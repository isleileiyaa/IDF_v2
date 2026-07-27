"""
Retrieval-fusion heads on top of a frozen Moirai 2.0 backbone.

Two variants live here, both following the same philosophy as the original
Chronos-Bolt-backed methods they port: the head outputs a 9-quantile
distribution over the forecast horizon (dim = num_quantiles *
prediction_length), trained with the same pinball/quantile loss Chronos-Bolt
uses, instead of the backbone's own native output format.

- Moirai2ModelForForecastingWithRetrieval: ports the RIDDE / idf_clean_dis
  fusion (models/ChronosBolt.py, augment='idf_clean_dis') -- learned
  retrieval scoring + fuse/routing gates + separate inv/dyn heads.
- Moirai2MoEModelForForecastingWithRetrieval: ports the plain 'moe' fusion
  (models/ChronosBolt.py, augment='moe', see _run_moe_fusion) -- self-attention
  over [query, retrieved_y_1, ..., retrieved_y_M] + per-token gate + softmax
  weighted sum, residual-added back onto the query.

Both share Moirai2RetrievalBackbone for backbone loading/freezing, `q`
extraction (a forward hook on self.backbone.encoder -- Moirai2Module.forward()
itself is never reimplemented), and the quantile loss/de-normalization math.

The backbone (Salesforce/moirai-2.0-R-small) is loaded from a vendored,
dependency-trimmed subset of uni2ts -- see TS-RAG/third_party/README.md for
why it's vendored instead of `pip install uni2ts`. The backbone is always
frozen; only each subclass's own head layers are trainable. Those names
don't overlap with the backbone's own parameter names (all prefixed
`backbone.`), so the existing "freeze everything, then unfreeze by layer-name
substring match" logic in pretrain.py (see --freeze_chronos_bolt) works on
these models unmodified.
"""

import os
import sys
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_THIRD_PARTY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party")
if _THIRD_PARTY_DIR not in sys.path:
    sys.path.insert(0, _THIRD_PARTY_DIR)

from uni2ts.model.moirai2 import Moirai2Module  # noqa: E402

from .ChronosBolt import InstanceNorm  # noqa: E402

# same 9-level grid Chronos-Bolt trains on (config.chronos_config["quantiles"])
QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

RIDDE_HEAD_LAYER_NAMES = [
    "encode_mlp",
    "ret_score_head",
    "fuse_gate",
    "routing_gate",
    "inv_pred_head",
    "dyn_pred_head_clean",
    "final_pred_head",
]
# kept for backward compatibility with existing imports (sanity_check_moirai2.py)
NEW_HEAD_LAYER_NAMES = RIDDE_HEAD_LAYER_NAMES

MOE_HEAD_LAYER_NAMES = [
    "encode_mlp",
    "mha",
    "ffn",
    "gate_layer",
    "quantile_pred_head",
]


class Moirai2RetrievalBackbone(nn.Module):
    """
    Shared plumbing: loads + freezes the Moirai2 backbone, extracts the query
    hidden state `q` via a forward hook, normalizes retrieved windows, and
    provides the quantile loss / de-normalization math shared by all heads.
    Subclasses add their own fusion head + forward().
    """

    def __init__(
        self,
        pretrained_model_name: str = "Salesforce/moirai-2.0-R-small",
        context_length: int = 512,
        prediction_length: int = 64,
    ):
        super().__init__()
        self.backbone = Moirai2Module.from_pretrained(pretrained_model_name)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        d = self.backbone.d_model
        patch_size = self.backbone.patch_size
        assert context_length % patch_size == 0, (
            f"context_length ({context_length}) must be divisible by "
            f"backbone patch_size ({patch_size})"
        )
        self.d_model = d
        self.patch_size = patch_size
        self.context_length = context_length
        self.context_token_length = context_length // patch_size
        self.prediction_length = prediction_length

        self.num_quantiles = len(QUANTILE_LEVELS)
        self.register_buffer("quantiles", torch.tensor(QUANTILE_LEVELS), persistent=False)
        self.pred_dim = self.num_quantiles * prediction_length
        self.median_idx = QUANTILE_LEVELS.index(0.5)

        self.instance_norm = InstanceNorm()

        self._reprs = None
        self.backbone.encoder.register_forward_hook(self._capture_reprs_hook)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()  # backbone is always frozen/eval, regardless of outer train()/eval() calls
        return self

    def _capture_reprs_hook(self, module, inputs, output):
        self._reprs = output

    def get_query_repr(self, context: torch.Tensor) -> torch.Tensor:
        """
        context: (B, context_length) raw (unnormalized) window -- Moirai2 has
        its own internal PackedStdScaler, so we feed it raw data, not our own
        instance_norm'd version.

        Returns q: (B, d_model), the causal hidden state at the last context
        patch position (index context_token_length - 1), obtained via a
        forward hook on self.backbone.encoder rather than by reimplementing
        Moirai2Module.forward().
        """
        B = context.shape[0]
        device = context.device
        patches = context.view(B, self.context_token_length, self.patch_size)
        observed_mask = torch.ones_like(patches, dtype=torch.bool)
        sample_id = torch.zeros(B, self.context_token_length, dtype=torch.long, device=device)
        time_id = torch.arange(self.context_token_length, device=device).unsqueeze(0).expand(B, -1)
        variate_id = torch.zeros(B, self.context_token_length, dtype=torch.long, device=device)
        prediction_mask = torch.zeros(B, self.context_token_length, dtype=torch.bool, device=device)

        self._reprs = None
        with torch.no_grad():
            self.backbone(
                patches, observed_mask, sample_id, time_id, variate_id, prediction_mask,
                training_mode=True,
            )
        assert self._reprs is not None, "forward hook on backbone.encoder did not fire"
        reprs = self._reprs  # (B, context_token_length, d_model)
        q = reprs[:, self.context_token_length - 1, :]
        return q

    def normalize_retrieved_y(self, retrieved_seq: torch.Tensor):
        """
        retrieved_seq: (B, top_k, r_L) raw retrieved x+y windows, r_L >= prediction_length.
        Returns (retrieved_y (B, M, L) in retrieved-window-normalized space, M).
        """
        L = self.prediction_length
        retrieved_seq_norm, _ = self.instance_norm(retrieved_seq)
        r_B, r_M, r_L = retrieved_seq_norm.shape
        assert r_L >= L, f"retrieved window length ({r_L}) must be >= prediction_length ({L})"
        retrieved_y = retrieved_seq_norm[..., -L:]  # (B, M, L)
        return retrieved_y, r_M

    def quantile_loss(
        self,
        quantile_preds: torch.Tensor,
        target: torch.Tensor,
        loc_scale,
    ) -> torch.Tensor:
        """
        quantile_preds: (B, num_quantiles, L), query-normalized space.
        target: (B, L) raw ground truth.
        loc_scale: the query's own (loc, scale) from self.instance_norm(context).
        Pinball/quantile loss, same formula and reduction as Chronos-Bolt.
        """
        target_norm, _ = self.instance_norm(target, loc_scale)
        # Guard against near-constant context windows: InstanceNorm's
        # constant-input check only catches exact equality, so a context
        # window that's almost-but-not-exactly flat gets a tiny nonzero
        # scale, and target_norm = (target - loc) / scale can explode to
        # O(1e4) even though target itself looks unremarkable -- squared,
        # that's enough to blow up training (observed: loss spiking to
        # ~170k-4.7M / ~60M and derailing an otherwise-healthy run).
        # Clamping in normalized space is scale-invariant and a no-op for
        # well-conditioned data (which stays well within +-100).
        target_norm = torch.clamp(target_norm, min=-100.0, max=100.0)
        target_norm = target_norm.unsqueeze(1)  # (B, 1, L), broadcasts against (B, num_quantiles, L)

        loss = 2 * torch.abs(
            (target_norm - quantile_preds)
            * ((target_norm <= quantile_preds).float() - self.quantiles.view(1, -1, 1))
        )
        loss = loss.mean(dim=-2)  # mean over quantile levels
        loss = loss.sum(dim=-1)  # sum over the horizon
        return loss.mean()  # mean over batch

    def denormalize_quantiles(self, quantile_preds: torch.Tensor, loc_scale) -> torch.Tensor:
        """quantile_preds: (B, num_quantiles, L) normalized -> same shape, query's raw scale."""
        B = quantile_preds.shape[0]
        flat = self.instance_norm.inverse(quantile_preds.reshape(B, -1), loc_scale)
        return flat.view(*quantile_preds.shape)


@dataclass
class Moirai2RiddeOutput:
    loss: Optional[torch.Tensor]
    quantile_preds: torch.Tensor  # (B, num_quantiles, L), de-normalized (query's own scale)
    point_forecast: torch.Tensor  # (B, L), median quantile, de-normalized
    # intermediates, kept around purely for the sanity-check script / debugging
    q: torch.Tensor
    h_ret: torch.Tensor
    h: torch.Tensor
    z_inv: torch.Tensor
    z_dyn: torch.Tensor
    y_inv: torch.Tensor  # (B, num_quantiles, L), normalized space
    y_dyn: torch.Tensor  # (B, num_quantiles, L), normalized space


class Moirai2ModelForForecastingWithRetrieval(Moirai2RetrievalBackbone):
    """RIDDE (idf_clean_dis) fusion head, ported to a frozen Moirai2 backbone."""

    def __init__(
        self,
        pretrained_model_name: str = "Salesforce/moirai-2.0-R-small",
        context_length: int = 512,
        prediction_length: int = 64,
    ):
        super().__init__(pretrained_model_name, context_length, prediction_length)
        d = self.d_model
        L = self.prediction_length
        pred_dim = self.pred_dim

        self.encode_mlp = nn.Sequential(
            nn.Linear(L, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )
        self.ret_score_head = nn.Linear(d * 2, 1)
        self.fuse_gate = nn.Linear(d * 2, d)
        self.routing_gate = nn.Linear(d * 2, d)
        self.inv_pred_head = nn.Linear(d, pred_dim)
        self.dyn_pred_head_clean = nn.Linear(d, pred_dim)
        self.final_pred_head = nn.Linear(pred_dim * 2, pred_dim)

    def forward(
        self,
        context: torch.Tensor,
        retrieved_seq: torch.Tensor,
        distances: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
    ) -> Moirai2RiddeOutput:
        """
        context: (B, context_length) raw
        retrieved_seq: (B, top_k, r_L) raw retrieved x+y windows, r_L >= prediction_length
        distances: unused here (kept for interface parity with the Chronos-Bolt path;
            idf_clean_dis learns its own retrieval weighting via ret_score_head instead
            of using `distances` directly)
        target: optional (B, prediction_length) raw ground-truth future, for the loss
        """
        B = context.shape[0]
        Q, L = self.num_quantiles, self.prediction_length

        # query's own normalization stats -- used to normalize `target` for the
        # loss and to de-normalize the final prediction, mirroring the Chronos-Bolt
        # idf_clean_dis convention (loss computed in query-normalized space).
        _, loc_scale = self.instance_norm(context)

        q = self.get_query_repr(context)  # (B, d)
        retrieved_y, r_M = self.normalize_retrieved_y(retrieved_seq)  # (B, M, L)

        retrieved_y_enc = torch.stack(
            [self.encode_mlp(retrieved_y[:, i, :]) for i in range(r_M)], dim=1
        )  # (B, M, d)

        q_expand = q.unsqueeze(1).expand(-1, r_M, -1)
        ret_score_in = torch.cat([q_expand, retrieved_y_enc], dim=-1)
        alpha = F.softmax(self.ret_score_head(ret_score_in), dim=1)
        h_ret = (alpha * retrieved_y_enc).sum(dim=1)  # (B, d)

        fuse_in = torch.cat([q, h_ret], dim=-1)
        fuse_lambda = torch.sigmoid(self.fuse_gate(fuse_in))
        h = fuse_lambda * q + (1 - fuse_lambda) * h_ret  # (B, d)

        route_in = torch.cat([h, h_ret], dim=-1)
        gamma = torch.sigmoid(self.routing_gate(route_in))
        z_inv = gamma * h
        z_dyn = (1 - gamma) * h

        y_inv = self.inv_pred_head(z_inv).view(B, Q, L)
        y_dyn = self.dyn_pred_head_clean(z_dyn).view(B, Q, L)

        final_in = torch.cat([y_inv.reshape(B, -1), y_dyn.reshape(B, -1)], dim=-1)  # (B, 2*pred_dim)
        quantile_preds = self.final_pred_head(final_in).view(B, Q, L)  # normalized space

        loss = None
        if target is not None:
            loss = self.quantile_loss(quantile_preds, target, loc_scale)

        quantile_preds = self.denormalize_quantiles(quantile_preds, loc_scale)
        point_forecast = quantile_preds[:, self.median_idx, :]

        return Moirai2RiddeOutput(
            loss=loss,
            quantile_preds=quantile_preds,
            point_forecast=point_forecast,
            q=q,
            h_ret=h_ret,
            h=h,
            z_inv=z_inv,
            z_dyn=z_dyn,
            y_inv=y_inv,
            y_dyn=y_dyn,
        )


@dataclass
class Moirai2MoEOutput:
    loss: Optional[torch.Tensor]
    quantile_preds: torch.Tensor  # (B, num_quantiles, L), de-normalized (query's own scale)
    point_forecast: torch.Tensor  # (B, L), median quantile, de-normalized
    q: torch.Tensor
    retrieved_y_enc: torch.Tensor  # (B, M, d)
    att_output: torch.Tensor  # (B, 1+M, d)
    alpha: torch.Tensor  # (B, 1+M, 1)
    h: torch.Tensor  # (B, d), fused representation after residual add


class Moirai2MoEModelForForecastingWithRetrieval(Moirai2RetrievalBackbone):
    """
    Plain 'moe' fusion (models/ChronosBolt.py::_run_moe_fusion), ported to a
    frozen Moirai2 backbone: self-attention over [q, retrieved_y_1..M],
    per-token sigmoid gate + softmax, weighted sum residual-added onto q,
    then a Linear head projecting to the 9-quantile grid (instead of the
    Chronos-Bolt path's native output_patch_embedding).
    """

    def __init__(
        self,
        pretrained_model_name: str = "Salesforce/moirai-2.0-R-small",
        context_length: int = 512,
        prediction_length: int = 64,
    ):
        super().__init__(pretrained_model_name, context_length, prediction_length)
        d = self.d_model
        L = self.prediction_length

        self.encode_mlp = nn.Sequential(
            nn.Linear(L, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )
        self.mha = nn.MultiheadAttention(embed_dim=d, num_heads=8, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )
        self.gate_layer = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 1),
        )
        self.dropout = nn.Dropout(p=0.2)
        self.quantile_pred_head = nn.Linear(d, self.pred_dim)

    def forward(
        self,
        context: torch.Tensor,
        retrieved_seq: torch.Tensor,
        distances: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
    ) -> Moirai2MoEOutput:
        B = context.shape[0]
        Q, L = self.num_quantiles, self.prediction_length

        _, loc_scale = self.instance_norm(context)

        q = self.get_query_repr(context)  # (B, d)
        retrieved_y, r_M = self.normalize_retrieved_y(retrieved_seq)  # (B, M, L)

        retrieved_y_enc = torch.stack(
            [self.encode_mlp(retrieved_y[:, i, :]) for i in range(r_M)], dim=1
        )  # (B, M, d)

        q_seq = q.unsqueeze(1)  # (B, 1, d)
        all_enc = torch.cat([q_seq, retrieved_y_enc], dim=1)  # (B, 1+M, d)
        att_output, _ = self.mha(all_enc, all_enc, all_enc)
        att_output = all_enc + att_output
        att_output = att_output + self.dropout(self.ffn(att_output))

        scores = torch.stack(
            [torch.sigmoid(self.gate_layer(att_output[:, i, :])) for i in range(r_M + 1)], dim=1
        )  # (B, 1+M, 1)
        alpha = F.softmax(scores, dim=1)
        fused = torch.sum(alpha * att_output, dim=1)  # (B, d)
        fused = self.dropout(fused)
        h = q + fused  # (B, d) -- residual add, mirrors Chronos-Bolt's sequence_output + fused.unsqueeze(1)

        quantile_preds = self.quantile_pred_head(h).view(B, Q, L)  # normalized space

        loss = None
        if target is not None:
            loss = self.quantile_loss(quantile_preds, target, loc_scale)

        quantile_preds = self.denormalize_quantiles(quantile_preds, loc_scale)
        point_forecast = quantile_preds[:, self.median_idx, :]

        return Moirai2MoEOutput(
            loss=loss,
            quantile_preds=quantile_preds,
            point_forecast=point_forecast,
            q=q,
            retrieved_y_enc=retrieved_y_enc,
            att_output=att_output,
            alpha=alpha,
            h=h,
        )
