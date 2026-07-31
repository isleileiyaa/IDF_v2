"""
Retrieval-fusion heads on top of a frozen TimesFM 2.5 backbone.

Uses the native `timesfm` package (pip installed -e from
TS-RAG/third_party/timesfm, see TS-RAG/third_party/README.md), NOT the
HuggingFace `transformers` port -- the two use different checkpoint repos
(`google/timesfm-2.5-200m-pytorch` here vs `google/timesfm-2.5-200m-transformers`
for `transformers.TimesFm2_5ModelForPrediction`) and are not interchangeable.

Two variants live here, mirroring models/Moirai2.py:
- TimesFM25ModelForForecastingWithRetrieval: ports the RIDDE / idf_clean_dis
  fusion (models/ChronosBolt.py, augment='idf_clean_dis').
- TimesFM25MoEModelForForecastingWithRetrieval: ports the plain 'moe' fusion
  (models/ChronosBolt.py, augment='moe', see _run_moe_fusion).

Both share TimesFM25RetrievalBackbone for backbone loading/freezing, `q`
extraction, and the quantile loss/de-normalization math, and both use the
same 9-quantile output convention (pred_dim = num_quantiles *
prediction_length) trained with the pinball/quantile loss, independent of
TimesFM's own native output heads (output_projection_point /
output_projection_quantiles), which are never used here.

Unlike Moirai2, no forward hook is needed: TimesFM_2p5_200M_torch_module's
own forward() already returns `output_embeddings` (the post-transformer,
pre-head hidden states) directly as the second element of its output tuple.
In prefill mode (context only, no future placeholder patches appended), the
returned sequence length is exactly context_length // patch_length with no
extra tokens, so q is simply output_embeddings[:, context_token_length - 1, :].

The backbone is always frozen; only each subclass's own head layers are
trainable. Those names don't overlap with the backbone's own parameter names
(all prefixed `backbone.`), so the existing "freeze everything, then
unfreeze by layer-name substring match" logic in pretrain.py (see
--freeze_chronos_bolt) works on these models unmodified.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch  # noqa: E402

from .ChronosBolt import InstanceNorm  # noqa: E402
from .Moirai2 import QUANTILE_LEVELS  # noqa: E402 -- reuse the same 9-level quantile grid

RIDDE_HEAD_LAYER_NAMES = [
    "encode_mlp",
    "ret_score_head",
    "fuse_gate",
    "routing_gate",
    "inv_pred_head",
    "dyn_pred_head_clean",
    "final_pred_head",
]

MOE_HEAD_LAYER_NAMES = [
    "encode_mlp",
    "mha",
    "ffn",
    "gate_layer",
    "quantile_pred_head",
]


class TimesFM25RetrievalBackbone(nn.Module):
    """
    Shared plumbing: loads + freezes the TimesFM 2.5 backbone, extracts the
    query hidden state `q` directly from backbone.forward()'s own return
    value (no hook needed), normalizes retrieved windows, and provides the
    quantile loss / de-normalization math shared by all heads.
    """

    def __init__(
        self,
        pretrained_model_name: str = "google/timesfm-2.5-200m-pytorch",
        context_length: int = 512,
        prediction_length: int = 64,
    ):
        super().__init__()
        # torch_compile=False: we don't need TimesFM's own inference-speed
        # optimization, and it would silently recompile on every new batch
        # shape (training bs vs per-dataset zeroshot bs vs sanity-check bs).
        wrapper = TimesFM_2p5_200M_torch.from_pretrained(pretrained_model_name, torch_compile=False)
        self.backbone = wrapper.model  # the actual nn.Module (TimesFM_2p5_200M_torch_module)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        d = self.backbone.md  # model_dims = 1280
        patch_size = self.backbone.p  # input_patch_len = 32
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

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()  # backbone is always frozen/eval, regardless of outer train()/eval() calls
        return self

    def get_query_repr(self, context: torch.Tensor) -> torch.Tensor:
        """
        context: (B, context_length) raw (unnormalized) window -- TimesFM has
        its own internal per-patch RevIN normalization, so we feed it raw data.

        Returns q: (B, d_model), output_embeddings at the last real context
        patch position (index context_token_length - 1), taken directly from
        backbone.forward()'s own return value. No hook needed -- prefill mode
        (context patches only, no future placeholder patches) yields exactly
        context_token_length positions with no extra tokens.
        """
        B = context.shape[0]
        patches = context.view(B, self.context_token_length, self.patch_size)
        masks = torch.zeros_like(patches, dtype=torch.bool)  # all observed, nothing padded

        with torch.no_grad():
            (_, output_embeddings, _, _), _ = self.backbone(patches, masks, decode_caches=None)
        q = output_embeddings[:, self.context_token_length - 1, :]
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
        Pinball/quantile loss, same formula and reduction as Chronos-Bolt / Moirai2.py.
        """
        target_norm, _ = self.instance_norm(target, loc_scale)
        # Same near-constant-context guard as models/Moirai2.py (see that
        # file's quantile_loss for the full incident writeup): a context
        # window that's almost-but-not-exactly flat gets InstanceNorm scale
        # near zero, so target_norm can explode to O(1e4) and, squared, blow
        # up training. Clamping in normalized space is scale-invariant and a
        # no-op for well-conditioned data.
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
class TimesFM25RiddeOutput:
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


class TimesFM25ModelForForecastingWithRetrieval(TimesFM25RetrievalBackbone):
    """RIDDE (idf_clean_dis) fusion head, ported to a frozen TimesFM 2.5 backbone."""

    def __init__(
        self,
        pretrained_model_name: str = "google/timesfm-2.5-200m-pytorch",
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
    ) -> TimesFM25RiddeOutput:
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

        return TimesFM25RiddeOutput(
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
class TimesFM25MoEOutput:
    loss: Optional[torch.Tensor]
    quantile_preds: torch.Tensor  # (B, num_quantiles, L), de-normalized (query's own scale)
    point_forecast: torch.Tensor  # (B, L), median quantile, de-normalized
    q: torch.Tensor
    retrieved_y_enc: torch.Tensor  # (B, M, d)
    att_output: torch.Tensor  # (B, 1+M, d)
    alpha: torch.Tensor  # (B, 1+M, 1)
    h: torch.Tensor  # (B, d), fused representation after residual add


class TimesFM25MoEModelForForecastingWithRetrieval(TimesFM25RetrievalBackbone):
    """
    Plain 'moe' fusion (models/ChronosBolt.py::_run_moe_fusion), ported to a
    frozen TimesFM 2.5 backbone: self-attention over [q, retrieved_y_1..M],
    per-token sigmoid gate + softmax, weighted sum residual-added onto q,
    then a Linear head projecting to the 9-quantile grid.
    """

    def __init__(
        self,
        pretrained_model_name: str = "google/timesfm-2.5-200m-pytorch",
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
    ) -> TimesFM25MoEOutput:
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

        return TimesFM25MoEOutput(
            loss=loss,
            quantile_preds=quantile_preds,
            point_forecast=point_forecast,
            q=q,
            retrieved_y_enc=retrieved_y_enc,
            att_output=att_output,
            alpha=alpha,
            h=h,
        )
