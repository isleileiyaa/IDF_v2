"""
RIDDE retrieval-fusion head on top of a frozen Moirai 2.0 backbone.

Unlike the Chronos-Bolt path (models/ChronosBolt.py, augment='idf_clean_dis'),
f_inv/f_dyn/f_g here output a plain point forecast (dim = prediction_length)
trained with L2 loss (paper Eq. 14), instead of Chronos-Bolt's 9-quantile
output. This keeps the RIDDE head's output format independent of whatever
format the backbone itself produces, so swapping backbones again later only
requires adapting how `q` (a d_model-dim query vector) is obtained.

The backbone (Salesforce/moirai-2.0-R-small) is loaded from a vendored,
dependency-trimmed subset of uni2ts -- see TS-RAG/third_party/README.md for
why it's vendored instead of `pip install uni2ts`. The backbone is always
frozen; only the layers listed in NEW_HEAD_LAYER_NAMES are trainable. Those
names don't overlap with the backbone's own parameter names (all prefixed
`backbone.`), so the existing "freeze everything, then unfreeze by layer-name
substring match" logic in pretrain.py (see --freeze_chronos_bolt) works on
this model unmodified.
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

NEW_HEAD_LAYER_NAMES = [
    "encode_mlp",
    "ret_score_head",
    "fuse_gate",
    "routing_gate",
    "inv_pred_head",
    "dyn_pred_head_clean",
    "final_pred_head",
]


@dataclass
class Moirai2RiddeOutput:
    loss: Optional[torch.Tensor]
    point_forecast: torch.Tensor  # (B, L), de-normalized (query's own scale)
    # intermediates, kept around purely for the sanity-check script / debugging
    q: torch.Tensor
    h_ret: torch.Tensor
    h: torch.Tensor
    z_inv: torch.Tensor
    z_dyn: torch.Tensor
    y_inv: torch.Tensor
    y_dyn: torch.Tensor
    final_pred: torch.Tensor  # (B, L), normalized (query loc/scale) space, pre inverse-transform


class Moirai2ModelForForecastingWithRetrieval(nn.Module):
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
        L = prediction_length

        self.instance_norm = InstanceNorm()

        self.encode_mlp = nn.Sequential(
            nn.Linear(L, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )
        self.ret_score_head = nn.Linear(d * 2, 1)
        self.fuse_gate = nn.Linear(d * 2, d)
        self.routing_gate = nn.Linear(d * 2, d)
        self.inv_pred_head = nn.Linear(d, L)
        self.dyn_pred_head_clean = nn.Linear(d, L)
        self.final_pred_head = nn.Linear(L * 2, L)

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
        target: optional (B, prediction_length) raw ground-truth future, for L2 loss
        """
        L = self.prediction_length

        # query's own normalization stats -- used only to normalize `target` for the
        # loss and to de-normalize the final prediction, mirroring the Chronos-Bolt
        # idf_clean_dis convention (loss computed in query-normalized space).
        _, loc_scale = self.instance_norm(context)

        q = self.get_query_repr(context)  # (B, d)

        retrieved_seq_norm, _ = self.instance_norm(retrieved_seq)
        r_B, r_M, r_L = retrieved_seq_norm.shape
        assert r_L >= L, f"retrieved window length ({r_L}) must be >= prediction_length ({L})"
        retrieved_y = retrieved_seq_norm[..., -L:]  # (B, M, L)

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

        y_inv = self.inv_pred_head(z_inv)  # (B, L)
        y_dyn = self.dyn_pred_head_clean(z_dyn)  # (B, L)

        final_in = torch.cat([y_inv, y_dyn], dim=-1)  # (B, 2L)
        final_pred = self.final_pred_head(final_in)  # (B, L), normalized space

        loss = None
        if target is not None:
            target_norm, _ = self.instance_norm(target, loc_scale)
            loss = F.mse_loss(final_pred, target_norm)

        point_forecast = self.instance_norm.inverse(final_pred, loc_scale)

        return Moirai2RiddeOutput(
            loss=loss,
            point_forecast=point_forecast,
            q=q,
            h_ret=h_ret,
            h=h,
            z_inv=z_inv,
            z_dyn=z_dyn,
            y_inv=y_inv,
            y_dyn=y_dyn,
            final_pred=final_pred,
        )
