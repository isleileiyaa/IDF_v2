import copy
import logging
import warnings
from dataclasses import dataclass, fields
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig
from transformers.models.t5.modeling_t5 import (
    ACT2FN,
    T5Config,
    T5LayerNorm,
    T5PreTrainedModel,
    T5Stack,
)
from transformers.utils import ModelOutput

from .base import BaseChronosPipeline, ForecastType

logger = logging.getLogger("autogluon.timeseries.models.chronos")


@dataclass
class ChronosBoltConfig:
    context_length: int
    prediction_length: int
    input_patch_size: int
    input_patch_stride: int
    quantiles: List[float]
    use_reg_token: bool = False


@dataclass
class ChronosBoltOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    total_loss: Optional[torch.Tensor] = None
    loss_forecast: Optional[torch.Tensor] = None
    loss_cons: Optional[torch.Tensor] = None
    loss_smooth: Optional[torch.Tensor] = None
    loss_inv: Optional[torch.Tensor] = None
    loss_dis: Optional[torch.Tensor] = None
    loss_ret: Optional[torch.Tensor] = None
    loss_dyn: Optional[torch.Tensor] = None
    use_disentangle_aux_loss: Optional[bool] = None
    aux_loss_enabled: Optional[bool] = None
    quantile_preds: Optional[torch.Tensor] = None
    attentions: Optional[torch.Tensor] = None
    cross_attentions: Optional[torch.Tensor] = None


class Patch(nn.Module):
    def __init__(self, patch_size: int, patch_stride: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]

        if length % self.patch_size != 0:
            padding_size = (
                *x.shape[:-1],
                self.patch_size - (length % self.patch_size),
            )
            padding = torch.full(size=padding_size, fill_value=torch.nan, dtype=x.dtype, device=x.device)
            x = torch.concat((padding, x), dim=-1)

        x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
        return x

    
class InstanceNorm(nn.Module):
    """
    Instance Normalization with handling for constant inputs.
    For constant inputs, the normalized output is set to 1, and inverse restores the original constant value.
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        loc_scale: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if loc_scale is None:
            # Compute loc and scale
            loc = torch.nan_to_num(torch.nanmean(x, dim=-1, keepdim=True), nan=0.0)
            scale = torch.nan_to_num(
                (x - loc).square().nanmean(dim=-1, keepdim=True).sqrt(),
                nan=1.0
            )

            # Detect constant inputs
            is_constant = torch.all(x == x[..., :1], dim=-1, keepdim=True)  # Batch-wise constant input detection

            # For constant inputs, set scale = 1
            scale = torch.where(is_constant, torch.ones_like(scale), scale)
        else:
            loc, scale = loc_scale

        # Normalize input
        normalized = (x - loc) / scale

        # For constant inputs, override normalized result to 1
        is_constant = torch.all(x == x[..., :1], dim=-1, keepdim=True) if loc_scale is None else (scale == 1)
        normalized = torch.where(is_constant, torch.ones_like(normalized), normalized)

        return normalized, (loc, scale)

    def inverse(self, x: torch.Tensor, loc_scale: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        loc, scale = loc_scale

        # Detect constant inputs during inverse
        is_constant = scale == 1

        # Restore original values for constant inputs
        original = torch.where(is_constant, loc, x * scale + loc)
        return original



class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        out_dim: int,
        act_fn_name: str,
        dropout_p: float = 0.0,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout_p)
        self.hidden_layer = nn.Linear(in_dim, h_dim)
        self.act = ACT2FN[act_fn_name]
        self.output_layer = nn.Linear(h_dim, out_dim)
        self.residual_layer = nn.Linear(in_dim, out_dim)

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = T5LayerNorm(out_dim)

    def forward(self, x: torch.Tensor):
        hid = self.act(self.hidden_layer(x))
        out = self.dropout(self.output_layer(hid))
        res = self.residual_layer(x)

        out = out + res

        if self.use_layer_norm:
            return self.layer_norm(out)
        return out


class ChronosBoltModelForForecasting(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"input_patch_embedding\.",
        r"output_patch_embedding\.",
    ]
    _keys_to_ignore_on_load_unexpected = [r"lm_head.weight"]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: T5Config):
        assert hasattr(config, "chronos_config"), "Not a Chronos config file"

        super().__init__(config)
        self.model_dim = config.d_model

        # TODO: remove filtering eventually, added for backward compatibility
        config_fields = {f.name for f in fields(ChronosBoltConfig)}
        self.chronos_config = ChronosBoltConfig(
            **{k: v for k, v in config.chronos_config.items() if k in config_fields}
        )

        # Only decoder_start_id (and optionally REG token)
        if self.chronos_config.use_reg_token:
            config.reg_token_id = 1

        config.vocab_size = 2 if self.chronos_config.use_reg_token else 1
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # Input patch embedding layer
        self.input_patch_embedding = ResidualBlock(
            in_dim=self.chronos_config.input_patch_size * 2,
            h_dim=config.d_ff,
            out_dim=config.d_model,
            act_fn_name=config.dense_act_fn,
            dropout_p=config.dropout_rate,
        )

        # patching layer
        self.patch = Patch(
            patch_size=self.chronos_config.input_patch_size,
            patch_stride=self.chronos_config.input_patch_stride,
        )

        # instance normalization, also referred to as "scaling" in Chronos and GluonTS
        self.instance_norm = InstanceNorm()

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        self._init_decoder(config)

        self.num_quantiles = len(self.chronos_config.quantiles)
        quantiles = torch.tensor(self.chronos_config.quantiles, dtype=self.dtype)
        self.register_buffer("quantiles", quantiles, persistent=False)

        self.output_patch_embedding = ResidualBlock(
            in_dim=config.d_model,
            h_dim=config.d_ff,
            out_dim=self.num_quantiles * self.chronos_config.prediction_length,
            act_fn_name=config.dense_act_fn,
            dropout_p=config.dropout_rate,
        )

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def _init_weights(self, module):
        super()._init_weights(module)
        """Initialize the weights"""
        factor = self.config.initializer_factor
        if isinstance(module, (self.__class__)):
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, ResidualBlock):
            module.hidden_layer.weight.data.normal_(
                mean=0.0,
                std=factor * ((self.chronos_config.input_patch_size * 2) ** -0.5),
            )
            if hasattr(module.hidden_layer, "bias") and module.hidden_layer.bias is not None:
                module.hidden_layer.bias.data.zero_()

            module.residual_layer.weight.data.normal_(
                mean=0.0,
                std=factor * ((self.chronos_config.input_patch_size * 2) ** -0.5),
            )
            if hasattr(module.residual_layer, "bias") and module.residual_layer.bias is not None:
                module.residual_layer.bias.data.zero_()

            module.output_layer.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.output_layer, "bias") and module.output_layer.bias is not None:
                module.output_layer.bias.data.zero_()

    def forward(
        self,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> ChronosBoltOutput:
        mask = mask.to(context.dtype) if mask is not None else torch.isnan(context).logical_not().to(context.dtype)

        batch_size, _ = context.shape
        if context.shape[-1] > self.chronos_config.context_length:
            context = context[..., -self.chronos_config.context_length :]
            mask = mask[..., -self.chronos_config.context_length :]

        # scaling
        context, loc_scale = self.instance_norm(context)

        # the scaling op above is done in 32-bit precision,
        # then the context is moved to model's dtype
        context = context.to(self.dtype)
        mask = mask.to(self.dtype)

        # patching
        patched_context = self.patch(context)
        patched_mask = torch.nan_to_num(self.patch(mask), nan=0.0)
        patched_context[~(patched_mask > 0)] = 0.0
        # concat context and mask along patch dim
        patched_context = torch.cat([patched_context, patched_mask], dim=-1)

        # attention_mask = 1 if at least one item in the patch is observed
        attention_mask = patched_mask.sum(dim=-1) > 0  # (batch_size, patched_seq_length)

        input_embeds = self.input_patch_embedding(patched_context)

        if self.chronos_config.use_reg_token:
            # Append [REG]
            reg_input_ids = torch.full(
                (batch_size, 1),
                self.config.reg_token_id,
                device=input_embeds.device,
            )
            reg_embeds = self.shared(reg_input_ids)
            input_embeds = torch.cat([input_embeds, reg_embeds], dim=-2)
            attention_mask = torch.cat([attention_mask, torch.ones_like(reg_input_ids)], dim=-1)

        encoder_outputs = self.encoder(
            attention_mask=attention_mask,
            inputs_embeds=input_embeds,
        )
        hidden_states = encoder_outputs[0]

        sequence_output = self.decode(input_embeds, attention_mask, hidden_states)

        quantile_preds_shape = (
            batch_size,
            self.num_quantiles,
            self.chronos_config.prediction_length,
        )
        quantile_preds = self.output_patch_embedding(sequence_output).view(*quantile_preds_shape)

        loss = None
        if target is not None:
            # normalize target
            target, _ = self.instance_norm(target, loc_scale)
            target = target.unsqueeze(1)  # type: ignore
            assert self.chronos_config.prediction_length >= target.shape[-1]

            target = target.to(quantile_preds.device)
            target_mask = (
                target_mask.unsqueeze(1).to(quantile_preds.device) if target_mask is not None else ~torch.isnan(target)
            )
            target[~target_mask] = 0.0

            # pad target and target_mask if they are shorter than model's prediction_length
            if self.chronos_config.prediction_length > target.shape[-1]:
                padding_shape = (*target.shape[:-1], self.chronos_config.prediction_length - target.shape[-1])
                target = torch.cat([target, torch.zeros(padding_shape).to(target)], dim=-1)
                target_mask = torch.cat([target_mask, torch.zeros(padding_shape).to(target_mask)], dim=-1)

            loss = (
                2
                * torch.abs(
                    (target - quantile_preds)
                    * ((target <= quantile_preds).float() - self.quantiles.view(1, self.num_quantiles, 1))
                )
                * target_mask.float()
            )
            loss = loss.mean(dim=-2)  # Mean over prediction horizon
            loss = loss.sum(dim=-1)  # Sum over quantile levels
            loss = loss.mean()  # Mean over batch

        # Unscale predictions
        quantile_preds = self.instance_norm.inverse(
            quantile_preds.view(batch_size, -1),
            loc_scale,
        ).view(*quantile_preds_shape)

        return ChronosBoltOutput(
            loss=loss,
            quantile_preds=quantile_preds,
        )

    def _init_decoder(self, config):
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

    def decode(
        self,
        input_embeds,
        attention_mask,
        hidden_states,
        output_attentions=False,
    ):
        """
        Parameters
        ----------
        input_embeds: torch.Tensor
            Patched and embedded inputs. Shape (batch_size, patched_context_length, d_model)
        attention_mask: torch.Tensor
            Attention mask for the patched context. Shape (batch_size, patched_context_length), type: torch.int64
        hidden_states: torch.Tensor
            Hidden states returned by the encoder. Shape (batch_size, patched_context_length, d_model)

        Returns
        -------
        last_hidden_state
            Last hidden state returned by the decoder, of shape (batch_size, 1, d_model)
        """
        batch_size = input_embeds.shape[0]
        decoder_input_ids = torch.full(
            (batch_size, 1),
            self.config.decoder_start_token_id,
            device=input_embeds.device,
        )
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            output_attentions=output_attentions,
            return_dict=True,
        )

        return decoder_outputs.last_hidden_state  # sequence_outputs, b x 1 x d_model


class ChronosBoltPipeline(BaseChronosPipeline):
    forecast_type: ForecastType = ForecastType.QUANTILES
    default_context_length: int = 2048
    # register this class name with this alias for backward compatibility
    _aliases = ["PatchedT5Pipeline"]

    def __init__(self, model: ChronosBoltModelForForecasting):
        super().__init__(inner_model=model)
        self.model = model

    @property
    def quantiles(self) -> List[float]:
        return self.model.config.chronos_config["quantiles"]

    def predict(  # type: ignore[override]
        self,
        context: Union[torch.Tensor, List[torch.Tensor]],
        prediction_length: Optional[int] = None,
        limit_prediction_length: bool = False,
    ):
        context_tensor = self._prepare_and_validate_context(context=context)

        model_context_length = self.model.config.chronos_config["context_length"]
        model_prediction_length = self.model.config.chronos_config["prediction_length"]
        if prediction_length is None:
            prediction_length = model_prediction_length

        if prediction_length > model_prediction_length:
            msg = (
                f"We recommend keeping prediction length <= {model_prediction_length}. "
                "The quality of longer predictions may degrade since the model is not optimized for it. "
            )
            if limit_prediction_length:
                msg += "You can turn off this check by setting `limit_prediction_length=False`."
                raise ValueError(msg)
            warnings.warn(msg)

        predictions = []
        remaining = prediction_length

        # We truncate the context here because otherwise batches with very long
        # context could take up large amounts of GPU memory unnecessarily.
        if context_tensor.shape[-1] > model_context_length:
            context_tensor = context_tensor[..., -model_context_length:]

        # TODO: We unroll the forecast of Chronos Bolt greedily with the full forecast
        # horizon that the model was trained with (i.e., 64). This results in variance collapsing
        # every 64 steps.
        while remaining > 0:
            with torch.no_grad():
                prediction = self.model(
                    context=context_tensor.to(
                        device=self.model.device,
                        dtype=torch.float32,  # scaling should be done in 32-bit precision
                    ),
                ).quantile_preds.to(context_tensor)

            predictions.append(prediction)
            remaining -= prediction.shape[-1]

            if remaining <= 0:
                break

            central_idx = torch.abs(torch.tensor(self.quantiles) - 0.5).argmin()
            central_prediction = prediction[:, central_idx]

            context_tensor = torch.cat([context_tensor, central_prediction], dim=-1)

        return torch.cat(predictions, dim=-1)[..., :prediction_length]

    def predict_quantiles(
        self, context: torch.Tensor, prediction_length: int, quantile_levels: List[float], **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # shape (batch_size, prediction_length, len(training_quantile_levels))
        predictions = (
            self.predict(
                context,
                prediction_length=prediction_length,
            )
            .detach()
            .cpu()
            .swapaxes(1, 2)
        )

        training_quantile_levels = self.quantiles

        if set(quantile_levels).issubset(set(training_quantile_levels)):
            # no need to perform intra/extrapolation
            quantiles = predictions[..., [training_quantile_levels.index(q) for q in quantile_levels]]
        else:
            # we rely on torch for interpolating quantiles if quantiles that
            # Chronos Bolt was trained on were not provided
            if min(quantile_levels) < min(training_quantile_levels) or max(quantile_levels) > max(
                training_quantile_levels
            ):
                logger.warning(
                    f"\tQuantiles to be predicted ({quantile_levels}) are not within the range of "
                    f"quantiles that Chronos-Bolt was trained on ({training_quantile_levels}). "
                    "Quantile predictions will be set to the minimum/maximum levels at which Chronos-Bolt "
                    "was trained on. This may significantly affect the quality of the predictions."
                )

            # TODO: this is a hack that assumes the model's quantiles during training (training_quantile_levels)
            # made up an equidistant grid along the quantile dimension. i.e., they were (0.1, 0.2, ..., 0.9).
            # While this holds for official Chronos-Bolt models, this may not be true in the future, and this
            # function may have to be revised.
            augmented_predictions = torch.cat(
                [predictions[..., [0]], predictions, predictions[..., [-1]]],
                dim=-1,
            )
            quantiles = torch.quantile(
                augmented_predictions, q=torch.tensor(quantile_levels, dtype=augmented_predictions.dtype), dim=-1
            ).permute(1, 2, 0)
        mean = predictions[:, :, training_quantile_levels.index(0.5)]
        return quantiles, mean

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """
        Load the model, either from a local path or from the HuggingFace Hub.
        Supports the same arguments as ``AutoConfig`` and ``AutoModel``
        from ``transformers``.
        """
        # if optimization_strategy is provided, pop this as it won't be used
        kwargs.pop("optimization_strategy", None)

        config = AutoConfig.from_pretrained(*args, **kwargs)
        assert hasattr(config, "chronos_config"), "Not a Chronos config file"

        context_length = kwargs.pop("context_length", None)
        if context_length is not None:
            config.chronos_config["context_length"] = context_length

        architecture = config.architectures[0]
        class_ = globals().get(architecture)

        # TODO: remove this once all models carry the correct architecture names in their configuration
        # and raise an error instead.
        if class_ is None:
            logger.warning(f"Unknown architecture: {architecture}, defaulting to ChronosBoltModelForForecasting")
            class_ = ChronosBoltModelForForecasting

        model = class_.from_pretrained(*args, **kwargs)
        return cls(model=model)

def compute_time_series_stats(tensor, dim=-1, keepdim=False):
    """
    mean, std, min, max
    """
    mean_val = tensor.mean(dim=dim, keepdim=keepdim)
    std_val = tensor.std(dim=dim, keepdim=keepdim)
    min_val = tensor.min(dim=dim, keepdim=keepdim)[0]
    max_val = tensor.max(dim=dim, keepdim=keepdim)[0]

    return torch.cat([mean_val, std_val, min_val, max_val], dim=-1)


class ChronosBoltModelForForecastingWithRetrieval(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"input_patch_embedding\.",
        r"output_patch_embedding\.",
    ]
    _keys_to_ignore_on_load_unexpected = [r"lm_head.weight"]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: T5Config, augment: str):
        assert hasattr(config, "chronos_config"), "Not a Chronos config file"

        super().__init__(config)
        self.model_dim = config.d_model
        self.augment = augment

        # TODO: remove filtering eventually, added for backward compatibility
        config_fields = {f.name for f in fields(ChronosBoltConfig)}
        self.chronos_config = ChronosBoltConfig(
            **{k: v for k, v in config.chronos_config.items() if k in config_fields}
        )

        # Only decoder_start_id (and optionally REG token)
        if self.chronos_config.use_reg_token:
            config.reg_token_id = 1

        config.vocab_size = 2 if self.chronos_config.use_reg_token else 1
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # Input patch embedding layer
        self.input_patch_embedding = ResidualBlock(
            in_dim=self.chronos_config.input_patch_size * 2,
            h_dim=config.d_ff,
            out_dim=config.d_model,
            act_fn_name=config.dense_act_fn,
            dropout_p=config.dropout_rate,
        )

        # patching layer
        self.patch = Patch(
            patch_size=self.chronos_config.input_patch_size,
            patch_stride=self.chronos_config.input_patch_stride,
        )

        # instance normalization, also referred to as "scaling" in Chronos and GluonTS
        self.instance_norm = InstanceNorm()

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        self._init_decoder(config)

        self.num_quantiles = len(self.chronos_config.quantiles)
        quantiles = torch.tensor(self.chronos_config.quantiles, dtype=self.dtype)
        self.register_buffer("quantiles", quantiles, persistent=False)

        self.output_patch_embedding = ResidualBlock(
            in_dim=config.d_model,
            h_dim=config.d_ff,
            out_dim=self.num_quantiles * self.chronos_config.prediction_length,
            act_fn_name=config.dense_act_fn,
            dropout_p=config.dropout_rate,
        )

        self.dropout = nn.Dropout(p=0.2)

        # Retrieval Augmentation Layer

        if 'gate' in self.augment:
            # gate: MLP + Linear
            self.gate_layer = nn.Sequential(
                nn.Linear(self.chronos_config.prediction_length * 2 + 512 + 12, config.d_model),
                nn.ReLU(),
                nn.Linear(config.d_model, config.d_model),
            )
            self.gate_linear1 = nn.Linear(self.chronos_config.prediction_length * 2 + 512 + 12, config.d_model)
            self.gate_linear2 = nn.Linear(config.d_model, 1)

        if 'moe' in self.augment:
            # moe: moe with topk+1 experts, use mha
            self.encode_mlp = nn.Sequential(
                nn.Linear(self.chronos_config.prediction_length, config.d_model),
                nn.ReLU(),
                nn.Linear(config.d_model, config.d_model),
            )
            self.mha = nn.MultiheadAttention(embed_dim=config.d_model, num_heads=8, batch_first=True)
            self.ffn = nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.ReLU(),
                nn.Linear(config.d_model, config.d_model),
            )
            self.gate_layer = nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.ReLU(),
                nn.Linear(config.d_model, 1),
            )

        if self.augment == 'moe_disentangle':
            self.disentangle_gate = nn.Linear(config.d_model * 2, config.d_model, bias=False)
            self.final_pred_head = nn.Linear(self.chronos_config.prediction_length * 2, self.chronos_config.prediction_length)
            pred_dim = self.num_quantiles * self.chronos_config.prediction_length
            self.inv_aux_backproj = nn.Linear(pred_dim, config.d_model)
            self.dyn_aux_backproj = nn.Linear(pred_dim, config.d_model)

        if self.augment == 'idf':
            self.encode_mlp = nn.Sequential(
                nn.Linear(self.chronos_config.prediction_length, config.d_model),
                nn.ReLU(),
                nn.Linear(config.d_model, config.d_model),
            )
            self.ret_score_head = nn.Linear(config.d_model * 2, 1)
            self.fuse_gate = nn.Linear(config.d_model * 2, config.d_model)
            self.routing_gate = nn.Linear(config.d_model * 2, config.d_model)
            self.inv_transition = nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.ReLU(),
                nn.Linear(config.d_model, config.d_model),
            )
            self.dyn_transition = nn.Sequential(
                nn.Linear(config.d_model * 2, config.d_model),
                nn.ReLU(),
                nn.Linear(config.d_model, config.d_model),
            )
            self.inv_head = nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.ReLU(),
                nn.Linear(config.d_model, config.d_model),
            )
            self.dyn_head = nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.ReLU(),
                nn.Linear(config.d_model, config.d_model),
            )
            self.final_head = nn.Sequential(
                nn.Linear(config.d_model * 2, config.d_model),
                nn.ReLU(),
                nn.Linear(config.d_model, config.d_model),
            )

        if self.augment in ['idf_branch', 'idf_x']:
            pred_dim = self.num_quantiles * self.chronos_config.prediction_length
            self.encode_mlp = nn.Sequential(
                nn.Linear(self.chronos_config.prediction_length, config.d_model),
                nn.ReLU(),
                nn.Linear(config.d_model, config.d_model),
            )
            self.ret_score_head = nn.Linear(config.d_model * 2, 1)
            self.fuse_gate = nn.Linear(config.d_model * 2, config.d_model)
            self.routing_gate = nn.Linear(config.d_model * 2, config.d_model)
            self.inv_pred_head = nn.Linear(config.d_model, pred_dim)
            self.inv_residual_head = nn.Linear(config.d_model, pred_dim)
            self.dyn_pred_head = nn.Linear(config.d_model * 2, pred_dim)
            self.final_pred_head = nn.Linear(pred_dim * 2, pred_dim)
            self.inv_aux_backproj = nn.Linear(pred_dim, config.d_model)
            self.dyn_aux_backproj = nn.Linear(pred_dim, config.d_model)

        if self.augment == 'idf_clean_dis':
            pred_dim = self.num_quantiles * self.chronos_config.prediction_length
            self.encode_mlp = nn.Sequential(
                nn.Linear(self.chronos_config.prediction_length, config.d_model),
                nn.ReLU(),
                nn.Linear(config.d_model, config.d_model),
            )
            self.ret_score_head = nn.Linear(config.d_model * 2, 1)
            self.fuse_gate = nn.Linear(config.d_model * 2, config.d_model)
            self.routing_gate = nn.Linear(config.d_model * 2, config.d_model)
            self.inv_pred_head = nn.Linear(config.d_model, pred_dim)
            self.dyn_pred_head_clean = nn.Linear(config.d_model, pred_dim)
            self.final_pred_head = nn.Linear(pred_dim * 2, pred_dim)
            self.inv_aux_backproj = nn.Linear(pred_dim, config.d_model)
            self.dyn_aux_backproj = nn.Linear(pred_dim, config.d_model)

        if self.augment == 'idf_clean_dis_deepmlp':
            pred_dim = self.num_quantiles * self.chronos_config.prediction_length
            self.encode_mlp = nn.Sequential(
                nn.Linear(self.chronos_config.prediction_length, config.d_model),
                nn.ReLU(),
                # nn.Dropout(self.dropout.p),
                nn.Linear(config.d_model, config.d_model),
                nn.ReLU(),
                # nn.Dropout(self.dropout.p),
                nn.Linear(config.d_model, config.d_model),
                nn.ReLU(),
                # nn.Dropout(self.dropout.p),
                nn.Linear(config.d_model, config.d_model),
            )
            self.ret_score_head = nn.Linear(config.d_model * 2, 1)
            self.fuse_gate = nn.Linear(config.d_model * 2, config.d_model)
            self.routing_gate = nn.Linear(config.d_model * 2, config.d_model)
            self.inv_pred_head = nn.Linear(config.d_model, pred_dim)
            self.dyn_pred_head_clean = nn.Linear(config.d_model, pred_dim)
            self.final_pred_head = nn.Linear(pred_dim * 2, pred_dim)
            self.inv_aux_backproj = nn.Linear(pred_dim, config.d_model)
            self.dyn_aux_backproj = nn.Linear(pred_dim, config.d_model)

        if self.augment == 'idf_clean_dis_ts3align':
            pred_dim = self.num_quantiles * self.chronos_config.prediction_length
            self.encode_mlp = nn.Sequential(
                nn.Linear(self.chronos_config.prediction_length, config.d_model),
                nn.ReLU(),
                nn.Linear(config.d_model, config.d_model),
            )
            # Keep the module available for compatibility, but the default TS3-aligned
            # branch uses uniform aggregation instead of learned query-conditioned scoring.
            self.ret_score_head = nn.Linear(config.d_model * 2, 1)
            self.fuse_gate = nn.Linear(config.d_model * 2, config.d_model)
            self.routing_gate = nn.Linear(config.d_model * 2, config.d_model)
            self.inv_pred_head = nn.Linear(config.d_model, pred_dim)
            self.dyn_pred_head_clean = nn.Linear(config.d_model, pred_dim)
            self.final_pred_head = nn.Linear(pred_dim * 2, pred_dim)
            self.inv_aux_backproj = nn.Linear(pred_dim, config.d_model)
            self.dyn_aux_backproj = nn.Linear(pred_dim, config.d_model)

        if self.augment == 'idf_h_linear_head':
            pred_dim = self.num_quantiles * self.chronos_config.prediction_length
            self.encode_mlp = nn.Sequential(
                nn.Linear(self.chronos_config.prediction_length, config.d_model),
                nn.ReLU(),
                nn.Linear(config.d_model, config.d_model),
            )
            self.ret_score_head = nn.Linear(config.d_model * 2, 1)
            self.fuse_gate = nn.Linear(config.d_model * 2, config.d_model)
            self.h_pred_head = nn.Linear(config.d_model, pred_dim)

        if self.augment == 'idf_h_native_head':
            self.encode_mlp = nn.Sequential(
                nn.Linear(self.chronos_config.prediction_length, config.d_model),
                nn.ReLU(),
                nn.Linear(config.d_model, config.d_model),
            )
            self.ret_score_head = nn.Linear(config.d_model * 2, 1)
            self.fuse_gate = nn.Linear(config.d_model * 2, config.d_model)

        if self.augment == 'idf_y_linear_head':
            pred_dim = self.num_quantiles * self.chronos_config.prediction_length
            self.encode_mlp = nn.Sequential(
                nn.Linear(self.chronos_config.prediction_length, config.d_model),
                nn.ReLU(),
                nn.Linear(config.d_model, config.d_model),
            )
            self.ret_score_head = nn.Linear(config.d_model * 2, 1)
            self.fuse_gate = nn.Linear(config.d_model * 2, config.d_model)
            self.h_pred_head = nn.Linear(config.d_model, pred_dim)
            self.y_linear_head = nn.Linear(pred_dim, pred_dim)

        if self.augment == 'idf_branch_gru':
            pred_dim = self.num_quantiles * self.chronos_config.prediction_length
            self.encode_mlp = nn.Sequential(
                nn.Linear(self.chronos_config.prediction_length, config.d_model),
                nn.ReLU(),
                nn.Linear(config.d_model, config.d_model),
            )
            self.ret_score_head = nn.Linear(config.d_model * 2, 1)
            self.fuse_gate = nn.Linear(config.d_model * 2, config.d_model)
            self.routing_gate = nn.Linear(config.d_model * 2, config.d_model)
            self.inv_pred_head = nn.Linear(config.d_model, pred_dim)
            self.inv_residual_head = nn.Linear(config.d_model, pred_dim)
            self.dyn_gru = nn.GRU(
                input_size=config.d_model * 2,
                hidden_size=config.d_model,
                batch_first=True,
            )
            self.dyn_out_head = nn.Linear(config.d_model, self.num_quantiles)
            self.final_pred_head = nn.Linear(pred_dim * 2, pred_dim)

        if self.augment == 'idf_branch_gru_q':
            pred_dim = self.num_quantiles * self.chronos_config.prediction_length
            self.encode_mlp = nn.Sequential(
                nn.Linear(self.chronos_config.prediction_length, config.d_model),
                nn.ReLU(),
                nn.Linear(config.d_model, config.d_model),
            )
            self.ret_score_head = nn.Linear(config.d_model * 2, 1)
            self.fuse_gate = nn.Linear(config.d_model * 2, config.d_model)
            self.routing_gate = nn.Linear(config.d_model * 2, config.d_model)
            self.inv_pred_head = nn.Linear(config.d_model, pred_dim)
            self.inv_residual_head = nn.Linear(config.d_model, pred_dim)
            self.dyn_gru = nn.GRU(
                input_size=config.d_model * 2 + self.num_quantiles,
                hidden_size=config.d_model,
                batch_first=True,
            )
            self.dyn_out_head = nn.Linear(config.d_model, self.num_quantiles)
            self.final_pred_head = nn.Linear(pred_dim * 2, pred_dim)

        if self.augment == 'idf_dual_direct_head':
            pred_dim = self.num_quantiles * self.chronos_config.prediction_length
            self.encode_mlp = nn.Sequential(
                nn.Linear(self.chronos_config.prediction_length, config.d_model),
                nn.ReLU(),
                nn.Linear(config.d_model, config.d_model),
            )
            self.ret_score_head = nn.Linear(config.d_model * 2, 1)
            self.fuse_gate = nn.Linear(config.d_model * 2, config.d_model)
            self.routing_gate = nn.Linear(config.d_model * 2, config.d_model)
            self.inv_hidden_proj = nn.Linear(config.d_model, config.d_model)
            self.dyn_ret_hidden_proj = nn.Linear(config.d_model * 2, config.d_model)
            self.final_pred_head = nn.Linear(pred_dim * 2, pred_dim)

        if self.augment == 'idf_residual':
            pred_dim = self.num_quantiles * self.chronos_config.prediction_length
            self.encode_mlp = nn.Sequential(
                nn.Linear(self.chronos_config.prediction_length, config.d_model),
                nn.ReLU(),
                nn.Linear(config.d_model, config.d_model),
            )
            self.ret_score_head = nn.Linear(config.d_model * 2, 1)
            self.fuse_gate = nn.Linear(config.d_model * 2, config.d_model)
            self.routing_gate = nn.Linear(config.d_model * 2, config.d_model)
            self.inv_pred_head = nn.Linear(config.d_model, pred_dim)
            self.dyn_pred_head = nn.Linear(config.d_model * 2, pred_dim)

        if self.augment == 'idf_dual_projector':
            pred_dim = self.num_quantiles * self.chronos_config.prediction_length
            self.fuse_gate = nn.Linear(config.d_model * 2, config.d_model)
            self.g_inv = nn.Linear(config.d_model * 2, config.d_model)
            self.g_dyn = nn.Linear(config.d_model * 2, config.d_model)
            self.f_inv = nn.Linear(config.d_model, pred_dim)
            self.f_dyn = nn.Linear(config.d_model, pred_dim)

        if self.augment == 'idf_dual_projector_mlp':
            pred_dim = self.num_quantiles * self.chronos_config.prediction_length
            self.retrieved_x_encoder_mlp = nn.Sequential(
                nn.Linear(self.chronos_config.context_length, config.d_model),
                nn.ReLU(),
                nn.Linear(config.d_model, config.d_model),
            )
            self.fuse_gate = nn.Linear(config.d_model * 2, config.d_model)
            self.g_inv = nn.Linear(config.d_model * 2, config.d_model)
            self.g_dyn = nn.Linear(config.d_model * 2, config.d_model)
            self.f_inv = nn.Linear(config.d_model, pred_dim)
            self.f_dyn = nn.Linear(config.d_model, pred_dim)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def _init_weights(self, module):
        super()._init_weights(module)
        """Initialize the weights"""
        factor = self.config.initializer_factor
        if isinstance(module, (self.__class__)):
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, ResidualBlock):
            module.hidden_layer.weight.data.normal_(
                mean=0.0,
                std=factor * ((self.chronos_config.input_patch_size * 2) ** -0.5),
            )
            if hasattr(module.hidden_layer, "bias") and module.hidden_layer.bias is not None:
                module.hidden_layer.bias.data.zero_()

            module.residual_layer.weight.data.normal_(
                mean=0.0,
                std=factor * ((self.chronos_config.input_patch_size * 2) ** -0.5),
            )
            if hasattr(module.residual_layer, "bias") and module.residual_layer.bias is not None:
                module.residual_layer.bias.data.zero_()

            module.output_layer.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.output_layer, "bias") and module.output_layer.bias is not None:
                module.output_layer.bias.data.zero_()
    
    def init_extra_weights(self, layers):
        """
        Initialize weights for multiple layers.
        
        Args:
            layers (list): List of layers (e.g., [self.retrieve_yin_layer, ...]).
        """
        factor = self.config.initializer_factor
        
        for layer in layers:
            if isinstance(layer, nn.Sequential):
                self.init_extra_weights(layer)
            if isinstance(layer, nn.Linear):
                # Initialize weights using normal distribution
                layer.weight.data.normal_(mean=0.0, std=factor * ((self.chronos_config.input_patch_size * 2) ** -0.5))
                
                # Initialize biases to zero if they exist
                if layer.bias is not None:
                    layer.bias.data.zero_()
            if isinstance(layer, nn.MultiheadAttention):
                # Initialize weights using normal distribution
                nn.init.xavier_uniform_(layer.in_proj_weight)
                nn.init.xavier_uniform_(layer.out_proj.weight)
                
                # Initialize biases to zero if they exist
                if layer.in_proj_bias is not None:
                    layer.in_proj_bias.data.zero_()
                if layer.out_proj.bias is not None:
                    layer.out_proj.bias.data.zero_()
            if isinstance(layer, nn.GRU):
                for name, param in layer.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param.data)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        param.data.zero_()

    def _run_moe_fusion(
        self,
        sequence_output: torch.Tensor,
        retrieved_y: torch.Tensor,
        r_M: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        retrieved_y_enc = []
        for i in range(r_M):
            retrieved_y_enc.append(self.encode_mlp(retrieved_y[:, i, :]))
        retrieved_y_enc = torch.stack(retrieved_y_enc, dim=1)
        all_enc = torch.cat([sequence_output, retrieved_y_enc], dim=1)
        att_output, _ = self.mha(all_enc, all_enc, all_enc)
        att_output = all_enc + att_output
        att_output = att_output + self.dropout(self.ffn(att_output))

        scores = []
        for i in range(r_M + 1):
            gate = torch.sigmoid(self.gate_layer(att_output[:, i, :]))
            scores.append(gate)
        scores = torch.stack(scores, dim=1)
        alpha = F.softmax(scores, dim=1)
        fused_sequance_output = torch.sum(alpha * att_output, dim=1)
        fused_sequance_output = self.dropout(fused_sequance_output)
        sequence_output = sequence_output + fused_sequance_output.unsqueeze(1)
        return sequence_output, retrieved_y_enc, alpha

    def _project_with_native_head(
        self,
        hidden_state: torch.Tensor,
        quantile_preds_shape: tuple[int, int, int],
    ) -> torch.Tensor:
        native_head_output = self.output_patch_embedding(hidden_state.unsqueeze(1))
        assert native_head_output.shape[-1] == self.num_quantiles * self.chronos_config.prediction_length, "output_patch_embedding output dim mismatch"
        return native_head_output.view(*quantile_preds_shape)

    def _debug_print_disentangle_shapes(self, **tensor_map: torch.Tensor) -> None:
        if not getattr(self, "debug_shapes", False) or getattr(self, "_debug_shapes_printed", False):
            return
        print("moe_disentangle debug shapes:")
        for name, tensor in tensor_map.items():
            print(f"  {name}: {tuple(tensor.shape)}")
        self._debug_shapes_printed = True

    @staticmethod
    def _zero_loss_like(reference_tensor: torch.Tensor) -> torch.Tensor:
        return reference_tensor.new_zeros(())

    def _encode_context_hidden(
        self,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is None:
            mask = torch.isnan(context).logical_not().to(context.dtype)
        else:
            mask = mask.to(context.dtype)

        if context.shape[-1] > self.chronos_config.context_length:
            context = context[..., -self.chronos_config.context_length :]
            mask = mask[..., -self.chronos_config.context_length :]

        context, _ = self.instance_norm(context)
        context = context.to(self.dtype)
        mask = mask.to(self.dtype)

        patched_context = self.patch(context)
        patched_mask = torch.nan_to_num(self.patch(mask), nan=0.0)
        patched_context[~(patched_mask > 0)] = 0.0
        patched_context = torch.cat([patched_context, patched_mask], dim=-1)
        attention_mask = patched_mask.sum(dim=-1) > 0

        input_embeds = self.input_patch_embedding(patched_context)

        if self.chronos_config.use_reg_token:
            batch_size = input_embeds.shape[0]
            reg_input_ids = torch.full(
                (batch_size, 1),
                self.config.reg_token_id,
                device=input_embeds.device,
            )
            reg_embeds = self.shared(reg_input_ids)
            input_embeds = torch.cat([input_embeds, reg_embeds], dim=-2)
            attention_mask = torch.cat([attention_mask, torch.ones_like(reg_input_ids)], dim=-1)

        encoder_outputs = self.encoder(
            attention_mask=attention_mask,
            inputs_embeds=input_embeds,
        )
        hidden_states = encoder_outputs[0]
        sequence_output = self.decode(input_embeds, attention_mask, hidden_states)
        return sequence_output.squeeze(1)

    def _debug_print_dual_projector_shapes(self, **tensor_map: torch.Tensor) -> None:
        if not getattr(self, "debug_shapes", False) or getattr(self, "_debug_shapes_printed", False):
            return
        print(f"{self.augment} debug shapes:")
        for name, tensor in tensor_map.items():
            print(f"  {name}: {tuple(tensor.shape)}")
        self._debug_shapes_printed = True

    def forward(
        self,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        retrieved_seq: Optional[torch.Tensor] = None,
        distances: Optional[torch.Tensor] = None,
    ) -> ChronosBoltOutput:
        mask = mask.to(context.dtype) if mask is not None else torch.isnan(context).logical_not().to(context.dtype)
        
        batch_size, _ = context.shape
        if context.shape[-1] > self.chronos_config.context_length:
            context = context[..., -self.chronos_config.context_length :]
            mask = mask[..., -self.chronos_config.context_length :]

        # scaling
        context, loc_scale = self.instance_norm(context)
        context = context.to(self.dtype)
        mask = mask.to(self.dtype)

        # patching
        patched_context = self.patch(context)
        patched_mask = torch.nan_to_num(self.patch(mask), nan=0.0)
        patched_context[~(patched_mask > 0)] = 0.0
        # concat context and mask along patch dim
        patched_context = torch.cat([patched_context, patched_mask], dim=-1)

        # attention_mask = 1 if at least one item in the patch is observed
        attention_mask = patched_mask.sum(dim=-1) > 0  # (batch_size, patched_seq_length)

        input_embeds = self.input_patch_embedding(patched_context)
        # import pdb; pdb.set_trace()

        if self.chronos_config.use_reg_token:
            # Append [REG]
            reg_input_ids = torch.full(
                (batch_size, 1),
                self.config.reg_token_id,
                device=input_embeds.device,
            )
            reg_embeds = self.shared(reg_input_ids)
            input_embeds = torch.cat([input_embeds, reg_embeds], dim=-2)
            attention_mask = torch.cat([attention_mask, torch.ones_like(reg_input_ids)], dim=-1)

        encoder_outputs = self.encoder(
            attention_mask=attention_mask,
            inputs_embeds=input_embeds,
        )
        hidden_states = encoder_outputs[0]

        sequence_output = self.decode(input_embeds, attention_mask, hidden_states)

        quantile_preds_shape = (
            batch_size,
            self.num_quantiles,
            self.chronos_config.prediction_length,
        )
        aux_h_ret = None
        aux_z_inv = None
        aux_z_dyn = None
        aux_y_inv = None
        aux_y_dyn = None
        dual_projector_metrics = None

        if self.augment == 'baseline':
            fused_quantile_preds = self.output_patch_embedding(sequence_output).view(*quantile_preds_shape)
        else:
            retrieved_seq, loc_scale_retrieved = self.instance_norm(retrieved_seq)

            # fuse retrieved sequence
            if 'moe' not in self.augment and self.augment != 'idf_branch' and self.augment != 'idf_x' and self.augment != 'idf_clean_dis' and self.augment != 'idf_clean_dis_deepmlp' and self.augment != 'idf_clean_dis_ts3align' and self.augment != 'idf_h_linear_head' and self.augment != 'idf_h_native_head' and self.augment != 'idf_y_linear_head' and self.augment != 'idf_residual' and self.augment != 'idf_branch_gru' and self.augment != 'idf_branch_gru_q' and self.augment != 'idf_dual_direct_head' and self.augment != 'idf_dual_projector' and self.augment != 'idf_dual_projector_mlp':
                weights = torch.softmax(-distances, dim=1)
                retrieved_seq = (weights.unsqueeze(-1) * retrieved_seq).sum(dim=1)
                retrieved_seq = retrieved_seq.unsqueeze(1)
            # B, L = target.shape
            L = self.chronos_config.prediction_length if self.augment in ['idf_branch', 'idf_x', 'idf_clean_dis', 'idf_clean_dis_deepmlp', 'idf_clean_dis_ts3align', 'idf_h_linear_head', 'idf_h_native_head', 'idf_y_linear_head', 'idf_residual', 'idf_branch_gru', 'idf_branch_gru_q', 'idf_dual_direct_head', 'idf_dual_projector', 'idf_dual_projector_mlp'] else 64
            r_B, r_M, r_L = retrieved_seq.shape
            assert r_L % 2 == 0, "L of retrieved_seq should be even"
            retrieved_x, retrieved_y = retrieved_seq.split((r_L-L, L), dim=2)
            retrieved_seq = retrieved_seq.to(self.dtype)

            if self.augment == 'moe':
                sequence_output, _, _ = self._run_moe_fusion(sequence_output, retrieved_y, r_M)

            if self.augment == 'moe_disentangle':
                h_q = sequence_output.squeeze(1)
                sequence_output, retrieved_y_enc, _ = self._run_moe_fusion(sequence_output, retrieved_y, r_M)
                e_final = sequence_output.squeeze(1)
                h_r = retrieved_y_enc

                sim = torch.sum(h_q.unsqueeze(1) * h_r, dim=-1)
                omega = F.softmax(sim, dim=1)
                h_ret = torch.sum(omega.unsqueeze(-1) * h_r, dim=1)
                assert h_ret.shape[-1] == e_final.shape[-1], "h_ret and e_final dimension mismatch"

                gate_in = torch.cat([e_final, h_ret], dim=-1)
                gamma = torch.sigmoid(self.disentangle_gate(gate_in))
                z_inv = gamma * e_final
                z_dyn = (1 - gamma) * e_final

                y_inv = self._project_with_native_head(z_inv, quantile_preds_shape)
                y_dyn = self._project_with_native_head(z_dyn, quantile_preds_shape)
                concat_y = torch.cat([y_inv, y_dyn], dim=-1)
                y_hat = self.final_pred_head(concat_y)
                fused_quantile_preds = y_hat
                aux_h_ret = h_ret
                aux_z_inv = z_inv
                aux_z_dyn = z_dyn
                aux_y_inv = y_inv
                aux_y_dyn = y_dyn

                self._debug_print_disentangle_shapes(
                    h_q=h_q,
                    h_r=h_r,
                    e_final=e_final,
                    h_ret=h_ret,
                    gamma=gamma,
                    z_inv=z_inv,
                    z_dyn=z_dyn,
                    y_inv=y_inv,
                    y_dyn=y_dyn,
                    y_hat=y_hat,
                )

            if self.augment == 'idf':
                retrieved_y_enc = []
                for i in range(r_M):
                    retrieved_y_enc.append(self.encode_mlp(retrieved_y[:, i, :]))
                retrieved_y_enc = torch.stack(retrieved_y_enc, dim=1)

                q = sequence_output.squeeze(1)
                q_expand = q.unsqueeze(1).expand(-1, r_M, -1)
                ret_score_in = torch.cat([q_expand, retrieved_y_enc], dim=-1)
                ret_score = self.ret_score_head(ret_score_in)
                alpha = F.softmax(ret_score, dim=1)
                weighted_R = alpha * retrieved_y_enc
                h_ret = weighted_R.sum(dim=1)

                fuse_in = torch.cat([q, h_ret], dim=-1)
                fuse_lambda = torch.sigmoid(self.fuse_gate(fuse_in))
                h = fuse_lambda * q + (1 - fuse_lambda) * h_ret

                route_in = torch.cat([h, h_ret], dim=-1)
                gamma = torch.sigmoid(self.routing_gate(route_in))
                z_inv = gamma * h
                z_dyn = (1 - gamma) * h

                z_inv_next = self.inv_transition(z_inv)
                dyn_in = torch.cat([z_dyn, h_ret], dim=-1)
                z_dyn_next = self.dyn_transition(dyn_in)

                h_inv = self.inv_head(z_inv_next)
                h_dyn = self.dyn_head(z_dyn_next)

                final_in = torch.cat([h_inv, h_dyn], dim=-1)
                idf_delta = self.final_head(final_in)
                sequence_output = sequence_output + self.dropout(idf_delta).unsqueeze(1)

            quantile_preds = self.output_patch_embedding(sequence_output).view(*quantile_preds_shape)
            if self.augment != 'moe_disentangle':
                fused_quantile_preds = quantile_preds

            if self.augment in ['idf_branch', 'idf_x']:
                retrieved_y_enc = []
                for i in range(r_M):
                    retrieved_y_enc.append(self.encode_mlp(retrieved_y[:, i, :]))
                retrieved_y_enc = torch.stack(retrieved_y_enc, dim=1)

                q = sequence_output.squeeze(1)
                q_expand = q.unsqueeze(1).expand(-1, r_M, -1)
                ret_score_in = torch.cat([q_expand, retrieved_y_enc], dim=-1)
                alpha = F.softmax(self.ret_score_head(ret_score_in), dim=1)
                h_ret = (alpha * retrieved_y_enc).sum(dim=1)

                fuse_in = torch.cat([q, h_ret], dim=-1)
                fuse_lambda = torch.sigmoid(self.fuse_gate(fuse_in))
                h = fuse_lambda * q + (1 - fuse_lambda) * h_ret

                route_in = torch.cat([h, h_ret], dim=-1)
                gamma = torch.sigmoid(self.routing_gate(route_in))
                z_inv = gamma * h
                z_dyn = (1 - gamma) * h

                y_inv_base = self.inv_pred_head(z_inv)
                y_inv_res = self.inv_residual_head(z_inv)
                y_inv = (y_inv_base + y_inv_res).view(*quantile_preds_shape)

                dyn_in = torch.cat([z_dyn, h_ret], dim=-1)
                y_dyn = self.dyn_pred_head(dyn_in).view(*quantile_preds_shape)

                final_in = torch.cat([y_inv.reshape(batch_size, -1), y_dyn.reshape(batch_size, -1)], dim=-1)
                fused_quantile_preds = self.final_pred_head(final_in).view(*quantile_preds_shape)
                aux_h_ret = h_ret
                aux_z_inv = z_inv
                aux_z_dyn = z_dyn
                aux_y_inv = y_inv
                aux_y_dyn = y_dyn

            if self.augment in ['idf_clean_dis', 'idf_clean_dis_deepmlp']:
                retrieved_y_enc = []
                for i in range(r_M):
                    retrieved_y_enc.append(self.encode_mlp(retrieved_y[:, i, :]))
                retrieved_y_enc = torch.stack(retrieved_y_enc, dim=1)

                q = sequence_output.squeeze(1)
                q_expand = q.unsqueeze(1).expand(-1, r_M, -1)
                ret_score_in = torch.cat([q_expand, retrieved_y_enc], dim=-1)
                alpha = F.softmax(self.ret_score_head(ret_score_in), dim=1)
                h_ret = (alpha * retrieved_y_enc).sum(dim=1)

                fuse_in = torch.cat([q, h_ret], dim=-1)
                fuse_lambda = torch.sigmoid(self.fuse_gate(fuse_in))
                h = fuse_lambda * q + (1 - fuse_lambda) * h_ret

                route_in = torch.cat([h, h_ret], dim=-1)
                gamma = torch.sigmoid(self.routing_gate(route_in))
                z_inv = gamma * h
                z_dyn = (1 - gamma) * h

                y_inv = self.inv_pred_head(z_inv).view(*quantile_preds_shape)
                y_dyn = self.dyn_pred_head_clean(z_dyn).view(*quantile_preds_shape)

                final_in = torch.cat([y_inv.reshape(batch_size, -1), y_dyn.reshape(batch_size, -1)], dim=-1)
                fused_quantile_preds = self.final_pred_head(final_in).view(*quantile_preds_shape)
                aux_h_ret = h_ret
                aux_z_inv = z_inv
                aux_z_dyn = z_dyn
                aux_y_inv = y_inv
                aux_y_dyn = y_dyn

            if self.augment == 'idf_clean_dis_ts3align':
                retrieved_y_enc = []
                for i in range(r_M):
                    retrieved_y_enc.append(self.encode_mlp(retrieved_y[:, i, :]))
                retrieved_y_enc = torch.stack(retrieved_y_enc, dim=1)

                q = sequence_output.squeeze(1)
                h_ret = retrieved_y_enc.mean(dim=1)

                fuse_in = torch.cat([q, h_ret], dim=-1)
                fuse_lambda = torch.sigmoid(self.fuse_gate(fuse_in))
                h = fuse_lambda * q + (1 - fuse_lambda) * h_ret

                route_in = torch.cat([h, h_ret], dim=-1)
                gamma = torch.sigmoid(self.routing_gate(route_in))
                z_inv = gamma * h
                z_dyn = (1 - gamma) * h

                y_inv = self.inv_pred_head(z_inv).view(*quantile_preds_shape)
                y_dyn = self.dyn_pred_head_clean(z_dyn).view(*quantile_preds_shape)

                final_in = torch.cat([y_inv.reshape(batch_size, -1), y_dyn.reshape(batch_size, -1)], dim=-1)
                fused_quantile_preds = self.final_pred_head(final_in).view(*quantile_preds_shape)
                aux_h_ret = h_ret
                aux_z_inv = z_inv
                aux_z_dyn = z_dyn
                aux_y_inv = y_inv
                aux_y_dyn = y_dyn

            if self.augment == 'idf_h_linear_head':
                retrieved_y_enc = []
                for i in range(r_M):
                    retrieved_y_enc.append(self.encode_mlp(retrieved_y[:, i, :]))
                retrieved_y_enc = torch.stack(retrieved_y_enc, dim=1)

                q = sequence_output.squeeze(1)
                q_expand = q.unsqueeze(1).expand(-1, r_M, -1)
                ret_score_in = torch.cat([q_expand, retrieved_y_enc], dim=-1)
                alpha = F.softmax(self.ret_score_head(ret_score_in), dim=1)
                h_ret = (alpha * retrieved_y_enc).sum(dim=1)

                fuse_in = torch.cat([q, h_ret], dim=-1)
                fuse_lambda = torch.sigmoid(self.fuse_gate(fuse_in))
                h = fuse_lambda * q + (1 - fuse_lambda) * h_ret

                fused_quantile_preds = self.h_pred_head(h).view(*quantile_preds_shape)

            if self.augment == 'idf_h_native_head':
                retrieved_y_enc = []
                for i in range(r_M):
                    retrieved_y_enc.append(self.encode_mlp(retrieved_y[:, i, :]))
                retrieved_y_enc = torch.stack(retrieved_y_enc, dim=1)

                q = sequence_output.squeeze(1)
                q_expand = q.unsqueeze(1).expand(-1, r_M, -1)
                ret_score_in = torch.cat([q_expand, retrieved_y_enc], dim=-1)
                alpha = F.softmax(self.ret_score_head(ret_score_in), dim=1)
                h_ret = (alpha * retrieved_y_enc).sum(dim=1)

                fuse_in = torch.cat([q, h_ret], dim=-1)
                fuse_lambda = torch.sigmoid(self.fuse_gate(fuse_in))
                h = fuse_lambda * q + (1 - fuse_lambda) * h_ret

                # Keep the native Chronos output head frozen by default and verify the
                # fused hidden state matches its expected [B, 1, d_model] input shape.
                assert h.shape == (batch_size, self.config.d_model), "idf_h_native_head expects h to have shape [B, d_model]"
                h_native = h.unsqueeze(1)
                assert h_native.shape == (batch_size, 1, self.config.d_model), "idf_h_native_head expects h.unsqueeze(1) to have shape [B, 1, d_model]"
                native_head_output = self.output_patch_embedding(h_native)
                assert native_head_output.shape[-1] == self.num_quantiles * self.chronos_config.prediction_length, "output_patch_embedding output dim mismatch"
                fused_quantile_preds = native_head_output.view(*quantile_preds_shape)

            if self.augment == 'idf_y_linear_head':
                retrieved_y_enc = []
                for i in range(r_M):
                    retrieved_y_enc.append(self.encode_mlp(retrieved_y[:, i, :]))
                retrieved_y_enc = torch.stack(retrieved_y_enc, dim=1)

                q = sequence_output.squeeze(1)
                q_expand = q.unsqueeze(1).expand(-1, r_M, -1)
                ret_score_in = torch.cat([q_expand, retrieved_y_enc], dim=-1)
                alpha = F.softmax(self.ret_score_head(ret_score_in), dim=1)
                h_ret = (alpha * retrieved_y_enc).sum(dim=1)

                fuse_in = torch.cat([q, h_ret], dim=-1)
                fuse_lambda = torch.sigmoid(self.fuse_gate(fuse_in))
                h = fuse_lambda * q + (1 - fuse_lambda) * h_ret

                y0 = self.h_pred_head(h)
                y_hat = self.y_linear_head(y0)

                fused_quantile_preds = y_hat.view(*quantile_preds_shape)

            if self.augment == 'idf_branch_gru':
                retrieved_y_enc = []
                for i in range(r_M):
                    retrieved_y_enc.append(self.encode_mlp(retrieved_y[:, i, :]))
                retrieved_y_enc = torch.stack(retrieved_y_enc, dim=1)

                q = sequence_output.squeeze(1)
                q_expand = q.unsqueeze(1).expand(-1, r_M, -1)
                ret_score_in = torch.cat([q_expand, retrieved_y_enc], dim=-1)
                alpha = F.softmax(self.ret_score_head(ret_score_in), dim=1)
                h_ret = (alpha * retrieved_y_enc).sum(dim=1)

                fuse_in = torch.cat([q, h_ret], dim=-1)
                fuse_lambda = torch.sigmoid(self.fuse_gate(fuse_in))
                h = fuse_lambda * q + (1 - fuse_lambda) * h_ret

                route_in = torch.cat([h, h_ret], dim=-1)
                gamma = torch.sigmoid(self.routing_gate(route_in))
                z_inv = gamma * h
                z_dyn = (1 - gamma) * h

                y_inv_base = self.inv_pred_head(z_inv)
                y_inv_res = self.inv_residual_head(z_inv)
                y_inv = (y_inv_base + y_inv_res).view(*quantile_preds_shape)

                dyn_in = torch.cat([z_dyn, h_ret], dim=-1)
                dyn_seq = dyn_in.unsqueeze(1).repeat(1, self.chronos_config.prediction_length, 1)
                gru_out, _ = self.dyn_gru(dyn_seq)
                y_dyn = self.dyn_out_head(gru_out).transpose(1, 2)

                final_in = torch.cat([y_inv.reshape(batch_size, -1), y_dyn.reshape(batch_size, -1)], dim=-1)
                fused_quantile_preds = self.final_pred_head(final_in).view(*quantile_preds_shape)

            if self.augment == 'idf_branch_gru_q':
                retrieved_y_enc = []
                for i in range(r_M):
                    retrieved_y_enc.append(self.encode_mlp(retrieved_y[:, i, :]))
                retrieved_y_enc = torch.stack(retrieved_y_enc, dim=1)

                q = sequence_output.squeeze(1)
                q_expand = q.unsqueeze(1).expand(-1, r_M, -1)
                ret_score_in = torch.cat([q_expand, retrieved_y_enc], dim=-1)
                alpha = F.softmax(self.ret_score_head(ret_score_in), dim=1)
                h_ret = (alpha * retrieved_y_enc).sum(dim=1)

                fuse_in = torch.cat([q, h_ret], dim=-1)
                fuse_lambda = torch.sigmoid(self.fuse_gate(fuse_in))
                h = fuse_lambda * q + (1 - fuse_lambda) * h_ret

                route_in = torch.cat([h, h_ret], dim=-1)
                gamma = torch.sigmoid(self.routing_gate(route_in))
                z_inv = gamma * h
                z_dyn = (1 - gamma) * h

                y_inv_base = self.inv_pred_head(z_inv)
                y_inv_res = self.inv_residual_head(z_inv)
                y_inv = (y_inv_base + y_inv_res).view(*quantile_preds_shape)

                dyn_base = torch.cat([z_dyn, h_ret], dim=-1)
                dyn_base_seq = dyn_base.unsqueeze(1).repeat(1, self.chronos_config.prediction_length, 1)
                quantile_step = quantile_preds.transpose(1, 2)
                dyn_seq = torch.cat([dyn_base_seq, quantile_step], dim=-1)
                gru_out, _ = self.dyn_gru(dyn_seq)
                y_dyn = self.dyn_out_head(gru_out).transpose(1, 2)

                final_in = torch.cat([y_inv.reshape(batch_size, -1), y_dyn.reshape(batch_size, -1)], dim=-1)
                fused_quantile_preds = self.final_pred_head(final_in).view(*quantile_preds_shape)

            if self.augment == 'idf_dual_direct_head':
                retrieved_y_enc = []
                for i in range(r_M):
                    retrieved_y_enc.append(self.encode_mlp(retrieved_y[:, i, :]))
                retrieved_y_enc = torch.stack(retrieved_y_enc, dim=1)

                q = sequence_output.squeeze(1)
                q_expand = q.unsqueeze(1).expand(-1, r_M, -1)
                ret_score_in = torch.cat([q_expand, retrieved_y_enc], dim=-1)
                alpha = F.softmax(self.ret_score_head(ret_score_in), dim=1)
                h_ret = (alpha * retrieved_y_enc).sum(dim=1)

                fuse_in = torch.cat([q, h_ret], dim=-1)
                fuse_lambda = torch.sigmoid(self.fuse_gate(fuse_in))
                h = fuse_lambda * q + (1 - fuse_lambda) * h_ret

                route_in = torch.cat([h, h_ret], dim=-1)
                gamma = torch.sigmoid(self.routing_gate(route_in))
                z_inv = gamma * h
                z_dyn = (1 - gamma) * h

                h_inv = self.inv_hidden_proj(z_inv)
                h_dyn = self.dyn_ret_hidden_proj(torch.cat([z_dyn, h_ret], dim=-1))

                y_inv = self.output_patch_embedding(h_inv.unsqueeze(1)).view(*quantile_preds_shape)
                y_dyn = self.output_patch_embedding(h_dyn.unsqueeze(1)).view(*quantile_preds_shape)

                final_in = torch.cat([y_inv.reshape(batch_size, -1), y_dyn.reshape(batch_size, -1)], dim=-1)
                fused_quantile_preds = self.final_pred_head(final_in).view(*quantile_preds_shape)

            if self.augment == 'idf_residual':
                retrieved_y_enc = []
                for i in range(r_M):
                    retrieved_y_enc.append(self.encode_mlp(retrieved_y[:, i, :]))
                retrieved_y_enc = torch.stack(retrieved_y_enc, dim=1)

                q = sequence_output.squeeze(1)
                q_expand = q.unsqueeze(1).expand(-1, r_M, -1)
                ret_score_in = torch.cat([q_expand, retrieved_y_enc], dim=-1)
                alpha = F.softmax(self.ret_score_head(ret_score_in), dim=1)
                h_ret = (alpha * retrieved_y_enc).sum(dim=1)

                fuse_in = torch.cat([q, h_ret], dim=-1)
                fuse_lambda = torch.sigmoid(self.fuse_gate(fuse_in))
                h = fuse_lambda * q + (1 - fuse_lambda) * h_ret

                route_in = torch.cat([h, h_ret], dim=-1)
                gamma = torch.sigmoid(self.routing_gate(route_in))
                z_inv = gamma * h
                z_dyn = (1 - gamma) * h

                delta_inv = self.inv_pred_head(z_inv).view(*quantile_preds_shape)
                dyn_in = torch.cat([z_dyn, h_ret], dim=-1)
                delta_dyn = self.dyn_pred_head(dyn_in).view(*quantile_preds_shape)
                delta = torch.tanh(delta_inv + delta_dyn) * 2

                fused_quantile_preds = quantile_preds + delta

            if self.augment == 'idf_dual_projector':
                tau = float(getattr(self, "tau", 0.1))
                tau = max(tau, 1e-6)

                q = sequence_output.squeeze(1)
                q_i = self._encode_context_hidden(retrieved_x.reshape(batch_size * r_M, -1)).reshape(batch_size, r_M, -1)

                q_expand = q.unsqueeze(1).expand(-1, r_M, -1)
                score = F.cosine_similarity(q_expand, q_i, dim=-1).unsqueeze(-1) / tau
                alpha = torch.softmax(score, dim=1)
                h_ret = (alpha * q_i).sum(dim=1)

                fuse_in = torch.cat([q, h_ret], dim=-1)
                fuse_lambda = torch.sigmoid(self.fuse_gate(fuse_in))
                h = fuse_lambda * q + (1 - fuse_lambda) * h_ret

                r_ret = q - h_ret
                z_inv = self.g_inv(torch.cat([h, h_ret], dim=-1))
                z_dyn = self.g_dyn(torch.cat([h, r_ret], dim=-1))

                h_ret_expand = h_ret.unsqueeze(1).expand(-1, r_M, -1)
                z_inv_i = self.g_inv(torch.cat([q_i, h_ret_expand], dim=-1).reshape(batch_size * r_M, -1)).reshape(batch_size, r_M, -1)

                y_inv = self.f_inv(z_inv).view(*quantile_preds_shape)
                y_dyn = self.f_dyn(z_dyn).view(*quantile_preds_shape)
                y_hat = y_inv + y_dyn
                fused_quantile_preds = y_hat

                dual_projector_metrics = {
                    "q": q,
                    "q_i": q_i,
                    "alpha": alpha,
                    "h_ret": h_ret,
                    "h": h,
                    "z_inv": z_inv,
                    "z_dyn": z_dyn,
                    "z_inv_i": z_inv_i,
                    "y_inv": y_inv,
                    "y_dyn": y_dyn,
                    "y_hat": y_hat,
                }

                self._debug_print_dual_projector_shapes(**dual_projector_metrics)

            if self.augment == 'idf_dual_projector_mlp':
                tau = float(getattr(self, "tau", 0.1))
                tau = max(tau, 1e-6)

                q = sequence_output.squeeze(1)
                q_i = self.retrieved_x_encoder_mlp(retrieved_x)

                q_expand = q.unsqueeze(1).expand(-1, r_M, -1)
                score = F.cosine_similarity(q_expand, q_i, dim=-1).unsqueeze(-1) / tau
                alpha = torch.softmax(score, dim=1)
                h_ret = (alpha * q_i).sum(dim=1)

                fuse_in = torch.cat([q, h_ret], dim=-1)
                fuse_lambda = torch.sigmoid(self.fuse_gate(fuse_in))
                h = fuse_lambda * q + (1 - fuse_lambda) * h_ret

                r_ret = q - h_ret
                z_inv = self.g_inv(torch.cat([h, h_ret], dim=-1))
                z_dyn = self.g_dyn(torch.cat([h, r_ret], dim=-1))

                h_ret_expand = h_ret.unsqueeze(1).expand(-1, r_M, -1)
                z_inv_i = self.g_inv(torch.cat([q_i, h_ret_expand], dim=-1).reshape(batch_size * r_M, -1)).reshape(batch_size, r_M, -1)

                y_inv = self.f_inv(z_inv).view(*quantile_preds_shape)
                y_dyn = self.f_dyn(z_dyn).view(*quantile_preds_shape)
                y_hat = y_inv + y_dyn
                fused_quantile_preds = y_hat

                dual_projector_metrics = {
                    "q": q,
                    "q_i": q_i,
                    "alpha": alpha,
                    "h_ret": h_ret,
                    "h": h,
                    "z_inv": z_inv,
                    "z_dyn": z_dyn,
                    "z_inv_i": z_inv_i,
                    "y_inv": y_inv,
                    "y_dyn": y_dyn,
                    "y_hat": y_hat,
                }

                self._debug_print_dual_projector_shapes(**dual_projector_metrics)

            if self.augment == 'gate':
                # Step 1
                retrieved_y = retrieved_y.repeat(1, self.num_quantiles, 1)
                x = context.unsqueeze(1).repeat(1, self.num_quantiles, 1)
                stats_preds = compute_time_series_stats(quantile_preds, dim=2, keepdim=True)
                stats_retrieved = compute_time_series_stats(retrieved_y, dim=2, keepdim=True)
                stats_x = compute_time_series_stats(x, dim=2, keepdim=True)
                # Step 2: Concatenate x, retrieved_y, quantile_preds
                concat_preds = torch.cat([quantile_preds, retrieved_y, x, stats_preds, stats_retrieved, stats_x], dim=-1).view(batch_size, self.num_quantiles, -1)
                gate = self.gate_layer(concat_preds) + self.gate_linear1(concat_preds)
                gate = torch.sigmoid(self.gate_linear2(gate))
                # Step 3: Fuse predictions
                fused_quantile_preds = gate * quantile_preds + (1 - gate) * retrieved_y

        loss = None
        loss_forecast = None
        loss_cons = None
        loss_smooth = None
        loss_inv = None
        loss_dis = None
        loss_ret = None
        loss_dyn = None
        use_disentangle_aux_loss = False
        aux_loss_enabled = False
        if target is not None:
            # normalize target
            target, _ = self.instance_norm(target, loc_scale)
            target = target.unsqueeze(1)  # type: ignore
            assert self.chronos_config.prediction_length >= target.shape[-1]

            target = target.to(fused_quantile_preds.device)
            target_mask = (
                target_mask.unsqueeze(1).to(fused_quantile_preds.device) if target_mask is not None else ~torch.isnan(target)
            )
            target[~target_mask] = 0.0

            # pad target and target_mask if they are shorter than model's prediction_length
            if self.chronos_config.prediction_length > target.shape[-1]:
                padding_shape = (*target.shape[:-1], self.chronos_config.prediction_length - target.shape[-1])
                target = torch.cat([target, torch.zeros(padding_shape).to(target)], dim=-1)
                target_mask = torch.cat([target_mask, torch.zeros(padding_shape).to(target_mask)], dim=-1)

            loss = (
                2
                * torch.abs(
                    (target - fused_quantile_preds)
                    * ((target <= fused_quantile_preds).float() - self.quantiles.view(1, self.num_quantiles, 1))
                )
                * target_mask.float()
            )
            loss = loss.mean(dim=-2)  # Mean over prediction horizon
            loss = loss.sum(dim=-1)  # Sum over quantile levels
            loss = loss.mean()  # Mean over batch

            loss_forecast = loss
            loss_cons = self._zero_loss_like(loss_forecast)
            loss_smooth = self._zero_loss_like(loss_forecast)
            loss_inv = self._zero_loss_like(loss_forecast)
            loss_dis = self._zero_loss_like(loss_forecast)
            loss_ret = self._zero_loss_like(loss_forecast)
            loss_dyn = self._zero_loss_like(loss_forecast)

            if self.augment in ['idf_dual_projector', 'idf_dual_projector_mlp']:
                assert dual_projector_metrics is not None, "idf_dual_projector metrics should be populated during forward"
                y_inv = dual_projector_metrics["y_inv"]
                y_dyn = dual_projector_metrics["y_dyn"]
                z_inv = dual_projector_metrics["z_inv"]
                z_dyn = dual_projector_metrics["z_dyn"]
                z_inv_i = dual_projector_metrics["z_inv_i"]

                z_inv_expand = z_inv.unsqueeze(1)
                loss_cons = ((z_inv_expand - z_inv_i) ** 2).sum(dim=-1).mean(dim=1).mean()

                y_inv_diff = y_inv[..., 1:] - y_inv[..., :-1]
                loss_smooth = (y_inv_diff ** 2).mean()

                loss_dis = ((z_inv * z_dyn).sum(dim=-1) ** 2).mean()

                res_target = target - y_inv.detach()
                loss_dyn = ((y_dyn - res_target) ** 2).mean()

                lambda1 = float(getattr(self, "lambda1", 0.0))
                lambda2 = float(getattr(self, "lambda2", 0.0))
                loss = loss_forecast + lambda1 * (loss_cons + loss_smooth + loss_dis) + lambda2 * loss_dyn

            use_disentangle_aux_loss = all(
                tensor is not None for tensor in (aux_h_ret, aux_z_inv, aux_z_dyn, aux_y_inv, aux_y_dyn)
            )
            rho1 = float(getattr(self, "rho1", 0.0))
            rho2 = float(getattr(self, "rho2", 0.0))
            rho3 = float(getattr(self, "rho3", 0.0))
            rho4 = float(getattr(self, "rho4", 0.0))
            aux_loss_enabled = (
                use_disentangle_aux_loss and any(rho > 0.0 for rho in (rho1, rho2, rho3, rho4))
            ) or (
                self.augment == 'idf_dual_projector'
                and any(v > 0.0 for v in (
                    float(getattr(self, "lambda1", 0.0)),
                    float(getattr(self, "lambda2", 0.0)),
                ))
            ) or (
                self.augment == 'idf_dual_projector_mlp'
                and any(v > 0.0 for v in (
                    float(getattr(self, "lambda1", 0.0)),
                    float(getattr(self, "lambda2", 0.0)),
                ))
            )

            if aux_loss_enabled:
                if self.augment not in ['idf_dual_projector', 'idf_dual_projector_mlp']:
                    inv_target = aux_z_inv.detach()
                    dyn_target = aux_z_dyn.detach()
                    y_inv_hidden = self.inv_aux_backproj(aux_y_inv.reshape(batch_size, -1))
                    y_dyn_hidden = self.dyn_aux_backproj(aux_y_dyn.reshape(batch_size, -1))

                    loss_inv = F.mse_loss(y_inv_hidden, inv_target)

                    z_inv_n = F.normalize(aux_z_inv.flatten(1), dim=-1)
                    z_dyn_n = F.normalize(aux_z_dyn.flatten(1), dim=-1)
                    loss_dis = (z_inv_n * z_dyn_n).sum(dim=-1).abs().mean()

                    ret_target = aux_h_ret.detach() if getattr(self, "aux_loss_detach_ret", True) else aux_h_ret
                    loss_ret = F.mse_loss(aux_z_inv, ret_target)

                    mse_dyn = F.mse_loss(y_dyn_hidden, dyn_target)
                    dyn_margin = float(getattr(self, "dyn_margin", 1.0))
                    loss_dyn = -torch.clamp(mse_dyn, max=dyn_margin)
                    loss = loss_forecast + rho1 * loss_inv + rho2 * loss_dis + rho3 * loss_ret + rho4 * loss_dyn

        # Unscale predictions
        fused_quantile_preds = self.instance_norm.inverse(
            fused_quantile_preds.view(batch_size, -1),
            loc_scale,
        ).view(*quantile_preds_shape)

        return ChronosBoltOutput(
            loss=loss,
            total_loss=loss,
            loss_forecast=loss_forecast,
            loss_cons=loss_cons,
            loss_smooth=loss_smooth,
            loss_inv=loss_inv,
            loss_dis=loss_dis,
            loss_ret=loss_ret,
            loss_dyn=loss_dyn,
            use_disentangle_aux_loss=use_disentangle_aux_loss,
            aux_loss_enabled=aux_loss_enabled,
            quantile_preds=fused_quantile_preds,
        )

    def _init_decoder(self, config):
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

    def decode(
        self,
        input_embeds,
        attention_mask,
        hidden_states,
        output_attentions=False,
    ):
        """
        Parameters
        ----------
        input_embeds: torch.Tensor
            Patched and embedded inputs. Shape (batch_size, patched_context_length, d_model)
        attention_mask: torch.Tensor
            Attention mask for the patched context. Shape (batch_size, patched_context_length), type: torch.int64
        hidden_states: torch.Tensor
            Hidden states returned by the encoder. Shape (batch_size, patched_context_length, d_model)

        Returns
        -------
        last_hidden_state
            Last hidden state returned by the decoder, of shape (batch_size, 1, d_model)
        """
        batch_size = input_embeds.shape[0]
        decoder_input_ids = torch.full(
            (batch_size, 1),
            self.config.decoder_start_token_id,
            device=input_embeds.device,
        )
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            output_attentions=output_attentions,
            return_dict=True,
        )

        return decoder_outputs.last_hidden_state  # sequence_outputs, b x 1 x d_model


class ChronosBoltPipelineWithRetrieval(BaseChronosPipeline):
    forecast_type: ForecastType = ForecastType.QUANTILES
    default_context_length: int = 2048
    # register this class name with this alias for backward compatibility
    _aliases = ["PatchedT5Pipeline"]

    def __init__(self, model: ChronosBoltModelForForecastingWithRetrieval):
        super().__init__(inner_model=model)
        self.model = model

    @property
    def quantiles(self) -> List[float]:
        return self.model.config.chronos_config["quantiles"]

    def predict(  # type: ignore[override]
        self,
        context: Union[torch.Tensor, List[torch.Tensor]],
        prediction_length: Optional[int] = None,
        limit_prediction_length: bool = False,
        retrieved_seq: Optional[torch.Tensor] = None,
        distances: Optional[torch.Tensor] = None,
    ):
        context_tensor = self._prepare_and_validate_context(context=context)

        model_context_length = self.model.config.chronos_config["context_length"]
        model_prediction_length = self.model.config.chronos_config["prediction_length"]
        if prediction_length is None:
            prediction_length = model_prediction_length

        if prediction_length > model_prediction_length:
            msg = (
                f"We recommend keeping prediction length <= {model_prediction_length}. "
                "The quality of longer predictions may degrade since the model is not optimized for it. "
            )
            if limit_prediction_length:
                msg += "You can turn off this check by setting `limit_prediction_length=False`."
                raise ValueError(msg)
            warnings.warn(msg)

        predictions = []
        remaining = prediction_length

        # We truncate the context here because otherwise batches with very long
        # context could take up large amounts of GPU memory unnecessarily.
        if context_tensor.shape[-1] > model_context_length:
            context_tensor = context_tensor[..., -model_context_length:]

        # TODO: We unroll the forecast of Chronos Bolt greedily with the full forecast
        # horizon that the model was trained with (i.e., 64). This results in variance collapsing
        # every 64 steps.
        while remaining > 0:
            with torch.no_grad():
                prediction = self.model(
                    context=context_tensor.to(
                        device=self.model.device,
                        dtype=torch.float32,  # scaling should be done in 32-bit precision
                    ),
                    retrieved_seq = retrieved_seq,
                    distances = distances,
                ).quantile_preds.to(context_tensor)

            predictions.append(prediction)
            remaining -= prediction.shape[-1]

            if remaining <= 0:
                break

            central_idx = torch.abs(torch.tensor(self.quantiles) - 0.5).argmin()
            central_prediction = prediction[:, central_idx]

            context_tensor = torch.cat([context_tensor, central_prediction], dim=-1)

        return torch.cat(predictions, dim=-1)[..., :prediction_length]

    def predict_quantiles(
        self, context: torch.Tensor, prediction_length: int, quantile_levels: List[float],
        retrieved_seq: Optional[torch.Tensor] = None,
        distances: Optional[torch.Tensor] = None, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # shape (batch_size, prediction_length, len(training_quantile_levels))
        predictions = (
            self.predict(
                context,
                prediction_length=prediction_length,
                retrieved_seq = retrieved_seq,
                distances = distances,
            )
            .detach()
            .cpu()
            .swapaxes(1, 2)
        )

        training_quantile_levels = self.quantiles

        if set(quantile_levels).issubset(set(training_quantile_levels)):
            # no need to perform intra/extrapolation
            quantiles = predictions[..., [training_quantile_levels.index(q) for q in quantile_levels]]
        else:
            # we rely on torch for interpolating quantiles if quantiles that
            # Chronos Bolt was trained on were not provided
            if min(quantile_levels) < min(training_quantile_levels) or max(quantile_levels) > max(
                training_quantile_levels
            ):
                logger.warning(
                    f"\tQuantiles to be predicted ({quantile_levels}) are not within the range of "
                    f"quantiles that Chronos-Bolt was trained on ({training_quantile_levels}). "
                    "Quantile predictions will be set to the minimum/maximum levels at which Chronos-Bolt "
                    "was trained on. This may significantly affect the quality of the predictions."
                )

            # TODO: this is a hack that assumes the model's quantiles during training (training_quantile_levels)
            # made up an equidistant grid along the quantile dimension. i.e., they were (0.1, 0.2, ..., 0.9).
            # While this holds for official Chronos-Bolt models, this may not be true in the future, and this
            # function may have to be revised.
            augmented_predictions = torch.cat(
                [predictions[..., [0]], predictions, predictions[..., [-1]]],
                dim=-1,
            )
            quantiles = torch.quantile(
                augmented_predictions, q=torch.tensor(quantile_levels, dtype=augmented_predictions.dtype), dim=-1
            ).permute(1, 2, 0)
        mean = predictions[:, :, training_quantile_levels.index(0.5)]
        return quantiles, mean

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """
        Load the model, either from a local path or from the HuggingFace Hub.
        Supports the same arguments as ``AutoConfig`` and ``AutoModel``
        from ``transformers``.
        """
        # if optimization_strategy is provided, pop this as it won't be used
        kwargs.pop("optimization_strategy", None)

        config = AutoConfig.from_pretrained(*args, **kwargs)
        assert hasattr(config, "chronos_config"), "Not a Chronos config file"

        context_length = kwargs.pop("context_length", None)
        if context_length is not None:
            config.chronos_config["context_length"] = context_length

        architecture = config.architectures[0]
        class_ = globals().get(architecture)

        # TODO: remove this once all models carry the correct architecture names in their configuration
        # and raise an error instead.
        if class_ is None:
            logger.warning(f"Unknown architecture: {architecture}, defaulting to ChronosBoltModelForForecastingWithRetrieval")
            class_ = ChronosBoltModelForForecastingWithRetrieval

        model = class_.from_pretrained(*args, **kwargs)
        return cls(model=model)
