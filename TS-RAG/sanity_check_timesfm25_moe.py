"""
Sanity check for models/TimesFM25.py's TimesFM25MoEModelForForecastingWithRetrieval
(the 'moe' fusion port) -- NOT a training script. See sanity_check_timesfm25.py
for the RIDDE (idf_clean_dis) variant; this mirrors it for the MoE head.
"""

import torch

from models.TimesFM25 import MOE_HEAD_LAYER_NAMES, TimesFM25MoEModelForForecastingWithRetrieval

torch.manual_seed(0)

BATCH_SIZE = 4
CONTEXT_LENGTH = 512
PREDICTION_LENGTH = 64
TOP_K = 5
RETRIEVED_WINDOW_LENGTH = 128  # must be >= PREDICTION_LENGTH


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    model = TimesFM25MoEModelForForecastingWithRetrieval(
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
    ).to(device)
    model.train()

    trainable, frozen = [], []
    for name, p in model.named_parameters():
        (trainable if p.requires_grad else frozen).append((name, p.numel()))
    n_trainable = sum(n for _, n in trainable)
    n_frozen = sum(n for _, n in frozen)
    print(f"\ntrainable params: {n_trainable:,} across {len(trainable)} tensors")
    print(f"frozen params:    {n_frozen:,} across {len(frozen)} tensors")
    backbone_leak = [name for name, _ in trainable if name.startswith("backbone.")]
    assert not backbone_leak, f"backbone params leaked into trainable set: {backbone_leak}"
    for expected in MOE_HEAD_LAYER_NAMES:
        assert any(name.startswith(expected) for name, _ in trainable), (
            f"expected head '{expected}' not found among trainable params"
        )
    print("OK: only the moe head layers are trainable; backbone is fully frozen.")

    context = torch.randn(BATCH_SIZE, CONTEXT_LENGTH, device=device)
    retrieved_seq = torch.randn(BATCH_SIZE, TOP_K, RETRIEVED_WINDOW_LENGTH, device=device)
    distances = torch.rand(BATCH_SIZE, TOP_K, device=device)
    target = torch.randn(BATCH_SIZE, PREDICTION_LENGTH, device=device)

    print(f"\ninput shapes:")
    print(f"  context:       {tuple(context.shape)}")
    print(f"  retrieved_seq: {tuple(retrieved_seq.shape)}")
    print(f"  distances:     {tuple(distances.shape)}")
    print(f"  target:        {tuple(target.shape)}")

    out = model(context=context, retrieved_seq=retrieved_seq, distances=distances, target=target)

    print(f"\nintermediate shapes:")
    for field in ["q", "retrieved_y_enc", "att_output", "alpha", "h", "quantile_preds", "point_forecast"]:
        t = getattr(out, field)
        print(f"  {field:15s} {tuple(t.shape)}")
    print(f"  {'loss':15s} {out.loss.item():.6f}")

    NQ = model.num_quantiles
    expected = {
        "q": (BATCH_SIZE, model.d_model),
        "retrieved_y_enc": (BATCH_SIZE, TOP_K, model.d_model),
        "att_output": (BATCH_SIZE, TOP_K + 1, model.d_model),
        "alpha": (BATCH_SIZE, TOP_K + 1, 1),
        "h": (BATCH_SIZE, model.d_model),
        "quantile_preds": (BATCH_SIZE, NQ, PREDICTION_LENGTH),
        "point_forecast": (BATCH_SIZE, PREDICTION_LENGTH),
    }
    for field, exp_shape in expected.items():
        actual = tuple(getattr(out, field).shape)
        assert actual == exp_shape, f"{field}: expected {exp_shape}, got {actual}"
    print("\nOK: all intermediate shapes match expectations.")

    out.loss.backward()
    missing_grad = [name for name, p in trainable if model.get_parameter(name).grad is None]
    backbone_has_grad = [
        name for name, p in model.named_parameters()
        if name.startswith("backbone.") and p.grad is not None
    ]
    assert not missing_grad, f"trainable params with no grad after backward: {missing_grad}"
    assert not backbone_has_grad, f"backbone params unexpectedly received grad: {backbone_has_grad}"
    print("OK: backward pass populates grads on all trainable heads and none on the backbone.")

    print("\nsanity check passed.")


if __name__ == "__main__":
    main()
