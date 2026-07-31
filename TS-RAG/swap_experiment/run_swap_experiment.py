"""
RIDDE branch-swap validation: does z_inv (invariant) actually carry
long-term/trend semantics and z_dyn (dynamic) actually carry short-term
semantics, as claimed?

For each candidate pair (A, B) selected by select_candidate_pairs.py +
dedup_candidate_pairs.py ("long-term similar, short-term different" ground
truth), runs the real ChronosBolt idf_clean_dis forward once to extract
z_inv/z_dyn for both samples (via hooks -- see ChronosBolt.py:1282-1291),
then:

  main experiment  (swap dynamic):   inv_pred_head(z_inv^A) + dyn_pred_head_clean(z_dyn^B) -> final_pred_head
  control experiment (swap invariant): inv_pred_head(z_inv^B) + dyn_pred_head_clean(z_dyn^A) -> final_pred_head

Everything is computed in the model's normalized space (final_pred_head's
raw output, before instance_norm.inverse) -- this sidesteps the fact that
loc_scale is per-sample and would otherwise bias which of A/B the swap
output "looks closer to" purely from unit conversion, not from the
disentanglement itself.

Point forecast = median-quantile slice (quantile closest to 0.5).
Distance = |logV(swap_pred) - logV(own_pred)| where logV(x) = log(Var(diff(x)) + eps),
matching diff_variance_analysis.py's oscillation-deviation metric.

Checkpoint: rho1=rho2=rho3=rho4=0 (the default/main results checkpoint,
results/forecast_evaluation/zeroshot_chronos_idf_clean_dis.txt).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from transformers import AutoConfig

TS_RAG_DIR = "/home/fenglei/TS-RAG-main/TS-RAG"
if TS_RAG_DIR not in sys.path:
    sys.path.insert(0, TS_RAG_DIR)

from models.ChronosBolt import ChronosBoltModelForForecastingWithRetrieval  # noqa: E402
from data_access import DatasetBundle, SEQ_LEN, PRED_LEN  # noqa: E402

RHO2 = sys.argv[1] if len(sys.argv) > 1 else "0"  # one of: 0, 0.001, 0.01, 0.1, 1, 10

PRETRAINED_MODEL_PATH = "/home/fenglei/TS-RAG-main/TS-RAG/checkpoints/base/"
CHECKPOINT_PATH = (
    "/home/fenglei/TS-RAG-main/TS-RAG/checkpoints/"
    f"data50m_idf_clean_dis_512_pred64_lookback512_top10_lr0.0003_drop0.2_adamw_cosanneal_"
    f"step10000_bs256_no_embeddingtuning_rho0_{RHO2}_0_0_final.pth"
)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPS = 1e-8
CHUNK_PAIRS = 64  # pairs per forward batch (2x samples)

CANDIDATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "candidate_pairs")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results_rho2_{RHO2}")

DATASETS = ["ETTh1", "ETTm1", "electricity", "ETTh2", "ETTm2", "weather", "exchange_rate",
            "traffic", "solar", "PEMS08", "AQWan", "Wind", "ZafNoo", "CzeLan"]
# ILI excluded: candidate-pair selection produced 0 pairs at any threshold
# (series too short -- ~130 non-overlapping test windows/channel total across
# all 7 channels, not enough to clear trend_corr>0.7/residual_corr<0.4 even
# loosened).
MA_WINDOW = 8
TREND_THR = 0.8
RESID_THR = 0.3


def load_model() -> ChronosBoltModelForForecastingWithRetrieval:
    config = AutoConfig.from_pretrained(PRETRAINED_MODEL_PATH)
    if hasattr(config, "chronos_config"):
        config.chronos_config["context_length"] = SEQ_LEN
        config.chronos_config["prediction_length"] = PRED_LEN
    model = ChronosBoltModelForForecastingWithRetrieval.from_pretrained(
        PRETRAINED_MODEL_PATH, config=config, augment="idf_clean_dis", low_cpu_mem_usage=False
    )
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for key, value in ckpt.items():
        new_state_dict[key.replace("module.", "")] = value
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"Loaded checkpoint: {CHECKPOINT_PATH}")
    print(f"  missing keys: {msg.missing_keys[:10]}")
    print(f"  unexpected keys: {msg.unexpected_keys[:10]}")
    model.to(DEVICE)
    model.eval()
    return model


def median_quantile_idx(model: ChronosBoltModelForForecastingWithRetrieval) -> int:
    quantiles = torch.tensor(model.chronos_config.quantiles)
    return int(torch.abs(quantiles - 0.5).argmin())


def log_diff_var(x: np.ndarray) -> np.ndarray:
    """x: (N, L) -> (N,) log(Var(diff(x, axis=-1)) + eps)."""
    d = np.diff(x, axis=-1)
    v = d.var(axis=-1)
    return np.log(v + EPS)


def run_forward_batch(model, ctx: torch.Tensor, ret_seq: torch.Tensor, dist: torch.Tensor, med_idx: int):
    """Runs one forward pass, hooks z_inv/z_dyn/final_pred_head-output.
    Returns numpy arrays: z_inv (N,768), z_dyn (N,768), own_point (N,PRED_LEN)."""
    captured = {}

    def pre_hook(name):
        def hook(module, args):
            captured[name] = args[0].detach()
        return hook

    def post_hook(name):
        def hook(module, args, output):
            captured[name] = output.detach()
        return hook

    h1 = model.inv_pred_head.register_forward_pre_hook(pre_hook("z_inv"))
    h2 = model.dyn_pred_head_clean.register_forward_pre_hook(pre_hook("z_dyn"))
    h3 = model.final_pred_head.register_forward_hook(post_hook("y_fused_norm"))
    try:
        with torch.no_grad():
            model(context=ctx, retrieved_seq=ret_seq, distances=dist)
    finally:
        h1.remove()
        h2.remove()
        h3.remove()

    n = ctx.shape[0]
    z_inv = captured["z_inv"]
    z_dyn = captured["z_dyn"]
    y_fused_norm = captured["y_fused_norm"].view(n, -1, PRED_LEN)  # (N, n_quantiles, PRED_LEN)
    own_point = y_fused_norm[:, med_idx, :].cpu().numpy()
    return z_inv, z_dyn, own_point


def swap_predict(model, z_inv_src: torch.Tensor, z_dyn_src: torch.Tensor, med_idx: int) -> np.ndarray:
    with torch.no_grad():
        y_inv = model.inv_pred_head(z_inv_src)
        y_dyn = model.dyn_pred_head_clean(z_dyn_src)
        final_in = torch.cat([y_inv, y_dyn], dim=-1)
        fused = model.final_pred_head(final_in).view(z_inv_src.shape[0], -1, PRED_LEN)
    return fused[:, med_idx, :].cpu().numpy()


def process_dataset(model, name: str, med_idx: int) -> pd.DataFrame:
    pairs_path = os.path.join(
        CANDIDATE_DIR, f"{name}_independent_pairs_ma{MA_WINDOW}_t{TREND_THR}_r{RESID_THR}.csv"
    )
    pairs = pd.read_csv(pairs_path)
    bundle = DatasetBundle(name)

    rows = []
    for chunk_start in range(0, len(pairs), CHUNK_PAIRS):
        chunk = pairs.iloc[chunk_start: chunk_start + CHUNK_PAIRS]
        ctx_list, ret_list, dist_list = [], [], []
        for row in chunk.itertuples(index=False):
            for start in (row.start_i, row.start_j):
                seq_x, seq_y, retrieved, dist = bundle.get_window(row.channel, int(start))
                ctx_list.append(seq_x)
                ret_list.append(retrieved)
                dist_list.append(dist)
        ctx = torch.tensor(np.stack(ctx_list), device=DEVICE)
        ret_seq = torch.tensor(np.stack(ret_list), device=DEVICE)
        dist = torch.tensor(np.stack(dist_list), device=DEVICE)

        z_inv, z_dyn, own_point = run_forward_batch(model, ctx, ret_seq, dist, med_idx)

        bs = len(chunk)
        # interleaved layout: [A_0, B_0, A_1, B_1, ...]
        idx_A = torch.arange(0, 2 * bs, 2, device=DEVICE)
        idx_B = torch.arange(1, 2 * bs, 2, device=DEVICE)
        z_inv_A, z_inv_B = z_inv[idx_A], z_inv[idx_B]
        z_dyn_A, z_dyn_B = z_dyn[idx_A], z_dyn[idx_B]
        own_A = own_point[0::2]
        own_B = own_point[1::2]

        swap_main = swap_predict(model, z_inv_A, z_dyn_B, med_idx)   # main: swap dynamic
        swap_ctrl = swap_predict(model, z_inv_B, z_dyn_A, med_idx)   # control: swap invariant

        logV_A = log_diff_var(own_A)
        logV_B = log_diff_var(own_B)
        logV_main = log_diff_var(swap_main)
        logV_ctrl = log_diff_var(swap_ctrl)

        d_A = np.abs(logV_main - logV_A)
        d_B = np.abs(logV_main - logV_B)
        d_ctrl_A = np.abs(logV_ctrl - logV_A)
        d_ctrl_B = np.abs(logV_ctrl - logV_B)

        for k in range(bs):
            rows.append(dict(
                dataset=name, channel=chunk.iloc[k]["channel"],
                start_i=int(chunk.iloc[k]["start_i"]), start_j=int(chunk.iloc[k]["start_j"]),
                trend_corr=chunk.iloc[k]["trend_corr"], residual_corr=chunk.iloc[k]["residual_corr"],
                d_A=d_A[k], d_B=d_B[k], d_ctrl_A=d_ctrl_A[k], d_ctrl_B=d_ctrl_B[k],
            ))
        print(f"  {name}: {chunk_start + bs}/{len(pairs)} pairs done", end="\r")
    print()
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, a_col: str, b_col: str) -> dict:
    a, b = df[a_col].values, df[b_col].values
    diff = b - a
    nz = diff != 0
    p = wilcoxon(a[nz], b[nz])[1] if nz.sum() >= 1 else float("nan")
    return dict(
        n=len(df),
        mean_d_A=a.mean(),
        mean_d_B=b.mean(),
        frac_closer_to_B=float((b < a).mean()),
        wilcoxon_p=p,
    )


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model = load_model()
    med_idx = median_quantile_idx(model)
    print(f"median quantile index = {med_idx} (quantiles={model.chronos_config.quantiles})")

    all_df = []
    for name in DATASETS:
        print(f"\n=== Processing {name} ===")
        df = process_dataset(model, name, med_idx)
        df.to_csv(os.path.join(RESULTS_DIR, f"{name}_pair_distances.csv"), index=False)
        all_df.append(df)
    combined = pd.concat(all_df, ignore_index=True)
    combined.to_csv(os.path.join(RESULTS_DIR, "all_pair_distances.csv"), index=False)

    print("\n\n" + "=" * 100)
    print("PER-DATASET REPORT")
    print("=" * 100)
    summary_rows = []
    for name in DATASETS + ["Combined"]:
        df = combined if name == "Combined" else combined[combined.dataset == name]
        main_stats = summarize(df, "d_A", "d_B")
        ctrl_stats = summarize(df, "d_ctrl_A", "d_ctrl_B")
        summary_rows.append(dict(dataset=name, experiment="main (swap dynamic)", **main_stats))
        summary_rows.append(dict(dataset=name, experiment="control (swap invariant)", **ctrl_stats))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "summary.csv"), index=False)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")

    for name in DATASETS + ["Combined"]:
        sub = summary_df[summary_df.dataset == name]
        print(f"\n--- {name} ---")
        print(sub.drop(columns=["dataset"]).to_string(index=False))

    print("\n\n" + "=" * 100)
    print("SIDE-BY-SIDE: main (swap dynamic) vs control (swap invariant)")
    print("=" * 100)
    pivot = summary_df.pivot(index="dataset", columns="experiment",
                              values=["n", "mean_d_A", "mean_d_B", "frac_closer_to_B", "wilcoxon_p"])
    print(pivot.to_string())


if __name__ == "__main__":
    main()
