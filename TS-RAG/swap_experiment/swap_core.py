"""Shared core for the RIDDE branch-swap experiment (see run_swap_experiment.py
for the full method description). Device- and checkpoint-parameterized so it
can be reused both for a single run and for a multi-checkpoint / multi-GPU
sweep (run_swap_sweep.py) without reloading each dataset's retrieval database
once per checkpoint.
"""
from __future__ import annotations

import os
import sys
from collections import OrderedDict

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

PRETRAINED_MODEL_PATH = "/home/fenglei/TS-RAG-main/TS-RAG/checkpoints/base/"
EPS = 1e-8

CANDIDATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "candidate_pairs")
MA_WINDOW = 8
TREND_THR = 0.8
RESID_THR = 0.3


def checkpoint_path_for(rho2: str) -> str:
    return (
        "/home/fenglei/TS-RAG-main/TS-RAG/checkpoints/"
        f"data50m_idf_clean_dis_512_pred64_lookback512_top10_lr0.0003_drop0.2_adamw_cosanneal_"
        f"step10000_bs256_no_embeddingtuning_rho0_{rho2}_0_0_final.pth"
    )


def load_model(checkpoint_path: str, device: torch.device) -> ChronosBoltModelForForecastingWithRetrieval:
    config = AutoConfig.from_pretrained(PRETRAINED_MODEL_PATH)
    if hasattr(config, "chronos_config"):
        config.chronos_config["context_length"] = SEQ_LEN
        config.chronos_config["prediction_length"] = PRED_LEN
    model = ChronosBoltModelForForecastingWithRetrieval.from_pretrained(
        PRETRAINED_MODEL_PATH, config=config, augment="idf_clean_dis", low_cpu_mem_usage=False
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    new_state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in ckpt.items())
    msg = model.load_state_dict(new_state_dict, strict=False)
    if msg.missing_keys or msg.unexpected_keys:
        print(f"  WARNING missing={msg.missing_keys[:5]} unexpected={msg.unexpected_keys[:5]}")
    model.to(device)
    model.eval()
    return model


def median_quantile_idx(model) -> int:
    quantiles = torch.tensor(model.chronos_config.quantiles)
    return int(torch.abs(quantiles - 0.5).argmin())


def log_diff_var(x: np.ndarray) -> np.ndarray:
    d = np.diff(x, axis=-1)
    return np.log(d.var(axis=-1) + EPS)


def run_forward_batch(model, ctx, ret_seq, dist, med_idx: int):
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
    y_fused_norm = captured["y_fused_norm"].view(n, -1, PRED_LEN)
    own_point = y_fused_norm[:, med_idx, :].cpu().numpy()
    return captured["z_inv"], captured["z_dyn"], own_point


def swap_predict(model, z_inv_src, z_dyn_src, med_idx: int) -> np.ndarray:
    with torch.no_grad():
        y_inv = model.inv_pred_head(z_inv_src)
        y_dyn = model.dyn_pred_head_clean(z_dyn_src)
        final_in = torch.cat([y_inv, y_dyn], dim=-1)
        fused = model.final_pred_head(final_in).view(z_inv_src.shape[0], -1, PRED_LEN)
    return fused[:, med_idx, :].cpu().numpy()


def load_pairs(name: str) -> pd.DataFrame:
    pairs_path = os.path.join(
        CANDIDATE_DIR, f"{name}_independent_pairs_ma{MA_WINDOW}_t{TREND_THR}_r{RESID_THR}.csv"
    )
    return pd.read_csv(pairs_path)


def process_dataset(model, bundle: DatasetBundle, pairs: pd.DataFrame, med_idx: int,
                     device: torch.device, chunk_pairs: int, name: str) -> pd.DataFrame:
    rows = []
    for chunk_start in range(0, len(pairs), chunk_pairs):
        chunk = pairs.iloc[chunk_start: chunk_start + chunk_pairs]
        ctx_list, ret_list, dist_list = [], [], []
        for row in chunk.itertuples(index=False):
            for start in (row.start_i, row.start_j):
                seq_x, seq_y, retrieved, dist = bundle.get_window(row.channel, int(start))
                ctx_list.append(seq_x)
                ret_list.append(retrieved)
                dist_list.append(dist)
        ctx = torch.tensor(np.stack(ctx_list), device=device)
        ret_seq = torch.tensor(np.stack(ret_list), device=device)
        dist = torch.tensor(np.stack(dist_list), device=device)

        z_inv, z_dyn, own_point = run_forward_batch(model, ctx, ret_seq, dist, med_idx)

        bs = len(chunk)
        idx_A = torch.arange(0, 2 * bs, 2, device=device)
        idx_B = torch.arange(1, 2 * bs, 2, device=device)
        z_inv_A, z_inv_B = z_inv[idx_A], z_inv[idx_B]
        z_dyn_A, z_dyn_B = z_dyn[idx_A], z_dyn[idx_B]
        own_A = own_point[0::2]
        own_B = own_point[1::2]

        swap_main = swap_predict(model, z_inv_A, z_dyn_B, med_idx)
        swap_ctrl = swap_predict(model, z_inv_B, z_dyn_A, med_idx)

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


def build_summary(combined: pd.DataFrame, dataset_names: list[str]) -> pd.DataFrame:
    summary_rows = []
    for name in dataset_names + ["Combined"]:
        df = combined if name == "Combined" else combined[combined.dataset == name]
        if len(df) == 0:
            continue
        main_stats = summarize(df, "d_A", "d_B")
        ctrl_stats = summarize(df, "d_ctrl_A", "d_ctrl_B")
        summary_rows.append(dict(dataset=name, experiment="main (swap dynamic)", **main_stats))
        summary_rows.append(dict(dataset=name, experiment="control (swap invariant)", **ctrl_stats))
    return pd.DataFrame(summary_rows)
