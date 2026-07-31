"""
Candidate-pair selection for the RIDDE branch-swap ablation ("does z_inv/z_dyn
really carry the invariant/dynamic semantics the paper claims").

Picks pairs of test-set forecast windows (i, j) from the SAME channel of the
SAME dataset whose ground-truth futures y_i, y_j are "long-term similar,
short-term different": high correlation of the moving-average trend, low
correlation of the residual.

Ground truth only -- no model, no predictions. Decomposition is a simple
centered moving average (not STL): STL needs a seasonal period, which is not
uniformly well-defined across all RIDDE eval datasets (e.g. exchange_rate has
weak/no clear seasonality), so a moving average is used for consistency across
datasets, exactly mirroring what test-time windows the RIDDE eval scripts use
(seq_len=512, pred_len=64, same train/val/test borders).

Windows are restricted to the SAME channel (cross-channel pairs compare
physically different quantities and would confound interpretation) and
sub-sampled at stride == pred_len so no two windows share a single
ground-truth step -- otherwise adjacent sliding windows (stride 1 in the
eval dataloader) would be near-duplicates and trivially inflate trend_corr.

Usage:
    python select_candidate_pairs.py
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

SEQ_LEN = 512
PRED_LEN = 64
MA_WINDOWS = [8, 12, 16]  # horizon/8 .. horizon/4, per spec
THRESHOLD_GRID = [
    (0.8, 0.3),
    (0.75, 0.35),
    (0.7, 0.4),
]

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "candidate_pairs")


@dataclass
class DatasetSpec:
    name: str
    kind: str  # 'ett_h' | 'ett_m' | 'custom'
    root_path: str
    data_path: str


DATASETS = [
    DatasetSpec("ETTh1", "ett_h", "/home/fenglei/TS-RAG-main/datasets/ETT-small", "ETTh1.csv"),
    DatasetSpec("ETTm1", "ett_m", "/home/fenglei/TS-RAG-main/datasets/ETT-small", "ETTm1.csv"),
    DatasetSpec("electricity", "custom", "/home/fenglei/TS-RAG-main/datasets/electricity", "electricity.csv"),
    DatasetSpec("ETTh2", "ett_h", "/home/fenglei/TS-RAG-main/datasets/ETT-small", "ETTh2.csv"),
    DatasetSpec("ETTm2", "ett_m", "/home/fenglei/TS-RAG-main/datasets/ETT-small", "ETTm2.csv"),
    DatasetSpec("weather", "custom", "/home/fenglei/TS-RAG-main/datasets/weather", "weather.csv"),
    DatasetSpec("exchange_rate", "custom", "/home/fenglei/TS-RAG-main/datasets/exchange_rate", "exchange_rate.csv"),
    DatasetSpec("traffic", "custom", "/home/fenglei/TS-RAG-main/datasets/traffic", "traffic.csv"),
    DatasetSpec("solar", "custom", "/home/fenglei/TS-RAG-main/datasets/solar", "solar.csv"),
    DatasetSpec("PEMS08", "custom", "/home/fenglei/TS-RAG-main/datasets/PEMS08", "PEMS08.csv"),
    DatasetSpec("AQWan", "custom", "/home/fenglei/TS-RAG-main/datasets/AQWan", "AQWan.csv"),
    DatasetSpec("Wind", "custom", "/home/fenglei/TS-RAG-main/datasets/Wind", "Wind.csv"),
    DatasetSpec("ILI", "custom", "/home/fenglei/TS-RAG-main/datasets/ILI", "ILI.csv"),
    DatasetSpec("ZafNoo", "custom", "/home/fenglei/TS-RAG-main/datasets/ZafNoo", "ZafNoo.csv"),
    DatasetSpec("CzeLan", "custom", "/home/fenglei/TS-RAG-main/datasets/CzeLan", "CzeLan.csv"),
]


def get_test_border(kind: str, n: int) -> tuple[int, int]:
    """Mirrors data_provider/data_loader.py test-split borders (set_type=2)."""
    if kind == "ett_h":
        border1 = 12 * 30 * 24 + 4 * 30 * 24 - SEQ_LEN
        border2 = 12 * 30 * 24 + 8 * 30 * 24
    elif kind == "ett_m":
        border1 = 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - SEQ_LEN
        border2 = 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4
    elif kind == "custom":
        num_test = int(n * 0.2)
        border1 = n - num_test - SEQ_LEN
        border2 = n
    else:
        raise ValueError(kind)
    return border1, border2


def moving_average(y: np.ndarray, w: int) -> np.ndarray:
    """Centered moving average with edge-padding, same length as y."""
    pad_l = w // 2
    pad_r = w - 1 - pad_l
    y_pad = np.pad(y, (pad_l, pad_r), mode="edge")
    kernel = np.ones(w) / w
    return np.convolve(y_pad, kernel, mode="valid")


def build_windows(data: np.ndarray, border1: int, border2: int) -> tuple[np.ndarray, np.ndarray]:
    """data: (T, C) full-series values (unscaled). Returns (starts, windows)
    where windows has shape (n_channels, n_windows, PRED_LEN); starts are the
    absolute row indices (into `data`) of each window's ground-truth horizon
    start, non-overlapping (stride = PRED_LEN)."""
    tot_len = (border2 - border1) - SEQ_LEN - PRED_LEN + 1
    if tot_len <= 0:
        raise ValueError("test span too short for seq_len+pred_len")
    s_begins = np.arange(0, tot_len, PRED_LEN)
    g_starts = border1 + s_begins + SEQ_LEN  # absolute row idx of horizon start
    n_channels = data.shape[1]
    windows = np.stack(
        [np.stack([data[g:g + PRED_LEN, c] for g in g_starts], axis=0) for c in range(n_channels)],
        axis=0,
    )  # (n_channels, n_windows, PRED_LEN)
    return g_starts, windows


def pairwise_corr_matrices(mat: np.ndarray) -> np.ndarray:
    """mat: (n_windows, PRED_LEN) -> (n_windows, n_windows) Pearson corr."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.corrcoef(mat)


def analyze_channel(y: np.ndarray, g_starts: np.ndarray, ma_w: int):
    n = y.shape[0]
    trends = np.stack([moving_average(y[i], ma_w) for i in range(n)])
    resids = y - trends
    trend_corr = pairwise_corr_matrices(trends)
    resid_corr = pairwise_corr_matrices(resids)
    iu = np.triu_indices(n, k=1)
    tc = trend_corr[iu]
    rc = resid_corr[iu]
    valid = ~(np.isnan(tc) | np.isnan(rc))
    idx_i, idx_j = iu[0][valid], iu[1][valid]
    tc, rc = tc[valid], rc[valid]
    return idx_i, idx_j, tc, rc, g_starts


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")

    summary_rows = []
    for spec in DATASETS:
        df = pd.read_csv(os.path.join(spec.root_path, spec.data_path))
        channels = [c for c in df.columns if c != "date"]
        data = df[channels].values.astype(np.float64)
        border1, border2 = get_test_border(spec.kind, len(df))
        g_starts, windows = build_windows(data, border1, border2)
        n_windows = windows.shape[1]
        print(f"\n=== {spec.name}: {len(channels)} channels, {n_windows} non-overlapping "
              f"test windows/channel (stride={PRED_LEN}) ===")

        for ma_w in MA_WINDOWS:
            all_tc, all_rc = [], []
            pair_rows = []
            for ci, ch in enumerate(channels):
                idx_i, idx_j, tc, rc, starts = analyze_channel(windows[ci], g_starts, ma_w)
                all_tc.append(tc)
                all_rc.append(rc)
                for a, b, t, r in zip(idx_i, idx_j, tc, rc):
                    pair_rows.append((ch, int(starts[a]), int(starts[b]), t, r))
            all_tc = np.concatenate(all_tc)
            all_rc = np.concatenate(all_rc)
            pairs_df = pd.DataFrame(pair_rows, columns=["channel", "start_i", "start_j", "trend_corr", "residual_corr"])

            print(f"\n-- MA window={ma_w} -- total pairs scanned: {len(pairs_df)}")
            print("trend_corr distribution:  " +
                  ", ".join(f"p{p}={np.percentile(all_tc, p):.3f}" for p in [10, 25, 50, 75, 90]))
            print("residual_corr distribution: " +
                  ", ".join(f"p{p}={np.percentile(all_rc, p):.3f}" for p in [10, 25, 50, 75, 90]))

            for trend_thr, resid_thr in THRESHOLD_GRID:
                sel = pairs_df[(pairs_df.trend_corr > trend_thr) & (pairs_df.residual_corr < resid_thr)]
                n_channels_covered = sel["channel"].nunique()
                print(f"   trend_corr>{trend_thr}, residual_corr<{resid_thr}: "
                      f"{len(sel)} pairs across {n_channels_covered}/{len(channels)} channels")
                summary_rows.append(dict(
                    dataset=spec.name, ma_window=ma_w, trend_thr=trend_thr, resid_thr=resid_thr,
                    n_pairs=len(sel), n_channels_covered=n_channels_covered,
                ))
                if (ma_w, trend_thr, resid_thr) == (MA_WINDOWS[len(MA_WINDOWS) // 2], 0.8, 0.3):
                    out_path = os.path.join(OUTPUT_DIR, f"{spec.name}_candidates_ma{ma_w}_t{trend_thr}_r{resid_thr}.csv")
                    sel.to_csv(out_path, index=False)

            # always dump the full unthresholded pair table for this MA window, for inspection
            full_path = os.path.join(OUTPUT_DIR, f"{spec.name}_all_pairs_ma{ma_w}.csv")
            pairs_df.to_csv(full_path, index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)
    print("\n\n=== Summary (n_pairs by dataset x MA window x threshold) ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
