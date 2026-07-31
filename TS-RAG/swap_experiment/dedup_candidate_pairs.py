"""
Greedy "each window used at most once" independence dedup on top of the
thresholded candidate pairs from select_candidate_pairs.py.

Raw candidate pairs are not independent: a single window with an unusually
flat/generic trend can co-occur in dozens of high trend_corr pairs, so the
raw pair count overstates how many genuinely distinct sample pairs are
available for the swap experiment. This greedily sorts by trend_corr
descending and keeps a pair only if neither of its two windows has been
used by an already-kept pair.

Reads the *_all_pairs_ma{W}.csv tables written by select_candidate_pairs.py,
applies the trend_corr/residual_corr threshold, then dedups.
"""
from __future__ import annotations

import os

import pandas as pd

CANDIDATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "candidate_pairs")

MA_WINDOW = 8
TREND_THR = 0.8
RESID_THR = 0.3

DATASETS = ["ETTh1", "ETTm1", "electricity", "ETTh2", "ETTm2", "weather", "exchange_rate",
            "traffic", "solar", "PEMS08", "AQWan", "Wind", "ILI", "ZafNoo", "CzeLan"]


def greedy_dedup(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("trend_corr", ascending=False)
    used = set()
    keep_rows = []
    for row in df.itertuples(index=False):
        key_i = (row.channel, row.start_i)
        key_j = (row.channel, row.start_j)
        if key_i in used or key_j in used:
            continue
        used.add(key_i)
        used.add(key_j)
        keep_rows.append(row)
    return pd.DataFrame(keep_rows, columns=df.columns)


def main():
    summary = []
    for name in DATASETS:
        full_path = os.path.join(CANDIDATE_DIR, f"{name}_all_pairs_ma{MA_WINDOW}.csv")
        pairs = pd.read_csv(full_path)
        thresholded = pairs[(pairs.trend_corr > TREND_THR) & (pairs.residual_corr < RESID_THR)]
        deduped = greedy_dedup(thresholded)

        n_channels_before = thresholded["channel"].nunique()
        n_channels_after = deduped["channel"].nunique()
        print(f"{name}: raw thresholded pairs={len(thresholded)} ({n_channels_before} channels) "
              f"-> independent pairs after dedup={len(deduped)} ({n_channels_after} channels)")

        out_path = os.path.join(CANDIDATE_DIR, f"{name}_independent_pairs_ma{MA_WINDOW}_t{TREND_THR}_r{RESID_THR}.csv")
        deduped.to_csv(out_path, index=False)
        summary.append(dict(dataset=name, raw_pairs=len(thresholded), independent_pairs=len(deduped),
                             channels_covered=n_channels_after))

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(CANDIDATE_DIR, "dedup_summary.csv"), index=False)
    print("\n=== Summary ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
