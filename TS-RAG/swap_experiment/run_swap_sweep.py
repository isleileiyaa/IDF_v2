"""
rho2 sweep over the RIDDE branch-swap experiment (see run_swap_experiment.py
for the method). The naive approach (run run_swap_experiment.py once per rho2
value) reloads every dataset's retrieval database (.pkl, up to 26GB for
electricity) from scratch for each of the 6 checkpoints -- that reload
dominates wall time (~80s just for electricity) while the actual GPU compute
per pair is ~1ms. This script restructures the loop to load each dataset's
DatasetBundle ONCE and reuse it across all rho2 checkpoints (models are cheap
to load/swap, ~1s each), and uses a much larger batch (chunk_pairs) since
GPU memory usage per pair is tiny.
"""
from __future__ import annotations

import os
import time

import pandas as pd
import torch

import swap_core as C
from data_access import DatasetBundle

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CHUNK_PAIRS = 512
ROOT = os.path.dirname(os.path.abspath(__file__))

RHO2_VALUES = ["0", "0.001", "0.01", "0.1", "1", "10"]

DATASETS = ["ETTh1", "ETTm1", "electricity", "ETTh2", "ETTm2", "weather", "exchange_rate",
            "traffic", "solar", "PEMS08", "AQWan", "Wind", "ZafNoo", "CzeLan"]


def main():
    models = {}  # rho2 -> model, cached across datasets
    med_idx = None

    all_dfs = {rho2: [] for rho2 in RHO2_VALUES}

    for name in DATASETS:
        t0 = time.time()
        bundle = DatasetBundle(name)
        pairs = C.load_pairs(name)
        t1 = time.time()
        print(f"[{name}] bundle loaded in {t1 - t0:.1f}s, {len(pairs)} pairs")

        for rho2 in RHO2_VALUES:
            if rho2 not in models:
                models[rho2] = C.load_model(C.checkpoint_path_for(rho2), DEVICE)
                if med_idx is None:
                    med_idx = C.median_quantile_idx(models[rho2])
                print(f"  loaded model rho2={rho2}")
            model = models[rho2]

            t2 = time.time()
            df = C.process_dataset(model, bundle, pairs, med_idx, DEVICE, CHUNK_PAIRS, name)
            t3 = time.time()
            print(f"  [{name}] rho2={rho2}: {t3 - t2:.2f}s ({(t3 - t2) / max(len(pairs), 1) * 1000:.2f} ms/pair)")

            out_dir = os.path.join(ROOT, f"results_rho2_{rho2}")
            os.makedirs(out_dir, exist_ok=True)
            df.to_csv(os.path.join(out_dir, f"{name}_pair_distances.csv"), index=False)
            all_dfs[rho2].append(df)

        del bundle

    # per-rho2 summary
    sweep_summary_rows = []
    for rho2 in RHO2_VALUES:
        combined = pd.concat(all_dfs[rho2], ignore_index=True)
        out_dir = os.path.join(ROOT, f"results_rho2_{rho2}")
        combined.to_csv(os.path.join(out_dir, "all_pair_distances.csv"), index=False)
        summary_df = C.build_summary(combined, DATASETS)
        summary_df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
        summary_df.insert(0, "rho2", rho2)
        sweep_summary_rows.append(summary_df)

    sweep_summary = pd.concat(sweep_summary_rows, ignore_index=True)
    sweep_summary.to_csv(os.path.join(ROOT, "rho2_sweep_summary.csv"), index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\n\n=== rho2 sweep: Combined row across all rho2 values ===")
    print(sweep_summary[sweep_summary.dataset == "Combined"].to_string(index=False))


if __name__ == "__main__":
    main()
