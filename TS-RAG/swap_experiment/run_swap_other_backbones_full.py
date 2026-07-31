"""
Same as run_swap_other_backbones.py but across all 14 usable datasets (ILI
excluded, 0 candidate pairs) instead of just electricity. Moirai2 and
TimesFM2.5 only have one trained idf_clean_dis checkpoint each (rho=0
equivalent, no disentangle-loss sweep was run for them).

Dataset-outer / backbone-inner loop (each dataset's retrieval database is
loaded once and reused for both backbones), same optimization as
run_swap_sweep.py.
"""
from __future__ import annotations

import os
import time

import pandas as pd
import torch

import swap_core as C
from data_access import DatasetBundle
from run_swap_other_backbones import load_moirai2, load_timesfm25

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CHUNK_PAIRS = 256
ROOT = os.path.dirname(os.path.abspath(__file__))

DATASETS = ["ETTh1", "ETTm1", "electricity", "ETTh2", "ETTm2", "weather", "exchange_rate",
            "traffic", "solar", "PEMS08", "AQWan", "Wind", "ZafNoo", "CzeLan"]

BACKBONES = {
    "Moirai2": load_moirai2,
    "TimesFM2.5": load_timesfm25,
}


def main():
    models = {}
    all_dfs = {name: [] for name in BACKBONES}

    for name in DATASETS:
        t0 = time.time()
        bundle = DatasetBundle(name)
        pairs = C.load_pairs(name)
        t1 = time.time()
        print(f"[{name}] bundle loaded in {t1 - t0:.1f}s, {len(pairs)} pairs")

        for backbone_name, loader in BACKBONES.items():
            if backbone_name not in models:
                models[backbone_name] = loader()
                print(f"  loaded backbone {backbone_name}")
            model = models[backbone_name]
            med_idx = model.median_idx

            t2 = time.time()
            df = C.process_dataset(model, bundle, pairs, med_idx, DEVICE, CHUNK_PAIRS, name)
            t3 = time.time()
            print(f"  [{name}] {backbone_name}: {t3 - t2:.2f}s "
                  f"({(t3 - t2) / max(len(pairs), 1) * 1000:.2f} ms/pair)")

            out_dir = os.path.join(ROOT, f"results_{backbone_name.replace('.', '')}")
            os.makedirs(out_dir, exist_ok=True)
            df.to_csv(os.path.join(out_dir, f"{name}_pair_distances.csv"), index=False)
            all_dfs[backbone_name].append(df)

        del bundle

    summary_rows = []
    for backbone_name in BACKBONES:
        combined = pd.concat(all_dfs[backbone_name], ignore_index=True)
        out_dir = os.path.join(ROOT, f"results_{backbone_name.replace('.', '')}")
        combined.to_csv(os.path.join(out_dir, "all_pair_distances.csv"), index=False)
        summary_df = C.build_summary(combined, DATASETS)
        summary_df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
        summary_df.insert(0, "backbone", backbone_name)
        summary_rows.append(summary_df)

    full_summary = pd.concat(summary_rows, ignore_index=True)
    full_summary.to_csv(os.path.join(ROOT, "other_backbones_full_summary.csv"), index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\n\n=== Moirai2 / TimesFM2.5, all datasets, rho=0 ===")
    print(full_summary.to_string(index=False))


if __name__ == "__main__":
    main()
