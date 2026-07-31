"""
Same branch-swap experiment (see run_swap_experiment.py) but on the Moirai2
and TimesFM2.5 RIDDE ports instead of Chronos-Bolt, electricity only, rho=0
checkpoints (these two backbones only have one trained checkpoint each --
no rho2 sweep was run for them).

Moirai2ModelForForecastingWithRetrieval / TimesFM25ModelForForecastingWithRetrieval
(models/Moirai2.py, models/TimesFM25.py) are 1:1 ports of the same
idf_clean_dis fusion head (same inv_pred_head/dyn_pred_head_clean/
final_pred_head/routing_gate submodule names, same z_inv=gamma*h /
z_dyn=(1-gamma)*h formula), so swap_core's hook-based extraction and swap
arithmetic apply unchanged -- only model construction/checkpoint loading
differs per backbone.
"""
from __future__ import annotations

import os
from collections import OrderedDict

import pandas as pd
import torch

import swap_core as C
from data_access import DatasetBundle

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET = "electricity"


def load_state_dict_into(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    new_state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in ckpt.items())
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"  loaded {checkpoint_path}")
    print(f"  missing={msg.missing_keys[:10]} unexpected={msg.unexpected_keys[:10]}")
    model.to(device)
    model.eval()
    return model


def load_moirai2():
    from models.Moirai2 import Moirai2ModelForForecastingWithRetrieval
    model = Moirai2ModelForForecastingWithRetrieval(context_length=C.SEQ_LEN, prediction_length=C.PRED_LEN)
    ckpt_path = (
        "/home/fenglei/TS-RAG-main/TS-RAG/checkpoints/"
        "moirai2_idf_clean_dis_512_pred64_lookback512_top10_lr0.000015_drop0.2_adamw_cosanneal_step10000_bs256_final.pth"
    )
    return load_state_dict_into(model, ckpt_path, DEVICE)


def load_timesfm25():
    from models.TimesFM25 import TimesFM25ModelForForecastingWithRetrieval
    model = TimesFM25ModelForForecastingWithRetrieval(context_length=C.SEQ_LEN, prediction_length=C.PRED_LEN)
    ckpt_path = (
        "/home/fenglei/TS-RAG-main/TS-RAG/checkpoints/"
        "timesfm25_idf_clean_dis_512_pred64_lookback512_top10_lr0.000005_drop0.2_adamw_cosanneal_step10000_bs256_final.pth"
    )
    return load_state_dict_into(model, ckpt_path, DEVICE)


BACKBONES = {
    "Moirai2": load_moirai2,
    "TimesFM2.5": load_timesfm25,
}


def main():
    bundle = DatasetBundle(DATASET)
    pairs = C.load_pairs(DATASET)
    print(f"{DATASET}: {len(pairs)} pairs")

    all_summaries = []
    for backbone_name, loader in BACKBONES.items():
        print(f"\n=== {backbone_name} ===")
        model = loader()
        med_idx = model.median_idx
        print(f"  median_idx={med_idx}")

        chunk_pairs = 64
        df = C.process_dataset(model, bundle, pairs, med_idx, DEVICE, chunk_pairs, DATASET)

        out_dir = os.path.join(ROOT, f"results_{backbone_name.replace('.', '')}")
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(os.path.join(out_dir, f"{DATASET}_pair_distances.csv"), index=False)

        summary_df = C.build_summary(df, [DATASET])
        summary_df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
        summary_df.insert(0, "backbone", backbone_name)
        all_summaries.append(summary_df)

        del model
        torch.cuda.empty_cache()

    combined_summary = pd.concat(all_summaries, ignore_index=True)
    combined_summary.to_csv(os.path.join(ROOT, "other_backbones_summary.csv"), index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\n\n=== Moirai2 / TimesFM2.5, electricity, rho=0 ===")
    print(combined_summary.to_string(index=False))


if __name__ == "__main__":
    main()
