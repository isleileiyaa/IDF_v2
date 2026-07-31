"""
Scan every available idf_clean_dis checkpoint (Moirai2 x2 lr variants,
TimesFM2.5 x2 lr variants -- all effectively rho=0, no disentangle-loss
sweep exists for these backbones) across all 14 usable datasets, and print a
HIT line the instant a (backbone, dataset) result clears Bonferroni-corrected
significance (alpha = 0.05/14, the same per-dataset family size used
throughout this analysis) in the direction the branch-disentanglement
hypothesis predicts:
  main (swap dynamic):    frac_closer_to_B > 0.5  and p < alpha
  control (swap invariant): frac_closer_to_B < 0.5  and p < alpha
So a hit can be read off the running log without waiting for the whole scan
to finish.
"""
from __future__ import annotations

import os
import time
from collections import OrderedDict

import pandas as pd
import torch

import swap_core as C
from data_access import DatasetBundle

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CHUNK_PAIRS = 256
ROOT = os.path.dirname(os.path.abspath(__file__))
ALPHA = 0.05 / 14

DATASETS = ["ETTh1", "ETTm1", "electricity", "ETTh2", "ETTm2", "weather", "exchange_rate",
            "traffic", "solar", "PEMS08", "AQWan", "Wind", "ZafNoo", "CzeLan"]

CKPT_DIR = "/home/fenglei/TS-RAG-main/TS-RAG/checkpoints"
CONFIGS = [
    ("Moirai2_lr1.5e-5", "moirai2",
     f"{CKPT_DIR}/moirai2_idf_clean_dis_512_pred64_lookback512_top10_lr0.000015_drop0.2_adamw_cosanneal_step10000_bs256_final.pth"),
    ("Moirai2_lr3e-4", "moirai2",
     f"{CKPT_DIR}/moirai2_idf_clean_dis_512_pred64_lookback512_top10_lr0.0003_drop0.2_adamw_cosanneal_step10000_bs256_final.pth"),
    ("TimesFM25_lr5e-6", "timesfm25",
     f"{CKPT_DIR}/timesfm25_idf_clean_dis_512_pred64_lookback512_top10_lr0.000005_drop0.2_adamw_cosanneal_step10000_bs256_final.pth"),
    ("TimesFM25_lr3e-4", "timesfm25",
     f"{CKPT_DIR}/timesfm25_idf_clean_dis_512_pred64_lookback512_top10_lr0.0003_drop0.2_adamw_cosanneal_step10000_bs256_final.pth"),
]


def load_state_dict_into(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    new_state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in ckpt.items())
    msg = model.load_state_dict(new_state_dict, strict=False)
    if msg.missing_keys or msg.unexpected_keys:
        print(f"  WARNING missing={msg.missing_keys[:5]} unexpected={msg.unexpected_keys[:5]}", flush=True)
    model.to(device)
    model.eval()
    return model


def load_backbone(kind, checkpoint_path):
    if kind == "moirai2":
        from models.Moirai2 import Moirai2ModelForForecastingWithRetrieval
        model = Moirai2ModelForForecastingWithRetrieval(context_length=C.SEQ_LEN, prediction_length=C.PRED_LEN)
    elif kind == "timesfm25":
        from models.TimesFM25 import TimesFM25ModelForForecastingWithRetrieval
        model = TimesFM25ModelForForecastingWithRetrieval(context_length=C.SEQ_LEN, prediction_length=C.PRED_LEN)
    else:
        raise ValueError(kind)
    return load_state_dict_into(model, checkpoint_path, DEVICE)


def main():
    hits = []
    all_summaries = []

    for name in DATASETS:
        t0 = time.time()
        bundle = DatasetBundle(name)
        pairs = C.load_pairs(name)
        t1 = time.time()
        print(f"[{name}] bundle loaded in {t1 - t0:.1f}s, {len(pairs)} pairs", flush=True)

        for cfg_name, kind, ckpt_path in CONFIGS:
            model = load_backbone(kind, ckpt_path)
            med_idx = model.median_idx

            df = C.process_dataset(model, bundle, pairs, med_idx, DEVICE, CHUNK_PAIRS, name)
            main_stats = C.summarize(df, "d_A", "d_B")
            ctrl_stats = C.summarize(df, "d_ctrl_A", "d_ctrl_B")

            print(f"  [{name}] {cfg_name}: main frac_B={main_stats['frac_closer_to_B']:.4f} "
                  f"p={main_stats['wilcoxon_p']:.4g} | ctrl frac_B={ctrl_stats['frac_closer_to_B']:.4f} "
                  f"p={ctrl_stats['wilcoxon_p']:.4g}", flush=True)

            main_hit = main_stats["frac_closer_to_B"] > 0.5 and main_stats["wilcoxon_p"] < ALPHA
            ctrl_hit = ctrl_stats["frac_closer_to_B"] < 0.5 and ctrl_stats["wilcoxon_p"] < ALPHA
            if main_hit:
                print(f"HIT main {cfg_name} {name} frac_B={main_stats['frac_closer_to_B']:.4f} "
                      f"p={main_stats['wilcoxon_p']:.4g}", flush=True)
                hits.append(dict(backbone=cfg_name, dataset=name, experiment="main", **main_stats))
            if ctrl_hit:
                print(f"HIT ctrl {cfg_name} {name} frac_B={ctrl_stats['frac_closer_to_B']:.4f} "
                      f"p={ctrl_stats['wilcoxon_p']:.4g}", flush=True)
                hits.append(dict(backbone=cfg_name, dataset=name, experiment="control", **ctrl_stats))

            out_dir = os.path.join(ROOT, f"results_{cfg_name}")
            os.makedirs(out_dir, exist_ok=True)
            df.to_csv(os.path.join(out_dir, f"{name}_pair_distances.csv"), index=False)

            df.insert(0, "backbone", cfg_name)
            all_summaries.append(df)

            del model
            torch.cuda.empty_cache()

        del bundle

    full = pd.concat(all_summaries, ignore_index=True)
    full.to_csv(os.path.join(ROOT, "scan_all_pair_distances.csv"), index=False)
    hits_df = pd.DataFrame(hits)
    hits_df.to_csv(os.path.join(ROOT, "scan_hits.csv"), index=False)

    print(f"\n\nSCAN DONE. total hits: {len(hits)}", flush=True)
    if len(hits):
        print(hits_df.to_string(index=False), flush=True)
    else:
        print("no hits found across any (backbone, dataset) combination", flush=True)


if __name__ == "__main__":
    main()
