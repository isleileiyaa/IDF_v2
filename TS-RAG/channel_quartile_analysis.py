"""
Per-channel smoothness quartile analysis for RIDDE vs baseline on a multichannel dataset.

For each channel c, computes a smoothness score S_c from the test-set ground truth
(relative roughness: mean over windows of mean((diff(y))^2) / var(y)), splits the
channels into 4 quartiles (Q1 = smoothest ... Q4 = most oscillatory), then reports
RIDDE (idf_clean_dis) vs baseline (frozen ChronosBoltRetrieve, augment_mode=baseline)
MSE/MAE and improvement% per quartile.
"""
import os
import json
import argparse
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from transformers import AutoConfig

from data_provider.data_factory import data_provider
from models.ChronosBolt import ChronosBoltModelForForecastingWithRetrieval
from retrieve import load_database, frequency_dict

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

DATASET_CONFIGS = {
    "electricity": {"root_path": "../datasets/electricity/", "data_path": "electricity.csv", "metadata_frequency": "hour"},
    "traffic": {"root_path": "../datasets/traffic/", "data_path": "traffic.csv", "metadata_frequency": "hour"},
    "solar": {"root_path": "../datasets/solar/", "data_path": "solar.csv", "metadata_frequency": "10minutes"},
    "PEMS08": {"root_path": "../datasets/PEMS08/", "data_path": "PEMS08.csv", "metadata_frequency": "5minutes"},
    "weather": {"root_path": "../datasets/weather/", "data_path": "weather.csv", "metadata_frequency": "10minutes"},
    "CzeLan": {"root_path": "../datasets/CzeLan/", "data_path": "CzeLan.csv", "metadata_frequency": "30minutes"},
    "ZafNoo": {"root_path": "../datasets/ZafNoo/", "data_path": "ZafNoo.csv", "metadata_frequency": "30minutes"},
    "AQWan": {"root_path": "../datasets/AQWan/", "data_path": "AQWan.csv", "metadata_frequency": "hour"},
    "Wind": {"root_path": "../datasets/Wind/", "data_path": "Wind.csv", "metadata_frequency": "15minutes"},
    "ILI": {"root_path": "../datasets/ILI/", "data_path": "ILI.csv", "metadata_frequency": "week"},
    "exchange_rate": {"root_path": "../datasets/exchange_rate/", "data_path": "exchange_rate.csv", "metadata_frequency": "hour"},
    "ETTh1": {"root_path": "../datasets/ETT-small/", "data_path": "ETTh1.csv", "metadata_frequency": "hour", "data": "ett_h_retrieve"},
    "ETTh2": {"root_path": "../datasets/ETT-small/", "data_path": "ETTh2.csv", "metadata_frequency": "hour", "data": "ett_h_retrieve"},
    "ETTm1": {"root_path": "../datasets/ETT-small/", "data_path": "ETTm1.csv", "metadata_frequency": "minute", "data": "ett_m_retrieve"},
    "ETTm2": {"root_path": "../datasets/ETT-small/", "data_path": "ETTm2.csv", "metadata_frequency": "minute", "data": "ett_m_retrieve"},
}


def build_args(dataset, augment_mode, checkpoint_model_path):
    cfg = DATASET_CONFIGS[dataset]
    return SimpleNamespace(
        model_id=f"{dataset}_zeroshot_512_pred_64_512_retrieve_64_{augment_mode}",
        root_path=cfg["root_path"],
        data_path=cfg["data_path"],
        data=cfg.get("data", "custom_retrieve"),
        features="M",
        freq="h",
        target="OT",
        embed="timeF",
        percent=100,
        max_len=-1,
        seq_len=512,
        pred_len=64,
        label_len=0,
        batch_size=256,
        num_workers=0,
        top_k=10,
        retrieval_database_dir="/home/fenglei/TS-RAG-main/retrieval_database/",
        dimension=768,
        embedding_model_type="chronos",
        metadata_frequency=cfg["metadata_frequency"],
        metadata_database_name=dataset,
        metadata={},
        mode="only_self_train",
        embedding_tuning=None,
        lookback_length=512,
        augment_mode=augment_mode,
        rawx_norm="zscore",
        retrieval_mode="embedding",
        tau=0.1,
        eval_split="test",
        checkpoint_model_path=checkpoint_model_path,
        pretrained_model_path="/home/fenglei/TS-RAG-main/TS-RAG/checkpoints/base/",
        debug_shapes=False,
        return_feature_id=0,
        gpu_loc=0,
    )


def load_eval_data(args):
    args.metadata["lookback_length"] = args.lookback_length
    args.metadata["frequency"] = args.metadata_frequency
    args.metadata["database_name"] = args.metadata_database_name.split(" ")

    retrieval_database_names = "_".join(args.metadata["database_name"])
    retrieved_filename = (
        f'{args.data_path.split(".")[0]}_retrieve_{retrieval_database_names}_'
        f'{args.metadata["lookback_length"]}_{args.mode}_{args.embedding_tuning}.csv'
    )
    retrieved_data_path = os.path.join(args.root_path, retrieved_filename)
    assert os.path.exists(retrieved_data_path), retrieved_data_path
    args.data_path = retrieved_filename

    kb_name = args.metadata["database_name"][0]
    kb_frequency = frequency_dict.get(kb_name, args.metadata["frequency"])
    database = load_database(
        os.path.join(args.retrieval_database_dir, f'{kb_name}_{kb_frequency}_{args.metadata["lookback_length"]}.pkl')
    )
    retriever_rawdata = [database[v]["raw_data"] for v in database.keys()]

    scaler = StandardScaler()
    retriever_rawdata = np.array(retriever_rawdata).T
    scaler.fit(retriever_rawdata)
    retriever_rawdata = scaler.transform(retriever_rawdata).T

    eval_data, eval_loader = data_provider(args, args.eval_split, retriever_rawdata=retriever_rawdata)
    return eval_data, eval_loader


def load_model(args, device):
    config = AutoConfig.from_pretrained(args.pretrained_model_path)
    if hasattr(config, "chronos_config"):
        config.chronos_config["context_length"] = args.seq_len
        config.chronos_config["prediction_length"] = args.pred_len
    model = ChronosBoltModelForForecastingWithRetrieval.from_pretrained(
        args.pretrained_model_path, config=config, augment=args.augment_mode, low_cpu_mem_usage=False
    )
    model.debug_shapes = False
    model._debug_shapes_printed = False
    model.tau = args.tau
    if args.augment_mode != "baseline":
        ckpt = torch.load(args.checkpoint_model_path, map_location="cpu")
        new_state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in ckpt.items())
        msg = model.load_state_dict(new_state_dict, strict=False)
        print(f"[{args.augment_mode}] loaded checkpoint {args.checkpoint_model_path}")
        print("  missing:", msg.missing_keys[:5], "unexpected:", msg.unexpected_keys[:5])
    else:
        print("[baseline] frozen Chronos-Bolt zero-shot, no retrieval checkpoint")
    model.to(device)
    model.eval()
    return model


def run_inference(model, eval_loader, device):
    preds, trues = [], []
    central_idx = int(torch.abs(torch.tensor(QUANTILES) - 0.5).argmin())
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark, retrieved_seqs, distances in eval_loader:
            batch_x = batch_x.float().to(device).squeeze()
            batch_y = batch_y.float().to(device).squeeze()
            retrieved_seqs = retrieved_seqs.float().to(device)
            distances = distances.float().to(device)
            outputs = model(context=batch_x, target=batch_y, retrieved_seq=retrieved_seqs, distances=distances)
            outputs = outputs.quantile_preds.to(batch_x)[:, central_idx]
            preds.append(outputs.detach().cpu())
            trues.append(batch_y.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    trues = torch.cat(trues, dim=0).numpy()
    return preds, trues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="electricity", choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--n_groups", type=int, default=4, help="number of smoothness groups (G1=smoothest ... Gn=most oscillatory)")
    parser.add_argument("--gpu_loc", type=int, default=0)
    parser.add_argument(
        "--ridde_checkpoint",
        type=str,
        default="/home/fenglei/TS-RAG-main/TS-RAG/checkpoints/"
        "data50m_idf_clean_dis_512_pred64_lookback512_top10_lr0.0003_drop0.2_adamw_"
        "cosanneal_step10000_bs256_no_embeddingtuning_rho0_0_0_0_final.pth",
    )
    parser.add_argument("--compare_mode", type=str, default="baseline", choices=["baseline", "moe"],
                         help="what to compare RIDDE (idf_clean_dis) against")
    parser.add_argument(
        "--compare_checkpoint",
        type=str,
        default="/home/fenglei/TS-RAG-main/TS-RAG/checkpoints/"
        "data50m_moe_512_pred64_lookback512_top10_lr0.0003_drop0.2_adamw_"
        "cosanneal_step10000_bs256_no_embeddingtuning_final.pth",
        help="checkpoint for --compare_mode=moe (ignored for baseline, which needs no checkpoint)",
    )
    parser.add_argument("--out_json", type=str, default=None)
    cli_args = parser.parse_args()
    if cli_args.out_json is None:
        cli_args.out_json = f"results/forecast_evaluation/channel_quartile_{cli_args.dataset}.json"

    device = torch.device(f"cuda:{cli_args.gpu_loc}" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # Build the eval dataset/loader once (shared across both models -- it's the
    # same windowing regardless of augment_mode, and the retrieval KB pickle can
    # be large (tens of GB) so loading it twice would double an already slow step).
    print(f"\n===== loading shared eval dataset ({cli_args.dataset}, test split) =====", flush=True)
    base_args = build_args(cli_args.dataset, "idf_clean_dis", cli_args.ridde_checkpoint)
    base_args.gpu_loc = cli_args.gpu_loc
    eval_data, eval_loader = load_eval_data(base_args)
    n_channel = eval_data.enc_in
    tot_len = eval_data.tot_len
    pred_len = base_args.pred_len
    print(f"n_channel={n_channel}, tot_len(windows/channel)={tot_len}", flush=True)

    compare_ckpt = None if cli_args.compare_mode == "baseline" else cli_args.compare_checkpoint
    results = {}
    for tag, augment_mode, ckpt in [
        ("ridde", "idf_clean_dis", cli_args.ridde_checkpoint),
        ("compare", cli_args.compare_mode, compare_ckpt),
    ]:
        print(f"\n===== running {tag} (augment_mode={augment_mode}) =====", flush=True)
        args = build_args(cli_args.dataset, augment_mode, ckpt)
        args.gpu_loc = cli_args.gpu_loc

        model = load_model(args, device)
        preds, trues = run_inference(model, eval_loader, device)
        print("preds/trues shape:", preds.shape, trues.shape, flush=True)
        assert preds.shape[0] == n_channel * tot_len

        preds = preds.reshape(n_channel, tot_len, pred_len)
        trues = trues.reshape(n_channel, tot_len, pred_len)
        results[tag] = {"preds": preds, "trues": trues}

        del model
        torch.cuda.empty_cache()

    # sanity check: ground truth should match between the two runs (same windows/order)
    diff = np.abs(results["ridde"]["trues"] - results["compare"]["trues"]).max()
    print(f"\nmax |trues diff| between ridde/{cli_args.compare_mode} runs: {diff:.6g} (should be ~0)")

    trues_ref = results["ridde"]["trues"]  # (n_channel, tot_len, pred_len)
    n_channel, tot_len, pred_len = trues_ref.shape

    # per-window smoothness: relative roughness = mean(diff(y)^2) / var(y)
    diffs = np.diff(trues_ref, axis=-1)  # (n_channel, tot_len, pred_len-1)
    window_roughness = np.mean(diffs ** 2, axis=-1)  # (n_channel, tot_len)
    window_var = np.var(trues_ref, axis=-1) + 1e-8  # (n_channel, tot_len)
    s = window_roughness / window_var  # (n_channel, tot_len)
    S_c = s.mean(axis=1)  # (n_channel,) channel-level smoothness score, higher = rougher

    order = np.argsort(S_c)  # ascending: smoothest first
    group_edges = np.array_split(order, cli_args.n_groups)  # G1..Gn channel index groups, smoothest first

    print("\nChannel smoothness score S_c: min={:.4f}, median={:.4f}, max={:.4f}".format(
        S_c.min(), np.median(S_c), S_c.max()))

    # per-channel (ungrouped) MSE improve%, for a channel-level Spearman correlation
    # against S_c -- a statistical test of the smoothness-improvement relationship
    # that doesn't depend on how channels happen to be binned into groups.
    ridde_p, ridde_t = results["ridde"]["preds"], results["ridde"]["trues"]
    cmp_p, cmp_t = results["compare"]["preds"], results["compare"]["trues"]
    per_ch_ridde_mse = np.mean((ridde_p - ridde_t) ** 2, axis=(1, 2))
    per_ch_compare_mse = np.mean((cmp_p - cmp_t) ** 2, axis=(1, 2))
    per_ch_improve_mse_pct = 100.0 * (per_ch_compare_mse - per_ch_ridde_mse) / per_ch_compare_mse

    from scipy.stats import spearmanr
    rho, pval = spearmanr(S_c, per_ch_improve_mse_pct)
    print(f"\nChannel-level Spearman correlation (S_c vs improve_MSE%): rho={rho:.4f}, p={pval:.4g}, n={n_channel}")
    print("(negative rho = improvement decreases as channels get more oscillatory, i.e. supports the hypothesis)")

    table_rows = []
    for qi, ch_idx in enumerate(group_edges, start=1):
        row = {"quartile": f"G{qi}", "n_channels": len(ch_idx),
               "S_c_range": [float(S_c[ch_idx].min()), float(S_c[ch_idx].max())]}
        for tag in ["ridde", "compare"]:
            p = results[tag]["preds"][ch_idx]
            t = results[tag]["trues"][ch_idx]
            mse = float(np.mean((p - t) ** 2))
            mae = float(np.mean(np.abs(p - t)))
            row[f"{tag}_mse"] = mse
            row[f"{tag}_mae"] = mae
        row["improve_mse_pct"] = 100.0 * (row["compare_mse"] - row["ridde_mse"]) / row["compare_mse"]
        row["improve_mae_pct"] = 100.0 * (row["compare_mae"] - row["ridde_mae"]) / row["compare_mae"]
        table_rows.append(row)

    compare_label = cli_args.compare_mode
    print()
    print("{:<4}{:>6}   {:>16}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}".format(
        "Grp", "n_ch", "S_c[min,max]", "RIDDE_MSE", f"{compare_label}_MSE", "impMSE%", "RIDDE_MAE", f"{compare_label}_MAE", "impMAE%"))
    for row in table_rows:
        print("{:<4}{:>6}   [{:.3f},{:.3f}]  {:>9.4f} {:>9.4f} {:>8.2f}%  {:>9.4f} {:>9.4f} {:>8.2f}%".format(
            row["quartile"], row["n_channels"], row["S_c_range"][0], row["S_c_range"][1],
            row["ridde_mse"], row["compare_mse"], row["improve_mse_pct"],
            row["ridde_mae"], row["compare_mae"], row["improve_mae_pct"]))

    os.makedirs(os.path.dirname(cli_args.out_json), exist_ok=True)
    with open(cli_args.out_json, "w") as f:
        json.dump({
            "rows": table_rows,
            "S_c": S_c.tolist(),
            "channel_order": order.tolist(),
            "per_channel_improve_mse_pct": per_ch_improve_mse_pct.tolist(),
            "spearman_rho": float(rho),
            "spearman_p": float(pval),
        }, f, indent=2)
    print(f"\nSaved -> {cli_args.out_json}")


if __name__ == "__main__":
    main()
