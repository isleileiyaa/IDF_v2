"""
RIDDE_NoRetrieval_Base_重新验证实验方案 -- steps 6-9.

Reads results/oracle_cache/{dataset}_truebase.npz (augment_mode=baseline
TRAINED with output_patch_embedding unfrozen -- 方案C fair No-Retrieval Base,
see pretrain.py's 'baseline' freeze branch), {dataset}_rag.npz (real
idf_clean_dis retrieval), and optionally the kill_retrieval negative control
-- tried as either {dataset}_shuffled.npz or {dataset}_base.npz (this repo
has cached it under the older 'base' tag; NEVER treat it as the real Base,
per the doc's explicit "禁止替代" -- it only answers "correct vs random
retrieval"). Computes, for the whole prediction horizon (no DCT band split --
that question is already closed; this is the sample-level question):

  - per-sample MSE for each available arm
  - Delta_i = l_i^Base - l_i^RAG, and normalized r_i = Delta_i / (l_i^Base + l_i^RAG + eps)
  - L^Base, L^RAG, L^Shuffled (means), Lfixed = min(L^Base, L^RAG)
  - Loracle = mean_i[min(l_i^Base, l_i^RAG)]  (min PER SAMPLE, then average --
    doc's explicit ordering requirement, do not average losses first)
  - Oracle gain (%) = (Lfixed - Loracle) / Lfixed * 100
  - p+ = P(Delta_i > 0) [RAG better], p- = P(Delta_i < 0) [RAG worse]
  - Delta_i distribution: mean, median, 25/75pct, 10/90pct
  - non-trivial effect proportion: |r_i| > tau for tau in {0.5%, 1%, 2%}
  - Shuffled vs RAG gap (L^Shuffled - L^RAG), to see whether whatever gain
    RAG has over Base actually depends on genuine query-neighbor correspondence
  - Go/No-Go classification per the doc's section 5 table

Usage:
    python analyze_sample_oracle.py --datasets ETTh1 ETTh2 ETTm1 ETTm2 weather exchange_rate electricity
"""
import os
import argparse
import numpy as np

EPS = 1e-8


def _to_2d(arr, name):
    """Normalize a cached preds/trues array to plain (N, L).

    See analyze_oracle.py's _to_2d docstring: utils.tools.test_retrieve's
    reshape line collapses an already-(N, L) ChronosBoltRetrieve output into
    (1, N, L). Handle that (and any other stray leading singleton dims) here.
    """
    arr = np.asarray(arr)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f'{name}: expected a 2D (N, L) array after squeezing leading '
                          f'singleton dims, got shape {arr.shape} -- inspect the .npz directly.')
    return arr


def _load_arm(cache_dir, dataset, tag):
    path = os.path.join(cache_dir, f'{dataset}_{tag}.npz')
    if not os.path.exists(path):
        return None
    d = np.load(path)
    pred = _to_2d(d['preds'], f'{dataset} {tag} preds')
    true = _to_2d(d['trues'], f'{dataset} {tag} trues')
    return pred, true


def _load_shuffled_arm(cache_dir, dataset):
    """The kill_retrieval negative control. Some utils/tools.py patches tag it
    'shuffled' (this repo's newer naming); others keep the original 'base' tag
    (the pre-existing shuffle-ablation cache from before the truebase work --
    kept as-is on purpose to avoid re-running/renaming an existing 1.1GB
    cache). Try both, in that order, so either tagging scheme works."""
    for tag in ('shuffled', 'base'):
        arm = _load_arm(cache_dir, dataset, tag)
        if arm is not None:
            return arm
    return None


def per_sample_mse(pred, true):
    return ((pred - true) ** 2).mean(axis=1)


def analyze_one(dataset, cache_dir):
    base = _load_arm(cache_dir, dataset, 'truebase')
    rag = _load_arm(cache_dir, dataset, 'rag')
    shuf = _load_shuffled_arm(cache_dir, dataset)

    if base is None or rag is None:
        missing = 'truebase' if base is None else 'rag'
        print(f'[{dataset}] missing {missing} cache, skipping')
        return None

    pred_b, true_b = base
    pred_r, true_r = rag
    n = min(pred_b.shape[0], pred_r.shape[0])
    if pred_b.shape[0] != pred_r.shape[0]:
        print(f'[{dataset}] WARNING: truebase n={pred_b.shape[0]} != rag n={pred_r.shape[0]}, '
              f'truncating to {n} -- cannot guarantee index-for-index pairing, treat with caution')
    pred_b, true_b = pred_b[:n], true_b[:n]
    pred_r, true_r = pred_r[:n], true_r[:n]

    true_mismatch = np.abs(true_b - true_r).mean()
    if true_mismatch > 1e-4:
        print(f'[{dataset}] WARNING: truebase/rag ground truth differ (mean abs diff='
              f'{true_mismatch:.6f}) -- eval runs may not be paired (same eval_split/order?)')

    l_b = per_sample_mse(pred_b, true_b)
    l_r = per_sample_mse(pred_r, true_r)

    L_b = l_b.mean()
    L_r = l_r.mean()
    L_fixed = min(L_b, L_r)

    # doc's ordering requirement: min PER SAMPLE first, then average
    L_oracle = np.minimum(l_b, l_r).mean()
    G_oracle = (L_fixed - L_oracle) / L_fixed * 100 if L_fixed > 0 else float('nan')

    delta = l_b - l_r  # >0 means RAG better (lower loss)
    r_eff = delta / (l_b + l_r + EPS)

    p_pos = (delta > 0).mean()
    p_neg = (delta < 0).mean()

    delta_stats = {
        'mean': delta.mean(),
        'median': np.median(delta),
        'p25': np.percentile(delta, 25),
        'p75': np.percentile(delta, 75),
        'p10': np.percentile(delta, 10),
        'p90': np.percentile(delta, 90),
    }

    nontrivial = {tau: (np.abs(r_eff) > tau).mean() for tau in (0.005, 0.01, 0.02)}

    L_shuf = None
    shuf_vs_rag = None
    if shuf is not None:
        pred_s, true_s = shuf
        ns = min(n, pred_s.shape[0])
        l_s = per_sample_mse(pred_s[:ns], true_s[:ns])
        L_shuf = l_s.mean()
        shuf_vs_rag = L_shuf - L_r  # >0 means real correspondence (RAG) beats shuffled

    return {
        'dataset': dataset, 'n': n,
        'L_base': L_b, 'L_rag': L_r, 'L_shuffled': L_shuf,
        'L_fixed': L_fixed, 'L_oracle': L_oracle, 'G_oracle': G_oracle,
        'p_pos': p_pos, 'p_neg': p_neg,
        'delta_stats': delta_stats, 'nontrivial': nontrivial,
        'shuf_vs_rag': shuf_vs_rag,
    }


def classify(row):
    """Section 5 Go/No-Go table, applied per-dataset (report the macro picture
    separately -- the doc's criterion is 'avg gain >= 2% AND >=5/7 datasets
    show stable positive space', which needs all rows first)."""
    g = row['G_oracle']
    rag_almost_always_worse = row['p_neg'] > 0.9  # RAG loses on >90% of samples: RAG not established
    if row['L_rag'] >= row['L_base'] and rag_almost_always_worse:
        return 'No-Go: RAG 未成立 (Base 几乎始终优于 RAG)'
    if g >= 2:
        return 'Go'
    if g >= 1:
        return 'Conditional Go'
    return 'No-Go: 空间不足'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+',
                     default=['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'weather', 'exchange_rate', 'electricity'])
    ap.add_argument('--cache_dir', default='results/oracle_cache')
    args = ap.parse_args()

    rows = [r for r in (analyze_one(ds, args.cache_dir) for ds in args.datasets) if r is not None]
    if not rows:
        print('No datasets with both truebase and rag caches found -- nothing to report.')
        return

    header = (f"{'Dataset':<14}{'Base MSE':>10}{'RAG MSE':>10}{'Shuf MSE':>10}{'p+':>7}{'p-':>7}"
              f"{'|r|>1%':>8}{'Oracle MSE':>11}{'Gain%':>8}{'Verdict':>28}")
    print(header)
    print('-' * len(header))
    for row in rows:
        shuf_str = f"{row['L_shuffled']:.5f}" if row['L_shuffled'] is not None else 'n/a'
        verdict = classify(row)
        print(f"{row['dataset']:<14}{row['L_base']:>10.5f}{row['L_rag']:>10.5f}{shuf_str:>10}"
              f"{row['p_pos']:>7.3f}{row['p_neg']:>7.3f}{row['nontrivial'][0.01]:>8.3f}"
              f"{row['L_oracle']:>11.5f}{row['G_oracle']:>8.2f}{verdict:>28}")

    gains = [row['G_oracle'] for row in rows]
    macro_gain = np.mean(gains)
    n_stable_positive = sum(1 for g in gains if g >= 1)
    print('-' * len(header))
    print(f"Macro avg Oracle gain: {macro_gain:.2f}%  |  "
          f"datasets with gain >= 1%: {n_stable_positive}/{len(rows)}  |  "
          f"datasets with gain >= 2%: {sum(1 for g in gains if g >= 2)}/{len(rows)}")

    print()
    print('Per-dataset Delta_i (l_base - l_rag) distribution:')
    dist_header = f"{'Dataset':<14}{'mean':>10}{'median':>10}{'p10':>10}{'p25':>10}{'p75':>10}{'p90':>10}"
    print(dist_header)
    for row in rows:
        s = row['delta_stats']
        print(f"{row['dataset']:<14}{s['mean']:>10.5f}{s['median']:>10.5f}{s['p10']:>10.5f}"
              f"{s['p25']:>10.5f}{s['p75']:>10.5f}{s['p90']:>10.5f}")

    print()
    print('Non-trivial effect proportion |r_i| > tau (sensitivity across tau):')
    tau_header = f"{'Dataset':<14}{'tau=0.5%':>10}{'tau=1%':>10}{'tau=2%':>10}"
    print(tau_header)
    for row in rows:
        nt = row['nontrivial']
        print(f"{row['dataset']:<14}{nt[0.005]:>10.3f}{nt[0.01]:>10.3f}{nt[0.02]:>10.3f}")

    print()
    print('Macro Go/No-Go read (section 5): Go needs avg gain >= 2% AND >= 5/7 datasets with '
          'stable positive space; Conditional Go ~= avg gain in 1-2% or gains concentrated in a few '
          'datasets; No-Go otherwise. See per-row Verdict column above for the per-dataset picture.')
    if macro_gain >= 2 and n_stable_positive >= max(5, int(0.7 * len(rows))):
        print('==> Macro verdict: GO')
    elif macro_gain >= 1:
        print('==> Macro verdict: CONDITIONAL GO')
    else:
        print('==> Macro verdict: NO-GO (insufficient space)')


if __name__ == '__main__':
    main()
