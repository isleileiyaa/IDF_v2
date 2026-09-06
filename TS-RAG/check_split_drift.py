"""
Quick diagnostic to run BEFORE writing up the round-2 Probe result.

Round 2 (build_probe.py with richer features + CV-based model selection over
logistic regression / gradient boosting) still failed (macro eta=-30.8%,
0/7), and ruled out plain single-fit overfitting: val_acc rose (0.73-0.79)
while test_acc stayed at chance (0.47-0.54) even with proper k-fold CV. That
pattern means the predictive relationship between these features and "which
arm wins" is not the same between the validation split and the test split.

Since these datasets are split chronologically (utils/tools.py's
get_borders: train/val/test are consecutive TIME periods, not random
subsets), that is exactly what a real concept drift would look like -- the
rule "when does RAG beat Base" itself changing over time, not a defect in
the features or the model. This script checks that directly and cheaply:
it only reads the already-cached preds/trues (no GPU, no rerun) and compares
P(RAG wins) between val and test per dataset via a two-proportion z-test,
plus the mean/sign of Delta_i on each split.

Usage:
    python check_split_drift.py --datasets ETTh1 ETTh2 ETTm1 ETTm2 weather exchange_rate electricity
"""
import os
import argparse
import numpy as np
from scipy import stats


def _to_2d(arr):
    arr = np.asarray(arr)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def per_sample_mse(pred, true):
    return ((pred - true) ** 2).mean(axis=1)


def load_delta(cache_dir, dataset, split):
    suffix = '' if split == 'test' else f'_{split}'
    bp = os.path.join(cache_dir, f'{dataset}_truebase{suffix}.npz')
    rp = os.path.join(cache_dir, f'{dataset}_rag{suffix}.npz')
    if not (os.path.exists(bp) and os.path.exists(rp)):
        print(f'[{dataset}/{split}] missing truebase or rag cache, skipping')
        return None
    b, r = np.load(bp), np.load(rp)
    pred_b, true_b = _to_2d(b['preds']), _to_2d(b['trues'])
    pred_r, true_r = _to_2d(r['preds']), _to_2d(r['trues'])
    n = min(pred_b.shape[0], pred_r.shape[0])
    l_b = per_sample_mse(pred_b[:n], true_b[:n])
    l_r = per_sample_mse(pred_r[:n], true_r[:n])
    delta = l_b - l_r  # >0 => RAG better
    y = (delta > 0).astype(int)
    return {'n': n, 'p_pos': float(y.mean()), 'y': y, 'delta_mean': float(delta.mean()),
            'delta_median': float(np.median(delta))}


def two_proportion_ztest(p1, n1, p2, n2):
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return float('nan'), float('nan')
    z = (p1 - p2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(z), float(p_value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+',
                     default=['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'weather', 'exchange_rate', 'electricity'])
    ap.add_argument('--cache_dir', default='results/oracle_cache')
    args = ap.parse_args()

    header = (f"{'Dataset':<14}{'n_val':>7}{'n_test':>7}{'p+_val':>9}{'p+_test':>9}"
              f"{'diff':>8}{'z':>8}{'p_value':>9}{'sign_flip?':>11}")
    print(header)
    print('-' * len(header))
    rows = []
    for ds in args.datasets:
        v = load_delta(args.cache_dir, ds, 'val')
        t = load_delta(args.cache_dir, ds, 'test')
        if v is None or t is None:
            continue
        z, pval = two_proportion_ztest(v['p_pos'], v['n'], t['p_pos'], t['n'])
        sign_flip = (v['delta_mean'] > 0) != (t['delta_mean'] > 0)
        rows.append((ds, v, t, z, pval, sign_flip))
        print(f"{ds:<14}{v['n']:>7}{t['n']:>7}{v['p_pos']:>9.3f}{t['p_pos']:>9.3f}"
              f"{v['p_pos']-t['p_pos']:>8.3f}{z:>8.2f}{pval:>9.4f}{str(sign_flip):>11}")

    if not rows:
        print('No datasets with both val and test caches found.')
        return

    print()
    print('Delta_i mean (l_base - l_rag; >0 means RAG better on average) per split:')
    print(f"{'Dataset':<14}{'delta_mean_val':>16}{'delta_mean_test':>17}{'delta_median_val':>18}{'delta_median_test':>19}")
    for ds, v, t, z, pval, sign_flip in rows:
        print(f"{ds:<14}{v['delta_mean']:>16.6f}{t['delta_mean']:>17.6f}"
              f"{v['delta_median']:>18.6f}{t['delta_median']:>19.6f}")

    n_sig = sum(1 for *_r, pval, _sf in rows if pval < 0.05)
    n_flip = sum(1 for *_r, sf in rows if sf)
    print()
    print(f"Datasets with significant p+ shift (p<0.05): {n_sig}/{len(rows)}  |  "
          f"datasets where the average winner flips between val and test: {n_flip}/{len(rows)}")
    if n_sig >= max(1, int(0.5 * len(rows))) or n_flip >= 1:
        print('==> Consistent with real val/test drift in the Base-vs-RAG label -- the predictive '
              'relationship a probe would need is not the same across time periods here, which is '
              'a property of the chronological split, not of the feature set or model choice.')
    else:
        print('==> No strong evidence of label-rate drift -- val and test agree on roughly how often '
              'RAG wins overall. The Probe failure is more likely a small-effective-sample / noise-'
              'dominated relationship rather than distribution shift per se.')


if __name__ == '__main__':
    main()
