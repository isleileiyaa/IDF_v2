"""
RIDDE_BaseRAG频段Oracle诊断方案 -- steps 3 & 4 (docx 表6 / 表 000-111 / Comp_delta).

Reads results/oracle_cache/{dataset}_base.npz and {dataset}_rag.npz (each with
'preds' and 'trues' arrays of shape (N, L), produced by zeroshot.py + the
test_retrieve patch), computes:

  - Base / RAG whole-horizon MSE
  - Sample-level Oracle (pick Base or RAG per sample, whole horizon)
  - 3-band Oracle (pick Base or RAG independently per DCT band: low/mid/high)
  - H^f_abs, H^f_rel = extra headroom from band-level vs sample-level oracle
  - 000-111 sign-combination proportions (raw, unthresholded)
  - Comp_delta for delta in {0.01, 0.02, 0.05}: fraction of samples where at
    least one band clearly favors Base AND at least one other clearly favors
    RAG, using the neutral-zone relative-effect definition from the docx.

Does NOT yet do the robustness battery (random-band permutation test,
cross-seed aggregation, block bootstrap, boundary sensitivity) -- those need
multiple seeds' worth of cached results first. Run this once the first
dataset's base/rag caches exist to sanity-check the pipeline and get a first
read on whether Hf looks big enough to bother with the rest.

Usage:
    python analyze_oracle.py --datasets ETTh1 ETTh2 ETTm1 ETTm2 weather exchange_rate
    python analyze_oracle.py --datasets ETTh1 --low 8 --mid 16   # override band edges
"""
import os
import argparse
import numpy as np
from scipy.fftpack import dct

EPS = 1e-8


def band_losses(pred, true, low, mid):
    """Per-sample, per-band MSE via an orthonormal DCT of the error vector.

    pred, true: (N, L). Returns dict with 'low','mid','high' each (N,) and
    'total' (N,) == whole-horizon MSE (bands sum to total exactly, since the
    DCT-II with norm='ortho' is an orthonormal transform -> Parseval holds).
    """
    err = pred - true  # (N, L)
    L = err.shape[1]
    E = dct(err, type=2, norm='ortho', axis=1)  # (N, L)
    sq = E ** 2
    out = {
        'low': sq[:, :low].sum(axis=1) / L,
        'mid': sq[:, low:mid].sum(axis=1) / L,
        'high': sq[:, mid:].sum(axis=1) / L,
    }
    out['total'] = out['low'] + out['mid'] + out['high']
    return out


def _to_2d(arr, name):
    """Normalize a cached preds/trues array to plain (N, L).

    utils.tools.test_retrieve's own reshape line (`preds.reshape(-1,
    preds.shape[-2], preds.shape[-1])`) is a no-op for most model paths but,
    for ChronosBoltRetrieve specifically, `pred` per batch is already 2D
    (batch, pred_len) -- so by the time that line runs on the *already
    concatenated* (N, L) array, `shape[-2]`/`shape[-1]` are N and L (its own
    two axes), and reshape(-1, N, L) collapses the leading dim to 1, giving
    (1, N, L) instead of (N, L). Confirmed by reproducing that exact line on
    a (500, 64) array -- it comes out (1, 500, 64). Handle that case (and any
    other stray singleton leading dims) here rather than assuming (N, L).
    """
    arr = np.asarray(arr)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f'{name}: expected a 2D (N, L) array after squeezing leading '
                          f'singleton dims, got shape {arr.shape} -- inspect the .npz directly.')
    return arr


def analyze_one(dataset, cache_dir, low, mid):
    base_path = os.path.join(cache_dir, f'{dataset}_base.npz')
    rag_path = os.path.join(cache_dir, f'{dataset}_rag.npz')
    if not (os.path.exists(base_path) and os.path.exists(rag_path)):
        print(f'[{dataset}] missing cache ({base_path} / {rag_path}), skipping')
        return None

    b = np.load(base_path)
    r = np.load(rag_path)
    pred_b = _to_2d(b['preds'], f'{dataset} base preds')
    true_b = _to_2d(b['trues'], f'{dataset} base trues')
    pred_r = _to_2d(r['preds'], f'{dataset} rag preds')
    true_r = _to_2d(r['trues'], f'{dataset} rag trues')

    if pred_b.shape != pred_r.shape:
        print(f'[{dataset}] WARNING: base shape {pred_b.shape} != rag shape {pred_r.shape}, '
              f'cannot pair samples index-for-index -- results below are unreliable until fixed')
    n = min(pred_b.shape[0], pred_r.shape[0])
    pred_b, true_b = pred_b[:n], true_b[:n]
    pred_r, true_r = pred_r[:n], true_r[:n]
    true_mismatch = np.abs(true_b - true_r).mean()
    if true_mismatch > 1e-4:
        print(f'[{dataset}] WARNING: base/rag ground-truth arrays differ '
              f'(mean abs diff={true_mismatch:.6f}) -- the two eval runs may not be paired '
              f'(different eval_split/order/dataset config). Treat results with caution.')

    lb = band_losses(pred_b, true_b, low, mid)
    lr = band_losses(pred_r, true_r, low, mid)

    base_mse = lb['total'].mean()
    rag_mse = lr['total'].mean()

    sample_oracle = np.minimum(lb['total'], lr['total']).mean()

    band_oracle_per_sample = (
        np.minimum(lb['low'], lr['low'])
        + np.minimum(lb['mid'], lr['mid'])
        + np.minimum(lb['high'], lr['high'])
    )
    band_oracle = band_oracle_per_sample.mean()

    hf_abs = sample_oracle - band_oracle
    hf_rel = hf_abs / sample_oracle * 100 if sample_oracle > 0 else float('nan')

    signs_raw = {
        band: (lb[band] > lr[band]).astype(int)
        for band in ('low', 'mid', 'high')
    }
    combo = signs_raw['low'] * 4 + signs_raw['mid'] * 2 + signs_raw['high']
    combo_counts = np.bincount(combo, minlength=8) / n

    r_eff = {
        band: (lb[band] - lr[band]) / (lb[band] + lr[band] + EPS)
        for band in ('low', 'mid', 'high')
    }
    comp = {}
    for delta in (0.01, 0.02, 0.05):
        s = {band: np.where(r_eff[band] > delta, 1, np.where(r_eff[band] < -delta, -1, 0))
             for band in ('low', 'mid', 'high')}
        stacked = np.stack([s['low'], s['mid'], s['high']], axis=1)
        has_favor_base = (stacked == -1).any(axis=1)
        has_favor_rag = (stacked == 1).any(axis=1)
        comp[delta] = (has_favor_base & has_favor_rag).mean()

    return {
        'dataset': dataset, 'n': n,
        'base_mse': base_mse, 'rag_mse': rag_mse,
        'sample_oracle': sample_oracle, 'band_oracle': band_oracle,
        'hf_abs': hf_abs, 'hf_rel': hf_rel,
        'combo_counts': combo_counts, 'comp': comp,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'weather', 'exchange_rate'])
    ap.add_argument('--cache_dir', default='results/oracle_cache')
    ap.add_argument('--low', type=int, default=8, help='low band = coeffs [0, low)')
    ap.add_argument('--mid', type=int, default=16, help='mid band = coeffs [low, mid); high = [mid, L)')
    args = ap.parse_args()

    rows = []
    for ds in args.datasets:
        res = analyze_one(ds, args.cache_dir, args.low, args.mid)
        if res is not None:
            rows.append(res)

    if not rows:
        print('No datasets with both base and rag caches found -- nothing to report.')
        return

    print()
    header = f"{'Dataset':<14}{'Base':>10}{'RAG':>10}{'SampleO.':>10}{'BandO.':>10}{'Hf_rel%':>10}{'Comp0.01':>10}{'Comp0.02':>10}{'Comp0.05':>10}"
    print(header)
    print('-' * len(header))
    for row in rows:
        print(f"{row['dataset']:<14}{row['base_mse']:>10.5f}{row['rag_mse']:>10.5f}"
              f"{row['sample_oracle']:>10.5f}{row['band_oracle']:>10.5f}{row['hf_rel']:>10.2f}"
              f"{row['comp'][0.01]:>10.3f}{row['comp'][0.02]:>10.3f}{row['comp'][0.05]:>10.3f}")

    macro = {
        k: np.mean([row[k] for row in rows])
        for k in ('base_mse', 'rag_mse', 'sample_oracle', 'band_oracle', 'hf_rel')
    }
    macro_comp = {d: np.mean([row['comp'][d] for row in rows]) for d in (0.01, 0.02, 0.05)}
    print('-' * len(header))
    print(f"{'Macro Avg.':<14}{macro['base_mse']:>10.5f}{macro['rag_mse']:>10.5f}"
          f"{macro['sample_oracle']:>10.5f}{macro['band_oracle']:>10.5f}{macro['hf_rel']:>10.2f}"
          f"{macro_comp[0.01]:>10.3f}{macro_comp[0.02]:>10.3f}{macro_comp[0.05]:>10.3f}")
    print()
    print('NOTE: Macro Avg. mixes datasets with different sampling rates -- the low/mid/high')
    print('bands do NOT mean the same real-world period across rows (see earlier discussion).')
    print('Read each dataset row on its own; treat Macro Avg. as a rough pointer, not a result.')

    print()
    print('000-111 sign combinations (bit order low,mid,high; 1 = RAG better on that band):')
    labels = [f'{i:03b}' for i in range(8)]
    for row in rows:
        counts_str = '  '.join(f'{lab}={c:.3f}' for lab, c in zip(labels, row['combo_counts']))
        print(f"  {row['dataset']:<12}{counts_str}")

    print()
    print('Still missing before this is decision-grade (see checklist):')
    print('  - random band-grouping permutation test (100x) vs these Hf_rel numbers')
    print('  - cross-seed repetition (this is a single seed per arm)')
    print('  - block bootstrap for significance')
    print('  - boundary sensitivity check on a held-out split')


if __name__ == '__main__':
    main()