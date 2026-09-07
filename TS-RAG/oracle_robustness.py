"""
oracle_robustness.py -- 文档第5节"必做对照与统计"里的两项：随机频率分组置换检验、
块 Bootstrap。读取跟 analyze_oracle.py 一样的 results/oracle_cache/{dataset}_{base|rag}.npz。

用法:
    python oracle_robustness.py --datasets ETTh1 ETTh2 ETTm1 ETTm2 weather exchange_rate electricity
"""
import os
import argparse
import numpy as np
from scipy.fftpack import dct

EPS = 1e-8


def _to_2d(arr, name):
    arr = np.asarray(arr)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f'{name}: expected 2D (N, L), got {arr.shape}')
    return arr


def load_pair(dataset, cache_dir):
    base_path = os.path.join(cache_dir, f'{dataset}_base.npz')
    rag_path = os.path.join(cache_dir, f'{dataset}_rag.npz')
    if not (os.path.exists(base_path) and os.path.exists(rag_path)):
        return None
    b, r = np.load(base_path), np.load(rag_path)
    pred_b = _to_2d(b['preds'], 'base preds')
    true_b = _to_2d(b['trues'], 'base trues')
    pred_r = _to_2d(r['preds'], 'rag preds')
    true_r = _to_2d(r['trues'], 'rag trues')
    n = min(pred_b.shape[0], pred_r.shape[0])
    return pred_b[:n] - true_b[:n], pred_r[:n] - true_r[:n]  # err_b, err_r


def group_losses(err, index_groups):
    """err: (N, L). index_groups: list of coefficient-index arrays.
    Returns list of (N,) per-sample per-group losses, same order as groups."""
    L = err.shape[1]
    E = dct(err, type=2, norm='ortho', axis=1)
    sq = E ** 2
    return [sq[:, idx].sum(axis=1) / L for idx in index_groups]


def hf_rel(losses_b, losses_r):
    total_b = sum(losses_b)
    total_r = sum(losses_r)
    sample_oracle = np.minimum(total_b, total_r).mean()
    band_oracle = sum(np.minimum(lb, lr) for lb, lr in zip(losses_b, losses_r)).mean()
    if sample_oracle <= 0:
        return float('nan')
    return (sample_oracle - band_oracle) / sample_oracle * 100


def random_grouping_test(err_b, err_r, group_sizes, n_perm, seed):
    """Null distribution: same group sizes (8/8/48), but coefficients assigned
    randomly instead of contiguous low/mid/high. Tests whether the real
    contiguous DCT bands beat pure 'more decision points' luck."""
    L = err_b.shape[1]
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(L)
        groups, start = [], 0
        for size in group_sizes:
            groups.append(perm[start:start + size])
            start += size
        null[i] = hf_rel(group_losses(err_b, groups), group_losses(err_r, groups))
    return null


def block_bootstrap(err_b, err_r, low, mid, block_len, n_boot, seed):
    """Resample contiguous blocks of sample indices (not i.i.d. samples) to
    respect that consecutive sliding-window samples overlap in time."""
    n = err_b.shape[0]
    groups = [np.arange(0, low), np.arange(low, mid), np.arange(mid, err_b.shape[1])]
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_len))
    boot = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, max(n - block_len, 1), size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_len) for s in starts])[:n]
        idx = np.clip(idx, 0, n - 1)
        boot[i] = hf_rel(group_losses(err_b[idx], groups), group_losses(err_r[idx], groups))
    return boot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', required=True)
    ap.add_argument('--cache_dir', default='results/oracle_cache')
    ap.add_argument('--low', type=int, default=8)
    ap.add_argument('--mid', type=int, default=16)
    ap.add_argument('--n_perm', type=int, default=100)
    ap.add_argument('--n_boot', type=int, default=200)
    ap.add_argument('--block_len', type=int, default=64,
                     help='bootstrap block length in samples; pre-register this, do not tune post-hoc')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    group_sizes = (args.low, args.mid - args.low, 64 - args.mid)
    header = f"{'Dataset':<14}{'ObsHf_rel%':>12}{'NullMean%':>12}{'NullP95%':>12}{'Percentile':>12}{'BootMean%':>12}{'Boot[2.5,97.5]':>22}"
    print(header)
    print('-' * len(header))

    for ds in args.datasets:
        pair = load_pair(ds, args.cache_dir)
        if pair is None:
            print(f'[{ds}] missing cache, skipping')
            continue
        err_b, err_r = pair

        real_groups = [np.arange(0, args.low), np.arange(args.low, args.mid), np.arange(args.mid, err_b.shape[1])]
        observed = hf_rel(group_losses(err_b, real_groups), group_losses(err_r, real_groups))

        null = random_grouping_test(err_b, err_r, group_sizes, args.n_perm, args.seed)
        percentile = (null < observed).mean() * 100

        boot = block_bootstrap(err_b, err_r, args.low, args.mid, args.block_len, args.n_boot, args.seed)
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

        print(f"{ds:<14}{observed:>12.3f}{null.mean():>12.3f}{np.percentile(null,95):>12.3f}"
              f"{percentile:>12.1f}{boot.mean():>12.3f}{f'[{ci_lo:.3f}, {ci_hi:.3f}]':>22}")

    print()
    print('Percentile = 观测到的真实连续频段 Hf_rel 超过了多少比例的随机分组结果。')
    print('文档建议 >95 才算"超过随机分组"；如果 Percentile 远低于95，说明真实频段划分')
    print('并不比瞎分好，Hf_rel 数值本身大概率就是决策自由度带来的假象，不是真信号。')


if __name__ == '__main__':
    main()