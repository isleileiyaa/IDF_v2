"""
D-RACS 方案 -- 阶段A：协议与 Oracle 复算（对应方案文档第19.1节 阶段A）。

不需要任何新的训练/推理，直接复用 Probe 两轮已经生成的
results/oracle_cache/{dataset}_truebase.npz 和 {dataset}_rag.npz
（测试集缓存）。计算：

  - d_t = r_t - b_t                       (RAG 相对 Base 的修正向量)
  - q_t = ||d_t||_W^2, W = I/pred_len      (修正能量，逐样本)
  - u_t = d_t / sqrt(q_t)                  (单位修正方向)
  - e_t = y_t - b_t                        (Base 的真实残差)
  - a_t = <e_t, u_t>_W                     (真实残差沿修正方向的投影效用)
  - alpha_t* = clip(a_t / sqrt(q_t), 0, 1)  (样本级 Line Oracle 闭式解)

然后报告：
  - Hard Oracle（我们之前算的那个二选一 Oracle，这里重新用同一套代码算一遍
    做交叉验证）
  - Line Oracle（连续 alpha 版本，理论上 >= Hard Oracle 的收益）
  - Interior Oracle 占比：0 < alpha_t* < 1 的样本比例（这部分样本是硬选择
    Oracle 漏掉、只有连续修正才能拿到的空间）
  - G_interior = (L_hard - L_line) / L_fixed （连续版比硬选择版多出的收益）

用法：
    python dracs/stage_a_oracle.py --datasets ETTh1 ETTh2 ETTm1 ETTm2 weather exchange_rate electricity
"""
import os
import argparse
import numpy as np

EPS = 1e-8


def _to_2d(arr):
    arr = np.asarray(arr)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def load_pair(cache_dir, dataset, split='test'):
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
    return {
        'pred_b': pred_b[:n], 'true': true_b[:n], 'pred_r': pred_r[:n], 'n': n,
    }


def compute_dqua(pred_b, pred_r, true):
    """Returns d, q, u, e, a, alpha_star -- all per-sample (N,) or (N, H)."""
    H = pred_b.shape[1]
    d = pred_r - pred_b                          # (N, H)
    q = (d ** 2).mean(axis=1)                    # (N,)  W = I/H
    safe_q = np.sqrt(np.clip(q, EPS, None))
    u = d / safe_q[:, None]                      # (N, H)
    e = true - pred_b                            # (N, H)
    a = (e * u).mean(axis=1)                     # (N,)  <e,u>_W
    alpha_star = np.clip(a / safe_q, 0.0, 1.0)
    return d, q, u, e, a, alpha_star


def loss_at_alpha(pred_b, d, true, alpha):
    """l_t(alpha) = mean((true - pred_b - alpha*d)^2) over horizon, per sample."""
    pred = pred_b + alpha[:, None] * d
    return ((true - pred) ** 2).mean(axis=1)


def analyze_one(dataset, cache_dir, split='test'):
    data = load_pair(cache_dir, dataset, split)
    if data is None:
        return None
    pred_b, pred_r, true, n = data['pred_b'], data['pred_r'], data['true'], data['n']

    d, q, u, e, a, alpha_star = compute_dqua(pred_b, pred_r, true)

    l0 = loss_at_alpha(pred_b, d, true, np.zeros(n))          # = Base loss
    l1 = loss_at_alpha(pred_b, d, true, np.ones(n))           # = RAG loss
    l_star = loss_at_alpha(pred_b, d, true, alpha_star)       # = Line Oracle per-sample loss

    # sanity check against the closed-form quadratic (doc section 3.1):
    # l(alpha) - l(0) = q*alpha^2 - 2*sqrt(q)*a*alpha
    l_star_closed = l0 + q * alpha_star ** 2 - 2 * np.sqrt(np.clip(q, EPS, None)) * a * alpha_star
    max_mismatch = np.abs(l_star - l_star_closed).max()

    L_base, L_rag = l0.mean(), l1.mean()
    L_fixed = min(L_base, L_rag)
    L_hard = np.minimum(l0, l1).mean()
    L_line = l_star.mean()

    G_interior = (L_hard - L_line) / L_fixed * 100 if L_fixed > 0 else float('nan')
    interior_frac = ((alpha_star > 1e-6) & (alpha_star < 1 - 1e-6)).mean()
    mean_alpha = alpha_star.mean()
    frac_alpha_0 = (alpha_star <= 1e-6).mean()
    frac_alpha_1 = (alpha_star >= 1 - 1e-6).mean()

    return {
        'dataset': dataset, 'n': n, 'max_closed_form_mismatch': max_mismatch,
        'L_base': L_base, 'L_rag': L_rag, 'L_fixed': L_fixed,
        'L_hard': L_hard, 'L_line': L_line,
        'G_hard_pct': (L_fixed - L_hard) / L_fixed * 100 if L_fixed > 0 else float('nan'),
        'G_line_pct': (L_fixed - L_line) / L_fixed * 100 if L_fixed > 0 else float('nan'),
        'G_interior_pct': G_interior,
        'interior_frac': interior_frac, 'mean_alpha': mean_alpha,
        'frac_alpha_0': frac_alpha_0, 'frac_alpha_1': frac_alpha_1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+',
                     default=['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'weather', 'exchange_rate', 'electricity'])
    ap.add_argument('--cache_dir', default='results/oracle_cache')
    ap.add_argument('--split', default='test')
    args = ap.parse_args()

    rows = [r for r in (analyze_one(ds, args.cache_dir, args.split) for ds in args.datasets) if r is not None]
    if not rows:
        print('No datasets with truebase+rag caches found.')
        return

    header = (f"{'Dataset':<14}{'L_base':>9}{'L_rag':>9}{'L_hard':>9}{'L_line':>9}"
              f"{'G_hard%':>8}{'G_line%':>8}{'G_interior%':>12}{'interior_frac':>14}{'mean_alpha':>11}")
    print(header)
    print('-' * len(header))
    for r in rows:
        print(f"{r['dataset']:<14}{r['L_base']:>9.5f}{r['L_rag']:>9.5f}{r['L_hard']:>9.5f}{r['L_line']:>9.5f}"
              f"{r['G_hard_pct']:>8.2f}{r['G_line_pct']:>8.2f}{r['G_interior_pct']:>12.2f}"
              f"{r['interior_frac']:>14.3f}{r['mean_alpha']:>11.3f}")

    worst_mismatch = max(r['max_closed_form_mismatch'] for r in rows)
    print()
    print(f"[sanity check] max |l_star - closed_form| across all datasets: {worst_mismatch:.2e} "
          f"(should be ~1e-10, confirms the quadratic-loss algebra is implemented correctly)")

    print()
    print('G_line% (连续修正 Oracle 相对固定最优端点的收益) vs 我们之前算过的 G_hard%(硬选择 Oracle) 对比：')
    print('G_interior% 是"连续修正比硬选择多挖出来的那部分" -- 越大说明硬选择漏掉的中间地带越值得做 D-RACS。')
    print(f"Macro G_hard%: {np.mean([r['G_hard_pct'] for r in rows]):.2f}  |  "
          f"Macro G_line%: {np.mean([r['G_line_pct'] for r in rows]):.2f}  |  "
          f"Macro interior_frac: {np.mean([r['interior_frac'] for r in rows]):.3f}")


if __name__ == '__main__':
    main()
