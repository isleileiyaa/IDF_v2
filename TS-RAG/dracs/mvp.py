"""
D-RACS 方案 -- 阶段B：MVP 最小实现（对应方案文档第7.3节 MVP 版本 + 第19.1节 阶段B）。

MVP 阶段的一个刻意简化（跟方案文档自己区分"MVP / 主模型"两档一致）：
  方案文档要求初始化记忆 M0 来自训练集上的 blocked OOF 预测(见文档5.2节)，
  这需要额外重新跑若干个"留出某个时间块"的 Base/RAG 训练，属于新的 GPU 工作。
  MVP 这一版改用【验证集】的 truebase/rag 缓存作为 M0 的替代来源 -- 验证集
  本来就没参与 Base/RAG 的梯度更新，不存在文档警告的"in-sample 过度乐观"
  问题，而且我们已经有现成缓存，不需要任何新训练。这不是文档要求的最终
  版本，只用于先把在线机制和下面4个诊断跑通、看数量级；正式结论仍需要
  按文档5.2节做真正的 blocked OOF。

在线流程（对应文档 7.2 节伪代码）：
  遍历测试集样本 t=0..T-1（模拟按时间顺序到达）：
    1. 把所有"预测发出后已经过了 pred_len 步、真实答案已揭晓"的历史测试
       样本，计算其 a_i 后加入在线记忆；
    2. 用 (验证集 M0 + 当前已揭晓的在线记忆) 作为候选池，按"状态+方向"联合
       距离找 K 个最近邻，做时间衰减加权，估计当前样本的方向效用 m_hat；
    3. 算出 alpha_t，产出预测，同时记录下面4类诊断：
         a) n_eff：时间衰减 + 距离核加权后的有效样本量
         b) 联合距离下实际找到的近邻数（够不够 K 个、以及有效池子多大）
         c) 当前时刻 t 是否处于"冷启动"（在线记忆里是否已有任何测试期样本
            揭晓），以及 D=pred_len 相对测试流长度占比
         d) 每个时刻可用历史反馈的总量（M0 + 已揭晓在线记忆的池子大小随时间
            的变化曲线）

用法：
    python dracs/mvp.py --dataset weather --K 16 --p_u 16 --eta_u 1.0 --eta_q 1.0 --tau_mult 2
"""
import os
import argparse
import numpy as np

EPS = 1e-8

STATE_FEATURE_KEYS = [
    'q_std', 'q_slope', 'q_roughness', 'q_cv', 'q_acf1', 'q_trend_r2', 'q_last_dev',
    'dist_mean', 'dist_min', 'dist_std', 'dist_top1_gap_norm',
    'neighbor_future_dispersion', 'neighbor_past_corr',
]


def _to_2d(arr):
    arr = np.asarray(arr)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _feat(npz, key, n, fill=0.0):
    if key in npz.files:
        return npz[key][:n]
    return np.full(n, fill)


def load_split_records(cache_dir, dataset, split):
    """Loads truebase/rag/probe_features for one split and builds per-sample
    state vector c_t (raw, unstandardized), direction u_t (raw, length pred_len),
    energy q_t, and (since both b and r predictions + true are all already
    known for any split we load offline) utility a_t -- for the val split this
    is genuinely legitimate ground truth (val was held out from training); for
    the test split we still compute a_t here for evaluation/diagnostics, but
    the online loop is only ALLOWED to use a test sample's a_t once that
    sample becomes eligible (see simulate())."""
    suffix = '' if split == 'test' else f'_{split}'
    bp = os.path.join(cache_dir, f'{dataset}_truebase{suffix}.npz')
    rp = os.path.join(cache_dir, f'{dataset}_rag{suffix}.npz')
    fp = os.path.join(cache_dir, f'{dataset}_rag_probe_features{suffix}.npz')
    for p in (bp, rp, fp):
        if not os.path.exists(p):
            print(f'[{dataset}/{split}] missing cache: {p}')
            return None
    b, r, f = np.load(bp), np.load(rp), np.load(fp)
    pred_b, true_b = _to_2d(b['preds']), _to_2d(b['trues'])
    pred_r, true_r = _to_2d(r['preds']), _to_2d(r['trues'])
    n = min(pred_b.shape[0], pred_r.shape[0], f[STATE_FEATURE_KEYS[0]].shape[0])
    pred_b, true, pred_r = pred_b[:n], true_b[:n], pred_r[:n]

    H = pred_b.shape[1]
    d = pred_r - pred_b
    q = (d ** 2).mean(axis=1)
    safe_q = np.sqrt(np.clip(q, EPS, None))
    u = d / safe_q[:, None]
    e = true - pred_b
    a = (e * u).mean(axis=1)

    divergence = np.abs(d).mean(axis=1)
    c = np.stack([_feat(f, k, n) for k in STATE_FEATURE_KEYS] + [divergence], axis=1)
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)

    return {'n': n, 'pred_b': pred_b, 'pred_r': pred_r, 'true': true,
            'd': d, 'q': q, 'u': u, 'a': a, 'c': c}


class StateScaler:
    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std < EPS] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean) / self.std


class DirectionPCA:
    """Frozen PCA fit once on the M0 pool's direction vectors u_i, per
    doc 4.2 ("训练 OOF 方向上拟合 PCA；固定后不再更新"). Implemented with
    plain numpy SVD to avoid adding a new dependency."""
    def fit(self, U, p_u):
        self.mean = U.mean(axis=0)
        Uc = U - self.mean
        # economy SVD
        _, _, Vt = np.linalg.svd(Uc, full_matrices=False)
        self.components = Vt[:p_u]  # (p_u, H)
        return self

    def transform(self, U):
        return (U - self.mean) @ self.components.T  # (N, p_u)


def build_joint_features(c_raw, u_raw, q_raw, state_scaler, dir_pca, log_q_scaler):
    c_std = state_scaler.transform(c_raw)
    u_proj = dir_pca.transform(u_raw)
    log_q = np.log(q_raw + EPS)
    log_q_std = (log_q - log_q_scaler[0]) / log_q_scaler[1]
    return c_std, u_proj, log_q_std


def simulate(dataset, cache_dir, K=16, p_u=16, eta_u=1.0, eta_q=1.0, tau_mult=2.0,
             n_min=8, eps_q=1e-6, eps_w=1e-8, eps_h=1e-6, seed=0, max_T=None):
    val = load_split_records(cache_dir, dataset, 'val')
    test = load_split_records(cache_dir, dataset, 'test')
    if val is None or test is None:
        return None

    if max_T is not None and test['n'] > max_T:
        # fast first-pass option: truncate the online stream so a debug run on
        # a large dataset (e.g. weather with many channels flattened into
        # samples) finishes quickly. This only shortens T; it does not change
        # M0 or any of the mechanics being validated.
        for k in ('pred_b', 'pred_r', 'true', 'd', 'u', 'c'):
            test[k] = test[k][:max_T]
        for k in ('q', 'a'):
            test[k] = test[k][:max_T]
        test['n'] = max_T

    H = test['pred_b'].shape[1]
    D = H  # feedback delay in sample-index units, since window stride S=1 (see writeup): D = ceil(H/S) = H
    tau = tau_mult * D

    # ---- fit scalers / PCA on M0 (validation split) ONLY ----
    state_scaler = StateScaler().fit(val['c'])
    dir_pca = DirectionPCA().fit(val['u'], p_u)
    log_q_val = np.log(val['q'] + EPS)
    log_q_scaler = (log_q_val.mean(), log_q_val.std() if log_q_val.std() > EPS else 1.0)

    val_c_std, val_u_proj, val_logq_std = build_joint_features(
        val['c'], val['u'], val['q'], state_scaler, dir_pca, log_q_scaler)

    T = test['n']
    n_val = len(val['a'])
    capacity = n_val + T  # pool can never grow past M0 + every test sample revealed once

    # Preallocate the pool once and fill by index, instead of np.vstack/concatenate
    # on every reveal event -- vstack-in-a-loop is O(T^2) memory copies and gets
    # slow fast once T is in the thousands (e.g. a dataset with many channels
    # flattened into samples), so this matters for real runs, not just synthetic ones.
    pool_c = np.zeros((capacity, val_c_std.shape[1]))
    pool_u = np.zeros((capacity, val_u_proj.shape[1]))
    pool_logq = np.zeros(capacity)
    pool_a = np.zeros(capacity)
    pool_is_online = np.zeros(capacity, dtype=bool)
    pool_reveal_t = np.full(capacity, -1)

    pool_c[:n_val] = val_c_std
    pool_u[:n_val] = val_u_proj
    pool_logq[:n_val] = val_logq_std
    pool_a[:n_val] = val['a']
    pool_len = n_val  # current number of valid rows in the pool

    test_c_std, test_u_proj, test_logq_std = build_joint_features(
        test['c'], test['u'], test['q'], state_scaler, dir_pca, log_q_scaler)

    rng = np.random.default_rng(seed)

    alpha_hist = np.zeros(T)
    yhat = test['pred_b'].copy()
    fallback_reason = np.array(['none'] * T, dtype=object)

    # diagnostics
    diag_n_eff = np.full(T, np.nan)
    diag_n_pool_total = np.zeros(T, dtype=int)          # (d) how much history is even available
    diag_n_online_revealed = np.zeros(T, dtype=int)      # online-only portion of the pool
    diag_n_knn_found = np.zeros(T, dtype=int)            # (b) raw neighbors found (<=K, limited by pool size)
    diag_is_cold_start = np.zeros(T, dtype=bool)         # (c) no online-revealed record yet at all

    pending_reveal_at = np.arange(T) + D  # test sample i becomes eligible at test-time-index i+D

    for t in range(T):
        # ---- reveal step: fold in any test samples whose future is now known ----
        newly_revealed = np.where(pending_reveal_at == t)[0]
        for i in newly_revealed:
            pool_c[pool_len] = test_c_std[i]
            pool_u[pool_len] = test_u_proj[i]
            pool_logq[pool_len] = test_logq_std[i]
            pool_a[pool_len] = test['a'][i]
            pool_is_online[pool_len] = True
            pool_reveal_t[pool_len] = t
            pool_len += 1

        diag_n_pool_total[t] = pool_len
        diag_n_online_revealed[t] = int(pool_is_online[:pool_len].sum())
        diag_is_cold_start[t] = diag_n_online_revealed[t] == 0

        q_t = test['q'][t]
        if q_t < eps_q:
            alpha_hist[t] = 0.0
            fallback_reason[t] = 'q_too_small'
            continue

        # ---- joint distance to every pool record (only the filled prefix) ----
        dc = pool_c[:pool_len] - test_c_std[t]
        du = pool_u[:pool_len] - test_u_proj[t]
        dlq = pool_logq[:pool_len] - test_logq_std[t]
        rho2 = (dc ** 2).sum(axis=1) + eta_u * (du ** 2).sum(axis=1) + eta_q * (dlq ** 2)

        n_avail = len(rho2)
        k_use = min(K, n_avail)
        diag_n_knn_found[t] = k_use
        if k_use == 0:
            alpha_hist[t] = 0.0
            fallback_reason[t] = 'no_neighbors'
            continue

        nn_idx = np.argpartition(rho2, k_use - 1)[:k_use]
        nn_rho2 = rho2[nn_idx]
        h_t = np.sqrt(np.partition(nn_rho2, -1)[-1]) + eps_h  # adaptive bandwidth = Kth NN distance
        kernel_w = np.exp(-nn_rho2 / (2 * h_t ** 2))

        # time decay: age measured from "feedback available time" -- M0 records
        # (from validation, i.e. genuinely pre-test) get age = t + D (at least as
        # old as the full test stream so far, never "fresher" than any online
        # record), online records get age = t - reveal_t.
        ages = np.where(pool_is_online[nn_idx], t - pool_reveal_t[nn_idx], t + D)
        time_w = np.exp(-ages / tau)

        w = kernel_w * time_w
        w_sum = w.sum()
        n_eff = (w_sum ** 2) / ((w ** 2).sum() + eps_w)
        diag_n_eff[t] = n_eff

        if n_eff < n_min:
            alpha_hist[t] = 0.0
            fallback_reason[t] = 'n_eff_too_low'
            continue

        m_hat = (w * pool_a[nn_idx]).sum() / (w_sum + eps_w)
        alpha_t = float(np.clip(m_hat / (np.sqrt(q_t) + eps_q), 0.0, 1.0))
        alpha_hist[t] = alpha_t
        if alpha_t <= 1e-6:
            fallback_reason[t] = 'alpha_clipped_to_0'

    yhat = test['pred_b'] + alpha_hist[:, None] * test['d']
    l_dracs = ((test['true'] - yhat) ** 2).mean(axis=1)
    l_base = ((test['true'] - test['pred_b']) ** 2).mean(axis=1)
    l_rag = ((test['true'] - test['pred_r']) ** 2).mean(axis=1)
    alpha_star = np.clip(test['a'] / np.sqrt(np.clip(test['q'], EPS, None)), 0, 1)
    l_line_oracle = ((test['true'] - (test['pred_b'] + alpha_star[:, None] * test['d'])) ** 2).mean(axis=1)

    L_base, L_rag = l_base.mean(), l_rag.mean()
    L_fixed = min(L_base, L_rag)
    L_dracs = l_dracs.mean()
    L_line = l_line_oracle.mean()
    eta_line = (L_fixed - L_dracs) / (L_fixed - L_line) * 100 if (L_fixed - L_line) > 1e-12 else float('nan')
    gain_best = (L_fixed - L_dracs) / L_fixed * 100

    cold_start_steps = int(diag_is_cold_start.sum())

    return {
        'dataset': dataset, 'T': T, 'D': D, 'tau': tau,
        'L_base': L_base, 'L_rag': L_rag, 'L_fixed': L_fixed,
        'L_dracs': L_dracs, 'L_line_oracle': L_line,
        'gain_best_pct': gain_best, 'eta_line_pct': eta_line,
        'cold_start_steps': cold_start_steps, 'cold_start_frac': cold_start_steps / T,
        'mean_alpha': alpha_hist.mean(), 'frac_alpha_gt0': (alpha_hist > 1e-6).mean(),
        'fallback_counts': {k: int((fallback_reason == k).sum()) for k in
                             ['q_too_small', 'no_neighbors', 'n_eff_too_low', 'alpha_clipped_to_0', 'none']},
        'diag_n_eff': diag_n_eff, 'diag_n_pool_total': diag_n_pool_total,
        'diag_n_online_revealed': diag_n_online_revealed, 'diag_n_knn_found': diag_n_knn_found,
        'diag_is_cold_start': diag_is_cold_start,
        'K': K, 'n_min': n_min,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--cache_dir', default='results/oracle_cache')
    ap.add_argument('--K', type=int, default=16)
    ap.add_argument('--p_u', type=int, default=16)
    ap.add_argument('--eta_u', type=float, default=1.0)
    ap.add_argument('--eta_q', type=float, default=1.0)
    ap.add_argument('--tau_mult', type=float, default=2.0, help='tau = tau_mult * D')
    ap.add_argument('--n_min', type=int, default=8)
    ap.add_argument('--max_T', type=int, default=None,
                     help='truncate the online test stream to this many samples for a fast first pass '
                          '(e.g. on a dataset with many channels flattened into samples); omit for the full run')
    args = ap.parse_args()

    r = simulate(args.dataset, args.cache_dir, K=args.K, p_u=args.p_u, eta_u=args.eta_u,
                 eta_q=args.eta_q, tau_mult=args.tau_mult, n_min=args.n_min, max_T=args.max_T)
    if r is None:
        return

    print(f"Dataset: {r['dataset']}   T(test samples)={r['T']}   D(feedback delay)={r['D']}   tau={r['tau']:.1f}")
    print(f"L_base={r['L_base']:.5f}  L_rag={r['L_rag']:.5f}  L_fixed={r['L_fixed']:.5f}  "
          f"L_line_oracle={r['L_line_oracle']:.5f}  L_dracs={r['L_dracs']:.5f}")
    print(f"Gain vs best fixed endpoint: {r['gain_best_pct']:.2f}%   "
          f"eta_line (Line-Oracle utilization): {r['eta_line_pct']:.2f}%")
    print()
    print("---- 4 项监控诊断 ----")
    print(f"(c) 冷启动步数: {r['cold_start_steps']} / {r['T']} = {r['cold_start_frac']*100:.2f}% "
          f"of test stream had ZERO online-revealed feedback yet (D={r['D']} steps of unavoidable blind spot "
          f"repeats at every single test step, on top of this one-time startup cold period)")
    print(f"(a) n_eff 分布 (时间衰减+联合核加权后的有效样本量): "
          f"mean={np.nanmean(r['diag_n_eff']):.2f}, median={np.nanmedian(r['diag_n_eff']):.2f}, "
          f"p10={np.nanpercentile(r['diag_n_eff'],10):.2f}, "
          f"frac below n_min({r['n_min']})={(r['diag_n_eff'] < r['n_min']).mean()*100:.1f}%")
    print(f"(b) 联合(状态+方向)匹配后实际找到的近邻数: mean={r['diag_n_knn_found'].mean():.2f} "
          f"(requested K={r['K']}), frac found < K: {(r['diag_n_knn_found'] < r['K']).mean()*100:.1f}%")
    print(f"(d) 历史反馈池子总大小随时间: 起始(含M0)={r['diag_n_pool_total'][0]}, "
          f"结束={r['diag_n_pool_total'][-1]}, 在线揭晓部分占比(结束时)="
          f"{r['diag_n_online_revealed'][-1]/max(r['diag_n_pool_total'][-1],1)*100:.1f}%")
    print()
    print(f"平均 alpha={r['mean_alpha']:.3f}   alpha>0 比例={r['frac_alpha_gt0']*100:.1f}%")
    print(f"回退原因计数: {r['fallback_counts']}")


if __name__ == '__main__':
    main()
