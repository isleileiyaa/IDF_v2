"""
新方案（RIDDE_新版目标函数与最终实验方案）第6.3节要求的时间稳健性指
标，在不需要任何新训练的情况下，直接用已有的测试集缓存
（results/oracle_cache/{dataset}_{tag}.npz，Probe/D-RACS 阶段就在用的
同一套缓存格式）算出来，先把"怎么衡量"这件事在真实数据上跑通、验证
代码本身没问题。

复用哪些概念（跟方案文档第6.3节符号对应）：
    R_e^method = 某个方法在时间环境 e 里的平均误差
    R_e^0      = Query-only 参考模型 B 在时间环境 e 里的平均误差
                 (B 就是已有的 truebase 缓存，不需要重新训练)
    NTR_block  = 有多少比例的时间环境，method 比 B 差
    WDR        = 最惨的那个时间环境，相对退化多少
    TRR_test   = 用 CVaR 挑出最差的一部分环境，算这些环境上"检索相对
                 参考模型变差了多少"的平均严重程度（只算变差的方向，
                 变好的环境记 0，跟文档4.3节 r_e = [Delta_e]_+ 一致）
    StdRisk    = method 的误差在各个时间环境之间波动多大

环境切分复用 env_blocks.py（channel-major 对齐，见该文件顶部说明）。

用法（真实数据，复用之前 Probe/D-RACS 阶段缓存下来的 truebase/rag npz，
默认拿 idf_clean_dis 的 RAG 缓存当 method，可以用 --method_tag 换成别
的方法比如以后新训出来的 idf_ridde_v2 / 新方案本身）：
    python ridde_trr/eval_temporal_robustness.py \\
        --datasets ETTh1 ETTh2 ETTm1 ETTm2 weather exchange_rate electricity \\
        --method_tag rag --ref_tag truebase --num_blocks 10 --alpha 0.7

先自检（不需要任何真实缓存，纯合成数据验证代码本身对不对）：
    python ridde_trr/eval_temporal_robustness.py --selftest
"""
import os
import argparse
import numpy as np

from env_blocks import assign_env_ids, KNOWN_ENC_IN

EPS = 1e-8


def _to_2d(arr):
    arr = np.asarray(arr)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def load_pair(cache_dir, dataset, tag, split='test'):
    suffix = '' if split == 'test' else f'_{split}'
    path = os.path.join(cache_dir, f'{dataset}_{tag}{suffix}.npz')
    if not os.path.exists(path):
        return None
    d = np.load(path)
    preds, trues = _to_2d(d['preds']), _to_2d(d['trues'])
    n = min(preds.shape[0], trues.shape[0])
    return preds[:n], trues[:n]


def per_sample_mse(preds, trues):
    return ((preds - trues) ** 2).mean(axis=1)


def env_risks(loss, env_id, num_blocks):
    """按环境聚合成 R_e（长度 num_blocks 的数组，某环境没有样本则为 NaN）。"""
    R = np.full(num_blocks, np.nan)
    for e in range(num_blocks):
        mask = env_id == e
        if mask.sum() > 0:
            R[e] = loss[mask].mean()
    return R


def cvar_of_relative_risk(R_method, R_ref, alpha, eps=EPS):
    """文档4.3/6.3节: Delta_e = log((R_method+eps)/(R_ref+eps))，
    r_e = [Delta_e]_+，TRR = CVaR_alpha(r_e)，即"最差的 (1-alpha) 比例
    环境"上 r_e 的均值。这里用经验分位数直接实现（环境数不多，排序取
    尾部即可；训练时用的 softplus(nu) 那套参数化是为了能对 nu 求梯度联
    合优化，评估时不需要，直接算分位数更直接、也更不容易引入新 bug）。
    """
    valid = ~np.isnan(R_method) & ~np.isnan(R_ref)
    R_method, R_ref = R_method[valid], R_ref[valid]
    delta = np.log((R_method + eps) / (R_ref + eps))
    r = np.clip(delta, 0.0, None)
    E = len(r)
    if E == 0:
        return float('nan')
    k = max(1, int(np.ceil((1 - alpha) * E)))
    worst = np.sort(r)[-k:]
    return worst.mean()


def analyze_one(dataset, cache_dir, method_tag, ref_tag, num_blocks, alpha, enc_in_override=None):
    m = load_pair(cache_dir, dataset, method_tag)
    r = load_pair(cache_dir, dataset, ref_tag)
    if m is None or r is None:
        print(f'[{dataset}] missing {method_tag} or {ref_tag} cache, skipping')
        return None
    preds_m, trues_m = m
    preds_r, trues_r = r
    n = min(len(preds_m), len(preds_r))
    preds_m, trues_m = preds_m[:n], trues_m[:n]
    preds_r, trues_r = preds_r[:n], trues_r[:n]

    enc_in = enc_in_override or KNOWN_ENC_IN.get(dataset)
    if enc_in is None:
        print(f'[{dataset}] 不认识这个数据集的 enc_in，用 --enc_in_override 手动指定')
        return None

    env_id, tot_len, _ = assign_env_ids(n, enc_in, num_blocks)

    loss_m = per_sample_mse(preds_m, trues_m)
    loss_r = per_sample_mse(preds_r, trues_r)

    R_m = env_risks(loss_m, env_id, num_blocks)
    R_r = env_risks(loss_r, env_id, num_blocks)

    valid = ~np.isnan(R_m) & ~np.isnan(R_r)
    ntr = (R_m[valid] > R_r[valid]).mean()
    wdr = ((R_m[valid] - R_r[valid]) / (R_r[valid] + EPS)).max()
    trr = cvar_of_relative_risk(R_m, R_r, alpha)
    std_risk = np.nanstd(R_m)

    return {
        'dataset': dataset, 'n': n, 'tot_len': tot_len, 'enc_in': enc_in,
        'overall_method_mse': loss_m.mean(), 'overall_ref_mse': loss_r.mean(),
        'NTR_block': ntr, 'WDR': wdr, 'TRR_test': trr, 'StdRisk': std_risk,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+',
                     default=['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'weather', 'exchange_rate', 'electricity'])
    ap.add_argument('--cache_dir', default='results/oracle_cache')
    ap.add_argument('--method_tag', default='rag')
    ap.add_argument('--ref_tag', default='truebase')
    ap.add_argument('--num_blocks', type=int, default=10)
    ap.add_argument('--alpha', type=float, default=0.7)
    ap.add_argument('--enc_in_override', type=int, default=None)
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    rows = []
    for ds in args.datasets:
        row = analyze_one(ds, args.cache_dir, args.method_tag, args.ref_tag,
                           args.num_blocks, args.alpha, args.enc_in_override)
        if row is not None:
            rows.append(row)

    if not rows:
        print('No datasets with both caches found.')
        return

    header = (f"{'Dataset':<14}{'n':>8}{'ov_method':>11}{'ov_ref':>9}"
              f"{'NTR%':>8}{'WDR%':>9}{'TRR_test':>10}{'StdRisk':>9}")
    print(header)
    print('-' * len(header))
    for row in rows:
        print(f"{row['dataset']:<14}{row['n']:>8}{row['overall_method_mse']:>11.5f}"
              f"{row['overall_ref_mse']:>9.5f}{row['NTR_block']*100:>8.1f}"
              f"{row['WDR']*100:>9.1f}{row['TRR_test']:>10.4f}{row['StdRisk']:>9.5f}")

    print()
    print('检查项: ov_method / ov_ref 应该跟之前 stage_a_oracle.py 报告的 '
          'L_rag / L_base 数字一致（同一份缓存，这里只是额外按时间切了环境）—— '
          '如果对不上，说明 enc_in 或者缓存文件选错了。')


def _selftest():
    """纯合成数据自检，不需要任何真实缓存文件。"""
    import tempfile

    rng = np.random.default_rng(0)
    enc_in, tot_len, num_blocks, H = 4, 200, 8, 16
    N = enc_in * tot_len

    env_id, got_tot_len, boundaries = assign_env_ids(N, enc_in, num_blocks)
    assert got_tot_len == tot_len

    trues = rng.normal(size=(N, H))
    noise_ref = rng.normal(scale=0.10, size=(N, H))
    noise_method = rng.normal(scale=0.05, size=(N, H))  # 平时 method 更准
    bad_env = num_blocks - 1
    is_bad = env_id == bad_env
    noise_method[is_bad] = rng.normal(scale=2.0, size=(is_bad.sum(), H))  # 最后一段狠狠变差

    preds_ref = trues + noise_ref
    preds_method = trues + noise_method

    with tempfile.TemporaryDirectory() as tmpdir:
        ds = 'SYNTH'
        np.savez(os.path.join(tmpdir, f'{ds}_truebase.npz'), preds=preds_ref, trues=trues)
        np.savez(os.path.join(tmpdir, f'{ds}_rag.npz'), preds=preds_method, trues=trues)

        row = analyze_one(ds, tmpdir, 'rag', 'truebase', num_blocks, alpha=0.7,
                           enc_in_override=enc_in)
        assert row is not None

        loss_m = per_sample_mse(preds_method, trues)
        loss_r = per_sample_mse(preds_ref, trues)
        R_m = env_risks(loss_m, env_id, num_blocks)
        R_r = env_risks(loss_r, env_id, num_blocks)

        worst_env = int(np.nanargmax((R_m - R_r) / (R_r + EPS)))
        assert worst_env == bad_env, f'WDR 应该抓出 env={bad_env}，实际抓到 env={worst_env}'

        ntr_expected = 1.0 / num_blocks
        assert abs(row['NTR_block'] - ntr_expected) < 1e-9, (
            f"NTR_block={row['NTR_block']}，期望恰好 1/{num_blocks}={ntr_expected}"
        )

        wdr_expected = (R_m[bad_env] - R_r[bad_env]) / (R_r[bad_env] + EPS)
        assert abs(row['WDR'] - wdr_expected) < 1e-6

        delta = np.log((R_m + EPS) / (R_r + EPS))
        r = np.clip(delta, 0.0, None)
        k = int(np.ceil((1 - 0.7) * num_blocks))
        assert k == 3
        worst_k = np.sort(r)[-k:]
        assert abs(row['TRR_test'] - worst_k.mean()) < 1e-9

        print('[eval_temporal_robustness] selftest PASSED')
        print(f"  worst_env(手算)={bad_env}, WDR={row['WDR']*100:.1f}%, "
              f"NTR_block={row['NTR_block']*100:.1f}%, TRR_test={row['TRR_test']:.4f}, "
              f"StdRisk={row['StdRisk']:.5f}")


if __name__ == '__main__':
    main()
