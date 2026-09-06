"""
RIDDE_NoRetrieval_Base_重新验证实验方案 -- section 6, the Probe stage.

Go was reached (analyze_sample_oracle.py): a real, sample-level Base/RAG
Oracle gap exists. That says nothing about whether the gap is PREDICTABLE
from information available at inference time (before the true future is
known) -- Oracle cheats by looking at the true future to pick the better
arm; a real Gate can't.

ROUND 1 of this script (logistic regression, 8 features: q_std/q_slope/
q_roughness/dist_mean/dist_min/dist_std/neighbor_future_dispersion/
divergence) FAILED cleanly across all 7 core datasets: macro eta=-35.4%,
0/7 beat Lfixed, 0/7 cleared the 25% bar, and every dataset showed the same
signature (val_acc 0.615-0.674 vs test_acc 0.489-0.554, i.e. near chance) --
a classic overfit-to-validation pattern.

ROUND 2 (this version) fixes two things at once:
  1. Adds features that are still 100% inference-time-safe (no peeking at
     the true future) but carry more information than round 1's basic
     stats: query-only q_cv/q_acf1/q_trend_r2/q_last_dev, retrieval-aware
     dist_top1_gap_norm/neighbor_past_corr, and a new post-prediction
     signal qwidth_ratio/qwidth_diff (each branch's OWN quantile-interval
     width, i.e. the model's own uncertainty about its own prediction --
     requires the updated utils/tools.py that saves 'qwidth' into every
     arm's preds/trues cache, not just the rag probe-features file).
  2. Replaces "fit once, score once" with model/hyperparameter selection
     via k-fold cross-validation done ENTIRELY inside the validation split
     (several logistic-regression regularization strengths + several
     HistGradientBoostingClassifier configs). This directly targets round
     1's overfitting signature: a single .fit()+.score() can't detect
     overfitting, k-fold CV can, because each fold's held-out part was
     never trained on. The test split is still touched exactly once, with
     the single config that won the internal CV -- so trying a stronger
     model does not turn into "pick whichever model happens to look best
     on the test set" (that would just be leakage by another name).

Feature groups (matching the doc's table, round 1 + round 2 additions):
  - Query-only:        q_std, q_slope, q_roughness,
                        q_cv, q_acf1, q_trend_r2, q_last_dev   (from batch_x alone)
  - Retrieval-aware:    dist_mean, dist_min, dist_std, dist_top1_gap_norm,
                        neighbor_future_dispersion, neighbor_past_corr  (from retrieved_seqs/distances)
  - Post-prediction:    divergence = mean|pred_rag - pred_base|,
                        qwidth_ratio, qwidth_diff (from each branch's own
                        quantile-interval width -- model's own uncertainty,
                        no future peeking)

Requires, per dataset, in --cache_dir (regenerate via the updated
utils/tools.py + script/zeroshot_chronos_truebase.sh /
script/zeroshot_chronos_idf_clean_dis.sh with EVAL_SPLIT=val and
EVAL_SPLIT=test, so every cache below carries the new 'qwidth' key):
  {dataset}_truebase_val.npz, {dataset}_rag_val.npz, {dataset}_rag_probe_features_val.npz
  {dataset}_truebase.npz,     {dataset}_rag.npz,     {dataset}_rag_probe_features.npz

Metric (doc's own, NOT plain classification accuracy):
    eta = (Lfixed - Lprobe) / (Lfixed - Loracle)
where Lprobe is the mean loss actually achieved by ROUTING with the probe's own
predicted label (not the oracle's cheating label). Doc's bar: eta >= 25-30%,
and Lprobe < Lfixed (beats blindly picking the single better fixed arm) on most
datasets, before investing in a full end-to-end Gate.

Usage:
    python build_probe.py --datasets ETTh1 ETTh2 ETTm1 ETTm2 weather exchange_rate electricity
"""
import os
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

EPS = 1e-8
FEATURE_NAMES = [
    'q_std', 'q_slope', 'q_roughness', 'q_cv', 'q_acf1', 'q_trend_r2', 'q_last_dev',
    'dist_mean', 'dist_min', 'dist_std', 'dist_top1_gap_norm',
    'neighbor_future_dispersion', 'neighbor_past_corr',
    'divergence', 'qwidth_ratio', 'qwidth_diff',
]

CANDIDATE_MODELS = [
    ('logreg_C0.01', lambda: LogisticRegression(max_iter=1000, C=0.01)),
    ('logreg_C0.1', lambda: LogisticRegression(max_iter=1000, C=0.1)),
    ('logreg_C1', lambda: LogisticRegression(max_iter=1000, C=1.0)),
    ('logreg_C10', lambda: LogisticRegression(max_iter=1000, C=10.0)),
    ('hgb_d2_lr0.05', lambda: HistGradientBoostingClassifier(max_depth=2, learning_rate=0.05, max_iter=100)),
    ('hgb_d3_lr0.05', lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=100)),
    ('hgb_d2_lr0.1', lambda: HistGradientBoostingClassifier(max_depth=2, learning_rate=0.1, max_iter=100)),
]


def _to_2d(arr, name):
    arr = np.asarray(arr)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f'{name}: expected 2D (N, L) after squeezing, got {arr.shape}')
    return arr


def per_sample_mse(pred, true):
    return ((pred - true) ** 2).mean(axis=1)


def _balanced_sample_weight(y):
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    w_per_class = {c: len(y) / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    return np.array([w_per_class[v] for v in y])


def _feat(npz, key, n, warned):
    """Fetch a feature column with a graceful fallback for caches that
    predate this round's new fields (so a partially-rerun cache_dir fails
    loud-but-survivable per dataset instead of crashing the whole run)."""
    if key in npz.files:
        return npz[key][:n]
    if key not in warned:
        print(f'  (note: "{key}" not in cache -- using 0 for it; rerun the eval scripts to get the real values)')
        warned.add(key)
    return np.zeros(n)


def load_split(cache_dir, dataset, split):
    suffix = '' if split == 'test' else f'_{split}'
    paths = {
        'base': os.path.join(cache_dir, f'{dataset}_truebase{suffix}.npz'),
        'rag': os.path.join(cache_dir, f'{dataset}_rag{suffix}.npz'),
        'feat': os.path.join(cache_dir, f'{dataset}_rag_probe_features{suffix}.npz'),
    }
    for tag, p in paths.items():
        if not os.path.exists(p):
            print(f'[{dataset}/{split}] missing {tag} cache at {p}')
            return None

    b = np.load(paths['base'])
    r = np.load(paths['rag'])
    f = np.load(paths['feat'])
    warned = set()

    pred_b = _to_2d(b['preds'], f'{dataset}/{split} base preds')
    true_b = _to_2d(b['trues'], f'{dataset}/{split} base trues')
    pred_r = _to_2d(r['preds'], f'{dataset}/{split} rag preds')
    true_r = _to_2d(r['trues'], f'{dataset}/{split} rag trues')

    n = min(pred_b.shape[0], pred_r.shape[0], f['q_std'].shape[0])
    pred_b, true_b = pred_b[:n], true_b[:n]
    pred_r, true_r = pred_r[:n], true_r[:n]

    l_b = per_sample_mse(pred_b, true_b)
    l_r = per_sample_mse(pred_r, true_r)
    divergence = np.abs(pred_r[:n] - pred_b[:n]).mean(axis=1)

    qwidth_b = _feat(b, 'qwidth', n, warned)
    qwidth_r = _feat(r, 'qwidth', n, warned)
    qwidth_ratio = qwidth_r / (np.abs(qwidth_b) + EPS)
    qwidth_diff = qwidth_b - qwidth_r

    X = np.stack([
        f['q_std'][:n], f['q_slope'][:n], f['q_roughness'][:n],
        _feat(f, 'q_cv', n, warned), _feat(f, 'q_acf1', n, warned),
        _feat(f, 'q_trend_r2', n, warned), _feat(f, 'q_last_dev', n, warned),
        f['dist_mean'][:n], f['dist_min'][:n], f['dist_std'][:n],
        _feat(f, 'dist_top1_gap_norm', n, warned),
        f['neighbor_future_dispersion'][:n], _feat(f, 'neighbor_past_corr', n, warned),
        divergence, qwidth_ratio, qwidth_diff,
    ], axis=1)
    assert X.shape[1] == len(FEATURE_NAMES), f'feature count mismatch: {X.shape[1]} vs {len(FEATURE_NAMES)}'
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return {'l_b': l_b, 'l_r': l_r, 'X': X, 'n': n}


def _route_loss(y_pred, l_b, l_r):
    return np.where(y_pred == 1, l_r, l_b).mean()


def _cv_route_loss(build_clf, Xv, y_val, l_b_val, l_r_val, n_splits, seed=0):
    """Mean out-of-fold routing loss for one (model, hyperparameter) config,
    computed ENTIRELY inside the validation split (never touches test).
    Lower is better. This is what actually lets a stronger/more flexible
    model (e.g. gradient boosting) get picked only when it generalizes,
    instead of just being tried once and possibly overfitting worse than
    round 1's plain logistic regression did."""
    min_class = min(np.bincount(y_val))
    k = max(2, min(n_splits, min_class))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    losses = []
    for train_idx, holdout_idx in skf.split(Xv, y_val):
        scaler = StandardScaler().fit(Xv[train_idx])
        Xtr = scaler.transform(Xv[train_idx])
        Xho = scaler.transform(Xv[holdout_idx])
        clf = build_clf()
        clf.fit(Xtr, y_val[train_idx], sample_weight=_balanced_sample_weight(y_val[train_idx]))
        y_pred = clf.predict(Xho)
        losses.append(_route_loss(y_pred, l_b_val[holdout_idx], l_r_val[holdout_idx]))
    return float(np.mean(losses))


def run_one(dataset, cache_dir, n_splits=5, seed=0):
    val = load_split(cache_dir, dataset, 'val')
    test = load_split(cache_dir, dataset, 'test')
    if val is None or test is None:
        return None

    delta_val = val['l_b'] - val['l_r']       # >0 => RAG better
    y_val = (delta_val > 0).astype(int)
    if y_val.min() == y_val.max():
        print(f'[{dataset}] val labels are all one class ({y_val[0]}) -- probe would be trivial, skipping')
        return None

    l_b_val, l_r_val = val['l_b'], val['l_r']

    # ---- model/hyperparameter selection: CV inside the val split ONLY ----
    cv_results = []
    best_name, best_build, best_cv_loss = None, None, float('inf')
    for name, build in CANDIDATE_MODELS:
        cv_loss = _cv_route_loss(build, val['X'], y_val, l_b_val, l_r_val, n_splits=n_splits, seed=seed)
        cv_results.append((name, cv_loss))
        if cv_loss < best_cv_loss:
            best_name, best_build, best_cv_loss = name, build, cv_loss

    # ---- refit the winning config on the FULL val split, touch test ONCE ----
    scaler = StandardScaler().fit(val['X'])
    Xv = scaler.transform(val['X'])
    Xt = scaler.transform(test['X'])

    clf = best_build()
    clf.fit(Xv, y_val, sample_weight=_balanced_sample_weight(y_val))
    y_pred_test = clf.predict(Xt)

    l_b_t, l_r_t = test['l_b'], test['l_r']
    L_base = l_b_t.mean()
    L_rag = l_r_t.mean()
    L_fixed = min(L_base, L_rag)
    L_oracle = np.minimum(l_b_t, l_r_t).mean()
    L_probe = np.where(y_pred_test == 1, l_r_t, l_b_t).mean()

    denom = L_fixed - L_oracle
    eta = (L_fixed - L_probe) / denom * 100 if denom > 1e-12 else float('nan')
    beats_fixed = L_probe < L_fixed

    val_acc = (clf.predict(Xv) == y_val).mean()
    y_test_true = (test['l_b'] - test['l_r'] > 0).astype(int)
    test_acc = (y_pred_test == y_test_true).mean()

    coef = dict(zip(FEATURE_NAMES, clf.coef_[0])) if hasattr(clf, 'coef_') else {}

    return {
        'dataset': dataset, 'n_val': val['n'], 'n_test': test['n'],
        'model': best_name, 'cv_route_loss': best_cv_loss,
        'L_base': L_base, 'L_rag': L_rag, 'L_fixed': L_fixed, 'L_oracle': L_oracle,
        'L_probe': L_probe, 'eta': eta, 'beats_fixed': beats_fixed,
        'val_acc': val_acc, 'test_acc': test_acc,
        'coef': coef, 'cv_results': cv_results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+',
                     default=['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'weather', 'exchange_rate', 'electricity'])
    ap.add_argument('--cache_dir', default='results/oracle_cache')
    ap.add_argument('--cv_splits', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    rows = [r for r in (run_one(ds, args.cache_dir, n_splits=args.cv_splits, seed=args.seed) for ds in args.datasets)
            if r is not None]
    if not rows:
        print('No datasets with complete val+test caches found -- nothing to report.')
        return

    header = (f"{'Dataset':<14}{'Model':<16}{'Lfixed':>10}{'Loracle':>10}{'Lprobe':>10}"
              f"{'eta%':>8}{'>fixed?':>9}{'val_acc':>9}{'test_acc':>9}")
    print(header)
    print('-' * len(header))
    for row in rows:
        print(f"{row['dataset']:<14}{row['model']:<16}{row['L_fixed']:>10.5f}{row['L_oracle']:>10.5f}"
              f"{row['L_probe']:>10.5f}{row['eta']:>8.1f}{str(row['beats_fixed']):>9}"
              f"{row['val_acc']:>9.3f}{row['test_acc']:>9.3f}")

    etas = [row['eta'] for row in rows]
    n_beats_fixed = sum(1 for row in rows if row['beats_fixed'])
    print('-' * len(header))
    print(f"Macro eta: {np.mean(etas):.1f}%  |  datasets beating Lfixed: {n_beats_fixed}/{len(rows)}  |  "
          f"datasets with eta >= 25%: {sum(1 for e in etas if e >= 25)}/{len(rows)}")
    print()
    print("Doc's bar: eta >= 25-30% AND Lprobe < Lfixed on most datasets, before investing in a full Gate.")
    if np.mean(etas) >= 25 and n_beats_fixed >= max(1, int(0.5 * len(rows))):
        print('==> Probe verdict: feasible, worth building a full Gate')
    elif np.mean(etas) >= 10:
        print('==> Probe verdict: weak/partial signal -- more feature engineering before a full Gate')
    else:
        print('==> Probe verdict: no usable signal in these features -- Oracle gap looks unpredictable from them')

    print()
    print('Per-dataset model selected (via CV on val split only) and its internal CV routing loss, '
          'plus every candidate tried (for transparency about what was searched):')
    for row in rows:
        cvs = ', '.join(f'{name}={loss:.5f}' for name, loss in row['cv_results'])
        print(f"  {row['dataset']:<12}selected={row['model']} (cv_loss={row['cv_route_loss']:.5f})")
        print(f"    all candidates: {cvs}")

    print()
    print('Per-dataset logistic regression coefficients where the selected model is linear '
          '(standardized features; sign/magnitude indicate which cues push toward "use RAG"; '
          'blank for datasets where a HistGradientBoostingClassifier won instead -- it has no '
          'linear coefficients to report):')
    for row in rows:
        if row['coef']:
            coefs = ', '.join(f'{k}={v:+.2f}' for k, v in row['coef'].items())
            print(f"  {row['dataset']:<12}{coefs}")
        else:
            print(f"  {row['dataset']:<12}(non-linear model选中, 无线性系数)")


if __name__ == '__main__':
    main()
