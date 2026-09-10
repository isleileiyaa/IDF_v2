"""
Post-hoc diagnostics for a trained idf_trr_dualpath (RIDDE_新版目标函数与最终
实验方案 Stage 3, Dual+ERM: 双路径结构 + L_sep) ChronosBoltRetrieve checkpoint.

（第一~四部分说明同前几版，此处不再重复；见 git 历史。）

---
新增(第五部分)：第四部分已经证明h_r低方差不是ret_score_head打分头的锅(均匀
权重跟学出来的权重效果几乎一样)，那问题到底是"检索到的候选本身就是同一批
近似重复的东西"，还是"候选之间其实差异不小，只是这种差异跟query是谁没什么
关系，纯粹是噪声，一平均就被抹掉了"——这两种情况都会导致h_r方差小，但含义
完全不同，前者要去修检索/相似度这一层，后者说明只是"平均"这个聚合方式本身
在这批数据上不合适，问题反而更靠后。

用方差分解(跟统计里的组内相关系数ICC是一回事)来分开这两种情况：
    V_within  = 同一个query下，r_M个候选相互之间的方差(候选够不够互相不同)
    V_between = 不同query的候选均值之间的方差(不同query拿到的候选整体上够不够不同)
    ICC = V_between / (V_within + V_between)
ICC接近0：候选间的差异基本是跟query无关的噪声，平均之后当然被抹平——就算候选
本身并不重复(V_within可能还不小)，只要这个差异不跟query挂钩，池化后依然会
导致h_r在样本间趋同。ICC接近1：不同query确实拿到了实质不同的候选，池化前后
的方差不该差太多。

encode_mlp在这个branch里对每个候选各调用一次(for i in range(r_M))，挂一个
hook能拿到每次调用的输入(原始的retrieved_y[:,i,:]，还没编码)和输出
(retrieved_y_enc[:,i,:]，编码之后)，同时对"编码前"和"编码后"两层都做这个
方差分解——如果编码前ICC就已经很低，问题在检索本身/检索库；如果编码前还行、
编码后骤降，问题在encode_mlp这个模块把差异抹掉了。
"""
import os
import sys
import torch
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import utils.tools as tools_mod  # noqa: E402

QUANTILES = tools_mod.quantiles
CENTRAL_IDX = min(range(len(QUANTILES)), key=lambda i: abs(QUANTILES[i] - 0.5))
NUM_Q = len(QUANTILES)

CAPTURED = {}


def capturing_test_retrieve(model, test_data, test_loader, args, device):
    num_batches = int(os.environ.get('NUM_BATCHES', '20'))
    num_vis_samples = int(os.environ.get('NUM_VIS_SAMPLES', '4'))
    pred_len = model.chronos_config.prediction_length

    hooked = {}

    def hook_inv(module, inp, out):
        hooked['z_inv'] = inp[0].detach()
        hooked['y_inv'] = out.detach()

    def hook_dyn(module, inp, out):
        hooked['z_dyn'] = inp[0].detach()
        hooked['y_dyn'] = out.detach()

    def hook_pinv_pre(module, inp):
        hooked['z_inv_in'] = inp[0].detach()

    def hook_pdyn_pre(module, inp):
        hooked['z_dyn_in'] = inp[0].detach()

    def hook_score_pre(module, inp):
        hooked['ret_score_in'] = inp[0].detach()  # (B, r_M, 2*d_model)

    def hook_score_post(module, inp, out):
        hooked['ret_score_raw'] = out.detach()  # (B, r_M, 1)，softmax之前

    encode_mlp_calls = []  # 每个batch forward前清空，累积这次batch里r_M次调用

    def hook_encode_mlp(module, inp, out):
        encode_mlp_calls.append((inp[0].detach(), out.detach()))

    handles = [
        model.f_inv.register_forward_hook(hook_inv),
        model.f_dyn.register_forward_hook(hook_dyn),
        model.P_inv.register_forward_pre_hook(hook_pinv_pre),
        model.P_dyn.register_forward_pre_hook(hook_pdyn_pre),
        model.ret_score_head.register_forward_pre_hook(hook_score_pre),
        model.ret_score_head.register_forward_hook(hook_score_post),
        model.encode_mlp.register_forward_hook(hook_encode_mlp),
    ]

    mse_list, mae_list = [], []
    cos_sim_list, rough_inv_list, rough_dyn_list, energy_share_list = [], [], [], []
    energy_share_sat_frac_list = []
    z_inv_list, z_dyn_list = [], []
    z_inv_in_list, z_dyn_in_list = [], []
    h_r_uniform_list = []
    omega_entropy_list = []
    raw_y_list = []  # 编码前的retrieved_y，(B, r_M, pred_len)按batch攒起来
    enc_y_list = []  # 编码后的retrieved_y_enc，(B, r_M, d_model)按batch攒起来
    vis_saved = False
    y_inv_vis = y_dyn_vis = y_hat_vis = true_vis = None

    def roughness(y):  # y: (B, Q, L) -> (B, Q)
        d2 = y[..., 2:] - 2 * y[..., 1:-1] + y[..., :-2]
        return torch.log(d2.var(dim=-1) + 1e-8)

    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, retrieved_seqs, distances) in enumerate(test_loader):
            if i >= num_batches:
                break
            batch_x = batch_x.float().to(device).squeeze()
            batch_y = batch_y.float().to(device).squeeze()
            retrieved_seqs = retrieved_seqs.float().to(device)
            distances = distances.float().to(device)

            encode_mlp_calls.clear()
            outputs = model(context=batch_x, target=batch_y, retrieved_seq=retrieved_seqs, distances=distances)

            raw_y_b = torch.stack([c[0] for c in encode_mlp_calls], dim=1)  # (B, r_M, pred_len)
            enc_y_b = torch.stack([c[1] for c in encode_mlp_calls], dim=1)  # (B, r_M, d_model)
            raw_y_list.append(raw_y_b.cpu())
            enc_y_list.append(enc_y_b.cpu())

            pred = outputs.quantile_preds.to(batch_x)[:, CENTRAL_IDX]
            true = batch_y
            mse_list.append(((pred - true) ** 2).mean(dim=-1).cpu())
            mae_list.append((pred - true).abs().mean(dim=-1).cpu())

            z_inv = hooked['z_inv']
            z_dyn = hooked['z_dyn']
            B = z_inv.shape[0]
            y_inv = hooked['y_inv'].view(B, NUM_Q, pred_len)
            y_dyn = hooked['y_dyn'].view(B, NUM_Q, pred_len)

            cos_sim_list.append(torch.nn.functional.cosine_similarity(z_inv, z_dyn, dim=-1).abs().cpu())
            z_inv_list.append(z_inv.cpu())
            z_dyn_list.append(z_dyn.cpu())
            z_inv_in_list.append(hooked['z_inv_in'].cpu())
            z_dyn_in_list.append(hooked['z_dyn_in'].cpu())

            ret_score_in = hooked['ret_score_in']  # (B, r_M, 2*d_model)
            d_model_probe = ret_score_in.shape[-1] // 2
            retrieved_y_enc_b = ret_score_in[..., d_model_probe:]  # (B, r_M, d_model)
            omega_b = torch.nn.functional.softmax(hooked['ret_score_raw'], dim=1)  # (B, r_M, 1)
            h_r_uniform_b = retrieved_y_enc_b.mean(dim=1)  # (B, d_model)
            r_M_probe = omega_b.shape[1]
            ent = -(omega_b.clamp_min(1e-12) * omega_b.clamp_min(1e-12).log()).sum(dim=1).squeeze(-1)
            ent_norm = ent / max(np.log(r_M_probe), 1e-8)
            h_r_uniform_list.append(h_r_uniform_b.cpu())
            omega_entropy_list.append(ent_norm.cpu())

            R_inv = roughness(y_inv).mean(dim=1)
            R_dyn = roughness(y_dyn).mean(dim=1)
            rough_inv_list.append(R_inv.cpu())
            rough_dyn_list.append(R_dyn.cpu())

            var_inv = y_inv.var(dim=-1).mean(dim=1)  # (B,)
            var_dyn = y_dyn.var(dim=-1).mean(dim=1)
            share = var_inv / (var_inv + var_dyn + 1e-8)
            energy_share_list.append(share.cpu())
            energy_share_sat_frac_list.append(((share < 0.1) | (share > 0.9)).float().mean().cpu())

            if not vis_saved:
                n = min(num_vis_samples, B)
                y_inv_vis = y_inv[:n, CENTRAL_IDX].cpu().numpy()
                y_dyn_vis = y_dyn[:n, CENTRAL_IDX].cpu().numpy()
                y_hat_vis = pred[:n].cpu().numpy()
                true_vis = true[:n].cpu().numpy()
                vis_saved = True

    for h in handles:
        h.remove()

    z_inv_all = torch.cat(z_inv_list, dim=0)  # (N, d_model)
    z_dyn_all = torch.cat(z_dyn_list, dim=0)  # (N, d_model)
    var_inv_dim = z_inv_all.var(dim=0)
    var_dyn_dim = z_dyn_all.var(dim=0)
    z_inv_c = z_inv_all - z_inv_all.mean(dim=0, keepdim=True)
    z_dyn_c = z_dyn_all - z_dyn_all.mean(dim=0, keepdim=True)
    N = z_inv_all.shape[0]
    xcov = (z_inv_c.t() @ z_dyn_c) / max(N - 1, 1)
    d_inv, d_dyn = z_inv_all.shape[-1], z_dyn_all.shape[-1]
    xcov_loss_eval = (xcov.pow(2).sum() / (d_inv * d_dyn)).item()

    z_inv_in_all = torch.cat(z_inv_in_list, dim=0)  # (N, 4*d_model)
    z_dyn_in_all = torch.cat(z_dyn_in_list, dim=0)  # (N, 2*d_model)
    d_model = model.f_inv.in_features
    e_q_a = z_inv_in_all[:, 0 * d_model:1 * d_model]
    h_r_a = z_inv_in_all[:, 1 * d_model:2 * d_model]
    eqhr_a = z_inv_in_all[:, 2 * d_model:3 * d_model]
    absdiff_a = z_inv_in_all[:, 3 * d_model:4 * d_model]
    e_q_b = z_dyn_in_all[:, 0 * d_model:1 * d_model]
    diff_b = z_dyn_in_all[:, 1 * d_model:2 * d_model]

    e_q_sanity_maxdiff = (e_q_a - e_q_b).abs().max().item()

    var_eq_dim = e_q_a.var(dim=0)
    var_hr_dim = h_r_a.var(dim=0)
    var_eqhr_dim = eqhr_a.var(dim=0)
    var_absdiff_dim = absdiff_a.var(dim=0)
    var_diff_dim = diff_b.var(dim=0)

    cos_eq_hr = torch.nn.functional.cosine_similarity(e_q_a, h_r_a, dim=-1)
    rel_resid = diff_b.norm(dim=-1) / (e_q_a.norm(dim=-1) + 1e-8)

    h_r_uniform_all = torch.cat(h_r_uniform_list, dim=0)  # (N, d_model)
    var_hr_uniform_dim = h_r_uniform_all.var(dim=0)
    omega_entropy_all = torch.cat(omega_entropy_list, dim=0)  # (N,)

    def icc_decompose(x):  # x: (N, r_M, D)
        v_within = x.var(dim=1, unbiased=True).mean()
        v_between = x.mean(dim=1).var(dim=0, unbiased=True).mean()
        icc = v_between / (v_within + v_between + 1e-12)
        return float(v_within), float(v_between), float(icc)

    raw_y_all = torch.cat(raw_y_list, dim=0)  # (N, r_M, pred_len)
    enc_y_all = torch.cat(enc_y_list, dim=0)  # (N, r_M, d_model)
    r_M_total = raw_y_all.shape[1]
    icc_chance_floor = 1.0 / (r_M_total + 1)
    vw_raw, vb_raw, icc_raw = icc_decompose(raw_y_all)
    vw_enc, vb_enc, icc_enc = icc_decompose(enc_y_all)

    CAPTURED['mse'] = torch.cat(mse_list).numpy()
    CAPTURED['mae'] = torch.cat(mae_list).numpy()
    CAPTURED['cos_sim'] = torch.cat(cos_sim_list).numpy()
    CAPTURED['roughness_inv'] = torch.cat(rough_inv_list).numpy()
    CAPTURED['roughness_dyn'] = torch.cat(rough_dyn_list).numpy()
    CAPTURED['energy_share_inv'] = torch.cat(energy_share_list).numpy()
    CAPTURED['energy_share_sat_frac'] = float(torch.stack(energy_share_sat_frac_list).mean())
    CAPTURED['var_inv_dim'] = var_inv_dim.numpy()
    CAPTURED['var_dyn_dim'] = var_dyn_dim.numpy()
    CAPTURED['xcov_loss_eval'] = xcov_loss_eval
    CAPTURED['n_z_samples'] = N
    CAPTURED['e_q_sanity_maxdiff'] = e_q_sanity_maxdiff
    CAPTURED['var_eq_dim'] = var_eq_dim.numpy()
    CAPTURED['var_hr_dim'] = var_hr_dim.numpy()
    CAPTURED['var_eqhr_dim'] = var_eqhr_dim.numpy()
    CAPTURED['var_absdiff_dim'] = var_absdiff_dim.numpy()
    CAPTURED['var_diff_dim'] = var_diff_dim.numpy()
    CAPTURED['cos_eq_hr'] = cos_eq_hr.numpy()
    CAPTURED['rel_resid'] = rel_resid.numpy()
    CAPTURED['var_hr_uniform_dim'] = var_hr_uniform_dim.numpy()
    CAPTURED['omega_entropy'] = omega_entropy_all.numpy()
    CAPTURED['r_M'] = r_M_total
    CAPTURED['icc_chance_floor'] = icc_chance_floor
    CAPTURED['vw_raw'] = vw_raw
    CAPTURED['vb_raw'] = vb_raw
    CAPTURED['icc_raw'] = icc_raw
    CAPTURED['vw_enc'] = vw_enc
    CAPTURED['vb_enc'] = vb_enc
    CAPTURED['icc_enc'] = icc_enc
    CAPTURED['y_inv_vis'] = y_inv_vis
    CAPTURED['y_dyn_vis'] = y_dyn_vis
    CAPTURED['y_hat_vis'] = y_hat_vis
    CAPTURED['true_vis'] = true_vis

    return float(CAPTURED['mse'].mean()), float(CAPTURED['mae'].mean())


tools_mod.test_retrieve = capturing_test_retrieve

DATASET_TABLE = {
    'ETTh1': dict(root='/home/fenglei/TS-RAG-main/datasets/ETT-small/', data='ett_h_retrieve', freq='hour'),
    'ETTh2': dict(root='/home/fenglei/TS-RAG-main/datasets/ETT-small/', data='ett_h_retrieve', freq='hour'),
    'ETTm1': dict(root='/home/fenglei/TS-RAG-main/datasets/ETT-small/', data='ett_m_retrieve', freq='minute'),
    'ETTm2': dict(root='/home/fenglei/TS-RAG-main/datasets/ETT-small/', data='ett_m_retrieve', freq='minute'),
    'weather': dict(root='/home/fenglei/TS-RAG-main/datasets/weather/', data='custom_retrieve', freq='10minutes'),
    'exchange_rate': dict(root='/home/fenglei/TS-RAG-main/datasets/exchange_rate/', data='custom_retrieve', freq='hour'),
    'electricity': dict(root='/home/fenglei/TS-RAG-main/datasets/electricity/', data='custom_retrieve', freq='hour'),
}

DATASET = os.environ.get('DATASET', 'ETTh1')
ds = DATASET_TABLE[DATASET]

CHECKPOINT_MODEL_PATH = os.environ['CHECKPOINT_MODEL_PATH']
GPU_LOC = os.environ.get('GPU_LOC', '1')
TAG = os.environ.get('TAG', os.path.basename(CHECKPOINT_MODEL_PATH).replace('.pth', ''))
BATCH_SIZE = os.environ.get('EVAL_BATCH_SIZE', '32')

sys.argv = [
    'zeroshot.py',
    '--root_path', ds['root'],
    '--data_path', f'{DATASET}.csv',
    '--model_id', f'{DATASET}_zeroshot_512_pred_64_512_retrieve_64_idf_trr_dualpath_diagnostics',
    '--data', ds['data'],
    '--top_k', '10',
    '--checkpoint_model_path', CHECKPOINT_MODEL_PATH,
    '--pretrained_model_path', '/home/fenglei/TS-RAG-main/TS-RAG/checkpoints/base/',
    '--seq_len', '512',
    '--label_len', '0',
    '--pred_len', '64',
    '--lookback_length', '512',
    '--batch_size', BATCH_SIZE,
    '--num_workers', '0',
    '--decay_fac', '0.5',
    '--freq', '0',
    '--percent', '100',
    '--model', 'ChronosBoltRetrieve',
    '--gpu_loc', GPU_LOC,
    '--tmax', '20',
    '--cos', '1',
    '--save_file_name', 'idf_trr_dualpath_diagnostics_tmp.txt',
    '--retrieval_database_dir', '/home/fenglei/TS-RAG-main/retrieval_database/',
    '--dimension', '768',
    '--embedding_model_type', 'chronos',
    '--metadata_frequency', ds['freq'],
    '--metadata_database_name', DATASET,
    '--augment_mode', 'idf_trr_dualpath',
]

import zeroshot  # noqa: E402,F401

print(f'\n=== idf_trr_dualpath diagnostics: dataset={DATASET}, checkpoint={CHECKPOINT_MODEL_PATH} ===')
mse, mae = CAPTURED['mse'], CAPTURED['mae']
print(f'n_samples={mse.shape[0]}')
print(f'overall MSE={mse.mean():.6f}  MAE={mae.mean():.6f}')

print('\n=== 第一部分：z_inv / z_dyn 本身的解耦质量 ===')
var_inv_dim, var_dyn_dim = CAPTURED['var_inv_dim'], CAPTURED['var_dyn_dim']
GAMMA0 = 0.1
print(f'n_samples_for_z_stats={CAPTURED["n_z_samples"]}  (d_inv=d_dyn={var_inv_dim.shape[0]})')
print(f'z_inv 每维方差: mean={var_inv_dim.mean():.4f}  min={var_inv_dim.min():.4f}  '
      f'低于gamma0({GAMMA0})阈值的维度占比={float((var_inv_dim < GAMMA0).mean()):.4f}')
print(f'z_dyn 每维方差: mean={var_dyn_dim.mean():.4f}  min={var_dyn_dim.min():.4f}  '
      f'低于gamma0({GAMMA0})阈值的维度占比={float((var_dyn_dim < GAMMA0).mean()):.4f}')
print(f'L_xcov（测试集上重新按训练公式算的）: {CAPTURED["xcov_loss_eval"]:.6f}')
print(f'|cos_sim(z_inv, z_dyn)| per-sample: mean={CAPTURED["cos_sim"].mean():.4f}  '
      f'std={CAPTURED["cos_sim"].std():.4f}')

print('\n=== 第二部分：y_inv / y_dyn 输出层面的隐性路由检查 ===')
share = CAPTURED['energy_share_inv']
print(f'branch energy share Var(y_inv)/(Var(y_inv)+Var(y_dyn)): mean={share.mean():.4f}  std={share.std():.4f}')
print(f'saturation_frac(<0.1 or >0.9)={CAPTURED["energy_share_sat_frac"]:.4f}')
print(f'  share<0.1 (几乎全是dyn路径): {(share < 0.1).mean():.4f}')
print(f'  0.1<=share<=0.9 (两条路径都有真实贡献): {((share >= 0.1) & (share <= 0.9)).mean():.4f}')
print(f'  share>0.9 (几乎全是inv路径): {(share > 0.9).mean():.4f}')

print('\n--- roughness ratio R(y) = log(Var(delta2 y) + eps) ---')
r_inv, r_dyn = CAPTURED['roughness_inv'], CAPTURED['roughness_dyn']
print(f'R(y_inv) mean={r_inv.mean():.4f}  R(y_dyn) mean={r_dyn.mean():.4f}  '
      f'R(y_dyn)-R(y_inv) mean={(r_dyn - r_inv).mean():.4f}')

print('\n=== 第三部分：P_inv/P_dyn 原始输入特征本身的方差诊断 ===')
print(f'sanity check: e_q最大逐元素差={CAPTURED["e_q_sanity_maxdiff"]:.2e}')
cos_eq_hr, rel_resid = CAPTURED['cos_eq_hr'], CAPTURED['rel_resid']
print(f'cos_sim(e_q, h_r) per-sample: mean={cos_eq_hr.mean():.4f}  std={cos_eq_hr.std():.4f}')
print(f'||e_q - h_r|| / ||e_q|| per-sample: mean={rel_resid.mean():.4f}  std={rel_resid.std():.4f}')

var_eq_dim, var_hr_dim = CAPTURED['var_eq_dim'], CAPTURED['var_hr_dim']
var_eqhr_dim, var_absdiff_dim, var_diff_dim = (
    CAPTURED['var_eqhr_dim'], CAPTURED['var_absdiff_dim'], CAPTURED['var_diff_dim']
)


def _dimstat(name, v, ref):
    ratio = float(v.mean() / (ref.mean() + 1e-12))
    print(f'  {name}: 每维方差 mean={v.mean():.4f}  min={v.min():.4f}  '
          f'(相对e_q自身方差的比例={ratio:.4f})')


print('各输入分量的逐维方差（跟e_q自身的方差对比）：')
_dimstat('e_q      (query编码本身，基准)', var_eq_dim, var_eq_dim)
_dimstat('h_r      (检索近邻聚合)       ', var_hr_dim, var_eq_dim)
_dimstat('e_q*h_r  (P_inv专用的乘积项)  ', var_eqhr_dim, var_eq_dim)
_dimstat('|e_q-h_r|(P_inv专用的绝对差项)', var_absdiff_dim, var_eq_dim)
_dimstat('e_q-h_r  (P_dyn用的带符号差)  ', var_diff_dim, var_eq_dim)

print('\n=== 第四部分：h_r自己方差这么小，是omega注意力权重太平，还是候选本身就同质？===')
var_hr_uniform_dim = CAPTURED['var_hr_uniform_dim']
omega_entropy = CAPTURED['omega_entropy']
ratio_actual = float(var_hr_dim.mean() / (var_eq_dim.mean() + 1e-12))
ratio_uniform = float(var_hr_uniform_dim.mean() / (var_eq_dim.mean() + 1e-12))
print(f'omega归一化熵(1=完全均匀权重, 0=只认一个候选): mean={omega_entropy.mean():.4f}  '
      f'std={omega_entropy.std():.4f}')
print(f'h_r(实际学出来的omega加权聚合) 相对e_q的方差比例: {ratio_actual:.4f}')
print(f'h_r_uniform(假设改成均匀权重1/r_M简单平均聚合) 相对e_q的方差比例: {ratio_uniform:.4f}')

print('\n=== 第五部分：候选之间的差异，到底跟query有没有关系？(方差分解/ICC) ===')
icc_floor = CAPTURED['icc_chance_floor']
print(f'r_M={CAPTURED["r_M"]}，如果检索完全跟query无关(纯随机)，ICC的理论期望值(有限'
      f'样本效应，不是0)大约是 1/(r_M+1) = {icc_floor:.4f}，可以当作"跟瞎检索没区别"的下限参照。')
print(f'[编码前] retrieved_y 原始值：组内方差={CAPTURED["vw_raw"]:.6f}  '
      f'组间方差={CAPTURED["vb_raw"]:.6f}  ICC={CAPTURED["icc_raw"]:.4f}')
print(f'[编码后] retrieved_y_enc (过完encode_mlp)：组内方差={CAPTURED["vw_enc"]:.6f}  '
      f'组间方差={CAPTURED["vb_enc"]:.6f}  ICC={CAPTURED["icc_enc"]:.4f}')

npz_dir = os.environ.get('NPZ_DIR', 'results/idf_trr_dualpath_diagnostics_npz')
os.makedirs(npz_dir, exist_ok=True)
out_path = os.path.join(npz_dir, f'{DATASET}_{TAG}.npz')
np.savez(
    out_path,
    mse=mse, mae=mae,
    cos_sim=CAPTURED['cos_sim'],
    roughness_inv=CAPTURED['roughness_inv'], roughness_dyn=CAPTURED['roughness_dyn'],
    energy_share_inv=share, energy_share_sat_frac=CAPTURED['energy_share_sat_frac'],
    var_inv_dim=var_inv_dim, var_dyn_dim=var_dyn_dim, xcov_loss_eval=CAPTURED['xcov_loss_eval'],
    var_eq_dim=CAPTURED['var_eq_dim'], var_hr_dim=CAPTURED['var_hr_dim'],
    var_eqhr_dim=CAPTURED['var_eqhr_dim'], var_absdiff_dim=CAPTURED['var_absdiff_dim'],
    var_diff_dim=CAPTURED['var_diff_dim'], cos_eq_hr=CAPTURED['cos_eq_hr'],
    rel_resid=CAPTURED['rel_resid'],
    var_hr_uniform_dim=CAPTURED['var_hr_uniform_dim'], omega_entropy=CAPTURED['omega_entropy'],
    icc_chance_floor=CAPTURED['icc_chance_floor'],
    vw_raw=CAPTURED['vw_raw'], vb_raw=CAPTURED['vb_raw'], icc_raw=CAPTURED['icc_raw'],
    vw_enc=CAPTURED['vw_enc'], vb_enc=CAPTURED['vb_enc'], icc_enc=CAPTURED['icc_enc'],
    y_inv_vis=CAPTURED['y_inv_vis'], y_dyn_vis=CAPTURED['y_dyn_vis'],
    y_hat_vis=CAPTURED['y_hat_vis'], true_vis=CAPTURED['true_vis'],
)
print(f'\nSaved raw arrays + a few sample curves to {out_path}')

if CAPTURED['y_inv_vis'] is not None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    y_inv_vis, y_dyn_vis = CAPTURED['y_inv_vis'], CAPTURED['y_dyn_vis']
    y_hat_vis, true_vis = CAPTURED['y_hat_vis'], CAPTURED['true_vis']
    n_vis = y_inv_vis.shape[0]

    fig, axes = plt.subplots(n_vis, 1, figsize=(9, 2.6 * n_vis), squeeze=False)
    for i in range(n_vis):
        ax = axes[i, 0]
        ax.plot(true_vis[i], label='true', color='black', linewidth=1.2)
        ax.plot(y_hat_vis[i], label='y_hat (fused)', color='tab:purple', linestyle='--', linewidth=1.0)
        ax.plot(y_inv_vis[i], label='y_inv', color='tab:blue', linewidth=1.2)
        ax.plot(y_dyn_vis[i], label='y_dyn', color='tab:orange', linewidth=1.2)
        ax.set_title(f'sample {i}')
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
    fig.suptitle(f'{DATASET} | {TAG}')
    fig.tight_layout()
    plot_dir = os.environ.get('PLOT_DIR', 'results/idf_trr_dualpath_diagnostics_plots')
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f'{DATASET}_{TAG}.png')
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f'Saved y_inv/y_dyn visualization to {plot_path}')
