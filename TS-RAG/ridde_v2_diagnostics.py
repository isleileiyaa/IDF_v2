"""
Post-hoc diagnostics for a trained idf_ridde_v2 (RIDDE "Training Objective ver
2.0") ChronosBoltRetrieve checkpoint, covering the per-stage monitoring items
from RIDDE_正则项消融实验清单.md (阶段2-4的"监测1/2/3"):
    - gamma routing-gate distribution / saturation fraction
    - |cos_sim(z_inv, z_dyn)|
    - roughness ratio R(y_inv) vs R(y_dyn) (paper Eq.23)
    - branch output energy share Var(y_inv)/(Var(y_inv)+Var(y_dyn))
    - retrieval-confidence c_i (paper Eq.19) distribution, and MSE/MAE
      stratified by c_i (high vs low half)
    - a few sample curves (y_inv/y_dyn/y_hat/true) saved for plotting

Reuses zeroshot.py's existing checkpoint/data loading by monkey-patching
utils.tools.test_retrieve before importing zeroshot, the same pattern as
dump_gamma.py -- except here we iterate over multiple batches (not just one)
and pull the diag_* fields that ChronosBoltOutput exposes for augment_mode ==
'idf_ridde_v2' (see models/ChronosBolt.py), instead of a single hooked tensor.

Usage:
    DATASET=ETTh1 CHECKPOINT_MODEL_PATH=/path/to/model_final.pth \
        python ridde_v2_diagnostics.py
"""
import os
import sys
import torch
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import utils.tools as tools_mod  # noqa: E402

QUANTILES = tools_mod.quantiles  # [0.1, ..., 0.9]
CENTRAL_IDX = min(range(len(QUANTILES)), key=lambda i: abs(QUANTILES[i] - 0.5))

CAPTURED = {}


def capturing_test_retrieve(model, test_data, test_loader, args, device):
    num_batches = int(os.environ.get('NUM_BATCHES', '20'))
    num_vis_samples = int(os.environ.get('NUM_VIS_SAMPLES', '4'))

    vis_captured = {}

    def hook_inv(module, inp, out):
        vis_captured['y_inv'] = out.detach()

    def hook_dyn(module, inp, out):
        vis_captured['y_dyn'] = out.detach()

    handle_inv = model.inv_pred_head.register_forward_hook(hook_inv)
    handle_dyn = model.dyn_pred_head_clean.register_forward_hook(hook_dyn)

    mse_list, mae_list = [], []
    c_i_list, gamma_list, cos_sim_list = [], [], []
    rough_inv_list, rough_dyn_list, energy_share_list = [], [], []
    gamma_sat_frac_list = []
    vis_saved = False
    y_inv_vis = y_dyn_vis = y_hat_vis = true_vis = None

    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, retrieved_seqs, distances) in enumerate(test_loader):
            if i >= num_batches:
                break
            batch_x = batch_x.float().to(device).squeeze()
            batch_y = batch_y.float().to(device).squeeze()
            retrieved_seqs = retrieved_seqs.float().to(device)
            distances = distances.float().to(device)

            outputs = model(context=batch_x, target=batch_y, retrieved_seq=retrieved_seqs, distances=distances)

            pred = outputs.quantile_preds.to(batch_x)[:, CENTRAL_IDX]  # (B, L)
            true = batch_y
            mse_list.append(((pred - true) ** 2).mean(dim=-1).cpu())
            mae_list.append((pred - true).abs().mean(dim=-1).cpu())

            if outputs.diag_c_i is not None:
                c_i_list.append(outputs.diag_c_i.cpu())
                gamma_list.append(outputs.diag_gamma_per_sample.cpu())
                cos_sim_list.append(outputs.diag_cos_sim_per_sample.cpu())
                rough_inv_list.append(outputs.diag_roughness_inv_per_sample.cpu())
                rough_dyn_list.append(outputs.diag_roughness_dyn_per_sample.cpu())
                energy_share_list.append(outputs.diag_energy_share_per_sample.cpu())
                gamma_sat_frac_list.append(outputs.diag_gamma_sat_frac.cpu())

            if not vis_saved and 'y_inv' in vis_captured:
                n = min(num_vis_samples, batch_x.shape[0])
                pred_dim_shape = pred.shape[-1]
                y_inv_vis = vis_captured['y_inv'][:n].view(n, len(QUANTILES), pred_dim_shape)[:, CENTRAL_IDX].cpu().numpy()
                y_dyn_vis = vis_captured['y_dyn'][:n].view(n, len(QUANTILES), pred_dim_shape)[:, CENTRAL_IDX].cpu().numpy()
                y_hat_vis = pred[:n].cpu().numpy()
                true_vis = true[:n].cpu().numpy()
                vis_saved = True

    handle_inv.remove()
    handle_dyn.remove()

    CAPTURED['mse'] = torch.cat(mse_list).numpy() if mse_list else np.array([])
    CAPTURED['mae'] = torch.cat(mae_list).numpy() if mae_list else np.array([])
    CAPTURED['c_i'] = torch.cat(c_i_list).numpy() if c_i_list else np.array([])
    CAPTURED['gamma'] = torch.cat(gamma_list).numpy() if gamma_list else np.array([])
    CAPTURED['cos_sim'] = torch.cat(cos_sim_list).numpy() if cos_sim_list else np.array([])
    CAPTURED['roughness_inv'] = torch.cat(rough_inv_list).numpy() if rough_inv_list else np.array([])
    CAPTURED['roughness_dyn'] = torch.cat(rough_dyn_list).numpy() if rough_dyn_list else np.array([])
    CAPTURED['energy_share_inv'] = torch.cat(energy_share_list).numpy() if energy_share_list else np.array([])
    CAPTURED['gamma_sat_frac'] = float(np.mean(gamma_sat_frac_list)) if gamma_sat_frac_list else float('nan')
    CAPTURED['y_inv_vis'] = y_inv_vis
    CAPTURED['y_dyn_vis'] = y_dyn_vis
    CAPTURED['y_hat_vis'] = y_hat_vis
    CAPTURED['true_vis'] = true_vis

    mse_all = CAPTURED['mse']
    mae_all = CAPTURED['mae']
    return (float(mse_all.mean()) if mse_all.size else 0.0,
            float(mae_all.mean()) if mae_all.size else 0.0)


tools_mod.test_retrieve = capturing_test_retrieve

# dataset table mirrors script/zeroshot_chronos_idf_ridde_v2.sh
DATASET_TABLE = {
    'ETTh1': dict(root='/home/fenglei/TS-RAG-main/datasets/ETT-small/', data='ett_h_retrieve', freq='hour'),
    'ETTh2': dict(root='/home/fenglei/TS-RAG-main/datasets/ETT-small/', data='ett_h_retrieve', freq='hour'),
    'ETTm1': dict(root='/home/fenglei/TS-RAG-main/datasets/ETT-small/', data='ett_m_retrieve', freq='minute'),
    'ETTm2': dict(root='/home/fenglei/TS-RAG-main/datasets/ETT-small/', data='ett_m_retrieve', freq='minute'),
    'weather': dict(root='/home/fenglei/TS-RAG-main/datasets/weather/', data='custom_retrieve', freq='10minutes'),
    'electricity': dict(root='/home/fenglei/TS-RAG-main/datasets/electricity/', data='custom_retrieve', freq='hour'),
    'exchange_rate': dict(root='/home/fenglei/TS-RAG-main/datasets/exchange_rate/', data='custom_retrieve', freq='hour'),
    'traffic': dict(root='/home/fenglei/TS-RAG-main/datasets/traffic/', data='custom_retrieve', freq='hour'),
    'solar': dict(root='/home/fenglei/TS-RAG-main/datasets/solar/', data='custom_retrieve', freq='10minutes'),
    'PEMS08': dict(root='/home/fenglei/TS-RAG-main/datasets/PEMS08/', data='custom_retrieve', freq='5minutes'),
    'AQWan': dict(root='/home/fenglei/TS-RAG-main/datasets/AQWan/', data='custom_retrieve', freq='hour'),
    'Wind': dict(root='/home/fenglei/TS-RAG-main/datasets/Wind/', data='custom_retrieve', freq='15minutes'),
    'ILI': dict(root='/home/fenglei/TS-RAG-main/datasets/ILI/', data='custom_retrieve', freq='week'),
    'ZafNoo': dict(root='/home/fenglei/TS-RAG-main/datasets/ZafNoo/', data='custom_retrieve', freq='30minutes'),
    'CzeLan': dict(root='/home/fenglei/TS-RAG-main/datasets/CzeLan/', data='custom_retrieve', freq='30minutes'),
}

DATASET = os.environ.get('DATASET', 'ETTh1')
ds = DATASET_TABLE[DATASET]

DEFAULT_CKPT = (
    '/home/fenglei/TS-RAG-main/TS-RAG/checkpoints/'
    'data50m_idf_ridde_v2_512_pred64_lookback512_top10_lr0.0003_drop0.2_adamw_'
    'cosanneal_step10000_bs256_no_embeddingtuning_sem0_xcov0_ord0_final.pth'
)
CHECKPOINT_MODEL_PATH = os.environ.get('CHECKPOINT_MODEL_PATH', DEFAULT_CKPT)
TAU = os.environ.get('TAU', '0.1')
ORD_MARGIN = os.environ.get('ORD_MARGIN', '0.0')
BATCH_SIZE = os.environ.get('EVAL_BATCH_SIZE', '32')
GPU_LOC = os.environ.get('GPU_LOC', '1')
TAG = os.environ.get('TAG', os.path.basename(CHECKPOINT_MODEL_PATH).replace('.pth', ''))

sys.argv = [
    'zeroshot.py',
    '--root_path', ds['root'],
    '--data_path', f'{DATASET}.csv',
    '--model_id', f'{DATASET}_zeroshot_512_pred_64_512_retrieve_64_idf_ridde_v2_diagnostics',
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
    '--save_file_name', 'ridde_v2_diagnostics_tmp.txt',
    '--retrieval_database_dir', '/home/fenglei/TS-RAG-main/retrieval_database/',
    '--dimension', '768',
    '--embedding_model_type', 'chronos',
    '--metadata_frequency', ds['freq'],
    '--metadata_database_name', DATASET,
    '--augment_mode', 'idf_ridde_v2',
    '--tau', TAU,
    '--ord_margin', ORD_MARGIN,
]

import zeroshot  # noqa: E402,F401  (executes module-level setup + our patched test_retrieve)

print(f'\n=== RIDDE ver2.0 diagnostics: dataset={DATASET}, checkpoint={CHECKPOINT_MODEL_PATH} ===')

mse, mae, c_i = CAPTURED['mse'], CAPTURED['mae'], CAPTURED['c_i']
n = mse.shape[0]
print(f'n_samples={n}')
print(f'overall MSE={mse.mean():.6f}  MAE={mae.mean():.6f}')

if c_i.size:
    print('\n--- gamma (routing gate) ---')
    print(f'mean={CAPTURED["gamma"].mean():.4f}  saturation_frac(<0.1 or >0.9)={CAPTURED["gamma_sat_frac"]:.4f}')

    print('\n--- |cos_sim(z_inv, z_dyn)| ---')
    print(f'mean={CAPTURED["cos_sim"].mean():.4f}  std={CAPTURED["cos_sim"].std():.4f}')

    print('\n--- roughness ratio R(y) = log(Var(delta2 y) + eps) ---')
    r_inv, r_dyn = CAPTURED['roughness_inv'], CAPTURED['roughness_dyn']
    print(f'R(y_inv) mean={r_inv.mean():.4f}  R(y_dyn) mean={r_dyn.mean():.4f}  '
          f'R(y_dyn)-R(y_inv) mean={(r_dyn - r_inv).mean():.4f} (want > 0)')

    print('\n--- branch energy share Var(y_inv)/(Var(y_inv)+Var(y_dyn)) ---')
    print(f'mean={CAPTURED["energy_share_inv"].mean():.4f}')

    print('\n--- retrieval-confidence c_i (paper Eq.19) ---')
    print(f'mean={c_i.mean():.4f}  median={np.median(c_i):.4f}  min={c_i.min():.4f}  max={c_i.max():.4f}')

    median_c = np.median(c_i)
    high_mask = c_i >= median_c
    low_mask = ~high_mask
    print('\n--- MSE/MAE stratified by c_i (median split) ---')
    print(f'high c_i (n={high_mask.sum()}): MSE={mse[high_mask].mean():.6f}  MAE={mae[high_mask].mean():.6f}')
    print(f'low  c_i (n={low_mask.sum()}):  MSE={mse[low_mask].mean():.6f}  MAE={mae[low_mask].mean():.6f}')
else:
    print('\n(no diag_* fields captured -- checkpoint/augment_mode is not idf_ridde_v2?)')

npz_dir = os.environ.get('NPZ_DIR', 'results/ridde_v2_diagnostics_npz')
os.makedirs(npz_dir, exist_ok=True)
out_path = os.path.join(npz_dir, f'{DATASET}_{TAG}.npz')
np.savez(
    out_path,
    mse=mse, mae=mae, c_i=c_i,
    gamma=CAPTURED['gamma'], cos_sim=CAPTURED['cos_sim'],
    roughness_inv=CAPTURED['roughness_inv'], roughness_dyn=CAPTURED['roughness_dyn'],
    energy_share_inv=CAPTURED['energy_share_inv'],
    y_inv_vis=CAPTURED['y_inv_vis'], y_dyn_vis=CAPTURED['y_dyn_vis'],
    y_hat_vis=CAPTURED['y_hat_vis'], true_vis=CAPTURED['true_vis'],
)
print(f'\nSaved raw arrays + a few sample curves to {out_path}')

if CAPTURED['y_inv_vis'] is not None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    y_inv_vis = CAPTURED['y_inv_vis']
    y_dyn_vis = CAPTURED['y_dyn_vis']
    y_hat_vis = CAPTURED['y_hat_vis']
    true_vis = CAPTURED['true_vis']
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
    plot_dir = os.environ.get('PLOT_DIR', 'results/ridde_v2_diagnostics_plots')
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f'{DATASET}_{TAG}.png')
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f'Saved y_inv/y_dyn visualization to {plot_path}')
