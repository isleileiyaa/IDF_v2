"""
Post-hoc diagnostics for a trained idf_clean_dis ("RIDDE ver1.0", L = L_pred +
rho2*L_dis) ChronosBoltRetrieve checkpoint, computing the same metrics as
ridde_v2_diagnostics.py (gamma distribution/saturation, |cos_sim(z_inv,z_dyn)|,
roughness ratio, branch energy share, sample curves) so ver1.0 and ver2.0
checkpoints can be compared apples-to-apples.

idf_clean_dis does not expose diag_* fields on ChronosBoltOutput (those only
exist for augment_mode == 'idf_ridde_v2'), so this script captures the needed
tensors via forward hooks instead:
    - routing_gate's raw (pre-sigmoid) output -> gamma
    - inv_pred_head / dyn_pred_head_clean's *inputs*  -> z_inv / z_dyn
    - inv_pred_head / dyn_pred_head_clean's *outputs* -> y_inv / y_dyn (flat,
      reshaped to (B, num_quantiles, L) here since the module itself returns
      the pre-.view() flat tensor)

Same monkeypatch-test_retrieve-before-import-zeroshot pattern as dump_gamma.py
/ ridde_v2_diagnostics.py.

Usage:
    DATASET=ETTh1 CHECKPOINT_MODEL_PATH=/path/to/model_final.pth \
        RHO1=0 RHO2=1 RHO3=0 RHO4=0 python v1_dis_diagnostics.py
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

    def hook_gate(module, inp, out):
        hooked['gamma'] = torch.sigmoid(out).detach()

    def hook_inv(module, inp, out):
        hooked['z_inv'] = inp[0].detach()
        hooked['y_inv'] = out.detach()

    def hook_dyn(module, inp, out):
        hooked['z_dyn'] = inp[0].detach()
        hooked['y_dyn'] = out.detach()

    handles = [
        model.routing_gate.register_forward_hook(hook_gate),
        model.inv_pred_head.register_forward_hook(hook_inv),
        model.dyn_pred_head_clean.register_forward_hook(hook_dyn),
    ]

    mse_list, mae_list = [], []
    gamma_list, cos_sim_list, rough_inv_list, rough_dyn_list, energy_share_list = [], [], [], [], []
    gamma_sat_frac_list = []
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

            outputs = model(context=batch_x, target=batch_y, retrieved_seq=retrieved_seqs, distances=distances)

            pred = outputs.quantile_preds.to(batch_x)[:, CENTRAL_IDX]
            true = batch_y
            mse_list.append(((pred - true) ** 2).mean(dim=-1).cpu())
            mae_list.append((pred - true).abs().mean(dim=-1).cpu())

            gamma = hooked['gamma']
            z_inv = hooked['z_inv']
            z_dyn = hooked['z_dyn']
            B = z_inv.shape[0]
            y_inv = hooked['y_inv'].view(B, NUM_Q, pred_len)
            y_dyn = hooked['y_dyn'].view(B, NUM_Q, pred_len)

            gamma_list.append(gamma.mean(dim=-1).cpu())
            gamma_sat_frac_list.append(((gamma < 0.1) | (gamma > 0.9)).float().mean().cpu())
            cos_sim_list.append(torch.nn.functional.cosine_similarity(z_inv, z_dyn, dim=-1).abs().cpu())

            R_inv = roughness(y_inv).mean(dim=1)
            R_dyn = roughness(y_dyn).mean(dim=1)
            rough_inv_list.append(R_inv.cpu())
            rough_dyn_list.append(R_dyn.cpu())

            var_inv = y_inv.var(dim=-1).mean(dim=1)
            var_dyn = y_dyn.var(dim=-1).mean(dim=1)
            energy_share_list.append((var_inv / (var_inv + var_dyn + 1e-8)).cpu())

            if not vis_saved:
                n = min(num_vis_samples, B)
                y_inv_vis = y_inv[:n, CENTRAL_IDX].cpu().numpy()
                y_dyn_vis = y_dyn[:n, CENTRAL_IDX].cpu().numpy()
                y_hat_vis = pred[:n].cpu().numpy()
                true_vis = true[:n].cpu().numpy()
                vis_saved = True

    for h in handles:
        h.remove()

    CAPTURED['mse'] = torch.cat(mse_list).numpy()
    CAPTURED['mae'] = torch.cat(mae_list).numpy()
    CAPTURED['gamma'] = torch.cat(gamma_list).numpy()
    CAPTURED['gamma_sat_frac'] = float(torch.stack(gamma_sat_frac_list).mean())
    CAPTURED['cos_sim'] = torch.cat(cos_sim_list).numpy()
    CAPTURED['roughness_inv'] = torch.cat(rough_inv_list).numpy()
    CAPTURED['roughness_dyn'] = torch.cat(rough_dyn_list).numpy()
    CAPTURED['energy_share_inv'] = torch.cat(energy_share_list).numpy()
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
}

DATASET = os.environ.get('DATASET', 'ETTh1')
ds = DATASET_TABLE[DATASET]

RHO1 = os.environ.get('RHO1', '0')
RHO2 = os.environ.get('RHO2', '0')
RHO3 = os.environ.get('RHO3', '0')
RHO4 = os.environ.get('RHO4', '0')
CHECKPOINT_MODEL_PATH = os.environ['CHECKPOINT_MODEL_PATH']
GPU_LOC = os.environ.get('GPU_LOC', '1')
TAG = os.environ.get('TAG', os.path.basename(CHECKPOINT_MODEL_PATH).replace('.pth', ''))
BATCH_SIZE = os.environ.get('EVAL_BATCH_SIZE', '32')

sys.argv = [
    'zeroshot.py',
    '--root_path', ds['root'],
    '--data_path', f'{DATASET}.csv',
    '--model_id', f'{DATASET}_zeroshot_512_pred_64_512_retrieve_64_idf_clean_dis_diagnostics',
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
    '--save_file_name', 'v1_dis_diagnostics_tmp.txt',
    '--retrieval_database_dir', '/home/fenglei/TS-RAG-main/retrieval_database/',
    '--dimension', '768',
    '--embedding_model_type', 'chronos',
    '--metadata_frequency', ds['freq'],
    '--metadata_database_name', DATASET,
    '--augment_mode', 'idf_clean_dis',
]

import zeroshot  # noqa: E402,F401

print(f'\n=== v1 L_dis diagnostics: dataset={DATASET}, checkpoint={CHECKPOINT_MODEL_PATH} ===')
mse, mae = CAPTURED['mse'], CAPTURED['mae']
print(f'n_samples={mse.shape[0]}')
print(f'overall MSE={mse.mean():.6f}  MAE={mae.mean():.6f}')

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

npz_dir = os.environ.get('NPZ_DIR', 'results/ridde_v2_diagnostics_npz')
os.makedirs(npz_dir, exist_ok=True)
out_path = os.path.join(npz_dir, f'{DATASET}_{TAG}.npz')
np.savez(
    out_path,
    mse=mse, mae=mae,
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
    plot_dir = os.environ.get('PLOT_DIR', 'results/ridde_v2_diagnostics_plots')
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f'{DATASET}_{TAG}.png')
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f'Saved y_inv/y_dyn visualization to {plot_path}')
