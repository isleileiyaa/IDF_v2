"""
Run a trained idf_clean_dis (RIDDE) ChronosBoltRetrieve checkpoint on one
eval batch of ETTh1 and dump the values of the routing gate Gamma:
    gamma = sigmoid(routing_gate([h, h_ret]))

Gamma is a per-sample activation (not a static weight), so this script
patches utils.tools.test_retrieve *before* importing zeroshot.py, so that
when zeroshot.py's module-level setup calls test_retrieve(...) to run the
usual full-dataset evaluation, it instead runs just one batch through a
forward hook on model.routing_gate and captures gamma. This reuses all of
zeroshot.py's existing checkpoint/data loading logic without duplicating it
and without paying for a full-dataset eval pass.
"""
import os
import sys
import torch
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import utils.tools as tools_mod  # noqa: E402

CAPTURED = {}

def capturing_test_retrieve(model, test_data, test_loader, args, device):
    captured = {}

    def hook(module, inp, out):
        captured['gamma'] = torch.sigmoid(out).detach()

    handle = model.routing_gate.register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        batch = next(iter(test_loader))
        batch_x, batch_y, batch_x_mark, batch_y_mark, retrieved_seqs, distances = batch
        batch_x = batch_x.float().to(device).squeeze()
        batch_y = batch_y.float().to(device).squeeze()
        retrieved_seqs = retrieved_seqs.float().to(device)
        distances = distances.float().to(device)
        model(context=batch_x, target=batch_y, retrieved_seq=retrieved_seqs, distances=distances)
    handle.remove()

    CAPTURED['gamma'] = captured['gamma']
    CAPTURED['batch_x'] = batch_x.detach().cpu()
    return 0.0, 0.0  # (mse, mae) placeholders so zeroshot.py's bookkeeping doesn't choke

tools_mod.test_retrieve = capturing_test_retrieve

# dataset table mirrors script/zeroshot_chronos_idf_clean_dis.sh
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

DATASET = os.environ.get('GAMMA_DATASET', 'ETTh1')
ds = DATASET_TABLE[DATASET]

sys.argv = [
    'zeroshot.py',
    '--root_path', ds['root'],
    '--data_path', f'{DATASET}.csv',
    '--model_id', f'{DATASET}_zeroshot_512_pred_64_512_retrieve_64_idf_clean_dis',
    '--data', ds['data'],
    '--top_k', '10',
    '--checkpoint_model_path',
    '/home/fenglei/TS-RAG-main/TS-RAG/checkpoints/'
    'data50m_idf_clean_dis_512_pred64_lookback512_top10_lr0.0003_drop0.2_adamw_'
    'cosanneal_step10000_bs256_no_embeddingtuning_rho0_0_0_0_final.pth',
    '--pretrained_model_path', '/home/fenglei/TS-RAG-main/TS-RAG/checkpoints/base/',
    '--seq_len', '512',
    '--label_len', '0',
    '--pred_len', '64',
    '--lookback_length', '512',
    '--batch_size', '32',
    '--num_workers', '0',
    '--decay_fac', '0.5',
    '--freq', '0',
    '--percent', '100',
    '--model', 'ChronosBoltRetrieve',
    '--gpu_loc', '0',
    '--tmax', '20',
    '--cos', '1',
    '--save_file_name', 'gamma_dump_tmp.txt',
    '--retrieval_database_dir', '/home/fenglei/TS-RAG-main/retrieval_database/',
    '--dimension', '768',
    '--embedding_model_type', 'chronos',
    '--metadata_frequency', ds['freq'],
    '--metadata_database_name', DATASET,
    '--augment_mode', 'idf_clean_dis',
]

import zeroshot  # noqa: E402,F401  (executes module-level setup + our patched test_retrieve)

gamma = CAPTURED['gamma']  # (B, d_model)
print(f'\n=== dataset: {DATASET} ===')
print(f'=== gamma tensor shape: {tuple(gamma.shape)} ===')
print(f'gamma dtype: {gamma.dtype}, device: {gamma.device}')

g = gamma.float().cpu().numpy()
print('\n--- global stats over all (B, d_model) entries ---')
print(f'mean={g.mean():.4f}  std={g.std():.4f}  min={g.min():.4f}  max={g.max():.4f}')

near0 = (g < 0.1).mean()
near1 = (g > 0.9).mean()
middle = 1 - near0 - near1
print(f'fraction near 0 (<0.1): {near0:.4f}')
print(f'fraction near 1 (>0.9): {near1:.4f}')
print(f'fraction in between:    {middle:.4f}')

print(f'\n--- per-sample mean gamma (avg over d_model), first {min(10, g.shape[0])} samples ---')
per_sample_mean = g.mean(axis=1)
for i, v in enumerate(per_sample_mean[:10]):
    print(f'sample {i}: mean(gamma)={v:.4f}')

print('\n--- gamma vector for sample 0, first 20 dims ---')
print(np.array2string(g[0, :20], precision=4, suppress_small=True))

out_path = f'/tmp/gamma_values_{DATASET}.npy'
np.save(out_path, g)
print(f'\nSaved full gamma tensor to {out_path}, shape', g.shape)
