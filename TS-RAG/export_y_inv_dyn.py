"""
导出一个真实eval batch上的y_inv/y_dyn/target/最终融合预测，用于v3 vs L_ord正则
两个checkpoint之间的可视化对比。

做法跟dump_gamma.py完全一样(已验证过的机制)：在import zeroshot.py之前monkey-patch
utils.tools.test_retrieve，让zeroshot.py模块级别的设置(参数解析/模型构建/checkpoint
加载/data_provider)照常跑，但真正的"整测试集算MSE"循环被替换成"只跑第一个batch，
把y_inv/y_dyn/target/融合预测的前N条样本存成.npz"。这样不用重新拼一遍zeroshot.py的
参数列表，也不用跑一次完整数据集的推理。

用法(环境变量控制)：
  EXPORT_CKPT=<checkpoint路径> EXPORT_AUGMODE=<augment_mode> EXPORT_DATASET=<ETTh2/ETTm1/...>
  EXPORT_OUT=<输出.npz路径> [EXPORT_N=2] python3 export_y_inv_dyn.py
"""
import os
import sys
import torch
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import utils.tools as tools_mod  # noqa: E402

CAPTURED = {}
N = int(os.environ.get('EXPORT_N', '2'))


def capturing_test_retrieve(model, test_data, test_loader, args, device):
    model.eval()
    with torch.no_grad():
        batch = next(iter(test_loader))
        batch_x, batch_y, batch_x_mark, batch_y_mark, retrieved_seqs, distances = batch
        batch_x = batch_x.float().to(device).squeeze()
        batch_y = batch_y.float().to(device).squeeze()
        retrieved_seqs = retrieved_seqs.float().to(device)
        distances = distances.float().to(device)
        outputs = model(context=batch_x, target=batch_y,
                         retrieved_seq=retrieved_seqs, distances=distances)

        q_tensor = torch.tensor(tools_mod.quantiles)  # utils/tools.py顶部的真实quantile列表
        central_idx = torch.abs(q_tensor - 0.5).argmin().item()

        assert outputs.y_inv is not None and outputs.y_dyn is not None, \
            f"augment_mode={args.augment_mode} 的forward()没有产出y_inv/y_dyn " \
            "(只有idf_clean_dis/idf_clean_dis_v3/idf_clean_dis_v4/idf_clean_dis_deepmlp支持)"

        CAPTURED['context'] = batch_x[:N].detach().cpu().numpy()
        CAPTURED['target'] = batch_y[:N].detach().cpu().numpy()
        CAPTURED['pred'] = outputs.quantile_preds[:N, central_idx, :].detach().cpu().numpy()
        CAPTURED['y_inv'] = outputs.y_inv[:N, central_idx, :].detach().cpu().numpy()
        CAPTURED['y_dyn'] = outputs.y_dyn[:N, central_idx, :].detach().cpu().numpy()

    return 0.0, 0.0  # (mse, mae) 占位，避免zeroshot.py后续统计逻辑报错


tools_mod.test_retrieve = capturing_test_retrieve

DATASET_TABLE = {
    'ETTh1': dict(root='/home/fenglei/TS-RAG-main/datasets/ETT-small/', data='ett_h_retrieve', freq='hour'),
    'ETTh2': dict(root='/home/fenglei/TS-RAG-main/datasets/ETT-small/', data='ett_h_retrieve', freq='hour'),
    'ETTm1': dict(root='/home/fenglei/TS-RAG-main/datasets/ETT-small/', data='ett_m_retrieve', freq='minute'),
    'ETTm2': dict(root='/home/fenglei/TS-RAG-main/datasets/ETT-small/', data='ett_m_retrieve', freq='minute'),
    'weather': dict(root='/home/fenglei/TS-RAG-main/datasets/weather/', data='custom_retrieve', freq='10minutes'),
    'electricity': dict(root='/home/fenglei/TS-RAG-main/datasets/electricity/', data='custom_retrieve', freq='hour'),
    'exchange_rate': dict(root='/home/fenglei/TS-RAG-main/datasets/exchange_rate/', data='custom_retrieve', freq='hour'),
}

DATASET = os.environ['EXPORT_DATASET']
ds = DATASET_TABLE[DATASET]
AUGMODE = os.environ['EXPORT_AUGMODE']

sys.argv = [
    'zeroshot.py',
    '--root_path', ds['root'],
    '--data_path', f'{DATASET}.csv',
    '--model_id', f'{DATASET}_zeroshot_512_pred_64_512_retrieve_64_{AUGMODE}',
    '--data', ds['data'],
    '--top_k', '10',
    '--checkpoint_model_path', os.environ['EXPORT_CKPT'],
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
    '--save_file_name', 'export_y_inv_dyn_tmp.txt',
    '--retrieval_database_dir', '/home/fenglei/TS-RAG-main/retrieval_database/',
    '--dimension', '768',
    '--embedding_model_type', 'chronos',
    '--metadata_frequency', ds['freq'],
    '--metadata_database_name', DATASET,
    '--augment_mode', AUGMODE,
]

import zeroshot  # noqa: E402,F401  (执行模块级别的设置 + 我们patch过的test_retrieve)

out_path = os.environ['EXPORT_OUT']
os.makedirs(os.path.dirname(out_path), exist_ok=True)
np.savez(out_path,
         context=CAPTURED['context'],
         target=CAPTURED['target'],
         pred=CAPTURED['pred'],
         y_inv=CAPTURED['y_inv'],
         y_dyn=CAPTURED['y_dyn'])
print(f"已保存到 {out_path}, context={CAPTURED['context'].shape} "
      f"target={CAPTURED['target'].shape} y_inv={CAPTURED['y_inv'].shape} "
      f"y_dyn={CAPTURED['y_dyn'].shape}")
