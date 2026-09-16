"""
诊断目的（对应"环境定义"自查里最关键的疑点）：
用已经训练好、配方对齐的 Query-only 参考模型 B（augment_mode='baseline'，
Stage4训练时本来就会加载它算 R_e^0），在全部~28个环境(parquet分片)上
分别算一次 R_e^0，只做推理，不训练、不反传、不碰任何检索模块。

要回答的问题：环境之间的 R_e^0 是否存在稳定、有实质意义的差异（支持"环境=
有意义的分布分组"），还是所有环境的 R_e^0 都在噪声范围内差不多（支持"环境
本质上是同一个大杂烩语料的随机子样本，CVaR很可能在拟合抽样噪声而不是真实
的分布差异"）——这是决定要不要继续投入下一次Stage4训练前，最该先查清楚的
一件事，而且几乎零成本（不需要GPU训练，一次前向推理扫一遍数据即可）。

关键代码依据（已在 models/ChronosBolt.py 里确认，不是猜测）：
  第1221-1222行：if self.augment == 'baseline': fused_quantile_preds =
      self.output_patch_embedding(sequence_output)...
  这个分支在结构上根本不会进入下面用 retrieved_seq/distances 的 else 分支，
  所以下面 forward() 调用时 retrieved_seq/distances 传 None 是安全的，不是
  假设、也不需要额外验证。

模型B的加载方式完全照抄 pretrain.py 第434-448行的原始代码（不重新发明一套
加载逻辑），避免引入新的、没有被10000步真实训练验证过的构造路径。
"""
import os
import sys
import math
from pathlib import Path

import numpy as np
import torch
import pyarrow.parquet as pq
from transformers import AutoConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.ChronosBolt import ChronosBoltModelForForecastingWithRetrieval

DATA_PATH = os.environ['DATA_PATH']
PRETRAINED_MODEL_PATH = os.environ.get('PRETRAINED_MODEL_PATH', './checkpoints/base/')
BASELINE_CKPT = os.environ['BASELINE_CKPT']
GPU_LOC = os.environ.get('GPU_LOC', '0')
CONTEXT_LENGTH = int(os.environ.get('CONTEXT_LENGTH', '512'))
PREDICTION_LENGTH = int(os.environ.get('PREDICTION_LENGTH', '64'))
SAMPLES_PER_ENV = int(os.environ.get('SAMPLES_PER_ENV', '3000'))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '128'))

STAGE4_TINY_CHUNK_BASENAMES = frozenset({
    'chronos-dataset-sample-50m-chunk-014.parquet',
    'chronos-dataset-sample-50m-chunk-029.parquet',
})

device = f'cuda:{GPU_LOC}' if torch.cuda.is_available() else 'cpu'
print(f"[env异质性诊断] device={device}")

config_B = AutoConfig.from_pretrained(PRETRAINED_MODEL_PATH)
if hasattr(config_B, "chronos_config"):
    config_B.chronos_config["context_length"] = CONTEXT_LENGTH
    config_B.chronos_config["prediction_length"] = PREDICTION_LENGTH
model_B = ChronosBoltModelForForecastingWithRetrieval(config=config_B, augment='baseline')
model_B.debug_shapes = False
model_B._debug_shapes_printed = True
b_state_dict = torch.load(BASELINE_CKPT, map_location='cpu')
missing, unexpected = model_B.load_state_dict(b_state_dict, strict=False)
print(f"[env异质性诊断] 参考模型B从 {BASELINE_CKPT} 加载完成 "
      f"(missing={len(missing)}, unexpected={len(unexpected)})")
model_B.to(device)
model_B.eval()
for p in model_B.parameters():
    p.requires_grad = False

all_chunk_files = sorted(Path(DATA_PATH).glob('*.parquet'))
env_files = [f for f in all_chunk_files if f.name not in STAGE4_TINY_CHUNK_BASENAMES]
print(f"[env异质性诊断] 共 {len(env_files)} 个环境文件")

results = []
with torch.no_grad():
    for env_id, f in enumerate(env_files):
        pf = pq.ParquetFile(f)
        total_rows = pf.metadata.num_rows
        n_batches_total = math.ceil(total_rows / BATCH_SIZE)
        n_batches_needed = max(1, math.ceil(SAMPLES_PER_ENV / BATCH_SIZE))
        stride = max(1, n_batches_total // n_batches_needed)

        losses = []
        n_collected = 0
        for bi, batch in enumerate(pf.iter_batches(batch_size=BATCH_SIZE, columns=['target'])):
            if bi % stride != 0:
                continue
            cols = batch.to_pydict()
            targets = cols['target']
            xs, ys = [], []
            for t in targets:
                t = np.asarray(t, dtype=np.float32)
                if len(t) < CONTEXT_LENGTH + PREDICTION_LENGTH:
                    continue
                xs.append(t[:CONTEXT_LENGTH])
                ys.append(t[CONTEXT_LENGTH:CONTEXT_LENGTH + PREDICTION_LENGTH])
            if not xs:
                continue
            x = torch.from_numpy(np.stack(xs)).float().to(device)
            y = torch.from_numpy(np.stack(ys)).float().to(device)
            outputs = model_B(context=x, target=y, retrieved_seq=None, distances=None)
            per_sample_loss = outputs.loss_forecast_per_sample
            if per_sample_loss is None:
                per_sample_loss = outputs.loss.expand(x.shape[0])
            losses.append(per_sample_loss.detach().cpu().numpy())
            n_collected += x.shape[0]
            if n_collected >= SAMPLES_PER_ENV:
                break

        losses = np.concatenate(losses)[:SAMPLES_PER_ENV] if losses else np.array([np.nan])
        R_e_0 = float(np.nanmean(losses))
        R_e_0_std = float(np.nanstd(losses))
        results.append((env_id, f.name, len(losses), R_e_0, R_e_0_std))
        print(f"env={env_id:2d} file={f.name} n={len(losses):5d} "
              f"R_e^0(mean)={R_e_0:.6f} R_e^0(within_std)={R_e_0_std:.6f}")

vals = np.array([r[3] for r in results])
within_std_mean = np.mean([r[4] for r in results])
print("\n=== 跨环境 R_e^0 分布汇总 ===")
print(f"环境数={len(vals)}")
print(f"跨环境 mean={vals.mean():.6f}  跨环境 std={vals.std():.6f}  "
      f"跨环境变异系数 cv={vals.std()/vals.mean():.4f}")
print(f"min={vals.min():.6f}  max={vals.max():.6f}  max/min={vals.max()/vals.min():.4f}")
print(f"环境内部(单环境采样噪声)典型std={within_std_mean:.6f}")
print(f"跨环境std / 环境内部std = {vals.std()/within_std_mean:.4f}  "
      f"(这个比值远大于1，说明环境间差异明显超出单环境内部的采样噪声，"
      f"支持'环境有实质分布差异'；如果这个比值接近或小于1，说明环境间的差异"
      f"跟一个环境内部随机抽样的波动是一个量级，更可能是噪声而不是真实的分布分组)")

sorted_results = sorted(results, key=lambda r: r[3])
print("\n=== 按 R_e^0 从小到大排序(供人工看是否有断层/分组结构，还是连续均匀分布) ===")
for r in sorted_results:
    print(f"  env={r[0]:2d} file={r[1]} R_e^0={r[3]:.6f}")
