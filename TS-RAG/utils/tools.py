import os
import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from datetime import datetime
from sklearn.utils import resample
from distutils.util import strtobool

from utils.metrics import metric

plt.switch_backend('agg')
quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss
        
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def convert_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                            numeric_series
                        ):
                            raise Exception(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                )
                            else:
                                raise Exception(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        )


def MASE(x, freq, pred, true):
    masep = np.mean(np.abs(x[:, freq:] - x[:, :-freq]))
    return np.mean(np.abs(pred - true) / (masep + 1e-8))

def boot_res(preds,labels ):
    n_iterations = 1000
    n_size = len(preds)
    stats = [] 
    res =  np.mean(np.abs(preds - labels), axis=(1, 2))  
    print(res.shape)
    assert len(res) == n_size
    for _ in range(n_iterations):
        sample = resample(res, n_samples=n_size , replace=True )  
        stats.append(np.mean(sample) )
    return stats
    
def bootstraptest(model, test_loader, args, device ):
    preds = []
    trues = []
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader)):
            
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            
            outputs = model(batch_x[:, -args.seq_len:, :], 0)
            
            outputs = outputs[:, -args.pred_len:, :]
            batch_y = batch_y[:, -args.pred_len:, :].to(device)

            pred = outputs.detach().cpu().numpy()
            true = batch_y.detach().cpu().numpy()
            
            preds.append(pred)
            trues.append(true)

    preds = np.array(preds)
    trues = np.array(trues)
    
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    
    return np.mean(np.abs(preds - trues), axis=(1, 2))  

def test(model, test_data, test_loader, args, device):
    preds = []
    trues = []
    prevs = []
    model.model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader)):
            
        
            batch_x = batch_x.float().to(device).squeeze()
            batch_y = batch_y.float().to(device).squeeze()
            
            outputs = model.predict_quantiles(context=torch.tensor(batch_x, dtype=torch.float32),
                prediction_length=args.pred_len,
                quantile_levels=[0.5],
                num_samples=5,
            )[1]
            
            # encoder - decoder
            outputs = outputs[:, -args.pred_len:]
            batch_y = batch_y[:, -args.pred_len:].to(device)

            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()
            prev = batch_x[:, -args.seq_len:].cpu()
            
            preds.append(pred)
            trues.append(true)
            prevs.append(prev)

    preds = torch.cat(preds, dim=0).numpy()
    trues = torch.cat(trues, dim=0).numpy()
    prevs = torch.cat(prevs, dim=0).numpy()
    
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    print('test shape:', preds.shape, trues.shape)
    
    mae, mse, rmse, mape, mspe, smape, nd = metric(preds, trues)
    print('mae:{:.4f}, mse:{:.4f}, rmse:{:.4f}, smape:{:.4f}'.format(mae, mse, rmse, smape))

    return mse, mae

def test_longer(model, test_data, test_loader, args, device):
    preds = []
    trues = []
    prevs = []
    if args.model == 'ChronosBolt':
        model.model.eval()
    else:
        model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader)):
            
        
            batch_x = batch_x.float().to(device).squeeze()
            batch_y = batch_y.float().to(device).squeeze()

            # rolling retrieve and predict
            predictions = []
            remaining = args.pred_len
            while remaining > 0:
                if args.model == 'ChronosBolt':
                    outputs = model.predict_quantiles(context=torch.tensor(batch_x, dtype=torch.float32),
                        prediction_length=args.pred_len,
                        quantile_levels=[0.5],
                        num_samples=5,
                    )[1]
                elif args.model == 'MOMENT':
                    outputs = model(x_enc=batch_x.unsqueeze(1))
                    outputs = outputs.forecast.squeeze(1)

                else:
                    raise ValueError('model error')

                # update
                if predictions == []:
                    predictions = outputs
                else:
                    predictions = torch.cat([predictions, outputs], dim=1)

                outputs = outputs.to(batch_x)

                batch_x = torch.cat([batch_x, outputs], dim=1)
                batch_x = batch_x[:, -args.seq_len:]
                remaining -= outputs.shape[-1]

                if remaining <=0:
                    predictions = predictions[:, :args.pred_len]
                    break
                
            pred = predictions.detach().cpu()
            true = batch_y.detach().cpu()
            
            preds.append(pred)
            trues.append(true)

    preds = torch.cat(preds, dim=0).numpy()
    trues = torch.cat(trues, dim=0).numpy()
    
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    if preds.shape != trues.shape:
        preds = preds[:, :, :trues.shape[-1]]
    print('test shape:', preds.shape, trues.shape)
    
    mae, mse, rmse, mape, mspe, smape, nd = metric(preds, trues)
    print('mae:{:.4f}, mse:{:.4f}, rmse:{:.4f}, smape:{:.4f}'.format(mae, mse, rmse, smape))

    return mse, mae

def test_retrieve(model, test_data, test_loader, args, device):
    preds = []
    trues = []
    prevs = []
    # Probe-feature accumulators, per RIDDE_NoRetrieval_Base_重新验证实验方案 步骤6之后
    # 的 Probe 阶段. 第二轮特征工程(第一轮 build_probe.py 全面失败, macro eta=-35.4%,
    # val_acc 明显高于 test_acc 的过拟合特征之后), 在原有3组之外新增若干个依然满足
    # "推理时可得, 不偷看真实未来"约束的特征:
    #   - query-only 新增: q_cv(变异系数), q_acf1(滞后1自相关), q_trend_r2(线性趋势
    #     拟合优度), q_last_dev(末值相对窗口均值/标准差的偏离) -- 都只用 batch_x 本身
    #   - retrieval-aware 新增: dist_top1_gap_norm(最近邻与次近邻的距离差, 按平均距离
    #     归一化, 衡量是否有一个明显更可信的近邻), neighbor_past_corr(query窗口和每个
    #     近邻自己历史窗口的皮尔逊相关系数均值, 比原始L2距离更能反映"形状"匹配度)
    #   - post-prediction 新增: qwidth, 模型自己输出的分位数区间宽度(q90-q10, 对
    #     pred_len取均值) -- 这是模型对自己这次预测有多不确定的自带信号, base/rag/
    #     shuffled 三个分支各自算各自的, 直接存进各自的 preds/trues 缓存里(见下方
    #     np.savez), 不需要额外重跑; build_probe.py 里用 base/rag 两个分支的 qwidth
    #     算比值和差值作为新特征.
    probe_q_std = []
    probe_q_slope = []
    probe_q_roughness = []
    probe_q_cv = []
    probe_q_acf1 = []
    probe_q_trend_r2 = []
    probe_q_last_dev = []
    probe_dist_mean = []
    probe_dist_min = []
    probe_dist_std = []
    probe_dist_top1_gap_norm = []
    probe_neighbor_future_dispersion = []
    probe_neighbor_past_corr = []
    qwidths = []
    model.eval()
    if not os.path.exists(f"./results/retrieve_visulization/{args.model_id.split('_')[0]}"):
        os.makedirs(f"./results/retrieve_visulization/{args.model_id.split('_')[0]}")

    if not os.path.exists(f"./results/retrieve_visulization/{args.model_id.split('_')[0]}_vis"):
        os.makedirs(f"./results/retrieve_visulization/{args.model_id.split('_')[0]}_vis")
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, retrieved_seqs, distances) in tqdm(enumerate(test_loader)):

            batch_x = batch_x.float().to(device).squeeze()
            batch_y = batch_y.float().to(device).squeeze()
            retrieved_seqs = retrieved_seqs.float().to(device)
            distances = distances.float().to(device)
            if getattr(args, 'kill_retrieval', False):
                perm = torch.randperm(retrieved_seqs.shape[0])
                retrieved_seqs = retrieved_seqs[perm]

            # ---- Probe features (cheap, available before seeing the true future) ----
            bx = batch_x.reshape(batch_x.shape[0], -1)  # (B, seq_len)
            q_diff = bx[:, 1:] - bx[:, :-1]
            q_mean = bx.mean(dim=1)
            q_std = bx.std(dim=1)
            q_var = q_std.clamp_min(1e-8) ** 2
            q_roughness = (q_diff ** 2).mean(dim=1) / q_var  # same "relative roughness" shape used elsewhere in this repo
            q_cv = q_std / q_mean.abs().clamp_min(1e-6)
            t_idx = torch.arange(bx.shape[1], device=bx.device, dtype=bx.dtype)
            t_centered = t_idx - t_idx.mean()
            q_slope = (bx * t_centered).sum(dim=1) / (t_centered ** 2).sum()

            bx_c = bx - q_mean.unsqueeze(1)
            ss_tot = (bx_c ** 2).sum(dim=1).clamp_min(1e-8)
            acf1_num = (bx_c[:, 1:] * bx_c[:, :-1]).sum(dim=1)
            q_acf1 = acf1_num / ss_tot
            fitted = q_slope.unsqueeze(1) * t_centered.unsqueeze(0)
            ss_res = ((bx_c - fitted) ** 2).sum(dim=1)
            q_trend_r2 = 1.0 - ss_res / ss_tot
            q_last_dev = (bx[:, -1] - q_mean) / q_std.clamp_min(1e-8)

            d_mean = distances.mean(dim=1)
            d_min = distances.min(dim=1).values
            d_std = distances.std(dim=1)
            if distances.shape[1] > 1:
                d_sorted, _ = torch.sort(distances, dim=1)
                d_top1_gap_norm = (d_sorted[:, 1] - d_sorted[:, 0]) / d_mean.clamp_min(1e-8)
            else:
                d_top1_gap_norm = torch.zeros_like(d_mean)

            # matches the model's own retrieved_seq -> retrieved_x/retrieved_y split (see
            # models/ChronosBolt.py's `retrieved_x, retrieved_y = retrieved_seq.split((r_L-L, L), dim=2)`),
            # computed here independently of the model internals purely from the raw tensors already in scope.
            neighbor_past = retrieved_seqs[:, :, :bx.shape[1]]          # (B, k, seq_len)
            neighbor_future = retrieved_seqs[:, :, -args.pred_len:]     # (B, k, pred_len)
            neighbor_future_mean = neighbor_future.mean(dim=-1)         # (B, k)
            neighbor_dispersion = neighbor_future_mean.var(dim=1)       # (B,)

            bx_exp = bx.unsqueeze(1)                                    # (B, 1, seq_len)
            bx_c_exp = bx_exp - bx_exp.mean(dim=2, keepdim=True)
            np_c = neighbor_past - neighbor_past.mean(dim=2, keepdim=True)
            corr_num = (bx_c_exp * np_c).sum(dim=2)
            corr_den = (bx_c_exp.pow(2).sum(dim=2).clamp_min(1e-8).sqrt()
                        * np_c.pow(2).sum(dim=2).clamp_min(1e-8).sqrt())
            neighbor_past_corr = (corr_num / corr_den).mean(dim=1)      # (B,) avg Pearson corr across k neighbors

            probe_q_std.append(q_std.detach().cpu())
            probe_q_slope.append(q_slope.detach().cpu())
            probe_q_roughness.append(q_roughness.detach().cpu())
            probe_q_cv.append(q_cv.detach().cpu())
            probe_q_acf1.append(q_acf1.detach().cpu())
            probe_q_trend_r2.append(q_trend_r2.detach().cpu())
            probe_q_last_dev.append(q_last_dev.detach().cpu())
            probe_dist_mean.append(d_mean.detach().cpu())
            probe_dist_min.append(d_min.detach().cpu())
            probe_dist_std.append(d_std.detach().cpu())
            probe_dist_top1_gap_norm.append(d_top1_gap_norm.detach().cpu())
            probe_neighbor_future_dispersion.append(neighbor_dispersion.detach().cpu())
            probe_neighbor_past_corr.append(neighbor_past_corr.detach().cpu())
            # ---- end probe features ----

            if args.model == 'ChronosBoltRetrieve':
                outputs = model(context = batch_x,
                                target = batch_y,
                                retrieved_seq = retrieved_seqs,
                                distances = distances)                  # ChronosBoltOutput
                qp = outputs.quantile_preds.to(batch_x)                 # (B, Q, pred_len)
                q_tensor = torch.tensor(quantiles)
                central_idx = torch.abs(q_tensor - 0.5).argmin()
                q10_idx = torch.abs(q_tensor - 0.1).argmin()
                q90_idx = torch.abs(q_tensor - 0.9).argmin()
                qwidth_batch = (qp[:, q90_idx, :] - qp[:, q10_idx, :]).abs().mean(dim=1)  # (B,)
                outputs = qp[:, central_idx]
            elif args.model == 'Moirai2Retrieve':
                outputs = model(context = batch_x,
                                target = batch_y,
                                retrieved_seq = retrieved_seqs,
                                distances = distances)                  # Moirai2RiddeOutput
                outputs = outputs.point_forecast
                qwidth_batch = torch.full((batch_x.shape[0],), float('nan'))
            elif args.model == 'TimesFM25Retrieve':
                outputs = model(context = batch_x,
                                target = batch_y,
                                retrieved_seq = retrieved_seqs,
                                distances = distances)                  # TimesFM25RiddeOutput / TimesFM25MoEOutput
                outputs = outputs.point_forecast
                qwidth_batch = torch.full((batch_x.shape[0],), float('nan'))
            elif args.model == 'MOMENTRetrieve':
                outputs = model(x_enc=batch_x.float().unsqueeze(1), retrieved_seq=retrieved_seqs.float())
                outputs = outputs.forecast.squeeze(1)
                qwidth_batch = torch.full((batch_x.shape[0],), float('nan'))
            else:
                raise ValueError('model error')

            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()

            preds.append(pred)
            trues.append(true)
            qwidths.append(qwidth_batch.detach().cpu())

    preds = torch.cat(preds, dim=0).numpy()
    trues = torch.cat(trues, dim=0).numpy()
    qwidths = torch.cat(qwidths, dim=0).numpy()

    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    print('test shape:', preds.shape, trues.shape)

    probe_features = {
        'q_std': torch.cat(probe_q_std).numpy(),
        'q_slope': torch.cat(probe_q_slope).numpy(),
        'q_roughness': torch.cat(probe_q_roughness).numpy(),
        'q_cv': torch.cat(probe_q_cv).numpy(),
        'q_acf1': torch.cat(probe_q_acf1).numpy(),
        'q_trend_r2': torch.cat(probe_q_trend_r2).numpy(),
        'q_last_dev': torch.cat(probe_q_last_dev).numpy(),
        'dist_mean': torch.cat(probe_dist_mean).numpy(),
        'dist_min': torch.cat(probe_dist_min).numpy(),
        'dist_std': torch.cat(probe_dist_std).numpy(),
        'dist_top1_gap_norm': torch.cat(probe_dist_top1_gap_norm).numpy(),
        'neighbor_future_dispersion': torch.cat(probe_neighbor_future_dispersion).numpy(),
        'neighbor_past_corr': torch.cat(probe_neighbor_past_corr).numpy(),
    }

    # Three-way cache tag: 'shuffled' (kill_retrieval / batch-permuted-neighbors arm --
    # Shuffled-RAG negative control ONLY, per the doc's explicit "禁止替代": must NOT be
    # treated as Base), 'truebase' (augment_mode='baseline' with a real trained checkpoint
    # loaded -- the legitimate No-Retrieval Base), or 'rag' (the real retrieval-augmented
    # run). A plain zero-shot augment_mode='baseline' call with no checkpoint (e.g.
    # zeroshot_chronos_baseline.sh, channel_quartile_analysis.py) is left untagged and not
    # cached here -- it isn't a matched-training-budget arm and shouldn't collide with
    # truebase (deviation from the literal patch: kept this checkpoint-presence gate from
    # the previous round rather than tagging every 'baseline' call as 'truebase', since that
    # would let a future plain zero-shot run silently overwrite the real truebase cache).
    if getattr(args, 'kill_retrieval', False):
        oracle_tag = 'shuffled'
    elif getattr(args, 'augment_mode', None) == 'baseline':
        ckpt_path = getattr(args, 'checkpoint_model_path', 'None')
        oracle_tag = 'truebase' if ckpt_path not in (None, 'None', '') and os.path.exists(ckpt_path) else None
    else:
        oracle_tag = 'rag'

    if oracle_tag is not None:
        # model_id is "{dataset}_zeroshot_...": split on '_zeroshot' rather than the first
        # '_' so dataset names containing underscores (e.g. exchange_rate) survive intact
        # (deviation from the literal patch's split('_')[0], which regresses a bug fixed
        # earlier this round -- exchange_rate would otherwise get truncated to 'exchange'
        # and stop matching the already-cached exchange_rate_*.npz files).
        dataset_name = args.model_id.split('_zeroshot')[0]
        os.makedirs('results/oracle_cache', exist_ok=True)
        # eval_split-aware filename: 'test' 保持原来不带后缀的文件名(现有测试集缓存不受影响)，
        # 其他 split(val) 加后缀，永远不会覆盖测试集缓存。
        eval_split = getattr(args, 'eval_split', 'test')
        split_suffix = '' if eval_split == 'test' else f'_{eval_split}'
        # qwidth (模型自己的分位数区间宽度) 现在对 truebase/rag/shuffled 三个分支都存进
        # 同一个 preds/trues 缓存里 -- 它是每个分支各自的输出, 不需要额外重跑; 第二轮
        # build_probe.py 里用 base 和 rag 两个分支的 qwidth 算比值/差值当新特征.
        np.savez(f'results/oracle_cache/{dataset_name}_{oracle_tag}{split_suffix}.npz',
                 preds=preds, trues=trues, qwidth=qwidths)
        if oracle_tag == 'rag':
            # Only the real RAG arm's retrieval geometry is meaningful as a probe feature
            # source (truebase/shuffled runs still have retrieved_seqs/distances in the
            # batch from the dataloader, but the model never used them to produce its
            # prediction, so they're not informative there).
            np.savez(f'results/oracle_cache/{dataset_name}_rag_probe_features{split_suffix}.npz', **probe_features)

    mae, mse, rmse, mape, mspe, smape, nd = metric(preds, trues)
    print('mae:{:.4f}, mse:{:.4f}, rmse:{:.4f}, smape:{:.4f}'.format(mae, mse, rmse, smape))

    return mse, mae

def scaler_inverse_transform(scaler, data, id):
    scale = scaler.scale_[id]
    mean = scaler.mean_[id]
    data = data.cpu().numpy()
    data_inv = data * scale + mean
    return data_inv

def retrieve_given_feat_id(retrievers, raw_datas, batch_x, feat_id, top_k, embedding_model, scaler):
    # devide batch_x based on feat_id
    distances = np.array([])
    retrieved_seqs = np.array([])
    unique_ids = feat_id.unique()
    for unique_id in unique_ids:
        idx = (feat_id == unique_id)
        batch_x_sub = batch_x[idx]
        batch_x_sub = scaler_inverse_transform(scaler, batch_x_sub, unique_id)
        query_sub, _ = embedding_model.embed(torch.from_numpy(batch_x_sub).to(feat_id.device))
        query_sub = query_sub[:,-1,:].squeeze().float().numpy()
        distances_sub, _, timestamp_idx_sub = retrievers[unique_id].search(query_sub, top_k=top_k)
        retrieved_seq_sub = np.array([
            [raw_datas[unique_id][idx: idx + 512 + 64] for idx in row] 
            for row in timestamp_idx_sub
        ])
        if distances.size == 0:
            distances = distances_sub
            retrieved_seqs = retrieved_seq_sub
        else:
            distances = np.concatenate([distances, distances_sub], axis=0)
            retrieved_seqs = np.concatenate([retrieved_seqs, retrieved_seq_sub], axis=0)
    distances = torch.from_numpy(distances)
    retrieved_seqs = torch.from_numpy(retrieved_seqs)
    return retrieved_seqs, distances

def test_real_time_retrieve(model, test_data, test_loader, args, device, retrievers, retriever_rawdata, embedding_model):
    preds = []
    trues = []
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, feat_id) in tqdm(enumerate(test_loader)):

            batch_x = batch_x.float().to(device).squeeze(-1)
            batch_y = batch_y.float().to(device).squeeze(-1)
            
            # rolling retrieve and predict
            predictions = []
            remaining = args.pred_len
            while remaining > 0:
                # retrieve
                batch_x_query = batch_x[:, -args.seq_len:].cpu()
                retrieved_seqs, distances = retrieve_given_feat_id(retrievers, retriever_rawdata, batch_x_query, feat_id, args.top_k, embedding_model, test_data.scaler)

                retrieved_seqs = retrieved_seqs.float().to(device)
                distances = distances.float().to(device)
                outputs = model(context = batch_x,
                                retrieved_seq = retrieved_seqs, 
                                distances = distances)                  # ChronosBoltOutput
                outputs = outputs.quantile_preds.to(batch_x)
            
                central_idx = torch.abs(torch.tensor(quantiles) - 0.5).argmin()
                outputs = outputs[:, central_idx]

                # update
                if predictions == []:
                    predictions = outputs
                else:
                    predictions = torch.cat([predictions, outputs], dim=1)
                batch_x = torch.cat([batch_x, outputs], dim=1)
                remaining -= outputs.shape[-1]
                
                if remaining <=0:
                    predictions = predictions[:, :args.pred_len]
                    break

            pred = predictions.detach().cpu()
            true = batch_y.detach().cpu()
            preds.append(pred)
            trues.append(true)

    preds = torch.cat(preds, dim=0).numpy()
    trues = torch.cat(trues, dim=0).numpy()

    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    print('test shape:', preds.shape, trues.shape)

    mae, mse, rmse, mape, mspe, smape, nd = metric(preds, trues)
    print('mae:{:.4f}, mse:{:.4f}, rmse:{:.4f}, smape:{:.4f}'.format(mae, mse, rmse, smape))

    return mse, mae

def multi_test_retrieve(model, test_loaders, args, device, dataset_probabilities):
    test_results = []
    for loader, prob in zip(test_loaders, dataset_probabilities):
        mse, mae = test_retrieve(model, None, loader, args, device, 1)
        test_results.append((mse, mae, prob))
    return test_results

def get_borders(dataset_name, seq_len, total_length=None):
    """
    get the border index according to the dataset name, sequence length and set type.
    """
    if 'ETTh' in dataset_name:
        border1s = [0, 12 * 30 * 24 - seq_len, 12 * 30 * 24 + 4 * 30 * 24 - seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
    elif 'ETTm' in dataset_name:
        border1s = [0, 12 * 30 * 24 * 4 - seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
    elif dataset_name in ['electricity', 'exchange_rate', 'weather', 'traffic', 'solar', 'Wind', 'ILI', 'ZafNoo', 'CzeLan']:
        if total_length is None:
            raise ValueError("need to provide total_length for {}".format(dataset_name))
        num_train = int(total_length * 0.7)
        num_test  = int(total_length * 0.2)
        num_vali  = total_length - num_train - num_test
        border1s = [0, num_train - seq_len, total_length - num_test - seq_len]
        border2s = [num_train, num_train + num_vali, total_length]
    elif dataset_name in ['PEMS08', 'AQWan']:
        # TFB (Qiu et al., VLDB 2024) documents PEMS08 and AQWan with a 6:2:2
        # split (unlike the 7:1:2 used for electricity/weather/traffic/solar
        # here), matching the same ratio already used for ETTh/ETTm above.
        if total_length is None:
            raise ValueError("need to provide total_length for {}".format(dataset_name))
        num_train = int(total_length * 0.6)
        num_vali  = int(total_length * 0.2)
        num_test  = total_length - num_train - num_vali
        border1s = [0, num_train - seq_len, total_length - num_test - seq_len]
        border2s = [num_train, num_train + num_vali, total_length]
    else:
        raise ValueError("Unknown dataset name: {}".format(dataset_name))
    return border1s, border2s
