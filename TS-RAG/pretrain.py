import os
import time
import wandb
import torch
import random
import argparse
import warnings
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from transformers import AutoConfig
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from models.moment import MOMENTPipelineWithRetrieval
from dataset import CustomPretrainDataset, Retriever_for_pretrain
from models.ChronosBolt import ChronosBoltModelForForecasting, ChronosBoltModelForForecastingWithRetrieval
from models.Moirai2 import Moirai2ModelForForecastingWithRetrieval, Moirai2MoEModelForForecastingWithRetrieval
    
warnings.filterwarnings('ignore')


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {'true', '1', 'yes', 'y'}:
        return True
    if value in {'false', '0', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean value: {value}')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='ChronosBoltRetrieve')

parser.add_argument('--model_id', type=str, default='ChronosBoltRetrieve_Pretrain')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

# retrieve
parser.add_argument('--embedding_tuning', type=str, default=None)
parser.add_argument('--top_k', type=int, default=10)
parser.add_argument('--embedding_model_type', type=str, default='chronos')
parser.add_argument('--retrieve_lookback_length', type=int, default=64)
parser.add_argument('--retrieval_database_path', type=str, default='../database/pretrain/retrieval_database_512.parquet')

# augment
parser.add_argument('--augment_mode', type=str, default='moe2')
parser.add_argument('--debug_shapes', action='store_true', help='print key tensor shapes once for debug')
parser.add_argument('--rho1', type=float, default=0.0)
parser.add_argument('--rho2', type=float, default=0.0)
parser.add_argument('--rho3', type=float, default=0.0)
parser.add_argument('--rho4', type=float, default=0.0)
parser.add_argument('--lambda1', type=float, default=0.0)
parser.add_argument('--lambda2', type=float, default=0.0)
parser.add_argument('--tau', type=float, default=0.1)
parser.add_argument('--dyn_margin', type=float, default=1.0)
parser.add_argument('--aux_loss_detach_ret', type=str2bool, default=True)

# model
parser.add_argument('--model', type=str, default='ChronosBoltRetrieve')
parser.add_argument('--freeze_chronos_bolt', action='store_true', help="freeze the params of chronos-bolt.")
parser.add_argument('--pretrained_model_path', type=str, default='./checkpoints/base/')
parser.add_argument('--context_length', type=int, default=512)
parser.add_argument('--prediction_length', type=int, default=64)

# pretrain
parser.add_argument('--data_path', type=str, default='../datasets/pretrain/50m-with-retrieval_512', help='pretrain data path')
parser.add_argument('--train_steps', type=int, default=200_000)
parser.add_argument('--evaluation_steps', type=int, default=10_000)
parser.add_argument('--optimizer', type=str, default='adamw')
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--tmax', type=int, default=20)
parser.add_argument('--drop_prob', type=float, default=0.2)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--shuffle_buffer_length', type=int, default=100_000)
parser.add_argument('--grad_clip_value', type=float, default=1.0)

# gpu
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--gpu_loc', type=int, default=0, help='main gpu location')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)


args = parser.parse_args()

# init wandb project
wandb.init(project=f'{args.model}_Pretrain', name=args.model_id)
wandb.config.update(args)


if torch.cuda.is_available():
    device = 'cuda:' + str(args.gpu_loc)
else:
    device = 'cpu'

time_now = time.time()

## load model, optimizer
config = AutoConfig.from_pretrained(args.pretrained_model_path)
if hasattr(config, "chronos_config"):
    config.chronos_config["context_length"] = args.context_length
    config.chronos_config["prediction_length"] = args.prediction_length
if args.model == 'ChronosBolt':
    model = ChronosBoltModelForForecasting.from_pretrained(args.pretrained_model_path, config=config)
    model.load_state_dict(torch.load('./checkpoints/base/autogluon_model.pth'), strict=False)
elif args.model == 'ChronosBoltRetrieve':
    model = ChronosBoltModelForForecastingWithRetrieval(config=config, augment=args.augment_mode)
    model.debug_shapes = args.debug_shapes
    model._debug_shapes_printed = False
    model.rho1 = args.rho1
    model.rho2 = args.rho2
    model.rho3 = args.rho3
    model.rho4 = args.rho4
    model.lambda1 = args.lambda1
    model.lambda2 = args.lambda2
    model.tau = args.tau
    model.dyn_margin = args.dyn_margin
    model.aux_loss_detach_ret = args.aux_loss_detach_ret
    model.load_state_dict(torch.load('./checkpoints/base/autogluon_model.pth'), strict=False)
    if 'moe' in args.augment_mode:
        model.init_extra_weights([model.encode_mlp, model.mha, model.ffn, model.gate_layer])
    if args.augment_mode == 'moe_disentangle':
        model.init_extra_weights([model.disentangle_gate, model.final_pred_head, model.inv_aux_backproj, model.dyn_aux_backproj])
    if 'gate' in args.augment_mode:
        model.init_extra_weights([model.gate_layer, model.gate_linear1, model.gate_linear2])
    if args.augment_mode == 'idf':
        model.init_extra_weights([
            model.encode_mlp,
            model.ret_score_head,
            model.fuse_gate,
            model.routing_gate,
            model.inv_transition,
            model.dyn_transition,
            model.inv_head,
            model.dyn_head,
            model.final_head,
        ])
    if args.augment_mode in ['idf_branch', 'idf_x']:
        model.init_extra_weights([
            model.encode_mlp,
            model.ret_score_head,
            model.fuse_gate,
            model.routing_gate,
            model.inv_pred_head,
            model.inv_residual_head,
            model.dyn_pred_head,
            model.final_pred_head,
            model.inv_aux_backproj,
            model.dyn_aux_backproj,
        ])
    if args.augment_mode == 'idf_clean_dis':
        model.init_extra_weights([
            model.encode_mlp,
            model.ret_score_head,
            model.fuse_gate,
            model.routing_gate,
            model.inv_pred_head,
            model.dyn_pred_head_clean,
            model.final_pred_head,
            model.inv_aux_backproj,
            model.dyn_aux_backproj,
        ])
    if args.augment_mode == 'idf_clean_dis_deepmlp':
        model.init_extra_weights([
            model.encode_mlp,
            model.ret_score_head,
            model.fuse_gate,
            model.routing_gate,
            model.inv_pred_head,
            model.dyn_pred_head_clean,
            model.final_pred_head,
            model.inv_aux_backproj,
            model.dyn_aux_backproj,
        ])
    if args.augment_mode == 'idf_clean_dis_ts3align':
        model.init_extra_weights([
            model.encode_mlp,
            model.fuse_gate,
            model.routing_gate,
            model.inv_pred_head,
            model.dyn_pred_head_clean,
            model.final_pred_head,
            model.inv_aux_backproj,
            model.dyn_aux_backproj,
        ])
    if args.augment_mode == 'idf_h_linear_head':
        model.init_extra_weights([
            model.encode_mlp,
            model.ret_score_head,
            model.fuse_gate,
            model.h_pred_head,
        ])
    if args.augment_mode == 'idf_h_native_head':
        model.init_extra_weights([
            model.encode_mlp,
            model.ret_score_head,
            model.fuse_gate,
        ])
    if args.augment_mode == 'idf_y_linear_head':
        model.init_extra_weights([
            model.encode_mlp,
            model.ret_score_head,
            model.fuse_gate,
            model.h_pred_head,
            model.y_linear_head,
        ])
    if args.augment_mode == 'idf_branch_gru':
        model.init_extra_weights([
            model.encode_mlp,
            model.ret_score_head,
            model.fuse_gate,
            model.routing_gate,
            model.inv_pred_head,
            model.inv_residual_head,
            model.dyn_gru,
            model.dyn_out_head,
            model.final_pred_head,
        ])
    if args.augment_mode == 'idf_branch_gru_q':
        model.init_extra_weights([
            model.encode_mlp,
            model.ret_score_head,
            model.fuse_gate,
            model.routing_gate,
            model.inv_pred_head,
            model.inv_residual_head,
            model.dyn_gru,
            model.dyn_out_head,
            model.final_pred_head,
        ])
    if args.augment_mode == 'idf_dual_direct_head':
        model.init_extra_weights([
            model.encode_mlp,
            model.ret_score_head,
            model.fuse_gate,
            model.routing_gate,
            model.inv_hidden_proj,
            model.dyn_ret_hidden_proj,
            model.final_pred_head,
        ])
    if args.augment_mode == 'idf_residual':
        model.init_extra_weights([
            model.encode_mlp,
            model.ret_score_head,
            model.fuse_gate,
            model.routing_gate,
            model.inv_pred_head,
            model.dyn_pred_head,
        ])
    if args.augment_mode == 'idf_dual_projector':
        model.init_extra_weights([
            model.fuse_gate,
            model.g_inv,
            model.g_dyn,
            model.f_inv,
            model.f_dyn,
        ])
    if args.augment_mode == 'idf_dual_projector_mlp':
        model.init_extra_weights([
            model.retrieved_x_encoder_mlp,
            model.fuse_gate,
            model.g_inv,
            model.g_dyn,
            model.f_inv,
            model.f_dyn,
        ])
elif args.model == 'Moirai2Retrieve':
    # Backbone (Salesforce/moirai-2.0-R-small) is loaded and frozen inside the
    # model class itself (requires_grad=False set in __init__); no checkpoint
    # load / init_extra_weights needed here, unlike ChronosBoltRetrieve.
    # args.augment_mode selects which fusion head to attach on top of the
    # frozen backbone -- both output a point forecast (L2 loss), unlike the
    # Chronos-Bolt path's 9-quantile heads.
    if args.augment_mode == 'idf_clean_dis':
        model = Moirai2ModelForForecastingWithRetrieval(
            context_length=args.context_length,
            prediction_length=args.prediction_length,
        )
    elif args.augment_mode == 'moe':
        model = Moirai2MoEModelForForecastingWithRetrieval(
            context_length=args.context_length,
            prediction_length=args.prediction_length,
        )
    else:
        raise ValueError(
            f"Moirai2Retrieve only supports augment_mode in ['idf_clean_dis', 'moe'], got {args.augment_mode!r}"
        )
elif args.model == 'MOMENTRetrieve':
    MOMENT_MODEL_PATH = "AutonLab/MOMENT-1-large"
    model = MOMENTPipelineWithRetrieval.from_pretrained(MOMENT_MODEL_PATH,
                                           model_kwargs={
                                               'task_name': 'forecasting',
                                               'forecast_horizon': 64,
                                           })
    model.init()
    if 'moe' in args.augment_mode:
        model.init_extra_weights([model.encode_mlp, model.mha, model.ffn, model.gate_layer, model.project_before_fusion, model.project_after_fusion])
    criterion = nn.MSELoss().to(device)
else:
    print('model error')
    exit()
print(f'{args.model} model loaded')

model.to(device)
if args.use_multi_gpu:
    args.devices = [int(i) for i in args.devices.split(',')]
    model = nn.DataParallel(model, device_ids=args.devices)
    
params = model.parameters()

if args.optimizer == 'adam':
    model_optim = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)
elif args.optimizer == 'adamw':
    model_optim = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)

# freeze params
if args.freeze_chronos_bolt:
    layers_to_unfreeze = ['gate_layer', 'encode_mlp', 'mha', 'ffn']
    if args.model == 'Moirai2Retrieve' and args.augment_mode == 'moe':
        # Moirai2MoEModelForForecastingWithRetrieval adds its own point-forecast
        # head (point_pred_head) instead of reusing a native backbone head, unlike
        # the Chronos-Bolt 'moe' path -- the base layers_to_unfreeze list above
        # doesn't cover it.
        layers_to_unfreeze.append('point_pred_head')
    if args.augment_mode == 'moe3':
        if args.model == 'ChronosBoltRetrieve':
            layers_to_unfreeze.append('output_patch_embedding')
        elif args.model == 'MOMENTRetrieve':
            # import pdb; pdb.set_trace()
            layers_to_unfreeze.append('head')
    elif args.augment_mode == 'gate':
        layers_to_unfreeze.append('gate_linear1')
        layers_to_unfreeze.append('gate_linear2')
    elif args.augment_mode == 'moe_disentangle':
        layers_to_unfreeze.extend([
            'encode_mlp',
            'mha',
            'ffn',
            'gate_layer',
            'disentangle_gate',
            'final_pred_head',
        ])
    elif args.augment_mode == 'idf':
        layers_to_unfreeze.extend([
            'encode_mlp',
            'ret_score_head',
            'fuse_gate',
            'routing_gate',
            'inv_transition',
            'dyn_transition',
            'inv_head',
            'dyn_head',
            'final_head',
        ])
    elif args.augment_mode in ['idf_branch', 'idf_x']:
        layers_to_unfreeze.extend([
            'encode_mlp',
            'ret_score_head',
            'fuse_gate',
            'routing_gate',
            'inv_pred_head',
            'inv_residual_head',
            'dyn_pred_head',
            'final_pred_head',
        ])
    elif args.augment_mode == 'idf_clean_dis':
        layers_to_unfreeze.extend([
            'encode_mlp',
            'ret_score_head',
            'fuse_gate',
            'routing_gate',
            'inv_pred_head',
            'dyn_pred_head_clean',
            'final_pred_head',
        ])
    elif args.augment_mode == 'idf_clean_dis_deepmlp':
        layers_to_unfreeze.extend([
            'encode_mlp',
            'ret_score_head',
            'fuse_gate',
            'routing_gate',
            'inv_pred_head',
            'dyn_pred_head_clean',
            'final_pred_head',
        ])
    elif args.augment_mode == 'idf_clean_dis_ts3align':
        layers_to_unfreeze.extend([
            'encode_mlp',
            'fuse_gate',
            'routing_gate',
            'inv_pred_head',
            'dyn_pred_head_clean',
            'final_pred_head',
            'inv_aux_backproj',
            'dyn_aux_backproj',
        ])
    elif args.augment_mode == 'idf_h_linear_head':
        layers_to_unfreeze.extend([
            'encode_mlp',
            'ret_score_head',
            'fuse_gate',
            'h_pred_head',
        ])
    elif args.augment_mode == 'idf_h_native_head':
        layers_to_unfreeze.extend([
            'encode_mlp',
            'ret_score_head',
            'fuse_gate',
        ])
    elif args.augment_mode == 'idf_y_linear_head':
        layers_to_unfreeze.extend([
            'encode_mlp',
            'ret_score_head',
            'fuse_gate',
            'h_pred_head',
            'y_linear_head',
        ])
    elif args.augment_mode == 'idf_branch_gru':
        layers_to_unfreeze.extend([
            'encode_mlp',
            'ret_score_head',
            'fuse_gate',
            'routing_gate',
            'inv_pred_head',
            'inv_residual_head',
            'dyn_gru',
            'dyn_out_head',
            'final_pred_head',
        ])
    elif args.augment_mode == 'idf_branch_gru_q':
        layers_to_unfreeze.extend([
            'encode_mlp',
            'ret_score_head',
            'fuse_gate',
            'routing_gate',
            'inv_pred_head',
            'inv_residual_head',
            'dyn_gru',
            'dyn_out_head',
            'final_pred_head',
        ])
    elif args.augment_mode == 'idf_dual_direct_head':
        layers_to_unfreeze.extend([
            'encode_mlp',
            'ret_score_head',
            'fuse_gate',
            'routing_gate',
            'inv_hidden_proj',
            'dyn_ret_hidden_proj',
            'final_pred_head',
        ])
    elif args.augment_mode == 'idf_residual':
        layers_to_unfreeze.extend([
            'encode_mlp',
            'ret_score_head',
            'fuse_gate',
            'routing_gate',
            'inv_pred_head',
            'dyn_pred_head',
        ])
    elif args.augment_mode == 'idf_dual_projector':
        layers_to_unfreeze.extend([
            'fuse_gate',
            'g_inv',
            'g_dyn',
            'f_inv',
            'f_dyn',
        ])
    elif args.augment_mode == 'idf_dual_projector_mlp':
        layers_to_unfreeze.extend([
            'retrieved_x_encoder_mlp',
            'fuse_gate',
            'g_inv',
            'g_dyn',
            'f_inv',
            'f_dyn',
        ])

    for param in model.parameters():
        param.requires_grad = False
    # unfreeze the specified layers
    for name, param in model.named_parameters():
        param.requires_grad = any(layer in name for layer in layers_to_unfreeze)

trainable_param_names = [name for name, param in model.named_parameters() if param.requires_grad]
print('Trainable parameters:')
for name in trainable_param_names:
    print(name)

model.train()

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=args.tmax, eta_min=1e-8)

# retrieval already done, do not need to load the embedding model
embedding_model = None

# load retriever
retriever = Retriever_for_pretrain(
    retrieval_database_path=args.retrieval_database_path,
    dimension=768,
    embedding_model=embedding_model,
)
retriever.build_index()

## load data
dataset = CustomPretrainDataset(
    args.data_path, 
    retriever=retriever, 
    mode='training',
    drop_prob=args.drop_prob,
    context_length=args.context_length,
    prediction_length=args.prediction_length,
    retrieve_lookback_length=args.retrieve_lookback_length,
    top_k=args.top_k,
).shuffle(shuffle_buffer_length=args.shuffle_buffer_length)

train_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)


## train
iter_count = 0
train_loss = []
try:
    total_steps = min(len(train_loader), args.train_steps)
except TypeError:
    total_steps = args.train_steps
pbar = tqdm(enumerate(train_loader), total=total_steps)
for i, batch in pbar:
    if i >= args.train_steps:
        print('training finished')
        break
    model.train()
    iter_count += 1
    model_optim.zero_grad()
    retrieved_seqs = torch.tensor(retriever.whole_seq[batch['indices']])
    
    if not args.use_multi_gpu:
        batch['x'] = batch['x'].float().to(device)
        batch['y'] = batch['y'].float().to(device)
        batch['distances'] = batch['distances'].float().to(device)
        retrieved_seqs = retrieved_seqs.float().to(device)
    if args.model == 'ChronosBoltRetrieve':
        outputs = model(context = batch['x'].float(),
                        target = batch['y'].float(),
                        retrieved_seq = retrieved_seqs.float(),
                        distances = batch['distances'].float())                  # ChronosBoltOutput
    elif args.model == 'Moirai2Retrieve':
        outputs = model(context = batch['x'].float(),
                        target = batch['y'].float(),
                        retrieved_seq = retrieved_seqs.float(),
                        distances = batch['distances'].float())                  # Moirai2RiddeOutput / Moirai2MoEOutput
    elif args.model == 'MOMENTRetrieve':
        outputs = model(x_enc=batch['x'].float().unsqueeze(1), retrieved_seq=retrieved_seqs.float())
        outputs = outputs.forecast.squeeze(1)                                                     
        loss = criterion(outputs, batch['y'].float())
    else:
        print('model error')
    if args.model == 'MOMENTRetrieve':
        pass
    else:
        loss = outputs.loss
    loss = loss.mean()
    if args.model == 'ChronosBoltRetrieve':
        loss_forecast = outputs.loss_forecast.mean() if outputs.loss_forecast is not None else loss
        loss_cons = outputs.loss_cons.mean() if outputs.loss_cons is not None else loss.new_zeros(())
        loss_smooth = outputs.loss_smooth.mean() if outputs.loss_smooth is not None else loss.new_zeros(())
        loss_inv = outputs.loss_inv.mean() if outputs.loss_inv is not None else loss.new_zeros(())
        loss_dis = outputs.loss_dis.mean() if outputs.loss_dis is not None else loss.new_zeros(())
        loss_ret = outputs.loss_ret.mean() if outputs.loss_ret is not None else loss.new_zeros(())
        loss_dyn = outputs.loss_dyn.mean() if outputs.loss_dyn is not None else loss.new_zeros(())
        aux_loss_enabled = getattr(outputs, 'aux_loss_enabled', False)
    else:
        loss_forecast = loss
        loss_cons = loss.new_zeros(())
        loss_smooth = loss.new_zeros(())
        loss_inv = loss.new_zeros(())
        loss_dis = loss.new_zeros(())
        loss_ret = loss.new_zeros(())
        loss_dyn = loss.new_zeros(())
        aux_loss_enabled = False
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        log_payload = {
            'loss': loss.item(),
            'loss_forecast': loss_forecast.item(),
            'loss_cons': loss_cons.item(),
            'loss_smooth': loss_smooth.item(),
            'loss_inv': loss_inv.item(),
            'loss_dis': loss_dis.item(),
            'loss_ret': loss_ret.item(),
            'loss_dyn': loss_dyn.item(),
            'lr': model_optim.param_groups[0]['lr']
            }
        wandb.log(log_payload)

    postfix = {
        'total': round(loss.item(), 4),
        'f': round(loss_forecast.item(), 4),
    }
    if aux_loss_enabled:
        if args.augment_mode in ['idf_dual_projector', 'idf_dual_projector_mlp']:
            postfix.update({
                'cons': round(loss_cons.item(), 4),
                'smooth': round(loss_smooth.item(), 4),
                'dis': round(loss_dis.item(), 4),
                'dyn': round(loss_dyn.item(), 4),
            })
        else:
            postfix.update({
                'cons': round(loss_cons.item(), 4),
                'smooth': round(loss_smooth.item(), 4),
                'inv': round(loss_inv.item(), 4),
                'dis': round(loss_dis.item(), 4),
                'ret': round(loss_ret.item(), 4),
                'dyn': round(loss_dyn.item(), 4),
            })
    pbar.set_postfix(postfix)

    train_loss.append(loss.item())

    if (i + 1) % args.evaluation_steps == 0:
        print("\titers: {0} | loss: {1:.7f}".format(i + 1, sum(train_loss) / len(train_loss)))
        train_loss = []
        speed = (time.time() - time_now) / iter_count
        print('\tspeed: {:.4f}s/iter'.format(speed))
        iter_count = 0
        time_now = time.time()
        # save model and optimizer
        if (i + 1) < args.train_steps and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
            save_path = os.path.join(args.checkpoints, args.model_id)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(model.state_dict(), os.path.join(save_path,f'model_steps{i}.pth'))
            torch.save(model_optim.state_dict(), os.path.join(save_path, f'optim_steps{i}.pth'))

        # adjust learning rate
        scheduler.step()
        print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))

    loss.backward()
    clip_grad_norm_(model.parameters(), args.grad_clip_value)
    model_optim.step()
save_path = os.path.join(args.checkpoints, f"{args.model_id}_final.pth")
torch.save(model.state_dict(), save_path)
print(f"✅ IDF checkpoint saved to {save_path}")
                
