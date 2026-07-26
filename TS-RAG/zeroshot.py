import os
import time
import torch
import random
import argparse
import warnings
import numpy as np

from transformers import AutoConfig
from chronos import ChronosPipeline
from sklearn.preprocessing import StandardScaler

try:
    from models.moment import MOMENTPipelineWithRetrieval
except ModuleNotFoundError:
    # MOMENT support depends on an autogluon.timeseries internal path
    # (models.gluonts.abstract_gluonts) that moved/was removed in newer
    # autogluon releases. Defer the failure until a MOMENT model is actually
    # requested, so non-MOMENT runs (e.g. ChronosBoltRetrieve) aren't blocked.
    MOMENTPipelineWithRetrieval = None
from utils.tools import test, test_retrieve
from retrieve import do_retrieve, load_database, frequency_dict
from data_provider.data_factory import data_provider
from models.ChronosBolt import ChronosBoltPipeline, ChronosBoltModelForForecastingWithRetrieval
from models.Moirai2 import Moirai2ModelForForecastingWithRetrieval, Moirai2MoEModelForForecastingWithRetrieval
warnings.filterwarnings('ignore')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='Chronos-bolt')

parser.add_argument('--model_id', type=str, required=True, default='test')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

parser.add_argument('--root_path', type=str, default='./dataset/traffic/')
parser.add_argument('--data_path', type=str, default='traffic.csv')
parser.add_argument('--data', type=str, default='custom')
parser.add_argument('--features', type=str, default='M')
parser.add_argument('--freq', type=int, default=1)
parser.add_argument('--target', type=str, default='OT')
parser.add_argument('--embed', type=str, default='timeF')
parser.add_argument('--percent', type=int, default=10)
parser.add_argument('--all', type=int, default=0)

parser.add_argument('--seq_len', type=int, default=512)
parser.add_argument('--pred_len', type=int, default=96)
parser.add_argument('--label_len', type=int, default=48)

parser.add_argument('--decay_fac', type=float, default=0.75)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--train_epochs', type=int, default=10)
parser.add_argument('--patience', type=int, default=3)

parser.add_argument('--gpt_layers', type=int, default=3)
parser.add_argument('--is_gpt', type=int, default=1)
parser.add_argument('--e_layers', type=int, default=3)
parser.add_argument('--d_model', type=int, default=768)
parser.add_argument('--n_heads', type=int, default=16)
parser.add_argument('--d_ff', type=int, default=512)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--enc_in', type=int, default=862)
parser.add_argument('--c_out', type=int, default=862)
parser.add_argument('--patch_size', type=int, default=16)
parser.add_argument('--kernel_size', type=int, default=25)

parser.add_argument('--pretrain', type=int, default=1)
parser.add_argument('--model', type=str, default='model')
parser.add_argument('--stride', type=int, default=8)
parser.add_argument('--max_len', type=int, default=-1)
parser.add_argument('--hid_dim', type=int, default=16)
parser.add_argument('--tmax', type=int, default=20)

parser.add_argument('--cos', type=int, default=0)
parser.add_argument('--train_ratio', type=float, default=1.0 , required=False)
parser.add_argument('--save_file_name', type=str, default=None)
parser.add_argument('--gpu_loc', type=int, default=1)
parser.add_argument('--n_scale', type=float, default=-1)
parser.add_argument('--method', type=str, default='')

# retrieve
parser.add_argument('--embedding_tuning', type=str, default=None)
parser.add_argument('--metadata', type=dict, default={})
parser.add_argument('--metadata_database_name', type=str, default='ETTh2')
parser.add_argument('--metadata_frequency', type=str, default='hour')
parser.add_argument('--mode', type=str, default='only_self_train')
parser.add_argument('--top_k', type=int, default=1)
parser.add_argument('--retrieval_database_dir', type=str, default='../retrieval_database/')
parser.add_argument('--dimension', type=int, default=768)
parser.add_argument('--embedding_model_type', type=str, default='chronos')
parser.add_argument('--save', type=bool, default=True)
parser.add_argument('--lookback_length', type=int, default=512)

# augment
parser.add_argument('--augment_mode', type=str, default='moe2')
parser.add_argument('--debug_shapes', action='store_true', help='print key tensor shapes once for debug')
parser.add_argument('--eval_split', type=str, default='test', choices=['val', 'test'])
parser.add_argument('--rawx_norm', type=str, default='zscore', choices=['zscore', 'minmax'])
parser.add_argument('--retrieval_mode', type=str, default=None, choices=['embedding', 'raw_x'])
parser.add_argument('--tau', type=float, default=0.1)

parser.add_argument('--checkpoint_model_path', type=str, default='None')
parser.add_argument('--pretrained_model_path', type=str, default='./checkpoints/base')

args = parser.parse_args()
args.return_feature_id = 0

if args.save_file_name is not None : 
    log_fine_name = args.save_file_name

if torch.cuda.is_available():
    device_address = 'cuda:' + str(args.gpu_loc)
else:
    device_address = 'cpu'
print(f'使用设备: {device_address}')

SEASONALITY_MAP = {
   "minutely": 1440,
   "10_minutes": 144,
   "half_hourly": 48,
   "hourly": 24,
   "daily": 7,
   "weekly": 1,
   "monthly": 12,
   "quarterly": 4,
   "yearly": 1
}
mses = []
maes = []
print(args.model_id)

args.metadata['lookback_length'] = args.lookback_length
args.metadata['frequency'] = args.metadata_frequency
args.metadata['database_name'] = args.metadata_database_name.split(' ')
ori_data_path = args.data_path

if args.retrieval_mode is None:
    args.retrieval_mode = 'raw_x' if args.augment_mode == 'idf_x' else 'embedding'
print(f'augment_mode={args.augment_mode}, retrieval_mode={args.retrieval_mode}')

best_model_path = os.environ.get("CHECKPOINT_MODEL_PATH", args.checkpoint_model_path)
print(f'best_model_path: {best_model_path}')
if args.augment_mode != 'baseline' and not os.path.exists(best_model_path):
    exit('no corresponding checkpoint!!')
    
if args.freq == 0:
    args.freq = 'h'

if 'retrieve' in args.model_id:
    retrieval_database_names = '_'.join(args.metadata['database_name'])
    if args.retrieval_mode == 'raw_x':
        retrieved_filename = f'{ori_data_path.split(".")[0]}_retrieve_{retrieval_database_names}_{args.metadata["lookback_length"]}_{args.mode}_{args.embedding_tuning}_idf_x_{args.rawx_norm}.csv'
    else:
        retrieved_filename = f'{ori_data_path.split(".")[0]}_retrieve_{retrieval_database_names}_{args.metadata["lookback_length"]}_{args.mode}_{args.embedding_tuning}.csv'
    retrieved_data_path = os.path.join(args.root_path, retrieved_filename)
    if os.path.exists(retrieved_data_path):
        print(f'----------retrieval for {args.model_id} has done!!----------')
    else:
        print(f'----------retrieving for {args.model_id} ...----------')
        retrieval_type = 'rawx' if args.retrieval_mode == 'raw_x' else 'embedding'
        print(f'retrieval_type = {retrieval_type}')
        if args.retrieval_mode == 'embedding' and 'chronos' in args.embedding_model_type:
            if args.embedding_tuning == None:
                model_path = "amazon/chronos-t5-base"
            else: 
                model_path = f"../tuning_results/{args.metadata_database_name}_{str(args.seq_len)}_chronos_{args.embedding_tuning}"
                if not os.path.exists(model_path):
                    exit('embedding model path does not exist!!')
            embedding_model = ChronosPipeline.from_pretrained(
                model_path,
                device_map=device_address,
                torch_dtype=torch.bfloat16,
            )
        else:
            embedding_model = None
        top_k = args.top_k if args.top_k > 20 else 20
        do_retrieve(
            ori_data_path.split('.')[0],
            args.retrieval_database_dir,
            args.root_path,
            args.metadata,
            args.mode,
            top_k,
            args.seq_len,
            args.pred_len,
            fix_seed,
            args.dimension,
            embedding_model,
            args.save,
            args.embedding_tuning,
            retrieval_type=retrieval_type,
            rawx_norm=args.rawx_norm,
        )
    print('retrieved_data_path = {}'.format(retrieved_data_path))
    args.data_path = retrieved_data_path.split('/')[-1]

    # load retrieved raw data, it will be used to reconstruct the retrieved data
    retriever_rawdata = []
    if args.mode == 'only_self' or args.mode == 'only_self_train':
        # database: {var1: {}, var2: {}, ...}
        # retriever_rawdata: [var1_raw_data, var2_raw_data, ...]
        # Use the KB dataset's own frequency (not the query's) to resolve its
        # .pkl filename -- required when the KB dataset differs from the
        # query dataset and has a different sampling rate (e.g. querying an
        # hourly ETTh series against a minutely ETTm knowledge base).
        kb_name = args.metadata["database_name"][0]
        kb_frequency = frequency_dict.get(kb_name, args.metadata["frequency"])
        database = load_database(os.path.join(args.retrieval_database_dir, f'{kb_name}_{kb_frequency}_{args.metadata["lookback_length"]}.pkl'))
        for variable in database.keys():
            retriever_rawdata.append(database[variable]['raw_data'])
    elif args.mode == 'all_vars':
        for database_name in args.metadata['database_name']:
            kb_frequency = frequency_dict.get(database_name, args.metadata["frequency"])
            database = load_database(os.path.join(args.retrieval_database_dir, f'{database_name}_{kb_frequency}_{args.metadata["lookback_length"]}.pkl'))
            for variable in database.keys():
                retriever_rawdata.append(database[variable]['raw_data'])

    # scale transform for retrieved data
    scaler = StandardScaler()
    retriever_rawdata = np.array(retriever_rawdata).T
    scaler.fit(retriever_rawdata)
    retriever_rawdata = scaler.transform(retriever_rawdata)         #(n_samples, n_features)
    retriever_rawdata = retriever_rawdata.T
    eval_data, eval_loader = data_provider(args, args.eval_split, retriever_rawdata=retriever_rawdata)

else:
    eval_data, eval_loader = data_provider(args, args.eval_split)

print(f'eval_split = {args.eval_split}')

if args.freq != 'h':
    args.freq = SEASONALITY_MAP[eval_data.freq]
    print("freq = {}".format(args.freq))
device = torch.device(device_address)

time_now = time.time()

if args.model == 'ChronosBolt':
    model = ChronosBoltPipeline.from_pretrained(args.pretrained_model_path)
    # model.model.load_state_dict(torch.load(args.pretrained_model_path+'/autogluon_model.pth'))
    model.model.to(device)
elif args.model == 'ChronosBoltRetrieve':
    config = AutoConfig.from_pretrained(args.pretrained_model_path)
    if hasattr(config, "chronos_config"):
        config.chronos_config["context_length"] = args.seq_len
        config.chronos_config["prediction_length"] = args.pred_len
    model = ChronosBoltModelForForecastingWithRetrieval.from_pretrained(args.pretrained_model_path, config=config, augment=args.augment_mode,low_cpu_mem_usage=False)
    model.debug_shapes = args.debug_shapes
    model._debug_shapes_printed = False
    model.tau = args.tau
    if args.augment_mode != 'baseline':
        ckpt = torch.load(best_model_path, map_location="cpu")
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for key, value in ckpt.items():
            new_key = key.replace("module.", "")
            new_state_dict[new_key] = value
        msg = model.load_state_dict(new_state_dict, strict=False)
        print("Loaded IDF checkpoint:", best_model_path)
        print("Missing keys:", msg.missing_keys[:10])
        print("Unexpected keys:", msg.unexpected_keys[:10])
    else:
        print("Running baseline without retrieval checkpoint.")
    model.to(device)
elif args.model == 'Moirai2Retrieve':
    # Backbone (Salesforce/moirai-2.0-R-small) is loaded and frozen inside the
    # model class itself; args.pretrained_model_path is not read by either
    # variant. args.augment_mode selects which fusion head to use, mirroring
    # pretrain.py's Moirai2Retrieve branch.
    if args.augment_mode == 'idf_clean_dis':
        model = Moirai2ModelForForecastingWithRetrieval(
            context_length=args.seq_len,
            prediction_length=args.pred_len,
        )
    elif args.augment_mode == 'moe':
        model = Moirai2MoEModelForForecastingWithRetrieval(
            context_length=args.seq_len,
            prediction_length=args.pred_len,
        )
    else:
        raise ValueError(
            f"Moirai2Retrieve only supports augment_mode in ['idf_clean_dis', 'moe'], got {args.augment_mode!r}"
        )
    if args.augment_mode != 'baseline':
        ckpt = torch.load(best_model_path, map_location="cpu")
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for key, value in ckpt.items():
            new_key = key.replace("module.", "")
            new_state_dict[new_key] = value
        msg = model.load_state_dict(new_state_dict, strict=False)
        print("Loaded RIDDE checkpoint:", best_model_path)
        print("Missing keys:", msg.missing_keys[:10])
        print("Unexpected keys:", msg.unexpected_keys[:10])
    else:
        print("Running baseline without retrieval checkpoint.")
    model.to(device)
elif args.model == "MOMENTRetrieve":
    MOMENT_MODEL_PATH = "AutonLab/MOMENT-1-large"
    model = MOMENTPipelineWithRetrieval.from_pretrained(MOMENT_MODEL_PATH,
                                           model_kwargs={
                                               'task_name': 'forecasting',
                                               'forecast_horizon': 64,
                                           })
    model.init()
    #state_dict = torch.load(best_model_path)
    state_dict = torch.load(best_model_path, map_location='cpu')
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)
    model.to(device)
else:
    print('model error')
    exit()

print("------------------------------------")

if 'retrieve' in args.model_id:
    mse, mae = test_retrieve(model, eval_data, eval_loader, args, device)
else:
    mse, mae = test(model, eval_data, eval_loader, args, device)
mses.append(round(mse,5))
maes.append(round(mae,5))

if len(maes)==0 : exit()
maes = np.array(maes)
mses = np.array(mses)
print("mse_mean = {:.4f}, mse_std = {:.4f}".format(np.mean(mses), np.std(mses)))
print("mae_mean = {:.4f}, mae_std = {:.4f}".format(np.mean(maes), np.std(maes)))
print("MSE: {:.4f}".format(np.mean(mses)))
print("MAE: {:.4f}".format(np.mean(maes)))
    
log_dir = 'results/forecast_evaluation'
os.makedirs(log_dir, exist_ok=True)
file_path = os.path.join(log_dir, log_fine_name)

with open(file_path, 'a') as f : 
    f.write("{}\n".format(args.model_id))
    f.write("mse:{:.4f}, std:{:.4f} ---- mae:{:.4f}, std:{:.4f}\n".format(np.mean(mses), np.std(mses) , np.mean(maes), np.std(maes)))
        
print(log_fine_name)
            
