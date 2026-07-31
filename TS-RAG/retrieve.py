import os
import pdb
import math
import torch
import faiss
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from chronos import ChronosPipeline

from utils.tools import get_borders

frequency_dict = {'ETTh1': 'hour', 'ETTh2': 'hour', 'ETTm1': 'minute', 'ETTm2': 'minute',
                    'electricity': 'hour', 'weather': '10minutes', 'traffic': 'hour', 'exchange_rate': 'hour', 'illness': 'hour',
                    'solar': '10minutes', 'PEMS08': '5minutes', 'AQWan': 'hour', 'Wind': '15minutes', 'ILI': 'week',
                    'ZafNoo': '30minutes', 'CzeLan': '30minutes'}
subdir_name_dict = {'ETTh1': 'ETT-small', 'ETTh2': 'ETT-small', 'ETTm1': 'ETT-small', 'ETTm2': 'ETT-small',
                    'electricity': 'electricity', 'weather': 'weather', 'traffic': 'traffic', 'solar': 'solar',
                    'PEMS08': 'PEMS08', 'AQWan': 'AQWan', 'Wind': 'Wind', 'ILI': 'ILI', 'ZafNoo': 'ZafNoo',
                    'CzeLan': 'CzeLan'}

# Self-retrieval KB build density: by default every dataset embeds a dense,
# stride-1 sliding window over its full history (one window per timestep),
# which is what makes the electricity KB (321 channels) ~26GB on disk. traffic
# has 862 channels over a comparable history, so a dense KB would be ~1.8x
# that (~45-50GB); solar has 137 channels over a full year at 10min frequency,
# comparable to electricity's own size (~22GB); PEMS08 is small (170 channels,
# 17856 timestamps) at ~9GB dense, but disk headroom on this machine is tight
# (shared with other jobs) so we still shrink it. Override the stride
# per-dataset here to only embed every Nth position instead, shrinking KB
# size/build time by ~N with a correspondingly coarser retrieval granularity.
# Datasets not listed keep the original dense stride=1 behavior.
RETRIEVAL_STRIDE = {'traffic': 6, 'solar': 3, 'PEMS08': 2}


def normalize_windows(windows, method='zscore', eps=1e-8):
    windows = np.asarray(windows, dtype=np.float32)
    if method == 'zscore':
        mean = windows.mean(axis=-1, keepdims=True)
        std = windows.std(axis=-1, keepdims=True)
        std = np.where(std < eps, 1.0, std)
        normalized = (windows - mean) / std
    elif method == 'minmax':
        min_v = windows.min(axis=-1, keepdims=True)
        max_v = windows.max(axis=-1, keepdims=True)
        span = np.where((max_v - min_v) < eps, 1.0, (max_v - min_v))
        normalized = (windows - min_v) / span
    else:
        raise ValueError(f'Unsupported normalization method: {method}')

    norms = np.linalg.norm(normalized, axis=-1, keepdims=True)
    norms = np.where(norms < eps, 1.0, norms)
    return normalized / norms

def create_database(raw_data, timestamps, lookback_length, embedding_model, metadata, stride=1):
    embeddings = []
    sliced_timestamps = []

    # window start positions: stride=1 embeds every position (dense, original
    # behavior); stride>1 only embeds every `stride`-th position to shrink
    # both build time and on-disk KB size by ~stride. `timestamp_idx` values
    # produced later by Retriever.search() are scaled back up by this same
    # stride (see Retriever.strides) so they still resolve to correct raw
    # sequence positions despite the sparser embedding array.
    positions = list(range(0, len(raw_data) - lookback_length + 1, stride))

    # batch embedding
    # NOTE: kept in sync with the batch size used in do_retrieve()'s query-side
    # embedding loop -- 512 OOMs on a 24GB GPU (T5 self-attention softmax
    # briefly upcasts to float32), so use the same smaller batch + cache
    # clearing here.
    batch_size = 64
    num_batchs = (len(positions) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batchs)):
        batch_positions = positions[batch_idx * batch_size: (batch_idx + 1) * batch_size]

        # get batch data
        batch_slices = [raw_data[i:i + lookback_length] for i in batch_positions]
        batch_slices = torch.tensor(batch_slices)

        # embedding
        batch_embeddings, _ = embedding_model.embed(batch_slices)
        eos_embeddings = batch_embeddings[:, -1, :].float().numpy()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # append to embeddings
        embeddings.extend(eos_embeddings)
        sliced_timestamps.extend([timestamps[i + lookback_length - 1] for i in batch_positions])

    embeddings = np.array(embeddings)
    sliced_timestamps = np.array(sliced_timestamps)

    metadata = {**metadata, 'stride': stride}
    database = {
        'raw_data': raw_data,
        'timestamps': sliced_timestamps,
        'embeddings': embeddings,
        'metadata': metadata
    }
    return database

def save_database(database, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(database, f)

def load_database(file_path):
    with open(file_path, 'rb') as f:
        database = pickle.load(f)
    return database

def generate_retrieval_database(dataset_name, lookback_length, embedding_model, database_dir, root_dir):
    root_dir = Path(root_dir)
    database_dir = Path(database_dir)
    data_path = root_dir / (dataset_name+'.csv')
    frequency = frequency_dict[dataset_name]
    stride = RETRIEVAL_STRIDE.get(dataset_name, 1)
    if stride != 1:
        print(f'generate_retrieval_database: using stride={stride} for {dataset_name} '
              f'(~{stride}x smaller/faster KB than the dense stride=1 default)')
    df = pd.read_csv(data_path)
    variables = df.columns[1:]

    databases = {}
    for variable in variables:
        raw_data = df[variable].tolist()
        timestamps = df['date'].tolist()
        metadata = {
            'dataset_name': dataset_name,
            'variable_name': variable,
            'lookback_length': lookback_length,
            'frequency': frequency,
            }
        database = create_database(raw_data, timestamps, lookback_length, embedding_model, metadata, stride=stride)
        databases[variable] = database

    save_database(databases, os.path.join(database_dir, f'{dataset_name}_{frequency}_{lookback_length}.pkl'))
    

class Retriever():
    def __init__(self, database_dir, root_dir, metadata, seed, dimension, embedding_model, embedding_tuning):
        self.database_dir = database_dir
        self.metadata = metadata
        self.d = dimension #768
        self.index = None
        self.Y = None
        self.seed = seed
        self.embedding_model = embedding_model
        self.root_dir = root_dir
        self.embedding_tuning = embedding_tuning

    def build_index(self, y_length, begin=None, end=None, variable_filter=None):
        self.raw_data = []
        self.retrieved_metadata = []
        self.timestamps = []
        self.boundary = [0]
        # per-segment build stride and slice offset (parallel to the
        # per-segment boundary sizes appended below), used in search() to
        # convert FAISS index positions back into raw sequence positions:
        # raw_position = (embed_begin + local_row_idx) * stride.
        self.strides = []
        self.embed_begins = []
        self.index = faiss.IndexFlatL2(self.d)  # euclidean distance

        database_paths = []
        for database_name in self.metadata['database_name']:
            # Use this source database's OWN frequency (not the query
            # dataset's) to build its .pkl filename — required for
            # cross-domain KBs where a merged database's sampling rate
            # differs from the query's (e.g. querying ETTh1 but merging in
            # ETTm1/ETTm2, which are minute-frequency, not hour-frequency).
            database_frequency = frequency_dict.get(database_name, self.metadata['frequency'])
            database_path = f'{database_name}_{database_frequency}_{self.metadata["lookback_length"]}.pkl'
            if self.embedding_tuning:
                self.database_dir = os.path.join(self.database_dir, self.embedding_tuning)
            if not os.path.exists(self.database_dir):
                print(f'{self.database_dir} does not exist, building the dir...')
                os.makedirs(self.database_dir)
            if os.path.exists(os.path.join(self.database_dir, database_path)):
                database_paths.append(database_path)
            else:
                print(f'{database_path} does not exist, building the database...')
                generate_retrieval_database(dataset_name=database_name, lookback_length=self.metadata['lookback_length'], embedding_model=self.embedding_model, database_dir=self.database_dir, root_dir=self.root_dir)
                database_paths.append(database_path)
        
        print(f'Build index with database: {database_paths}')

        # load database
        for database_name, database_path in zip(self.metadata['database_name'], database_paths):
            print(f'load database: {database_path}')
            database = load_database(os.path.join(self.database_dir, database_path))
            # filter by metadata['variable']
            for key in database.keys():
                if variable_filter is None or key in variable_filter:
                    embeddings = database[key]['embeddings']
                    embeddings = embeddings.reshape(-1, self.d).astype('float32')  # reshape to (n, d)
                    stride = database[key]['metadata'].get('stride', 1)

                    # filter embeddings: when begin/end are not explicitly given,
                    # slice by THIS source database's own train boundary rather
                    # than the query dataset's, so cross-domain knowledge bases
                    # (database_name != query dataset) use their own correct
                    # train split instead of an unrelated, possibly out-of-range
                    # boundary borrowed from the query dataset.
                    if begin is None or end is None:
                        src_border1s, src_border2s = get_borders(
                            database_name, self.metadata['lookback_length'], len(database[key]['raw_data'])
                        )
                        filter_begin = src_border1s[0] if begin is None else begin
                        filter_end = src_border2s[0] if end is None else end
                    else:
                        filter_begin = begin
                        filter_end = end

                    # filter_begin/filter_end are in raw-sequence units; the
                    # embeddings array is indexed in stride units (row j ==
                    # raw position j*stride when stride>1), so rescale.
                    embed_begin = -(-filter_begin // stride)  # ceil division
                    embed_end = -(-filter_end // stride)
                    embeddings = embeddings[embed_begin:embed_end, :]

                    self.index.add(embeddings)

                    self.timestamps.append(database[key]['timestamps'])
                    self.raw_data.append(database[key]['raw_data'])
                    self.retrieved_metadata.append(database[key]['metadata'])
                    self.boundary.append(embeddings.shape[0])
                    self.strides.append(stride)
                    self.embed_begins.append(embed_begin)

        self.boundary = [sum(self.boundary[:i]) for i in range(1, len(self.boundary)+1)]

    def search(self, query_vector, top_k, drop_first=False, params=None):
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        # drop first or last
        if params is None:
            distances, indices = self.index.search(query_vector, top_k + 1)
        else:
            distances, indices = self.index.search(query_vector, top_k + 1, params=params)
        if drop_first:
            distances = distances[:, 1:]
            indices = indices[:, 1:]
        else:
            distances = distances[:, :-1]
            indices = indices[:, :-1]
        
        # find boundary_idx and timestamp_idx
        boundary_idx_batch = np.digitize(indices, self.boundary) - 1
        local_row_idx_batch = indices - np.array(self.boundary)[boundary_idx_batch]
        # local_row_idx_batch is 0-based within this segment's (possibly
        # strided) embeddings slice; convert back to a raw sequence position.
        # For the default stride=1, embed_begins is always 0 (see build_index),
        # so this reduces to the original `indices - boundary_offset` value.
        embed_begin_batch = np.array(self.embed_begins)[boundary_idx_batch]
        stride_batch = np.array(self.strides)[boundary_idx_batch]
        timestamp_idx_batch = (local_row_idx_batch + embed_begin_batch) * stride_batch
        # if np.any(timestamp_idx_batch < 0):
        #     print('timestamp_idx_batch has negative values')
        #     pdb.set_trace()

        return distances, boundary_idx_batch, timestamp_idx_batch

def do_retrieve(original_data_name, retrieval_database_dir, root_dir, metadata, mode, top_k, context_length, prediction_length, seed, dimension, embedding_model, save=True, embedding_tuning=None, retrieval_type='embedding', rawx_norm='zscore'):
    '''
    input: the original data, retrieval database, metadata and retrieve mode
    output: retrieved_data
    '''
    # load original data
    original_data_path = os.path.join(root_dir, original_data_name+'.csv')
    original_data = pd.read_csv(original_data_path)
    variable_names = original_data.columns[1:]  # exclude 'date'
    print(f'There are {len(variable_names)} variables in the original data')

    # initialize the retrieved data
    boundary_idx_matrix = np.full((len(original_data), len(variable_names), top_k), np.nan)
    timestamp_idx_matrix = np.full((len(original_data), len(variable_names), top_k), np.nan)
    distance_matrix = np.full((len(original_data), len(variable_names), top_k), np.nan)

    # get borders
    border1s, border2s = get_borders(original_data_name, context_length, len(original_data))

    if mode == 'only_self_train':
        print(f'----------For one variable, we retrieve data from itself training set----------')
        # each variable retrieve from historical data of itself
        for var_idx, var_name in enumerate(variable_names):
            print(f'----------Retrieving for variable: {var_name}')
            retriever = Retriever(database_dir=retrieval_database_dir, metadata=metadata, seed=seed, dimension=dimension, embedding_model=embedding_model, root_dir=root_dir, embedding_tuning=embedding_tuning)

            # Each source database (self or, for cross-domain KBs, a different
            # dataset) is sliced by its own train boundary inside build_index;
            # border1s/border2s here only gate whether the QUERY window itself
            # falls in train (skip retrieval) below.
            retriever.build_index(y_length=prediction_length, variable_filter=[var_name])
            
            sequence = original_data[var_name].values

            ## batch search
            start_idx_list = list(range(0, len(sequence) - context_length - prediction_length + 1))
            end_idx_list = [start_idx + context_length for start_idx in start_idx_list]
            
            # get batch of start_idx and end_idx
            # NOTE: the embedding model's self-attention softmax briefly
            # upcasts scores to float32, so batch_size=512 at seq_len=512 can
            # allocate several GiB for a single batch; combined with CUDA
            # allocator fragmentation across many batches this can OOM on a
            # 24GB GPU when the retrieval CSV isn't already cached (i.e. the
            # embeddings actually have to be computed, as opposed to loading
            # a precomputed *_retrieve_*.csv from disk). Use a smaller batch
            # and release cached (but unused) allocator memory each batch.
            search_batch_size = 64
            batch_num = math.ceil(len(start_idx_list) / search_batch_size)
            for batch_idx in tqdm(range(batch_num)):
                start_idx_batch = start_idx_list[batch_idx*search_batch_size:min((batch_idx+1)*search_batch_size, len(start_idx_list))]
                end_idx_batch = end_idx_list[batch_idx*search_batch_size:min((batch_idx+1)*search_batch_size, len(start_idx_list))]
                
                # for training and validation set, do not need to search
                if end_idx_batch[-1] <= border2s[0]:
                    boundary_idx_batch = np.zeros((len(start_idx_batch), top_k))
                    timestamp_idx_batch = np.zeros((len(start_idx_batch), top_k))
                    distance_batch = np.zeros((len(start_idx_batch), top_k))
                    boundary_idx_matrix[start_idx_batch, var_idx, :] = boundary_idx_batch
                    timestamp_idx_matrix[start_idx_batch, var_idx, :] = timestamp_idx_batch
                    distance_matrix[start_idx_batch, var_idx, :] = distance_batch

                else:
                    seq_x_batch = np.array([sequence[start_idx:end_idx] for start_idx, end_idx in zip(start_idx_batch, end_idx_batch)])
                    if retrieval_type == 'embedding':
                        query_vector_batch, _ = embedding_model.embed(torch.tensor(seq_x_batch))
                        query_vector_batch = query_vector_batch[:,-1,:].squeeze().float().numpy()
                        distances_batch, boundary_idx_batch, timestamp_idx_batch = retriever.search(query_vector_batch, top_k=top_k)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    elif retrieval_type == 'rawx':
                        query_norm = normalize_windows(seq_x_batch, method=rawx_norm)
                        train_sequence = np.asarray(retriever.raw_data[0], dtype=np.float32)
                        train_begin = border1s[0]
                        train_end = border2s[0]
                        candidate_windows = np.array([
                            train_sequence[idx:idx + context_length]
                            for idx in range(train_begin, train_end - context_length - prediction_length + 1)
                        ], dtype=np.float32)
                        candidate_norm = normalize_windows(candidate_windows, method=rawx_norm)
                        sims = query_norm @ candidate_norm.T
                        kth = min(top_k - 1, sims.shape[1] - 1)
                        top_idx = np.argpartition(-sims, kth=kth, axis=1)[:, :top_k]
                        top_sims = np.take_along_axis(sims, top_idx, axis=1)
                        order = np.argsort(-top_sims, axis=1)
                        top_idx = np.take_along_axis(top_idx, order, axis=1)
                        top_sims = np.take_along_axis(top_sims, order, axis=1)
                        distances_batch = 1.0 - top_sims
                        boundary_idx_batch = np.zeros_like(top_idx)
                        timestamp_idx_batch = top_idx + train_begin
                    else:
                        raise ValueError(f'Unsupported retrieval_type: {retrieval_type}')
                    boundary_idx_matrix[start_idx_batch, var_idx, :] = boundary_idx_batch
                    timestamp_idx_matrix[start_idx_batch, var_idx, :] = timestamp_idx_batch
                    distance_matrix[start_idx_batch, var_idx, :] = distances_batch
            # import pdb; pdb.set_trace()
            
    boundary_idx_df = pd.DataFrame(boundary_idx_matrix.reshape(len(original_data), -1), columns=[f'boundary_idx_{var}_{k}' for var in variable_names for k in range(top_k)])
    timestamp_idx_df = pd.DataFrame(timestamp_idx_matrix.reshape(len(original_data), -1), columns=[f'timestamp_idx_{var}_{k}' for var in variable_names for k in range(top_k)])
    distance_df = pd.DataFrame(distance_matrix.reshape(len(original_data), -1), columns=[f'distance_{var}_{k}' for var in variable_names for k in range(top_k)])
    retrieved_data = pd.concat([original_data, boundary_idx_df, timestamp_idx_df, distance_df], axis=1)

    assert (pd.concat([boundary_idx_df.isna().sum().reset_index(drop=True), 
                   timestamp_idx_df.isna().sum().reset_index(drop=True), 
                   distance_df.isna().sum().reset_index(drop=True)], axis=1).nunique(axis=1) == 1).all(), "NaN counts are not the same in all columns"
    
    if save:
        retrieval_database_names = '_'.join(metadata['database_name'])
        retrieval_suffix = ''
        if retrieval_type == 'rawx':
            retrieval_suffix = f'_idf_x_{rawx_norm}'
        retrieved_data_path = os.path.join(root_dir, f'{original_data_name}_retrieve_{retrieval_database_names}_{metadata["lookback_length"]}_{mode}_{embedding_tuning}{retrieval_suffix}.csv')
        print(f'Saving the retrieved data to {retrieved_data_path}')
        retrieved_data.to_csv(retrieved_data_path, index=False)

    return retrieved_data


if __name__ == "__main__":
    if 1:
        original_data_path = './datasets/ETT-small/ETTh1.csv'
        original_data_name = 'ETTh1'
        retrieval_database_dir = '../retrieval_database/'
        root_dir = './datasets/ETT-small'
        metadata = {
            'database_name': ['ETTh2'],
            'lookback_length': 96,
            'frequency': 'hour',
        }
        mode = 'only_self'
        dimension = 768
        save = True
        seed = 42
        top_k = 20
        context_length = 96
        prediction_length = 96

        embedding_model = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-base",
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )

        do_retrieve(original_data_name, retrieval_database_dir, root_dir, metadata, mode, top_k, context_length, prediction_length, seed, dimension, embedding_model, save)
    
    if 0:
        retrieval_database_dir = '../retrieval_database/'
        root_dir = './datasets/ETT-small'
        metadata = {
            'database_name': ['ETTh2'],
            'lookback_length': 96,
            'frequency': 'hour',
        }
        seed = 42
        dimension = 768
        embedding_model = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-base",
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )
        retriever = Retriever(database_dir=retrieval_database_dir, root_dir=root_dir, metadata=metadata, seed=seed, dimension=dimension, embedding_model=embedding_model)
        retriever.build_index(y_length=96)

        query_vector = torch.randn(32, 768)
        distances, boundary_idx_list, timestamp_idx_list = retriever.search(query_vector, top_k=20)
        pdb.set_trace()
        print('done')
