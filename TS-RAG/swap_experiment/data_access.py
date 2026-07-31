"""
Builds the exact same test-set Dataset objects (context/target/retrieved_seq)
that the RIDDE zeroshot eval pipeline (zeroshot.py + data_provider) uses, so
the branch-swap experiment reads real eval-time inputs instead of
reimplementing retrieval/scaling logic.

Mirrors zeroshot.py lines ~134-232 for the 'retrieve' + augment_mode=idf_clean_dis
path, but skips do_retrieve() (the *_retrieve_*.csv caches already exist) and
skips DataLoader construction -- callers index the Dataset directly for the
specific (channel, window) pairs picked by select_candidate_pairs.py.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

TS_RAG_DIR = "/home/fenglei/TS-RAG-main/TS-RAG"
if TS_RAG_DIR not in sys.path:
    sys.path.insert(0, TS_RAG_DIR)

from retrieve import load_database, frequency_dict  # noqa: E402
from data_provider.data_factory import data_dict  # noqa: E402

SEQ_LEN = 512
PRED_LEN = 64
LABEL_LEN = 0
TOP_K = 10
LOOKBACK_LENGTH = 512
RETRIEVAL_DATABASE_DIR = "/home/fenglei/TS-RAG-main/retrieval_database/"
MODE = "only_self_train"


@dataclass
class DatasetCfg:
    data: str
    root_path: str
    metadata_frequency: str
    raw_csv: str


DATASET_CONFIG = {
    "ETTh1": DatasetCfg("ett_h_retrieve", "/home/fenglei/TS-RAG-main/datasets/ETT-small/", "hour", "ETTh1.csv"),
    "ETTm1": DatasetCfg("ett_m_retrieve", "/home/fenglei/TS-RAG-main/datasets/ETT-small/", "minute", "ETTm1.csv"),
    "electricity": DatasetCfg("custom_retrieve", "/home/fenglei/TS-RAG-main/datasets/electricity/", "hour", "electricity.csv"),
    "ETTh2": DatasetCfg("ett_h_retrieve", "/home/fenglei/TS-RAG-main/datasets/ETT-small/", "hour", "ETTh2.csv"),
    "ETTm2": DatasetCfg("ett_m_retrieve", "/home/fenglei/TS-RAG-main/datasets/ETT-small/", "minute", "ETTm2.csv"),
    "weather": DatasetCfg("custom_retrieve", "/home/fenglei/TS-RAG-main/datasets/weather/", "10minutes", "weather.csv"),
    "exchange_rate": DatasetCfg("custom_retrieve", "/home/fenglei/TS-RAG-main/datasets/exchange_rate/", "hour", "exchange_rate.csv"),
    "traffic": DatasetCfg("custom_retrieve", "/home/fenglei/TS-RAG-main/datasets/traffic/", "hour", "traffic.csv"),
    "solar": DatasetCfg("custom_retrieve", "/home/fenglei/TS-RAG-main/datasets/solar/", "10minutes", "solar.csv"),
    "PEMS08": DatasetCfg("custom_retrieve", "/home/fenglei/TS-RAG-main/datasets/PEMS08/", "5minutes", "PEMS08.csv"),
    "AQWan": DatasetCfg("custom_retrieve", "/home/fenglei/TS-RAG-main/datasets/AQWan/", "hour", "AQWan.csv"),
    "Wind": DatasetCfg("custom_retrieve", "/home/fenglei/TS-RAG-main/datasets/Wind/", "15minutes", "Wind.csv"),
    "ILI": DatasetCfg("custom_retrieve", "/home/fenglei/TS-RAG-main/datasets/ILI/", "week", "ILI.csv"),
    "ZafNoo": DatasetCfg("custom_retrieve", "/home/fenglei/TS-RAG-main/datasets/ZafNoo/", "30minutes", "ZafNoo.csv"),
    "CzeLan": DatasetCfg("custom_retrieve", "/home/fenglei/TS-RAG-main/datasets/CzeLan/", "30minutes", "CzeLan.csv"),
}


def get_test_border(name: str, n: int) -> tuple[int, int]:
    """Mirrors data_provider/data_loader.py test-split borders (set_type=2).
    Must match select_candidate_pairs.get_test_border exactly (same formula)."""
    cfg = DATASET_CONFIG[name]
    if cfg.data == "ett_h_retrieve":
        border1 = 12 * 30 * 24 + 4 * 30 * 24 - SEQ_LEN
        border2 = 12 * 30 * 24 + 8 * 30 * 24
    elif cfg.data == "ett_m_retrieve":
        border1 = 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - SEQ_LEN
        border2 = 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4
    elif cfg.data == "custom_retrieve":
        num_test = int(n * 0.2)
        border1 = n - num_test - SEQ_LEN
        border2 = n
    else:
        raise ValueError(cfg.data)
    return border1, border2


class DatasetBundle:
    def __init__(self, name: str):
        self.name = name
        cfg = DATASET_CONFIG[name]
        self.cfg = cfg

        retrieved_filename = f"{name}_retrieve_{name}_{LOOKBACK_LENGTH}_{MODE}_None.csv"
        retrieved_data_path = os.path.join(cfg.root_path, retrieved_filename)
        assert os.path.exists(retrieved_data_path), retrieved_data_path

        kb_frequency = frequency_dict.get(name, cfg.metadata_frequency)
        database = load_database(os.path.join(RETRIEVAL_DATABASE_DIR, f"{name}_{kb_frequency}_{LOOKBACK_LENGTH}.pkl"))
        retriever_rawdata = [database[v]["raw_data"] for v in database.keys()]

        scaler = StandardScaler()
        retriever_rawdata = np.array(retriever_rawdata).T
        scaler.fit(retriever_rawdata)
        retriever_rawdata = scaler.transform(retriever_rawdata).T

        Data = data_dict[cfg.data]
        model_id = f"{name}_zeroshot_{SEQ_LEN}_pred_{PRED_LEN}_{LOOKBACK_LENGTH}_retrieve_{PRED_LEN}_idf_clean_dis"
        self.dataset = Data(
            model_id=model_id,
            root_path=cfg.root_path,
            data_path=retrieved_filename,
            flag="test",
            size=[SEQ_LEN, LABEL_LEN, PRED_LEN],
            features="M",
            target="OT",
            timeenc=1,  # matches zeroshot.py default (--embed=timeF -> timeenc=1)
            freq="h",  # matches zeroshot.py: --freq 0 is coerced to 'h' for all datasets
            percent=100,
            max_len=-1,
            train_all=False,
            top_k=TOP_K,
            retriever_rawdata=retriever_rawdata,
            mode=MODE,
        )

        df_raw = pd.read_csv(os.path.join(cfg.root_path, cfg.raw_csv))
        self.channels = [c for c in df_raw.columns if c != "date"]
        self.border1, self.border2 = get_test_border(name, len(df_raw))
        assert self.dataset.tot_len == (self.border2 - self.border1) - SEQ_LEN - PRED_LEN + 1

    def index_for(self, channel: str, abs_horizon_start_row: int) -> int:
        feat_id = self.channels.index(channel)
        s_begin = abs_horizon_start_row - self.border1 - SEQ_LEN
        assert 0 <= s_begin < self.dataset.tot_len, (
            f"{self.name}/{channel}: s_begin={s_begin} out of range [0, {self.dataset.tot_len})"
        )
        return feat_id * self.dataset.tot_len + s_begin

    def get_window(self, channel: str, abs_horizon_start_row: int):
        """Returns (seq_x [seq_len], seq_y [pred_len], retrieved_seq [top_k, seq_len+pred_len], distances [top_k])."""
        idx = self.index_for(channel, abs_horizon_start_row)
        seq_x, seq_y, seq_x_mark, timestamp_idx, retrieved_seqs, distances = self.dataset[idx]
        return (
            seq_x.squeeze(-1).astype(np.float32),
            seq_y.squeeze(-1).astype(np.float32),
            retrieved_seqs.numpy().astype(np.float32),
            np.asarray(distances, dtype=np.float32),
        )
