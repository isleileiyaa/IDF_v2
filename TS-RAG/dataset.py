import torch
import faiss
import numpy as np
import pandas as pd
from pathlib import Path

from gluonts.itertools import Cyclic
from torch.utils.data import IterableDataset
from gluonts.dataset.common import FileDataset

# RIDDE_新版目标函数与最终实验方案 Stage 4 (5.2/5.3节)：pretrain_pairs_ctx512/ 下30个
# chunk 分片里，chunk-014(7757行)、chunk-029(6223行) 比其余~28个分片(每个约100万行)
# 小两个数量级，是随机切分后的尾部余量，不能跟其余分片当成同等规模的"环境"，默认排除。
STAGE4_TINY_CHUNK_BASENAMES = frozenset({
    'chronos-dataset-sample-50m-chunk-014.parquet',
    'chronos-dataset-sample-50m-chunk-029.parquet',
})


class PseudoShuffledIterableDataset(IterableDataset):
    """
    Shuffle entries from an iterable by temporarily accumulating them
    in an intermediate buffer.

    Parameters
    ----------
    base_dataset
        The original iterable object, representing the dataset.
    shuffle_buffer_length
        Size of the buffer use to shuffle entries from the base dataset.
    """

    def __init__(self, base_dataset, shuffle_buffer_length: int = 100) -> None:
        super().__init__()
        self.base_dataset = base_dataset
        self.shuffle_buffer_length = shuffle_buffer_length
        self.generator = torch.Generator()

    def __iter__(self):
        shuffle_buffer = []

        for element in self.base_dataset:
            shuffle_buffer.append(element)
            if len(shuffle_buffer) >= self.shuffle_buffer_length:
                idx = torch.randint(
                    len(shuffle_buffer), size=(), generator=self.generator
                )
                yield shuffle_buffer.pop(idx)

        while shuffle_buffer:
            idx = torch.randint(len(shuffle_buffer), size=(), generator=self.generator)
            yield shuffle_buffer.pop(idx)


class ShuffleMixin:
    """
    Mix-in class that datasets can inherit from to get
    shuffling functionality.
    """

    def shuffle(self, shuffle_buffer_length: int = 100):
        return PseudoShuffledIterableDataset(self, shuffle_buffer_length)


class CustomPretrainDataset(IterableDataset, ShuffleMixin):
    def __init__(
        self,
        dataset_path,
        retriever,
        mode="training",
        drop_prob=0.2,
        context_length=512,
        prediction_length=64,
        retrieve_lookback_length=64,
        top_k=5,
    ):
        super().__init__()

        assert mode in ("training", "validation", "test")

        self.drop_prob = drop_prob
        self.dataset_path = Path(dataset_path)
        self.mode = mode
        self.retriever = retriever
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.retrieve_lookback_length = retrieve_lookback_length
        self.top_k = top_k

        # Ensure the dataset path exists
        if not self.dataset_path.is_dir():
            raise ValueError(f"Provided dataset_path {dataset_path} is not a directory.")
        
        # check files, all should be parquet
        if not all([f.suffix == ".parquet" for f in self.dataset_path.iterdir()]):
            raise ValueError("All files in the dataset_path should be parquet files.")

        # lazy loading
        self.dataset = FileDataset(self.dataset_path, freq="1H")

        if self.mode == "training":
            self.dataset = Cyclic(self.dataset)


    def __iter__(self):
        iterable = iter(self.dataset)
        if self.mode == "training":
            while True:
                entry = next(iterable)
                entry = {f: entry[f] for f in ['target', 'distances', 'indices']}
                entry['x'] = entry['target'][:self.context_length]
                entry['y'] = entry['target'][self.context_length:]
                entry['distances'] = entry['distances'][:self.top_k]
                entry['indices'] = entry['indices'][:self.top_k]

                if self.drop_prob > 0:
                    target = entry['target'].copy()
                    drop_p = np.random.uniform(low=0.0, high=self.drop_prob)
                    mask = np.random.choice(
                        [True, False], size=len(target), p=[drop_p, 1 - drop_p]
                    )
                    target[mask] = np.nan
                    entry['target'] = target
                yield entry

        else:
            for entry in iterable:
                entry = {f: entry[f] for f in ['target', 'distances', 'indices']}
                entry['x'] = entry['target'][:self.context_length]
                entry['y'] = entry['target'][self.context_length:]
                yield entry


class Retriever_for_pretrain():
    def __init__(self, retrieval_database_path, dimension, embedding_model):
        self.retrieval_database_path = retrieval_database_path
        self.d = dimension #768
        self.index = None
        self.Y = None
        self.embedding_model = embedding_model

    def build_index(self):
        self.index = faiss.IndexFlatL2(self.d)  # euclidean distance

        database = pd.read_parquet(self.retrieval_database_path)
        embeddings = np.vstack(database["embedding"].to_numpy())
        self.x = database['x'].values
        self.y = database['y'].values
        self.whole_seq = np.concatenate([self.x.tolist(), self.y.tolist()], axis=-1)
        self.index.add(embeddings)

    def embedding(self, x_tensor):
        embeddings, _ = self.embedding_model.embed(x_tensor)
        return embeddings[:, -1, :].float().numpy()

    def search(self, query_vector, top_k, params=None):
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        # drop first or last
        if params is None:
            distances, indices = self.index.search(query_vector, top_k + 1)
        else:
            distances, indices = self.index.search(query_vector, top_k + 1, params=params)
        # drop first if first distance is 0
        mask = distances[:, 0] == 0
        distances = np.where(
            mask[:, None],
            distances[:, 1:], 
            distances[:, :-1]
        )
        indices = np.where(
            mask[:, None],
            indices[:, 1:],
            indices[:, :-1]
        )

        return indices, distances


class EnvironmentBalancedPretrainDataset(IterableDataset):
    """
    RIDDE_新版目标函数与最终实验方案 Stage 4 (5.2/5.3节)：每个训练 step 从 G 个"环境"
    里各抽 m 个样本拼成一个大小 G*m 的 batch，这样外面(pretrain.py)按 batch 里带的
    env_id 分组，一次 forward 就能同时估计这 G 个环境各自的 R_e^ret/R_e^0，进而算
    Δ_e、r_e 和 CVaR(L_TRR)，不需要另外写 batch_sampler。

    用法要求：DataLoader 必须用 batch_size = env_group_size * env_batch_size，
    且不能再套一层 shuffle（IterableDataset 本来也不支持 DataLoader 级别的
    shuffle=True）——本类自己在 __iter__ 里控制"每 env_batch_size 个连续样本属于
    同一个环境、每 env_group_size 段换一次环境"的结构，DataLoader 只是照顺序取。

    !!!环境定义的说明，如实记录（2026-09-09 更新：重叠窗口泄露诊断结果）!!!
    方案文档5.2节要求环境由源域训练数据的连续时间块构造。实际检查过
    datasets/pretrain/pretrain_pairs_ctx512/ 下的数据后发现，"chronos-dataset-
    sample-50m"这份语料每一行的 start 字段全部是统一的 1970-01-01 占位值，没有真实
    时间戳，也没有 item_id 之类可以标识"属于哪个原始序列"的字段；往上追溯到
    build_pretrain_pairs_rawx.py 也证实这个占位值是从更上游的输入文件透传过来的，不是
    这一步引入的 bug——所以按方案5.2节字面要求的"连续时间块"目前无法构造。

    但针对"这30个chunk分片本身是怎么切出来的"做过额外的抽样诊断（每个chunk内部相邻行
    的滑窗重合比例、以及6个chunk之间样本级重叠情况），结果是：
      - chunk内部相邻行滑窗重合比例 99.3%~100%（30个chunk全部如此）——chunk内部
        完整保留了原始序列的滑窗顺序，没有在单窗口粒度上打乱。
      - 抽查的6个chunk之间（约1800行样本）未发现任何跨chunk的近乎重复窗口。
    这两条证据一致指向：30个chunk之间的切分发生在"原始序列/来源对象"这个粒度上（整条
    序列被完整分配到某一个chunk），而不是把有重叠的滑窗样本打散后随机分配到不同chunk。
    也就是说，chunk之间**不存在重叠窗口泄露**——虽然仍不知道chunk之间有没有时间先后
    关系，但chunk作为"环境"至少是干净、互不重叠的固定分组，不是退化的随机负对照。
    （注：这是6/30个chunk、每chunk约300行的抽样结果，不是全量穷举。）

    结论：本类目前用"哪个 parquet 分片文件"代替"时间环境"，准确的说法是**"序列/来源
    对象级别的固定环境划分"**，不是方案7.5节里作为反面对照组的"random blocks"（因为
    没有发现随机打散重叠窗口的证据），但也还不是"chronological blocks"完整方法（因为
    chunk之间目前没有已知的时间顺序）。L_TRR/CVaR的公式本身对"环境具体是什么"没有要求，
    只要求环境是训练前定死、互不泄露的固定分组——这一点现在的chunk划分是满足的，所以
    不需要改动模型/损失代码，只需要在汇报结果时如实说明"环境"的定义是序列/来源对象级，
    不是时间连续块。如果后续需要真正的时间环境，仍然要往上游追更原始的数据源，这是另一
    件独立的数据侧工作，不在本次改动范围内。
    """

    def __init__(
        self,
        env_files,
        drop_prob=0.2,
        context_length=512,
        prediction_length=64,
        top_k=5,
        env_group_size=8,
        env_batch_size=32,
        env_shuffle_buffer_length=2000,
        max_resident_envs=None,
    ):
        super().__init__()
        self.env_files = [Path(f) for f in env_files]
        self.num_envs = len(self.env_files)
        assert self.num_envs >= 2, (
            f"至少需要2个环境才能算CVaR，当前只传入了 {self.num_envs} 个环境文件"
        )
        assert env_group_size <= self.num_envs, (
            f"env_group_size({env_group_size}) 不能超过环境总数({self.num_envs})"
        )
        self.drop_prob = drop_prob
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.top_k = top_k
        self.env_group_size = env_group_size
        self.env_batch_size = env_batch_size
        self.env_shuffle_buffer_length = env_shuffle_buffer_length
        # 冒烟测试暴露的第二个问题：改用 pyarrow 流式读取后单个环境的常驻内存从
        # ~16GB 降到 ~1GB 量级，但 __iter__ 原来会把 self.num_envs(比如28)个
        # 环境的 stream 全部创建出来且永久留着——训练跑起来后，随机采样几十步
        # 就能把28个环境基本都"激活"一遍，每个环境的 pyarrow 流式reader+shuffle
        # buffer 都不会被释放，实测20步冒烟测试就把这台机器61GB内存吃到只剩
        # 327MB，再多跑几步大概率还是会被 OOM 杀掉。这里用一个有上限的 LRU 缓存
        # 限制"同时活着"的环境数量，超过上限就淘汰最久未用的环境(整个 generator
        # 丢弃，内存跟着回收)，需要时重新从头 warm-up(pyarrow流式读取现在很快，
        # 重新warm-up的代价可以接受)。默认上限=env_group_size，也就是不跨round
        # 复用——最保守、最不容易OOM的设置；机器内存富余的话可以调大以减少重复
        # warm-up 的开销。
        self.max_resident_envs = max_resident_envs if max_resident_envs is not None else env_group_size
        assert self.max_resident_envs >= env_group_size, (
            f"max_resident_envs({self.max_resident_envs}) 不能小于 env_group_size({env_group_size})，"
            f"否则同一个round内需要的环境数量都放不下"
        )

    def _make_env_stream(self, env_file):
        """单个环境(单个parquet分片)的无限循环+局部shuffle样本流，跟
        CustomPretrainDataset 的 training 模式逐条处理逻辑保持一致，只是数据源
        从"整个目录"缩小到"这一个分片文件"。

        !!!性能/内存说明(冒烟测试踩过的坑，不要改回 gluonts.FileDataset)!!!
        这里没有用 gluonts.FileDataset(单文件)+Cyclic，是因为实测过：对这批
        ~100万行/文件的parquet分片，gluonts.FileDataset 在第一次迭代时会把
        整个文件一次性物化成逐行dict(哪怕只想读2000条也一样)，单个文件就要
        吃掉 ~16GB 常驻内存、耗时 ~6.4秒；EnvironmentBalancedPretrainDataset
        需要 env_group_size 个环境的 stream 同时存活(后面训练过程中会有更多
        环境陆续被激活并常驻)，几个环境同时物化就会直接把机器内存打爆——这正是
        Stage4冒烟测试第一次跑就被 SIGKILL(OOM)的根因，用 pyarrow 的 Python
        进程内存监控实测复现确认过。改用 pyarrow.parquet.ParquetFile.iter_batches
        做真正的流式读取：同一个文件全量流式读一遍只要 ~1.9秒、峰值常驻内存
        ~660MB，比 gluonts 的路径低两个数量级，且内存不随读取进度增长。
        """
        import pyarrow.parquet as pq

        def _infinite_rows():
            pf = pq.ParquetFile(env_file)
            while True:
                for batch in pf.iter_batches(batch_size=1000, columns=['target', 'distances', 'indices']):
                    cols = batch.to_pydict()
                    for i in range(batch.num_rows):
                        yield {
                            'target': np.asarray(cols['target'][i], dtype=np.float32),
                            'distances': np.asarray(cols['distances'][i], dtype=np.float32),
                            'indices': np.asarray(cols['indices'][i], dtype=np.int64),
                        }

        base_iter = _infinite_rows()
        buffer = []
        generator = torch.Generator()

        def _next_processed_entry():
            entry = next(base_iter)
            entry['x'] = entry['target'][:self.context_length]
            entry['y'] = entry['target'][self.context_length:]
            entry['distances'] = entry['distances'][:self.top_k]
            entry['indices'] = entry['indices'][:self.top_k]
            if self.drop_prob > 0:
                target = entry['target'].copy()
                drop_p = np.random.uniform(low=0.0, high=self.drop_prob)
                mask = np.random.choice(
                    [True, False], size=len(target), p=[drop_p, 1 - drop_p]
                )
                target[mask] = np.nan
                entry['target'] = target
                # 修复(发车前检查发现)：CustomPretrainDataset里dropout发生后会
                # 重新从(被打了NaN的)target切一次x/y，这样drop_prob才会真正影响到
                # 喂给模型的数据；这里最初漏了这一步，导致x/y在dropout之前就切好了，
                # drop_prob对训练数据实际上没有任何作用。
                entry['x'] = entry['target'][:self.context_length]
                entry['y'] = entry['target'][self.context_length:]
            return entry

        while True:
            buffer.append(_next_processed_entry())
            if len(buffer) >= self.env_shuffle_buffer_length:
                idx = int(torch.randint(len(buffer), size=(), generator=generator))
                yield buffer.pop(idx)

    def __iter__(self):
        # 有上限的 LRU 缓存：最多同时保留 self.max_resident_envs 个环境的
        # stream，超过上限就丢弃最久未使用的那个(整个 generator 连带它的
        # pyarrow reader/shuffle buffer一起被回收)。用 dict 的插入顺序当
        # LRU 顺序：命中时 pop 再重新插入(移到末尾=最近使用)，插入新环境时
        # 从最前面(最久未用)开始淘汰。
        resident_streams = {}
        env_rng = np.random.default_rng()

        def _get_stream(env_id):
            if env_id in resident_streams:
                stream = resident_streams.pop(env_id)
            else:
                stream = self._make_env_stream(self.env_files[env_id])
                if len(resident_streams) >= self.max_resident_envs:
                    resident_streams.pop(next(iter(resident_streams)))
            resident_streams[env_id] = stream  # 重新插入=标记为最近使用
            return stream

        while True:
            env_ids = env_rng.choice(
                self.num_envs, size=self.env_group_size, replace=False
            )
            for env_id in env_ids:
                env_id = int(env_id)
                stream = _get_stream(env_id)
                for _ in range(self.env_batch_size):
                    entry = next(stream)
                    entry['env_id'] = env_id
                    yield entry


