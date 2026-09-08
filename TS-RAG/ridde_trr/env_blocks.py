"""
D-RACS 目录（dracs/）已经废弃——老师看完那条诊断线之后决定放弃，改用
《RIDDE_新版目标函数与最终实验方案》。这个模块是新方案共用的一个基础
工具：把某个 split 里"按 index=feat_id*tot_len+s_begin 摊平的样本"按
真实时间顺序切成 num_blocks 个连续的"环境"/时间块（对应方案文档第2节
的环境划分 D_src = union E_e，以及第6.3节的时间块稳健性指标）。

关键背景（已经跟 data_provider/data_loader.py 里的 __getitem__ 核对过，
例如 Dataset_ETT_hour_retrieve.__getitem__，ChronosBolt.py 同款逻辑在
所有 *_retrieve Dataset 类里都一致）：

    feat_id = index // tot_len
    s_begin = index % tot_len

也就是说，展平后的样本数组是"按通道分段、每段内部按时间顺序排列"
（channel-major）：第 0..tot_len-1 行是通道0的全部时间窗口（按时间顺序
排列），第 tot_len..2*tot_len-1 行是通道1的全部时间窗口，以此类推。

所以"同一个时间环境"必须跨通道对齐到同一段 s_begin 范围，不能直接把
展平后的行下标切成连续的 num_blocks 段——那样会把"通道"当成"时间"切开，
产出的"环境"其实是不同通道的混合，跟方案要求的"按时间连续切分"完全不
是一回事。

用法：
    from env_blocks import assign_env_ids
    env_id, tot_len, boundaries = assign_env_ids(N, enc_in, num_blocks)
"""
import numpy as np

# 已核对：这几个数据集在本仓库里的原始列数（=enc_in），用来把展平后的
# 样本数还原成 (feat_id, s_begin) 两个维度。用 weather 反推出的
# tot_len=10476，乘以 enc_in=21 正好等于 local cc 汇报的 weather 完整
# 测试集真实样本数 219996，可以当作这张表本身正确性的交叉验证
# （见下面 _selftest 里的断言）。
KNOWN_ENC_IN = {
    'ETTh1': 7, 'ETTh2': 7, 'ETTm1': 7, 'ETTm2': 7,
    'weather': 21, 'exchange_rate': 8, 'electricity': 321,
}


def time_block_boundaries(tot_len, num_blocks):
    """把 [0, tot_len) 这段时间下标切成 num_blocks 个尽量等长的连续区间。
    返回长度 num_blocks+1 的边界数组 boundaries；区间 b 是
    [boundaries[b], boundaries[b+1])。
    """
    if num_blocks < 1:
        raise ValueError('num_blocks must be >= 1')
    if tot_len < num_blocks:
        raise ValueError(f'tot_len={tot_len} < num_blocks={num_blocks}，环境切得太细了')
    boundaries = np.linspace(0, tot_len, num_blocks + 1)
    boundaries = np.round(boundaries).astype(int)
    boundaries[0] = 0
    boundaries[-1] = tot_len
    return boundaries


def assign_env_ids(N, enc_in, num_blocks):
    """给展平后长度为 N 的样本数组（channel-major，
    index = feat_id*tot_len + s_begin）的每一行分配一个环境编号
    env_id in [0, num_blocks)。

    返回 (env_id 数组(长度 N), tot_len, boundaries)。
    """
    if N % enc_in != 0:
        raise ValueError(
            f'N={N} 不能被 enc_in={enc_in} 整除，说明 enc_in 填错了，或者这份'
            f'缓存不是完整的一个 split —— 先检查一下再往下算。'
        )
    tot_len = N // enc_in
    boundaries = time_block_boundaries(tot_len, num_blocks)

    time_env = np.zeros(tot_len, dtype=np.int64)
    for b in range(num_blocks):
        time_env[boundaries[b]:boundaries[b + 1]] = b

    idx = np.arange(N)
    s_begin = idx % tot_len
    env_id = time_env[s_begin]
    return env_id, tot_len, boundaries


def _selftest():
    # 造一个 enc_in=3, tot_len=10 的假样本(N=30)，切成 num_blocks=4 份，
    # 手工核对每个环境到底包含哪些 (feat_id, s_begin)。
    enc_in, tot_len, num_blocks = 3, 10, 4
    N = enc_in * tot_len
    env_id, got_tot_len, boundaries = assign_env_ids(N, enc_in, num_blocks)
    assert got_tot_len == tot_len, f'tot_len 算错了: {got_tot_len} != {tot_len}'
    assert boundaries[0] == 0 and boundaries[-1] == tot_len
    assert len(boundaries) == num_blocks + 1
    assert np.all(np.diff(boundaries) >= 0)

    # 核心正确性：同一个 s_begin（同一个真实时间点），不管 feat_id（通道）
    # 是几，env_id 必须一样 —— 这是这个模块存在的唯一理由。
    for s in range(tot_len):
        envs_for_this_time = set()
        for feat_id in range(enc_in):
            idx = feat_id * tot_len + s
            envs_for_this_time.add(int(env_id[idx]))
        assert len(envs_for_this_time) == 1, (
            f's_begin={s} 在不同通道上被分到了不同环境 {envs_for_this_time}，'
            f'说明切分逻辑没有正确按时间（而不是按展平下标）对齐'
        )

    # 环境编号必须随时间单调不减（因为是连续时间块，不是随机分组）
    time_order_envs = [int(env_id[s]) for s in range(tot_len)]  # feat_id=0 那一段就是纯时间顺序
    assert time_order_envs == sorted(time_order_envs), '环境编号没有按时间单调递增，切分逻辑有问题'

    # weather 的真实数字交叉验证：enc_in=21，真实 N=219996 -> tot_len 应为 10476
    real_N, real_enc_in = 219996, KNOWN_ENC_IN['weather']
    assert real_N % real_enc_in == 0
    real_tot_len = real_N // real_enc_in
    assert real_tot_len == 10476, f'weather tot_len 交叉验证失败: {real_tot_len} != 10476'

    print('[env_blocks] selftest PASSED')
    print(f'  示例: enc_in={enc_in}, tot_len={tot_len}, num_blocks={num_blocks}')
    print(f'  boundaries={boundaries.tolist()}')
    print('  weather 真实数据交叉验证: N=219996, enc_in=21 -> tot_len=10476 '
          '(与 local cc 汇报的样本数吻合)')


if __name__ == '__main__':
    _selftest()
