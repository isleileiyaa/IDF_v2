"""
环境均衡采样器（对应《RIDDE_新版目标函数与最终实验方案》第5.3节
"环境平衡采样"："每个优化 step 从多个环境中抽取相同数量样本，使单次
更新能够估计环境风险和 CVaR"，建议 G=4~8 个环境、每个环境 m=8~32 个
样本）。

刻意做成一个不依赖真实数据集、只需要一个 env_id 数组就能跑的独立组件，
方便先用小数据/合成数据验证"每步从 G 个环境里各抽 m 个样本"这件事本
身实现对不对，不用等新模型和新数据管线（真实的 dataset 类、真实的
env_id 生成）都搭好才能测。真正接训练时，env_id 数组来自
env_blocks.assign_env_ids（用真实的 N / enc_in / num_blocks 算出来），
在 dataset 构造完之后算一次即可，因为环境划分在整个训练过程中是固定的
（方案文档第5.2节："环境边界、数量和构造规则在主实验前固定"）。

用法（接入真实训练时）：
    from env_blocks import assign_env_ids
    from env_sampler import EnvironmentBalancedBatchSampler

    env_id, tot_len, _ = assign_env_ids(len(train_dataset), enc_in, num_blocks)
    sampler = EnvironmentBalancedBatchSampler(
        env_id, envs_per_step=6, samples_per_env=16,
        steps_per_epoch=200, seed=PRETRAIN_SEED,
    )
    loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=0)
    # 每个 batch 里样本自带的 env_id（用 env_id[原始index] 查）就是训练时
    # 用来分组算 R_e^ret / Delta_e / CVaR 的分组依据。

先自检（不需要真实数据集，纯合成 env_id 数组验证代码本身对不对）：
    python ridde_trr/env_sampler.py
"""
import numpy as np


class EnvironmentBalancedBatchSampler:
    """每个 batch 固定由 envs_per_step 个不同环境、每个环境 samples_per_env
    个样本拼成。环境内部用有放回抽样——方案文档允许的环境大小差异很大
    （比如某些时间段样本天然就少），有放回抽样保证小环境也能正常凑够
    m 个样本，不需要额外处理"环境太小"的边界情况。
    """

    def __init__(self, env_ids, envs_per_step, samples_per_env, steps_per_epoch, seed=2021):
        self.env_ids = np.asarray(env_ids)
        self.envs_per_step = envs_per_step
        self.samples_per_env = samples_per_env
        self.steps_per_epoch = steps_per_epoch
        self.rng = np.random.default_rng(seed)

        self.env_to_indices = {}
        for e in np.unique(self.env_ids):
            self.env_to_indices[int(e)] = np.where(self.env_ids == e)[0]
        self.envs = sorted(self.env_to_indices.keys())

        if envs_per_step > len(self.envs):
            raise ValueError(
                f'envs_per_step={envs_per_step} 超过了实际环境数 {len(self.envs)}'
            )

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            chosen_envs = self.rng.choice(self.envs, size=self.envs_per_step, replace=False)
            batch = []
            for e in chosen_envs:
                pool = self.env_to_indices[int(e)]
                picked = self.rng.choice(pool, size=self.samples_per_env, replace=True)
                batch.extend(picked.tolist())
            self.rng.shuffle(batch)
            yield batch


def _selftest():
    rng = np.random.default_rng(1)
    env_sizes = [50, 5, 200, 30, 80, 3, 60, 40]
    env_ids = np.concatenate([np.full(sz, e) for e, sz in enumerate(env_sizes)])
    rng.shuffle(env_ids)

    envs_per_step, samples_per_env, steps_per_epoch = 4, 16, 20
    sampler = EnvironmentBalancedBatchSampler(env_ids, envs_per_step, samples_per_env,
                                               steps_per_epoch, seed=2021)

    assert len(sampler) == steps_per_epoch

    env_appearance_count = {e: 0 for e in range(len(env_sizes))}
    n_batches = 0
    for batch in sampler:
        n_batches += 1
        assert len(batch) == envs_per_step * samples_per_env, (
            f'batch 大小不对: {len(batch)} != {envs_per_step * samples_per_env}'
        )
        envs_in_batch = set(int(env_ids[i]) for i in batch)
        assert len(envs_in_batch) == envs_per_step, (
            f'一个 batch 里应该有 {envs_per_step} 个不同环境，实际有 {len(envs_in_batch)} 个'
        )
        for e in envs_in_batch:
            env_appearance_count[e] += 1
            count = sum(1 for i in batch if env_ids[i] == e)
            assert count == samples_per_env, (
                f'环境 {e} 在这个 batch 里贡献了 {count} 个样本，应该是 {samples_per_env} 个'
            )
        for e in envs_in_batch:
            if env_sizes[e] < samples_per_env:
                picked_from_this_env = [i for i in batch if env_ids[i] == e]
                assert len(picked_from_this_env) == samples_per_env

    assert n_batches == steps_per_epoch
    total_appearances = sum(env_appearance_count.values())
    assert total_appearances == steps_per_epoch * envs_per_step

    print('[env_sampler] selftest PASSED')
    print(f'  env_sizes={env_sizes}')
    print(f'  env_appearance_count(20步里每个环境被选中几次)={env_appearance_count}')
    print('  小环境(样本数3、5，小于 samples_per_env=16)也被正常抽到并靠有放回凑够了16个')


if __name__ == '__main__':
    _selftest()
