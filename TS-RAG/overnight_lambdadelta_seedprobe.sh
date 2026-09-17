#!/bin/bash
# 噪声底噪检验: 跟lambda_delta=7.0(learned头,fromscratch)完全相同的配置，只换随机种子
# (PROBE_SEED，默认1234，原始跑用的是pretrain.py默认种子2021)，训练+评测，
# 拿它跟原始结果(0.2096/0.2632)对比，量化"仅仅换种子"能带来多大的MSE/MAE波动。
set -u
source /home/fenglei/miniconda3/etc/profile.d/conda.sh
conda activate tsrag
cd /home/fenglei/TS-RAG-main/TS-RAG
export WANDB_MODE=offline
export TRAIN_STEPS=10000
export EVALUATION_STEPS=10000
export SHUFFLE_BUFFER_LENGTH=2000
export PROBE_SEED="${PROBE_SEED:-1234}"
unset CHECKPOINTS_DIR
mkdir -p logs

RETRY_WAIT=1800
MODEL_ID="idf_clean_dis_v4_fusion_learned_lambdadelta7.0_fromscratch_full10000_seedprobe${PROBE_SEED}"
CKPT="checkpoints/${MODEL_ID}/${MODEL_ID}_final.pth"

attempt=1
while [ ! -f "$CKPT" ]; do
    echo "[overnight_lambdadelta_seedprobe seed=${PROBE_SEED}] 第${attempt}次训练尝试开始: $(date)"
    bash script/learned_lambdadelta7_seed_probe.sh \
        > "logs/${MODEL_ID}_attempt${attempt}.log" 2>&1
    if [ -f "$CKPT" ]; then
        echo "[overnight_lambdadelta_seedprobe seed=${PROBE_SEED}] 训练成功(第${attempt}次): $(date)"
        break
    fi
    echo "[overnight_lambdadelta_seedprobe seed=${PROBE_SEED}] 第${attempt}次训练失败(无checkpoint)，${RETRY_WAIT}秒后重试: $(date)"
    attempt=$((attempt+1))
    sleep "$RETRY_WAIT"
done

RESULT_FILE="results/forecast_evaluation/zeroshot_chronos_idf_clean_dis_fusion_learned_lambdadelta7.0_fromscratch_full10000_seedprobe${PROBE_SEED}.txt"
attempt=1
while true; do
    n=$(grep -c "^ETTh1\|^ETTh2\|^ETTm1\|^ETTm2\|^weather\|^exchange_rate" "$RESULT_FILE" 2>/dev/null || echo 0)
    if [ "$n" -eq 6 ]; then
        echo "[overnight_lambdadelta_seedprobe seed=${PROBE_SEED}] 评测已完整(6/6)，跳过: $(date)"
        break
    fi
    rm -f "$RESULT_FILE"
    echo "[overnight_lambdadelta_seedprobe seed=${PROBE_SEED}] 第${attempt}次评测尝试开始: $(date)"
    DATASETS="ETTh1 ETTh2 ETTm1 ETTm2 weather exchange_rate" \
        AUGMENT_MODE="idf_clean_dis_v4" \
        FUSION_MODE="learned" \
        SAVE_SUFFIX="fusion_learned_lambdadelta7.0_fromscratch_full10000_seedprobe${PROBE_SEED}" \
        CHECKPOINT_MODEL_PATH="$CKPT" \
        bash script/zeroshot_chronos_idf_clean_dis.sh \
        > "logs/zeroshot_lambdadelta7_seedprobe${PROBE_SEED}_attempt${attempt}.log" 2>&1
    n=$(grep -c "^ETTh1\|^ETTh2\|^ETTm1\|^ETTm2\|^weather\|^exchange_rate" "$RESULT_FILE" 2>/dev/null || echo 0)
    if [ "$n" -eq 6 ]; then
        echo "[overnight_lambdadelta_seedprobe seed=${PROBE_SEED}] 评测成功(第${attempt}次): $(date)"
        break
    fi
    echo "[overnight_lambdadelta_seedprobe seed=${PROBE_SEED}] 第${attempt}次评测失败(得到${n}/6条)，${RETRY_WAIT}秒后重试: $(date)"
    attempt=$((attempt+1))
    sleep "$RETRY_WAIT"
done

echo "==== overnight_lambdadelta_seedprobe seed=${PROBE_SEED} 全部完成: $(date) ===="
