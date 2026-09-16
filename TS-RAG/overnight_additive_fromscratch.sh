#!/bin/bash
# 过夜任务(GPU1): idf_clean_dis_v4 + fusion_mode=additive + lambda_ord=0.01，真正从零训练
# (无init_from_checkpoint)，10000步训练+评测。被杀后每30分钟自动重试。
set -u
source /home/fenglei/miniconda3/etc/profile.d/conda.sh
conda activate tsrag
cd /home/fenglei/TS-RAG-main/TS-RAG
export CUDA_VISIBLE_DEVICES=1
export WANDB_MODE=offline
export TRAIN_STEPS=10000
export EVALUATION_STEPS=10000
export SHUFFLE_BUFFER_LENGTH=2000
unset CHECKPOINTS_DIR
mkdir -p logs

RETRY_WAIT=1800
MODEL_ID="idf_clean_dis_v4_fusion_additive_ord0.01_fromscratch_full10000"
CKPT="checkpoints/${MODEL_ID}/${MODEL_ID}_final.pth"

attempt=1
while [ ! -f "$CKPT" ]; do
    echo "[overnight_additive_fromscratch] 第${attempt}次训练尝试开始: $(date)"
    bash script/additive_ord0.01_fromscratch_full10000.sh \
        > "logs/${MODEL_ID}_attempt${attempt}.log" 2>&1
    if [ -f "$CKPT" ]; then
        echo "[overnight_additive_fromscratch] 训练成功(第${attempt}次): $(date)"
        break
    fi
    echo "[overnight_additive_fromscratch] 第${attempt}次训练失败(无checkpoint)，${RETRY_WAIT}秒后重试: $(date)"
    attempt=$((attempt+1))
    sleep "$RETRY_WAIT"
done

RESULT_FILE="results/forecast_evaluation/zeroshot_chronos_idf_clean_dis_fusion_additive_ord0.01_fromscratch_full10000.txt"
attempt=1
while true; do
    n=$(grep -c "^ETTh1\|^ETTh2\|^ETTm1\|^ETTm2\|^weather\|^exchange_rate" "$RESULT_FILE" 2>/dev/null || echo 0)
    if [ "$n" -eq 6 ]; then
        echo "[overnight_additive_fromscratch] 评测已完整(6/6)，跳过: $(date)"
        break
    fi
    rm -f "$RESULT_FILE"
    echo "[overnight_additive_fromscratch] 第${attempt}次评测尝试开始: $(date)"
    DATASETS="ETTh1 ETTh2 ETTm1 ETTm2 weather exchange_rate" \
        AUGMENT_MODE="idf_clean_dis_v4" \
        FUSION_MODE="additive" \
        SAVE_SUFFIX="fusion_additive_ord0.01_fromscratch_full10000" \
        CHECKPOINT_MODEL_PATH="$CKPT" \
        bash script/zeroshot_chronos_idf_clean_dis.sh \
        > "logs/zeroshot_additive_fromscratch_full10000_attempt${attempt}.log" 2>&1
    n=$(grep -c "^ETTh1\|^ETTh2\|^ETTm1\|^ETTm2\|^weather\|^exchange_rate" "$RESULT_FILE" 2>/dev/null || echo 0)
    if [ "$n" -eq 6 ]; then
        echo "[overnight_additive_fromscratch] 评测成功(第${attempt}次): $(date)"
        break
    fi
    echo "[overnight_additive_fromscratch] 第${attempt}次评测失败(得到${n}/6条)，${RETRY_WAIT}秒后重试: $(date)"
    attempt=$((attempt+1))
    sleep "$RETRY_WAIT"
done

echo "==== overnight_additive_fromscratch 全部完成: $(date) ===="
