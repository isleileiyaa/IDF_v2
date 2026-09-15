#!/bin/bash
# 过夜任务(GPU0): idf_ridde_v2 真正的零正则基线(rho_sem=rho_xcov=rho_ord=0)，10000步训练+评测。
# 跟之前overnight_gpu0/gpu1同一套1小时自动重试逻辑。
set -u
source /home/fenglei/miniconda3/etc/profile.d/conda.sh
conda activate tsrag
cd /home/fenglei/TS-RAG-main/TS-RAG
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline
export TRAIN_STEPS=10000
export EVALUATION_STEPS=10000
mkdir -p logs

RETRY_WAIT=3600
CKPT_DIR="checkpoints/idf_ridde_v2_true_zero_baseline"
CKPT="${CKPT_DIR}/data50m_idf_ridde_v2_512_pred64_lookback512_top10_lr0.0003_drop0.2_adamw_cosanneal_step10000_bs256_no_embeddingtuning_sem0_xcov0_ord0_final.pth"
export CHECKPOINTS_DIR="$CKPT_DIR"
mkdir -p "$CKPT_DIR"

attempt=1
while [ ! -f "$CKPT" ]; do
    echo "[overnight_gpu0_zerobaseline] 第${attempt}次训练尝试开始: $(date)"
    bash script/pretrain_idf_ridde_v2.sh 0 0 0 \
        > "logs/idf_ridde_v2_true_zero_baseline_attempt${attempt}.log" 2>&1
    if [ -f "$CKPT" ]; then
        echo "[overnight_gpu0_zerobaseline] 训练成功(第${attempt}次): $(date)"
        break
    fi
    echo "[overnight_gpu0_zerobaseline] 第${attempt}次训练失败(无checkpoint，大概率被资源争抢杀掉)，${RETRY_WAIT}秒后重试: $(date)"
    attempt=$((attempt+1))
    sleep "$RETRY_WAIT"
done

RESULT_FILE="results/forecast_evaluation/zeroshot_chronos_idf_ridde_v2_zero_baseline.txt"
attempt=1
while true; do
    n=$(grep -c "^ETTh1\|^ETTh2\|^ETTm1\|^ETTm2\|^weather\|^exchange_rate" "$RESULT_FILE" 2>/dev/null || echo 0)
    if [ "$n" -eq 6 ]; then
        echo "[overnight_gpu0_zerobaseline] 评测已完整(6/6)，跳过: $(date)"
        break
    fi
    rm -f "$RESULT_FILE"
    echo "[overnight_gpu0_zerobaseline] 第${attempt}次评测尝试开始: $(date)"
    DATASETS="ETTh1 ETTh2 ETTm1 ETTm2 weather exchange_rate" \
        SAVE_SUFFIX="zero_baseline" \
        CHECKPOINT_MODEL_PATH="$CKPT" \
        bash script/zeroshot_chronos_idf_ridde_v2.sh \
        > "logs/zeroshot_idf_ridde_v2_zero_baseline_attempt${attempt}.log" 2>&1
    n=$(grep -c "^ETTh1\|^ETTh2\|^ETTm1\|^ETTm2\|^weather\|^exchange_rate" "$RESULT_FILE" 2>/dev/null || echo 0)
    if [ "$n" -eq 6 ]; then
        echo "[overnight_gpu0_zerobaseline] 评测成功(第${attempt}次): $(date)"
        break
    fi
    echo "[overnight_gpu0_zerobaseline] 第${attempt}次评测失败(得到${n}/6条)，${RETRY_WAIT}秒后重试: $(date)"
    attempt=$((attempt+1))
    sleep "$RETRY_WAIT"
done

echo "==== overnight_gpu0_zerobaseline 全部完成: $(date) ===="
