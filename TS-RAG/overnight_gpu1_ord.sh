#!/bin/bash
# 过夜任务(GPU1): rho_ord=0.0001/0.01/0.1/1.0 各10000步训练+评测(idf_ridde_v2)。
# 如果训练中途被杀(资源争抢)，每隔1小时自动重试，直到拿到checkpoint为止；
# 评测同理，直到拿到完整的6个数据集结果为止。
set -u
source /home/fenglei/miniconda3/etc/profile.d/conda.sh
conda activate tsrag
cd /home/fenglei/TS-RAG-main/TS-RAG
export CUDA_VISIBLE_DEVICES=1
export WANDB_MODE=offline
mkdir -p logs

RETRY_WAIT=3600

train_with_retry() {
    local RHO_ORD=$1
    local CKPT_DIR="checkpoints/ord_full10000_rho${RHO_ORD}"
    local CKPT="${CKPT_DIR}/data50m_idf_ridde_v2_512_pred64_lookback512_top10_lr0.0003_drop0.2_adamw_cosanneal_step10000_bs256_no_embeddingtuning_sem0_xcov0_ord${RHO_ORD}_final.pth"
    mkdir -p "$CKPT_DIR"
    export CHECKPOINTS_DIR="$CKPT_DIR"
    local attempt=1
    while [ ! -f "$CKPT" ]; do
        echo "[overnight_gpu1] rho_ord=${RHO_ORD} 第${attempt}次训练尝试开始: $(date)"
        bash script/pretrain_idf_ridde_v2.sh 0 0 "${RHO_ORD}" \
            > "logs/ord_full10000_${RHO_ORD}_attempt${attempt}.log" 2>&1
        if [ -f "$CKPT" ]; then
            echo "[overnight_gpu1] rho_ord=${RHO_ORD} 训练成功(第${attempt}次): $(date)"
            break
        fi
        echo "[overnight_gpu1] rho_ord=${RHO_ORD} 第${attempt}次训练失败(无checkpoint，大概率被资源争抢杀掉)，${RETRY_WAIT}秒后重试: $(date)"
        attempt=$((attempt+1))
        sleep "$RETRY_WAIT"
    done
}

eval_with_retry() {
    local RHO_ORD=$1
    local CKPT="checkpoints/ord_full10000_rho${RHO_ORD}/data50m_idf_ridde_v2_512_pred64_lookback512_top10_lr0.0003_drop0.2_adamw_cosanneal_step10000_bs256_no_embeddingtuning_sem0_xcov0_ord${RHO_ORD}_final.pth"
    local RESULT_FILE="results/forecast_evaluation/zeroshot_chronos_idf_ridde_v2_ord_full10000_${RHO_ORD}.txt"
    local attempt=1
    while true; do
        local n
        n=$(grep -c "^ETTh1\|^ETTh2\|^ETTm1\|^ETTm2\|^weather\|^exchange_rate" "$RESULT_FILE" 2>/dev/null || echo 0)
        if [ "$n" -eq 6 ]; then
            echo "[overnight_gpu1] rho_ord=${RHO_ORD} 评测已完整(6/6)，跳过: $(date)"
            break
        fi
        rm -f "$RESULT_FILE"
        echo "[overnight_gpu1] rho_ord=${RHO_ORD} 第${attempt}次评测尝试开始: $(date)"
        DATASETS="ETTh1 ETTh2 ETTm1 ETTm2 weather exchange_rate" \
            SAVE_SUFFIX="ord_full10000_${RHO_ORD}" \
            CHECKPOINT_MODEL_PATH="$CKPT" \
            bash script/zeroshot_chronos_idf_ridde_v2.sh \
            > "logs/zeroshot_ord_full10000_${RHO_ORD}_attempt${attempt}.log" 2>&1
        n=$(grep -c "^ETTh1\|^ETTh2\|^ETTm1\|^ETTm2\|^weather\|^exchange_rate" "$RESULT_FILE" 2>/dev/null || echo 0)
        if [ "$n" -eq 6 ]; then
            echo "[overnight_gpu1] rho_ord=${RHO_ORD} 评测成功(第${attempt}次): $(date)"
            break
        fi
        echo "[overnight_gpu1] rho_ord=${RHO_ORD} 第${attempt}次评测失败(得到${n}/6条)，${RETRY_WAIT}秒后重试: $(date)"
        attempt=$((attempt+1))
        sleep "$RETRY_WAIT"
    done
}

echo "==== overnight_gpu1 开始: $(date) ===="
for R in 0.0001 0.01 0.1 1.0; do
    train_with_retry "$R"
    eval_with_retry "$R"
done
echo "==== overnight_gpu1 全部完成: $(date) ===="
