#!/bin/bash
# 过夜任务: idf_clean_dis_v4 + fusion_mode=additive + lambda_ord=$LAMBDA_ORD，真正从零训练
# (无init_from_checkpoint)，10000步训练+评测。被杀后每30分钟自动重试。
# 用法: LAMBDA_ORD=0.001 CUDA_VISIBLE_DEVICES=0 bash overnight_additive_ord_fromscratch.sh
set -u
source /home/fenglei/miniconda3/etc/profile.d/conda.sh
conda activate tsrag
cd /home/fenglei/TS-RAG-main/TS-RAG
export WANDB_MODE=offline
export TRAIN_STEPS=10000
export EVALUATION_STEPS=10000
export SHUFFLE_BUFFER_LENGTH=2000
unset CHECKPOINTS_DIR
mkdir -p logs

LAMBDA_ORD="${LAMBDA_ORD:?必须指定LAMBDA_ORD}"
ORD_MARGIN="${ORD_MARGIN:-0.0}"
TAU="${TAU:-0.1}"
export LAMBDA_ORD ORD_MARGIN TAU
RETRY_WAIT=1800
if [ "$ORD_MARGIN" = "0.0" ] && [ "$TAU" = "0.1" ]; then
    MODEL_ID="idf_clean_dis_v4_fusion_additive_ord${LAMBDA_ORD}_fromscratch_full10000"
else
    MODEL_ID="idf_clean_dis_v4_fusion_additive_ord${LAMBDA_ORD}_margin${ORD_MARGIN}_tau${TAU}_fromscratch_full10000"
fi
CKPT="checkpoints/${MODEL_ID}/${MODEL_ID}_final.pth"

attempt=1
while [ ! -f "$CKPT" ]; do
    echo "[overnight_additive_ord_fromscratch lambda_ord=${LAMBDA_ORD}] 第${attempt}次训练尝试开始: $(date)"
    bash script/additive_ord_fromscratch_full10000.sh \
        > "logs/${MODEL_ID}_attempt${attempt}.log" 2>&1
    if [ -f "$CKPT" ]; then
        echo "[overnight_additive_ord_fromscratch lambda_ord=${LAMBDA_ORD}] 训练成功(第${attempt}次): $(date)"
        break
    fi
    echo "[overnight_additive_ord_fromscratch lambda_ord=${LAMBDA_ORD}] 第${attempt}次训练失败(无checkpoint)，${RETRY_WAIT}秒后重试: $(date)"
    attempt=$((attempt+1))
    sleep "$RETRY_WAIT"
done

RESULT_FILE="results/forecast_evaluation/zeroshot_chronos_idf_clean_dis_fusion_${MODEL_ID#idf_clean_dis_v4_fusion_}.txt"
attempt=1
while true; do
    n=$(grep -c "^ETTh1\|^ETTh2\|^ETTm1\|^ETTm2\|^weather\|^exchange_rate" "$RESULT_FILE" 2>/dev/null || echo 0)
    if [ "$n" -eq 6 ]; then
        echo "[overnight_additive_ord_fromscratch lambda_ord=${LAMBDA_ORD}] 评测已完整(6/6)，跳过: $(date)"
        break
    fi
    rm -f "$RESULT_FILE"
    echo "[overnight_additive_ord_fromscratch lambda_ord=${LAMBDA_ORD}] 第${attempt}次评测尝试开始: $(date)"
    DATASETS="ETTh1 ETTh2 ETTm1 ETTm2 weather exchange_rate" \
        AUGMENT_MODE="idf_clean_dis_v4" \
        FUSION_MODE="additive" \
        SAVE_SUFFIX="fusion_${MODEL_ID#idf_clean_dis_v4_fusion_}" \
        CHECKPOINT_MODEL_PATH="$CKPT" \
        bash script/zeroshot_chronos_idf_clean_dis.sh \
        > "logs/zeroshot_fusion_${MODEL_ID#idf_clean_dis_v4_fusion_}_attempt${attempt}.log" 2>&1
    n=$(grep -c "^ETTh1\|^ETTh2\|^ETTm1\|^ETTm2\|^weather\|^exchange_rate" "$RESULT_FILE" 2>/dev/null || echo 0)
    if [ "$n" -eq 6 ]; then
        echo "[overnight_additive_ord_fromscratch lambda_ord=${LAMBDA_ORD}] 评测成功(第${attempt}次): $(date)"
        break
    fi
    echo "[overnight_additive_ord_fromscratch lambda_ord=${LAMBDA_ORD}] 第${attempt}次评测失败(得到${n}/6条)，${RETRY_WAIT}秒后重试: $(date)"
    attempt=$((attempt+1))
    sleep "$RETRY_WAIT"
done

echo "==== overnight_additive_ord_fromscratch lambda_ord=${LAMBDA_ORD} 全部完成: $(date) ===="
