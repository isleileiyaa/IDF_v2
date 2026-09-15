#!/bin/bash
# 过夜任务(GPU1): idf_clean_dis_v4 + L_xcov，三档lambda_xcov=0.0001/0.001/0.01，各10000步训练+评测。
# 被杀后每30分钟自动重试。
set -u
source /home/fenglei/miniconda3/etc/profile.d/conda.sh
conda activate tsrag
cd /home/fenglei/TS-RAG-main/TS-RAG
export CUDA_VISIBLE_DEVICES=1
export WANDB_MODE=offline
mkdir -p logs

V3_CKPT="checkpoints/idf_clean_dis_v3_full10000_tau017/idf_clean_dis_v3_full10000_tau017_final.pth"
RETRY_WAIT=1800

train_with_retry() {
    local LAMBDA_XCOV=$1
    local MODEL_ID="idf_clean_dis_v4_xcov${LAMBDA_XCOV}_full10000"
    local CKPT_DIR="checkpoints/${MODEL_ID}"
    local CKPT="${CKPT_DIR}/${MODEL_ID}_final.pth"
    mkdir -p "$CKPT_DIR"
    local attempt=1
    while [ ! -f "$CKPT" ]; do
        echo "[overnight_gpu1_xcov] lambda_xcov=${LAMBDA_XCOV} 第${attempt}次训练尝试开始: $(date)"
        python3 pretrain.py \
            --model_id "$MODEL_ID" --model ChronosBoltRetrieve --top_k 10 \
            --retrieve_lookback_length 512 \
            --retrieval_database_path /home/fenglei/TS-RAG-main/retrieval_database/pretrain/retrieval_database_512.parquet \
            --augment_mode idf_clean_dis_v4 \
            --pretrained_model_path /home/fenglei/TS-RAG-main/TS-RAG/checkpoints/base/ \
            --context_length 512 --prediction_length 64 \
            --data_path /home/fenglei/TS-RAG-main/datasets/pretrain/pretrain_pairs_ctx512 \
            --train_steps 10000 --evaluation_steps 10000 \
            --optimizer adamw --learning_rate 0.0003 --weight_decay 0.01 --tmax 20 \
            --drop_prob 0.2 --batch_size 256 --grad_clip_value 1.0 --shuffle_buffer_length 2000 \
            --rho1 0 --rho2 1000 --rho3 0 --rho4 0 --dyn_margin 1.0 --aux_loss_detach_ret true \
            --tau_dis 0.17 --lambda_sem 0 --lambda_xcov "$LAMBDA_XCOV" \
            --freeze_chronos_bolt --init_from_checkpoint "$V3_CKPT" --checkpoints "$CKPT_DIR" \
            > "logs/${MODEL_ID}_attempt${attempt}.log" 2>&1
        if [ -f "$CKPT" ]; then
            echo "[overnight_gpu1_xcov] lambda_xcov=${LAMBDA_XCOV} 训练成功(第${attempt}次): $(date)"
            break
        fi
        echo "[overnight_gpu1_xcov] lambda_xcov=${LAMBDA_XCOV} 第${attempt}次训练失败(无checkpoint)，${RETRY_WAIT}秒后重试: $(date)"
        attempt=$((attempt+1))
        sleep "$RETRY_WAIT"
    done
}

eval_with_retry() {
    local LAMBDA_XCOV=$1
    local MODEL_ID="idf_clean_dis_v4_xcov${LAMBDA_XCOV}_full10000"
    local CKPT="checkpoints/${MODEL_ID}/${MODEL_ID}_final.pth"
    local RESULT_FILE="results/forecast_evaluation/zeroshot_chronos_idf_clean_dis_eval_xcov${LAMBDA_XCOV}_full10000.txt"
    local attempt=1
    while true; do
        local n
        n=$(grep -c "^ETTh1\|^ETTh2\|^ETTm1\|^ETTm2\|^weather\|^exchange_rate" "$RESULT_FILE" 2>/dev/null || echo 0)
        if [ "$n" -eq 6 ]; then
            echo "[overnight_gpu1_xcov] lambda_xcov=${LAMBDA_XCOV} 评测已完整(6/6)，跳过: $(date)"
            break
        fi
        rm -f "$RESULT_FILE"
        echo "[overnight_gpu1_xcov] lambda_xcov=${LAMBDA_XCOV} 第${attempt}次评测尝试开始: $(date)"
        DATASETS="ETTh1 ETTh2 ETTm1 ETTm2 weather exchange_rate" \
            AUGMENT_MODE="idf_clean_dis_v4" \
            SAVE_SUFFIX="eval_xcov${LAMBDA_XCOV}_full10000" \
            CHECKPOINT_MODEL_PATH="$CKPT" \
            bash script/zeroshot_chronos_idf_clean_dis.sh \
            > "logs/zeroshot_v4_xcov${LAMBDA_XCOV}_full10000_attempt${attempt}.log" 2>&1
        n=$(grep -c "^ETTh1\|^ETTh2\|^ETTm1\|^ETTm2\|^weather\|^exchange_rate" "$RESULT_FILE" 2>/dev/null || echo 0)
        if [ "$n" -eq 6 ]; then
            echo "[overnight_gpu1_xcov] lambda_xcov=${LAMBDA_XCOV} 评测成功(第${attempt}次): $(date)"
            break
        fi
        echo "[overnight_gpu1_xcov] lambda_xcov=${LAMBDA_XCOV} 第${attempt}次评测失败(得到${n}/6条)，${RETRY_WAIT}秒后重试: $(date)"
        attempt=$((attempt+1))
        sleep "$RETRY_WAIT"
    done
}

echo "==== overnight_gpu1_xcov 开始: $(date) ===="
for LAMBDA_XCOV in 0.0001 0.001 0.01; do
    train_with_retry "$LAMBDA_XCOV"
    eval_with_retry "$LAMBDA_XCOV"
done
echo "==== overnight_gpu1_xcov 全部完成: $(date) ===="
