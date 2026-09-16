#!/bin/bash
# 等当前的v4_lambda_sem_full10000.sh(λ=40/100那两条)跑完，
# 再补跑λ=10(修复mkdir -p缺失的bug)，最后对3个checkpoint跑zeroshot(6个数据集，不含electricity)。
set -u
source /home/fenglei/miniconda3/etc/profile.d/conda.sh && conda activate tsrag
export WANDB_MODE=offline

MASTER_PID=2002576
echo "===== 等待主进程(PID=$MASTER_PID, λ=40/100)结束: $(date) ====="
while kill -0 "$MASTER_PID" 2>/dev/null; do
    sleep 30
done
echo "===== 主进程已结束: $(date) ====="

V3_CKPT="checkpoints/idf_clean_dis_v3_full10000_tau017/idf_clean_dis_v3_full10000_tau017_final.pth"
MODEL_ID="idf_clean_dis_v4_lambda10_full10000"
CKPT_DIR="checkpoints/${MODEL_ID}"
mkdir -p "$CKPT_DIR"
echo "===== 补跑 lambda_sem=10 (mkdir修复后): $(date) ====="
python3 pretrain.py \
    --model_id "$MODEL_ID" \
    --model ChronosBoltRetrieve \
    --top_k 10 \
    --retrieve_lookback_length 512 \
    --retrieval_database_path /home/fenglei/TS-RAG-main/retrieval_database/pretrain/retrieval_database_512.parquet \
    --augment_mode idf_clean_dis_v4 \
    --pretrained_model_path /home/fenglei/TS-RAG-main/TS-RAG/checkpoints/base/ \
    --context_length 512 \
    --prediction_length 64 \
    --data_path /home/fenglei/TS-RAG-main/datasets/pretrain/pretrain_pairs_ctx512 \
    --train_steps 10000 \
    --evaluation_steps 10000 \
    --optimizer adamw \
    --learning_rate 0.0003 \
    --weight_decay 0.01 \
    --tmax 20 \
    --drop_prob 0.2 \
    --batch_size 256 \
    --grad_clip_value 1.0 \
    --shuffle_buffer_length 10000 \
    --rho1 0 --rho2 1000 --rho3 0 --rho4 0 \
    --dyn_margin 1.0 \
    --aux_loss_detach_ret true \
    --tau_dis 0.17 \
    --lambda_sem 10 \
    --freeze_chronos_bolt \
    --init_from_checkpoint "$V3_CKPT" \
    --checkpoints "$CKPT_DIR" \
    > "logs/${MODEL_ID}.log" 2>&1
echo "===== lambda_sem=10 补跑结束，exit code $?: $(date) ====="

echo "===== 开始对3个checkpoint跑zeroshot(6个数据集,不含electricity): $(date) ====="
for LAMBDA in 10 40 100; do
    MODEL_ID="idf_clean_dis_v4_lambda${LAMBDA}_full10000"
    CKPT="checkpoints/${MODEL_ID}/${MODEL_ID}_final.pth"
    if [ ! -f "$CKPT" ]; then
        echo "===== 跳过 lambda_sem=${LAMBDA} 评测：checkpoint不存在($CKPT) ====="
        continue
    fi
    echo "===== 开始评测 lambda_sem=${LAMBDA} ====="
    DATASETS="ETTh1 ETTh2 ETTm1 ETTm2 weather exchange_rate" \
        AUGMENT_MODE="idf_clean_dis_v4" \
        SAVE_SUFFIX="eval_lambda${LAMBDA}_full10000" \
        CHECKPOINT_MODEL_PATH="$CKPT" \
        bash script/zeroshot_chronos_idf_clean_dis.sh \
        > "logs/zeroshot_v4_lambda${LAMBDA}_full10000.log" 2>&1
    echo "===== lambda_sem=${LAMBDA} 评测结束，exit code $? ====="
done
echo "===== 全部完成: $(date) ====="
