#!/bin/bash
# idf_clean_dis_v4 lambda_sem=0.1/1 完整10000步训练，GPU0。
set -u
source /home/fenglei/miniconda3/etc/profile.d/conda.sh
conda activate tsrag
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline
mkdir -p logs

V3_CKPT="checkpoints/idf_clean_dis_v3_full10000_tau017/idf_clean_dis_v3_full10000_tau017_final.pth"

train_one() {
    local LAMBDA=$1
    local MODEL_ID="idf_clean_dis_v4_lambda${LAMBDA}_full10000"
    local CKPT_DIR="checkpoints/${MODEL_ID}"
    local LOGFILE="logs/${MODEL_ID}.log"
    mkdir -p "$CKPT_DIR"
    echo "===== 开始训练 lambda_sem=${LAMBDA}（10000步）====="
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
        --lambda_sem "$LAMBDA" \
        --freeze_chronos_bolt \
        --init_from_checkpoint "$V3_CKPT" \
        --checkpoints "$CKPT_DIR" \
        > "$LOGFILE" 2>&1
    echo "===== lambda_sem=${LAMBDA} 结束，exit code $? ====="
}

echo "==== v4_lambda_sem_small_full10000 开始: $(date) ===="
train_one 0.1
train_one 1
echo "==== v4_lambda_sem_small_full10000 全部完成: $(date) ===="
