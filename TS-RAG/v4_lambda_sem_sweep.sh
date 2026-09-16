#!/bin/bash
# idf_clean_dis_v4: lambda_sem 短程sweep（300步 x 3档：10/20/40）
# 目的：只是找一个能让loss_sem*lambda_sem落在loss_forecast的10%~20%区间的量级，
# 不代表最终训练要用哪个值。rho2=1000/tau_dis=0.17（v3机制原样保留）。
set -u
source /home/fenglei/miniconda3/etc/profile.d/conda.sh
conda activate tsrag
export WANDB_MODE=offline
mkdir -p logs

V3_CKPT="checkpoints/idf_clean_dis_v3_full10000_tau017/idf_clean_dis_v3_full10000_tau017_final.pth"

run_one() {
    local LAMBDA=$1
    local MODEL_ID="smoke_idf_clean_dis_v4_lambda${LAMBDA}_300"
    local CKPT_DIR="checkpoints/${MODEL_ID}"
    local LOGFILE="logs/${MODEL_ID}.log"
    mkdir -p "$CKPT_DIR"
    echo "[v4_lambda_sweep] 开始 lambda_sem=${LAMBDA}，日志: $LOGFILE"
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
        --train_steps 300 \
        --evaluation_steps 300 \
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
    echo "[v4_lambda_sweep] lambda_sem=${LAMBDA} 退出码: $?"
}

echo "==== v4_lambda_sweep 开始: $(date) ===="
run_one 10
run_one 20
run_one 40
echo "==== v4_lambda_sweep 全部完成: $(date) ===="
