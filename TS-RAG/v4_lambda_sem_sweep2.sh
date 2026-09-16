#!/bin/bash
# idf_clean_dis_v4 lambda_sem sweep v2：scale_q clamp_min(1e-2)保护补丁之后重跑，
# 档位10/20/40/60/100，每档300步，跑完立即用dump_gamma.py(eval模式)测真实饱和度。
set -u
source /home/fenglei/miniconda3/etc/profile.d/conda.sh
conda activate tsrag
export WANDB_MODE=offline
mkdir -p logs

V3_CKPT="checkpoints/idf_clean_dis_v3_full10000_tau017/idf_clean_dis_v3_full10000_tau017_final.pth"

train_one() {
    local LAMBDA=$1
    local MODEL_ID="smoke_idf_clean_dis_v4_lambda${LAMBDA}_300v2"
    local CKPT_DIR="checkpoints/${MODEL_ID}"
    local LOGFILE="logs/${MODEL_ID}.log"
    mkdir -p "$CKPT_DIR"
    echo "[v4_lambda_sweep2] 开始训练 lambda_sem=${LAMBDA}，日志: $LOGFILE"
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
    echo "[v4_lambda_sweep2] lambda_sem=${LAMBDA} 训练退出码: $?"
}

eval_gamma_one() {
    local LAMBDA=$1
    local MODEL_ID="smoke_idf_clean_dis_v4_lambda${LAMBDA}_300v2"
    local CKPT="/home/fenglei/TS-RAG-main/TS-RAG/checkpoints/${MODEL_ID}/${MODEL_ID}_final.pth"
    local LOGFILE="logs/dump_gamma_v4_lambda${LAMBDA}_300v2.log"
    echo "[v4_lambda_sweep2] 开始eval模式饱和度诊断 lambda_sem=${LAMBDA}，日志: $LOGFILE"
    GAMMA_CKPT="$CKPT" GAMMA_AUGMODE="idf_clean_dis_v4" GAMMA_DATASET=ETTh1 python3 dump_gamma.py > "$LOGFILE" 2>&1
    echo "[v4_lambda_sweep2] lambda_sem=${LAMBDA} 饱和度诊断退出码: $?"
}

echo "==== v4_lambda_sweep2 开始: $(date) ===="
for L in 10 20 40 60 100; do
    train_one "$L"
    eval_gamma_one "$L"
done
echo "==== v4_lambda_sweep2 全部完成: $(date) ===="
