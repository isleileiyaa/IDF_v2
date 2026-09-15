#!/bin/bash
# idf_clean_dis_v3 正式10000步训练：三档tau_dis(0.20/0.17/0.13)，rho2=1000固定，
# 依次顺序跑(不同时抢同一块卡)。每条命令独立日志，某一条报错不阻塞后面的继续跑。
set -u
export WANDB_MODE=offline
mkdir -p logs

CKPT="checkpoints/data50m_idf_clean_dis_512_pred64_lookback512_top10_lr0.0003_drop0.2_adamw_cosanneal_step10000_bs256_no_embeddingtuning_rho0_0.01_0_0_final.pth"

run_one() {
    local TAU_LABEL=$1
    local TAU_VALUE=$2
    local MODEL_ID="idf_clean_dis_v3_full10000_tau${TAU_LABEL}"
    local CKPT_DIR="checkpoints/${MODEL_ID}"
    local LOGFILE="logs/v3_full10000_tau${TAU_LABEL}.log"

    echo "=================================================================="
    echo "[v3_full_run] 开始 tau_dis=${TAU_VALUE} (label=${TAU_LABEL})，日志: ${LOGFILE}"
    echo "=================================================================="
    mkdir -p "$CKPT_DIR"

    python3 pretrain.py \
        --model_id "$MODEL_ID" \
        --model ChronosBoltRetrieve \
        --top_k 10 \
        --retrieve_lookback_length 512 \
        --retrieval_database_path /home/fenglei/TS-RAG-main/retrieval_database/pretrain/retrieval_database_512.parquet \
        --augment_mode idf_clean_dis_v3 \
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
        --tau_dis "$TAU_VALUE" \
        --freeze_chronos_bolt \
        --init_from_checkpoint "$CKPT" \
        --checkpoints "$CKPT_DIR" \
        > "$LOGFILE" 2>&1
    local EXIT_CODE=$?
    echo "[v3_full_run] tau_dis=${TAU_VALUE} 退出码: ${EXIT_CODE}"
    return 0   # 无论成功失败都返回0，保证调用方(;分隔)继续往下走
}

run_one "020" "0.20"
run_one "017" "0.17"
run_one "013" "0.13"

echo "=================================================================="
echo "[v3_full_run] 三条全部尝试完毕: $(date)"
echo "=================================================================="
