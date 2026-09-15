#!/bin/bash
# 用PRETRAIN_SEED=2022重跑baseline(idf_clean_dis, rho2=0.01)和三档tau(idf_clean_dis_v3,
# rho2=1000)，各10000步训练+7数据集zeroshot评测，衡量seed2021那次的差异有多少是run间
# 噪声。每条训练/评测独立，某一条失败不阻塞后面继续。
set -u
export WANDB_MODE=offline
mkdir -p logs

DATASETS7="ETTh1 ETTh2 ETTm1 ETTm2 weather exchange_rate electricity"
V3_INIT_CKPT="checkpoints/data50m_idf_clean_dis_512_pred64_lookback512_top10_lr0.0003_drop0.2_adamw_cosanneal_step10000_bs256_no_embeddingtuning_rho0_0.01_0_0_final.pth"

train_baseline() {
    local MODEL_ID="idf_clean_dis_baseline_seed2022"
    local CKPT_DIR="checkpoints/${MODEL_ID}"
    local LOGFILE="logs/v3_seed2022_train_baseline.log"
    mkdir -p "$CKPT_DIR"
    echo "[v3_seed2022] 开始训练 baseline(seed2022)，日志: $LOGFILE"
    PRETRAIN_SEED=2022 python3 pretrain.py \
        --model_id "$MODEL_ID" \
        --model ChronosBoltRetrieve \
        --top_k 10 \
        --retrieve_lookback_length 512 \
        --retrieval_database_path /home/fenglei/TS-RAG-main/retrieval_database/pretrain/retrieval_database_512.parquet \
        --augment_mode idf_clean_dis \
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
        --rho1 0 --rho2 0.01 --rho3 0 --rho4 0 \
        --dyn_margin 1.0 \
        --aux_loss_detach_ret true \
        --freeze_chronos_bolt \
        --checkpoints "$CKPT_DIR" \
        > "$LOGFILE" 2>&1
    echo "[v3_seed2022] baseline(seed2022) 训练退出码: $?"
}

train_v3() {
    local TAU_LABEL=$1
    local TAU_VALUE=$2
    local MODEL_ID="idf_clean_dis_v3_full10000_tau${TAU_LABEL}_seed2022"
    local CKPT_DIR="checkpoints/${MODEL_ID}"
    local LOGFILE="logs/v3_seed2022_train_tau${TAU_LABEL}.log"
    mkdir -p "$CKPT_DIR"
    echo "[v3_seed2022] 开始训练 tau=${TAU_VALUE}(seed2022)，日志: $LOGFILE"
    PRETRAIN_SEED=2022 python3 pretrain.py \
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
        --init_from_checkpoint "$V3_INIT_CKPT" \
        --checkpoints "$CKPT_DIR" \
        > "$LOGFILE" 2>&1
    echo "[v3_seed2022] tau=${TAU_VALUE}(seed2022) 训练退出码: $?"
}

eval_one() {
    local TAG=$1
    local AUGMODE=$2
    local CKPT=$3
    local LOGFILE="logs/v3_seed2022_eval_${TAG}.log"
    echo "[v3_seed2022] 开始评测 $TAG (augment_mode=$AUGMODE)，日志: $LOGFILE"
    if [ ! -f "$CKPT" ]; then
        echo "[v3_seed2022] 跳过评测 $TAG：checkpoint不存在 ($CKPT)，训练大概率失败了"
        return 0
    fi
    DATASETS="$DATASETS7" AUGMENT_MODE="$AUGMODE" SAVE_SUFFIX="eval_${TAG}_seed2022" \
        CHECKPOINT_MODEL_PATH="$CKPT" \
        bash script/zeroshot_chronos_idf_clean_dis.sh > "$LOGFILE" 2>&1
    echo "[v3_seed2022] 评测 $TAG 退出码: $?"
}

echo "=================================================================="
echo "[v3_seed2022] 开始: $(date)"
echo "=================================================================="

train_baseline
eval_one "baseline" "idf_clean_dis" "checkpoints/idf_clean_dis_baseline_seed2022/idf_clean_dis_baseline_seed2022_final.pth"

train_v3 "020" "0.20"
eval_one "tau020" "idf_clean_dis_v3" "checkpoints/idf_clean_dis_v3_full10000_tau020_seed2022/idf_clean_dis_v3_full10000_tau020_seed2022_final.pth"

train_v3 "017" "0.17"
eval_one "tau017" "idf_clean_dis_v3" "checkpoints/idf_clean_dis_v3_full10000_tau017_seed2022/idf_clean_dis_v3_full10000_tau017_seed2022_final.pth"

train_v3 "013" "0.13"
eval_one "tau013" "idf_clean_dis_v3" "checkpoints/idf_clean_dis_v3_full10000_tau013_seed2022/idf_clean_dis_v3_full10000_tau013_seed2022_final.pth"

echo "=================================================================="
echo "[v3_seed2022] 全部完成: $(date)"
echo "=================================================================="
