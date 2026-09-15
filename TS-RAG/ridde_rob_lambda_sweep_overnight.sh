#!/bin/bash
# RIDDE_rob lambda_rob 夜间扫描：新baseline(clean_pred)下，对 0.5/1/2/4/8 这几个候选值
# 各跑一次完整10000步训练 + 7数据集zeroshot评测，每个候选跑完立刻清理掉中间resume快照
# (model_steps*/optim_steps*)只保留final.pth，避免5个候选的中间产物同时占盘把磁盘写满。
# 单个候选失败(训练或评测报错)不影响后续候选继续跑——夜里没人盯着，不能因为一个候选
# 出问题就让整晚白跑。
set -u
export WANDB_MODE=offline
LOG_PREFIX="[RIDDE_rob_sweep]"
mkdir -p logs

INIT_CKPT="checkpoints/data50m_idf_trr_dualpath_learnfuse_512_pred64_lookback512_top10_lr0.0003_drop0.2_adamw_cosanneal_step10000_bs256_no_embeddingtuning_lambdasep0.0_final.pth"
NEW_F0="checkpoints/truebase_rerun/data50m_baseline_512_pred64_lookback512_top10_lr0.0003_drop0.2_adamw_cosanneal_step10000_bs256_no_embeddingtuning_seed2021_final.pth"

LAMBDAS="0.5 1 2 4 8"

echo "$LOG_PREFIX 开始时间: $(date)"
echo "$LOG_PREFIX 候选lambda_rob值: $LAMBDAS"

for LAM in $LAMBDAS; do
    TAG=$(echo "$LAM" | tr '.' '_')
    MODEL_ID="data50m_idf_trr_dualpath_learnfuse_rob_512_pred64_lookback512_top10_lr0.0003_drop0.2_adamw_cosanneal_step10000_bs256_no_embeddingtuning_lambdasep0.0_lambdarob${TAG}_newbaseline_sweep"
    CKPT_DIR="checkpoints/ridde_rob_sweep_lambda${TAG}"
    TRAIN_LOG="logs/ridde_rob_sweep_train_lambda${TAG}_$(date +%Y%m%d_%H%M%S).log"
    EVAL_LOG="logs/ridde_rob_sweep_eval_lambda${TAG}_$(date +%Y%m%d_%H%M%S).log"
    FINAL_CKPT="${CKPT_DIR}/${MODEL_ID}_final.pth"

    echo ""
    echo "=================================================================="
    echo "$LOG_PREFIX [lambda_rob=$LAM] 开始训练，日志: $TRAIN_LOG"
    echo "=================================================================="
    mkdir -p "$CKPT_DIR"

    python3 pretrain.py \
        --model_id "$MODEL_ID" \
        --model ChronosBoltRetrieve \
        --top_k 10 \
        --retrieve_lookback_length 512 \
        --retrieval_database_path /home/fenglei/TS-RAG-main/retrieval_database/pretrain/retrieval_database_512.parquet \
        --augment_mode idf_trr_dualpath_learnfuse \
        --pretrained_model_path /home/fenglei/TS-RAG-main/TS-RAG/checkpoints/base/ \
        --context_length 512 \
        --prediction_length 64 \
        --data_path /home/fenglei/TS-RAG-main/datasets/pretrain/pretrain_pairs_ctx512 \
        --train_steps 10000 \
        --evaluation_steps 1000 \
        --optimizer adamw \
        --learning_rate 0.0003 \
        --weight_decay 0.01 \
        --tmax 20 \
        --drop_prob 0.2 \
        --batch_size 256 \
        --grad_clip_value 1.0 \
        --shuffle_buffer_length 10000 \
        --lambda_sep 0.0 \
        --lambda_rob "$LAM" \
        --kl_radius 0.02 \
        --rob_inner_steps 3 \
        --rob_step_size 0.2 \
        --rob_random_restarts 1 \
        --rob_bisection_steps 24 \
        --init_from_checkpoint "$INIT_CKPT" \
        --f0_checkpoint_path "$NEW_F0" \
        --freeze_chronos_bolt \
        --checkpoints "$CKPT_DIR" \
        > "$TRAIN_LOG" 2>&1
    TRAIN_EXIT=$?
    echo "$LOG_PREFIX [lambda_rob=$LAM] 训练退出码: $TRAIN_EXIT"

    if [ "$TRAIN_EXIT" -ne 0 ] || [ ! -f "$FINAL_CKPT" ]; then
        echo "$LOG_PREFIX [lambda_rob=$LAM] 训练失败或没有产出final checkpoint，跳过这个候选的zeroshot评测，继续下一个候选。"
        continue
    fi

    echo "$LOG_PREFIX [lambda_rob=$LAM] 训练完成，开始7数据集zeroshot评测，日志: $EVAL_LOG"
    DATASETS="ETTh1 ETTh2 ETTm1 ETTm2 weather exchange_rate electricity" \
        AUGMENT_MODE="idf_trr_dualpath_learnfuse" \
        SAVE_SUFFIX="ridde_rob_sweep_lambda${TAG}" \
        CHECKPOINT_MODEL_PATH="$FINAL_CKPT" \
        bash script/zeroshot_chronos_idf_trr_dualpath.sh > "$EVAL_LOG" 2>&1
    EVAL_EXIT=$?
    echo "$LOG_PREFIX [lambda_rob=$LAM] 评测退出码: $EVAL_EXIT"

    # 清理中间resume快照(model_steps*/optim_steps*)，只留final.pth，控制磁盘占用。
    SUBDIR="${CKPT_DIR}/${MODEL_ID}"
    if [ -d "$SUBDIR" ]; then
        rm -rf "$SUBDIR"
        echo "$LOG_PREFIX [lambda_rob=$LAM] 已清理中间resume快照目录: $SUBDIR"
    fi
    echo "$LOG_PREFIX [lambda_rob=$LAM] 完成。剩余磁盘: $(df -h /home | tail -1)"
done

echo ""
echo "=================================================================="
echo "$LOG_PREFIX 全部候选跑完，结束时间: $(date)"
echo "$LOG_PREFIX 结果文件: results/forecast_evaluation/zeroshot_chronos_idf_trr_dualpath_ridde_rob_sweep_lambda*.txt"
echo "=================================================================="
