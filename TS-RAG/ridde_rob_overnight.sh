#!/bin/bash
set -e
export WANDB_MODE=offline
LOG_PREFIX="[RIDDE_rob_overnight]"
echo "$LOG_PREFIX 开始时间: $(date)"

# OLD_F0: 9月4日那个来源存疑、已被排查过的旧checkpoint，只用来做对比参照。
# NEW_F0: 这次重训(CHECKPOINTS_DIR=checkpoints/truebase_rerun)产出的、来源干净
# 可追溯的新checkpoint——鲁棒训练实际要用的f_0是这一份，不是OLD_F0。
OLD_F0="checkpoints/data50m_baseline_512_pred64_lookback512_top10_lr0.0003_drop0.2_adamw_cosanneal_step10000_bs256_no_embeddingtuning_seed2021_final.pth"
NEW_F0="checkpoints/truebase_rerun/data50m_baseline_512_pred64_lookback512_top10_lr0.0003_drop0.2_adamw_cosanneal_step10000_bs256_no_embeddingtuning_seed2021_final.pth"
INIT_CKPT="checkpoints/data50m_idf_trr_dualpath_learnfuse_512_pred64_lookback512_top10_lr0.0003_drop0.2_adamw_cosanneal_step10000_bs256_no_embeddingtuning_lambdasep0.0_final.pth"

# ---- 第1步：确认TrueBase重训确实是刚跑完的、干净的 ----
TRUEBASE_LOG=$(ls -t logs/truebase_rerun_*.log 2>/dev/null | head -1)
if [ -z "$TRUEBASE_LOG" ]; then
    echo "$LOG_PREFIX 致命错误：找不到logs/truebase_rerun_*.log，说明重训没有按预期落日志。停止，不往下走。"
    exit 1
fi
echo "$LOG_PREFIX 使用日志: $TRUEBASE_LOG"

# 从日志里抽取Trainable parameters:后面的参数名单，必须全部包含output_patch_embedding，
# 一个都不能是别的模块——这是这次重训要拿到的直接证据。
# 注意：参数名单后面没有空行、直接紧跟着tqdm的\r进度条(实测确认过)，不能靠"空行"
# 判断截止；改成"只要这一行是纯粹的点号分隔标识符(没有空格/百分号等)就继续收，
# 遇到第一行不是这个格式的(也就是tqdm那一行)就停"。
TRAINABLE_BLOCK=$(awk '/^Trainable parameters:/{flag=1; next} flag && $0 ~ /^[A-Za-z_][A-Za-z0-9_.]*$/{print; next} flag{flag=0}' "$TRUEBASE_LOG")
if [ -z "$TRAINABLE_BLOCK" ]; then
    echo "$LOG_PREFIX 致命错误：日志里没抓到Trainable parameters:清单(可能训练还没真正开始或者日志格式跟预期不一样)。停止，不往下走，等人工检查。"
    exit 1
fi
echo "$LOG_PREFIX 可训练参数清单："
echo "$TRAINABLE_BLOCK"
BAD_LINES=$(echo "$TRAINABLE_BLOCK" | grep -v "output_patch_embedding" || true)
if [ -n "$BAD_LINES" ]; then
    echo "$LOG_PREFIX 致命错误：可训练参数里出现了不含output_patch_embedding的key："
    echo "$BAD_LINES"
    echo "$LOG_PREFIX 说明这次重训跟预期的'只解冻output_patch_embedding'不一致，停止，不往下走，等人工检查。"
    exit 1
fi
echo "$LOG_PREFIX 验证通过：可训练参数确实只有output_patch_embedding相关的key。"

# ---- 第2步：确认新checkpoint文件(NEW_F0)确实是这次重训刚写出来的(mtime要新) ----
if [ ! -f "$NEW_F0" ]; then
    echo "$LOG_PREFIX 致命错误：$NEW_F0 不存在，重训似乎没有成功写出checkpoint。停止。"
    exit 1
fi
CKPT_AGE_MIN=$(( ( $(date +%s) - $(stat -c %Y "$NEW_F0") ) / 60 ))
echo "$LOG_PREFIX 新f0 checkpoint($NEW_F0) mtime距现在: ${CKPT_AGE_MIN} 分钟"
if [ "$CKPT_AGE_MIN" -gt 60 ]; then
    echo "$LOG_PREFIX 警告：这个checkpoint的mtime超过60分钟，可能不是刚重训出来的那个文件。停止，等人工检查，不要假设它是新的。"
    exit 1
fi
echo "$LOG_PREFIX 确认：checkpoint mtime够新，可以认为是这次重训的产物。"

# ---- 第3步：新旧checkpoint的zeroshot对比(仅供参考，不作为通过/失败判据，只是留证据) ----
echo "$LOG_PREFIX 用新checkpoint(NEW_F0)跑zeroshot评估(2个数据集，够快)..."
DATASETS="ETTh1 ETTh2" CHECKPOINT_MODEL_PATH="$NEW_F0" SAVE_FILE_NAME="zeroshot_chronos_truebase_new.txt" \
  bash script/zeroshot_chronos_truebase.sh 2>&1 | tee "logs/truebase_rerun_eval_new_$(date +%Y%m%d_%H%M%S).log"

echo "$LOG_PREFIX 用旧checkpoint(OLD_F0)跑同样两个数据集，供对比..."
DATASETS="ETTh1 ETTh2" CHECKPOINT_MODEL_PATH="$OLD_F0" SAVE_FILE_NAME="zeroshot_chronos_truebase_old.txt" \
  bash script/zeroshot_chronos_truebase.sh 2>&1 | tee "logs/truebase_rerun_eval_old_$(date +%Y%m%d_%H%M%S).log"

echo "$LOG_PREFIX 新旧checkpoint评估结果已存档(results/forecast_evaluation/zeroshot_chronos_truebase_new.txt / _old.txt)，供你早上review，不影响下面是否继续。"

# ---- 第4步：CPU自测(STEP3)再跑一遍，确认代码状态没有被这段时间的任何改动破坏 ----
python3 utils/ridde_robust_loss.py
echo "$LOG_PREFIX ridde_robust_loss.py自测: 见上方PASS输出"

# ---- 第5步：冒烟测试(真实GPU，小步数，用刚验证过的NEW_F0) ----
echo "$LOG_PREFIX 开始冒烟测试(小步数)..."
# TRAIN_STEPS(50) < EVALUATION_STEPS(1000)时，pretrain.py中途的周期性保存(会自动
# mkdir)不会触发，训练结束时的最终保存又不会自己建目录——这是之前反复踩过的老
# 毛病，这里提前建好目录，避免冒烟测试在最后一步(保存checkpoint)时才崩溃。
mkdir -p checkpoints/smoke_ridde_rob
python3 pretrain.py \
    --model_id smoke_test_ridde_rob \
    --model ChronosBoltRetrieve \
    --top_k 10 \
    --retrieve_lookback_length 512 \
    --retrieval_database_path /home/fenglei/TS-RAG-main/retrieval_database/pretrain/retrieval_database_512.parquet \
    --augment_mode idf_trr_dualpath_learnfuse \
    --pretrained_model_path /home/fenglei/TS-RAG-main/TS-RAG/checkpoints/base/ \
    --context_length 512 \
    --prediction_length 64 \
    --data_path /home/fenglei/TS-RAG-main/datasets/pretrain/pretrain_pairs_ctx512 \
    --train_steps 50 \
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
    --lambda_rob 0.1 \
    --kl_radius 0.02 \
    --rob_inner_steps 3 \
    --rob_step_size 0.2 \
    --rob_random_restarts 1 \
    --rob_bisection_steps 24 \
    --init_from_checkpoint "$INIT_CKPT" \
    --f0_checkpoint_path "$NEW_F0" \
    --freeze_chronos_bolt \
    --checkpoints checkpoints/smoke_ridde_rob \
    2>&1 | tee "logs/smoke_test_ridde_rob_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "=================================================================="
echo "$LOG_PREFIX 冒烟测试跑完，到此为止——不会自动往下启动正式训练。"
echo "$LOG_PREFIX 请在上面的日志里人工确认这几件事，再决定要不要正式训练："
echo "$LOG_PREFIX 1. 50步里有没有报错/NaN/Inf"
echo "$LOG_PREFIX 2. loss_rob 是不是一个合理的正数(不是0，也不是异常大)"
echo "$LOG_PREFIX 3. active_fraction 是不是在0~1之间的一个正常值(不是恒为0，也不是恒为1)"
echo "$LOG_PREFIX 4. rob_kl_max 是不是接近但不超过kl_radius=0.02(允许小的数值误差)"
echo "$LOG_PREFIX 结束时间: $(date)"
echo "=================================================================="
