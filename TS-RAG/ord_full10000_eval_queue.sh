#!/bin/bash
set -u
source /home/fenglei/miniconda3/etc/profile.d/conda.sh
conda activate tsrag
cd /home/fenglei/TS-RAG-main/TS-RAG
export WANDB_MODE=offline
while kill -0 2112569 2>/dev/null; do sleep 20; done
echo "===== rho_ord训练(PID 2112569)已结束: $(date) ====="
for R in 0.0001 0.01 0.1 1.0; do
    CKPT="checkpoints/ord_full10000_rho${R}/data50m_idf_ridde_v2_512_pred64_lookback512_top10_lr0.0003_drop0.2_adamw_cosanneal_step10000_bs256_no_embeddingtuning_sem0_xcov0_ord${R}_final.pth"
    if [ ! -f "$CKPT" ]; then
        echo "===== 跳过 rho_ord=${R}：checkpoint不存在($CKPT) ====="
        continue
    fi
    echo "===== 开始评测 rho_ord=${R} ====="
    DATASETS="ETTh1 ETTh2 ETTm1 ETTm2 weather exchange_rate" \
        SAVE_SUFFIX="ord_full10000_${R}" \
        CHECKPOINT_MODEL_PATH="$CKPT" \
        bash script/zeroshot_chronos_idf_ridde_v2.sh \
        > "logs/zeroshot_ord_full10000_${R}.log" 2>&1
    echo "===== rho_ord=${R} 评测结束，exit code $? ====="
done
echo "===== rho_ord 4档全部评测完成: $(date) ====="
