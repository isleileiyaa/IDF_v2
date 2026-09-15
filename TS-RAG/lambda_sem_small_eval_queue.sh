#!/bin/bash
set -u
source /home/fenglei/miniconda3/etc/profile.d/conda.sh
conda activate tsrag
cd /home/fenglei/TS-RAG-main/TS-RAG
export WANDB_MODE=offline
while kill -0 2112568 2>/dev/null; do sleep 20; done
echo "===== λ_sem训练(PID 2112568)已结束: $(date) ====="
for LAMBDA in 0.1 1; do
    MODEL_ID="idf_clean_dis_v4_lambda${LAMBDA}_full10000"
    CKPT="checkpoints/${MODEL_ID}/${MODEL_ID}_final.pth"
    if [ ! -f "$CKPT" ]; then
        echo "===== 跳过 lambda_sem=${LAMBDA}：checkpoint不存在 ====="
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
echo "===== λ_sem=0.1/1 全部评测完成: $(date) ====="
