#!/bin/bash
# idf_ridde_v2 rho_ord=0.0001/0.01/0.1/1.0 完整10000步训练，GPU1。
# 用脚本自身的train_steps/evaluation_steps默认值(10000)，不覆盖。
set -u
source /home/fenglei/miniconda3/etc/profile.d/conda.sh
conda activate tsrag
cd /home/fenglei/TS-RAG-main/TS-RAG
export CUDA_VISIBLE_DEVICES=1
export WANDB_MODE=offline
mkdir -p logs

echo "==== ord_full10000 开始: $(date) ===="
for RHO_ORD in 0.0001 0.01 0.1 1.0; do
    export CHECKPOINTS_DIR="checkpoints/ord_full10000_rho${RHO_ORD}"
    mkdir -p "$CHECKPOINTS_DIR"
    echo "===== 开始 rho_ord=${RHO_ORD}（GPU1，10000步完整训练）====="
    free -h
    bash script/pretrain_idf_ridde_v2.sh 0 0 ${RHO_ORD} 2>&1 | tee "logs/ord_full10000_${RHO_ORD}.log"
    echo "===== rho_ord=${RHO_ORD} 结束 ====="
done
echo "==== ord_full10000 全部完成: $(date) ===="
