#!/bin/bash
# 在同一张卡上按顺序跑多个lambda_ord值(每个都是完整的10000步fromscratch训练+评测)。
# 用法: CUDA_VISIBLE_DEVICES=0 bash overnight_additive_ord_queue.sh 0.001 1
set -u
cd /home/fenglei/TS-RAG-main/TS-RAG
for LAMBDA_ORD in "$@"; do
    echo "==== queue: 开始 lambda_ord=${LAMBDA_ORD} $(date) ===="
    LAMBDA_ORD="$LAMBDA_ORD" bash overnight_additive_ord_fromscratch.sh
    echo "==== queue: 完成 lambda_ord=${LAMBDA_ORD} $(date) ===="
done
echo "==== queue: 全部lambda_ord跑完 $(date) ===="
