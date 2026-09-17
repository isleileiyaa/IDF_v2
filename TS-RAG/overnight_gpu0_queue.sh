#!/bin/bash
# GPU0过夜队列: tau_dis sweep(0.10, 0.20) -> lambda_delta噪声底噪检验(seed=1234) -> lambda_delta=5.0
set -u
cd /home/fenglei/TS-RAG-main/TS-RAG
export CUDA_VISIBLE_DEVICES=0

echo "==== gpu0 queue: 开始 tau_dis=0.10 $(date) ===="
TAU_DIS=0.10 bash overnight_v3_taudis.sh
echo "==== gpu0 queue: 完成 tau_dis=0.10 $(date) ===="

echo "==== gpu0 queue: 开始 tau_dis=0.20 $(date) ===="
TAU_DIS=0.20 bash overnight_v3_taudis.sh
echo "==== gpu0 queue: 完成 tau_dis=0.20 $(date) ===="

echo "==== gpu0 queue: 开始 lambda_delta=7.0 seedprobe(1234) $(date) ===="
PROBE_SEED=1234 bash overnight_lambdadelta_seedprobe.sh
echo "==== gpu0 queue: 完成 lambda_delta=7.0 seedprobe(1234) $(date) ===="

echo "==== gpu0 queue: 开始 lambda_delta=5.0 $(date) ===="
LAMBDA_DELTA=5.0 bash overnight_learned_lambdadelta.sh
echo "==== gpu0 queue: 完成 lambda_delta=5.0 $(date) ===="

echo "==== gpu0 queue: 全部完成 $(date) ===="
