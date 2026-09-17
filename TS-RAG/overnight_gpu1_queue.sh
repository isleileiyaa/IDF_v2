#!/bin/bash
# GPU1过夜队列: tau_dis sweep(0.13, 0.25) -> lambda_delta=10.0 -> lambda_delta=15.0
set -u
cd /home/fenglei/TS-RAG-main/TS-RAG
export CUDA_VISIBLE_DEVICES=1

echo "==== gpu1 queue: 开始 tau_dis=0.13 $(date) ===="
TAU_DIS=0.13 bash overnight_v3_taudis.sh
echo "==== gpu1 queue: 完成 tau_dis=0.13 $(date) ===="

echo "==== gpu1 queue: 开始 tau_dis=0.25 $(date) ===="
TAU_DIS=0.25 bash overnight_v3_taudis.sh
echo "==== gpu1 queue: 完成 tau_dis=0.25 $(date) ===="

echo "==== gpu1 queue: 开始 lambda_delta=10.0 $(date) ===="
LAMBDA_DELTA=10.0 bash overnight_learned_lambdadelta.sh
echo "==== gpu1 queue: 完成 lambda_delta=10.0 $(date) ===="

echo "==== gpu1 queue: 开始 lambda_delta=15.0 $(date) ===="
LAMBDA_DELTA=15.0 bash overnight_learned_lambdadelta.sh
echo "==== gpu1 queue: 完成 lambda_delta=15.0 $(date) ===="

echo "==== gpu1 queue: 全部完成 $(date) ===="
