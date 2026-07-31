#!/bin/bash
# Full-scale disentanglement-loss ablation sweep for RIDDE (idf_clean_dis).
# For each (dis_mode, rho2, rho_dis_output) combo: 10000-step pretrain, then
# zero-shot eval on the first 7 datasets. Sequential on a single GPU.
set -uo pipefail
cd /home/fenglei/TS-RAG-main/TS-RAG
source /home/fenglei/miniconda3/etc/profile.d/conda.sh
conda activate tsrag
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline
export TRAIN_STEPS=10000
export EVALUATION_STEPS=10000
export DATASETS="ETTh1 ETTh2 ETTm1 ETTm2 weather exchange_rate electricity"

mkdir -p logs/dis_ablation

# combos: dis_mode rho2 rho_dis_output
combos=(
  "output 0 0.01"
  "output 0 0.1"
  "output 0 1"
  "both 0.01 0.01"
  "both 0.01 0.1"
  "both 0.01 1"
)

for combo in "${combos[@]}"; do
  read -r dis_mode rho2 rho_out <<< "$combo"
  tag="dismode${dis_mode}_rho2_${rho2}_rhoout_${rho_out}"
  echo "=== [$(date)] PRETRAIN start: $tag ==="
  bash script/pretrain_idf_clean_dis_dismode.sh "$dis_mode" "$rho2" "$rho_out" \
    > "logs/dis_ablation/pretrain_${tag}.log" 2>&1
  echo "=== [$(date)] PRETRAIN done: $tag (exit $?) ==="

  echo "=== [$(date)] ZEROSHOT start: $tag ==="
  SAVE_FILE_NAME="zeroshot_${tag}.txt" \
  bash script/zeroshot_chronos_idf_clean_dis_dismode.sh "$dis_mode" "$rho2" "$rho_out" \
    > "logs/dis_ablation/zeroshot_${tag}.log" 2>&1
  echo "=== [$(date)] ZEROSHOT done: $tag (exit $?) ==="
done

echo "=== ALL COMBOS DONE ==="
