#!/bin/bash
# tau(Eq.19置信度权重温度参数)200步快速诊断：tau=1/5/20/50，观察c_i分布变化。
# 只是诊断，不代表最终训练要用哪个值。
set -u
source /home/fenglei/miniconda3/etc/profile.d/conda.sh
conda activate tsrag
cd /home/fenglei/TS-RAG-main/TS-RAG
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline
mkdir -p logs

V3_CKPT="checkpoints/idf_clean_dis_v3_full10000_tau017/idf_clean_dis_v3_full10000_tau017_final.pth"

for TAU in 1 5 20 50; do
    MODEL_ID="tau_probe_${TAU}"
    CKPT_DIR="/tmp/tau_probe_${TAU}"
    mkdir -p "$CKPT_DIR"
    echo "===== 开始 tau=${TAU} 200步诊断 ====="
    python3 pretrain.py \
        --model_id "$MODEL_ID" --model ChronosBoltRetrieve --top_k 10 \
        --retrieve_lookback_length 512 \
        --retrieval_database_path /home/fenglei/TS-RAG-main/retrieval_database/pretrain/retrieval_database_512.parquet \
        --augment_mode idf_clean_dis_v4 \
        --pretrained_model_path /home/fenglei/TS-RAG-main/TS-RAG/checkpoints/base/ \
        --context_length 512 --prediction_length 64 \
        --data_path /home/fenglei/TS-RAG-main/datasets/pretrain/pretrain_pairs_ctx512 \
        --train_steps 200 --evaluation_steps 200 \
        --optimizer adamw --learning_rate 0.0003 --weight_decay 0.01 --tmax 20 \
        --drop_prob 0.2 --batch_size 256 --grad_clip_value 1.0 --shuffle_buffer_length 2000 \
        --rho1 0 --rho2 1000 --rho3 0 --rho4 0 --dyn_margin 1.0 --aux_loss_detach_ret true \
        --tau_dis 0.17 --lambda_sem 0 --lambda_ord 0.01 --tau "$TAU" \
        --freeze_chronos_bolt --init_from_checkpoint "$V3_CKPT" --checkpoints "$CKPT_DIR" \
        > "logs/tau_probe_${TAU}.log" 2>&1
    echo "===== tau=${TAU} 结束，exit code $? ====="
    rm -rf "$CKPT_DIR"
done
echo "==== tau_probe_sweep 全部完成 ===="
