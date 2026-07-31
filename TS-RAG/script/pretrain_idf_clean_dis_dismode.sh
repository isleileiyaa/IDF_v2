cd /home/fenglei/TS-RAG-main/TS-RAG
run_file=pretrain.py
top_k=10
retrieve_lookback_length=512
retrieval_database_path="${PRETRAIN_RETRIEVAL_DATABASE_PATH:-/home/fenglei/TS-RAG-main/retrieval_database/pretrain/retrieval_database_${retrieve_lookback_length}.parquet}"
augment_mode=idf_clean_dis
context_length=512
prediction_length=64
checkpoints="${CHECKPOINTS_DIR:-/home/fenglei/TS-RAG-main/TS-RAG/checkpoints}"
data_path="${PRETRAIN_DATA_PATH:-/home/fenglei/TS-RAG-main/datasets/pretrain/pretrain_pairs_ctx${retrieve_lookback_length}}"
train_steps=${TRAIN_STEPS:-10000}
evaluation_steps=${EVALUATION_STEPS:-10000}
optimizer=adamw
lr=0.0003
weight_decay=0.01
tmax=20
drop_prob=0.2
batch_size=256
shuffle_buffer_length=${SHUFFLE_BUFFER_LENGTH:-10000}
pretrained_model_path="${PRETRAINED_MODEL_PATH:-/home/fenglei/TS-RAG-main/TS-RAG/checkpoints/base/}"

# dis_mode: latent | output | both -- see ChronosBolt.py forward() for the gating logic.
#   latent -> only rho2 * L_dis(z_inv, z_dyn)                 [reproduces original RIDDE / Table 4]
#   output -> only rho_dis_output * L_dis(y_hat_inv, y_hat_dyn)
#   both   -> both terms, independent weights
dis_mode=${1:-${DIS_MODE:-output}}
rho2=${2:-${RHO2:-0.01}}
rho_dis_output=${3:-${RHO_DIS_OUTPUT:-0.01}}
dyn_margin=${DYN_MARGIN:-1.0}
aux_loss_detach_ret=${AUX_LOSS_DETACH_RET:-true}

model_id="data50m_${augment_mode}_${context_length}_pred${prediction_length}_lookback${retrieve_lookback_length}_top${top_k}_lr${lr}_drop${drop_prob}_${optimizer}_cosanneal_step${train_steps}_bs${batch_size}_no_embeddingtuning_dismode${dis_mode}_rho2_${rho2}_rhoout_${rho_dis_output}"
echo "Launching pretrain dis_mode=$dis_mode rho2=$rho2 rho_dis_output=$rho_dis_output"
echo "model_id=$model_id"
python $run_file \
    --model_id $model_id \
    --model ChronosBoltRetrieve \
    --top_k $top_k \
    --retrieve_lookback_length $retrieve_lookback_length \
    --retrieval_database_path $retrieval_database_path \
    --augment_mode $augment_mode \
    --pretrained_model_path $pretrained_model_path \
    --context_length $context_length \
    --prediction_length $prediction_length \
    --data_path $data_path \
    --train_steps $train_steps \
    --evaluation_steps $evaluation_steps \
    --optimizer $optimizer \
    --learning_rate $lr \
    --weight_decay $weight_decay \
    --tmax $tmax \
    --drop_prob $drop_prob \
    --batch_size $batch_size \
    --grad_clip_value 1.0 \
    --shuffle_buffer_length $shuffle_buffer_length \
    --rho1 0 \
    --rho2 $rho2 \
    --rho3 0 \
    --rho4 0 \
    --dis_mode $dis_mode \
    --rho_dis_output $rho_dis_output \
    --dyn_margin $dyn_margin \
    --aux_loss_detach_ret $aux_loss_detach_ret \
    --freeze_chronos_bolt \
    --checkpoints $checkpoints
