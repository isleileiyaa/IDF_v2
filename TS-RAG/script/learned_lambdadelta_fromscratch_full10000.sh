cd /home/fenglei/TS-RAG-main/TS-RAG
export WANDB_MODE=offline
run_file=pretrain.py
top_k=10
retrieve_lookback_length=512
retrieval_database_path="${PRETRAIN_RETRIEVAL_DATABASE_PATH:-/home/fenglei/TS-RAG-main/retrieval_database/pretrain/retrieval_database_${retrieve_lookback_length}.parquet}"
augment_mode=idf_clean_dis_v4
context_length=512
prediction_length=64
lambda_delta="${LAMBDA_DELTA:-7.0}"
model_id="idf_clean_dis_v4_fusion_learned_lambdadelta${lambda_delta}_fromscratch_full10000"
checkpoints="${CHECKPOINTS_DIR:-/home/fenglei/TS-RAG-main/TS-RAG/checkpoints/${model_id}}"
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
rho1=0
rho2=1000
rho3=0
rho4=0
tau_dis=0.17
lambda_sem=0
lambda_ord=0
lambda_xcov=0
huber_kappa="${HUBER_KAPPA:-1.0}"
dyn_margin=${DYN_MARGIN:-1.0}
aux_loss_detach_ret=${AUX_LOSS_DETACH_RET:-true}

mkdir -p "$checkpoints"
echo "Launching pretrain (fromscratch, no init_from_checkpoint) with rho=($rho1, $rho2, $rho3, $rho4), tau_dis=$tau_dis, fusion_mode=learned, lambda_delta=$lambda_delta, huber_kappa=$huber_kappa"
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
    --rho1 $rho1 \
    --rho2 $rho2 \
    --rho3 $rho3 \
    --rho4 $rho4 \
    --tau_dis $tau_dis \
    --lambda_sem $lambda_sem \
    --lambda_ord $lambda_ord \
    --lambda_xcov $lambda_xcov \
    --lambda_delta $lambda_delta \
    --huber_kappa $huber_kappa \
    --dyn_margin $dyn_margin \
    --aux_loss_detach_ret $aux_loss_detach_ret \
    --freeze_chronos_bolt \
    --checkpoints $checkpoints
