cd /home/fenglei/TS-RAG-main/TS-RAG
run_file=pretrain.py
top_k=10
retrieve_lookback_length=512
retrieval_database_path="${PRETRAIN_RETRIEVAL_DATABASE_PATH:-/home/fenglei/TS-RAG-main/retrieval_database/pretrain/retrieval_database_${retrieve_lookback_length}.parquet}"
augment_mode=idf_ridde_v2
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
# Stage gating (RIDDE_正则项消融实验清单.md): stage2 = rho_sem only,
# stage3 = +rho_xcov, stage4 = +rho_ord. tau (retrieval-confidence temperature,
# Eq.19) reuses the existing --tau flag.
rho_sem=${1:-${RHO_SEM:-0}}
rho_xcov=${2:-${RHO_XCOV:-0}}
rho_ord=${3:-${RHO_ORD:-0}}
tau=${TAU:-0.1}
ord_margin=${ORD_MARGIN:-0.0}

model_id="data50m_${augment_mode}_${context_length}_pred${prediction_length}_lookback${retrieve_lookback_length}_top${top_k}_lr${lr}_drop${drop_prob}_${optimizer}_cosanneal_step${train_steps}_bs${batch_size}_no_embeddingtuning_sem${rho_sem}_xcov${rho_xcov}_ord${rho_ord}"
echo "Launching pretrain with (rho_sem, rho_xcov, rho_ord)=($rho_sem, $rho_xcov, $rho_ord)"
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
    --rho_sem $rho_sem \
    --rho_xcov $rho_xcov \
    --rho_ord $rho_ord \
    --tau $tau \
    --ord_margin $ord_margin \
    --freeze_chronos_bolt \
    --checkpoints $checkpoints
