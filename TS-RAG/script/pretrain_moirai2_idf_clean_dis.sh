cd /home/fenglei/TS-RAG-main/TS-RAG
run_file=pretrain.py
top_k=10
retrieve_lookback_length=512
retrieval_database_path="${PRETRAIN_RETRIEVAL_DATABASE_PATH:-/home/fenglei/TS-RAG-main/retrieval_database/pretrain/retrieval_database_${retrieve_lookback_length}.parquet}"
# augment_mode selects which fusion head pretrain.py attaches to the frozen
# Moirai2 backbone (see models/Moirai2.py): 'idf_clean_dis' for the RIDDE
# port, 'moe' for the plain moe-fusion port (script/pretrain_moirai2_moe.sh).
augment_mode=idf_clean_dis
context_length=512
prediction_length=64
checkpoints="${CHECKPOINTS_DIR:-/home/fenglei/TS-RAG-main/TS-RAG/checkpoints}"
data_path="${PRETRAIN_DATA_PATH:-/home/fenglei/TS-RAG-main/datasets/pretrain/pretrain_pairs_ctx${retrieve_lookback_length}}"
train_steps=${TRAIN_STEPS:-10000}
# evaluation_steps < train_steps so the CosineAnnealingLR scheduler actually
# fires more than once and real decay happens over the run (with
# evaluation_steps==train_steps it barely moves). lr tuned via a 4-round,
# 23-config sweep on 6 datasets (ETTh1/ETTh2/ETTm1/ETTm2/weather/exchange_rate):
# lr=0.0003 (old default) -> avg mse 0.2635; lr=0.000015 -> avg mse 0.2380.
evaluation_steps=${EVALUATION_STEPS:-1000}
optimizer=adamw
lr=0.000015
weight_decay=0.01
tmax=20
drop_prob=0.2
batch_size=256
shuffle_buffer_length=${SHUFFLE_BUFFER_LENGTH:-10000}

model_id="moirai2_${augment_mode}_${context_length}_pred${prediction_length}_lookback${retrieve_lookback_length}_top${top_k}_lr${lr}_drop${drop_prob}_${optimizer}_cosanneal_step${train_steps}_bs${batch_size}"
echo "Launching Moirai2 RIDDE pretrain"
echo "model_id=$model_id"
python $run_file \
    --model_id $model_id \
    --model Moirai2Retrieve \
    --top_k $top_k \
    --retrieve_lookback_length $retrieve_lookback_length \
    --retrieval_database_path $retrieval_database_path \
    --augment_mode $augment_mode \
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
    --freeze_chronos_bolt \
    --checkpoints $checkpoints
