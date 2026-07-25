cd /home/fenglei/TS-RAG-main/TS-RAG
run_file=pretrain.py

# model
top_k=10
retrieve_lookback_length=512
retrieval_database_path="${PRETRAIN_RETRIEVAL_DATABASE_PATH:-/home/fenglei/TS-RAG-main/retrieval_database/pretrain/retrieval_database_${retrieve_lookback_length}.parquet}"
augment_mode=moe
context_length=512
prediction_length=64

# training parameters
train_steps=10000
evaluation_steps=1000
optimizer=adamw
lr=0.0003
weight_decay=0.01
tmax=20
drop_prob=0.2
batch_size=256
shuffle_buffer_length=10000
checkpoints="/home/fenglei/TS-RAG-main/TS-RAG/checkpoints"
pretrained_model_path="/home/fenglei/TS-RAG-main/TS-RAG/checkpoints/base/"
data_path="/home/fenglei/TS-RAG-main/datasets/pretrain/pretrain_pairs_ctx${retrieve_lookback_length}"

model_id="data50m_${augment_mode}_${context_length}_pred${prediction_length}_lookback${retrieve_lookback_length}_top${top_k}_lr${lr}_drop${drop_prob}_${optimizer}_cosanneal_step${train_steps}_bs${batch_size}_no_embeddingtuning"

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
    --shuffle_buffer_length $shuffle_buffer_length \
    --freeze_chronos_bolt \
    --checkpoints $checkpoints
