export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

augment_mode=idf_y_linear_head
model=ChronosBoltRetrieve
run_file="/home/fenglei/TS-RAG-main/TS-RAG/pretrain.py"

context_length=512
prediction_length=64
retrieve_lookback_length=512
top_k=10

lr=0.0003
drop_prob=0.2
optimizer=adamw
train_steps=10000
evaluation_steps=1000
batch_size=256

checkpoints="/home/fenglei/TS-RAG-main/TS-RAG/checkpoints"
pretrained_model_path="/home/fenglei/TS-RAG-main/TS-RAG/checkpoints/base/"
retrieval_database_path="/home/fenglei/TS-RAG-main/retrieval_database/pretrain/retrieval_database_512.parquet"
data_path="/home/fenglei/TS-RAG-main/datasets/pretrain/pretrain_pairs_ctx512"

model_id="data50m_${augment_mode}_${context_length}_pred${prediction_length}_lookback${retrieve_lookback_length}_top${top_k}_lr${lr}_drop${drop_prob}_${optimizer}_cosanneal_step${train_steps}_bs${batch_size}_no_embeddingtuning"

python $run_file \
    --model_id $model_id \
    --model $model \
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
    --weight_decay 0.01 \
    --tmax 20 \
    --drop_prob $drop_prob \
    --batch_size $batch_size \
    --shuffle_buffer_length 10000 \
    --freeze_chronos_bolt \
    --checkpoints $checkpoints
