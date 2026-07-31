cd /home/fenglei/TS-RAG-main/TS-RAG
run_file=pretrain.py
top_k=10
retrieve_lookback_length=512
retrieval_database_path="${PRETRAIN_RETRIEVAL_DATABASE_PATH:-/home/fenglei/TS-RAG-main/retrieval_database/pretrain/retrieval_database_${retrieve_lookback_length}.parquet}"
# augment_mode selects which fusion head pretrain.py attaches to the frozen
# TimesFM 2.5 backbone (see models/TimesFM25.py): only 'idf_clean_dis' (RIDDE)
# exists for this backbone so far, no moe port.
augment_mode=idf_clean_dis
context_length=512
prediction_length=64
checkpoints="${CHECKPOINTS_DIR:-/home/fenglei/TS-RAG-main/TS-RAG/checkpoints}"
data_path="${PRETRAIN_DATA_PATH:-/home/fenglei/TS-RAG-main/datasets/pretrain/pretrain_pairs_ctx${retrieve_lookback_length}}"
train_steps=${TRAIN_STEPS:-10000}
# evaluation_steps < train_steps so the CosineAnnealingLR scheduler actually
# decays over the run instead of firing once at the very end (see the note
# in pretrain_moirai2_idf_clean_dis.sh for why this matters).
evaluation_steps=${EVALUATION_STEPS:-1000}
optimizer=adamw
# lr tuned via HPO sweep over {3e-5, 1e-4, 3e-4, 6e-4, 1e-3, 2e-3, 1.5e-5,
# 5e-6, 2e-6} on this backbone: training loss and 6-dataset zeroshot MSE/MAE
# diverge (lowest training loss was 1e-3, but its zeroshot MSE/MAE was among
# the worst -- overfitting to the pretrain retrieval distribution), so the
# choice below is picked by zeroshot MSE/MAE, not training loss. 5e-6 beat
# both its neighbors (1.5e-5 and 2e-6), confirming it's the actual optimum,
# not just monotonic improvement that ran out of search budget. Only RIDDE
# (idf_clean_dis) was tuned for this backbone -- moe/CrossRAG are left at
# their defaults, matching the same policy used for the Moirai2 backbone.
lr=${LEARNING_RATE:-0.000005}
weight_decay=0.01
tmax=20
drop_prob=0.2
batch_size=256
shuffle_buffer_length=${SHUFFLE_BUFFER_LENGTH:-10000}

model_id="timesfm25_${augment_mode}_${context_length}_pred${prediction_length}_lookback${retrieve_lookback_length}_top${top_k}_lr${lr}_drop${drop_prob}_${optimizer}_cosanneal_step${train_steps}_bs${batch_size}"
echo "Launching TimesFM 2.5 RIDDE pretrain"
echo "model_id=$model_id"
python $run_file \
    --model_id $model_id \
    --model TimesFM25Retrieve \
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
