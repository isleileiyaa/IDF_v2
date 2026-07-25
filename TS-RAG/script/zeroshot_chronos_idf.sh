export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

save_suffix="${SAVE_SUFFIX:-}"
filename="${SAVE_FILE_NAME:-zeroshot_chronos_idf${save_suffix:+_${save_suffix}}.txt}"
model=ChronosBoltRetrieve
gpu_loc=0
run_file="/home/fenglei/TS-RAG-main/TS-RAG/zeroshot.py"
seq_len=512
pred_len=64
datasets="ETTh1"
lookback_length=512
augment_mode=idf
top_k=10
retrieval_mode="${RETRIEVAL_MODE:-embedding}"

batch_size=256
retrieval_database_dir="/home/fenglei/TS-RAG-main/retrieval_database/"
ett_root_path="${ETT_ROOT_PATH:-/home/fenglei/TS-RAG-main/TS-RAG/datasets/ETT-small/}"
custom_datasets_root="${CUSTOM_DATASETS_ROOT:-/home/fenglei/TS-RAG-main/datasets}"
pretrained_model_path="/home/fenglei/TS-RAG-main/TS-RAG/checkpoints/base/"
checkpoint_model_path="${CHECKPOINT_MODEL_PATH:-/home/fenglei/TS-RAG-main/TS-RAG/checkpoints/idf_final.pth}"

for dataset in $datasets;
do
retrieve_database_name=$dataset

if [ "$dataset" = 'ETTm1' ] || [ "$dataset" = 'ETTm2' ]; then
    data='ett_m_retrieve'
    metadata_frequency='minute'
    root_path="$ett_root_path"
elif [ "$dataset" = 'ETTh1' ] || [ "$dataset" = 'ETTh2' ]; then
    data='ett_h_retrieve'
    metadata_frequency='hour'
    root_path="$ett_root_path"
elif [ "$dataset" = 'electricity' ] || [ "$dataset" = 'exchange_rate' ]; then
    data='custom_retrieve'
    metadata_frequency='hour'
    root_path="${custom_datasets_root}/${dataset}/"
elif [ "$dataset" = 'weather' ]; then
    data='custom_retrieve'
    metadata_frequency='10minutes'
    root_path="${custom_datasets_root}/${dataset}/"
fi

python $run_file \
    --root_path "$root_path" \
    --data_path "${dataset}.csv" \
    --model_id "${dataset}_zeroshot_${seq_len}_pred_${pred_len}_${lookback_length}_retrieve_${pred_len}_idf" \
    --data $data \
    --top_k $top_k \
    --checkpoint_model_path $checkpoint_model_path \
    --pretrained_model_path $pretrained_model_path \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --lookback_length $lookback_length \
    --batch_size $batch_size \
    --num_workers 0 \
    --decay_fac 0.5 \
    --freq 0 \
    --percent 100 \
    --model $model \
    --gpu_loc $gpu_loc \
    --tmax 20 \
    --cos 1 \
    --save_file_name $filename \
    --retrieval_database_dir "$retrieval_database_dir" \
    --dimension 768 \
    --embedding_model_type chronos \
    --metadata_frequency $metadata_frequency \
    --metadata_database_name $retrieve_database_name \
    --augment_mode $augment_mode \
    --retrieval_mode $retrieval_mode

done
