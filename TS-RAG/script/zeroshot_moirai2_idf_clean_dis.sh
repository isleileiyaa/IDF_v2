export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

save_suffix="${SAVE_SUFFIX:-}"
filename="${SAVE_FILE_NAME:-zeroshot_moirai2_idf_clean_dis${save_suffix:+_${save_suffix}}.txt}"
model=Moirai2Retrieve
gpu_loc=0
run_file="/home/fenglei/TS-RAG-main/TS-RAG/zeroshot.py"
seq_len=512
pred_len=64
datasets="${DATASETS:-ETTh1 ETTh2 ETTm1 ETTm2 weather exchange_rate electricity}"
lookback_length=512
augment_mode=idf_clean_dis
top_k=10

batch_size=256
retrieval_database_dir="/home/fenglei/TS-RAG-main/retrieval_database/"
ett_root_path="${ETT_ROOT_PATH:-/home/fenglei/TS-RAG-main/datasets/ETT-small/}"
custom_datasets_root="${CUSTOM_DATASETS_ROOT:-/home/fenglei/TS-RAG-main/datasets/}"
# NOTE: Moirai2ModelForForecastingWithRetrieval does not read
# --pretrained_model_path (its backbone is always
# Salesforce/moirai-2.0-R-small, loaded from HuggingFace Hub inside the model
# class itself) -- unlike the Chronos-Bolt zeroshot script, no
# pretrained_model_path arg is passed below.
default_checkpoint_model_path="/home/fenglei/TS-RAG-main/TS-RAG/checkpoints/moirai2_idf_clean_dis_512_pred64_lookback512_top10_lr0.000015_drop0.2_adamw_cosanneal_step10000_bs256_final.pth"
checkpoint_model_path="${CHECKPOINT_MODEL_PATH:-$default_checkpoint_model_path}"
echo "checkpoint_model_path=$checkpoint_model_path"

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
elif [ "$dataset" = 'traffic' ]; then
    data='custom_retrieve'
    metadata_frequency='hour'
    root_path="${custom_datasets_root}/${dataset}/"
elif [ "$dataset" = 'solar' ]; then
    data='custom_retrieve'
    metadata_frequency='10minutes'
    root_path="${custom_datasets_root}/${dataset}/"
elif [ "$dataset" = 'PEMS08' ]; then
    data='custom_retrieve'
    metadata_frequency='5minutes'
    root_path="${custom_datasets_root}/${dataset}/"
elif [ "$dataset" = 'AQWan' ]; then
    data='custom_retrieve'
    metadata_frequency='hour'
    root_path="${custom_datasets_root}/${dataset}/"
elif [ "$dataset" = 'Wind' ]; then
    data='custom_retrieve'
    metadata_frequency='15minutes'
    root_path="${custom_datasets_root}/${dataset}/"
elif [ "$dataset" = 'ILI' ]; then
    data='custom_retrieve'
    metadata_frequency='week'
    root_path="${custom_datasets_root}/${dataset}/"
elif [ "$dataset" = 'ZafNoo' ]; then
    data='custom_retrieve'
    metadata_frequency='30minutes'
    root_path="${custom_datasets_root}/${dataset}/"
elif [ "$dataset" = 'CzeLan' ]; then
    data='custom_retrieve'
    metadata_frequency='30minutes'
    root_path="${custom_datasets_root}/${dataset}/"
else
    echo "Unknown dataset: $dataset"
    echo "Supported datasets: ETTh1 ETTh2 ETTm1 ETTm2 weather electricity exchange_rate traffic solar PEMS08 AQWan Wind ILI ZafNoo CzeLan"
    exit 1
fi

python $run_file \
    --root_path "$root_path" \
    --data_path "${dataset}.csv" \
    --model_id "${dataset}_zeroshot_${seq_len}_pred_${pred_len}_${lookback_length}_retrieve_${pred_len}_moirai2_idf_clean_dis" \
    --data $data \
    --top_k $top_k \
    --checkpoint_model_path $checkpoint_model_path \
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
    --augment_mode $augment_mode

done
