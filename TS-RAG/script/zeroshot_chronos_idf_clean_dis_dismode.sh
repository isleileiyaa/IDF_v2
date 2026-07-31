export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

save_suffix="${SAVE_SUFFIX:-}"
filename="${SAVE_FILE_NAME:-zeroshot_chronos_idf_clean_dis_dismode${save_suffix:+_${save_suffix}}.txt}"
model=ChronosBoltRetrieve
gpu_loc=0
run_file="/home/fenglei/TS-RAG-main/TS-RAG/zeroshot.py"
seq_len=512
pred_len=64
datasets="${DATASETS:-ETTh1 ETTh2 ETTm1 ETTm2 weather exchange_rate electricity traffic solar PEMS08 AQWan Wind ILI ZafNoo CzeLan}"
lookback_length=512
augment_mode=idf_clean_dis
top_k=10

dis_mode=${1:-${DIS_MODE:-output}}
rho2=${2:-${RHO2:-0.01}}
rho_dis_output=${3:-${RHO_DIS_OUTPUT:-0.01}}

batch_size=256
retrieval_database_dir="/home/fenglei/TS-RAG-main/retrieval_database/"
ett_root_path="${ETT_ROOT_PATH:-/home/fenglei/TS-RAG-main/datasets/ETT-small/}"
custom_datasets_root="${CUSTOM_DATASETS_ROOT:-/home/fenglei/TS-RAG-main/datasets/}"
pretrained_model_path="/home/fenglei/TS-RAG-main/TS-RAG/checkpoints/base/"
default_checkpoint_model_path="/home/fenglei/TS-RAG-main/TS-RAG/checkpoints/data50m_idf_clean_dis_512_pred64_lookback512_top10_lr0.0003_drop0.2_adamw_cosanneal_step10000_bs256_no_embeddingtuning_dismode${dis_mode}_rho2_${rho2}_rhoout_${rho_dis_output}_final.pth"
checkpoint_model_path="${CHECKPOINT_MODEL_PATH:-$default_checkpoint_model_path}"
echo "Using zeroshot dis_mode=$dis_mode rho2=$rho2 rho_dis_output=$rho_dis_output"
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
    --model_id "${dataset}_zeroshot_${seq_len}_pred_${pred_len}_${lookback_length}_retrieve_${pred_len}_idf_clean_dis_dismode${dis_mode}" \
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
    --augment_mode $augment_mode

done
