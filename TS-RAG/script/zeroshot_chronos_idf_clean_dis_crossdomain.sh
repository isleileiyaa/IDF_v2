export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Replicates the "Cross-Domain" retrieval-knowledge-base setting from the
# TS-RAG (Table 7) / Cross-RAG (R4) papers: for each of the four ETT
# datasets, the retrieval knowledge base is built from the OTHER three ETT
# datasets' training splits instead of the target dataset's own training
# split (in-domain). Only the ETT family is used here, matching what both
# papers actually validated (they never test retrieval across structurally
# different domains, e.g. ETT <-> electricity/weather).

save_suffix="${SAVE_SUFFIX:-}"
filename="${SAVE_FILE_NAME:-zeroshot_chronos_idf_clean_dis_crossdomain${save_suffix:+_${save_suffix}}.txt}"
model=ChronosBoltRetrieve
gpu_loc=0
run_file="/home/fenglei/TS-RAG-main/TS-RAG/zeroshot.py"
seq_len=512
pred_len=64
datasets="${DATASETS:-ETTh1 ETTh2 ETTm1 ETTm2}"
lookback_length=512
augment_mode=idf_clean_dis
top_k=10
rho1=${1:-${RHO1:-0}}
rho2=${2:-${RHO2:-0}}
rho3=${3:-${RHO3:-0}}
rho4=${4:-${RHO4:-0}}

batch_size=256
retrieval_database_dir="/home/fenglei/TS-RAG-main/retrieval_database/"
ett_root_path="${ETT_ROOT_PATH:-/home/fenglei/TS-RAG-main/datasets/ETT-small/}"
pretrained_model_path="/home/fenglei/TS-RAG-main/TS-RAG/checkpoints/base/"
default_checkpoint_model_path="/home/fenglei/TS-RAG-main/TS-RAG/checkpoints/data50m_idf_clean_dis_512_pred64_lookback512_top10_lr0.0003_drop0.2_adamw_cosanneal_step10000_bs256_no_embeddingtuning_rho${rho1}_${rho2}_${rho3}_${rho4}_final.pth"
checkpoint_model_path="${CHECKPOINT_MODEL_PATH:-$default_checkpoint_model_path}"
echo "Using zeroshot rho=($rho1, $rho2, $rho3, $rho4)"
echo "checkpoint_model_path=$checkpoint_model_path"

all_ett="ETTh1 ETTh2 ETTm1 ETTm2"

for dataset in $datasets;
do
retrieve_database_name=""
for other in $all_ett; do
    if [ "$other" != "$dataset" ]; then
        retrieve_database_name="$retrieve_database_name $other"
    fi
done
retrieve_database_name="$(echo $retrieve_database_name | xargs)"

if [ "$dataset" = 'ETTm1' ] || [ "$dataset" = 'ETTm2' ]; then
    data='ett_m_retrieve'
    metadata_frequency='minute'
    root_path="$ett_root_path"
elif [ "$dataset" = 'ETTh1' ] || [ "$dataset" = 'ETTh2' ]; then
    data='ett_h_retrieve'
    metadata_frequency='hour'
    root_path="$ett_root_path"
else
    echo "Unknown dataset: $dataset (this script only supports the ETT family)"
    exit 1
fi

echo "dataset=$dataset  KB(cross-domain)=$retrieve_database_name"

python $run_file \
    --root_path "$root_path" \
    --data_path "${dataset}.csv" \
    --model_id "${dataset}_zeroshot_${seq_len}_pred_${pred_len}_${lookback_length}_retrieve_${pred_len}_idf_clean_dis_crossdomain" \
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
    --metadata_database_name "$retrieve_database_name" \
    --augment_mode $augment_mode

done
