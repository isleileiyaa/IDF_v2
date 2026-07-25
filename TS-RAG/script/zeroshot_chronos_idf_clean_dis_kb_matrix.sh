export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

# Replicates Cross-RAG's Table 7 / TS-RAG's Table 7 "different knowledge base"
# setup: for each of the four ETT datasets as TARGET, evaluate with the
# retrieval knowledge base built from each of the four ETT datasets in turn
# (including the target itself, i.e. the in-domain diagonal). Each cell uses
# exactly ONE source database, so this reduces to the same single-source
# retrieval code path as ordinary in-domain retrieval -- no multi-database
# boundary_idx bookkeeping is involved.

save_suffix="${SAVE_SUFFIX:-}"
filename="${SAVE_FILE_NAME:-zeroshot_chronos_idf_clean_dis_kb_matrix${save_suffix:+_${save_suffix}}.txt}"
model=ChronosBoltRetrieve
gpu_loc=0
run_file="/home/fenglei/TS-RAG-main/TS-RAG/zeroshot.py"
seq_len=512
pred_len=64
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
targets="${TARGETS:-$all_ett}"
kbs="${KBS:-$all_ett}"

freq_of() {
    case "$1" in
        ETTh1|ETTh2) echo hour ;;
        ETTm1|ETTm2) echo minute ;;
    esac
}

data_of() {
    case "$1" in
        ETTh1|ETTh2) echo ett_h_retrieve ;;
        ETTm1|ETTm2) echo ett_m_retrieve ;;
    esac
}

for dataset in $targets;
do
data="$(data_of $dataset)"
metadata_frequency="$(freq_of $dataset)"
root_path="$ett_root_path"

for kb in $kbs;
do
retrieve_database_name="$kb"

echo "target=$dataset  KB=$kb"

python $run_file \
    --root_path "$root_path" \
    --data_path "${dataset}.csv" \
    --model_id "${dataset}_zeroshot_${seq_len}_pred_${pred_len}_${lookback_length}_retrieve_${pred_len}_idf_clean_dis_kb_${kb}" \
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
done
