export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

# Same 4x4 KB-matrix setup as zeroshot_chronos_idf_clean_dis_kb_matrix.sh, but
# for augment_mode=moe -- the actual TS-RAG paper method (ARM module), as
# opposed to idf_clean_dis which is a locally-authored variant. Reuses the
# already-cached *_retrieve_*.csv files (retrieval is augment-mode-agnostic),
# so this should only need to run the (fast) model forward pass per cell.

save_suffix="${SAVE_SUFFIX:-}"
filename="${SAVE_FILE_NAME:-zeroshot_chronos_moe_kb_matrix${save_suffix:+_${save_suffix}}.txt}"
model=ChronosBoltRetrieve
gpu_loc=0
run_file="/home/fenglei/TS-RAG-main/TS-RAG/zeroshot.py"
seq_len=512
pred_len=64
lookback_length=512
augment_mode=moe
top_k=10

batch_size=256
retrieval_database_dir="/home/fenglei/TS-RAG-main/retrieval_database/"
ett_root_path="${ETT_ROOT_PATH:-/home/fenglei/TS-RAG-main/datasets/ETT-small/}"
pretrained_model_path="/home/fenglei/TS-RAG-main/TS-RAG/checkpoints/base/"
default_checkpoint_model_path="/home/fenglei/TS-RAG-main/TS-RAG/checkpoints/data50m_moe_512_pred64_lookback512_top10_lr0.0003_drop0.2_adamw_cosanneal_step10000_bs256_no_embeddingtuning_final.pth"
checkpoint_model_path="${CHECKPOINT_MODEL_PATH:-$default_checkpoint_model_path}"
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
    --model_id "${dataset}_zeroshot_${seq_len}_pred_${pred_len}_${lookback_length}_retrieve_${pred_len}_moe_kb_${kb}" \
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
