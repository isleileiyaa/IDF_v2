export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

save_suffix="${SAVE_SUFFIX:-}"
filename="${SAVE_FILE_NAME:-zeroshot_chronos_bolt_official${save_suffix:+_${save_suffix}}.txt}"
model=ChronosBolt
gpu_loc=0
run_file="/home/fenglei/TS-RAG-main/TS-RAG/zeroshot.py"
seq_len=512
pred_len=64
datasets="ETTh1"

batch_size=256
ett_root_path="${ETT_ROOT_PATH:-/home/fenglei/TS-RAG-main/datasets/ETT-small/}"
custom_datasets_root="${CUSTOM_DATASETS_ROOT:-/home/fenglei/TS-RAG-main/datasets}"
pretrained_model_path="/home/fenglei/TS-RAG-main/TS-RAG/checkpoints/base/"
checkpoint_model_path="None"
augment_mode=baseline

for dataset in $datasets;
do
if [ "$dataset" = 'ETTm1' ] || [ "$dataset" = 'ETTm2' ]; then
    data='ett_m'
    root_path="$ett_root_path"
elif [ "$dataset" = 'ETTh1' ] || [ "$dataset" = 'ETTh2' ]; then
    data='ett_h'
    root_path="$ett_root_path"
elif [ "$dataset" = 'electricity' ] || [ "$dataset" = 'exchange_rate' ] || [ "$dataset" = 'weather' ]; then
    data='custom'
    root_path="${custom_datasets_root}/${dataset}/"
fi

python $run_file \
    --root_path "$root_path" \
    --data_path "${dataset}.csv" \
    --model_id "${dataset}_zeroshot_${seq_len}_pred_${pred_len}_chronosbolt_baseline" \
    --data $data \
    --checkpoint_model_path $checkpoint_model_path \
    --pretrained_model_path $pretrained_model_path \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size $batch_size \
    --num_workers 0 \
    --freq 0 \
    --percent 100 \
    --model $model \
    --gpu_loc $gpu_loc \
    --save_file_name $filename \
    --augment_mode $augment_mode 
    
done
