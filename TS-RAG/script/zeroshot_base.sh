export CUDA_VISIBLE_DEVICES="0"

filename=zeroshot_chronos_baseline.txt
model=ChronosBoltRetrieve
gpu_loc=0
run_file=/home/fenglei/TS-RAG-main/TS-RAG/zeroshot.py

seq_len=512
pred_len=64
datasets="ETTh1"
lookback_length=512
augment_mode=baseline
top_k=10

batch_size=256

pretrained_model_path="./checkpoints/base/"
checkpoint_model_path=None

for dataset in $datasets;
do

if [ "$dataset" = 'ETTh1' ] || [ "$dataset" = 'ETTh2' ]; then
    data='ett_h'
    root_path='../datasets/ETT-small/'
fi

python $run_file \
    --root_path $root_path \
    --data_path ${dataset}.csv \
    --model_id "${dataset}_baseline" \
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
    --model $model \
    --gpu_loc $gpu_loc \
    --save_file_name $filename \
    --augment_mode $augment_mode

done