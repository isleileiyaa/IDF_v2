export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# RIDDE_新版目标函数与最终实验方案 Stage 3 (Dual+ERM) 的 zero-shot 评估。
# 建议先只在 ETTh1 上跑通(默认 DATASETS 就只给了 ETTh1)，确认跑得通、缓存
# tag 没有把 idf_clean_dis 的 rag 缓存覆盖掉之后，再把 DATASETS 换成全部
# 7 个数据集去跟 ridde_trr/eval_temporal_robustness.py 里已经跑出来的基线
# 表(idf_clean_dis 的 rag vs truebase)对比。
save_suffix="${SAVE_SUFFIX:-}"
filename="${SAVE_FILE_NAME:-zeroshot_chronos_idf_trr_dualpath${save_suffix:+_${save_suffix}}.txt}"
model=ChronosBoltRetrieve
gpu_loc=0
run_file="/home/fenglei/TS-RAG-main/TS-RAG/zeroshot.py"
seq_len=512
pred_len=64
datasets="${DATASETS:-ETTh1}"
lookback_length=512
augment_mode=${AUGMENT_MODE:-idf_trr_dualpath}
top_k=10
lambda_sep=${LAMBDA_SEP:-0.01}

batch_size=256
retrieval_database_dir="/home/fenglei/TS-RAG-main/retrieval_database/"
ett_root_path="${ETT_ROOT_PATH:-/home/fenglei/TS-RAG-main/datasets/ETT-small/}"
custom_datasets_root="${CUSTOM_DATASETS_ROOT:-/home/fenglei/TS-RAG-main/datasets/}"
pretrained_model_path="/home/fenglei/TS-RAG-main/TS-RAG/checkpoints/base/"
# 跟 pretrain_idf_trr_dualpath.sh 里的 model_id 拼接规则保持一致；如果那边
# 实际跑的时候用了非默认的 TRAIN_STEPS/lambda_sep，这里要么同步改
# train_steps_in_ckpt/lambda_sep，要么直接用 CHECKPOINT_MODEL_PATH 手动指定
# 实际的 checkpoint 路径(更保险，推荐这么做)。
train_steps_in_ckpt=${TRAIN_STEPS_IN_CKPT:-10000}
default_checkpoint_model_path="/home/fenglei/TS-RAG-main/TS-RAG/checkpoints/data50m_${augment_mode}_512_pred64_lookback512_top10_lr0.0003_drop0.2_adamw_cosanneal_step${train_steps_in_ckpt}_bs256_no_embeddingtuning_lambdasep${lambda_sep}_final.pth"
checkpoint_model_path="${CHECKPOINT_MODEL_PATH:-$default_checkpoint_model_path}"
echo "checkpoint_model_path=$checkpoint_model_path"
eval_split="${EVAL_SPLIT:-test}"
echo "eval_split=$eval_split"

for dataset in $datasets;
do
retrieve_database_name=$dataset

if [ "$dataset" = 'ETTm1' ] || [ "$dataset" = 'ETTm2' ]; then
    data='ett_m_retrieve'; metadata_frequency='minute'; root_path="$ett_root_path"
elif [ "$dataset" = 'ETTh1' ] || [ "$dataset" = 'ETTh2' ]; then
    data='ett_h_retrieve'; metadata_frequency='hour'; root_path="$ett_root_path"
elif [ "$dataset" = 'exchange_rate' ] || [ "$dataset" = 'electricity' ]; then
    data='custom_retrieve'; metadata_frequency='hour'; root_path="${custom_datasets_root}/${dataset}/"
elif [ "$dataset" = 'weather' ]; then
    data='custom_retrieve'; metadata_frequency='10minutes'; root_path="${custom_datasets_root}/${dataset}/"
else
    echo "Unknown dataset: $dataset"; exit 1
fi

python $run_file \
    --root_path "$root_path" \
    --data_path "${dataset}.csv" \
    --model_id "${dataset}_zeroshot_${seq_len}_pred_${pred_len}_${lookback_length}_retrieve_${pred_len}_${augment_mode}" \
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
    --eval_split "$eval_split"

done
