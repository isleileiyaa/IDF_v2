cd /home/fenglei/TS-RAG-main/TS-RAG
run_file=pretrain.py
top_k=10
retrieve_lookback_length=512
retrieval_database_path="${PRETRAIN_RETRIEVAL_DATABASE_PATH:-/home/fenglei/TS-RAG-main/retrieval_database/pretrain/retrieval_database_${retrieve_lookback_length}.parquet}"
augment_mode=idf_trr_dualpath
context_length=512
prediction_length=64
checkpoints="${CHECKPOINTS_DIR:-/home/fenglei/TS-RAG-main/TS-RAG/checkpoints}"
data_path="${PRETRAIN_DATA_PATH:-/home/fenglei/TS-RAG-main/datasets/pretrain/pretrain_pairs_ctx${retrieve_lookback_length}}"
train_steps=${TRAIN_STEPS:-10000}
evaluation_steps=${EVALUATION_STEPS:-10000}
gpu_loc=${GPU_LOC:-0}
optimizer=adamw
lr=0.0003
weight_decay=0.01
tmax=20
drop_prob=0.2
# 注意：Stage 4 下这个 batch_size 只影响 model_id 的命名，实际训练 batch size
# 由 env_group_size * env_batch_size 决定(见下面)，pretrain.py 里会打印提示。
batch_size=256
shuffle_buffer_length=${SHUFFLE_BUFFER_LENGTH:-10000}
pretrained_model_path="${PRETRAINED_MODEL_PATH:-/home/fenglei/TS-RAG-main/TS-RAG/checkpoints/base/}"

# Stage 3 的 L_sep 部分，跟之前一致，不变。
lambda_sep=${LAMBDA_SEP:-0.01}
beta_var=${BETA_VAR:-1.0}
gamma0_var=${GAMMA0_VAR:-0.1}

# RIDDE_新版目标函数与最终实验方案 Stage 4：L_TRR/CVaR 新增参数。
# !!!环境定义说明，务必知晓（2026-09-09 更新：重叠窗口泄露诊断结果）!!!：现有
# pretrain_pairs_ctx512/ 下的30个parquet分片没有真实时间戳/序列身份，无法构造方案
# 5.2节字面要求的"连续时间块"。但抽样诊断(chunk内部滑窗重合比例99.3%~100%、6个chunk
# 间样本级重叠数=0)证实30个chunk之间是按原始序列/来源对象切分的，不存在重叠窗口跨环境
# 泄露——"环境"是干净的序列/来源对象级固定分组，不是随机打散的负对照。这次结果可以作为
# "环境定义为序列/来源对象级(非时间连续)"的正式CVaR结果去汇报，但不能称为方案里"连续
# 时间块"意义上的时间稳健性结果。
lambda_trr=${LAMBDA_TRR:-0.1}
cvar_alpha=${CVAR_ALPHA:-0.6}
env_group_size=${ENV_GROUP_SIZE:-8}
env_batch_size=${ENV_BATCH_SIZE:-32}
env_shuffle_buffer_length=${ENV_SHUFFLE_BUFFER_LENGTH:-2000}
nu_init=${NU_INIT:-0.0}
# 必须指向已经训练好、配方对齐(同backbone/数据/步数/优化器)的 augment_mode=baseline
# checkpoint(方案文档4.2节的模型B)。默认指向确认过的 seed2021、10000步 baseline。
trr_reference_model_path="${TRR_REFERENCE_MODEL_PATH:-/home/fenglei/TS-RAG-main/TS-RAG/checkpoints/data50m_baseline_512_pred64_lookback512_top10_lr0.0003_drop0.2_adamw_cosanneal_step10000_bs256_no_embeddingtuning_seed2021_final.pth}"

# 种子：baseline checkpoint 明确是 seed2021 训出来的。之前 dualpath 的正式 checkpoint
# 训练时到底传没传 PRETRAIN_SEED 环境变量、传的是不是2021，日志里查不到实锤证据。
# 为了让这次 Stage 4 的种子跟 baseline 严格对齐、不再有任何不确定性，这里显式写死。
export PRETRAIN_SEED=2021

model_id="data50m_${augment_mode}_${context_length}_pred${prediction_length}_lookback${retrieve_lookback_length}_top${top_k}_lr${lr}_drop${drop_prob}_${optimizer}_cosanneal_step${train_steps}_bs${batch_size}_no_embeddingtuning_lambdasep${lambda_sep}_lambdatrr${lambda_trr}_alpha${cvar_alpha}_seed${PRETRAIN_SEED}"
echo "Launching pretrain (idf_trr_dualpath, Stage 4 L_TRR/CVaR), lambda_sep=$lambda_sep beta_var=$beta_var lambda_trr=$lambda_trr cvar_alpha=$cvar_alpha env_group_size=$env_group_size env_batch_size=$env_batch_size PRETRAIN_SEED=$PRETRAIN_SEED"
echo "model_id=$model_id"
echo "[提醒] 这次的\"环境\"已确认是序列/来源对象级的固定分组(抽样未发现跨chunk重叠泄露)，不是随机负对照，但也不是按时间连续切分——汇报时按此如实说明。"
python $run_file \
    --model_id $model_id \
    --model ChronosBoltRetrieve \
    --top_k $top_k \
    --retrieve_lookback_length $retrieve_lookback_length \
    --retrieval_database_path $retrieval_database_path \
    --augment_mode $augment_mode \
    --pretrained_model_path $pretrained_model_path \
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
    --lambda_sep $lambda_sep \
    --beta_var $beta_var \
    --gamma0_var $gamma0_var \
    --lambda_trr $lambda_trr \
    --cvar_alpha $cvar_alpha \
    --env_group_size $env_group_size \
    --env_batch_size $env_batch_size \
    --env_shuffle_buffer_length $env_shuffle_buffer_length \
    --nu_init $nu_init \
    --trr_reference_model_path $trr_reference_model_path \
    --gpu_loc $gpu_loc \
    --freeze_chronos_bolt \
    --checkpoints $checkpoints
