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
# 先用小的 TRAIN_STEPS(比如 200~500)跑一次冒烟测试，确认新架构训得动、
# loss_forecast 在下降、loss_xcov/loss_var 不是 NaN/爆炸，再跑正式的完整
# 预算(不设 TRAIN_STEPS 时默认 10000，跟 idf_clean_dis 当年用的一样，方便
# 后面公平对比)。
train_steps=${TRAIN_STEPS:-10000}
evaluation_steps=${EVALUATION_STEPS:-10000}
optimizer=adamw
lr=0.0003
weight_decay=0.01
tmax=20
drop_prob=0.2
batch_size=256
shuffle_buffer_length=${SHUFFLE_BUFFER_LENGTH:-10000}
pretrained_model_path="${PRETRAINED_MODEL_PATH:-/home/fenglei/TS-RAG-main/TS-RAG/checkpoints/base/}"
# RIDDE_新版目标函数与最终实验方案 Stage 3 (Dual+ERM)：只接 L_sep，不接
# L_TRR/CVaR(那是 Stage 4)。lambda_sep 先给个不大的默认值(0.01)，避免一
# 上来就让 L_sep 主导训练、盖过预测损失本身 —— 具体量级还需要看冒烟测试
# 里 loss_forecast 和 lambda_sep*loss_sep 两者的相对大小再调。
lambda_sep=${LAMBDA_SEP:-0.01}
beta_var=${BETA_VAR:-1.0}
gamma0_var=${GAMMA0_VAR:-0.1}

model_id="data50m_${augment_mode}_${context_length}_pred${prediction_length}_lookback${retrieve_lookback_length}_top${top_k}_lr${lr}_drop${drop_prob}_${optimizer}_cosanneal_step${train_steps}_bs${batch_size}_no_embeddingtuning_lambdasep${lambda_sep}"
echo "Launching pretrain (idf_trr_dualpath), lambda_sep=$lambda_sep beta_var=$beta_var gamma0_var=$gamma0_var"
echo "model_id=$model_id"
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
    --freeze_chronos_bolt \
    --checkpoints $checkpoints
