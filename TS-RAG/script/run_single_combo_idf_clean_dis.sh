#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
ROOT_DIR="${ROOT_DIR:-$(cd "${PROJECT_DIR}/.." && pwd)}"

PRETRAIN_PY="${PROJECT_DIR}/pretrain.py"
ZEROSHOT_PY="${PROJECT_DIR}/zeroshot.py"
PYTHON_BIN="${PYTHON_BIN:-python}"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${PROJECT_DIR}/checkpoints}"
PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-${PROJECT_DIR}/checkpoints/base/}"
PRETRAIN_DATA_PATH="${PRETRAIN_DATA_PATH:-${ROOT_DIR}/datasets/pretrain/pretrain_pairs_ctx512}"
PRETRAIN_RETRIEVAL_DATABASE_PATH="${PRETRAIN_RETRIEVAL_DATABASE_PATH:-${ROOT_DIR}/retrieval_database/pretrain/retrieval_database_512.parquet}"
RETRIEVAL_DATABASE_DIR="${RETRIEVAL_DATABASE_DIR:-${ROOT_DIR}/retrieval_database/}"
ETT_ROOT_PATH="${ETT_ROOT_PATH:-${ROOT_DIR}/datasets/ETT-small/}"
CUSTOM_DATASETS_ROOT="${CUSTOM_DATASETS_ROOT:-${ROOT_DIR}/datasets}"

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/sweep_outputs/idf_clean_dis}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
RESULT_CSV="${RESULT_CSV:-${OUTPUT_DIR}/results_long.csv}"
RESULT_TXT="${RESULT_TXT:-${OUTPUT_DIR}/results_readable.txt}"
STATUS_DIR="${STATUS_DIR:-${OUTPUT_DIR}/status}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${STATUS_DIR}"

cd "${PROJECT_DIR}"

top_k="$1"
rho2="$2"

augment_mode="${AUGMENT_MODE:-idf_clean_dis}"
context_length="${CONTEXT_LENGTH:-512}"
prediction_length="${PREDICTION_LENGTH:-64}"
retrieve_lookback_length="${RETRIEVE_LOOKBACK_LENGTH:-512}"

train_steps="${TRAIN_STEPS:-10000}"
evaluation_steps="${EVALUATION_STEPS:-10000}"
optimizer="${OPTIMIZER:-adamw}"
lr="${LR:-0.00003}"
weight_decay="${WEIGHT_DECAY:-0.01}"
tmax="${TMAX:-20}"
drop_prob="${DROP_PROB:-0.0}"
batch_size="${BATCH_SIZE:-256}"
shuffle_buffer_length="${SHUFFLE_BUFFER_LENGTH:-10000}"
gpu_loc="${GPU_LOC:-0}"

rho1=0
rho3=0
rho4=0

datasets=(ETTh1 ETTh2 ETTm1 ETTm2 weather exchange_rate electricity)

model_id="data50m_${augment_mode}_${context_length}_pred${prediction_length}_lookback${retrieve_lookback_length}_top${top_k}_lr${lr}_drop${drop_prob}_${optimizer}_cosanneal_step${train_steps}_bs${batch_size}_no_embeddingtuning_rho${rho1}_${rho2}_${rho3}_${rho4}"
train_log="${LOG_DIR}/${model_id}_train.log"
done_flag="${STATUS_DIR}/${model_id}.done"

dataset_args() {
    local dataset="$1"
    case "${dataset}" in
        ETTm1|ETTm2)
            echo "ett_m_retrieve|minute|${ETT_ROOT_PATH}|${dataset}.csv"
            ;;
        ETTh1|ETTh2)
            echo "ett_h_retrieve|hour|${ETT_ROOT_PATH}|${dataset}.csv"
            ;;
        electricity|exchange_rate)
            echo "custom_retrieve|hour|${CUSTOM_DATASETS_ROOT}/${dataset}/|${dataset}.csv"
            ;;
        weather)
            echo "custom_retrieve|10minutes|${CUSTOM_DATASETS_ROOT}/${dataset}/|${dataset}.csv"
            ;;
        *)
            echo "Unknown dataset: ${dataset}" >&2
            exit 1
            ;;
    esac
}

extract_metric() {
    local log_path="$1"
    local metric_name="$2"
    local line
    line="$(grep -E "^${metric_name}:" "${log_path}" | tail -n 1 || true)"
    if [[ -z "${line}" ]]; then
        echo ""
    else
        echo "${line}" | awk '{print $2}' | sed 's/[^0-9.eE+-]//g'
    fi
}

cleanup_ckpt() {
    local final_ckpt="${CHECKPOINTS_DIR}/${model_id}_final.pth"
    local exp_dir="${CHECKPOINTS_DIR}/${model_id}"
    rm -f "${final_ckpt}" || true
    rm -rf "${exp_dir}" || true
}

if [[ -f "${done_flag}" ]]; then
    echo "Skipping finished combo: top_k=${top_k}, rho2=${rho2}"
    exit 0
fi

trap cleanup_ckpt EXIT

if [[ ! -f "${RESULT_CSV}" ]]; then
    echo "top_k,rho2,dataset,mse,mae,ckpt" > "${RESULT_CSV}"
fi

if [[ ! -f "${RESULT_TXT}" ]]; then
    : > "${RESULT_TXT}"
fi

echo "=================================================="
echo "Running combo: top_k=${top_k}, rho2=${rho2}"
echo "model_id=${model_id}"
echo "=================================================="

"${PYTHON_BIN}" "${PRETRAIN_PY}" \
    --model_id "${model_id}" \
    --model ChronosBoltRetrieve \
    --top_k "${top_k}" \
    --retrieve_lookback_length "${retrieve_lookback_length}" \
    --retrieval_database_path "${PRETRAIN_RETRIEVAL_DATABASE_PATH}" \
    --augment_mode "${augment_mode}" \
    --pretrained_model_path "${PRETRAINED_MODEL_PATH}" \
    --context_length "${context_length}" \
    --prediction_length "${prediction_length}" \
    --data_path "${PRETRAIN_DATA_PATH}" \
    --train_steps "${train_steps}" \
    --evaluation_steps "${evaluation_steps}" \
    --optimizer "${optimizer}" \
    --learning_rate "${lr}" \
    --weight_decay "${weight_decay}" \
    --tmax "${tmax}" \
    --drop_prob "${drop_prob}" \
    --batch_size "${batch_size}" \
    --grad_clip_value 1.0 \
    --shuffle_buffer_length "${shuffle_buffer_length}" \
    --rho1 "${rho1}" \
    --rho2 "${rho2}" \
    --rho3 "${rho3}" \
    --rho4 "${rho4}" \
    --freeze_chronos_bolt \
    --checkpoints "${CHECKPOINTS_DIR}" \
    > "${train_log}" 2>&1

ckpt_path="${CHECKPOINTS_DIR}/${model_id}_final.pth"
if [[ ! -f "${ckpt_path}" ]]; then
    echo "Checkpoint not found: ${ckpt_path}" >&2
    exit 1
fi

for dataset in "${datasets[@]}"; do
    mapping="$(dataset_args "${dataset}")"
    IFS='|' read -r data_name metadata_frequency root_path data_path <<< "${mapping}"

    eval_log="${LOG_DIR}/${model_id}_${dataset}_test.log"
    save_file_name="sweep_${model_id}_${dataset}.txt"

    CHECKPOINT_MODEL_PATH="${ckpt_path}" "${PYTHON_BIN}" "${ZEROSHOT_PY}" \
        --root_path "${root_path}" \
        --data_path "${data_path}" \
        --model_id "${dataset}_zeroshot_${context_length}_pred_${prediction_length}_${retrieve_lookback_length}_retrieve_${prediction_length}_${augment_mode}" \
        --data "${data_name}" \
        --top_k "${top_k}" \
        --checkpoint_model_path "${ckpt_path}" \
        --pretrained_model_path "${PRETRAINED_MODEL_PATH}" \
        --seq_len "${context_length}" \
        --label_len 0 \
        --pred_len "${prediction_length}" \
        --lookback_length "${retrieve_lookback_length}" \
        --batch_size "${batch_size}" \
        --num_workers 0 \
        --decay_fac 0.5 \
        --freq 0 \
        --percent 100 \
        --model ChronosBoltRetrieve \
        --gpu_loc "${gpu_loc}" \
        --tmax "${tmax}" \
        --cos 1 \
        --save_file_name "${save_file_name}" \
        --retrieval_database_dir "${RETRIEVAL_DATABASE_DIR}" \
        --dimension 768 \
        --embedding_model_type chronos \
        --metadata_frequency "${metadata_frequency}" \
        --metadata_database_name "${dataset}" \
        --augment_mode "${augment_mode}" \
        --eval_split test \
        > "${eval_log}" 2>&1

    mse="$(extract_metric "${eval_log}" "MSE")"
    mae="$(extract_metric "${eval_log}" "MAE")"

    if [[ -z "${mse}" || -z "${mae}" ]]; then
        echo "Failed to parse metrics for dataset=${dataset}, top_k=${top_k}, rho2=${rho2}" >&2
        exit 1
    fi

    echo "${top_k},${rho2},${dataset},${mse},${mae},${ckpt_path}" >> "${RESULT_CSV}"
    printf "top_k=%s rho2=%s dataset=%s mse=%s mae=%s\n" "${top_k}" "${rho2}" "${dataset}" "${mse}" "${mae}" >> "${RESULT_TXT}"
done

printf "done top_k=%s rho2=%s\n" "${top_k}" "${rho2}" > "${done_flag}"
echo "Finished combo: top_k=${top_k}, rho2=${rho2}"
