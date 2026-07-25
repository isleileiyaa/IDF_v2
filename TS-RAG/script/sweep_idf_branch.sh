#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/fenglei/TS-RAG-main"
PROJECT_DIR="${ROOT_DIR}/TS-RAG"
PRETRAIN_PY="${PROJECT_DIR}/pretrain.py"
ZEROSHOT_PY="${PROJECT_DIR}/zeroshot.py"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${PROJECT_DIR}/checkpoints}"
PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-${PROJECT_DIR}/checkpoints/base/}"
PRETRAIN_DATA_PATH="${PRETRAIN_DATA_PATH:-${ROOT_DIR}/datasets/pretrain/pretrain_pairs_ctx512}"
PRETRAIN_RETRIEVAL_DATABASE_PATH="${PRETRAIN_RETRIEVAL_DATABASE_PATH:-${ROOT_DIR}/retrieval_database/pretrain/retrieval_database_512.parquet}"
RETRIEVAL_DATABASE_DIR="${RETRIEVAL_DATABASE_DIR:-${ROOT_DIR}/retrieval_database/}"
ETT_ROOT_PATH="${ETT_ROOT_PATH:-${ROOT_DIR}/datasets/ETT-small/}"
CUSTOM_DATASETS_ROOT="${CUSTOM_DATASETS_ROOT:-${ROOT_DIR}/datasets}"

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/sweep_outputs/idf_branch}"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

SUMMARY_CSV="${OUTPUT_DIR}/summary.csv"
BEST_GLOBAL_TXT="${OUTPUT_DIR}/best_global.txt"
echo "top_k,lr,dropout,ckpt,val_mse,val_mae,test_mse,test_mae" > "${SUMMARY_CSV}"
: > "${BEST_GLOBAL_TXT}"

top_ks=(3 5 8 10 12 )
lrs=(1e-4 3e-4 5e-4 )
dropouts=( 0.15 0.2 0.25)
datasets=(ETTh1 ETTh2 ETTm1 ETTm2 weather )

retrieve_lookback_length=512
context_length=512
prediction_length=64
train_steps="${TRAIN_STEPS:-10000}"
evaluation_steps="${EVALUATION_STEPS:-1000}"
optimizer="${OPTIMIZER:-adamw}"
weight_decay="${WEIGHT_DECAY:-0.01}"
tmax="${TMAX:-20}"
batch_size="${BATCH_SIZE:-256}"
shuffle_buffer_length="${SHUFFLE_BUFFER_LENGTH:-10000}"
gpu_loc="${GPU_LOC:-0}"

best_global_val_mse=""
best_global_line=""

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

run_eval() {
    local ckpt_path="$1"
    local dataset="$2"
    local top_k="$3"
    local eval_split="$4"
    local log_path="$5"

    local mapping
    mapping="$(dataset_args "${dataset}")"
    IFS='|' read -r data_name metadata_frequency root_path data_path <<< "${mapping}"

    CHECKPOINT_MODEL_PATH="${ckpt_path}" python "${ZEROSHOT_PY}" \
        --root_path "${root_path}" \
        --data_path "${data_path}" \
        --model_id "${dataset}_zeroshot_512_pred_64_512_retrieve_64_idf_branch" \
        --data "${data_name}" \
        --top_k "${top_k}" \
        --checkpoint_model_path "${ckpt_path}" \
        --pretrained_model_path "${PRETRAINED_MODEL_PATH}" \
        --seq_len 512 \
        --label_len 0 \
        --pred_len 64 \
        --lookback_length 512 \
        --batch_size 256 \
        --num_workers 0 \
        --decay_fac 0.5 \
        --freq 0 \
        --percent 100 \
        --model ChronosBoltRetrieve \
        --gpu_loc "${gpu_loc}" \
        --tmax 20 \
        --cos 1 \
        --save_file_name "$(basename "${log_path}").txt" \
        --retrieval_database_dir "${RETRIEVAL_DATABASE_DIR}" \
        --dimension 768 \
        --embedding_model_type chronos \
        --metadata_frequency "${metadata_frequency}" \
        --metadata_database_name "${dataset}" \
        --augment_mode idf_branch \
        --eval_split "${eval_split}" \
        > "${log_path}" 2>&1
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

mean_of_values() {
    python - "$@" <<'PY'
import sys
vals = [float(x) for x in sys.argv[1:]]
print(sum(vals) / len(vals))
PY
}

is_less_than() {
    python - "$1" "$2" <<'PY'
import sys
cur = float(sys.argv[1])
best = float(sys.argv[2])
sys.exit(0 if cur < best else 1)
PY
}

for top_k in "${top_ks[@]}"; do
    for lr in "${lrs[@]}"; do
        for dropout in "${dropouts[@]}"; do
            model_id="data50m_idf_branch_${context_length}_pred${prediction_length}_lookback${retrieve_lookback_length}_top${top_k}_lr${lr}_drop${dropout}_${optimizer}_cosanneal_step${train_steps}_bs${batch_size}_no_embeddingtuning"
            train_log="${LOG_DIR}/${model_id}_train.log"

            python "${PRETRAIN_PY}" \
                --model_id "${model_id}" \
                --model ChronosBoltRetrieve \
                --top_k "${top_k}" \
                --retrieve_lookback_length "${retrieve_lookback_length}" \
                --retrieval_database_path "${PRETRAIN_RETRIEVAL_DATABASE_PATH}" \
                --augment_mode idf_branch \
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
                --drop_prob "${dropout}" \
                --batch_size "${batch_size}" \
                --grad_clip_value 1.0 \
                --shuffle_buffer_length "${shuffle_buffer_length}" \
                --freeze_chronos_bolt \
                --checkpoints "${CHECKPOINTS_DIR}" \
                > "${train_log}" 2>&1

            experiment_dir="${CHECKPOINTS_DIR}/${model_id}"
            shopt -s nullglob
            ckpts=()
            if [[ -d "${experiment_dir}" ]]; then
                ckpts=( "${experiment_dir}"/model_steps*.pth )
            fi
            final_ckpt="${CHECKPOINTS_DIR}/${model_id}_final.pth"
            if [[ ${#ckpts[@]} -eq 0 && -f "${final_ckpt}" ]]; then
                ckpts=( "${final_ckpt}" )
            fi
            shopt -u nullglob

            if [[ ${#ckpts[@]} -eq 0 ]]; then
                echo "No checkpoint found for ${model_id}" >&2
                exit 1
            fi

            best_ckpt=""
            best_val_mse=""
            best_val_mae=""

            for ckpt_path in "${ckpts[@]}"; do
                val_mses=()
                val_maes=()

                for dataset in "${datasets[@]}"; do
                    eval_log="${LOG_DIR}/$(basename "${ckpt_path}" .pth)_${dataset}_val.log"
                    run_eval "${ckpt_path}" "${dataset}" "${top_k}" "val" "${eval_log}"

                    mse="$(extract_metric "${eval_log}" "MSE")"
                    mae="$(extract_metric "${eval_log}" "MAE")"
                    if [[ -z "${mse}" || -z "${mae}" ]]; then
                        echo "Failed to parse val metrics for ${ckpt_path} on ${dataset}" >&2
                        exit 1
                    fi
                    val_mses+=( "${mse}" )
                    val_maes+=( "${mae}" )
                done

                avg_val_mse="$(mean_of_values "${val_mses[@]}")"
                avg_val_mae="$(mean_of_values "${val_maes[@]}")"

                if [[ -z "${best_ckpt}" ]]; then
                    best_ckpt="${ckpt_path}"
                    best_val_mse="${avg_val_mse}"
                    best_val_mae="${avg_val_mae}"
                else
                    if is_less_than "${avg_val_mse}" "${best_val_mse}"; then
                        best_ckpt="${ckpt_path}"
                        best_val_mse="${avg_val_mse}"
                        best_val_mae="${avg_val_mae}"
                    fi
                fi
            done

            test_mses=()
            test_maes=()
            for dataset in "${datasets[@]}"; do
                test_log="${LOG_DIR}/$(basename "${best_ckpt}" .pth)_${dataset}_test.log"
                run_eval "${best_ckpt}" "${dataset}" "${top_k}" "test" "${test_log}"

                mse="$(extract_metric "${test_log}" "MSE")"
                mae="$(extract_metric "${test_log}" "MAE")"
                if [[ -z "${mse}" || -z "${mae}" ]]; then
                    echo "Failed to parse test metrics for ${best_ckpt} on ${dataset}" >&2
                    exit 1
                fi
                test_mses+=( "${mse}" )
                test_maes+=( "${mae}" )
            done

            avg_test_mse="$(mean_of_values "${test_mses[@]}")"
            avg_test_mae="$(mean_of_values "${test_maes[@]}")"

            best_keep="${CHECKPOINTS_DIR}/${model_id}_best_val.pth"
            cp -f "${best_ckpt}" "${best_keep}"
            best_ckpt="${best_keep}"

            echo "${top_k},${lr},${dropout},${best_ckpt},${best_val_mse},${best_val_mae},${avg_test_mse},${avg_test_mae}" >> "${SUMMARY_CSV}"

            for ckpt_path in "${ckpts[@]}"; do
                if [[ "${ckpt_path}" != "${best_ckpt}" ]]; then
                    rm -f "${ckpt_path}"
                fi
            done

            if [[ -z "${best_global_val_mse}" ]]; then
                best_global_val_mse="${best_val_mse}"
                best_global_line="top_k=${top_k}
lr=${lr}
dropout=${dropout}
ckpt=${best_ckpt}
val_mse=${best_val_mse}
val_mae=${best_val_mae}
test_mse=${avg_test_mse}
test_mae=${avg_test_mae}"
            else
                if is_less_than "${best_val_mse}" "${best_global_val_mse}"; then
                    best_global_val_mse="${best_val_mse}"
                    best_global_line="top_k=${top_k}
lr=${lr}
dropout=${dropout}
ckpt=${best_ckpt}
val_mse=${best_val_mse}
val_mae=${best_val_mae}
test_mse=${avg_test_mse}
test_mae=${avg_test_mae}"
                fi
            fi
        done
    done
done

printf '%s\n' "${best_global_line}" > "${BEST_GLOBAL_TXT}"
echo "summary.csv written to ${SUMMARY_CSV}"
echo "best_global.txt written to ${BEST_GLOBAL_TXT}"
