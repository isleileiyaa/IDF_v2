#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
ROOT_DIR="${ROOT_DIR:-$(cd "${PROJECT_DIR}/.." && pwd)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/sweep_outputs/idf_clean_dis}"

RUN_SINGLE="${RUN_SINGLE:-${SCRIPT_DIR}/run_single_combo_idf_clean_dis.sh}"
ALL_RHO2_VALUES=(1e-3 1e-2 1e-1 1 10)
RHO_VALUES="${RHO_VALUES:-}"
RHO_INDEXES="${RHO_INDEXES:-}"

mkdir -p "${OUTPUT_DIR}"

rho2_values=()

if [[ -n "${RHO_VALUES}" ]]; then
    read -r -a rho2_values <<< "${RHO_VALUES}"
elif [[ -n "${RHO_INDEXES}" ]]; then
    IFS=',' read -r -a rho_indexes <<< "${RHO_INDEXES}"
    for idx in "${rho_indexes[@]}"; do
        if ! [[ "${idx}" =~ ^[0-9]+$ ]]; then
            echo "Invalid RHO_INDEXES entry: ${idx}" >&2
            exit 1
        fi
        if (( idx < 0 || idx >= ${#ALL_RHO2_VALUES[@]} )); then
            echo "RHO index out of range: ${idx}" >&2
            exit 1
        fi
        rho2_values+=( "${ALL_RHO2_VALUES[idx]}" )
    done
else
    rho2_values=( "${ALL_RHO2_VALUES[@]}" )
fi

echo "Assigned rho2 values: ${rho2_values[*]}"

for rho2 in "${rho2_values[@]}"; do
    for top_k in $(seq 1 15); do
        echo "top_k=${top_k} rho2=${rho2}"
        bash "${RUN_SINGLE}" "${top_k}" "${rho2}"
    done
done

echo "All assigned rho2 buckets finished."
