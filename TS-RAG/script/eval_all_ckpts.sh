#!/usr/bin/env bash
set -u

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <augment_mode> <checkpoint_dir> <zeroshot_script>"
    exit 1
fi

augment_mode="$1"
checkpoint_dir="$2"
zeroshot_script="$3"

if [ ! -d "$checkpoint_dir" ]; then
    echo "Checkpoint directory not found: $checkpoint_dir"
    exit 1
fi

if [ ! -f "$zeroshot_script" ]; then
    echo "Zeroshot script not found: $zeroshot_script"
    exit 1
fi

results_file="${RESULTS_FILE:-best_search_${augment_mode}.txt}"
logs_dir="${LOGS_DIR:-logs/best_search_${augment_mode}}"
mkdir -p "$logs_dir"

: > "$results_file"
echo "augment_mode=${augment_mode}" >> "$results_file"
echo "checkpoint_dir=${checkpoint_dir}" >> "$results_file"
echo "zeroshot_script=${zeroshot_script}" >> "$results_file"
echo "" >> "$results_file"

best_ckpt=""
best_mse=""
best_mae=""

shopt -s nullglob
ckpt_found=0
for ckpt_path in "$checkpoint_dir"/model_steps*.pth; do
    ckpt_found=1
    ckpt_name="$(basename "$ckpt_path")"
    ckpt_stem="${ckpt_name%.pth}"
    log_path="${logs_dir}/${ckpt_stem}.log"

    echo "Evaluating ${ckpt_name} ..."
    CHECKPOINT_MODEL_PATH="$ckpt_path" SAVE_SUFFIX="$ckpt_stem" bash "$zeroshot_script" > "$log_path" 2>&1
    status=$?

    mse_lines="$(grep -E 'mse_mean = ' "$log_path" || true)"
    mae_lines="$(grep -E 'mae_mean = ' "$log_path" || true)"

    if [ "$status" -ne 0 ] || [ -z "$mse_lines" ] || [ -z "$mae_lines" ]; then
        echo "${ckpt_name} FAILED" | tee -a "$results_file"
        continue
    fi

    mse_value="$(printf '%s\n' "$mse_lines" | python -c 'import sys
vals = []
for line in sys.stdin:
    parts = line.strip().split()
    if len(parts) >= 3:
        try:
            vals.append(float(parts[2].rstrip(",")))
        except ValueError:
            pass
print(f"{sum(vals) / len(vals):.6f}" if vals else "")')"
    mae_value="$(printf '%s\n' "$mae_lines" | python -c 'import sys
vals = []
for line in sys.stdin:
    parts = line.strip().split()
    if len(parts) >= 3:
        try:
            vals.append(float(parts[2].rstrip(",")))
        except ValueError:
            pass
print(f"{sum(vals) / len(vals):.6f}" if vals else "")')"

    if [ -z "$mse_value" ] || [ -z "$mae_value" ]; then
        echo "${ckpt_name} FAILED" | tee -a "$results_file"
        continue
    fi

    echo "${ckpt_name} mse=${mse_value} mae=${mae_value}" | tee -a "$results_file"

    if [ -z "$best_ckpt" ]; then
        best_ckpt="$ckpt_name"
        best_mse="$mse_value"
        best_mae="$mae_value"
    else
        python - "$mse_value" "$best_mse" <<'PY'
import sys
cur = float(sys.argv[1])
best = float(sys.argv[2])
sys.exit(0 if cur < best else 1)
PY
        if [ "$?" -eq 0 ]; then
            best_ckpt="$ckpt_name"
            best_mse="$mse_value"
            best_mae="$mae_value"
        fi
    fi
done
shopt -u nullglob

if [ "$ckpt_found" -eq 0 ]; then
    echo "No model_steps*.pth found in ${checkpoint_dir}" | tee -a "$results_file"
    exit 1
fi

echo "" >> "$results_file"
if [ -n "$best_ckpt" ]; then
    echo "BEST checkpoint=${best_ckpt} mse=${best_mse} mae=${best_mae}" | tee -a "$results_file"
else
    echo "BEST checkpoint=NONE" | tee -a "$results_file"
fi
