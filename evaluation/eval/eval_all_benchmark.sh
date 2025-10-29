#!/bin/bash

# --- 前置设置 (与原脚本相同) ---
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_FLASH_ATTN_VERSION=2

# 定义基础目录
BASE_DIR=$(dirname "$(readlink -f "$0")")

# 定义数据集配置
declare -A IMG_PATHS=(
  ["screenspot_pro"]="Your data dir/screenspotPro_imgs"
  ["screenspot_v2"]="Your data dir/screenspotv2_image"
  ["I2E"]="Your data dir/I2E_imgs"
  ["MMBench"]="Your data dir/MMBench_imgs"
  ["showclick"]="Your data dir/showdown_imgs"
)

declare -A TEST_PATHS=(
  ["screenspot_pro"]="${BASE_DIR}/SSpro_annotation"
  ["screenspot_v2"]="${BASE_DIR}/SSV2_annotation"
  ["I2E"]="${BASE_DIR}/I2E_annotation"
  ["MMBench"]="${BASE_DIR}/MMBench_GUI_L2_annotation"
  ["showclick"]="${BASE_DIR}/Showdown_annotation"
)

MODEL_PATH="$1"
MODEL_TYPE="${2:-qwen2_5vl}"
MAX_PIXELS="${3:-2116800}"
MAGIC_PROMPT="${4:-true}"
USE_GUIDE_TEXT="${5:-true}"

echo "Using model type: ${MODEL_TYPE}"
echo "Using model path: ${MODEL_PATH}"
echo "Using max pixels: ${MAX_PIXELS}"
echo "Using magic prompt: ${MAGIC_PROMPT}"

TASK="all"
LANGUAGE="en"
GT_TYPE="positive"
INST_STYLE="instruction"

MODEL_NAME=$(basename "${MODEL_PATH}")
OUTPUT_DIR="${BASE_DIR}/results/${MODEL_NAME}"
mkdir -p "${OUTPUT_DIR}"

DATASETS_TO_RUN=("screenspot_pro" "screenspot_v2" "I2E" "MMBench" "showclick")
NUM_DATASETS=${#DATASETS_TO_RUN[@]}
SPLIT_POINT=$(( (NUM_DATASETS + 1) / 2 ))

GROUP1_DATASETS=("${DATASETS_TO_RUN[@]:0:$SPLIT_POINT}")
GROUP2_DATASETS=("${DATASETS_TO_RUN[@]:$SPLIT_POINT}")

echo "Total datasets: ${NUM_DATASETS}"
echo "Group 1 (GPUs 0,1,2,3) will process: ${GROUP1_DATASETS[*]}"
echo "Group 2 (GPUs 4,5,6,7) will process: ${GROUP2_DATASETS[*]}"


run_evaluation_group() {
    local gpus="$1"

    eval "local datasets=(\"\${@:2}\")"

    for dataset in "${datasets[@]}"; do
        echo "========================================"
        echo "[GPU Group ${gpus}] Processing dataset: ${dataset}"
        echo "Model: ${MODEL_NAME}"
        echo "========================================"
        
        LOG_PATH="${OUTPUT_DIR}/${dataset}.json"
        
        CUDA_VISIBLE_DEVICES="${gpus}" python eval_screenspot_pro.py \
            --model_type "${MODEL_TYPE}" \
            --model_name_or_path "${MODEL_PATH}" \
            --screenspot_imgs "${IMG_PATHS[$dataset]}" \
            --screenspot_test "${TEST_PATHS[$dataset]}" \
            --task "${TASK}" \
            --language "${LANGUAGE}" \
            --gt_type "${GT_TYPE}" \
            --log_path "${LOG_PATH}" \
            --inst_style "${INST_STYLE}" \
            --magic_prompt "${MAGIC_PROMPT}" \
            --max_pixels "${MAX_PIXELS}" \
            --use_guide_text "${USE_GUIDE_TEXT}"
        echo "[GPU Group ${gpus}] Finished processing ${dataset}, results saved to ${LOG_PATH}"
        echo "========================================"
    done
}

run_evaluation_group "0,1,2,3" "${GROUP1_DATASETS[@]}" &

if [ ${#GROUP2_DATASETS[@]} -gt 0 ]; then
    run_evaluation_group "4,5,6,7" "${GROUP2_DATASETS[@]}" &
fi


echo "Waiting for all parallel evaluations to complete..."
wait
echo "All parallel evaluations have finished."
