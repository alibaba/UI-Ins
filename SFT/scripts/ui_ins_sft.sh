#!/bin/bash
#!/bin/bash

export RUN_NAME="UI-Ins"
export DATASETS_NAME="UI-Ins"
DEFAULT_MASTER_PORT=29500
echo "--- [Step 1/4] Setting up environment variables ---"
export TIMESTAMP=$(date +%Y_%m_%d_%H)
export should_log=True

if [ -n "$MASTER_ADDR" ]; then
    echo "Distributed environment detected. Using variables from the environment:"
    export NNODES=${NNODES}
    export NODE_RANK=${NODE_RANK}
    export MASTER_ADDR=${MASTER_ADDR}
    export MASTER_PORT=${MASTER_PORT}
else
    echo "No distributed environment detected. Configuring for single-machine training."
    export NNODES=1
    export NODE_RANK=0
    export MASTER_ADDR="localhost"
    export MASTER_PORT=${DEFAULT_MASTER_PORT}
fi

echo " - NNODES: ${NNODES}"
echo " - NODE_RANK: ${NODE_RANK}"
echo " - MASTER_ADDR: ${MASTER_ADDR}"
echo " - MASTER_PORT: ${MASTER_PORT}"
echo "----------------------------------------------------"


echo "--- [Step 2/4] Changing working directory ---"
cd /root/code/llama-factory || { echo "Failed to change directory to /root/code/llama-factory"; exit 1; }
echo "Current directory: $(pwd)"
echo "----------------------------------------------------"
echo "--- [Step 3/4] Starting llamafactory training ---"

export FORCE_TORCHRUN=1

llamafactory-cli train \
    --model_name_or_path Qwen/Qwen2.5-VL-32B-Instruct \
    --image_max_pixels 12845056 \
    --video_max_pixels 16384 \
    --trust_remote_code \
    --stage sft \
    --do_train \
    --finetuning_type full \
    --freeze_vision_tower \
    --freeze_multi_modal_projector \
    --freeze_language_model false \
    --deepspeed examples/deepspeed/ds_z3_config.json \
    --dataset "${DATASETS_NAME}" \
    --template qwen2_vl \
    --cutoff_len 32768 \
    --overwrite_cache \
    --preprocessing_num_workers 32 \
    --dataloader_num_workers 32 \
    --output_dir "Your CKPT Dir/${RUN_NAME:-local_job}-${DATASETS_NAME}-${TIMESTAMP}" \
    --logging_steps 10 \
    --save_steps 837 \
    --plot_loss \
    --overwrite_output_dir \
    --save_only_model false \
    --report_to wandb \
    --run_name "${RUN_NAME:-local_run}-${DATASETS_NAME}" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5.0e-6 \
    --num_train_epochs 2.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 \
    --ddp_timeout 180000000 \
    --flash_attn fa2 \
    --enable_liger_kernel \
    --tokenized_path "Your tokenized path"
echo "--- [Step 4/4] Training script finished. ---"