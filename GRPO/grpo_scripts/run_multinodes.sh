#!/bin/bash
set -x

export GPUS_PER_NODE=${NPROC_PER_NODE:-8}
export NNODES=${NNODES:-1}
export NODE_RANK=${RANK:-0}
export MASTER_PORT=${MASTER_PORT:-6379}
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1

if [ $NODE_RANK -eq 0 ]; then
    echo "Starting Ray head node on port ${MASTER_PORT} with ${GPUS_PER_NODE} GPUs..."
    ray start --head --port=${MASTER_PORT} --num-gpus=${GPUS_PER_NODE} --block &
    echo "Waiting for Ray head to initialize..."
    sleep 5m
    echo "Starting the training script on the head node..."
    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        custom_reward_function.path=grpo_scripts/reward_fn.py \
        custom_reward_function.name=${REWARD_FUNC} \
        data.train_files=${TRAIN_FILE} \
        data.val_files=${TEST_FILE} \
        data.train_batch_size=${BATCH_SIZE} \
        data.max_prompt_length=10240 \
        data.max_response_length=256 \
        data.filter_overlong_prompts=False \
        data.filter_overlong_prompts_workers=64 \
        data.truncation='right' \
        data.image_key=images \
        actor_rollout_ref.model.path=${ACTOR_MODEL} \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.strategy="fsdp" \
        actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=10496 \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.optim.lr=${LR} \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.disable_log_stats=False \
        actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
        actor_rollout_ref.rollout.tensor_model_parallel_size=${TENSOR_PARREL_SIZE} \
        actor_rollout_ref.ref.ulysses_sequence_parallel_size=2 \
        actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
        actor_rollout_ref.rollout.enable_chunked_prefill=False \
        actor_rollout_ref.rollout.max_num_batched_tokens=102400 \
        actor_rollout_ref.rollout.max_num_seqs=10240 \
        actor_rollout_ref.rollout.enforce_eager=True \
        actor_rollout_ref.actor.entropy_checkpointing=True \
        actor_rollout_ref.rollout.free_cache_engine=True \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE} \
        actor_rollout_ref.rollout.n=${ROLLOUT_NUM} \
        algorithm.use_kl_in_reward=False \
        trainer.total_epochs=3 \
        trainer.project_name='qwen_sfted_rl' \
        trainer.experiment_name=${WANDB_PROJECT} \
        trainer.logger=['console','tensorboard'] \
        trainer.nnodes=${NNODES} \
        trainer.n_gpus_per_node=${GPUS_PER_NODE} \
        trainer.save_freq=25 \
        trainer.val_before_train=False \
        trainer.test_freq=0 \
        trainer.critic_warmup=0 \
        trainer.default_local_dir=${CHECKPOINT_DIR}

else
    if [ -z "${MASTER_ADDR}" ]; then
        echo "FATAL: MASTER_ADDR is not set. Worker node cannot find the head node."
        exit 1
    fi
    echo "Starting Ray worker node, attempting to connect to ${MASTER_ADDR}:${MASTER_PORT}..."
    ray start --address=${MASTER_ADDR}:${MASTER_PORT} --num-gpus=${GPUS_PER_NODE} --block
fi