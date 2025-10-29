# RL for UI-Ins

## Data Preparation
You should first organize data as a jsonl file in the following format:
```json
{
    "id": "idx_xxx", 
    "image": "image path", 
    "bbox": [x1,y1,x2,y2(all number should resized in 0-1000)], 
    "conversations": 
    [
        {
            "from": "human", "value": "instruction here"
        }, 
        {
            "from": "gpt", "value": " "
        }
    ]
}
```

Then, package the jsonl by the data_processing/trans_to_parquet.py:

```bash
python trans_to_parquet.py
```
## Training
To train UI-Ins by GRPO, use the following command:

```bash
TENSOR_PARREL_SIZE=2 \
REWARD_FUNC=click_reward \
ROLLOUT_NUM=2 \
LR=1e-5 \
TRAIN_FILE=train.parquet \
TEST_FILE=test.parquet \
ACTOR_MODEL=SFTed model \
CHECKPOINT_DIR=output \
BATCH_SIZE=256 \
MINI_BATCH_SIZE=256 \
MICRO_BATCH_SIZE=1 \
GPU_MEMORY_UTILIZATION=0.6 \
LOG_PROB_MICRO_BATCH_SIZE=1 \
sh grpo_scripts/run_multinodes.sh
```

## Model merge
To merge model, you should use the following command:
```bash
python grpo_scripts/model_merger.py --loacl_dir your_ckpt_path/global_step_xx/actor
```
