# Evaluation

## Data Preparation

We have already provide whole annotations of grounding benchmarks in ScreenSpot-Pro format, you can download each benchmark images from following links:

- MMBench-GUI: [here](https://huggingface.co/datasets/OpenGVLab/MMBench-GUI)

- UI-I2E Bench: [here](https://huggingface.co/datasets/vaundys/I2E-Bench)

- ScreenSpot-Pro: [here](https://huggingface.co/datasets/likaixin/ScreenSpot-Pro)

- ScreenSpot-V2: [here](https://huggingface.co/datasets/OS-Copilot/ScreenSpot-v2)

- ShowDown-Click-Dev: [here](https://huggingface.co/datasets/generalagents/showdown-clicks)

## Quick Evaluation

You can directly use the following command:

```bash
MAGIC_PROMPT=False && \
USE_GUIDE_TEXT=False && \
model_dir=Your model path && \
sh eval/eval_all_benchmarks.sh ${model_dir} qwen2_5vl 12845056 MAGIC_PROMPT USE_GUIDE_TEXT
```

