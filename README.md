# AEE: Adaptive Early Exit for Reasoning Models

This repository is built on top of [DEER](https://arxiv.org/abs/2504.15895) (Dynamic Early Exit in Reasoning Models) and introduces **AEE (Adaptive Early Exit)** for more robust early-exit decisions during reasoning.

## What AEE Adds Beyond DEER

AEE extends DEER's confidence-based exit mechanism with additional decision signals:

- **EMA-smoothed confidence**: Applies exponential moving average to trial-answer confidence to reduce single-step noise.
- **Adaptive threshold**: Uses a history-aware dynamic threshold instead of a fixed threshold.
- **Temporal consistency window**: Checks whether the latest `n` trial answers are consistent.
- **Disjunctive exit**: Exit when either condition is met:
  - Smoothed confidence exceeds the adaptive threshold, or
  - Trial answers in the recent window are consistent and non-empty.

## Project Structure

- `vllm-aee.py` / `vllm-aee-qwen3.py`: Main AEE inference scripts
- `vllm-deer.py` / `vllm-deer-qwen3.py`: DEER baseline scripts
- `vllm-vanilla-cot.py`: Vanilla Chain-of-Thought scripts
- `vanilla_deer.py`: Transformers-based DEER version
- `check.py`: Evaluation script
- `data/`: Benchmark datasets (MATH, GSM8K, AIME, AMC, GPQA, OlympiadBench, etc.)
- `prompts/`: Prompt templates
- `utils/`: Data processing and grading utilities
- `bashes/`: Handy shell scripts for running experiments

## Environment Setup

We recommend two separate environments:

- **deer** (for DeepSeek-R1-Distill and other general models)  
  See `requirements-deer.txt` (e.g., `vllm==0.6.1`, `transformers==4.46.3`)
- **deer-qwen3** (for Qwen3 models)  
  See `requirements-deer-qwen3.txt` (e.g., `vllm==0.17.1`, `transformers==4.57.6`)

Example with conda:

```bash
conda create -n deer python=3.10 -y
conda activate deer
pip install -r requirements-deer.txt
```

```bash
conda create -n deer-qwen3 python=3.10 -y
conda activate deer-qwen3
pip install -r requirements-deer-qwen3.txt
```

## Quick Start

Run the following commands from the `Code/AEE` directory.

### 1) Run AEE (DeepSeek-R1-Distill)

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 conda run -n deer python vllm-aee.py \
  --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
  --dataset_dir "./data/" \
  --output_path "./outputs" \
  --dataset "math_small" \
  --threshold 0.95 \
  --base_threshold 0.90 \
  --ema_alpha 0.7 \
  --adaptive_beta 1.0 \
  --consistency_window 2 \
  --max_generated_tokens 2048 \
  --think_ratio 0.6 \
  --batch_size 64 \
  --policy avg1 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.80
```

### 2) Run AEE (Qwen3)

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 conda run -n deer-qwen3 python vllm-aee-qwen3.py \
  --model_name_or_path "Qwen/Qwen3-4B" \
  --dataset_dir "./data/" \
  --output_path "./outputs" \
  --dataset "math_small" \
  --threshold 0.95 \
  --base_threshold 0.90 \
  --ema_alpha 0.7 \
  --adaptive_beta 1.0 \
  --consistency_window 2 \
  --max_generated_tokens 1024 \
  --think_ratio 0.8 \
  --batch_size 16 \
  --policy avg2 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85
```

### 3) Evaluate Outputs

```bash
PYTHONNOUSERSITE=1 conda run -n deer python check.py \
  --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
  --data_name "math_small" \
  --generation_path "YOUR_OUTPUT.jsonl"
```

## Supported Models and Datasets

- Models:
  - DeepSeek-R1-Distill-Qwen (1.5B, 7B)
  - Qwen3 (4B, 14B)
- Datasets:
  - MATH, GSM8K, AIME, AMC, GPQA, OlympiadBench

## Acknowledgement and Citation

This project extends the DEER framework. If you use this code, please cite the original DEER paper:

```bibtex
@article{yang2025dynamic,
  title={Dynamic Early Exit in Reasoning Models},
  author={Yang, Chenxu and Si, Qingyi and Duan, Yongjie and Zhu, Zheliang and Zhu, Chenyu and Lin, Zheng and Cao, Li and Wang, Weiping},
  journal={arXiv preprint arXiv:2504.15895},
  year={2025}
}
```
