#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=./
export HF_ENDPOINT=https://hf-mirror.com

DATA_DIR="${DATA_DIR:-data/retrieval/G1/}"
LOG_DIR="${LOG_DIR:-log/eval/Qwen3}"

python src/main.py \
  --data_path "$DATA_DIR" \
  --model_type qwen3 \
  --model_name_or_path "Qwen/Qwen3-0.6B" \
  --max_seq_length 512 \
  --torch_dtype auto \
  --train_batch_size 8 \
  --learning_rate 2e-5 \
  --num_epochs 0 \
  --log_path "$LOG_DIR"
