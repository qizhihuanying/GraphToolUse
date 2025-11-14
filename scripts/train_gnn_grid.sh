#!/usr/bin/env bash

set -euo pipefail

export PYTHONPATH=./
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export HF_ENDPOINT=https://hf-mirror.com

# ===== 可配置参数 =====
LOG_ROOT="log/GNN-G3"
DATASET="G3"
GPUS=(0 0 4 4)  
LEARNING_RATES=(1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 3e-6)
GNN_LAYERS_LIST=(2 3 5 10 50)
EPOCHS_LIST=(4 5)
BATCH_SIZE=32
WARMUP_STEPS=0
GNN_HIDDEN_DIM=768
GNN_WEIGHT_DECAY=0
MODEL_NAME="query_aware_gnn"
DATA_PATH="data/retrieval_graph/${DATASET}"
OUTPUT_ROOT="retriever_model"

TOTAL_JOBS=$(( ${#LEARNING_RATES[@]} * ${#GNN_LAYERS_LIST[@]} * ${#EPOCHS_LIST[@]} ))
NUM_SLOTS=${#GPUS[@]}
if [[ ${NUM_SLOTS} -eq 0 ]]; then
  echo "GPUS 数组不能为空" >&2
  exit 1
fi

# 预生成所有组合
combos=()
for lr in "${LEARNING_RATES[@]}"; do
  for layers in "${GNN_LAYERS_LIST[@]}"; do
    for epochs in "${EPOCHS_LIST[@]}"; do
      combos+=("${lr},${layers},${epochs}")
    done
  done
done

start_job() {
  local combo="$1"
  local gpu="$2"
  IFS=',' read -r lr layers epochs <<< "$combo"
  local run_name="gnn_${DATASET}_lr=${lr}_L=${layers}_epoch=${epochs}"
  local log_path="${LOG_ROOT}"
  local output_path="${OUTPUT_ROOT}/${run_name}"

  printf '[JOB %s] GPU=%s LR=%s LAYERS=%s EPOCHS=%s\n' "$run_name" "$gpu" "$lr" "$layers" "$epochs" >&2
  CUDA_VISIBLE_DEVICES="${gpu}" \
  python src/main.py \
    --data_path "$DATA_PATH" \
    --model_type gnn \
    --model_name_or_path "$MODEL_NAME" \
    --output_path "$output_path" \
    --num_epochs "$epochs" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$lr" \
    --warmup_steps "$WARMUP_STEPS" \
    --log_path "$log_path" \
    --gnn_hidden_dim "$GNN_HIDDEN_DIM" \
    --gnn_layers "$layers" \
    --gnn_weight_decay "$GNN_WEIGHT_DECAY" \
    --gpu_id "$gpu" \
    &
}

index=0
launched=0

while [[ ${index} -lt ${TOTAL_JOBS} ]]; do
  batch_size=$(( TOTAL_JOBS - index ))
  if [[ ${batch_size} -gt ${NUM_SLOTS} ]]; then
    batch_size=${NUM_SLOTS}
  fi

  pids=()
  for ((slot=0; slot<batch_size; slot++)); do
    combo=${combos[$((index + slot))]}
    gpu=${GPUS[$slot]}
    start_job "$combo" "$gpu"
    pid=$!
    pids+=("$pid")
    launched=$((launched + 1))
  done

  for pid in "${pids[@]}"; do
    wait "$pid"
  done

  index=$((index + batch_size))
  echo "完成 ${index}/${TOTAL_JOBS} 个作业"
done

echo "所有 ${TOTAL_JOBS} 个超参组合已完成。"
