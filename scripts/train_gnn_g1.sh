#!/usr/bin/env bash

export PYTHONPATH=./
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export HF_ENDPOINT=https://hf-mirror.com
export DATA_PATH="data/retrieval_graph/G1"
export MODEL_NAME="query_aware_gnn"
export OUTPUT_PATH="retriever_model"
export GPU_ID="0"
export LOG_PATH="log/GNN-G1"

python src/main.py \
    --data_path $DATA_PATH \
    --model_type gnn \
    --model_name_or_path $MODEL_NAME \
    --output_path $OUTPUT_PATH \
    --num_epochs 5 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --warmup_steps 500 \
    --gpu_id $GPU_ID \
    --log_path $LOG_PATH \
    --gnn_hidden_dim 768 \
    --gnn_layers 2 \
    --gnn_weight_decay 0.01
