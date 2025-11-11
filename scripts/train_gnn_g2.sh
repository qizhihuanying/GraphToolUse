#!/usr/bin/env bash

export PYTHONPATH=./
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export HF_ENDPOINT=https://hf-mirror.com
export DATA_PATH="data/retrieval_graph/G2"
export MODEL_NAME="query_aware_gnn"
export OUTPUT_PATH="retriever_model"
export GPU_ID="6"
export LOG_PATH="log/GNN-G2"

export CUDA_VISIBLE_DEVICES="$GPU_ID"

python src/main.py \
    --data_path $DATA_PATH \
    --model_type gnn \
    --model_name_or_path $MODEL_NAME \
    --output_path $OUTPUT_PATH \
    --num_epochs 5 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --warmup_steps 0 \
    --gpu_id $GPU_ID \
    --log_path $LOG_PATH \
    --gnn_hidden_dim 768 \
    --gnn_layers 2 \
    --gnn_weight_decay 0
