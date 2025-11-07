export PYTHONPATH=./
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export HF_ENDPOINT=https://hf-mirror.com
export DATA_DIR="data/retrieval/G1/"
export MODEL_NAME="bert-base-uncased"
export OUTPUT_PATH="retriever_model"
export GPU_ID="2"
export LOG_PATH="log/G1"

python src/main.py \
    --data_path $DATA_DIR \
    --model_name_or_path $MODEL_NAME \
    --output_path $OUTPUT_PATH \
    --num_epochs 5 \
    --train_batch_size 32 \
    --learning_rate 2e-5 \
    --warmup_steps 500 \
    --max_seq_length 256 \
    --gpu_id $GPU_ID \
    --log_path $LOG_PATH
