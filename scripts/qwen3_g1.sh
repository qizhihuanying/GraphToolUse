export PYTHONPATH=./
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export HF_ENDPOINT=https://hf-mirror.com
export DATA_DIR="data/retrieval/G1/"
export OUTPUT_PATH="retriever_model"
export MODEL_NAME="Qwen/Qwen3-Embedding-0.6B"
export GPU_ID="7"
export LOG_PATH="${LOG_PATH:-log/Qwen3_G1}"

python src/main.py \
    --data_path $DATA_DIR \
    --model_type qwen3 \
    --model_name_or_path $MODEL_NAME \
    --output_path $OUTPUT_PATH \
    --num_epochs 0 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --warmup_steps 500 \
    --max_seq_length 256 \
    --torch_dtype auto \
    --gpu_id $GPU_ID \
    --log_path $LOG_PATH
