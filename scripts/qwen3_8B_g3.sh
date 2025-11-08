export PYTHONPATH=./
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export HF_ENDPOINT=https://hf-mirror.com
export DATA_DIR="data/retrieval/G3/"
export OUTPUT_PATH="retriever_model"
export MODEL_NAME="Qwen/Qwen3-Embedding-4B"
export GPU_ID="7"
export LOG_PATH="${LOG_PATH:-log/Qwen3_8B_G3}"

python src/main.py \
    --data_path $DATA_DIR \
    --model_type qwen3 \
    --model_name_or_path $MODEL_NAME \
    --output_path $OUTPUT_PATH \
    --num_epochs 0 \
    --batch_size 1 \
    --learning_rate 2e-5 \
    --warmup_steps 500 \
    --max_seq_length 256 \
    --torch_dtype fp16 \
    --gpu_id $GPU_ID \
    --log_path $LOG_PATH
