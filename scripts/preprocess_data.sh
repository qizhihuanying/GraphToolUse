#!/bin/bash

set -euo pipefail

export PYTHONPATH=./

DATASETS=("G1" "G2" "G3")

for DATASET in "${DATASETS[@]}"; do
    QUERY_FILE="data/instruction/${DATASET}_query.json"
    INDEX_FILE="data/retrieval_test_query_ids/${DATASET}_test_query_ids.json"
    OUTPUT_DIR="data/retrieval/${DATASET}"

    echo "Preprocessing retriever data for ${DATASET}..."
    python src/preprocess_data.py \
        --query_file "${QUERY_FILE}" \
        --index_file "${INDEX_FILE}" \
        --dataset_name "${DATASET}" \
        --output_dir "${OUTPUT_DIR}"
done

echo "Retriever preprocessing complete for datasets: ${DATASETS[*]}"
