#!/usr/bin/env bash

set -euo pipefail

export PYTHONPATH=./

DATASETS=("G1" "G2" "G3")

for DATASET in "${DATASETS[@]}"; do
    QUERY_FILE="data/instruction/${DATASET}_query.json"
    TEST_IDS_FILE="data/retrieval_test_query_ids/${DATASET}_test_query_ids.json"
    OUTPUT_DIR="data/retrieval_graph/${DATASET}"

    echo "Preprocessing graph data for ${DATASET}..."
    python src/preprocess_graph_data.py \
        --query_file "$QUERY_FILE" \
        --test_query_ids "$TEST_IDS_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --dataset_name "$DATASET"

done

echo "Graph preprocessing complete for datasets: ${DATASETS[*]}"
