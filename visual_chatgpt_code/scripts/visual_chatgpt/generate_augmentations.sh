#!/bin/bash
# Full augmentation: Generate augmented queries for entire QVHighlights dataset

set -e

# Activate cgdetr conda environment
source /home/Kajal_project/miniconda3/bin/activate cgdetr

echo "=========================================="
echo "Visual ChatGPT - Full Augmentation"
echo "=========================================="

# Configuration
DATASET_PATH="${1:-data/highlight_train_release.jsonl}"
OUTPUT_DIR="${2:-data/augmented}"
AUGMENT_RATIO="${3:-5}"
STRATEGY="${4:-semantic_paraphrase}"

# Parse optional arguments
MAX_SAMPLES="${5:-}"  # Empty = process all

echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_DIR"
echo "Augmentation ratio: $AUGMENT_RATIO"
echo "Strategy: $STRATEGY"
if [ -n "$MAX_SAMPLES" ]; then
    echo "Max samples: $MAX_SAMPLES"
fi

# Check if dataset exists
if [ ! -f "$DATASET_PATH" ]; then
    echo "Error: Dataset not found at $DATASET_PATH"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build command
CMD="python -m visual_chatgpt.generate_augmentations \
    --dataset_path $DATASET_PATH \
    --output_dir $OUTPUT_DIR \
    --augment_ratio $AUGMENT_RATIO \
    --strategy $STRATEGY \
    --load_in_4bit \
    --device cuda \
    --temperature 0.8 \
    --top_p 0.9 \
    --similarity_threshold 0.75 \
    --keyword_overlap_min 0.5 \
    --checkpoint_interval 100"

# Add max_samples if provided
if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max_samples $MAX_SAMPLES"
fi

# Run augmentation
echo "Starting augmentation..."
eval $CMD

echo "=========================================="
echo "Augmentation complete!"
echo "Output saved to: $OUTPUT_DIR"
echo "=========================================="

# Print statistics
STATS_FILE="$OUTPUT_DIR/stats_augmented_$(basename $DATASET_PATH .jsonl).json"
if [ -f "$STATS_FILE" ]; then
    echo ""
    echo "Statistics:"
    cat "$STATS_FILE"
fi
