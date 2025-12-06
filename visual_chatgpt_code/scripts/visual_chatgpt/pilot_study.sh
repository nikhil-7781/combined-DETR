#!/bin/bash
# Pilot study: Test augmentation pipeline on 100 samples

set -e

# Activate cgdetr conda environment
source /home/Kajal_project/miniconda3/bin/activate cgdetr

echo "=========================================="
echo "Visual ChatGPT - Pilot Study"
echo "=========================================="

# Configuration
DATASET_PATH="data/highlight_train_release.jsonl"
OUTPUT_DIR="data/augmented"
PILOT_SAMPLES=100
AUGMENT_RATIO=5
STRATEGY="semantic_paraphrase"

# Check if dataset exists
if [ ! -f "$DATASET_PATH" ]; then
    echo "Error: Dataset not found at $DATASET_PATH"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run pilot study
echo "Running pilot study with $PILOT_SAMPLES samples..."
python -m visual_chatgpt.generate_augmentations \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --pilot \
    --pilot_samples "$PILOT_SAMPLES" \
    --augment_ratio "$AUGMENT_RATIO" \
    --strategy "$STRATEGY" \
    --device cuda \
    --temperature 0.8 \
    --similarity_threshold 0.75 \
    --keyword_overlap_min 0.5 \
    --checkpoint_interval 50

echo "=========================================="
echo "Pilot study complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
