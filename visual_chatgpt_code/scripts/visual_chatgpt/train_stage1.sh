#!/bin/bash

# Training script for Visual ChatGPT - Stage 1: Adapter Warm-up
# This stage freezes CG-DETR and trains only the adapter modules

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cgdetr

# Configuration
PROJECT_ROOT="/home/Kajal_project/vamshi/baseline/CGDETR"
TRAIN_DATA="${PROJECT_ROOT}/data/highlight_train_release.jsonl"
VAL_DATA="${PROJECT_ROOT}/data/highlight_val_release.jsonl"
AUGMENTED_DATA="${PROJECT_ROOT}/visual_chatgpt/augmented_data/train_augmented.jsonl"
CGDETR_CHECKPOINT="${PROJECT_ROOT}/results/qvhighlights/model_best.ckpt"
OUTPUT_DIR="${PROJECT_ROOT}/visual_chatgpt/results/stage1_adapter_warmup"

# Training hyperparameters
BATCH_SIZE=32
AUGMENTATION_RATIO=0.5
NUM_EPOCHS=20
LR=1e-4
ADAPTER_DIM=128
NUM_TOKEN_ADAPTERS=2
FUSION_TYPE="gate"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run training
echo "Starting Stage 1: Adapter Warm-up Training"
echo "Output directory: ${OUTPUT_DIR}"
echo "================================================"

python -m visual_chatgpt.train_visual_chatgpt \
    --train_data "${TRAIN_DATA}" \
    --val_data "${VAL_DATA}" \
    --augmented_data "${AUGMENTED_DATA}" \
    --cgdetr_checkpoint "${CGDETR_CHECKPOINT}" \
    --adapter_dim ${ADAPTER_DIM} \
    --num_token_adapters ${NUM_TOKEN_ADAPTERS} \
    --fusion_type ${FUSION_TYPE} \
    --training_stage 1 \
    --num_epochs ${NUM_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --augmentation_ratio ${AUGMENTATION_RATIO} \
    --lr ${LR} \
    --weight_decay 1e-4 \
    --adapter_l2_weight 1e-5 \
    --max_grad_norm 1.0 \
    --output_dir "${OUTPUT_DIR}" \
    --save_interval 5 \
    --log_interval 10 \
    --device cuda \
    --num_workers 4 \
    2>&1 | tee "${OUTPUT_DIR}/training.log"

echo "================================================"
echo "Stage 1 training complete!"
echo "Best model saved at: ${OUTPUT_DIR}/checkpoints/stage1_best.pth"
