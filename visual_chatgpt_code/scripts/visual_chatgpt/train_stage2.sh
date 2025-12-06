#!/bin/bash

# Training script for Visual ChatGPT - Stage 2: Joint Fine-tuning
# This stage unfreezes CG-DETR and trains the entire model end-to-end

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cgdetr

# Configuration
PROJECT_ROOT="/home/Kajal_project/vamshi/baseline/CGDETR"
TRAIN_DATA="${PROJECT_ROOT}/data/highlight_train_release.jsonl"
VAL_DATA="${PROJECT_ROOT}/data/highlight_val_release.jsonl"
AUGMENTED_DATA="${PROJECT_ROOT}/visual_chatgpt/augmented_data/train_augmented.jsonl"

# Load Stage 1 checkpoint
STAGE1_CHECKPOINT="${PROJECT_ROOT}/visual_chatgpt/results/stage1_adapter_warmup/checkpoints/stage1_best.pth"
OUTPUT_DIR="${PROJECT_ROOT}/visual_chatgpt/results/stage2_joint_finetuning"

# Training hyperparameters
BATCH_SIZE=16  # Smaller batch for joint training
AUGMENTATION_RATIO=0.5
NUM_EPOCHS=10  # Fewer epochs for fine-tuning
LR=5e-5  # Lower learning rate for fine-tuning
ADAPTER_DIM=128
NUM_TOKEN_ADAPTERS=2
FUSION_TYPE="gate"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run training
echo "Starting Stage 2: Joint Fine-tuning"
echo "Loading Stage 1 checkpoint: ${STAGE1_CHECKPOINT}"
echo "Output directory: ${OUTPUT_DIR}"
echo "================================================"

python -m visual_chatgpt.train_visual_chatgpt \
    --train_data "${TRAIN_DATA}" \
    --val_data "${VAL_DATA}" \
    --augmented_data "${AUGMENTED_DATA}" \
    --cgdetr_checkpoint "${STAGE1_CHECKPOINT}" \
    --adapter_dim ${ADAPTER_DIM} \
    --num_token_adapters ${NUM_TOKEN_ADAPTERS} \
    --fusion_type ${FUSION_TYPE} \
    --training_stage 2 \
    --num_epochs ${NUM_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --augmentation_ratio ${AUGMENTATION_RATIO} \
    --lr ${LR} \
    --weight_decay 1e-4 \
    --adapter_l2_weight 1e-6 \
    --max_grad_norm 0.5 \
    --output_dir "${OUTPUT_DIR}" \
    --save_interval 2 \
    --log_interval 10 \
    --device cuda \
    --num_workers 4 \
    2>&1 | tee "${OUTPUT_DIR}/training.log"

echo "================================================"
echo "Stage 2 training complete!"
echo "Final model saved at: ${OUTPUT_DIR}/checkpoints/stage2_best.pth"
