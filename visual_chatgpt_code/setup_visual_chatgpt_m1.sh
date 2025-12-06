#!/bin/bash
# Setup script for Visual ChatGPT Milestone 1

set -e

echo "=========================================="
echo "Visual ChatGPT - Setup (Milestone 1)"
echo "=========================================="

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Working directory: $(pwd)"

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip install -r requirements_visual_chatgpt.txt

# Download spaCy model
echo ""
echo "Downloading spaCy model..."
python -m spacy download en_core_web_sm

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p data/augmented
mkdir -p logs

# Test installation
echo ""
echo "Running installation tests..."
python visual_chatgpt/test_installation.py

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "You can now run:"
echo "  1. Pilot study: bash scripts/visual_chatgpt/pilot_study.sh"
echo "  2. Full augmentation: bash scripts/visual_chatgpt/generate_augmentations.sh"
echo ""
