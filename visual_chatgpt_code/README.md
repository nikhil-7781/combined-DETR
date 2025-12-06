# Visual ChatGPT - Semantic Query Augmentation Guide

This guide explains how to use the semantic query augmentation pipeline to generate diverse, high-quality training data for Video Moment Retrieval using Llama-3.

## Overview

The augmentation pipeline uses a Large Language Model (Llama-3.1-8B-Instruct) to rewrite existing video queries. It includes a rigorous filtering process to ensure that generated queries preserve the exact meaning of the original human annotations.

### Key Components
1.  **Generator** (`visual_chatgpt/llm_augmentation/query_generator.py`): Generates variations using Llama-3.
2.  **Filter** (`visual_chatgpt/llm_augmentation/quality_filter.py`): Validates quality using CLIP similarity and linguistic checks.
3.  **Prompt Manager** (`visual_chatgpt/llm_augmentation/prompt_templates.py`): Manages different rewriting strategies.

## Prerequisites

Ensure you have the required dependencies installed:

```bash
pip install -r requirements_visual_chatgpt.txt
```

You also need a HuggingFace token with access to Llama-3.1-8B-Instruct.
```bash
huggingface-cli login
```

## How to Run Augmentation

The main script is `visual_chatgpt/generate_augmentations.py`. You can run it as a module.

### 1. Run a Pilot Study (Recommended)
Before processing the entire dataset, run a small pilot study to verify settings and quality.

```bash
python -m visual_chatgpt.generate_augmentations \
    --dataset_path data/highlight_train_release.jsonl \
    --output_dir data/augmented \
    --pilot \
    --pilot_samples 50
```
*   **Output**: Check `data/augmented/pilot_highlight_train_release.jsonl` to see the results.

### 2. Run Full Augmentation
To generate augmentations for the entire dataset:

```bash
python -m visual_chatgpt.generate_augmentations \
    --dataset_path data/highlight_train_release.jsonl \
    --output_dir data/augmented \
    --strategy mixed_strategy \
    --augment_ratio 5 \
    --similarity_threshold 0.75
```

## Configuration Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--dataset_path` | Path to the input JSONL file (Required) | - |
| `--output_dir` | Directory to save results | `data/augmented` |
| `--strategy` | Augmentation style: `semantic_paraphrase`, `synonym_replacement`, `structure_variation`, `mixed_strategy` | `semantic_paraphrase` |
| `--augment_ratio` | Number of variations per query | `5` |
| `--similarity_threshold` | CLIP semantic similarity cutoff (0.0-1.0) | `0.75` |
| `--load_in_4bit` | Use 4-bit quantization (saves GPU memory) | `True` |
| `--device` | `cuda` or `cpu` | `cuda` |

## Output Format

The output JSONL file will contain the original data plus a new `augmented_queries` field:

```json
{
  "qid": 123,
  "query": "A person opens the door",
  "duration": 15.5,
  "relevant_windows": [[0, 5]],
  "augmented_queries": [
    "The door is opened by a person",
    "An individual opens the entryway",
    ...
  ]
}
```
