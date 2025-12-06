#!/usr/bin/env python3
"""
Main script for semantic data augmentation.
Usage: python -m visual_chatgpt.generate_augmentations [args]
"""

import sys
import os
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visual_chatgpt.config_augmentation import AugmentationConfig
from visual_chatgpt.llm_augmentation.query_generator import LlamaQueryGenerator
from visual_chatgpt.llm_augmentation.quality_filter import SemanticQualityFilter
from visual_chatgpt.llm_augmentation.diversity_metrics import DiversityMetrics
from visual_chatgpt.llm_augmentation.batch_augmenter import BatchQueryAugmenter


def setup_logging(level: str):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('augmentation.log')
        ]
    )


def main():
    """Main augmentation pipeline."""
    # Parse config
    config = AugmentationConfig()
    opt = config.parse()
    
    # Setup logging
    setup_logging(opt.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("VISUAL CHATGPT - SEMANTIC DATA AUGMENTATION")
    logger.info("=" * 80)
    logger.info(f"Dataset: {opt.dataset_path}")
    logger.info(f"Output: {opt.output_dir}/{opt.output_filename}")
    logger.info(f"Strategy: {opt.strategy}")
    logger.info(f"Augmentation ratio: {opt.augment_ratio}")
    logger.info("=" * 80)
    
    # Initialize components
    logger.info("Initializing Llama-3.1-8B-Instruct...")
    generator = LlamaQueryGenerator(
        model_name=opt.model_name,
        device=opt.device,
        load_in_4bit=opt.load_in_4bit,
        max_new_tokens=opt.max_new_tokens,
        temperature=opt.temperature,
        top_p=opt.top_p
    )
    
    logger.info("Initializing quality filter...")
    quality_filter = SemanticQualityFilter(
        similarity_threshold=opt.similarity_threshold,
        keyword_overlap_min=opt.keyword_overlap_min,
        device=opt.device
    )
    
    logger.info("Initializing diversity calculator...")
    diversity_calculator = DiversityMetrics(device=opt.device)
    
    # Create batch augmenter
    augmenter = BatchQueryAugmenter(
        generator=generator,
        quality_filter=quality_filter,
        diversity_calculator=diversity_calculator,
        output_dir=opt.output_dir
    )
    
    # Run augmentation
    if opt.pilot:
        logger.info(f"Running PILOT STUDY with {opt.pilot_samples} samples")
        stats = augmenter.pilot_study(
            dataset_path=opt.dataset_path,
            n_samples=opt.pilot_samples,
            augment_ratio=opt.augment_ratio,
            strategy=opt.strategy
        )
    else:
        logger.info("Running FULL AUGMENTATION")
        stats = augmenter.augment_dataset(
            dataset_path=opt.dataset_path,
            output_filename=opt.output_filename,
            augment_ratio=opt.augment_ratio,
            strategy=opt.strategy,
            max_samples=opt.max_samples,
            checkpoint_interval=opt.checkpoint_interval
        )
    
    # Summary
    logger.info("=" * 80)
    logger.info("AUGMENTATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Original queries: {stats['total_original']}")
    logger.info(f"Generated variations: {stats['total_generated']}")
    logger.info(f"Filtered variations: {stats['total_filtered']}")
    logger.info(f"Filter pass rate: {stats['filter_pass_rate']:.2%}")
    logger.info(f"Average per query: {stats['avg_filtered_per_query']:.2f}")
    logger.info("=" * 80)
    
    # Cleanup
    generator.cleanup()
    
    logger.info("Done!")
    
    return stats


if __name__ == "__main__":
    main()
