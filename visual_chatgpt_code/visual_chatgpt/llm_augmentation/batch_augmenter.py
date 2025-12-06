"""
Batch augmentation pipeline orchestrator.
Coordinates LLM generation, quality filtering, and output saving.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import logging
from datetime import datetime

from .query_generator import LlamaQueryGenerator
from .quality_filter import SemanticQualityFilter
from .diversity_metrics import DiversityMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file."""
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]


def save_jsonl(data: List[Dict], path: str):
    """Save data to JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    logger.info(f"Saved {len(data)} items to {path}")


class BatchQueryAugmenter:
    """
    End-to-end pipeline for semantic data augmentation.
    
    Pipeline:
    1. Load original dataset
    2. Generate variations with Llama
    3. Apply quality filtering
    4. Compute diversity metrics
    5. Save augmented dataset
    """
    
    def __init__(
        self,
        generator: LlamaQueryGenerator,
        quality_filter: SemanticQualityFilter,
        diversity_calculator: DiversityMetrics,
        output_dir: str = "data/augmented"
    ):
        """
        Initialize batch augmenter.
        
        Args:
            generator: Llama query generator instance
            quality_filter: Quality filter instance
            diversity_calculator: Diversity metrics instance
            output_dir: Output directory for augmented data
        """
        self.generator = generator
        self.quality_filter = quality_filter
        self.diversity_calculator = diversity_calculator
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def augment_dataset(
        self,
        dataset_path: str,
        output_filename: str,
        augment_ratio: int = 5,
        strategy: str = "semantic_paraphrase",
        max_samples: Optional[int] = None,
        checkpoint_interval: int = 100,
    ) -> Dict:
        """
        Augment entire dataset with quality control.
        
        Args:
            dataset_path: Path to original dataset JSONL
            output_filename: Output filename (without path)
            augment_ratio: Number of variations per query
            strategy: Augmentation strategy
            max_samples: Max samples to process (None = all)
            checkpoint_interval: Save checkpoint every N samples
        
        Returns:
            Statistics dict
        """
        logger.info(f"Starting augmentation pipeline for {dataset_path}")
        logger.info(f"Strategy: {strategy}, Augmentation ratio: {augment_ratio}")
        
        # Load original dataset
        original_data = load_jsonl(dataset_path)
        if max_samples:
            original_data = original_data[:max_samples]
        
        logger.info(f"Loaded {len(original_data)} original samples")
        
        # Prepare output path
        output_path = self.output_dir / output_filename
        checkpoint_path = self.output_dir / f"checkpoint_{output_filename}"
        
        # Process in batches
        augmented_data = []
        generation_stats = {
            'total_original': len(original_data),
            'total_generated': 0,
            'total_filtered': 0,
            'generation_failures': 0,
        }
        
        for idx, sample in enumerate(tqdm(original_data, desc="Augmenting queries")):
            try:
                # Extract metadata
                video_metadata = {
                    'duration': sample.get('duration', 150.0),
                    'start_time': 0.0,
                    'end_time': 0.0
                }
                
                # Use first relevant window if available
                if 'relevant_windows' in sample and sample['relevant_windows']:
                    video_metadata['start_time'] = sample['relevant_windows'][0][0]
                    video_metadata['end_time'] = sample['relevant_windows'][0][1]
                
                # Generate variations
                variations = self.generator.generate_variations(
                    original_query=sample['query'],
                    video_metadata=video_metadata,
                    strategy=strategy,
                    num_variations=augment_ratio
                )
                
                generation_stats['total_generated'] += len(variations)
                
                # Quality filtering
                filtered_variations, details = self.quality_filter.filter_batch(
                    original_queries=[sample['query']],
                    generated_variations=[variations],
                    video_metadatas=[video_metadata]
                )
                
                filtered_variations = filtered_variations[0]  # Unpack
                generation_stats['total_filtered'] += len(filtered_variations)
                
                # Create augmented samples
                for var_idx, var_query in enumerate(filtered_variations):
                    aug_sample = {
                        'qid': f"aug_{sample['qid']}_{var_idx}",
                        'original_qid': sample['qid'],
                        'query': var_query,
                        'vid': sample['vid'],
                        'duration': sample['duration'],
                        'relevant_windows': sample.get('relevant_windows', []),
                        'relevant_clip_ids': sample.get('relevant_clip_ids', []),
                        'saliency_scores': sample.get('saliency_scores', []),
                        'augmentation_strategy': strategy,
                        'augmentation_index': var_idx,
                        'quality_score': details[0][var_idx]['overall_score'] if details[0] else 0.0
                    }
                    augmented_data.append(aug_sample)
            
            except Exception as e:
                logger.warning(f"Failed to augment sample {sample.get('qid', idx)}: {e}")
                generation_stats['generation_failures'] += 1
                continue
            
            # Checkpoint saving
            if (idx + 1) % checkpoint_interval == 0:
                save_jsonl(augmented_data, checkpoint_path)
                logger.info(f"Checkpoint saved: {len(augmented_data)} augmented samples")
        
        # Final save
        save_jsonl(augmented_data, output_path)
        logger.info(f"Augmentation complete! Saved to {output_path}")
        
        # Compute diversity metrics
        logger.info("Computing diversity metrics...")
        diversity_metrics = self.diversity_calculator.compute_dataset_diversity(
            augmented_data
        )
        self.diversity_calculator.log_diversity_report(diversity_metrics)
        
        # Combine statistics
        stats = {
            **generation_stats,
            **diversity_metrics,
            'filter_pass_rate': generation_stats['total_filtered'] / max(generation_stats['total_generated'], 1),
            'avg_filtered_per_query': generation_stats['total_filtered'] / generation_stats['total_original']
        }
        
        # Save statistics
        stats_path = self.output_dir / f"stats_{output_filename.replace('.jsonl', '.json')}"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Statistics saved to {stats_path}")
        
        return stats
    
    def pilot_study(
        self,
        dataset_path: str,
        n_samples: int = 100,
        augment_ratio: int = 5,
        strategy: str = "semantic_paraphrase"
    ) -> Dict:
        """
        Run pilot study on small sample to validate pipeline.
        
        Args:
            dataset_path: Path to original dataset
            n_samples: Number of samples for pilot
            augment_ratio: Variations per query
            strategy: Augmentation strategy
        
        Returns:
            Pilot statistics
        """
        logger.info(f"Running pilot study with {n_samples} samples")
        
        stats = self.augment_dataset(
            dataset_path=dataset_path,
            output_filename=f"pilot_{os.path.basename(dataset_path)}",
            augment_ratio=augment_ratio,
            strategy=strategy,
            max_samples=n_samples,
            checkpoint_interval=50
        )
        
        # Log pilot summary
        logger.info("=" * 60)
        logger.info("PILOT STUDY SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Samples processed: {stats['total_original']}")
        logger.info(f"Variations generated: {stats['total_generated']}")
        logger.info(f"Variations after filtering: {stats['total_filtered']}")
        logger.info(f"Filter pass rate: {stats['filter_pass_rate']:.2%}")
        logger.info(f"Avg variations per query: {stats['avg_filtered_per_query']:.2f}")
        logger.info("=" * 60)
        
        return stats
