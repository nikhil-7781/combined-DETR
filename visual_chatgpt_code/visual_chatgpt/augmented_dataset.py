"""
Augmented dataset loader for Visual ChatGPT.
Extends CG-DETR's StartEndDataset to handle augmented queries.
"""

import os
import random
import json
import logging
from typing import List, Dict, Optional
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset

# Import base dataset from CG-DETR
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cg_detr.start_end_dataset import StartEndDataset, start_end_collate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]


class AugmentedStartEndDataset(StartEndDataset):
    """
    Extended dataset that includes LLM-augmented queries.
    
    Features:
    - Loads both original and augmented queries
    - Tracks query source (augmented vs human)
    - Smart sampling with configurable augmentation ratio
    - Compatible with CG-DETR's data format
    """
    
    def __init__(
        self,
        augmented_data_path: Optional[str] = None,
        augmentation_ratio: float = 0.5,
        use_augmented: bool = True,
        *args,
        **kwargs
    ):
        """
        Args:
            augmented_data_path: Path to augmented queries JSONL
            augmentation_ratio: Ratio of augmented queries in batch (0.0-1.0)
            use_augmented: Whether to use augmented queries
            *args, **kwargs: Arguments for base StartEndDataset
        """
        # Initialize base dataset
        super().__init__(*args, **kwargs)
        
        self.augmented_data_path = augmented_data_path
        self.augmentation_ratio = augmentation_ratio
        self.use_augmented = use_augmented and augmented_data_path is not None
        
        # Load augmented data if provided
        if self.use_augmented:
            self.augmented_data = self._load_augmented_data()
            self.qid_to_augmented = self._build_augmentation_map()
            logger.info(f"Loaded {len(self.augmented_data)} augmented queries")
            logger.info(f"Coverage: {len(self.qid_to_augmented)}/{len(self.data)} original queries have augmentations")
        else:
            self.augmented_data = []
            self.qid_to_augmented = {}
            logger.info("Augmented data not used")
    
    def _load_augmented_data(self) -> List[Dict]:
        """Load and filter augmented queries."""
        if not os.path.exists(self.augmented_data_path):
            logger.warning(f"Augmented data not found: {self.augmented_data_path}")
            return []
        
        data = load_jsonl(self.augmented_data_path)
        
        # Filter by quality threshold if available
        filtered = [
            d for d in data 
            if d.get('quality_score', 1.0) >= 0.8  # Keep high-quality only
        ]
        
        logger.info(
            f"Loaded {len(filtered)}/{len(data)} augmented queries "
            f"(quality filtered)"
        )
        
        return filtered
    
    def _build_augmentation_map(self) -> Dict[int, List[Dict]]:
        """Map each original query to its augmented versions."""
        mapping = defaultdict(list)
        
        for aug_data in self.augmented_data:
            original_qid = aug_data.get('original_qid', aug_data.get('qid'))
            mapping[original_qid].append(aug_data)
        
        return dict(mapping)
    
    def __getitem__(self, index):
        """
        Get item with augmentation awareness.
        
        Strategy:
        - With probability augmentation_ratio, return augmented query
        - Otherwise, return original human-annotated query
        - Track source for adapter routing
        """
        # Decide whether to use augmentation
        use_augmented = (
            self.use_augmented and
            self.load_labels and  # Only in training mode
            random.random() < self.augmentation_ratio and
            index in self.qid_to_augmented
        )
        
        if use_augmented:
            # Sample one augmented version
            aug_variants = self.qid_to_augmented[index]
            selected = random.choice(aug_variants)
            
            # Build sample from augmented data
            sample = self._build_sample_from_augmented(selected, index)
            sample['query_source'] = 'augmented'
            sample['original_qid'] = selected.get('original_qid', index)
        else:
            # Use original human annotation
            sample = super().__getitem__(index)
            sample['query_source'] = 'human'
            sample['original_qid'] = index
        
        return sample
    
    def _build_sample_from_augmented(self, aug_data: Dict, index: int) -> Dict:
        """
        Build a sample dict from augmented data.
        Follows same format as base dataset.
        """
        # Get base sample for video/label data
        base_sample = super().__getitem__(index)
        
        # Replace query with augmented version
        # Keep same video features and labels
        base_sample['query'] = aug_data['query']
        
        # Update query features if pre-extracted
        # (In practice, we re-encode augmented queries)
        # For now, keep base features - will be re-encoded in model
        
        return base_sample
    
    def get_augmentation_stats(self) -> Dict:
        """Return statistics about augmentation coverage."""
        if not self.use_augmented:
            return {'augmented_enabled': False}
        
        return {
            'augmented_enabled': True,
            'total_original': len(self.data),
            'total_augmented': len(self.augmented_data),
            'coverage': len(self.qid_to_augmented) / max(len(self.data), 1),
            'avg_augmentations_per_query': (
                sum(len(augs) for augs in self.qid_to_augmented.values()) /
                max(len(self.qid_to_augmented), 1)
            ),
            'augmentation_ratio': self.augmentation_ratio
        }


def augmented_collate_fn(batch):
    """
    Custom collate function that preserves query source metadata.
    
    Returns:
        model_inputs: Standard CG-DETR inputs
        targets: Labels
        metadata: Query source tracking
    """
    # Extract metadata
    query_sources = [item.pop('query_source', 'human') for item in batch]
    original_qids = [item.pop('original_qid', -1) for item in batch]
    
    # Use standard CG-DETR collate
    model_inputs, targets = start_end_collate(batch)
    
    # Add metadata
    metadata = {
        'query_sources': query_sources,
        'original_qids': original_qids
    }
    
    return model_inputs, targets, metadata


class MixedBatchSampler:
    """
    Smart batch sampler that ensures balanced augmented/human queries.
    
    Maintains exact augmentation_ratio per batch.
    """
    
    def __init__(
        self,
        dataset: AugmentedStartEndDataset,
        batch_size: int,
        augmentation_ratio: float = 0.5,
        shuffle: bool = True
    ):
        """
        Args:
            dataset: AugmentedStartEndDataset instance
            batch_size: Batch size
            augmentation_ratio: Target ratio of augmented queries
            shuffle: Whether to shuffle indices
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.augmentation_ratio = augmentation_ratio
        self.shuffle = shuffle
        
        # Separate indices for augmented-capable vs human-only
        self.aug_capable_indices = list(dataset.qid_to_augmented.keys())
        self.human_only_indices = [
            i for i in range(len(dataset))
            if i not in dataset.qid_to_augmented
        ]
        
        logger.info(f"Batch sampler: {len(self.aug_capable_indices)} augmented-capable, "
                   f"{len(self.human_only_indices)} human-only")
    
    def __iter__(self):
        """Generate batches with controlled augmentation ratio."""
        # Calculate samples per batch
        n_aug = int(self.batch_size * self.augmentation_ratio)
        n_human = self.batch_size - n_aug
        
        # Shuffle if needed
        aug_indices = self.aug_capable_indices.copy()
        human_indices = self.human_only_indices.copy()
        
        if self.shuffle:
            random.shuffle(aug_indices)
            random.shuffle(human_indices)
        
        # Generate batches
        aug_ptr = 0
        human_ptr = 0
        
        while aug_ptr < len(aug_indices) or human_ptr < len(human_indices):
            batch = []
            
            # Add augmented samples
            for _ in range(n_aug):
                if aug_ptr < len(aug_indices):
                    batch.append(aug_indices[aug_ptr])
                    aug_ptr += 1
            
            # Add human samples
            for _ in range(n_human):
                if human_ptr < len(human_indices):
                    batch.append(human_indices[human_ptr])
                    human_ptr += 1
                elif aug_ptr < len(aug_indices):
                    # Fill with augmented if human exhausted
                    batch.append(aug_indices[aug_ptr])
                    aug_ptr += 1
            
            if batch:
                if self.shuffle:
                    random.shuffle(batch)
                yield batch
    
    def __len__(self):
        """Number of batches."""
        total_samples = len(self.aug_capable_indices) + len(self.human_only_indices)
        return (total_samples + self.batch_size - 1) // self.batch_size
