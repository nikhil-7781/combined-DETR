"""
Diversity metrics for evaluating augmentation quality.
Measures lexical, syntactic, and semantic diversity.
"""

import torch
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict
import spacy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiversityMetrics:
    """
    Compute diversity metrics for augmented queries.
    
    Metrics:
    - Lexical diversity (Type-Token Ratio, unique n-grams)
    - Syntactic diversity (POS tag patterns)
    - Semantic coverage (embedding space spread)
    - Self-BLEU (inverse diversity measure)
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize diversity metrics calculator.
        
        Args:
            device: cuda or cpu
        """
        self.device = device
        
        # Load spaCy for linguistic analysis
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("Downloading spaCy model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        # Load CLIP for semantic diversity
        self._load_clip_encoder()
    
    def _load_clip_encoder(self):
        """Load CLIP text encoder."""
        try:
            from transformers import CLIPTextModel, CLIPTokenizer
            
            model_name = "openai/clip-vit-base-patch32"
            self.clip_tokenizer = CLIPTokenizer.from_pretrained(model_name)
            self.clip_model = CLIPTextModel.from_pretrained(model_name).to(self.device)
            self.clip_model.eval()
        except Exception as e:
            logger.warning(f"CLIP loading failed: {e}")
            self.clip_model = None
    
    def compute_dataset_diversity(
        self,
        augmented_data: List[Dict]
    ) -> Dict[str, float]:
        """
        Compute diversity metrics across entire augmented dataset.
        
        Args:
            augmented_data: List of augmented query dicts with 'query' key
        
        Returns:
            Dictionary of diversity metrics
        """
        # Group by original query
        query_groups = defaultdict(list)
        for item in augmented_data:
            orig_qid = item.get('original_qid', item.get('qid'))
            query_groups[orig_qid].append(item['query'])
        
        # Compute metrics
        metrics = {
            'num_original_queries': len(query_groups),
            'total_augmented_queries': len(augmented_data),
            'avg_augmentations_per_query': len(augmented_data) / max(len(query_groups), 1),
            'lexical_diversity_ttr': self.lexical_diversity_ttr(query_groups),
            'lexical_diversity_ngrams': self.unique_ngram_ratio(query_groups),
            'syntactic_diversity': self.syntactic_diversity(query_groups),
            'semantic_coverage': self.semantic_coverage(query_groups),
        }
        
        return metrics
    
    def lexical_diversity_ttr(self, query_groups: Dict) -> float:
        """
        Compute Type-Token Ratio (TTR) across all augmented queries.
        Higher = more diverse vocabulary.
        
        Args:
            query_groups: Dict mapping original_qid to list of variations
        
        Returns:
            TTR score [0, 1]
        """
        all_tokens = []
        
        for variations in query_groups.values():
            for query in variations:
                doc = self.nlp(query.lower())
                tokens = [token.text for token in doc if not token.is_punct]
                all_tokens.extend(tokens)
        
        if not all_tokens:
            return 0.0
        
        unique_tokens = set(all_tokens)
        ttr = len(unique_tokens) / len(all_tokens)
        
        return ttr
    
    def unique_ngram_ratio(self, query_groups: Dict, n: int = 2) -> float:
        """
        Compute ratio of unique n-grams.
        
        Args:
            query_groups: Query variations grouped by original
            n: N-gram size
        
        Returns:
            Unique n-gram ratio [0, 1]
        """
        all_ngrams = []
        
        for variations in query_groups.values():
            for query in variations:
                tokens = query.lower().split()
                ngrams = [
                    tuple(tokens[i:i+n]) 
                    for i in range(len(tokens) - n + 1)
                ]
                all_ngrams.extend(ngrams)
        
        if not all_ngrams:
            return 0.0
        
        unique_ngrams = set(all_ngrams)
        ratio = len(unique_ngrams) / len(all_ngrams)
        
        return ratio
    
    def syntactic_diversity(self, query_groups: Dict) -> float:
        """
        Measure variety in grammatical structures (POS patterns).
        
        Args:
            query_groups: Query variations
        
        Returns:
            Syntactic diversity score [0, 1]
        """
        all_pos_patterns = []
        
        for variations in query_groups.values():
            for query in variations:
                doc = self.nlp(query)
                pos_pattern = tuple([token.pos_ for token in doc])
                all_pos_patterns.append(pos_pattern)
        
        if not all_pos_patterns:
            return 0.0
        
        unique_patterns = set(all_pos_patterns)
        diversity = len(unique_patterns) / len(all_pos_patterns)
        
        return diversity
    
    def semantic_coverage(self, query_groups: Dict) -> float:
        """
        Measure spread in CLIP embedding space.
        Higher = more diverse semantic representations.
        
        Args:
            query_groups: Query variations
        
        Returns:
            Average pairwise distance in embedding space
        """
        if self.clip_model is None:
            return 0.0
        
        # Collect all queries
        all_queries = []
        for variations in query_groups.values():
            all_queries.extend(variations)
        
        if len(all_queries) < 2:
            return 0.0
        
        # Encode in batches
        batch_size = 32
        embeddings = []
        
        for i in range(0, len(all_queries), batch_size):
            batch = all_queries[i:i+batch_size]
            
            inputs = self.clip_tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                batch_emb = outputs.pooler_output
                embeddings.append(batch_emb.cpu())
        
        # Concatenate
        all_embeddings = torch.cat(embeddings, dim=0)
        
        # Normalize
        all_embeddings = torch.nn.functional.normalize(all_embeddings, p=2, dim=1)
        
        # Compute average pairwise distance
        # Use sampling for efficiency if too many queries
        if len(all_embeddings) > 500:
            # Sample 500 random pairs
            indices = torch.randperm(len(all_embeddings))[:500]
            sample_emb = all_embeddings[indices]
            distances = torch.cdist(sample_emb, sample_emb)
        else:
            distances = torch.cdist(all_embeddings, all_embeddings)
        
        # Average distance (excluding diagonal)
        mask = torch.eye(len(distances), dtype=torch.bool)
        avg_distance = distances[~mask].mean().item()
        
        return avg_distance
    
    def compute_within_group_diversity(
        self,
        variations: List[str]
    ) -> Dict[str, float]:
        """
        Compute diversity within a single query's variations.
        
        Args:
            variations: List of variations for one original query
        
        Returns:
            Diversity metrics for this group
        """
        if len(variations) < 2:
            return {'diversity': 0.0}
        
        # Lexical diversity
        all_tokens = []
        for query in variations:
            tokens = query.lower().split()
            all_tokens.extend(tokens)
        
        ttr = len(set(all_tokens)) / len(all_tokens) if all_tokens else 0.0
        
        # Semantic diversity
        if self.clip_model is not None:
            inputs = self.clip_tokenizer(
                variations,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                embeddings = outputs.pooler_output
            
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            distances = torch.cdist(embeddings, embeddings)
            
            mask = torch.eye(len(distances), dtype=torch.bool)
            avg_distance = distances[~mask].mean().item()
        else:
            avg_distance = 0.0
        
        return {
            'lexical_diversity': ttr,
            'semantic_diversity': avg_distance,
            'num_variations': len(variations)
        }
    
    def log_diversity_report(self, metrics: Dict):
        """
        Pretty-print diversity metrics.
        
        Args:
            metrics: Metrics dict from compute_dataset_diversity
        """
        logger.info("=" * 60)
        logger.info("DIVERSITY METRICS REPORT")
        logger.info("=" * 60)
        
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"{key:.<40} {value:.4f}")
            else:
                logger.info(f"{key:.<40} {value}")
        
        logger.info("=" * 60)
