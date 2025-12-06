"""
Quality filtering for LLM-generated query variations.
Ensures semantic fidelity and detects hallucinations.
"""

import torch
import torch.nn.functional as F
import spacy
import numpy as np
from typing import Dict, Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticQualityFilter:
    """
    Multi-layered quality control for augmented queries.
    
    Validation checks:
    1. Semantic similarity (CLIP text embeddings)
    2. Keyword overlap (content word preservation)
    3. Action consistency (verb matching)
    4. Length reasonability
    5. Hallucination detection
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.75,
        keyword_overlap_min: float = 0.5,
        min_length: int = 5,
        max_length: int = 150,
        device: str = "cuda"
    ):
        """
        Initialize quality filter.
        
        Args:
            similarity_threshold: Minimum cosine similarity for semantic match
            keyword_overlap_min: Minimum keyword overlap ratio
            min_length: Minimum query length in characters
            max_length: Maximum query length in characters
            device: cuda or cpu
        """
        self.similarity_threshold = similarity_threshold
        self.keyword_overlap_min = keyword_overlap_min
        self.min_length = min_length
        self.max_length = max_length
        self.device = device
        
        # Load spaCy for linguistic analysis
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("Downloading spaCy model 'en_core_web_sm'...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        # Load CLIP text encoder for semantic similarity
        self._load_clip_encoder()
    
    def _load_clip_encoder(self):
        """Load CLIP text encoder for semantic similarity computation."""
        try:
            from transformers import CLIPTextModel, CLIPTokenizer
            
            model_name = "openai/clip-vit-base-patch32"
            self.clip_tokenizer = CLIPTokenizer.from_pretrained(model_name)
            self.clip_model = CLIPTextModel.from_pretrained(model_name).to(self.device)
            self.clip_model.eval()
            
            logger.info("CLIP text encoder loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load CLIP: {e}. Semantic similarity check disabled.")
            self.clip_model = None
            self.clip_tokenizer = None
    
    def validate_query(
        self,
        original: str,
        generated: str,
        video_metadata: Dict = None
    ) -> Tuple[bool, float, Dict]:
        """
        Comprehensive validation of generated query.
        
        Args:
            original: Original human-annotated query
            generated: LLM-generated variation
            video_metadata: Optional video context
        
        Returns:
            valid: Whether query passes all checks
            score: Overall quality score [0, 1]
            details: Breakdown of individual metrics
        """
        details = {}
        
        # 1. Length check
        length_ok = self.min_length <= len(generated) <= self.max_length
        details['length_ok'] = length_ok
        details['generated_length'] = len(generated)
        
        if not length_ok:
            return False, 0.0, details
        
        # 2. Semantic similarity check
        if self.clip_model is not None:
            sim_score = self.compute_semantic_similarity(original, generated)
            details['semantic_similarity'] = sim_score
            sem_ok = sim_score >= self.similarity_threshold
        else:
            sim_score = 1.0  # Skip if CLIP not available
            details['semantic_similarity'] = None
            sem_ok = True
        
        # 3. Keyword overlap check
        keyword_score = self.check_keyword_overlap(original, generated)
        details['keyword_overlap'] = keyword_score
        keyword_ok = keyword_score >= self.keyword_overlap_min
        
        # 4. Action consistency check
        action_match = self.verify_action_consistency(original, generated)
        details['action_match'] = action_match
        
        # 5. Hallucination detection
        no_hallucination = self.detect_hallucinations(original, generated)
        details['no_hallucination'] = no_hallucination
        
        # 6. Diversity check (not too similar to original)
        not_duplicate = generated.lower() != original.lower()
        details['not_duplicate'] = not_duplicate
        
        # Aggregate validation
        valid = (
            length_ok and
            sem_ok and
            keyword_ok and
            action_match and
            no_hallucination and
            not_duplicate
        )
        
        # Compute overall quality score
        score = (
            0.35 * sim_score +
            0.25 * keyword_score +
            0.15 * float(action_match) +
            0.15 * float(no_hallucination) +
            0.10 * float(not_duplicate)
        )
        
        details['overall_score'] = score
        
        return valid, score, details
    
    def compute_semantic_similarity(self, query1: str, query2: str) -> float:
        """
        Compute cosine similarity using CLIP text embeddings.
        
        Args:
            query1: First query
            query2: Second query
        
        Returns:
            Cosine similarity score [0, 1]
        """
        if self.clip_model is None:
            return 1.0
        
        try:
            # Tokenize
            inputs = self.clip_tokenizer(
                [query1, query2],
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Encode
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                embeddings = outputs.pooler_output  # [2, 512]
            
            # Normalize and compute similarity
            embeddings = F.normalize(embeddings, p=2, dim=1)
            similarity = F.cosine_similarity(
                embeddings[0:1], embeddings[1:2], dim=1
            ).item()
            
            return max(0.0, similarity)  # Clamp to [0, 1]
        
        except Exception as e:
            logger.warning(f"Similarity computation failed: {e}")
            return 0.0
    
    def check_keyword_overlap(self, original: str, generated: str) -> float:
        """
        Check overlap of content words (nouns, verbs, adjectives).
        
        Args:
            original: Original query
            generated: Generated query
        
        Returns:
            Overlap ratio [0, 1]
        """
        doc1 = self.nlp(original.lower())
        doc2 = self.nlp(generated.lower())
        
        # Extract content words
        content_pos = {"NOUN", "VERB", "ADJ", "PROPN"}
        keywords1 = {token.lemma_ for token in doc1 if token.pos_ in content_pos}
        keywords2 = {token.lemma_ for token in doc2 if token.pos_ in content_pos}
        
        if not keywords1:
            return 1.0  # Empty original, trivially passes
        
        # Compute overlap
        intersection = keywords1 & keywords2
        overlap = len(intersection) / len(keywords1)
        
        return overlap
    
    def verify_action_consistency(self, original: str, generated: str) -> bool:
        """
        Verify that main action verbs are preserved or synonymous.
        
        Args:
            original: Original query
            generated: Generated query
        
        Returns:
            True if actions are consistent
        """
        doc1 = self.nlp(original)
        doc2 = self.nlp(generated)
        
        # Extract main verbs
        verbs1 = [token for token in doc1 if token.pos_ == "VERB"]
        verbs2 = [token for token in doc2 if token.pos_ == "VERB"]
        
        if not verbs1:
            return True  # No verbs in original
        
        # Check if any verb matches or is similar
        for v1 in verbs1:
            for v2 in verbs2:
                # Exact lemma match
                if v1.lemma_ == v2.lemma_:
                    return True
                # Similarity check (using word vectors)
                if v1.has_vector and v2.has_vector:
                    similarity = v1.similarity(v2)
                    if similarity > 0.6:
                        return True
        
        # More lenient: if at least 1 verb exists in generated
        return len(verbs2) > 0
    
    def detect_hallucinations(self, original: str, generated: str) -> bool:
        """
        Detect if LLM added information not in original query.
        
        Args:
            original: Original query
            generated: Generated query
        
        Returns:
            True if no hallucinations detected
        """
        doc1 = self.nlp(original)
        doc2 = self.nlp(generated)
        
        # Extract named entities
        entities1 = {ent.text.lower() for ent in doc1.ents}
        entities2 = {ent.text.lower() for ent in doc2.ents}
        
        # Allow some generic entities
        generic_terms = {
            "person", "people", "someone", "individual",
            "thing", "object", "item", "place", "area"
        }
        
        # Check for new specific entities
        new_entities = entities2 - entities1 - generic_terms
        
        # Flag if too many new entities (likely hallucination)
        if len(new_entities) > 2:
            return False
        
        # Check for suspicious additions (numbers, colors, etc.)
        # Extract numbers
        numbers1 = {token.text for token in doc1 if token.like_num}
        numbers2 = {token.text for token in doc2 if token.like_num}
        new_numbers = numbers2 - numbers1
        
        if len(new_numbers) > 1:
            return False
        
        return True
    
    def filter_batch(
        self,
        original_queries: List[str],
        generated_variations: List[List[str]],
        video_metadatas: List[Dict] = None
    ) -> Tuple[List[List[str]], List[List[Dict]]]:
        """
        Filter a batch of generated variations.
        
        Args:
            original_queries: List of original queries
            generated_variations: List of variation lists
            video_metadatas: Optional video metadata list
        
        Returns:
            filtered_variations: List of filtered variation lists
            details_list: List of detail dicts for each variation
        """
        if video_metadatas is None:
            video_metadatas = [{}] * len(original_queries)
        
        filtered_variations = []
        all_details = []
        
        for orig_query, variations, metadata in zip(
            original_queries, generated_variations, video_metadatas
        ):
            filtered = []
            details_for_query = []
            
            for var in variations:
                valid, score, details = self.validate_query(
                    orig_query, var, metadata
                )
                
                if valid:
                    filtered.append(var)
                    details_for_query.append(details)
            
            filtered_variations.append(filtered)
            all_details.append(details_for_query)
        
        return filtered_variations, all_details
    
    def get_statistics(self, details_list: List[List[Dict]]) -> Dict:
        """
        Compute aggregate statistics from validation details.
        
        Args:
            details_list: List of detail lists from filter_batch
        
        Returns:
            Statistics dict
        """
        # Flatten details
        all_details = [d for sublist in details_list for d in sublist]
        
        if not all_details:
            return {}
        
        stats = {
            'total_validated': len(all_details),
            'avg_semantic_similarity': np.mean([
                d['semantic_similarity'] for d in all_details 
                if d['semantic_similarity'] is not None
            ]) if all_details else 0.0,
            'avg_keyword_overlap': np.mean([
                d['keyword_overlap'] for d in all_details
            ]),
            'action_match_rate': np.mean([
                float(d['action_match']) for d in all_details
            ]),
            'no_hallucination_rate': np.mean([
                float(d['no_hallucination']) for d in all_details
            ]),
            'avg_quality_score': np.mean([
                d['overall_score'] for d in all_details
            ])
        }
        
        return stats
