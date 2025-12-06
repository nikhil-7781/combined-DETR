"""
Query generator using Llama-3.1-8B-Instruct for semantic augmentation.
Handles model loading, prompt formatting, and batch generation.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import HfFolder
import json
import logging
from typing import List, Dict, Optional
from tqdm import tqdm
import re

from .prompt_templates import PromptTemplateManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LlamaQueryGenerator:
    """
    Llama-3.1-8B-Instruct interface for semantic query augmentation.
    Supports 4-bit quantization for memory efficiency.
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "cuda",
        load_in_4bit: bool = True,
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ):
        """
        Initialize Llama model for query generation.
        
        Args:
            model_name: HuggingFace model identifier
            device: cuda or cpu
            load_in_4bit: Use 4-bit quantization for memory efficiency
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more diverse)
            top_p: Nucleus sampling threshold
        """
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        logger.info(f"Loading {model_name}...")
        
        # Get HuggingFace token
        hf_token = HfFolder.get_token()
        if hf_token is None:
            logger.warning("No HuggingFace token found. Model access may fail for gated models.")
        
        # Load tokenizer (use_auth_token for transformers < 4.34)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_auth_token=hf_token  # Explicitly pass the token
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        # Load model without quantization (use_auth_token for transformers < 4.34)
        logger.info(f"Loading model in full precision (FP16 on GPU, FP32 on CPU)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
            use_auth_token=hf_token  # Explicitly pass the token
        )
        
        if device == "cuda" and not hasattr(self.model, 'hf_device_map'):
            self.model = self.model.to(device)
        
        self.model.eval()
        
        logger.info(f"Model loaded successfully on {device}")
        logger.info(f"Model memory footprint: {self.model.get_memory_footprint() / 1e9:.2f} GB")
    
    def generate_variations(
        self,
        original_query: str,
        video_metadata: Dict,
        strategy: str = "semantic_paraphrase",
        num_variations: int = 5,
    ) -> List[str]:
        """
        Generate semantic variations of a query.
        
        Args:
            original_query: Original human-annotated query
            video_metadata: Dict with 'duration', 'start_time', 'end_time'
            strategy: Augmentation strategy from PromptTemplateManager
            num_variations: Number of variations to generate
        
        Returns:
            List of generated query variations
        """
        # Get prompt
        system_prompt, user_prompt = PromptTemplateManager.get_prompt(
            strategy=strategy,
            original_query=original_query,
            num_variations=num_variations,
            duration=video_metadata.get('duration', 150.0),
            start_time=video_metadata.get('start_time', 0.0),
            end_time=video_metadata.get('end_time', 10.0)
        )
        
        # Format for Llama-3.1-Instruct
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # Parse JSON output
        variations = self._parse_variations(generated_text, num_variations)
        
        return variations
    
    def _parse_variations(self, generated_text: str, expected_count: int) -> List[str]:
        """
        Parse generated text to extract query variations.
        Handles JSON parsing and fallback strategies.
        
        Args:
            generated_text: Raw LLM output
            expected_count: Expected number of variations
        
        Returns:
            List of parsed variations
        """
        variations = []
        
        # Try JSON parsing first
        try:
            # Find JSON block
            json_match = re.search(r'\{[\s\S]*"variations"[\s\S]*\}', generated_text)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                variations = data.get("variations", [])
        except json.JSONDecodeError:
            logger.warning("JSON parsing failed, trying fallback extraction")
        
        # Fallback: extract quoted strings
        if not variations:
            # Look for patterns like "variation text" or numbered lists
            quoted_variations = re.findall(r'"([^"]{10,})"', generated_text)
            if quoted_variations:
                variations = quoted_variations[:expected_count]
        
        # Additional fallback: numbered list extraction
        if not variations:
            numbered_variations = re.findall(r'\d+\.\s*(.+?)(?:\n|$)', generated_text)
            if numbered_variations:
                variations = [v.strip().strip('"') for v in numbered_variations[:expected_count]]
        
        # Validate and clean
        variations = [v.strip() for v in variations if v and len(v) > 5]
        
        if len(variations) < expected_count:
            logger.warning(f"Only extracted {len(variations)}/{expected_count} variations")
        
        return variations[:expected_count]
    
    def generate_batch(
        self,
        queries: List[str],
        video_metadatas: List[Dict],
        strategy: str = "semantic_paraphrase",
        num_variations: int = 5,
    ) -> List[List[str]]:
        """
        Generate variations for a batch of queries.
        
        Args:
            queries: List of original queries
            video_metadatas: List of metadata dicts
            strategy: Augmentation strategy
            num_variations: Variations per query
        
        Returns:
            List of variation lists (one per query)
        """
        all_variations = []
        
        for query, metadata in tqdm(
            zip(queries, video_metadatas),
            total=len(queries),
            desc="Generating variations"
        ):
            variations = self.generate_variations(
                query,
                metadata,
                strategy=strategy,
                num_variations=num_variations
            )
            all_variations.append(variations)
        
        return all_variations
    
    def cleanup(self):
        """Free GPU memory."""
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
