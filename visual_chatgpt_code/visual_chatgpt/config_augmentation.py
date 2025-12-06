"""
Configuration for semantic data augmentation.
"""

import argparse
import os


class AugmentationConfig:
    """Configuration parser for augmentation pipeline."""
    
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Semantic Data Augmentation with Llama-3.1-8B-Instruct"
        )
        self.initialize()
    
    def initialize(self):
        # Dataset paths
        self.parser.add_argument(
            "--dataset_path",
            type=str,
            required=True,
            help="Path to original dataset JSONL file"
        )
        self.parser.add_argument(
            "--output_dir",
            type=str,
            default="data/augmented",
            help="Output directory for augmented data"
        )
        self.parser.add_argument(
            "--output_filename",
            type=str,
            default=None,
            help="Output filename (default: augmented_{input_filename})"
        )
        
        # LLM configuration
        self.parser.add_argument(
            "--model_name",
            type=str,
            default="meta-llama/Llama-3.1-8B-Instruct",
            help="HuggingFace model name"
        )
        self.parser.add_argument(
            "--load_in_4bit",
            action="store_true",
            default=True,
            help="Use 4-bit quantization"
        )
        self.parser.add_argument(
            "--device",
            type=str,
            default="cuda",
            choices=["cuda", "cpu"],
            help="Device for model inference"
        )
        
        # Generation parameters
        self.parser.add_argument(
            "--augment_ratio",
            type=int,
            default=5,
            help="Number of variations per original query"
        )
        self.parser.add_argument(
            "--strategy",
            type=str,
            default="semantic_paraphrase",
            choices=["semantic_paraphrase", "synonym_replacement", 
                    "structure_variation", "mixed_strategy"],
            help="Augmentation strategy"
        )
        self.parser.add_argument(
            "--temperature",
            type=float,
            default=0.8,
            help="Sampling temperature for LLM"
        )
        self.parser.add_argument(
            "--top_p",
            type=float,
            default=0.9,
            help="Nucleus sampling threshold"
        )
        self.parser.add_argument(
            "--max_new_tokens",
            type=int,
            default=512,
            help="Maximum tokens to generate"
        )
        
        # Quality filtering
        self.parser.add_argument(
            "--similarity_threshold",
            type=float,
            default=0.75,
            help="Minimum semantic similarity threshold"
        )
        self.parser.add_argument(
            "--keyword_overlap_min",
            type=float,
            default=0.5,
            help="Minimum keyword overlap ratio"
        )
        
        # Processing options
        self.parser.add_argument(
            "--max_samples",
            type=int,
            default=None,
            help="Maximum samples to process (for testing)"
        )
        self.parser.add_argument(
            "--checkpoint_interval",
            type=int,
            default=100,
            help="Save checkpoint every N samples"
        )
        self.parser.add_argument(
            "--pilot",
            action="store_true",
            help="Run pilot study (100 samples)"
        )
        self.parser.add_argument(
            "--pilot_samples",
            type=int,
            default=100,
            help="Number of samples for pilot study"
        )
        
        # Logging
        self.parser.add_argument(
            "--log_level",
            type=str,
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            help="Logging level"
        )
    
    def parse(self, args=None):
        """Parse arguments and validate."""
        opt = self.parser.parse_args(args)
        
        # Set output filename if not provided
        if opt.output_filename is None:
            input_basename = os.path.basename(opt.dataset_path)
            opt.output_filename = f"augmented_{input_basename}"
        
        # Validate paths
        if not os.path.exists(opt.dataset_path):
            raise FileNotFoundError(f"Dataset not found: {opt.dataset_path}")
        
        return opt


if __name__ == "__main__":
    config = AugmentationConfig()
    opt = config.parse()
    print(opt)
