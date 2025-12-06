"""
Visual ChatGPT Model: CG-DETR enhanced with nested adapters.
Integrates adapter modules into CG-DETR's text processing pathway.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import logging

from .nested_adapter.adapter_layers import (
    NestedAdapterModule,
    DualStreamAdapter,
    AdapterConfig,
    count_adapter_parameters,
    freeze_base_model
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisualChatGPT_CGDETR(nn.Module):
    """
    CG-DETR enhanced with nested adapters for augmented query handling.
    
    Architecture:
    - Base: Pre-trained CG-DETR model (can be frozen)
    - Adapters: Inserted into text encoder pathway
    - Dual streams: Separate processing for augmented vs human queries
    - Fusion: Intelligent combination at inference
    
    Training Stages:
    1. Stage 1 (Adapter warm-up): Freeze CG-DETR, train adapters only
    2. Stage 2 (Joint fine-tuning): Unfreeze CG-DETR, train end-to-end
    """
    
    def __init__(
        self,
        cgdetr_model: nn.Module,
        adapter_config: AdapterConfig,
        freeze_cgdetr: bool = True
    ):
        """
        Args:
            cgdetr_model: Pre-trained CG-DETR model
            adapter_config: Adapter configuration
            freeze_cgdetr: Whether to freeze base CG-DETR weights
        """
        super().__init__()
        
        self.cgdetr = cgdetr_model
        self.adapter_config = adapter_config
        self.freeze_cgdetr = freeze_cgdetr
        
        # Freeze base model if requested
        if freeze_cgdetr:
            freeze_base_model(self.cgdetr, freeze=True)
            logger.info("Base CG-DETR model frozen")
        
        # Get hidden dimension from CG-DETR
        self.hidden_dim = cgdetr_model.hidden_dim
        
        # Initialize adapter modules
        self._init_adapters()
        
        # Log parameter counts
        self._log_parameters()
    
    def _init_adapters(self):
        """Initialize adapter modules at key insertion points."""
        config = self.adapter_config
        
        # Text input adapter (after text projection)
        self.text_input_adapter = NestedAdapterModule(
            hidden_dim=self.hidden_dim,
            adapter_dim=config.adapter_dim,
            num_token_adapters=config.num_token_adapters,
            dropout=config.dropout,
            fusion_type=config.fusion_type
        )
        
        # Adapters for text encoder layers (if CG-DETR has them)
        if hasattr(self.cgdetr, 'txtproj_encoder'):
            num_layers = len(self.cgdetr.txtproj_encoder.layers)
            self.text_encoder_adapters = nn.ModuleList([
                DualStreamAdapter(
                    hidden_dim=self.hidden_dim,
                    adapter_dim=config.adapter_dim,
                    dropout=config.dropout,
                    fusion_type=config.fusion_type
                )
                for _ in range(num_layers)
            ])
            logger.info(f"Initialized {num_layers} text encoder adapters")
        else:
            self.text_encoder_adapters = None
        
        # Query source classifier (optional, for automatic detection)
        self.query_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 2),  # Binary: augmented vs human
        )
    
    def _log_parameters(self):
        """Log parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        adapter_params = count_adapter_parameters(self)
        
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Adapter parameters: {adapter_params:,}")
        logger.info(f"Adapter ratio: {adapter_params/total_params:.2%}")
    
    def detect_query_source(
        self,
        src_txt: torch.Tensor,
        src_txt_mask: torch.Tensor
    ) -> str:
        """
        Classify query as augmented or human-annotated.
        
        Args:
            src_txt: Text features (batch, L_txt, D_txt)
            src_txt_mask: Text mask
        
        Returns:
            Query source: 'augmented' or 'human'
        """
        # Pool text features
        pooled = (src_txt * src_txt_mask.unsqueeze(-1)).sum(1) / \
                 src_txt_mask.sum(1, keepdim=True)
        
        # Classify
        with torch.no_grad():
            logits = self.query_classifier(pooled)
            predictions = torch.argmax(logits, dim=-1)
        
        # Map to source string
        sources = ["augmented", "human"]
        batch_sources = [sources[p.item()] for p in predictions]
        
        # Return most common (for batch consistency)
        from collections import Counter
        most_common = Counter(batch_sources).most_common(1)[0][0]
        
        return most_common
    
    def forward(
        self,
        src_txt: torch.Tensor,
        src_txt_mask: torch.Tensor,
        src_vid: torch.Tensor,
        src_vid_mask: torch.Tensor,
        query_source: Optional[str] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with adapter-enhanced processing.
        
        Args:
            src_txt: Text features (batch, L_txt, D_txt)
            src_txt_mask: Text attention mask
            src_vid: Video features (batch, L_vid, D_vid)
            src_vid_mask: Video attention mask
            query_source: 'augmented', 'human', or None (auto-detect)
            **kwargs: Additional arguments for CG-DETR
        
        Returns:
            Model outputs (predictions, losses, etc.)
        """
        # Auto-detect query source if not provided
        if query_source is None:
            query_source = self.detect_query_source(src_txt, src_txt_mask)
        
        # Apply text input adapters
        adapted_txt = self.text_input_adapter(
            text_features=src_txt,
            query_source=query_source,
            video_features=src_vid
        )
        
        # Process through CG-DETR with adapter augmentation
        outputs = self._cgdetr_forward_with_adapters(
            adapted_txt,
            src_txt_mask,
            src_vid,
            src_vid_mask,
            query_source,
            **kwargs
        )
        
        return outputs
    
    def _cgdetr_forward_with_adapters(
        self,
        src_txt: torch.Tensor,
        src_txt_mask: torch.Tensor,
        src_vid: torch.Tensor,
        src_vid_mask: torch.Tensor,
        query_source: str,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Modified CG-DETR forward pass with adapter insertion.
        
        This is a simplified version - in practice, we'd hook into
        CG-DETR's actual forward pass at specific points.
        """
        # For now, use CG-DETR's forward directly
        # In full implementation, we'd modify the text encoder path
        
        # Call base CG-DETR forward
        outputs = self.cgdetr(
            src_txt=src_txt,
            src_txt_mask=src_txt_mask,
            src_vid=src_vid,
            src_vid_mask=src_vid_mask,
            **kwargs
        )
        
        return outputs
    
    def set_training_stage(self, stage: int):
        """
        Set training stage and adjust parameter freezing.
        
        Args:
            stage: 1 (adapter warm-up) or 2 (joint fine-tuning)
        """
        if stage == 1:
            # Stage 1: Freeze CG-DETR, train adapters only
            freeze_base_model(self.cgdetr, freeze=True)
            for param in self.parameters():
                if 'adapter' in param or 'classifier' in param:
                    param.requires_grad = True
            logger.info("Training stage 1: Adapter warm-up")
        
        elif stage == 2:
            # Stage 2: Unfreeze CG-DETR for joint fine-tuning
            freeze_base_model(self.cgdetr, freeze=False)
            for param in self.parameters():
                param.requires_grad = True
            logger.info("Training stage 2: Joint fine-tuning")
        
        else:
            raise ValueError(f"Unknown training stage: {stage}")
        
        # Log trainable parameters
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters: {trainable:,}")
    
    def save_adapters(self, path: str):
        """Save only adapter weights (for efficient storage)."""
        adapter_state = {}
        for name, param in self.named_parameters():
            if 'adapter' in name.lower() or 'classifier' in name.lower():
                adapter_state[name] = param.data
        
        torch.save({
            'adapter_state_dict': adapter_state,
            'adapter_config': self.adapter_config.to_dict()
        }, path)
        
        logger.info(f"Saved adapters to {path}")
    
    def load_adapters(self, path: str):
        """Load adapter weights."""
        checkpoint = torch.load(path, map_location='cpu')
        
        # Load adapter parameters
        for name, param in self.named_parameters():
            if name in checkpoint['adapter_state_dict']:
                param.data = checkpoint['adapter_state_dict'][name]
        
        logger.info(f"Loaded adapters from {path}")


def build_visual_chatgpt_model(
    cgdetr_checkpoint: str,
    adapter_config: Optional[AdapterConfig] = None,
    freeze_cgdetr: bool = True,
    device: str = 'cuda'
) -> VisualChatGPT_CGDETR:
    """
    Build Visual ChatGPT model from CG-DETR checkpoint.
    
    Args:
        cgdetr_checkpoint: Path to CG-DETR checkpoint
        adapter_config: Adapter configuration (uses default if None)
        freeze_cgdetr: Whether to freeze base model
        device: Device to load model on
    
    Returns:
        VisualChatGPT_CGDETR model
    """
    # Load base CG-DETR model
    from cg_detr.model import build_model
    from cg_detr.config import BaseOptions
    
    # Parse CG-DETR config from checkpoint
    # (In practice, load from checkpoint metadata)
    opt = BaseOptions().parse(['--dset_name', 'hl'])
    cgdetr_model, criterion = build_model(opt)
    
    # Load checkpoint
    if cgdetr_checkpoint:
        logger.info(f"Loading CG-DETR from {cgdetr_checkpoint}")
        checkpoint = torch.load(cgdetr_checkpoint, map_location='cpu')
        cgdetr_model.load_state_dict(checkpoint['model'], strict=False)
    
    # Default adapter config if not provided
    if adapter_config is None:
        adapter_config = AdapterConfig(
            hidden_dim=opt.hidden_dim,
            adapter_dim=opt.hidden_dim // 4,
            num_token_adapters=2,
            dropout=0.1,
            fusion_type='gate',
            freeze_base=freeze_cgdetr
        )
    
    # Build Visual ChatGPT model
    model = VisualChatGPT_CGDETR(
        cgdetr_model=cgdetr_model,
        adapter_config=adapter_config,
        freeze_cgdetr=freeze_cgdetr
    )
    
    model = model.to(device)
    
    logger.info("Visual ChatGPT model built successfully")
    
    return model
