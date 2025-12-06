"""
Adapter layers for parameter-efficient fine-tuning.
Based on Houlsby et al. (2019) adapter architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AdapterLayer(nn.Module):
    """
    Lightweight adapter module for parameter-efficient fine-tuning.
    
    Architecture:
        Input → LayerNorm → Down-project → GELU → Up-project → Dropout → Residual
    
    Args:
        hidden_dim: Hidden dimension of the model
        adapter_dim: Bottleneck dimension (typically hidden_dim // 4)
        dropout: Dropout probability
        init_scale: Initialization scale for adapter weights
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        adapter_dim: int = 64,
        dropout: float = 0.1,
        init_scale: float = 1e-3
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.adapter_dim = adapter_dim
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Down-projection
        self.down_project = nn.Linear(hidden_dim, adapter_dim)
        
        # Activation
        self.activation = nn.GELU()
        
        # Up-projection
        self.up_project = nn.Linear(adapter_dim, hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize with small weights for stability
        self._init_weights(init_scale)
    
    def _init_weights(self, init_scale: float):
        """Initialize adapter weights with small values."""
        nn.init.normal_(self.down_project.weight, std=init_scale)
        nn.init.zeros_(self.down_project.bias)
        nn.init.normal_(self.up_project.weight, std=init_scale)
        nn.init.zeros_(self.up_project.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.
        
        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)
        
        Returns:
            Output tensor of same shape with adapter transformation
        """
        residual = x
        
        # Normalize
        x = self.layer_norm(x)
        
        # Adapter transformation
        x = self.down_project(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_project(x)
        x = self.dropout(x)
        
        # Residual connection
        return residual + x


class DualStreamAdapter(nn.Module):
    """
    Dual-stream adapter for handling augmented vs human-annotated queries.
    
    Maintains two separate adapter paths:
    - augmented_adapter: For LLM-generated query variations
    - human_adapter: For original human annotations
    
    At inference, can fuse both streams or route to specific stream.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        adapter_dim: int = 64,
        dropout: float = 0.1,
        fusion_type: str = "gate"
    ):
        """
        Args:
            hidden_dim: Model hidden dimension
            adapter_dim: Adapter bottleneck dimension
            dropout: Dropout rate
            fusion_type: Fusion strategy ('gate', 'average', 'concat')
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.fusion_type = fusion_type
        
        # Separate adapters for each stream
        self.augmented_adapter = AdapterLayer(hidden_dim, adapter_dim, dropout)
        self.human_adapter = AdapterLayer(hidden_dim, adapter_dim, dropout)
        
        # Fusion mechanism
        if fusion_type == "gate":
            # Gated fusion: learn adaptive weighting
            self.fusion_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        elif fusion_type == "concat":
            # Concatenation fusion
            self.fusion_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        # 'average' fusion needs no parameters
    
    def forward(
        self,
        x: torch.Tensor,
        query_source: Optional[str] = None
    ) -> torch.Tensor:
        """
        Forward pass with stream-aware routing.
        
        Args:
            x: Input tensor (batch, seq_len, hidden_dim)
            query_source: 'augmented', 'human', or None (fusion mode)
        
        Returns:
            Adapted features
        """
        if query_source == "augmented":
            # Route through augmented adapter only
            return self.augmented_adapter(x)
        
        elif query_source == "human":
            # Route through human adapter only
            return self.human_adapter(x)
        
        else:
            # Fusion mode: combine both streams
            aug_out = self.augmented_adapter(x)
            human_out = self.human_adapter(x)
            
            if self.fusion_type == "average":
                # Simple averaging
                return (aug_out + human_out) / 2
            
            elif self.fusion_type == "gate":
                # Gated fusion
                concat = torch.cat([aug_out, human_out], dim=-1)
                gate = self.fusion_gate(concat)  # (batch, seq_len, 1)
                return gate * aug_out + (1 - gate) * human_out
            
            elif self.fusion_type == "concat":
                # Concatenate and project
                concat = torch.cat([aug_out, human_out], dim=-1)
                return self.fusion_proj(concat)
            
            else:
                raise ValueError(f"Unknown fusion type: {self.fusion_type}")


class NestedAdapterModule(nn.Module):
    """
    Hierarchical adapter structure with multiple insertion points.
    
    Levels:
    1. Token-level adapters (fine-grained)
    2. Sequence-level adapter (holistic)
    3. Cross-modal adapter (video-text alignment)
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        adapter_dim: int = 64,
        num_token_adapters: int = 2,
        dropout: float = 0.1,
        fusion_type: str = "gate"
    ):
        """
        Args:
            hidden_dim: Model hidden dimension
            adapter_dim: Adapter bottleneck dimension
            num_token_adapters: Number of token-level adapter layers
            dropout: Dropout rate
            fusion_type: Fusion strategy
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_token_adapters = num_token_adapters
        
        # Token-level adapters (applied sequentially)
        self.token_adapters = nn.ModuleList([
            DualStreamAdapter(hidden_dim, adapter_dim, dropout, fusion_type)
            for _ in range(num_token_adapters)
        ])
        
        # Sequence-level adapter (larger capacity)
        self.sequence_adapter = AdapterLayer(
            hidden_dim,
            adapter_dim * 2,  # Double capacity for sequence-level
            dropout
        )
        
        # Cross-modal adapter (for video-text interaction)
        self.crossmodal_adapter = AdapterLayer(
            hidden_dim,
            adapter_dim,
            dropout
        )
    
    def forward(
        self,
        text_features: torch.Tensor,
        query_source: Optional[str] = None,
        video_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Multi-level adaptation pipeline.
        
        Args:
            text_features: Text features (batch, seq_len, hidden_dim)
            query_source: 'augmented', 'human', or None
            video_features: Optional video features for cross-modal adaptation
        
        Returns:
            Adapted text features
        """
        # 1. Token-level adaptation
        adapted_text = text_features
        for adapter in self.token_adapters:
            adapted_text = adapter(adapted_text, query_source)
        
        # 2. Sequence-level adaptation
        adapted_text = self.sequence_adapter(adapted_text)
        
        # 3. Cross-modal adaptation (if video features provided)
        if video_features is not None:
            # Simple approach: concatenate and adapt
            # In practice, use attention for cross-modal interaction
            adapted_text = self.crossmodal_adapter(adapted_text)
        
        return adapted_text


class AdapterConfig:
    """Configuration for adapter modules."""
    
    def __init__(
        self,
        hidden_dim: int = 256,
        adapter_dim: int = 64,
        num_token_adapters: int = 2,
        dropout: float = 0.1,
        fusion_type: str = "gate",
        freeze_base: bool = True,
        adapter_init_scale: float = 1e-3
    ):
        self.hidden_dim = hidden_dim
        self.adapter_dim = adapter_dim
        self.num_token_adapters = num_token_adapters
        self.dropout = dropout
        self.fusion_type = fusion_type
        self.freeze_base = freeze_base
        self.adapter_init_scale = adapter_init_scale
    
    def to_dict(self):
        """Convert config to dictionary."""
        return {
            'hidden_dim': self.hidden_dim,
            'adapter_dim': self.adapter_dim,
            'num_token_adapters': self.num_token_adapters,
            'dropout': self.dropout,
            'fusion_type': self.fusion_type,
            'freeze_base': self.freeze_base,
            'adapter_init_scale': self.adapter_init_scale
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary."""
        return cls(**config_dict)


def count_adapter_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters in adapter modules.
    
    Args:
        model: Model containing adapters
    
    Returns:
        Number of trainable adapter parameters
    """
    adapter_params = 0
    for name, param in model.named_parameters():
        if 'adapter' in name.lower() and param.requires_grad:
            adapter_params += param.numel()
    return adapter_params


def freeze_base_model(model: nn.Module, freeze: bool = True):
    """
    Freeze/unfreeze base model parameters (non-adapter).
    
    Args:
        model: Model to freeze
        freeze: Whether to freeze or unfreeze
    """
    for name, param in model.named_parameters():
        if 'adapter' not in name.lower():
            param.requires_grad = not freeze
