"""
Training script for Visual ChatGPT model.
Implements 2-stage training with adapter warmup and joint fine-tuning.
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from cg_detr.config import BaseOptions
from cg_detr.model import build_model
from visual_chatgpt.visual_chatgpt_model import (
    VisualChatGPT_CGDETR,
    build_visual_chatgpt_model
)
from visual_chatgpt.nested_adapter.adapter_layers import AdapterConfig
from visual_chatgpt.augmented_dataset import (
    AugmentedStartEndDataset,
    augmented_collate_fn,
    MixedBatchSampler
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VisualChatGPTTrainer:
    """Trainer for Visual ChatGPT model."""
    
    def __init__(
        self,
        model: VisualChatGPT_CGDETR,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        config: dict,
        device: str = 'cuda'
    ):
        self.model = model
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_metric = 0.0
        
        # Output directory
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpointing
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Logging
        self.log_file = self.output_dir / 'training.log'
        self.metrics_file = self.output_dir / 'metrics.jsonl'
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                src_txt=batch['src_txt'],
                src_txt_mask=batch['src_txt_mask'],
                src_vid=batch['src_vid'],
                src_vid_mask=batch['src_vid_mask'],
                query_source=batch.get('query_source', None)
            )
            
            # Compute loss
            loss_dict = self.criterion(outputs, batch)
            
            # Total loss (weighted sum)
            loss = sum(loss_dict[k] * self.config['loss_weights'].get(k, 1.0)
                      for k in loss_dict.keys())
            
            # Adapter regularization
            if self.config.get('adapter_l2_weight', 0.0) > 0:
                adapter_l2 = self._compute_adapter_l2()
                loss = loss + self.config['adapter_l2_weight'] * adapter_l2
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.get('max_grad_norm', 0.0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['max_grad_norm']
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # Log metrics periodically
            if batch_idx % self.config.get('log_interval', 10) == 0:
                self._log_metrics({
                    'epoch': epoch,
                    'step': self.global_step,
                    'loss': loss.item(),
                    'lr': self.optimizer.param_groups[0]['lr'],
                    **{k: v.item() for k, v in loss_dict.items()}
                })
        
        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    src_txt=batch['src_txt'],
                    src_txt_mask=batch['src_txt_mask'],
                    src_vid=batch['src_vid'],
                    src_vid_mask=batch['src_vid_mask'],
                    query_source=batch.get('query_source', None)
                )
                
                # Compute loss
                loss_dict = self.criterion(outputs, batch)
                loss = sum(loss_dict.values())
                
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        
        # Log validation metrics
        self._log_metrics({
            'epoch': epoch,
            'step': self.global_step,
            'val_loss': avg_loss
        })
        
        return {'val_loss': avg_loss}
    
    def train(self, num_epochs: int, training_stage: int = 1):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            training_stage: 1 (adapter warmup) or 2 (joint fine-tuning)
        """
        logger.info(f"Starting training stage {training_stage} for {num_epochs} epochs")
        
        # Set training stage
        self.model.set_training_stage(training_stage)
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            
            # Train
            train_metrics = self.train_epoch(self.current_epoch)
            logger.info(f"Epoch {self.current_epoch}: {train_metrics}")
            
            # Validate
            val_metrics = self.validate(self.current_epoch)
            logger.info(f"Validation: {val_metrics}")
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Save checkpoint
            if self.current_epoch % self.config.get('save_interval', 5) == 0:
                self.save_checkpoint(f'stage{training_stage}_epoch{self.current_epoch}')
            
            # Save best model
            val_loss = val_metrics['val_loss']
            if val_loss < self.best_val_metric or self.best_val_metric == 0:
                self.best_val_metric = val_loss
                self.save_checkpoint(f'stage{training_stage}_best')
                logger.info(f"New best model: val_loss = {val_loss:.4f}")
        
        logger.info(f"Training stage {training_stage} complete")
    
    def _compute_adapter_l2(self) -> torch.Tensor:
        """Compute L2 regularization for adapter parameters."""
        adapter_params = [
            p for name, p in self.model.named_parameters()
            if 'adapter' in name.lower() and p.requires_grad
        ]
        
        if len(adapter_params) == 0:
            return torch.tensor(0.0, device=self.device)
        
        return sum(torch.norm(p, p=2) for p in adapter_params)
    
    def _log_metrics(self, metrics: Dict):
        """Log metrics to file."""
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f'{name}.pth'
        
        torch.save({
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_metric': self.best_val_metric,
            'config': self.config
        }, checkpoint_path)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_metric = checkpoint['best_val_metric']
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")


def create_dataloaders(config: dict) -> tuple:
    """Create train and validation dataloaders."""
    # Training dataset
    train_dataset = AugmentedStartEndDataset(
        data_path=config['train_data'],
        augmented_data_path=config.get('augmented_data', None),
        augmentation_ratio=config.get('augmentation_ratio', 0.5),
        clip_length=config.get('clip_length', 2.0),
        ctx_mode=config.get('ctx_mode', 'video_tef')
    )
    
    # Validation dataset (no augmentation)
    val_dataset = AugmentedStartEndDataset(
        data_path=config['val_data'],
        augmented_data_path=None,  # No augmentation for validation
        augmentation_ratio=0.0,
        clip_length=config.get('clip_length', 2.0),
        ctx_mode=config.get('ctx_mode', 'video_tef')
    )
    
    # Create dataloaders with mixed batch sampling
    train_sampler = MixedBatchSampler(
        dataset=train_dataset,
        batch_size=config['batch_size'],
        augmentation_ratio=config.get('augmentation_ratio', 0.5),
        shuffle=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=augmented_collate_fn,
        num_workers=config.get('num_workers', 4)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=augmented_collate_fn,
        num_workers=config.get('num_workers', 4)
    )
    
    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Val dataset: {len(val_dataset)} samples")
    
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="Train Visual ChatGPT model")
    
    # Data paths
    parser.add_argument('--train_data', type=str, required=True,
                       help='Path to training data')
    parser.add_argument('--val_data', type=str, required=True,
                       help='Path to validation data')
    parser.add_argument('--augmented_data', type=str, default=None,
                       help='Path to augmented queries')
    
    # Model configuration
    parser.add_argument('--cgdetr_checkpoint', type=str, required=True,
                       help='Path to pre-trained CG-DETR checkpoint')
    parser.add_argument('--adapter_dim', type=int, default=128,
                       help='Adapter bottleneck dimension')
    parser.add_argument('--num_token_adapters', type=int, default=2,
                       help='Number of token-level adapters')
    parser.add_argument('--fusion_type', type=str, default='gate',
                       choices=['gate', 'average', 'concat'],
                       help='Fusion type for dual streams')
    
    # Training configuration
    parser.add_argument('--training_stage', type=int, default=1,
                       choices=[1, 2],
                       help='Training stage: 1 (adapter warmup) or 2 (joint)')
    parser.add_argument('--num_epochs', type=int, default=20,
                       help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--augmentation_ratio', type=float, default=0.5,
                       help='Ratio of augmented queries in each batch')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--adapter_l2_weight', type=float, default=1e-5,
                       help='L2 regularization weight for adapters')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Maximum gradient norm for clipping')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--save_interval', type=int, default=5,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Log metrics every N batches')
    
    # Device configuration
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    
    args = parser.parse_args()
    
    # Configuration dictionary
    config = vars(args)
    config['loss_weights'] = {}  # Can be customized
    
    # Build model
    adapter_config = AdapterConfig(
        hidden_dim=256,  # From CG-DETR
        adapter_dim=args.adapter_dim,
        num_token_adapters=args.num_token_adapters,
        dropout=0.1,
        fusion_type=args.fusion_type,
        freeze_base=(args.training_stage == 1)
    )
    
    model = build_visual_chatgpt_model(
        cgdetr_checkpoint=args.cgdetr_checkpoint,
        adapter_config=adapter_config,
        freeze_cgdetr=(args.training_stage == 1),
        device=args.device
    )
    
    # Build criterion (use CG-DETR's criterion)
    from cg_detr.config import BaseOptions
    from cg_detr.model import build_model
    
    opt = BaseOptions().parse(['--dset_name', 'hl'])
    _, criterion = build_model(opt)
    criterion = criterion.to(args.device)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)
    
    # Build optimizer
    if args.training_stage == 1:
        # Only optimize adapter parameters
        params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    else:
        # Optimize all parameters
        params_to_optimize = model.parameters()
    
    optimizer = AdamW(
        params_to_optimize,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Build scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.lr * 0.01
    )
    
    # Create trainer
    trainer = VisualChatGPTTrainer(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=args.device
    )
    
    # Train
    trainer.train(
        num_epochs=args.num_epochs,
        training_stage=args.training_stage
    )
    
    logger.info("Training complete!")


if __name__ == '__main__':
    main()
