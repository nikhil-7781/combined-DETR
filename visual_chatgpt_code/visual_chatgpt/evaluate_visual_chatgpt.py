"""
Evaluation script for Visual ChatGPT model.
Tests on both original and augmented queries, with paraphrase robustness analysis.
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from visual_chatgpt.visual_chatgpt_model import build_visual_chatgpt_model
from visual_chatgpt.augmented_dataset import (
    AugmentedStartEndDataset,
    augmented_collate_fn
)
from standalone_eval.eval import eval_submission

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VisualChatGPTEvaluator:
    """Evaluator for Visual ChatGPT model."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        data_loader: DataLoader,
        config: dict,
        device: str = 'cuda'
    ):
        self.model = model
        self.data_loader = data_loader
        self.config = config
        self.device = device
        
        # Results storage
        self.predictions = []
        self.query_source_metrics = defaultdict(list)
    
    @torch.no_grad()
    def evaluate(self) -> Dict:
        """Run full evaluation."""
        self.model.eval()
        
        logger.info("Starting evaluation...")
        
        for batch in tqdm(self.data_loader, desc="Evaluating"):
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
            
            # Extract predictions
            self._process_predictions(outputs, batch)
        
        # Compute metrics
        metrics = self._compute_metrics()
        
        # Analyze by query source
        source_analysis = self._analyze_by_source()
        metrics['source_analysis'] = source_analysis
        
        # Paraphrase robustness (if augmented queries available)
        if self.config.get('evaluate_robustness', False):
            robustness_metrics = self._evaluate_paraphrase_robustness()
            metrics['robustness'] = robustness_metrics
        
        return metrics
    
    def _process_predictions(self, outputs: Dict, batch: Dict):
        """Process model outputs into predictions."""
        # This depends on CG-DETR's output format
        # Typically: {'pred_spans': (N, 2), 'pred_saliency': (N, L)}
        
        batch_size = outputs['pred_spans'].shape[0]
        
        for i in range(batch_size):
            pred = {
                'qid': batch['qid'][i] if 'qid' in batch else i,
                'vid': batch['vid'][i] if 'vid' in batch else '',
                'pred_span': outputs['pred_spans'][i].cpu().tolist(),
                'query_source': batch.get('query_source', 'unknown'),
            }
            
            # Include ground truth if available
            if 'timestamps' in batch:
                pred['gt_span'] = batch['timestamps'][i].cpu().tolist()
            
            self.predictions.append(pred)
            
            # Track by source
            source = batch.get('query_source', 'unknown')
            self.query_source_metrics[source].append(pred)
    
    def _compute_metrics(self) -> Dict:
        """Compute evaluation metrics."""
        # Save predictions
        pred_file = Path(self.config['output_dir']) / 'predictions.jsonl'
        with open(pred_file, 'w') as f:
            for pred in self.predictions:
                f.write(json.dumps(pred) + '\n')
        
        logger.info(f"Saved predictions to {pred_file}")
        
        # Use standalone eval if available
        if Path(self.config.get('gt_file', '')).exists():
            metrics = eval_submission(
                submission_path=str(pred_file),
                gt_path=self.config['gt_file']
            )
        else:
            # Basic metrics
            metrics = self._compute_basic_metrics()
        
        return metrics
    
    def _compute_basic_metrics(self) -> Dict:
        """Compute basic IoU and accuracy metrics."""
        ious = []
        
        for pred in self.predictions:
            if 'gt_span' in pred:
                iou = self._compute_iou(pred['pred_span'], pred['gt_span'])
                ious.append(iou)
        
        if len(ious) == 0:
            return {}
        
        ious = np.array(ious)
        
        metrics = {
            'mIoU': float(np.mean(ious)),
            'R@0.3': float(np.mean(ious >= 0.3)),
            'R@0.5': float(np.mean(ious >= 0.5)),
            'R@0.7': float(np.mean(ious >= 0.7)),
        }
        
        return metrics
    
    def _compute_iou(self, pred_span: List[float], gt_span: List[float]) -> float:
        """Compute IoU between predicted and ground truth spans."""
        pred_start, pred_end = pred_span
        gt_start, gt_end = gt_span
        
        # Intersection
        intersection_start = max(pred_start, gt_start)
        intersection_end = min(pred_end, gt_end)
        intersection = max(0, intersection_end - intersection_start)
        
        # Union
        union = (pred_end - pred_start) + (gt_end - gt_start) - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _analyze_by_source(self) -> Dict:
        """Analyze metrics by query source."""
        source_metrics = {}
        
        for source, preds in self.query_source_metrics.items():
            if len(preds) == 0:
                continue
            
            # Compute metrics for this source
            ious = []
            for pred in preds:
                if 'gt_span' in pred:
                    iou = self._compute_iou(pred['pred_span'], pred['gt_span'])
                    ious.append(iou)
            
            if len(ious) > 0:
                ious = np.array(ious)
                source_metrics[source] = {
                    'count': len(preds),
                    'mIoU': float(np.mean(ious)),
                    'R@0.5': float(np.mean(ious >= 0.5)),
                }
        
        return source_metrics
    
    def _evaluate_paraphrase_robustness(self) -> Dict:
        """
        Evaluate robustness to paraphrases.
        Groups augmented queries by original query and measures consistency.
        """
        # Group predictions by original query
        query_groups = defaultdict(list)
        
        for pred in self.predictions:
            if pred['query_source'] == 'augmented':
                # Assumes qid format: original_id_aug_N
                original_qid = '_'.join(pred['qid'].split('_')[:-2])
                query_groups[original_qid].append(pred)
        
        # Compute consistency metrics
        span_variances = []
        iou_consistencies = []
        
        for original_qid, group in query_groups.items():
            if len(group) < 2:
                continue
            
            # Measure span variance
            spans = [pred['pred_span'] for pred in group]
            span_variance = np.var(spans, axis=0).mean()
            span_variances.append(span_variance)
            
            # Measure pairwise IoU consistency
            pairwise_ious = []
            for i in range(len(spans)):
                for j in range(i + 1, len(spans)):
                    iou = self._compute_iou(spans[i], spans[j])
                    pairwise_ious.append(iou)
            
            if len(pairwise_ious) > 0:
                iou_consistencies.append(np.mean(pairwise_ious))
        
        robustness_metrics = {
            'num_query_groups': len(query_groups),
            'avg_span_variance': float(np.mean(span_variances)) if span_variances else 0.0,
            'avg_consistency_iou': float(np.mean(iou_consistencies)) if iou_consistencies else 0.0,
        }
        
        logger.info(f"Paraphrase robustness: {robustness_metrics}")
        
        return robustness_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Visual ChatGPT model")
    
    # Data paths
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test data')
    parser.add_argument('--augmented_data', type=str, default=None,
                       help='Path to augmented test queries')
    parser.add_argument('--gt_file', type=str, default=None,
                       help='Path to ground truth file for official eval')
    
    # Model checkpoint
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    
    # Evaluation configuration
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--evaluate_robustness', action='store_true',
                       help='Evaluate paraphrase robustness')
    parser.add_argument('--split_by_source', action='store_true',
                       help='Split results by query source')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for predictions and metrics')
    
    # Device configuration
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration dictionary
    config = vars(args)
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    
    # For now, use placeholder - actual implementation would load from checkpoint
    # model = build_visual_chatgpt_model(...)
    # checkpoint = torch.load(args.checkpoint, map_location=args.device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.warning("Model loading not fully implemented - placeholder")
    
    # Create dataset
    test_dataset = AugmentedStartEndDataset(
        data_path=args.test_data,
        augmented_data_path=args.augmented_data,
        augmentation_ratio=0.5 if args.augmented_data else 0.0,
        clip_length=2.0,
        ctx_mode='video_tef'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=augmented_collate_fn,
        num_workers=args.num_workers
    )
    
    logger.info(f"Test dataset: {len(test_dataset)} samples")
    
    # Create evaluator (placeholder model for now)
    # evaluator = VisualChatGPTEvaluator(
    #     model=model,
    #     data_loader=test_loader,
    #     config=config,
    #     device=args.device
    # )
    
    # Run evaluation
    # metrics = evaluator.evaluate()
    
    # Save metrics
    # metrics_file = output_dir / 'metrics.json'
    # with open(metrics_file, 'w') as f:
    #     json.dump(metrics, f, indent=2)
    
    # logger.info(f"Evaluation complete! Metrics saved to {metrics_file}")
    # logger.info(f"Results:\n{json.dumps(metrics, indent=2)}")
    
    logger.info("Evaluation script structure complete (full implementation pending)")


if __name__ == '__main__':
    main()
