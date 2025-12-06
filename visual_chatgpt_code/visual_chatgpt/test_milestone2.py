#!/usr/bin/env python3
"""
Test script to verify Milestone 2 implementation.
Checks that all components can be imported and initialized.
"""

import sys
import os
import torch
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from visual_chatgpt.nested_adapter.adapter_layers import (
            AdapterLayer, DualStreamAdapter, NestedAdapterModule, AdapterConfig
        )
        from visual_chatgpt.augmented_dataset import (
            AugmentedStartEndDataset, augmented_collate_fn, MixedBatchSampler
        )
        from visual_chatgpt.visual_chatgpt_model import (
            VisualChatGPT_CGDETR, build_visual_chatgpt_model
        )
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_adapter_layer():
    """Test AdapterLayer initialization and forward pass."""
    print("\nTesting AdapterLayer...")
    try:
        from visual_chatgpt.nested_adapter.adapter_layers import AdapterLayer
        
        adapter = AdapterLayer(hidden_dim=256, adapter_dim=64, dropout=0.1)
        
        # Test forward pass
        x = torch.randn(2, 10, 256)  # (batch, seq_len, hidden_dim)
        output = adapter(x)
        
        assert output.shape == x.shape, f"Shape mismatch: {output.shape} != {x.shape}"
        
        # Count parameters
        num_params = sum(p.numel() for p in adapter.parameters())
        expected_params = 2 * 256 * 64  # down + up projections
        
        print(f"  Parameters: {num_params:,} (expected ~{expected_params:,})")
        print("✅ AdapterLayer works correctly")
        return True
    except Exception as e:
        print(f"❌ AdapterLayer failed: {e}")
        return False


def test_dual_stream_adapter():
    """Test DualStreamAdapter with different fusion types."""
    print("\nTesting DualStreamAdapter...")
    try:
        from visual_chatgpt.nested_adapter.adapter_layers import DualStreamAdapter
        
        for fusion_type in ['gate', 'average', 'concat']:
            adapter = DualStreamAdapter(
                hidden_dim=256,
                adapter_dim=64,
                fusion_type=fusion_type
            )
            
            # Test forward pass
            x = torch.randn(2, 10, 256)
            output = adapter(x, query_source='augmented')
            
            if fusion_type == 'concat':
                assert output.shape == (2, 10, 256), f"Concat fusion output shape wrong"
            else:
                assert output.shape == x.shape, f"Shape mismatch for {fusion_type}"
            
            print(f"  ✓ {fusion_type} fusion works")
        
        print("✅ DualStreamAdapter works for all fusion types")
        return True
    except Exception as e:
        print(f"❌ DualStreamAdapter failed: {e}")
        return False


def test_nested_adapter_module():
    """Test NestedAdapterModule."""
    print("\nTesting NestedAdapterModule...")
    try:
        from visual_chatgpt.nested_adapter.adapter_layers import NestedAdapterModule
        
        adapter = NestedAdapterModule(
            hidden_dim=256,
            adapter_dim=64,
            num_token_adapters=2
        )
        
        # Test forward pass
        text_features = torch.randn(2, 10, 256)
        video_features = torch.randn(2, 20, 256)
        
        output = adapter(
            text_features=text_features,
            query_source='human',
            video_features=video_features
        )
        
        assert output.shape == text_features.shape
        
        # Count parameters
        num_params = sum(p.numel() for p in adapter.parameters())
        print(f"  Parameters: {num_params:,}")
        print("✅ NestedAdapterModule works correctly")
        return True
    except Exception as e:
        print(f"❌ NestedAdapterModule failed: {e}")
        return False


def test_adapter_config():
    """Test AdapterConfig serialization."""
    print("\nTesting AdapterConfig...")
    try:
        from visual_chatgpt.nested_adapter.adapter_layers import AdapterConfig
        
        config = AdapterConfig(
            hidden_dim=256,
            adapter_dim=64,
            num_token_adapters=2,
            dropout=0.1,
            fusion_type='gate'
        )
        
        # Test serialization
        config_dict = config.to_dict()
        config_restored = AdapterConfig.from_dict(config_dict)
        
        assert config_restored.hidden_dim == config.hidden_dim
        assert config_restored.fusion_type == config.fusion_type
        
        print("✅ AdapterConfig serialization works")
        return True
    except Exception as e:
        print(f"❌ AdapterConfig failed: {e}")
        return False


def test_augmented_dataset():
    """Test AugmentedStartEndDataset (check initialization only)."""
    print("\nTesting AugmentedStartEndDataset...")
    try:
        from visual_chatgpt.augmented_dataset import AugmentedStartEndDataset
        
        # Check class can be imported and has required methods
        required_methods = ['__init__', '__len__', '__getitem__']
        for method in required_methods:
            assert hasattr(AugmentedStartEndDataset, method), f"Missing method: {method}"
        
        print("  ✓ AugmentedStartEndDataset class structure is valid")
        print("  ⚠️  Skipping dataset loading (requires CG-DETR dataset setup)")
        print("✅ AugmentedStartEndDataset structure verified")
        return True
    except Exception as e:
        print(f"❌ AugmentedDataset failed: {e}")
        return False


def test_mixed_batch_sampler():
    """Test MixedBatchSampler."""
    print("\nTesting MixedBatchSampler...")
    try:
        from visual_chatgpt.augmented_dataset import MixedBatchSampler
        
        # Create dummy dataset with required attributes
        class DummyDataset:
            def __init__(self):
                self.qid_to_augmented = {
                    f'q{i}': [] for i in range(100)
                }
                self.datalist = [
                    {'qid': f'q{i}', 'query_source': 'human'}
                    for i in range(100)
                ]
            
            def __len__(self):
                return len(self.datalist)
            
            def __getitem__(self, idx):
                return self.datalist[idx]
        
        dataset = DummyDataset()
        sampler = MixedBatchSampler(
            dataset=dataset,
            batch_size=8,
            augmentation_ratio=0.5,
            shuffle=False
        )
        
        # Test batch generation
        batches = list(sampler)
        print(f"  Generated {len(batches)} batches")
        print(f"  First batch size: {len(batches[0])}")
        
        print("✅ MixedBatchSampler works correctly")
        return True
    except Exception as e:
        print(f"❌ MixedBatchSampler failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Milestone 2: Nested Adapter Architecture - Verification")
    print("=" * 60)
    
    results = {
        'Imports': test_imports(),
        'AdapterLayer': test_adapter_layer(),
        'DualStreamAdapter': test_dual_stream_adapter(),
        'NestedAdapterModule': test_nested_adapter_module(),
        'AdapterConfig': test_adapter_config(),
        'AugmentedDataset': test_augmented_dataset(),
        'MixedBatchSampler': test_mixed_batch_sampler(),
    }
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:.<30} {status}")
    
    all_passed = all(results.values())
    
    print("=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED - Milestone 2 implementation verified!")
        print("=" * 60)
        return 0
    else:
        print("❌ SOME TESTS FAILED - Check errors above")
        print("=" * 60)
        return 1


if __name__ == '__main__':
    sys.exit(main())
