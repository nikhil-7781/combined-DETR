"""
Test script to verify QDECA module integration with QD-DETR.

This script performs shape verification and basic forward pass testing
to ensure the QDECA module is properly integrated.
"""

import torch
import sys
sys.path.insert(0, '/Users/nikhilvaidyanath/Desktop/QD-DETR')

from qd_detr.qdeca import QDECA, LearnedDecomposer


def test_learned_decomposer():
    """Test the LearnedDecomposer module."""
    print("=" * 60)
    print("Testing LearnedDecomposer...")
    print("=" * 60)

    B, L_txt, D = 4, 26, 256
    num_heads = 8

    # Create module
    decomposer = LearnedDecomposer(d_model=D, num_heads=num_heads)

    # Create dummy inputs
    src_txt = torch.randn(B, L_txt, D)
    src_txt_mask = torch.ones(B, L_txt, dtype=torch.bool)
    # Mask out last 6 tokens to simulate padding
    src_txt_mask[:, -6:] = False

    # Forward pass
    event_tokens, object_tokens, temporal_tokens, \
    event_mask, object_mask, temporal_mask = decomposer(src_txt, src_txt_mask)

    # Verify shapes
    assert event_tokens.shape == (B, L_txt, D), f"Event tokens shape mismatch: {event_tokens.shape}"
    assert object_tokens.shape == (B, L_txt, D), f"Object tokens shape mismatch: {object_tokens.shape}"
    assert temporal_tokens.shape == (B, L_txt, D), f"Temporal tokens shape mismatch: {temporal_tokens.shape}"
    assert event_mask.shape == (B, L_txt), f"Event mask shape mismatch: {event_mask.shape}"
    assert object_mask.shape == (B, L_txt), f"Object mask shape mismatch: {object_mask.shape}"
    assert temporal_mask.shape == (B, L_txt), f"Temporal mask shape mismatch: {temporal_mask.shape}"

    print(f"✓ Input shape: {src_txt.shape}")
    print(f"✓ Event tokens shape: {event_tokens.shape}")
    print(f"✓ Object tokens shape: {object_tokens.shape}")
    print(f"✓ Temporal tokens shape: {temporal_tokens.shape}")
    print(f"✓ All masks shape: {event_mask.shape}")
    print("✓ LearnedDecomposer test passed!\n")


def test_qdeca_module():
    """Test the full QDECA module."""
    print("=" * 60)
    print("Testing QDECA Module...")
    print("=" * 60)

    B, L_vid, L_txt, D = 4, 75, 26, 256
    num_heads = 8

    # Create module
    qdeca = QDECA(d_model=D, num_heads=num_heads, max_txt_len=32)

    # Create dummy inputs
    src_vid = torch.randn(B, L_vid, D)
    src_txt = torch.randn(B, L_txt, D)
    src_txt_mask = torch.ones(B, L_txt, dtype=torch.bool)
    src_txt_mask[:, -6:] = False  # Simulate padding

    # Forward pass
    src_vid_qdeca = qdeca(src_vid, src_txt, src_txt_mask)

    # Verify shape preservation
    assert src_vid_qdeca.shape == (B, L_vid, D), \
        f"Output shape mismatch: expected {(B, L_vid, D)}, got {src_vid_qdeca.shape}"

    print(f"✓ Video input shape: {src_vid.shape}")
    print(f"✓ Text input shape: {src_txt.shape}")
    print(f"✓ Video output shape: {src_vid_qdeca.shape}")
    print(f"✓ Shape preserved: {src_vid.shape == src_vid_qdeca.shape}")

    # Test gating weights
    gate_weights = qdeca.get_gate_weights(src_vid)
    assert gate_weights.shape == (B, L_vid, 3), f"Gate weights shape mismatch: {gate_weights.shape}"

    # Verify softmax (should sum to 1)
    gate_sum = gate_weights.sum(dim=-1)
    assert torch.allclose(gate_sum, torch.ones_like(gate_sum), atol=1e-6), \
        "Gate weights do not sum to 1"

    print(f"✓ Gate weights shape: {gate_weights.shape}")
    print(f"✓ Gate weights sum to 1: {torch.allclose(gate_sum, torch.ones_like(gate_sum), atol=1e-6)}")
    print(f"✓ Average gate weights (event, object, temporal): "
          f"[{gate_weights[..., 0].mean():.3f}, "
          f"{gate_weights[..., 1].mean():.3f}, "
          f"{gate_weights[..., 2].mean():.3f}]")
    print("✓ QDECA module test passed!\n")


def test_qdeca_with_empty_temporal():
    """Test QDECA with null temporal token fallback."""
    print("=" * 60)
    print("Testing QDECA with Null Temporal Token...")
    print("=" * 60)

    B, L_vid, L_txt, D = 4, 75, 26, 256
    num_heads = 8

    # Create module
    qdeca = QDECA(d_model=D, num_heads=num_heads, max_txt_len=32)

    # Create inputs where temporal component would be zero
    src_vid = torch.randn(B, L_vid, D)
    src_txt = torch.zeros(B, L_txt, D)  # Zero text to trigger null temporal
    src_txt_mask = torch.ones(B, L_txt, dtype=torch.bool)

    # Forward pass (should not crash)
    src_vid_qdeca = qdeca(src_vid, src_txt, src_txt_mask)

    assert src_vid_qdeca.shape == (B, L_vid, D), "Shape mismatch with null temporal"

    print(f"✓ Video output shape: {src_vid_qdeca.shape}")
    print("✓ Null temporal token fallback working!\n")


def test_gradient_flow():
    """Test that gradients flow through QDECA."""
    print("=" * 60)
    print("Testing Gradient Flow...")
    print("=" * 60)

    B, L_vid, L_txt, D = 2, 75, 26, 256
    num_heads = 8

    # Create module
    qdeca = QDECA(d_model=D, num_heads=num_heads, max_txt_len=32)

    # Create inputs with requires_grad
    src_vid = torch.randn(B, L_vid, D, requires_grad=True)
    src_txt = torch.randn(B, L_txt, D, requires_grad=True)
    src_txt_mask = torch.ones(B, L_txt, dtype=torch.bool)

    # Forward pass
    src_vid_qdeca = qdeca(src_vid, src_txt, src_txt_mask)

    # Compute dummy loss and backward
    loss = src_vid_qdeca.sum()
    loss.backward()

    # Check gradients exist
    assert src_vid.grad is not None, "No gradient for video input"
    assert src_txt.grad is not None, "No gradient for text input"

    print(f"✓ Video gradient shape: {src_vid.grad.shape}")
    print(f"✓ Text gradient shape: {src_txt.grad.shape}")
    print(f"✓ Video gradient mean: {src_vid.grad.mean().item():.6f}")
    print(f"✓ Text gradient mean: {src_txt.grad.mean().item():.6f}")
    print("✓ Gradient flow test passed!\n")


def test_batch_independence():
    """Test that QDECA processes batches independently."""
    print("=" * 60)
    print("Testing Batch Independence...")
    print("=" * 60)

    L_vid, L_txt, D = 75, 26, 256
    num_heads = 8

    # Create module
    qdeca = QDECA(d_model=D, num_heads=num_heads, max_txt_len=32)
    qdeca.eval()  # Deterministic mode

    # Create two identical samples
    src_vid_single = torch.randn(1, L_vid, D)
    src_txt_single = torch.randn(1, L_txt, D)
    src_txt_mask_single = torch.ones(1, L_txt, dtype=torch.bool)

    # Process as batch of 1
    with torch.no_grad():
        out_single = qdeca(src_vid_single, src_txt_single, src_txt_mask_single)

    # Process as batch of 2 (duplicated)
    src_vid_batch = src_vid_single.repeat(2, 1, 1)
    src_txt_batch = src_txt_single.repeat(2, 1, 1)
    src_txt_mask_batch = src_txt_mask_single.repeat(2, 1)

    with torch.no_grad():
        out_batch = qdeca(src_vid_batch, src_txt_batch, src_txt_mask_batch)

    # Verify both batch elements are identical
    assert torch.allclose(out_batch[0], out_batch[1], atol=1e-5), \
        "Batch elements not processed identically"
    assert torch.allclose(out_batch[0], out_single[0], atol=1e-5), \
        "Batch output differs from single output"

    print("✓ Batch independence verified")
    print("✓ Outputs are deterministic and identical\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("QDECA MODULE INTEGRATION TESTS")
    print("=" * 60 + "\n")

    try:
        test_learned_decomposer()
        test_qdeca_module()
        test_qdeca_with_empty_temporal()
        test_gradient_flow()
        test_batch_independence()

        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nQDECA module is properly integrated and functional.")
        print("You can now train with --use_qdeca flag.\n")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
