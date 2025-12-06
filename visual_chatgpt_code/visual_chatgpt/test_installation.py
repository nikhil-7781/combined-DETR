#!/usr/bin/env python3
"""
Test script to verify augmentation pipeline installation and basic functionality.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from visual_chatgpt.llm_augmentation import (
            query_generator,
            prompt_templates,
            quality_filter,
            diversity_metrics,
            batch_augmenter
        )
        print("✓ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_prompt_templates():
    """Test prompt template generation."""
    print("\nTesting prompt templates...")
    
    try:
        from visual_chatgpt.llm_augmentation.prompt_templates import PromptTemplateManager
        
        system_prompt, user_prompt = PromptTemplateManager.get_prompt(
            strategy="semantic_paraphrase",
            original_query="A person opens a door",
            num_variations=3,
            duration=150.0,
            start_time=10.0,
            end_time=15.0
        )
        
        assert len(system_prompt) > 0
        assert "A person opens a door" in user_prompt
        assert "3" in user_prompt
        
        print("✓ Prompt templates working correctly")
        return True
    except Exception as e:
        print(f"✗ Prompt template test failed: {e}")
        return False


def test_quality_filter():
    """Test quality filter initialization."""
    print("\nTesting quality filter...")
    
    try:
        from visual_chatgpt.llm_augmentation.quality_filter import SemanticQualityFilter
        
        filter = SemanticQualityFilter(device="cpu")
        
        # Test validation
        original = "A person opens a door"
        generated = "An individual pushes the door open"
        
        valid, score, details = filter.validate_query(original, generated)
        
        print(f"  Validation result: {valid}")
        print(f"  Quality score: {score:.3f}")
        print(f"  Keyword overlap: {details['keyword_overlap']:.3f}")
        
        print("✓ Quality filter working correctly")
        return True
    except Exception as e:
        print(f"✗ Quality filter test failed: {e}")
        return False


def test_dependencies():
    """Test that all required dependencies are installed."""
    print("\nTesting dependencies...")
    
    required = [
        ("transformers", "Transformers library"),
        ("torch", "PyTorch"),
        ("spacy", "spaCy NLP"),
    ]
    
    all_ok = True
    for module_name, description in required:
        try:
            __import__(module_name)
            print(f"✓ {description}")
        except ImportError:
            print(f"✗ {description} not installed")
            all_ok = False
    
    # Check spaCy model
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("✓ spaCy model 'en_core_web_sm'")
    except OSError:
        print("✗ spaCy model 'en_core_web_sm' not installed")
        print("  Run: python -m spacy download en_core_web_sm")
        all_ok = False
    
    return all_ok


def main():
    """Run all tests."""
    print("=" * 60)
    print("Visual ChatGPT - Installation Test")
    print("=" * 60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Dependencies", test_dependencies()))
    results.append(("Prompt Templates", test_prompt_templates()))
    results.append(("Quality Filter", test_quality_filter()))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {test_name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("=" * 60)
    if all_passed:
        print("All tests passed! Ready to run augmentation.")
        print("\nNext steps:")
        print("1. Run pilot study: bash scripts/visual_chatgpt/pilot_study.sh")
        print("2. Check results in: data/augmented/")
        return 0
    else:
        print("Some tests failed. Please install missing dependencies.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
