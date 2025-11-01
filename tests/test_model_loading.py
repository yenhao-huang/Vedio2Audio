#!/usr/bin/env python3
"""
Test script to verify model loading functionality for both Wan2.1 and Wan2.2
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.text2vedio import Text2Vedio
import torch

def test_model_loading(model_name, quantized=False):
    """
    Test loading a specific model

    Args:
        model_name: Model key ('wan_2_1' or 'wan_2_2')
        quantized: Whether to use quantization
    """
    print(f"\n{'='*80}")
    print(f"Testing Model: {model_name} (quantized={quantized})")
    print(f"{'='*80}\n")

    try:
        # Initialize the model
        t2v = Text2Vedio(model_name=model_name, quantized=quantized)

        print(f"\n✅ SUCCESS: {model_name} loaded successfully!")
        print(f"   Model HuggingFace ID: {t2v.model_name}")
        print(f"   Model key: {t2v.model_name}")
        print(f"   Device: {t2v.device}")
        print(f"   Dtype: {t2v.dtype}")
        print(f"   Initialization time: {t2v.metrics['initialization_time']:.2f}s")

        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            memory_allocated = torch.cuda.memory_allocated(device_id) / (1024 ** 3)
            print(f"   GPU Memory allocated: {memory_allocated:.2f} GB")

        # Clean up
        del t2v
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return True

    except Exception as e:
        print(f"\n❌ FAILED: {model_name} loading failed!")
        print(f"   Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all model loading tests"""

    print("\n" + "="*80)
    print("Model Loading Test Suite")
    print("="*80)

    results = {}

    # Test 1: Wan2.1 without quantization
    results['wan_2_1_no_quant'] = test_model_loading('wan_2_1', quantized=False)

    # Clean up GPU memory between tests
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("\n🧹 Cleared GPU cache")

    # Test 2: Wan2.1 with quantization
    print("\n" + "-"*80)
    results['wan_2_1_quant'] = test_model_loading('wan_2_1', quantized=True)

    # Clean up GPU memory between tests
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("\n🧹 Cleared GPU cache")

    # Test 3: Wan2.2 without quantization
    print("\n" + "-"*80)
    results['wan_2_2_no_quant'] = test_model_loading('wan_2_2', quantized=False)

    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Print summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")

    total_tests = len(results)
    passed_tests = sum(results.values())

    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️ {total_tests - passed_tests} test(s) failed")
        return 1

if __name__ == "__main__":
    exit(main())
