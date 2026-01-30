"""
Quick test script to verify all new rebuttal modules are working.
Run this before running the full experiments.
"""
import sys
import torch
import numpy as np
from pathlib import Path

def test_imports():
    """Test that all new modules can be imported"""
    print("Testing imports...")
    
    try:
        from speech_dataset import load_keyword_spotting_wrapper, SpeechCommandsDataset
        print("  ✓ speech_dataset")
    except Exception as e:
        print(f"  ✗ speech_dataset: {e}")
        return False
    
    try:
        from ternary_baseline import TernarySeparableCNN, TernaryConv1d, build_ternary_separable
        print("  ✓ ternary_baseline")
    except Exception as e:
        print(f"  ✗ ternary_baseline: {e}")
        return False
    
    try:
        from synthesis_profiler import SynthesisProfiler, profile_hypertiny_model
        print("  ✓ synthesis_profiler")
    except Exception as e:
        print(f"  ✗ synthesis_profiler: {e}")
        return False
    
    try:
        from nas_compatibility import NASWithHyperTinyPW, NASInspiredBackbone
        print("  ✓ nas_compatibility")
    except Exception as e:
        print(f"  ✗ nas_compatibility: {e}")
        return False
    
    return True


def test_ternary_model():
    """Test ternary quantization model"""
    print("\nTesting ternary model...")
    
    from ternary_baseline import TernarySeparableCNN
    
    try:
        model = TernarySeparableCNN(in_ch=1, base=8, num_classes=2)
        x = torch.randn(2, 1, 1800)
        
        # Forward pass
        y = model(x)
        assert y.shape == (2, 2), f"Expected (2, 2), got {y.shape}"
        
        # Flash computation
        breakdown = model.compute_total_flash_bytes()
        assert 'total' in breakdown
        assert breakdown['total'] > 0
        
        print(f"  ✓ Forward pass: {y.shape}")
        print(f"  ✓ Flash usage: {breakdown['total']/1024:.2f} KB")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_nas_model():
    """Test NAS compatibility model"""
    print("\nTesting NAS compatibility...")
    
    from nas_compatibility import NASInspiredBackbone, NASWithHyperTinyPW
    
    try:
        # Baseline NAS
        config = [(16, 3, 1, 2), (24, 5, 2, 4)]
        baseline = NASInspiredBackbone(in_ch=1, num_classes=2, config=config)
        
        x = torch.randn(2, 1, 1800)
        y1 = baseline(x)
        assert y1.shape == (2, 2)
        
        # With compression
        compressed = NASWithHyperTinyPW(in_ch=1, num_classes=2, 
                                        nas_config=config, compress_pw_layers=True)
        y2 = compressed(x)
        assert y2.shape == (2, 2)
        
        stats = compressed.compute_compression_stats()
        assert 'compression_ratio' in stats
        
        print(f"  ✓ Baseline forward: {y1.shape}")
        print(f"  ✓ Compressed forward: {y2.shape}")
        print(f"  ✓ Compression ratio: {stats['compression_ratio']:.2f}x")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_synthesis_profiler():
    """Test synthesis profiler"""
    print("\nTesting synthesis profiler...")
    
    from synthesis_profiler import SynthesisProfiler
    import torch.nn as nn
    
    try:
        # Create dummy generator
        class DummyGen(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(16, 128)
                self.z = nn.Parameter(torch.randn(16))
            
            def forward(self):
                return self.fc(self.z).view(16, 8, 1)
        
        gen = DummyGen()
        
        def gen_fn():
            with torch.no_grad():
                return gen()
        
        profiler = SynthesisProfiler(device='cpu', warmup=2, repeats=5)
        synth_time, synth_energy, weight_bytes = profiler.profile_synthesis(
            gen_fn, (16, 8, 1), "test_layer"
        )
        
        assert synth_time > 0
        assert synth_energy > 0
        assert weight_bytes == 16 * 8 * 1 * 4  # FP32
        
        print(f"  ✓ Synthesis time: {synth_time:.3f}ms")
        print(f"  ✓ Weight size: {weight_bytes} bytes")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_speech_dataset_structure():
    """Test speech dataset structure (without actual data)"""
    print("\nTesting speech dataset structure...")
    
    from speech_dataset import TINYML_KEYWORDS, SPEECH_COMMANDS_CLASSES
    
    try:
        assert len(TINYML_KEYWORDS) == 12
        assert len(SPEECH_COMMANDS_CLASSES) == 35
        assert 'yes' in TINYML_KEYWORDS
        assert 'no' in TINYML_KEYWORDS
        
        print(f"  ✓ Keywords defined: {len(TINYML_KEYWORDS)}")
        print(f"  ✓ Full classes: {len(SPEECH_COMMANDS_CLASSES)}")
        print("  ℹ To test loading, download Speech Commands dataset")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    print("=" * 70)
    print("REBUTTAL MODULES TEST SUITE")
    print("=" * 70)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Ternary Model", test_ternary_model()))
    results.append(("NAS Model", test_nas_model()))
    results.append(("Synthesis Profiler", test_synthesis_profiler()))
    results.append(("Speech Dataset", test_speech_dataset_structure()))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {name}")
    
    print("-" * 70)
    print(f"Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nAll tests passed! Ready to run experiments.")
        print("\nNext steps:")
        print("  1. Download Speech Commands dataset (optional)")
        print("  2. Run: python run_rebuttal_experiments.py --experiments all")
        return 0
    else:
        print("\nWARNING: Some tests failed. Check errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
