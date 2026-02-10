"""
Quick VM Tests - Fast validation without training
Run this on any VM to quickly test all code paths
"""
import sys
import time
from pathlib import Path

# Add tinyml to path if running from scripts folder
sys.path.insert(0, str(Path(__file__).parent.parent / 'tinyml'))

def print_header(text):
    print("\n" + "="*60)
    print(text)
    print("="*60)

def print_test(name):
    print(f"\n[TEST] {name}...")

def print_ok(msg):
    print(f"  [OK] {msg}")

def print_fail(msg):
    print(f"  [FAIL] {msg}")

def test_imports():
    """Test 1: All modules can be imported"""
    print_test("Module Imports")
    
    modules = [
        ('models', ['HyperTinyPW', 'safe_build_model']),
        ('experiments', ['ExpCfg', 'train_regular_cnn']),
        ('datasets', ['ApneaECGWindows', 'create_synthetic_ecg_data']),
        ('data_loaders', ['load_apnea_ecg_loaders_impl']),
        ('ternary_baseline', ['TernaryQuantizer']),
        ('synthesis_profiler', ['SynthesisProfiler']),
    ]
    
    for module_name, attributes in modules:
        try:
            module = __import__(module_name)
            for attr in attributes:
                assert hasattr(module, attr), f"Missing {attr}"
            print_ok(f"{module_name}")
        except Exception as e:
            print_fail(f"{module_name}: {e}")
            return False
    
    return True

def test_model_creation():
    """Test 2: Model instantiation and forward pass"""
    print_test("Model Creation & Forward Pass")
    
    try:
        from models import HyperTinyPW
        import torch
        
        # Create small model
        model = HyperTinyPW(
            num_classes=2,
            in_channels=1,
            base_channels=8,
            num_blocks=2,
            latent_dim=32,
            seq_len=100
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        print_ok(f"Model created: {total_params:,} parameters")
        
        # Forward pass
        x = torch.randn(4, 1, 100)
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (4, 2), f"Wrong output shape: {output.shape}"
        print_ok(f"Forward pass: {x.shape} -> {output.shape}")
        
        # Backward pass
        y = torch.randint(0, 2, (4,))
        output = model(x)
        loss = torch.nn.functional.cross_entropy(output, y)
        loss.backward()
        
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad, "No gradients computed"
        print_ok(f"Backward pass: gradients computed")
        
        return True
        
    except Exception as e:
        print_fail(f"Model creation failed: {e}")
        return False

def test_quantization():
    """Test 3: Ternary quantization"""
    print_test("Ternary Quantization")
    
    try:
        from ternary_baseline import TernaryQuantizer
        import torch
        
        quantizer = TernaryQuantizer(threshold=0.7)
        
        # Test quantization
        weight = torch.randn(50, 50)
        quantized, scale = quantizer.quantize(weight)
        
        unique_vals = torch.unique(quantized).tolist()
        assert all(v in [-1, 0, 1] for v in unique_vals), f"Invalid quantized values: {unique_vals}"
        print_ok(f"Quantized to: {unique_vals}")
        
        # Size calculation
        original_kb = (50 * 50 * 4) / 1024
        quantized_kb = (50 * 50 * 0.25) / 1024
        ratio = original_kb / quantized_kb
        print_ok(f"Compression: {original_kb:.2f} KB -> {quantized_kb:.2f} KB ({ratio:.1f}x)")
        
        return True
        
    except Exception as e:
        print_fail(f"Quantization failed: {e}")
        return False

def test_synthesis_profiler():
    """Test 4: Synthesis profiling"""
    print_test("Synthesis Profiler")
    
    try:
        from synthesis_profiler import SynthesisProfiler
        from models import HyperTinyPW
        import torch
        
        model = HyperTinyPW(
            num_classes=2,
            in_channels=1,
            base_channels=8,
            num_blocks=2,
            latent_dim=32,
            seq_len=100
        )
        
        profiler = SynthesisProfiler(model, device='cpu')
        print_ok("Profiler created")
        
        # Profile inference
        x = torch.randn(1, 1, 100)
        start = time.time()
        n_runs = 20
        for _ in range(n_runs):
            with torch.no_grad():
                _ = model(x)
        elapsed_ms = (time.time() - start) / n_runs * 1000
        
        print_ok(f"Inference time: {elapsed_ms:.2f} ms/sample")
        
        return True
        
    except Exception as e:
        print_fail(f"Profiler failed: {e}")
        return False

def test_data_generation():
    """Test 5: Synthetic data generation"""
    print_test("Data Generation")
    
    try:
        from datasets import create_synthetic_ecg_data
        import torch
        
        # Generate small dataset
        X, y = create_synthetic_ecg_data(n_samples=100, seq_len=1800)
        
        assert X.shape == (100, 1, 1800), f"Wrong X shape: {X.shape}"
        assert y.shape == (100,), f"Wrong y shape: {y.shape}"
        print_ok(f"Generated data: X={X.shape}, y={y.shape}")
        
        # Check class balance
        n_class0 = torch.sum(y == 0).item()
        n_class1 = torch.sum(y == 1).item()
        print_ok(f"Class distribution: 0={n_class0}, 1={n_class1}")
        
        return True
        
    except Exception as e:
        print_fail(f"Data generation failed: {e}")
        return False

def test_experiment_setup():
    """Test 6: Experiment configuration"""
    print_test("Experiment Setup")
    
    try:
        from experiments import ExpCfg, seed_everything
        import torch
        
        # Create config
        cfg = ExpCfg(
            epochs=5,
            batch_size=32,
            lr=0.001,
            device='cpu'
        )
        print_ok(f"Config created: epochs={cfg.epochs}, batch_size={cfg.batch_size}")
        
        # Test seeding
        seed_everything(42)
        r1 = torch.rand(5)
        
        seed_everything(42)
        r2 = torch.rand(5)
        
        assert torch.allclose(r1, r2), "Seeding not working"
        print_ok("Random seeding works")
        
        return True
        
    except Exception as e:
        print_fail(f"Experiment setup failed: {e}")
        return False

def run_all_tests():
    """Run all quick tests"""
    print_header("HyperTinyPW - Quick VM Tests")
    print("Fast validation of all code paths (no training)\n")
    
    tests = [
        ("Module Imports", test_imports),
        ("Model Creation", test_model_creation),
        ("Quantization", test_quantization),
        ("Synthesis Profiler", test_synthesis_profiler),
        ("Data Generation", test_data_generation),
        ("Experiment Setup", test_experiment_setup),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print_fail(f"Unexpected error: {e}")
            results.append((name, False))
    
    # Summary
    print_header("Test Summary")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "[OK]" if success else "[FAIL]"
        print(f"  {status} {name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[SUCCESS] All tests passed! Code paths validated.")
        print("\nNext steps:")
        print("  1. Run synthesis experiment: python run_experiments.py --experiments synthesis")
        print("  2. Run ternary comparison: python run_experiments.py --experiments ternary")
        print("  3. Run full suite: python run_experiments.py --experiments all")
        return True
    else:
        print(f"\n[FAILURE] {total - passed} test(s) failed")
        return False

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
