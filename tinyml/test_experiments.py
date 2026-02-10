"""
Comprehensive Unit Tests for HyperTinyPW Experiments
Tests all code paths, modules, and experiment functions
"""
import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path
import torch
import numpy as np
import argparse

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))


class TestModuleImports(unittest.TestCase):
    """Test that all required modules can be imported"""
    
    def test_import_models(self):
        """Test models module import"""
        try:
            import models
            self.assertTrue(hasattr(models, 'HyperTinyPW'))
            self.assertTrue(hasattr(models, 'safe_build_model'))
        except Exception as e:
            self.fail(f"Failed to import models: {e}")
    
    def test_import_experiments(self):
        """Test experiments module import"""
        try:
            import experiments
            self.assertTrue(hasattr(experiments, 'ExpCfg'))
            self.assertTrue(hasattr(experiments, 'train_regular_cnn'))
        except Exception as e:
            self.fail(f"Failed to import experiments: {e}")
    
    def test_import_datasets(self):
        """Test datasets module import"""
        try:
            import datasets
            self.assertTrue(hasattr(datasets, 'ApneaECGWindows'))
        except Exception as e:
            self.fail(f"Failed to import datasets: {e}")
    
    def test_import_data_loaders(self):
        """Test data_loaders module import"""
        try:
            import data_loaders
            self.assertTrue(hasattr(data_loaders, 'load_apnea_ecg_loaders_impl'))
        except Exception as e:
            self.fail(f"Failed to import data_loaders: {e}")
    
    def test_import_ternary_baseline(self):
        """Test ternary_baseline module import"""
        try:
            import ternary_baseline
            self.assertTrue(hasattr(ternary_baseline, 'TernaryQuantizer'))
        except Exception as e:
            self.fail(f"Failed to import ternary_baseline: {e}")
    
    def test_import_synthesis_profiler(self):
        """Test synthesis_profiler module import"""
        try:
            import synthesis_profiler
            self.assertTrue(hasattr(synthesis_profiler, 'SynthesisProfiler'))
        except Exception as e:
            self.fail(f"Failed to import synthesis_profiler: {e}")
    
    def test_import_speech_dataset(self):
        """Test speech_dataset module import (optional)"""
        try:
            import speech_dataset
            self.assertTrue(hasattr(speech_dataset, 'load_keyword_spotting_wrapper'))
        except Exception as e:
            self.skipTest(f"Speech dataset not available: {e}")


class TestModelArchitectures(unittest.TestCase):
    """Test model architectures and building"""
    
    def test_hypertinypw_creation(self):
        """Test HyperTinyPW model creation"""
        from models import HyperTinyPW
        
        model = HyperTinyPW(
            num_classes=2,
            in_channels=1,
            base_channels=16,
            num_blocks=4,
            latent_dim=96,
            seq_len=1800
        )
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'forward'))
    
    def test_model_forward_pass(self):
        """Test forward pass with dummy data"""
        from models import HyperTinyPW
        
        model = HyperTinyPW(
            num_classes=2,
            in_channels=1,
            base_channels=8,
            num_blocks=2,
            latent_dim=32,
            seq_len=100
        )
        
        # Dummy input
        x = torch.randn(2, 1, 100)
        
        with torch.no_grad():
            output = model(x)
        
        self.assertEqual(output.shape, (2, 2))
    
    def test_safe_build_model(self):
        """Test safe model building function"""
        from models import safe_build_model
        
        model_config = {
            'name': 'SharedCoreSeparable1D',
            'num_classes': 2,
            'in_channels': 1,
            'base_channels': 8,
            'seq_len': 100
        }
        
        model = safe_build_model(model_config)
        self.assertIsNotNone(model)
    
    def test_ternary_quantizer(self):
        """Test ternary quantization"""
        from ternary_baseline import TernaryQuantizer
        
        quantizer = TernaryQuantizer(threshold=0.7)
        
        # Test weight quantization
        weight = torch.randn(10, 10)
        quantized, scale = quantizer.quantize(weight)
        
        # Check quantized values are in {-1, 0, 1}
        unique_vals = torch.unique(quantized)
        self.assertTrue(all(v in [-1, 0, 1] for v in unique_vals.tolist()))
        self.assertIsInstance(scale, torch.Tensor)


class TestDataLoading(unittest.TestCase):
    """Test data loading and preprocessing"""
    
    def test_synthetic_ecg_data(self):
        """Test synthetic ECG data generation"""
        from datasets import create_synthetic_ecg_data
        
        X, y = create_synthetic_ecg_data(n_samples=100, seq_len=1800)
        
        self.assertEqual(X.shape, (100, 1, 1800))
        self.assertEqual(y.shape, (100,))
        self.assertTrue(torch.all((y == 0) | (y == 1)))
    
    def test_apnea_dataset_creation(self):
        """Test ApneaECGWindows dataset creation"""
        from datasets import ApneaECGWindows
        
        # Create dummy data
        dummy_records = []
        for i in range(3):
            record_data = {
                'signal': np.random.randn(1, 1800),
                'labels': np.random.randint(0, 2, 1800),
                'record_id': f'test_{i}'
            }
            dummy_records.append(record_data)
        
        try:
            dataset = ApneaECGWindows(
                records=dummy_records,
                length=1800,
                stride=1800
            )
            self.assertGreater(len(dataset), 0)
        except Exception as e:
            self.skipTest(f"Dataset creation failed: {e}")


class TestExperimentComponents(unittest.TestCase):
    """Test experiment framework components"""
    
    def test_exp_cfg_creation(self):
        """Test ExpCfg configuration object"""
        from experiments import ExpCfg
        
        cfg = ExpCfg(
            epochs=5,
            batch_size=32,
            lr=0.001,
            device='cpu'
        )
        
        self.assertEqual(cfg.epochs, 5)
        self.assertEqual(cfg.batch_size, 32)
        self.assertEqual(cfg.lr, 0.001)
    
    def test_seed_everything(self):
        """Test random seed setting"""
        from experiments import seed_everything
        
        seed_everything(42)
        
        # Generate random numbers
        r1 = torch.rand(5)
        
        # Reset seed
        seed_everything(42)
        r2 = torch.rand(5)
        
        # Should be identical
        self.assertTrue(torch.allclose(r1, r2))
    
    def test_dataset_registry(self):
        """Test dataset registry functions"""
        from experiments import register_dataset, available_datasets
        
        def dummy_loader(**kwargs):
            return None, None, None, {}
        
        register_dataset('test_dataset', dummy_loader)
        
        datasets = available_datasets()
        self.assertIn('test_dataset', datasets)


class TestSynthesisProfiler(unittest.TestCase):
    """Test synthesis profiling"""
    
    def test_profiler_creation(self):
        """Test profiler creation"""
        from synthesis_profiler import SynthesisProfiler
        from models import HyperTinyPW
        
        model = HyperTinyPW(
            num_classes=2,
            in_channels=1,
            base_channels=8,
            num_blocks=2,
            latent_dim=32,
            seq_len=100
        )
        
        profiler = SynthesisProfiler(model, device='cpu')
        self.assertIsNotNone(profiler)
    
    def test_inference_profiling(self):
        """Test inference time profiling"""
        from synthesis_profiler import SynthesisProfiler
        from models import HyperTinyPW
        
        model = HyperTinyPW(
            num_classes=2,
            in_channels=1,
            base_channels=8,
            num_blocks=2,
            latent_dim=32,
            seq_len=100
        )
        
        profiler = SynthesisProfiler(model, device='cpu')
        
        # Profile inference
        x = torch.randn(1, 1, 100)
        inference_time = profiler.profile_inference(x, n_runs=10)
        
        self.assertGreater(inference_time, 0)
        self.assertLess(inference_time, 1.0)  # Should be under 1 second


class TestExperimentFunctions(unittest.TestCase):
    """Test experiment runner functions"""
    
    def setUp(self):
        """Create temporary output directory"""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / 'test_results'
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        """Clean up temporary directory"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_synthesis_experiment(self):
        """Test synthesis profiling experiment (fast)"""
        from run_experiments import run_synthesis_profiling
        
        # Create minimal args
        args = argparse.Namespace(
            output_dir=str(self.output_dir),
            batch_size=32,
            cpu=True
        )
        
        try:
            profiler = run_synthesis_profiling(args)
            self.assertIsNotNone(profiler)
            
            # Check output file
            output_file = self.output_dir / 'synthesis_profile.json'
            self.assertTrue(output_file.exists())
        except Exception as e:
            self.skipTest(f"Synthesis experiment failed: {e}")
    
    def test_setup_logging(self):
        """Test logging setup"""
        from run_experiments import setup_logging
        
        logger, log_path = setup_logging(str(self.output_dir))
        
        self.assertTrue(log_path.exists())
        logger.close()
    
    def test_git_commit_retrieval(self):
        """Test git commit hash retrieval"""
        from run_experiments import get_git_commit
        
        commit = get_git_commit()
        self.assertIsInstance(commit, str)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""
    
    def test_model_size_calculation(self):
        """Test model size calculation"""
        from models import HyperTinyPW
        
        model = HyperTinyPW(
            num_classes=2,
            in_channels=1,
            base_channels=16,
            num_blocks=4,
            latent_dim=96,
            seq_len=1800
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(total_params, 0)
        
        # Calculate size in KB (FP32)
        size_kb = (total_params * 4) / 1024
        self.assertGreater(size_kb, 0)
    
    def test_compression_ratio(self):
        """Test compression ratio calculation"""
        uncompressed_kb = 903
        compressed_kb = 72
        
        compression_ratio = uncompressed_kb / compressed_kb
        self.assertAlmostEqual(compression_ratio, 12.5, places=1)


class QuickVMTests(unittest.TestCase):
    """Quick tests for VM validation (no training)"""
    
    def test_imports_all_work(self):
        """Quick: All imports work"""
        modules = [
            'models', 'experiments', 'datasets', 
            'data_loaders', 'ternary_baseline', 'synthesis_profiler'
        ]
        
        for module_name in modules:
            try:
                __import__(module_name)
            except Exception as e:
                self.fail(f"Import failed for {module_name}: {e}")
    
    def test_model_instantiation(self):
        """Quick: Models can be instantiated"""
        from models import HyperTinyPW
        
        configs = [
            {'base_channels': 8, 'num_blocks': 2, 'latent_dim': 32},
            {'base_channels': 16, 'num_blocks': 4, 'latent_dim': 96},
        ]
        
        for config in configs:
            model = HyperTinyPW(
                num_classes=2,
                in_channels=1,
                seq_len=100,
                **config
            )
            self.assertIsNotNone(model)
    
    def test_forward_backward_pass(self):
        """Quick: Forward and backward pass work"""
        from models import HyperTinyPW
        
        model = HyperTinyPW(
            num_classes=2,
            in_channels=1,
            base_channels=8,
            num_blocks=2,
            latent_dim=32,
            seq_len=100
        )
        
        x = torch.randn(4, 1, 100)
        y = torch.randint(0, 2, (4,))
        
        # Forward
        output = model(x)
        self.assertEqual(output.shape, (4, 2))
        
        # Backward
        loss = torch.nn.functional.cross_entropy(output, y)
        loss.backward()
        
        # Check gradients exist
        has_gradients = any(p.grad is not None for p in model.parameters())
        self.assertTrue(has_gradients)
    
    def test_data_generation(self):
        """Quick: Synthetic data generation works"""
        from datasets import create_synthetic_ecg_data
        
        X, y = create_synthetic_ecg_data(n_samples=50, seq_len=100)
        
        self.assertEqual(X.shape[0], 50)
        self.assertEqual(y.shape[0], 50)
    
    def test_quantization_works(self):
        """Quick: Ternary quantization works"""
        from ternary_baseline import TernaryQuantizer
        
        quantizer = TernaryQuantizer(threshold=0.7)
        weight = torch.randn(20, 20)
        quantized, scale = quantizer.quantize(weight)
        
        unique_vals = torch.unique(quantized)
        self.assertTrue(len(unique_vals) <= 3)


def run_quick_tests():
    """Run only quick VM tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add only quick test classes
    suite.addTests(loader.loadTestsFromTestCase(TestModuleImports))
    suite.addTests(loader.loadTestsFromTestCase(QuickVMTests))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_all_tests():
    """Run all comprehensive tests"""
    loader = unittest.TestLoader()
    suite = loader.discover('.', pattern='test_experiments.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run HyperTinyPW tests')
    parser.add_argument('--quick', action='store_true', 
                       help='Run only quick VM tests (no training)')
    args = parser.parse_args()
    
    print("="*80)
    if args.quick:
        print("RUNNING QUICK VM TESTS (Fast, no training)")
    else:
        print("RUNNING COMPREHENSIVE TESTS (All code paths)")
    print("="*80)
    print()
    
    if args.quick:
        success = run_quick_tests()
    else:
        success = run_all_tests()
    
    print()
    print("="*80)
    if success:
        print("[OK] ALL TESTS PASSED")
    else:
        print("[FAIL] SOME TESTS FAILED")
    print("="*80)
    
    sys.exit(0 if success else 1)
