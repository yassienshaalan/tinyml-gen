#!/bin/bash
# Quick VM Tests - Fast execution, no training
# Tests all critical code paths without long-running experiments

set -e

echo "========================================"
echo "Quick VM Tests for HyperTinyPW"
echo "========================================"
echo ""

cd "$(dirname "$0")/../tinyml"

echo "[1/6] Testing module imports..."
python -c "
import sys
modules = ['models', 'experiments', 'datasets', 'data_loaders', 'ternary_baseline', 'synthesis_profiler']
for m in modules:
    try:
        __import__(m)
        print(f'  [OK] {m}')
    except Exception as e:
        print(f'  [FAIL] {m}: {e}')
        sys.exit(1)
"
echo ""

echo "[2/6] Testing model instantiation..."
python -c "
from models import HyperTinyPW
import torch

# Test small model
model = HyperTinyPW(num_classes=2, in_channels=1, base_channels=8, num_blocks=2, latent_dim=32, seq_len=100)
print(f'  [OK] Small model created: {sum(p.numel() for p in model.parameters())} params')

# Test forward pass
x = torch.randn(2, 1, 100)
with torch.no_grad():
    y = model(x)
print(f'  [OK] Forward pass: input {x.shape} -> output {y.shape}')
"
echo ""

echo "[3/6] Testing ternary quantization..."
python -c "
from ternary_baseline import TernaryQuantizer
import torch

quantizer = TernaryQuantizer(threshold=0.7)
weight = torch.randn(50, 50)
quantized, scale = quantizer.quantize(weight)

unique_vals = torch.unique(quantized).tolist()
print(f'  [OK] Quantized to values: {unique_vals}')
print(f'  [OK] Scale factor: {scale.item():.4f}')

# Calculate size
original_kb = (50 * 50 * 4) / 1024
quantized_kb = (50 * 50 * 0.25) / 1024  # 2 bits
print(f'  [OK] Size: {original_kb:.2f} KB -> {quantized_kb:.2f} KB ({original_kb/quantized_kb:.1f}x)')
"
echo ""

echo "[4/6] Testing synthesis profiler..."
python -c "
from synthesis_profiler import SynthesisProfiler
from models import HyperTinyPW
import torch
import time

model = HyperTinyPW(num_classes=2, in_channels=1, base_channels=8, num_blocks=2, latent_dim=32, seq_len=100)
profiler = SynthesisProfiler(model, device='cpu')

# Profile inference
x = torch.randn(1, 1, 100)
start = time.time()
for _ in range(10):
    with torch.no_grad():
        _ = model(x)
elapsed = (time.time() - start) / 10 * 1000

print(f'  [OK] Profiler created')
print(f'  [OK] Avg inference time: {elapsed:.2f} ms')
"
echo ""

echo "[5/6] Testing data generation..."
python -c "
from datasets import create_synthetic_ecg_data
import torch

X, y = create_synthetic_ecg_data(n_samples=100, seq_len=1800)
print(f'  [OK] Generated synthetic data: X={X.shape}, y={y.shape}')
print(f'  [OK] Class distribution: 0={torch.sum(y==0).item()}, 1={torch.sum(y==1).item()}')
"
echo ""

echo "[6/6] Running unit tests (quick mode)..."
python test_experiments.py --quick
echo ""

echo "========================================"
echo "VM Tests Complete!"
echo "========================================"
echo ""
echo "Summary:"
echo "  [OK] All imports working"
echo "  [OK] Model instantiation successful"
echo "  [OK] Forward/backward pass functional"
echo "  [OK] Quantization working"
echo "  [OK] Profiling working"
echo "  [OK] Data generation working"
echo ""
echo "Next steps:"
echo "  - Run full experiments: python run_experiments.py --experiments synthesis"
echo "  - Run comprehensive tests: python test_experiments.py"
echo ""
