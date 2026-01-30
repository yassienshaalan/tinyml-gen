"""
Example: How to Use New Rebuttal Components

This script demonstrates how to use each new component individually.
"""
import torch
import sys
from pathlib import Path

# Ensure imports work
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("REBUTTAL COMPONENTS USAGE EXAMPLES")
print("=" * 70)

# ============================================================================
# Example 1: Ternary Quantization Baseline
# ============================================================================
print("\n" + "─" * 70)
print("Example 1: Ternary Quantization Baseline")
print("─" * 70)

from ternary_baseline import TernarySeparableCNN

print("\nCreating ternary-quantized model...")
ternary_model = TernarySeparableCNN(in_ch=1, base=16, num_classes=2)

# Test forward pass
x = torch.randn(2, 1, 1800)
y = ternary_model(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {y.shape}")

# Compute flash memory usage
breakdown = ternary_model.compute_total_flash_bytes()
print(f"\nFlash Memory Breakdown:")
for component, bytes_used in breakdown.items():
    if component != 'total':
        print(f"  {component}: {bytes_used/1024:.2f} KB")
print(f"  {'='*30}")
print(f"  Total: {breakdown['total']/1024:.2f} KB")

print("\n✓ Ternary model working!")

# ============================================================================
# Example 2: NAS Compatibility
# ============================================================================
print("\n" + "─" * 70)
print("Example 2: NAS Compatibility")
print("─" * 70)

from nas_compatibility import NASInspiredBackbone, NASWithHyperTinyPW

print("\nCreating NAS-inspired architecture...")
# Define a custom NAS config: (out_channels, kernel_size, stride, expansion)
nas_config = [
    (16, 3, 1, 3),   # First stage
    (24, 5, 2, 6),   # Downsample
    (32, 5, 2, 6),   # Downsample
]

# Baseline NAS model
baseline_nas = NASInspiredBackbone(in_ch=1, num_classes=2, config=nas_config)
baseline_params = baseline_nas.count_params()
print(f"Baseline NAS model: {baseline_params:,} parameters ({baseline_params*4/1024:.2f} KB)")

# NAS + HyperTinyPW compression
compressed_nas = NASWithHyperTinyPW(
    in_ch=1, 
    num_classes=2, 
    nas_config=nas_config,
    latent_dim=16,
    compress_pw_layers=True
)

# Test forward pass
x = torch.randn(2, 1, 1800)
y1 = baseline_nas(x)
y2 = compressed_nas(x)
print(f"Baseline output: {y1.shape}")
print(f"Compressed output: {y2.shape}")

# Compression stats
stats = compressed_nas.compute_compression_stats()
print(f"\nCompression Statistics:")
print(f"  Original: {stats['original_kb']:.2f} KB")
print(f"  Compressed: {stats['compressed_kb']:.2f} KB")
print(f"  Ratio: {stats['compression_ratio']:.2f}x")
print(f"  Savings: {stats['savings_kb']:.2f} KB")
print(f"  Layers compressed: {stats['num_compressed_layers']}")

print("\n✓ NAS compatibility demonstrated!")

# ============================================================================
# Example 3: Synthesis Profiling
# ============================================================================
print("\n" + "─" * 70)
print("Example 3: Boot-Time Synthesis Profiling")
print("─" * 70)

from synthesis_profiler import SynthesisProfiler
import torch.nn as nn

print("\nCreating a dummy generator for profiling...")

class SimpleGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 256)
        )
        self.z = nn.Parameter(torch.randn(16))
    
    def forward(self):
        return self.net(self.z).view(16, 16, 1)

gen = SimpleGenerator()

# Generator function for profiling
def gen_fn():
    with torch.no_grad():
        return gen()

# Profile synthesis
profiler = SynthesisProfiler(device='cpu', warmup=2, repeats=10)
synth_time, synth_energy, weight_bytes = profiler.profile_synthesis(
    gen_fn, (16, 16, 1), "example_layer"
)

print(f"\nSynthesis Profiling Results:")
print(f"  Synthesis time: {synth_time:.3f} ms (one-shot at boot)")
print(f"  Synthesis energy: {synth_energy:.4f} mJ")
print(f"  Generated weights: {weight_bytes} bytes ({weight_bytes/1024:.2f} KB)")

# Create a dummy conv layer for inference profiling
dummy_conv = nn.Conv1d(16, 16, kernel_size=1)
dummy_input = torch.randn(2, 16, 100)

inf_time, inf_energy = profiler.profile_inference_layer(
    dummy_conv, dummy_input, "example_layer"
)

print(f"\nInference Profiling Results:")
print(f"  Inference time: {inf_time:.3f} ms (per run)")
print(f"  Inference energy: {inf_energy:.4f} mJ")

amortization = synth_time / inf_time
print(f"\nAmortization Analysis:")
print(f"  Break-even after: {amortization:.1f} inference runs")
print(f"  For always-on @1Hz: amortized in {amortization:.1f} seconds")

print("\n✓ Synthesis profiling complete!")

# ============================================================================
# Example 4: Speech Dataset Structure (without actual data)
# ============================================================================
print("\n" + "─" * 70)
print("Example 4: Speech Dataset Structure")
print("─" * 70)

from speech_dataset import TINYML_KEYWORDS, SPEECH_COMMANDS_CLASSES

print(f"\nKeyword Spotting Configuration:")
print(f"  TinyML keywords (12-class): {', '.join(TINYML_KEYWORDS[:5])}...")
print(f"  Full Speech Commands (35-class): {', '.join(SPEECH_COMMANDS_CLASSES[:5])}...")
print(f"\nTo use keyword spotting:")
print(f"  1. Download Speech Commands v0.02")
print(f"  2. Set SPEECH_COMMANDS_ROOT environment variable")
print(f"  3. Use load_keyword_spotting_wrapper()")

print("\n✓ Speech dataset structure defined!")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("""
All components are working correctly!

Next Steps:
  1. Run full experiments: python run_rebuttal_experiments.py --experiments all
  2. Or test individual components as shown above
  3. Integrate results into your paper

Documentation:
  - Quick Start: QUICKSTART_REBUTTAL.md
  - Full Guide: REBUTTAL_EXPERIMENTS.md
  - Implementation: REBUTTAL_IMPLEMENTATION_SUMMARY.md
""")

print("=" * 70)
print("All examples completed successfully!")
print("=" * 70)
