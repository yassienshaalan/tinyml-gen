"""
Boot-Time Synthesis Profiling Module
For rebuttal: measure one-shot synthesis cost vs. steady-state inference
"""
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import json


@dataclass
class SynthesisProfile:
    """Profile data for weight synthesis operation"""
    layer_name: str
    synthesis_time_ms: float
    synthesis_energy_mj: float  # Estimated in millijoules
    steady_inference_time_ms: float
    steady_inference_energy_mj: float
    weight_size_bytes: int
    generator_size_bytes: int
    compression_ratio: float
    sram_peak_bytes: int


class SynthesisProfiler:
    """
    Profile boot-time synthesis and steady-state inference separately.
    Provides detailed metrics for deployment-critical analysis.
    """
    def __init__(self, device='cuda', warmup=5, repeats=20):
        self.device = device
        self.warmup = warmup
        self.repeats = repeats
        self.profiles: List[SynthesisProfile] = []
        
        # Energy model constants (based on ARM Cortex-M7 @ 216MHz)
        # These are approximate values from literature
        self.ENERGY_PER_MAC_NJ = 0.3  # nanojoules per MAC
        self.ENERGY_PER_FLOP_NJ = 0.5  # nanojoules per FLOP (synthesis)
        self.SRAM_READ_ENERGY_NJ = 0.1  # per byte
        self.FLASH_READ_ENERGY_NJ = 0.05  # per byte
        
    def profile_synthesis(self, generator_fn, weight_shape, layer_name="layer"):
        """
        Profile a weight synthesis operation.
        
        Args:
            generator_fn: Callable that generates weights (no args)
            weight_shape: Expected output weight shape
            layer_name: Identifier for this layer
        """
        # Warmup
        for _ in range(self.warmup):
            _ = generator_fn()
        
        # Time synthesis
        if self.device == 'cuda':
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            for _ in range(self.repeats):
                weights = generator_fn()
            end.record()
            
            torch.cuda.synchronize()
            synthesis_time_ms = start.elapsed_time(end) / self.repeats
        else:
            start = time.perf_counter()
            for _ in range(self.repeats):
                weights = generator_fn()
            end = time.perf_counter()
            synthesis_time_ms = ((end - start) / self.repeats) * 1000
        
        # Estimate energy (based on operations count)
        weight_bytes = np.prod(weight_shape) * 4  # FP32
        # Rough estimate: synthesis involves ~10-20 FLOPs per output weight
        estimated_flops = np.prod(weight_shape) * 15
        synthesis_energy_mj = (estimated_flops * self.ENERGY_PER_FLOP_NJ) / 1e6
        
        return synthesis_time_ms, synthesis_energy_mj, weight_bytes
    
    def profile_inference_layer(self, layer, input_tensor, layer_name="layer"):
        """
        Profile inference time for a single layer.
        
        Args:
            layer: nn.Module to profile
            input_tensor: Input tensor
            layer_name: Identifier
        """
        layer.eval()
        with torch.no_grad():
            # Warmup
            for _ in range(self.warmup):
                _ = layer(input_tensor)
            
            # Time inference
            if self.device == 'cuda':
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                for _ in range(self.repeats):
                    output = layer(input_tensor)
                end.record()
                
                torch.cuda.synchronize()
                inference_time_ms = start.elapsed_time(end) / self.repeats
            else:
                start = time.perf_counter()
                for _ in range(self.repeats):
                    output = layer(input_tensor)
                end = time.perf_counter()
                inference_time_ms = ((end - start) / self.repeats) * 1000
        
        # Estimate energy (based on MAC operations for conv/linear)
        if isinstance(layer, (nn.Conv1d, nn.Conv2d)):
            # MACs = output_size * kernel_size * in_channels
            output_size = output.numel() / output.shape[0]  # per sample
            kernel_size = np.prod(layer.kernel_size) if hasattr(layer.kernel_size, '__iter__') else layer.kernel_size
            macs = output_size * kernel_size * layer.in_channels / layer.groups
        elif isinstance(layer, nn.Linear):
            macs = layer.in_features * layer.out_features
        else:
            macs = 0  # Approximate for other layers
        
        inference_energy_mj = (macs * self.ENERGY_PER_MAC_NJ) / 1e6
        
        return inference_time_ms, inference_energy_mj
    
    def profile_model_with_synthesis(self, model, input_shape, 
                                    synthesized_layers: Dict[str, Tuple]):
        """
        Full model profiling with synthesis overhead analysis.
        
        Args:
            model: Full model
            input_shape: Input tensor shape (B, C, L)
            synthesized_layers: Dict mapping layer_name -> (generator_fn, weight_shape, generator_bytes)
        """
        self.profiles = []
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)
        
        for layer_name, (generator_fn, weight_shape, gen_bytes) in synthesized_layers.items():
            print(f"Profiling {layer_name}...")
            
            # Profile synthesis
            synth_time, synth_energy, weight_bytes = self.profile_synthesis(
                generator_fn, weight_shape, layer_name
            )
            
            # Get the actual layer for inference profiling
            layer = self._get_layer_by_name(model, layer_name)
            if layer is None:
                print(f"  Warning: Could not find layer {layer_name}")
                continue
            
            # Profile inference
            # For this, we need appropriate input shape for the layer
            # Run forward up to this layer to get correct input
            layer_input = self._get_layer_input(model, layer_name, dummy_input)
            
            if layer_input is not None:
                inf_time, inf_energy = self.profile_inference_layer(
                    layer, layer_input, layer_name
                )
            else:
                inf_time, inf_energy = 0, 0
            
            # Compute compression ratio
            compression_ratio = weight_bytes / (gen_bytes + weight_bytes * 0.1)  # Rough estimate
            
            # Estimate SRAM peak (generator params + temporary weights)
            sram_peak = gen_bytes + weight_bytes
            
            profile = SynthesisProfile(
                layer_name=layer_name,
                synthesis_time_ms=synth_time,
                synthesis_energy_mj=synth_energy,
                steady_inference_time_ms=inf_time,
                steady_inference_energy_mj=inf_energy,
                weight_size_bytes=weight_bytes,
                generator_size_bytes=gen_bytes,
                compression_ratio=compression_ratio,
                sram_peak_bytes=sram_peak
            )
            
            self.profiles.append(profile)
            
            print(f"  Synthesis: {synth_time:.3f}ms, {synth_energy:.4f}mJ")
            print(f"  Inference: {inf_time:.3f}ms, {inf_energy:.4f}mJ")
            print(f"  Amortization: {synth_time/inf_time:.1f}x inference runs")
    
    def _get_layer_by_name(self, model, name):
        """Get layer by dotted name"""
        parts = name.split('.')
        module = model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None
        return module
    
    def _get_layer_input(self, model, layer_name, input_tensor):
        """Run forward pass up to layer to capture its input"""
        activations = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = input[0].detach()
            return hook
        
        # Register hook
        layer = self._get_layer_by_name(model, layer_name)
        if layer is None:
            return None
        
        handle = layer.register_forward_hook(hook_fn(layer_name))
        
        # Forward pass
        with torch.no_grad():
            _ = model(input_tensor)
        
        handle.remove()
        
        return activations.get(layer_name)
    
    def get_summary_table(self) -> str:
        """Generate markdown table summary"""
        lines = []
        lines.append("| Layer | Synthesis (ms) | Inference (ms) | Amortization | Energy Ratio | Compression |")
        lines.append("|-------|----------------|----------------|--------------|--------------|-------------|")
        
        for p in self.profiles:
            amort = p.synthesis_time_ms / max(0.001, p.steady_inference_time_ms)
            energy_ratio = p.synthesis_energy_mj / max(0.001, p.steady_inference_energy_mj)
            lines.append(
                f"| {p.layer_name} | {p.synthesis_time_ms:.3f} | "
                f"{p.steady_inference_time_ms:.3f} | {amort:.1f}x | "
                f"{energy_ratio:.1f}x | {p.compression_ratio:.2f}x |"
            )
        
        # Totals
        total_synth = sum(p.synthesis_time_ms for p in self.profiles)
        total_inf = sum(p.steady_inference_time_ms for p in self.profiles)
        total_synth_energy = sum(p.synthesis_energy_mj for p in self.profiles)
        total_inf_energy = sum(p.steady_inference_energy_mj for p in self.profiles)
        
        lines.append(
            f"| **Total** | **{total_synth:.3f}** | **{total_inf:.3f}** | "
            f"**{total_synth/max(0.001,total_inf):.1f}x** | "
            f"**{total_synth_energy/max(0.001,total_inf_energy):.1f}x** | - |"
        )
        
        return "\n".join(lines)
    
    def export_json(self, filepath: str):
        """Export profiles to JSON"""
        data = {
            'profiles': [
                {
                    'layer_name': p.layer_name,
                    'synthesis_time_ms': p.synthesis_time_ms,
                    'synthesis_energy_mj': p.synthesis_energy_mj,
                    'steady_inference_time_ms': p.steady_inference_time_ms,
                    'steady_inference_energy_mj': p.steady_inference_energy_mj,
                    'weight_size_bytes': p.weight_size_bytes,
                    'generator_size_bytes': p.generator_size_bytes,
                    'compression_ratio': p.compression_ratio,
                    'sram_peak_bytes': p.sram_peak_bytes,
                }
                for p in self.profiles
            ],
            'summary': {
                'total_synthesis_time_ms': sum(p.synthesis_time_ms for p in self.profiles),
                'total_inference_time_ms': sum(p.steady_inference_time_ms for p in self.profiles),
                'total_synthesis_energy_mj': sum(p.synthesis_energy_mj for p in self.profiles),
                'total_inference_energy_mj': sum(p.steady_inference_energy_mj for p in self.profiles),
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Exported profile to {filepath}")


def profile_hypertiny_model(model, input_shape=(1, 1, 1800), device='cuda'):
    """
    Convenience function to profile a HyperTinyPW-style model.
    Automatically detects synthesized layers.
    """
    profiler = SynthesisProfiler(device=device)
    
    # Auto-detect synthesized layers (those with generators)
    synthesized = {}
    
    # Look for generator modules
    if hasattr(model, 'gen') and hasattr(model, 'pw_head'):
        # Extract generator info
        gen_params = sum(p.numel() for p in model.gen.parameters())
        head_params = sum(p.numel() for p in model.pw_head.parameters())
        gen_bytes = (gen_params + head_params) * 4  # FP32
        
        # Find the synthesized PW layer
        # This is model-specific; adapt to your architecture
        if hasattr(model, 'last_pw_out') and hasattr(model, 'last_pw_in'):
            weight_shape = (model.last_pw_out, model.last_pw_in, 1)
            
            def generator_fn():
                with torch.no_grad():
                    h = model.gen()
                    w = model.pw_head(h)
                    return w.view(*weight_shape)
            
            synthesized['synthesized_pw'] = (generator_fn, weight_shape, gen_bytes)
    
    if not synthesized:
        print("Warning: No synthesized layers detected. Profiling skipped.")
        return profiler
    
    print(f"Detected {len(synthesized)} synthesized layer(s)")
    profiler.profile_model_with_synthesis(model, input_shape, synthesized)
    
    return profiler


if __name__ == '__main__':
    # Example usage
    print("Boot-Time Synthesis Profiling Module")
    print("=" * 60)
    
    # Dummy generator for testing
    class DummyGenerator(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(16, 64),
                nn.ReLU(),
                nn.Linear(64, 512)
            )
            self.z = nn.Parameter(torch.randn(16))
        
        def forward(self):
            return self.net(self.z)
    
    gen = DummyGenerator()
    
    def gen_fn():
        with torch.no_grad():
            return gen().view(32, 16, 1)
    
    profiler = SynthesisProfiler(device='cpu')
    
    # Profile synthesis
    synth_time, synth_energy, weight_bytes = profiler.profile_synthesis(
        gen_fn, (32, 16, 1), "dummy_layer"
    )
    
    print(f"\nSynthesis: {synth_time:.3f}ms, {synth_energy:.4f}mJ")
    print(f"Weight size: {weight_bytes} bytes ({weight_bytes/1024:.2f} KB)")
