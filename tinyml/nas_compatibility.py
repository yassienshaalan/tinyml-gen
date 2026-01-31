"""
NAS Compatibility Experiment
For rebuttal: demonstrate that HyperTinyPW can be applied to NAS-derived architectures
Shows orthogonality to search-based methods like MCUNet/Once-for-All

VERSION: Fixed compression calculation (commit 0767909)
BUG FIX: Removed double-counting of generator overhead in compressed parameter count
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict
from models import SharedPWGenerator, PWHead, DepthwiseSeparable1D


class NASInspiredBackbone(nn.Module):
    """
    Simplified NAS-like architecture (inspired by MCUNet/MobileNetV3).
    Variable kernel sizes, expansion ratios, and layer depths.
    """
    def __init__(self, in_ch=1, num_classes=2, config: List[Tuple] = None):
        """
        Args:
            config: List of (out_ch, kernel_size, stride, expansion) tuples
                   If None, uses a default MCUNet-inspired config
        """
        super().__init__()
        
        if config is None:
            # Default NAS-inspired config
            # Format: (out_channels, kernel_size, stride, expansion_ratio)
            config = [
                (16, 3, 1, 1),   # Initial expansion
                (24, 5, 2, 6),   # Expanded bottleneck
                (24, 3, 1, 6),   # Repeated
                (32, 5, 2, 6),   # Downsample
                (32, 3, 1, 6),   # Repeated
                (48, 5, 2, 6),   # Final stage
            ]
        
        self.config = config
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU6(inplace=True)
        )
        
        # Build blocks according to config
        self.blocks = nn.ModuleList()
        in_channels = 16
        
        for out_ch, k, stride, exp in config:
            self.blocks.append(
                InvertedResidualBlock(
                    in_channels, out_ch, k, stride, exp
                )
            )
            in_channels = out_ch
        
        # Head
        self.final_channels = in_channels
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(self.final_channels, num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.gap(x)
        x = x.flatten(1)
        x = self.classifier(x)
        
        return x
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def get_pw_layers(self) -> List[nn.Conv1d]:
        """Extract all 1x1 pointwise convolutions for analysis"""
        pw_layers = []
        for module in self.modules():
            if isinstance(module, nn.Conv1d) and module.kernel_size[0] == 1:
                pw_layers.append(module)
        return pw_layers


class InvertedResidualBlock(nn.Module):
    """
    Mobile Inverted Bottleneck (MBConv) block.
    Used in NAS architectures like MCUNet, EfficientNet.
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, expansion=6):
        super().__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_ch == out_ch)
        
        hidden_dim = int(in_ch * expansion)
        
        layers = []
        
        # Expansion (pointwise)
        if expansion != 1:
            layers.extend([
                nn.Conv1d(in_ch, hidden_dim, 1, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        
        # Depthwise
        layers.extend([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, stride, 
                     padding=kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU6(inplace=True)
        ])
        
        # Projection (pointwise)
        layers.extend([
            nn.Conv1d(hidden_dim, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class NASWithHyperTinyPW(nn.Module):
    """
    NAS-derived backbone with HyperTinyPW applied to pointwise layers.
    Demonstrates orthogonality: NAS finds architecture, HyperTinyPW compresses it.
    """
    def __init__(self, in_ch=1, num_classes=2, nas_config=None, 
                 latent_dim=16, compress_pw_layers=True):
        """
        Args:
            nas_config: NAS-derived architecture config
            latent_dim: Dimension of shared latent code for weight generation
            compress_pw_layers: If True, replace PW layers with synthesized weights
        """
        super().__init__()
        
        # Build base NAS architecture
        self.backbone = NASInspiredBackbone(in_ch, num_classes, nas_config)
        
        self.compress_pw = compress_pw_layers
        
        if compress_pw_layers:
            # Identify and prepare to synthesize PW layers
            pw_layers = self.backbone.get_pw_layers()
            print(f"Found {len(pw_layers)} pointwise layers to compress")
            
            # For simplicity, we'll compress the largest PW layers
            # In practice, you'd compress all or strategically select
            self.compressed_pw_specs = []
            
            for i, pw in enumerate(pw_layers):
                if pw.weight.numel() > 256:  # Only compress large enough layers
                    spec = {
                        'layer_idx': i,
                        'in_ch': pw.in_channels,
                        'out_ch': pw.out_channels,
                        'weight_size': pw.weight.numel()
                    }
                    self.compressed_pw_specs.append(spec)
            
            # Create shared generator
            if self.compressed_pw_specs:
                self.gen = SharedPWGenerator(z_dim=latent_dim, hidden=96)
                
                # Create heads for each compressed layer
                self.pw_heads = nn.ModuleList([
                    PWHead(h_dim=96, flat_out=spec['weight_size'])
                    for spec in self.compressed_pw_specs
                ])
                
                print(f"Compressing {len(self.compressed_pw_specs)} PW layers with shared generator")
    
    def forward(self, x):
        # If compression is enabled, synthesize weights at forward time
        # (In deployment, this would be done once at boot)
        if self.compress_pw and hasattr(self, 'gen'):
            self._synthesize_weights()
        
        return self.backbone(x)
    
    def _synthesize_weights(self):
        """Synthesize weights for compressed PW layers"""
        with torch.no_grad():
            h = self.gen()
            
            # Generate weights for each compressed layer
            pw_layers = self.backbone.get_pw_layers()
            
            for spec, head in zip(self.compressed_pw_specs, self.pw_heads):
                w_flat = head(h)
                w = w_flat.view(spec['out_ch'], spec['in_ch'], 1)
                
                # Update the actual layer's weights
                layer = pw_layers[spec['layer_idx']]
                layer.weight.data.copy_(w)
    
    def compute_compression_stats(self) -> Dict:
        """Compute compression ratio and memory savings"""
        stats = {}
        
        # Original model size
        orig_params = self.backbone.count_params()
        orig_bytes = orig_params * 4  # FP32
        
        if self.compress_pw and hasattr(self, 'gen'):
            # Size of generator + heads (amortized overhead)
            gen_params = sum(p.numel() for p in self.gen.parameters())
            head_params = sum(sum(p.numel() for p in head.parameters()) 
                            for head in self.pw_heads)
            
            # Total params in PW layers that we're compressing
            pw_layer_params = sum(spec['weight_size'] for spec in self.compressed_pw_specs)
            
            # Size of uncompressed parts (DW, BN, classifier, etc.)
            other_params = orig_params - pw_layer_params
            
            # Compressed size = generator + heads + non-PW layers
            # PW layers are synthesized, so we don't store them
            compressed_params = gen_params + head_params + other_params
            compressed_bytes = compressed_params * 4  # FP32
            
            stats['original_kb'] = orig_bytes / 1024
            stats['compressed_kb'] = compressed_bytes / 1024
            stats['compression_ratio'] = orig_bytes / compressed_bytes
            stats['savings_kb'] = (orig_bytes - compressed_bytes) / 1024
            stats['num_compressed_layers'] = len(self.compressed_pw_specs)
            stats['pw_params_removed'] = pw_layer_params
            stats['generator_params'] = gen_params + head_params
        else:
            stats['original_kb'] = orig_bytes / 1024
            stats['compressed_kb'] = orig_bytes / 1024
            stats['compression_ratio'] = 1.0
            stats['savings_kb'] = 0
            stats['num_compressed_layers'] = 0
        
        return stats


def experiment_nas_compatibility(device='cuda'):
    """
    Run experiment showing HyperTinyPW applied to NAS architectures.
    """
    print("=" * 80)
    print("NAS Compatibility Experiment")
    print("=" * 80)
    
    # Define several NAS-inspired configs
    configs = {
        'mcunet_tiny': [
            (16, 3, 1, 3),
            (24, 5, 2, 4),
            (32, 5, 2, 4),
            (48, 3, 1, 6),
        ],
        'mcunet_medium': [
            (16, 3, 1, 4),
            (24, 5, 2, 6),
            (24, 3, 1, 6),
            (32, 5, 2, 6),
            (32, 5, 1, 6),
            (48, 5, 2, 6),
        ],
        'efficient_tiny': [
            (16, 3, 1, 1),
            (24, 3, 2, 4),
            (32, 5, 2, 4),
            (48, 3, 1, 4),
        ]
    }
    
    results = []
    
    for name, config in configs.items():
        print(f"\n{name.upper()}")
        print("-" * 80)
        
        # Baseline NAS architecture
        baseline = NASInspiredBackbone(in_ch=1, num_classes=2, config=config)
        baseline_params = baseline.count_params()
        baseline_kb = baseline_params * 4 / 1024
        
        print(f"Baseline: {baseline_params:,} params ({baseline_kb:.2f} KB)")
        
        # NAS + HyperTinyPW
        compressed = NASWithHyperTinyPW(
            in_ch=1, num_classes=2, nas_config=config,
            latent_dim=16, compress_pw_layers=True
        )
        
        stats = compressed.compute_compression_stats()
        
        print(f"With HyperTinyPW:")
        print(f"  Compressed: {stats['compressed_kb']:.2f} KB")
        print(f"  Compression ratio: {stats['compression_ratio']:.2f}x")
        print(f"  Savings: {stats['savings_kb']:.2f} KB")
        print(f"  Layers compressed: {stats['num_compressed_layers']}")
        
        results.append({
            'config': name,
            'baseline_kb': baseline_kb,
            'compressed_kb': stats['compressed_kb'],
            'ratio': stats['compression_ratio'],
            'savings_kb': stats['savings_kb']
        })
        
        # Test forward pass
        x = torch.randn(2, 1, 1800).to(device)
        baseline = baseline.to(device)
        compressed = compressed.to(device)
        
        with torch.no_grad():
            y1 = baseline(x)
            y2 = compressed(x)
        
        print(f"  Forward pass: baseline {y1.shape}, compressed {y2.shape} ✓")
    
    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Config':<20} {'Baseline (KB)':<15} {'Compressed (KB)':<17} {'Ratio':<10} {'Savings (KB)'}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['config']:<20} {r['baseline_kb']:<15.2f} {r['compressed_kb']:<17.2f} "
              f"{r['ratio']:<10.2f}x {r['savings_kb']:.2f}")
    
    avg_ratio = np.mean([r['ratio'] for r in results])
    print("-" * 80)
    print(f"Average compression ratio: {avg_ratio:.2f}x")
    
    return results


if __name__ == '__main__':
    # Run NAS compatibility experiment
    results = experiment_nas_compatibility(device='cpu')
    
    print("\n✓ Experiment complete!")
    print("\nKey Insight:")
    print("HyperTinyPW is orthogonal to NAS - it compresses the PW layers in")
    print("whatever architecture NAS finds, without modifying the search process.")
