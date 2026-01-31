"""
Ternary Weight Quantization Baseline
For rebuttal: comparison to aggressive quantization approaches under matched flash budgets

VERSION: Fixed compression calculation (commit 176cd90)
BUG FIX: Corrected ternary compression factor from 0.8 to 0.08 (12x compression for 2-bit)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


def ternary_quantize_weights(weights: torch.Tensor, threshold: float = 0.7) -> torch.Tensor:
    """
    Quantize weights to {-1, 0, +1} using threshold-based ternarization.
    
    Args:
        weights: Input weight tensor
        threshold: Relative threshold (0-1). Values with |w| > threshold*|w_max| → ±1, else 0
    
    Returns:
        Ternary weights {-1, 0, +1}
    """
    with torch.no_grad():
        # Compute threshold
        w_abs = weights.abs()
        w_max = w_abs.max()
        thresh_val = threshold * w_max
        
        # Ternarize: |w| > thresh → sign(w), else 0
        ternary = torch.zeros_like(weights)
        ternary[weights > thresh_val] = 1.0
        ternary[weights < -thresh_val] = -1.0
        
        return ternary


def compute_ternary_scale(weights: torch.Tensor, ternary: torch.Tensor) -> float:
    """
    Compute optimal scale factor for ternary weights to minimize reconstruction error.
    scale = (W^T * T) / (T^T * T)
    """
    with torch.no_grad():
        numerator = (weights * ternary).sum()
        denominator = (ternary * ternary).sum()
        if denominator == 0:
            return 1.0
        return (numerator / denominator).item()


class TernaryConv1d(nn.Module):
    """
    Conv1d with ternary weights {-1, 0, +1}.
    Stores: ternary values (2 bits per weight) + 1 scale per filter
    """
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, groups=1, bias=False,
                 threshold=0.7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.threshold = threshold
        
        # Full-precision weights for training
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size))
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Ternary weights and scales (computed on-the-fly during training, cached during eval)
        self.register_buffer('ternary_weight', None)
        self.register_buffer('scales', None)
        
    def ternarize(self):
        """Compute ternary weights and per-filter scales"""
        with torch.no_grad():
            ternary = torch.zeros_like(self.weight)
            scales = torch.zeros(self.out_channels, device=self.weight.device)
            
            # Per-filter ternarization
            for i in range(self.out_channels):
                w_filter = self.weight[i]
                t_filter = ternary_quantize_weights(w_filter, self.threshold)
                scale = compute_ternary_scale(w_filter, t_filter)
                
                ternary[i] = t_filter
                scales[i] = scale
            
            return ternary, scales
    
    def forward(self, x):
        if self.training:
            # Straight-through estimator: forward with ternary, backward with full-precision
            ternary, scales = self.ternarize()
            # Apply scales
            ternary_scaled = ternary * scales.view(-1, 1, 1)
            
            # STE: detach ternary, keep gradient path through original weight
            ternary_scaled = ternary_scaled.detach() + self.weight - self.weight.detach()
            
            return F.conv1d(x, ternary_scaled, self.bias, self.stride, self.padding, groups=self.groups)
        else:
            # Use cached ternary weights
            if self.ternary_weight is None:
                self.ternary_weight, self.scales = self.ternarize()
            
            ternary_scaled = self.ternary_weight * self.scales.view(-1, 1, 1)
            return F.conv1d(x, ternary_scaled, self.bias, self.stride, self.padding, groups=self.groups)
    
    def compute_flash_bytes(self) -> int:
        """
        Compute flash memory usage:
        - Ternary weights: 2 bits per weight (packed)
        - Scales: 4 bytes (FP32) per output channel
        """
        num_weights = self.out_channels * (self.in_channels // self.groups) * self.kernel_size
        ternary_bytes = (num_weights * 2 + 7) // 8  # 2 bits per weight, packed
        scale_bytes = self.out_channels * 4  # FP32 scales
        bias_bytes = self.out_channels * 4 if self.bias is not None else 0
        return ternary_bytes + scale_bytes + bias_bytes


class TernarySeparableBlock(nn.Module):
    """
    Depthwise-separable block with ternary pointwise conv.
    Depthwise: standard INT8 (small footprint due to groups)
    Pointwise: ternary quantization (major compression target)
    """
    def __init__(self, in_ch, out_ch, k=5, stride=1, padding=None, 
                 use_residual=True, ternary_threshold=0.7):
        super().__init__()
        padding = (k // 2) if padding is None else padding
        self.use_residual = use_residual and (in_ch == out_ch) and (stride == 1)
        
        # Depthwise: keep full precision (or INT8) - small anyway due to groups
        self.dw = nn.Conv1d(in_ch, in_ch, kernel_size=k, stride=stride, 
                           padding=padding, groups=in_ch, bias=False)
        
        # Pointwise: ternary quantization
        self.pw = TernaryConv1d(in_ch, out_ch, kernel_size=1, 
                                threshold=ternary_threshold, bias=False)
        
        self.bn1 = nn.BatchNorm1d(in_ch)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x
        
        # Depthwise
        out = self.dw(x)
        out = self.bn1(out)
        out = self.act(out)
        
        # Ternary pointwise
        out = self.pw(out)
        out = self.bn2(out)
        
        if self.use_residual:
            out = out + identity
        
        return self.act(out)
    
    def compute_flash_bytes(self) -> int:
        """Total flash usage for this block"""
        # Depthwise: assume INT8
        dw_weights = self.dw.in_channels * self.dw.kernel_size[0]
        dw_bytes = dw_weights  # INT8
        
        # Pointwise: ternary
        pw_bytes = self.pw.compute_flash_bytes()
        
        # BatchNorm: 2 params per channel (gamma, beta) in FP32
        bn_bytes = (self.bn1.num_features + self.bn2.num_features) * 2 * 4
        
        return dw_bytes + pw_bytes + bn_bytes


class TernarySeparableCNN(nn.Module):
    """
    Baseline separable CNN with ternary pointwise convolutions.
    For fair comparison under matched flash budgets.
    """
    def __init__(self, in_ch=1, base=16, num_classes=2, 
                 ternary_threshold=0.7, input_length=1800):
        super().__init__()
        self.base = base
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2)
        )
        
        # Ternary separable blocks
        self.blocks = nn.ModuleList([
            TernarySeparableBlock(base, base*2, k=5, stride=2, 
                                 use_residual=False, ternary_threshold=ternary_threshold),
            TernarySeparableBlock(base*2, base*2, k=5, stride=1, 
                                 use_residual=True, ternary_threshold=ternary_threshold),
            TernarySeparableBlock(base*2, base*4, k=5, stride=2, 
                                 use_residual=False, ternary_threshold=ternary_threshold),
            TernarySeparableBlock(base*4, base*4, k=5, stride=1, 
                                 use_residual=True, ternary_threshold=ternary_threshold),
        ])
        
        # Global pooling and classifier
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base*4, num_classes)
        
    def forward(self, x):
        x = self.stem(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.gap(x)
        x = x.flatten(1)
        x = self.fc(x)
        
        return x
    
    def compute_total_flash_bytes(self) -> dict:
        """
        Compute total flash memory usage broken down by component.
        Returns dict with detailed breakdown.
        """
        breakdown = {}
        
        # Stem
        stem_conv = self.stem[0]
        stem_bytes = (stem_conv.out_channels * stem_conv.in_channels * 
                     stem_conv.kernel_size[0])  # Assume INT8
        stem_bn = 2 * stem_conv.out_channels * 4  # BN params
        breakdown['stem'] = stem_bytes + stem_bn
        
        # Blocks
        for i, block in enumerate(self.blocks):
            breakdown[f'block_{i}'] = block.compute_flash_bytes()
        
        # Classifier
        fc_bytes = self.fc.in_features * self.fc.out_features * 4  # FP32 for final layer
        if self.fc.bias is not None:
            fc_bytes += self.fc.out_features * 4
        breakdown['classifier'] = fc_bytes
        
        breakdown['total'] = sum(breakdown.values())
        
        return breakdown


def train_with_ternary_annealing(model, train_loader, optimizer, 
                                 epoch, num_epochs, device='cuda'):
    """
    Training loop with threshold annealing for better ternary quantization.
    Gradually increase sparsity (decrease threshold) during training.
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Anneal threshold: start at 0.9 (less sparse), end at 0.5 (more sparse)
    progress = epoch / max(1, num_epochs)
    threshold = 0.9 - 0.4 * progress
    
    # Update all TernaryConv1d modules
    for module in model.modules():
        if isinstance(module, TernaryConv1d):
            module.threshold = threshold
    
    criterion = nn.CrossEntropyLoss()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    return total_loss / len(train_loader), 100. * correct / total


# Model registration for experiments.py
def build_ternary_separable(in_ch: int, num_classes: int, 
                           base: int = 16, ternary_threshold: float = 0.7,
                           input_length: int = 1800, **kwargs):
    """Builder function compatible with experiment framework"""
    return TernarySeparableCNN(
        in_ch=in_ch,
        base=base,
        num_classes=num_classes,
        ternary_threshold=ternary_threshold,
        input_length=input_length
    )


if __name__ == '__main__':
    # Quick test
    model = TernarySeparableCNN(in_ch=1, base=16, num_classes=2)
    x = torch.randn(4, 1, 1800)
    y = model(x)
    print(f"Output shape: {y.shape}")
    
    # Flash usage
    breakdown = model.compute_total_flash_bytes()
    print("\nFlash Memory Breakdown (bytes):")
    for k, v in breakdown.items():
        print(f"  {k}: {v:,} ({v/1024:.2f} KB)")
    print(f"\nTotal: {breakdown['total']/1024:.2f} KB")
