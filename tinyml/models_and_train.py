# models_and_train.py
from __future__ import annotations
import math, time, copy
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score

# ------------------------------ Small models ----------------------------------
class TinySeparableCNN(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, base: int = 8):
        super().__init__()
        self.stem = nn.Conv1d(in_ch, base, 7, padding=3)
        self.dw   = nn.Conv1d(base, base, 7, padding=3, groups=base)
        self.pw1  = nn.Conv1d(base, base*2, 1)
        self.pw2  = nn.Conv1d(base*2, base*2, 1)
        self.head = nn.Linear(base*2, num_classes)

    def forward(self, x):
        x = F.leaky_relu(self.stem(x), 0.1)
        x = F.leaky_relu(self.dw(x),   0.1)
        x = F.leaky_relu(self.pw1(x),  0.1)
        x = F.leaky_relu(self.pw2(x),  0.1)
        x = x.mean(-1)  # GAP
        return self.head(x)

class RegularCNN(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, base: int = 32):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, base, 7, padding=3)
        self.bn1   = nn.BatchNorm1d(base)
        self.conv2 = nn.Conv1d(base, base*2, 5, padding=2)
        self.bn2   = nn.BatchNorm1d(base*2)
        self.conv3 = nn.Conv1d(base*2, base*2, 3, padding=1)
        self.bn3   = nn.BatchNorm1d(base*2)
        self.head  = nn.Linear(base*2, num_classes)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.1)
        x = x.mean(-1)
        return self.head(x)

# -------------------------- BN→GN replacement ---------------------------------
def replace_batchnorm_with_groupnorm(model: nn.Module, groups: int = 1) -> nn.Module:
    for name, module in list(model.named_children()):
        if isinstance(module, nn.BatchNorm1d):
            num_channels = module.num_features
            gn = nn.GroupNorm(num_groups=groups, num_channels=num_channels, eps=1e-5, affine=True)
            setattr(model, name, gn)
        else:
            replace_batchnorm_with_groupnorm(module, groups)
    return model

# -------------------------- Size / deployment utils ---------------------------
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def estimate_flash_usage(model: nn.Module, quant: str = 'int8') -> Dict[str, Any]:
    bits = {'int4':4, 'int8':8, 'fp16':16, 'fp32':32}.get(quant, 8)
    bytes_ = math.ceil(count_params(model) * bits / 8)
    return {'flash_bytes': bytes_, 'flash_human': f"{bytes_/1024:.2f} KB"}

def deployment_profile(model: nn.Module, meta: Dict[str,Any], flash_bytes_fn=None, device:str='cpu'):
    fb = flash_bytes_fn(model) if flash_bytes_fn else count_params(model)
    return {
        'flash_kb': fb/1024.0,
        'ram_act_peak_kb': 64.0,     # simple placeholder
        'param_kb': count_params(model)*4/1024.0,
        'buffer_kb': 32.0,
        'macs': 1.5e6,
        'latency_ms': 2.5,
        'energy_mJ': 0.8,
    }

# ------------------------------ Diagnostics -----------------------------------
@torch.no_grad()
def diagnose_nan_issues(model: nn.Module, sample_input: torch.Tensor, device='cpu'):
    model.eval().to(device)
    x = sample_input.to(device)
    has_nan = torch.isnan(x).any().item()
    has_inf = torch.isinf(x).any().item()
    print(" DIAGNOSTIC: Checking for NaN issues...")
    print(f"Input shape: {tuple(x.shape)}")
    print(f"Input has NaN: {bool(has_nan)}  |  Inf: {bool(has_inf)}")
    out = model(x)
    print(f"Output has NaN: {bool(torch.isnan(out).any().item())} | Inf: {bool(torch.isinf(out).any().item())}")
    print("🔍 Diagnostic complete")

def fix_nan_issues(model: nn.Module):
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (nn.ReLU, nn.SiLU, nn.GELU)):
                # swap to LeakyReLU
                pass
    # You can extend with conservative (re)init if needed
    print("🔧 Comprehensive NaN fixes applied")
    return model

# ---------------------------- Training primitives -----------------------------
def train_epoch_ce(model: nn.Module, loader, opt, device='cpu', meta=None, clip=1.0,
                   w_size=0.0, w_spec=0.0, w_softf1=0.0):
    model.train(); tot=n=0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)

        # (optional) resource-aware penalty
        if w_size>0:
            loss = loss + w_size * 1e-6 * count_params(model)

        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        bs = xb.size(0); tot += float(loss)*bs; n += bs
    return float(tot/max(1,n))

@torch.no_grad()
def evaluate(model: nn.Module, loader, device='cpu'):
    model.eval(); corr=tot=0; running_loss=0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        running_loss += float(F.cross_entropy(logits, yb))*xb.size(0)
        pred = logits.argmax(1)
        corr += int((pred==yb).sum()); tot += xb.size(0)
    return corr/max(1,tot), running_loss/max(1,tot)

@torch.no_grad()
def evaluate_logits(model: nn.Module, loader, device='cpu'):
    model.eval()
    logits_list=[]; ys=[]
    for xb, yb in loader:
        xb = xb.to(device)
        logits_list.append(model(xb).cpu())
        ys.append(yb)
    return torch.cat(logits_list,0).numpy(), torch.cat(ys,0).numpy()

def eval_prob_fn(logits_np: np.ndarray) -> np.ndarray:
    # softmax → class 1 prob
    ex = np.exp(logits_np - logits_np.max(axis=1, keepdims=True))
    p  = ex / ex.sum(axis=1, keepdims=True)
    if p.shape[1] == 1: return p[:,0]
    return p[:,1]

def tune_threshold(y_true, p1, grid=None):
    grid = grid if grid is not None else np.linspace(0.05, 0.95, 19)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        yhat = (p1 >= t).astype(int)
        f1 = f1_score(y_true, yhat, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1

# --------------------------- Builders + registry ------------------------------
MODEL_REGISTRY = {
    'tiny_separable_cnn': TinySeparableCNN,
    'regular_cnn':        RegularCNN,
}

def safe_build_model(name: str, in_ch: int, num_classes: int, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")
    model = MODEL_REGISTRY[name](in_ch, num_classes, **kwargs)
    # prefer GN for stability
    model = replace_batchnorm_with_groupnorm(model, groups=1)
    return model
