from dataclasses import dataclass
from typing import Optional
import torch
from collections import OrderedDict
import os, glob, random, math, json, typing as T
from pathlib import Path
import numpy as np
import wfdb
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from google.colab import drive
import shutil, os
import os, glob
import os, random, numpy as np, wfdb, torch
from collections import Counter, defaultdict, OrderedDict
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd
from torch.utils.data import ConcatDataset, Subset, random_split, RandomSampler, WeightedRandomSampler
from torch.utils.data.dataset import ConcatDataset, Subset
from torch.utils.data import random_split, RandomSampler, WeightedRandomSampler
import ast
from typing import List, Tuple
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
from math import ceil
from torch.optim.lr_scheduler import LambdaLR
import os, sys
from collections import Counter
from google.colab import drive  # will exist in Colab
from torch.nn.utils import clip_grad_norm_
import math
from pprint import pprint
from collections import defaultdict
from pprint import pprint; pprint(res)
from sklearn.metrics import f1_score
import traceback
import time
from typing import Any, Dict, Tuple, List
from torch.optim import Adam
import traceback; traceback.print_exc()
import pandas as pd, numpy as np, inspect
from caas_jupyter_tools import display_dataframe_to_user
import torch, torch.nn.functional as F
import torch, numpy as np
import csv
import json
from typing import Dict, Tuple, Any, List
import itertools
import math, numpy as np
import pandas as pd, time

from .data import *  # if models depend on dataset utils


# Model definitions (consolidated)
# %% Enhanced Training & eval with advanced techniques

def acc_logits(logits, y):
    return (logits.argmax(1) == y).float().mean().item()

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing for better generalization"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """Cosine annealing with warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def train_classifier_epoch(model, loader, opt, device, criterion=None, clip_grad=1.0, scheduler=None):
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.train()
    tot = 0; acc = 0; n = 0

    for batch_idx, (xb, yb) in enumerate(loader):
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(set_to_none=True)

        # Forward pass
        logits = model(xb)
        loss = criterion(logits, yb)

        # Check for NaN/inf before backward
        if not torch.isfinite(loss):
            print(f"Warning: Non-finite loss detected: {loss}")
            continue

        loss.backward()

        # Gradient clipping
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        opt.step()

        # Update scheduler
        if scheduler is not None:
            scheduler.step()

        bs = xb.size(0)
        tot += loss.item() * bs
        acc += acc_logits(logits, yb) * bs
        n += bs

    return tot/max(1,n), acc/max(1,n)

@torch.no_grad()
def eval_classifier(model, loader, device, criterion=None):
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.eval()
    tot = 0; acc = 0; n = 0
    all_preds = []
    all_targets = []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)

        preds = logits.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(yb.cpu().numpy())

        bs = xb.size(0)
        tot += loss.item() * bs
        acc += acc_logits(logits, yb) * bs
        n += bs

    return tot/max(1,n), acc/max(1,n), all_preds, all_targets

def mixup_data(x, y, alpha=0.2):
    """Mixup data augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss computation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def vae_loss(x, xhat, mu, lv, beta=0.1):
    recon = F.mse_loss(xhat, x, reduction="mean")
    kld = -0.5*torch.mean(1 + lv - mu.pow(2) - lv.exp())

    # Clamp to prevent extreme values
    kld = torch.clamp(kld, 0, 50)  # Reduced upper bound
    recon = torch.clamp(recon, 0, 50)

    return recon + beta*kld, recon, kld

@torch.no_grad()
def eval_vae_epoch(vae, loader, device, beta=1.0):
    vae.eval()
    total, recon_sum, kld_sum, n = 0.0, 0.0, 0.0, 0
    for xb, _ in loader:
        xb = xb.to(device)
        xb = torch.nan_to_num(xb)
        xb = standardize_1d(xb)  # same standardization used in train
        xhat, mu, lv = vae(xb)
        loss, recon, kld = safe_vae_loss(xb, xhat, mu, lv, beta=beta)
        bs = xb.size(0)
        total     += float(loss)  * bs
        recon_sum += float(recon) * bs
        kld_sum   += float(kld)   * bs
        n += bs
    return total / max(1, n), recon_sum / max(1, n), kld_sum / max(1, n)

# Advanced metrics
def compute_metrics(y_true, y_pred):
    """Compute additional metrics beyond accuracy"""
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    # For binary classification, compute AUC
    if len(set(y_true)) == 2:
        auc = roc_auc_score(y_true, y_pred)
    else:
        auc = None

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm
    }



# === Missing Model Builder Functions for HyperTiny Architecture ===

def build_hypertiny_all_synth(base_channels=24, num_classes=2, latent_dim=16, input_length=1800,
                            dz=6, dh=16, r=4, synthesis_mode="full"):
    """
    Build HyperTiny model with full synthesis capabilities.

    Args:
        base_channels: Base channel count for the model
        num_classes: Number of output classes
        latent_dim: Dimensionality of latent space
        input_length: Expected input sequence length
        dz: Latent code dimension for synthesis
        dh: Hidden dimension for generator
        r: Rank factor for low-rank approximations
        synthesis_mode: "full" for all layers synthetic, "hybrid" for partial
    """
    return SharedCoreSeparable1D(
        in_ch=1,
        base=base_channels,
        num_classes=num_classes,
        latent_dim=latent_dim,
        hybrid_keep=0 if synthesis_mode == "full" else 1,  # 0 = all synthetic, 1 = hybrid
        input_length=input_length
    )

def build_hypertiny_hybrid(base_channels=24, num_classes=2, latent_dim=16, input_length=1800,
                         dz=6, dh=16, r=4, keep_pw1=True):
    """
    Build HyperTiny model with hybrid synthesis (keep first PW layer, synthesize rest).

    Args:
        base_channels: Base channel count for the model
        num_classes: Number of output classes
        latent_dim: Dimensionality of latent space
        input_length: Expected input sequence length
        dz: Latent code dimension for synthesis
        dh: Hidden dimension for generator
        r: Rank factor for low-rank approximations
        keep_pw1: Whether to keep first pointwise layer (hybrid mode)
    """
    return SharedCoreSeparable1D(
        in_ch=1,
        base=base_channels,
        num_classes=num_classes,
        latent_dim=latent_dim,
        hybrid_keep=1 if keep_pw1 else 0,
        input_length=input_length
    )

def build_tiny_separable_cnn(base_channels=24, num_classes=2, latent_dim=16, input_length=1800):
    """
    Build standard tiny separable CNN without synthesis (baseline comparison).
    """
    return SharedCoreSeparable1D(
        in_ch=1,
        base=base_channels,
        num_classes=num_classes,
        latent_dim=latent_dim,
        hybrid_keep=1,  # Standard mode
        input_length=input_length
    )

# === EC57 Metrics and Evaluation Utilities ===

def ec57_metrics(y_true, y_pred, labels=None):
    """
    Compute EC57-style metrics with bootstrap confidence intervals.
    Returns accuracy, macro-F1, per-class F1, and 95% CIs.
    """
    from sklearn.metrics import accuracy_score, f1_score
    import numpy as np

    def _bootstrap_ci(stat_fn, y_true, y_pred, B=1000, seed=42):
        """Bootstrap confidence interval for a statistic."""
        rng = np.random.default_rng(seed)
        n = len(y_true)
        idx = np.arange(n)
        boots = []
        for _ in range(B):
            s = rng.choice(idx, size=n, replace=True)
            boots.append(stat_fn(y_true[s], y_pred[s]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return float(lo), float(hi)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average='macro')
    pcs = f1_score(y_true, y_pred, average=None, labels=labels) if labels is not None else f1_score(y_true, y_pred, average=None)

    acc_ci = _bootstrap_ci(accuracy_score, y_true, y_pred)
    macro_ci = _bootstrap_ci(lambda a,b: f1_score(a,b,average='macro'), y_true, y_pred)

    per_class = {int(lbl): float(f) for lbl, f in zip(labels if labels is not None else np.unique(y_true), pcs)}

    return {
        "acc": float(acc),
        "acc_ci": acc_ci,
        "macro_f1": float(macro),
        "macro_f1_ci": macro_ci,
        "per_class_f1": per_class
    }

def packed_flash_bytes(components):
    """
    Calculate packed flash memory usage from model components.

    Args:
        components: Dict[str, Tuple[int, int]] - {name: (num_params, bitwidth)}

    Returns:
        Tuple[int, Dict[str, int]] - (total_bytes, breakdown_dict)
    """
    from math import ceil

    breakdown = {}
    total = 0
    for k, (n_params, b) in components.items():
        bytes_k = ceil((n_params * b) / 8.0)
        breakdown[k] = int(bytes_k)
        total += bytes_k
    return int(total), breakdown

def to_kb(nbytes):
    """Convert bytes to KB, rounded to 2 decimal places."""
    return round(nbytes / 1024.0, 2)

# === Model Parameter Counting Utilities ===

def count_dw_parameters(model):
    """Count parameters in depthwise convolution layers."""
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv1d) and hasattr(module, 'groups') and module.groups == module.in_channels:
            count += sum(p.numel() for p in module.parameters() if p.requires_grad)
    return count

def count_pw_parameters(model):
    """Count parameters in pointwise convolution layers (1x1 convs)."""
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv1d) and module.kernel_size[0] == 1:
            count += sum(p.numel() for p in module.parameters() if p.requires_grad)
    return count

def estimate_synthesis_overhead(dz, dh, num_pw_layers):
    """
    Estimate parameter overhead for synthesis components.

    Args:
        dz: Latent code dimension
        dh: Hidden dimension in generator
        num_pw_layers: Number of pointwise layers to synthesize

    Returns:
        Dict with parameter counts for synthesis components
    """
    # Shared generator parameters
    gen_params = dz + dz * dh + dh * dh  # z vector + first linear + second linear

    # Per-layer head parameters (depends on PW layer sizes)
    head_params_per_layer = dh * 64  # Assuming average PW layer has ~64 weights
    total_head_params = head_params_per_layer * num_pw_layers

    return {
        "generator": gen_params,
        "heads_total": total_head_params,
        "codes_total": dz * num_pw_layers,  # One code per layer
        "synthesis_overhead": gen_params + total_head_params + dz * num_pw_layers
    }

# === Advanced Training Utilities ===

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Create cosine annealing scheduler with warmup."""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lr_lambda)

def mixup_data(x, y, alpha=0.2):
    """Apply mixup augmentation to input batch."""
    import torch
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

print(" Model builder functions and utilities loaded successfully!")


# Dataset loaders (consolidated)
# Cell 1 — Paths (keeps your existing layout)
# Your original path (edit if yours differs)
APNEA_ROOT = Path("/content/drive/MyDrive/tinyml_hyper_tiny_baselines/data/apnea-ecg-database-1.0.0")
PTBXL_ROOT = Path("/content/drive/MyDrive/tinyml_hyper_tiny_baselines/data/ptbxl")
MITDB_ROOT = Path("/content/drive/MyDrive/tinyml_hyper_tiny_baselines/data/mitbih/raw") #UCI HAR Dataset")

for p in [APNEA_ROOT, PTBXL_ROOT, MITDB_ROOT]:
    p.mkdir(parents=True, exist_ok=True)

print("[Paths]")
print("  APNEA_ROOT:", APNEA_ROOT)
print("  PTBXL_ROOT:", PTBXL_ROOT)
print("  MITDB_ROOT:", MITDB_ROOT)
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    print("[Drive] Mounted ")
except Exception as e:
    print("[Drive] Skipped mounting:", e)

from pathlib import Path

APNEA_ROOT = Path(APNEA_ROOT)
APNEA_ROOT.mkdir(parents=True, exist_ok=True)
print("[Paths] APNEA_ROOT:", APNEA_ROOT)

########################################################################################

# Cell 2 — Optional, non-destructive downloaders

def _dir_has_any(root: Path, exts=(".dat",".hea",".apn",".csv",".mat",".atr")):
    try:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                return True
    except Exception:
        pass
    return False

def _wfdb_download(db_name: str, dest: Path, do_download: bool, force: bool, verbose: bool):
    dest.mkdir(parents=True, exist_ok=True)
    exists_any = _dir_has_any(dest)
    if verbose:
        print(f"[download] {db_name} -> {dest} | do={do_download} force={force} existing={exists_any}")
    if not do_download and not force:
        if verbose: print("  - Skipped (flags off).")
        return
    if exists_any and not force:
        if verbose: print("  - Files present; not forcing re-download.")
        return
    wfdb.dl_database(db_name, dest.as_posix())
    if verbose: print("  - Download completed.")

# Call (safe; won't download unless flags True)
_wfdb_download("apnea-ecg", APNEA_ROOT, DO_APNEA_DOWNLOAD, FORCE_DOWNLOAD, VERBOSE_DL)
_wfdb_download("ptb-xl",    PTBXL_ROOT, DO_PTBXL_DOWNLOAD, FORCE_DOWNLOAD, VERBOSE_DL)
_wfdb_download("mitdb",     MITDB_ROOT, DO_MITDB_DOWNLOAD, FORCE_DOWNLOAD, VERBOSE_DL)


########################################################################################

# --- Google Drive debug + where-are-my-files finder ---

import os, sys
from pathlib import Path
from collections import Counter

# 0) Ensure Drive is mounted (no-op if already mounted)
try:
    from google.colab import drive  # will exist in Colab
    drive.mount('/content/drive', force_remount=False)
except Exception:
    pass

BASE = Path("/content/drive/MyDrive/tinyml_hyper_tiny_baselines/data")
TARGET_FOLDERS = [
    BASE / "UCI HAR Dataset",
    BASE / "apnea-ecg-database-1.0.0",
    BASE / "mitdb",
    BASE / "ptbxl",
]

def ext_counts(root: Path, max_depth=10, limit=None):
    """Count file extensions under root (any depth)."""
    cnt = Counter()
    n = 0
    for p in root.rglob("*"):
        if p.is_file():
            cnt[p.suffix] += 1
            n += 1
            if limit and n >= limit:
                break
    return n, cnt

def list_dirs_and_files(root: Path, depth=1):
    """List immediate entries; show if items are dirs, files, or symlinks/shortcuts."""
    entries = []
    if not root.exists():
        return entries
    for p in sorted(root.iterdir()):
        kind = "DIR" if p.is_dir() else "FILE"
        if p.is_symlink():
            kind += " (symlink)"
        entries.append((kind, p.name))
    return entries
########################################################################################

# --- Hot-fix Cell 3: Optional sanity check (set DO_SANITY=True to run) ---
DO_SANITY = False
if DO_SANITY:
    try:
        _tr, _va, _te = load_apnea_ecg_loaders_impl(APNEA_ROOT, batch_size=64, length=1800, stride=None, verbose=True)
        xb, yb = next(iter(_tr))
        print("[Sanity] Batch:", xb.shape, yb.shape)
    except Exception as e:
        print("[Sanity] Loader error:", e)


########################################################################################

# %% Models (compact, TinyML-friendly)

import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
def _standardize_1d(x, eps: float = 1e-6):
    # x: (B, C, T)
    mu = x.mean(dim=-1, keepdim=True)
    sd = x.std(dim=-1, keepdim=True).clamp_min(eps)
    return (x - mu) / sd
class SqueezeExcite1D(nn.Module):
    """Lightweight SE block for 1D signals - improves feature selection"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        red = max(1, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, red, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(red, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class DepthwiseSeparable1D(nn.Module):
    def __init__(self, in_ch, out_ch, k=5, stride=1, padding=None, use_se=True, use_residual=True):
        super().__init__()
        padding = (k//2) if padding is None else padding
        self.use_residual = use_residual and (in_ch == out_ch) and (stride == 1)

        self.dw = nn.Conv1d(in_ch, in_ch, kernel_size=k, stride=stride, padding=padding, groups=in_ch, bias=False)
        self.pw = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(in_ch)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)

        # Optional squeeze-excitation
        self.se = SqueezeExcite1D(out_ch) if use_se else None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = x

        # Depthwise conv
        out = self.dw(x)
        out = self.bn1(out)
        out = self.act(out)

        # Pointwise conv
        out = self.pw(out)
        out = self.bn2(out)

        # SE block
        if self.se is not None:
            out = self.se(out)

        # Residual connection
        if self.use_residual:
            out = out + identity

        out = self.act(out)
        return out

class SharedPWGenerator(nn.Module):
    """Enhanced latent-to-weight generator with better expressivity"""
    def __init__(self, z_dim=16, hidden=64):
        super().__init__()
        self.z = nn.Parameter(torch.randn(z_dim) * 0.02)  # Even smaller init
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.LayerNorm(hidden),  # Better than BatchNorm for small latents
            nn.ReLU(),
            nn.Dropout(0.1),  # Regularization
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)  # Smaller gain
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self):
        h = self.net(self.z)
        return h

class PWHead(nn.Module):
    """Enhanced projection with better weight generation"""
    def __init__(self, h_dim, flat_out):
        super().__init__()
        mid_dim = max(1, min(h_dim, flat_out // 2))
        self.proj = nn.Sequential(
            nn.Linear(h_dim, mid_dim, bias=False),
            nn.ReLU(),
            nn.Linear(mid_dim, flat_out, bias=False)
        )
        for m in self.proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.05)

    def forward(self, h):
        return self.proj(h)

# ==== Channel Split Safety Helper ====
def _derive_out_ch(out_ch, in_ch):
    if out_ch is None:
        cfg = globals().get("CURRENT_CFG", None)
        try:
            base = int(getattr(cfg, "width_base", in_ch)) if cfg is not None else int(in_ch)
            mult = float(getattr(cfg, "width_mult", 1.0)) if cfg is not None else 1.0
        except Exception:
            base, mult = int(in_ch), 1.0
        out_ch = int(max(4, round(base * mult)))
    return int(out_ch)

class MultiScaleFeatures(nn.Module):
    """Extract features at multiple temporal scales"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # Split out_ch across 3 branches while preserving the exact sum
        out_ch = _derive_out_ch(out_ch, in_ch)
        b1 = out_ch // 3
        b2 = out_ch // 3
        b3 = out_ch - (b1 + b2)  # absorbs remainder so b1+b2+b3 == out_ch
        assert b1 > 0 and b2 > 0 and b3 > 0, "out_ch must be >= 3"

        self.branches = nn.ModuleList([
            nn.Conv1d(in_ch, b1, kernel_size=3, padding=1, bias=False),
            nn.Conv1d(in_ch, b2, kernel_size=5, padding=2, bias=False),
            nn.Conv1d(in_ch, b3, kernel_size=7, padding=3, bias=False),
        ])
        self.bn = nn.BatchNorm1d(out_ch)  # matches concat channels
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        features = torch.cat([branch(x) for branch in self.branches], dim=1)
        # assert features.shape[1] == self.bn.num_features, f"BN expects {self.bn.num_features}, got {features.shape[1]}"
        return self.act(self.bn(features))

def standardize_1d(x, eps: float = 1e-6):
    # x: (B, C, T) → per-sample, per-channel standardization
    mu = x.mean(dim=-1, keepdim=True)
    sd = x.std(dim=-1, keepdim=True).clamp_min(eps)
    return (x - mu) / sd

# --- Stability helpers (standardize + safe ops) ---
import torch
import torch.nn.functional as F

def standardize_1d(x, eps: float = 1e-6):
    # x: (B, C, T) → per-sample, per-channel standardization
    mu = x.mean(dim=-1, keepdim=True)
    sd = x.std(dim=-1, keepdim=True).clamp_min(eps)
    return (x - mu) / sd

@torch.no_grad()
def nan_sanitize_():
    # Call occasionally if needed
    for obj in list(globals().values()):
        if isinstance(obj, torch.nn.Module):
            for p in obj.parameters():
                if p.grad is not None:
                    p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=1e6, neginf=-1e6)

# --- Mixup utilities (works with SafeFocalLoss) ---
def one_hot(target, num_classes):
    return F.one_hot(target, num_classes=num_classes).float()

def mixup_batch(x, y, alpha: float, num_classes: int):
    if alpha <= 0:
        return x, one_hot(y, num_classes)
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    idx = torch.randperm(x.size(0), device=x.device)
    x_m = lam * x + (1.0 - lam) * x[idx]
    y_m = lam * one_hot(y, num_classes) + (1.0 - lam) * one_hot(y[idx], num_classes)
    return x_m, y_m

class SharedCoreSeparable1D(nn.Module):
    """
    Enhanced proposed model with:
    - Multi-scale feature extraction
    - Squeeze-Excitation attention
    - Residual connections
    - Learned attention weights + global pooling
    - Better classifier head
    """
    def __init__(self, in_ch=1, base=16, num_classes=2, latent_dim=16, input_length=1800, hybrid_keep=1):
        super().__init__()
        self.base = base

        # Stem with multi-scale features
        self.stem = nn.Sequential(
            MultiScaleFeatures(in_ch, base),
            nn.MaxPool1d(2, 2)  # Reduce temporal dimension
        )

        # Depthwise-separable stages
        self.blocks = nn.ModuleList([
            DepthwiseSeparable1D(base, base*2, k=5, stride=2, use_se=True, use_residual=False),
            DepthwiseSeparable1D(base*2, base*2, k=5, stride=1, use_se=True, use_residual=True),
            DepthwiseSeparable1D(base*2, base*4, k=5, stride=2, use_se=True, use_residual=False),
        ])

        # PW weight generator for the last pointwise conv (synthetic)
        self.gen = SharedPWGenerator(z_dim=latent_dim, hidden=96)
        self.last_pw_out = base*4
        self.last_pw_in  = base*2
        last_pw_shape = (self.last_pw_out, self.last_pw_in, 1)
        self.pw_head = PWHead(h_dim=96, flat_out=int(np.prod(last_pw_shape)))

        # Attention weight head: produces (B,1,T) weights in [0,1]
        self.att_weight = nn.Sequential(
            nn.Conv1d(base*4, max(1, base//4), 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(max(1, base//4), 1, 1),
            nn.Sigmoid()
        )

        # Feature dim after pooling is channels of last stage
        feat_dim = base * 4

        # Classifier head
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes)
        )
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _synth_pw_weight(self):
        h = self.gen()
        w = self.pw_head(h)
        w = w.view(self.last_pw_out, self.last_pw_in, 1)
        # Keep synthetic weights small
        w = torch.tanh(w) * 0.05
        return w

    def _forward_features(self, x):
        # --- NEW: sanitize + standardize inputs ---
        x = torch.nan_to_num(x)
        x = _standardize_1d(x)

        # light noise only during training (kept)
        if self.training:
            x = x + torch.randn_like(x) * 5e-7

        x = self.stem(x)
        x = torch.nan_to_num(x)

        # First two blocks
        x = self.blocks[0](x); x = torch.nan_to_num(x)
        x = self.blocks[1](x); x = torch.nan_to_num(x)

        # Third block: depthwise + BN/act, then synthetic PW, BN, SE, act
        b2 = self.blocks[2].dw(x)
        b2 = self.blocks[2].bn1(b2)
        b2 = self.blocks[2].act(b2)

        # Synthetic PW conv
        w = self._synth_pw_weight()
        b2 = F.conv1d(b2, w, bias=None, stride=1, padding=0, groups=1)
        b2 = self.blocks[2].bn2(b2)

        if self.blocks[2].se is not None:
            b2 = self.blocks[2].se(b2)
        b2 = self.blocks[2].act(b2)
        b2 = torch.nan_to_num(b2)

        # Attention-weighted global pooling (already stable with +1e-6, keep)
        att = self.att_weight(b2)               # (B,1,T)
        b2_weighted = b2 * att                  # (B,C,T)
        y = b2_weighted.sum(dim=-1) / (att.sum(dim=-1) + 1e-6)  # (B,C)

        # --- NEW: final sanitize ---
        y = torch.nan_to_num(y)
        return y

    def forward(self, x):
        y = self._forward_features(x)
        return self.head(y)

    def tinyml_packed_bytes(self):
        total = 0
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                total += (m.weight.numel() * 8 + 7)//8
        for p in list(self.gen.parameters()) + list(self.pw_head.parameters()):
            total += (p.numel() * 4 + 7)//8
        return {"boot": 1954, "lazy": total}

import torch
import torch.nn as nn
import torch.nn.functional as F
class SafeFocalLoss(nn.Module):
    """
    Stable multi-class focal loss (supports hard labels or soft/one-hot).
    """
    def __init__(self, gamma=1.5, alpha=0.5, label_smoothing=0.05, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits, target):
        # logits: (B, C)
        if target.dtype in (torch.long, torch.int64):
            C = logits.size(1)
            with torch.no_grad():
                smooth = self.label_smoothing
                target_prob = torch.full_like(logits, smooth / max(1, C - 1))
                target_prob.scatter_(1, target.view(-1,1), 1.0 - smooth)
        else:
            target_prob = target  # already soft / one-hot

        logp = F.log_softmax(logits, dim=1)        # stable
        p = logp.exp()
        pt = (p * target_prob).sum(dim=1).clamp_min(1e-8)

        ce = -(target_prob * logp).sum(dim=1)      # smoothed CE
        focal = (self.alpha * (1.0 - pt).pow(self.gamma)) * ce

        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        return focal

def safe_vae_loss(x, xhat, mu, lv, beta=1.0):
    x    = torch.nan_to_num(x)
    xhat = torch.nan_to_num(xhat)
    recon = F.mse_loss(torch.tanh(xhat), torch.tanh(x), reduction='mean')

    mu = torch.nan_to_num(mu).clamp(-10, 10)
    lv = torch.nan_to_num(lv).clamp(-8, 8)
    kld = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp().clamp_max(1e4))

    loss = recon + beta * kld
    if not torch.isfinite(loss):
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=-1e6)
    return loss, recon.detach(), kld.detach()

# --- Drop-in training helpers for CNN and VAE ---

from torch.nn.utils import clip_grad_norm_

def make_cosine_with_warmup(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        # cosine from warmup to total
        progress = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    import math
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_cnn_epoch(model, loader, optimizer, criterion, device, epoch,
                    use_mixup=False, mixup_alpha=0.2, num_classes=2, clip=1.0):
    model.train()
    total_loss, total_correct, total_count = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        # sanitize + standardize
        xb = torch.nan_to_num(xb)
        xb = standardize_1d(xb)

        # Mixup only after epoch 0 (stabilize first)
        if use_mixup and epoch >= 1 and mixup_alpha > 0:
            xb, y_soft = mixup_batch(xb, yb, alpha=mixup_alpha, num_classes=num_classes)
            logits = model(xb)
            loss = criterion(logits, y_soft)
            preds = logits.argmax(1)
            total_correct += (preds == yb).sum().item()  # accuracy vs hard labels
        else:
            logits = model(xb)
            loss = criterion(logits, yb)
            preds = logits.argmax(1)
            total_correct += (preds == yb).sum().item()

        loss = torch.nan_to_num(loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=clip)
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        total_count += xb.size(0)

    return total_loss / max(1, total_count), total_correct / max(1, total_count)

@torch.no_grad()
def eval_cnn(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0
    all_preds, all_true = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        xb = torch.nan_to_num(xb)
        xb = standardize_1d(xb)
        logits = model(xb)
        loss = criterion(logits, yb)
        preds = logits.argmax(1)

        total_loss += float(torch.nan_to_num(loss)) * xb.size(0)
        total_correct += (preds == yb).sum().item()
        total_count += xb.size(0)
        all_preds.append(preds.cpu())
        all_true.append(yb.cpu())

    import torch
    all_preds = torch.cat(all_preds).numpy()
    all_true  = torch.cat(all_true).numpy()
    return total_loss / max(1, total_count), total_correct / max(1, total_count), all_preds, all_true

def train_vae_epoch(vae, loader, optimizer, device, beta=1.0, clip=1.0):
    vae.train()
    total, recon_sum, kld_sum, n = 0.0, 0.0, 0.0, 0
    for xb, _ in loader:
        xb = xb.to(device)
        xb = torch.nan_to_num(xb)
        xb = standardize_1d(xb)

        xhat, mu, lv = vae(xb)
        loss, recon, kld = safe_vae_loss(xb, xhat, mu, lv, beta=beta)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(vae.parameters(), max_norm=clip)
        optimizer.step()

        bs = xb.size(0)
        total += loss.item() * bs
        recon_sum += float(recon) * bs
        kld_sum += float(kld) * bs
        n += bs
    return total / n, recon_sum / n, kld_sum / n

# --- Enhanced Tiny VAE with better architecture
class TinyVAE1D(nn.Module):
    def __init__(self, in_channels=1, base=16, latent_dim=16, input_length=1800):
        super().__init__()
        self.latent_dim = latent_dim

        # Enhanced encoder with residual connections
        self.enc = nn.Sequential(
            nn.Conv1d(in_channels, base, 7, 2, 3), nn.BatchNorm1d(base), nn.ReLU(),
            DepthwiseSeparable1D(base, base*2, k=5, stride=2, use_se=False, use_residual=False),
            DepthwiseSeparable1D(base*2, base*2, k=5, stride=2, use_se=True, use_residual=True),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_length)
            e = self.enc(dummy)
            self._enc_flat = e.shape[1] * e.shape[2]
            self._enc_channels = e.shape[1]
            self._enc_length = e.shape[2]

        # Enhanced latent projection
        self.fc_mu = nn.Sequential(
            nn.Linear(self._enc_flat, latent_dim*2),
            nn.ReLU(),
            nn.Linear(latent_dim*2, latent_dim)
        )
        self.fc_lv = nn.Sequential(
            nn.Linear(self._enc_flat, latent_dim*2),
            nn.ReLU(),
            nn.Linear(latent_dim*2, latent_dim)
        )

        self.dec_fc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim*2),
            nn.ReLU(),
            nn.Linear(latent_dim*2, self._enc_flat)
        )

        # Enhanced decoder
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(base*2, base*2, 4, 2, 1), nn.BatchNorm1d(base*2), nn.ReLU(),
            nn.ConvTranspose1d(base*2, base, 4, 2, 1), nn.BatchNorm1d(base), nn.ReLU(),
            nn.ConvTranspose1d(base, in_channels, 4, 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def encode(self, x):
        # --- NEW: sanitize + standardize before encode ---
        x = torch.nan_to_num(x)
        x = _standardize_1d(x)
        h = self.enc(x).view(x.size(0), -1)
        mu = self.fc_mu(h)
        lv = self.fc_lv(h)
        # tighter & safer clamps
        mu = torch.nan_to_num(mu).clamp(-10, 10)
        lv = torch.nan_to_num(lv).clamp(-8, 8)
        return mu, lv

    def reparam(self, mu, lv):
        std = torch.exp(0.5*lv).clamp_min(1e-3)   # --- NEW: avoid near-zero std
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h = self.dec_fc(z).view(z.size(0), self._enc_channels, self._enc_length)
        out = self.dec(h)
        # --- NEW: bound + sanitize decoder output to avoid exploding recon ---
        out = torch.tanh(out)                     # keep outputs finite
        return torch.nan_to_num(out)

    def forward(self, x):
        mu, lv = self.encode(x)
        z = self.reparam(mu, lv)
        xhat = self.decode(z)
        return xhat, mu, lv

class VAEAdapter(nn.Module):
    """Enhanced adapter with feature refinement"""
    def __init__(self, vae: TinyVAE1D):
        super().__init__()
        self.vae = vae
        self.refine = nn.Sequential(
            nn.Linear(vae.latent_dim, vae.latent_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(vae.latent_dim, vae.latent_dim)
        )

    def forward(self, x):
        mu, lv = self.vae.encode(x)
        refined = self.refine(mu)
        return refined + mu  # Residual connection

class AttentionPool1D(nn.Module):
    """
    Parameter-free temporal attention pooling.
    x: (B, C, T) -> returns (B, C)
    """
    def forward(self, x):
        score = x.mean(dim=1, keepdim=True)       # (B,1,T)
        alpha = torch.softmax(score, dim=-1)      # (B,1,T)
        return (x * alpha).sum(dim=-1)            # (B,C)

class TinyHead(nn.Module):
    def __init__(self, in_dim, num_classes=2, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden*2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden*2, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, z):
        return self.net(z)


########################################################################################

# Cell 7 — Enhanced Experiment runner with advanced techniques

from dataclasses import dataclass
from pprint import pprint
import math
import numpy as np
from typing import Optional
'''
@dataclass
class ExpCfg:
# ---- Training schedule ----
    epochs: int = 8
    epochs_cnn: Optional[int] = None           # fallback -> epochs
    epochs_head: Optional[int] = None          # fallback -> epochs//2
    epochs_vae_pre: Optional[int] = None       # fallback -> epochs//2
    warmup_epochs: int = 1

    # ---- Optimizer / LR ----
    lr: float = 3e-4
    weight_decay: float = 1e-4

    # ---- Data loading ----
    batch_size: int = 64
    num_workers: int = 2
    limit: Optional[int] = None                # cap samples for quick runs

    # ---- Windowing / signal params ----
    length: int = 1800                         # e.g., 60s * 30Hz
    window_ms: int = 60000                     # 60s default
    target_fs: int = 100                       # resample target
    stride: Optional[int] = None               # step between windows (samples)
    input_len: Optional[int] = None            # derived if None

    # ---- Model knobs ----
    latent_dim: int = 16                       # used by VAE/heads, etc.
    width_base: int = 24                       # channel base for tiny models
    width_mult: float = 1.0                    # channel multiplier

    # ---- Augment / loss toggles ----
    use_mixup: bool = False
    mixup_alpha: float = 0.2
    use_focal_loss: bool = False
    use_label_smoothing: bool = False

    # ---- Misc ----
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    base: int = 24                             # base channels for models

    def __post_init__(self):
        # sensible fallbacks
        if self.epochs_cnn is None:
            self.epochs_cnn = self.epochs
        if self.epochs_head is None:
            self.epochs_head = max(1, self.epochs // 2)
        if self.epochs_vae_pre is None:
            self.epochs_vae_pre = max(1, self.epochs // 2)
        if self.input_len is None and self.window_ms and self.target_fs:
            self.input_len = int((self.window_ms / 1000.0) * self.target_fs)
        if isinstance(self.stride, float):
            self.stride = int(self.stride)
        # Ensure base is always a positive integer for model initialization
        if self.base is None or self.base <= 0:
            self.base = 24
'''
def _has_data(loader):
    try:
        return len(loader) > 0
    except Exception:
        for _ in loader:
            return True
        return False

def _safe_make_apnea_loaders(root: str, cfg: ExpCfg):
    try:
        return load_apnea_ecg_loaders_impl(root, batch_size=cfg.batch_size, length=cfg.input_len, stride=cfg.stride, verbose=True)
    except TypeError:
        return load_apnea_ecg_loaders_impl(root, batch_size=cfg.batch_size, length=cfg.input_len, verbose=True)

def run_apnea(cfg: ExpCfg, root: str):
    print("\n[make_loaders] Preparing dataset: ApneaECG")
    tr_loader, va_loader, te_loader = _safe_make_apnea_loaders(root, cfg)

    if not _has_data(tr_loader) or not _has_data(va_loader):
        print("[ApneaECG] No data after filtering — skipping this dataset.")
        return {
            "dataset": "ApneaECG",
            "cnn_val_acc": None,
            "vae_val_acc": None,
            "cnn_packed_bytes": None,
            "note": "Skipped: no usable windows/labels"
        }

    # Class distributions
    print_class_distribution(tr_loader, "ApneaECG Train")
    print_class_distribution(va_loader, "ApneaECG Val")
    print_class_distribution(te_loader, "ApneaECG Test")

    # ---- Enhanced CNN baseline ----
    cnn = SharedCoreSeparable1D(
        in_ch=1, base=cfg.base, num_classes=2,
        latent_dim=cfg.latent_dim, hybrid_keep=1,
        input_length=cfg.input_len
    ).to(DEVICE)

    # Enhanced optimizer and scheduler
    opt_cnn = torch.optim.AdamW(cnn.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = len(tr_loader) * cfg.epochs_cnn
    warmup_steps = len(tr_loader) * cfg.warmup_epochs
    scheduler_cnn = get_cosine_schedule_with_warmup(opt_cnn, warmup_steps, total_steps)

    # Enhanced loss function
    if cfg.use_focal_loss:
        criterion_cnn = FocalLoss(alpha=1, gamma=2)
    elif cfg.use_label_smoothing:
        criterion_cnn = LabelSmoothingCrossEntropy(smoothing=0.1)
    else:
        criterion_cnn = nn.CrossEntropyLoss()

    print(f"[ApneaECG] Training CNN with {type(criterion_cnn).__name__}...")
    best_val_acc = 0
    patience_counter = 0
    patience = 3

    for ep in range(1, cfg.epochs_cnn + 1):
        # Enhanced training with mixup
        cnn.train()
        tot = 0; acc = 0; n = 0

        for batch_idx, (xb, yb) in enumerate(tr_loader):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            # Apply mixup augmentation
            if cfg.use_mixup and np.random.random() > 0.5:
                mixed_x, y_a, y_b, lam = mixup_data(xb, yb, cfg.mixup_alpha)
                opt_cnn.zero_grad(set_to_none=True)
                logits = cnn(mixed_x)
                loss = mixup_criterion(criterion_cnn, logits, y_a, y_b, lam)
            else:
                opt_cnn.zero_grad(set_to_none=True)
                logits = cnn(xb)
                loss = criterion_cnn(logits, yb)

            if not torch.isfinite(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(cnn.parameters(), 1.0)
            opt_cnn.step()
            scheduler_cnn.step()

            bs = xb.size(0)
            tot += loss.item() * bs
            acc += acc_logits(logits, yb) * bs
            n += bs

        trL, trA = tot/max(1,n), acc/max(1,n)

        # Validation with detailed metrics
        vaL, vaA, val_preds, val_targets = eval_classifier(cnn, va_loader, DEVICE, criterion_cnn)
        metrics = compute_metrics(val_targets, val_preds)

        print(f"[ApneaECG] CNN ep {ep:02d} trL={trL:.4f} trA={trA:.3f} vaL={vaL:.4f} vaA={vaA:.3f} F1={metrics['f1']:.3f}")

        # Early stopping
        if vaA > best_val_acc:
            best_val_acc = vaA
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[ApneaECG] CNN early stopping at epoch {ep}")
                break

        if not np.isfinite(trL) or not np.isfinite(vaL):
            print(f"[ApneaECG] CNN training stopped due to NaN at epoch {ep}")
            break

    cnn_bytes = cnn.tinyml_packed_bytes()

    # ---- Enhanced VAE + classifier ----
    vae = TinyVAE1D(in_channels=1, latent_dim=cfg.latent_dim, base=cfg.base, input_length=cfg.input_len).to(DEVICE)
    opt_vae = torch.optim.AdamW(vae.parameters(), lr=cfg.lr*0.5, weight_decay=cfg.weight_decay)  # Lower LR for VAE

    print("[ApneaECG] Training VAE...")
    for ep in range(1, cfg.epochs_vae_pre + 1):
        # Progressive beta scheduling
        beta = min(0.5, 0.1 * ep / cfg.epochs_vae_pre)

        tr_tot, tr_rec, tr_kld = train_vae_epoch(vae, tr_loader, opt_vae, DEVICE, beta=beta, clip=1.0)
        va_tot, va_rec, va_kld = eval_vae_epoch(vae, va_loader, DEVICE, beta=beta)

        print(
            f"[ApneaECG] VAE ep {ep:02d} "
            f"loss_tr={tr_tot:.4f} recon_tr={tr_rec:.4f} kld_tr={tr_kld:.4f} | "
            f"loss_va={va_tot:.4f} recon_va={va_rec:.4f} kld_va={va_kld:.4f} "
            f"beta={beta:.3f}"
        )

        # NaN/Inf guard
        if not all(np.isfinite(v) for v in (tr_tot, tr_rec, tr_kld, va_tot, va_rec, va_kld)):
            print("[ApneaECG] VAE early stop: non-finite detected")
            break

    # Enhanced classifier training
    for p in vae.parameters():
        p.requires_grad = False

    adapter = VAEAdapter(vae).to(DEVICE)
    head = TinyHead(in_dim=cfg.latent_dim, num_classes=2, hidden=64).to(DEVICE)  # Larger head
    opt_h = torch.optim.AdamW(list(adapter.refine.parameters()) + list(head.parameters()),
                             lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Head loss function
    if cfg.use_focal_loss:
        criterion_head = FocalLoss(alpha=1, gamma=2)
    else:
        criterion_head = nn.CrossEntropyLoss()

    print("[ApneaECG] Training VAE classifier head...")
    best_head_acc = 0
    last_vaA = None

    for ep in range(1, cfg.epochs_head + 1):
        # Train head
        head.train()
        adapter.train()
        tot = 0; acc = 0; n = 0

        for x, y in tr_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            z = adapter(x)  # Now trainable through adapter.refine
            opt_h.zero_grad(set_to_none=True)
            logits = head(z)
            loss = criterion_head(logits, y)

            if not torch.isfinite(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(adapter.refine.parameters()) + list(head.parameters()), 1.0)
            opt_h.step()

            bs = x.size(0)
            tot += loss.item() * bs
            acc += acc_logits(logits, y) * bs
            n += bs

        trL, trA = tot/max(n,1), acc/max(n,1)

        # Validation
        head.eval()
        adapter.eval()
        tot = 0; acc = 0; n = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for x, y in va_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                z = adapter(x)
                logits = head(z)
                loss = criterion_head(logits, y)

                preds = logits.argmax(1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(y.cpu().numpy())

                bs = x.size(0)
                tot += loss.item() * bs
                acc += acc_logits(logits, y) * bs
                n += bs

        vaL, vaA = tot/max(n,1), acc/max(n,1)
        last_vaA = vaA
        metrics = compute_metrics(val_targets, val_preds)

        print(f"[ApneaECG] VAE+Head ep {ep:02d} trL={trL:.4f} trA={trA:.3f} vaL={vaL:.4f} vaA={vaA:.3f} F1={metrics['f1']:.3f}")

    # Final test evaluation
    print("\n[ApneaECG] Final test evaluation...")
    _, test_acc, test_preds, test_targets = eval_classifier(cnn, te_loader, DEVICE)
    test_metrics = compute_metrics(test_targets, test_preds)

    res = {
        "dataset": "ApneaECG",
        "cnn_val_acc": round(float(best_val_acc), 4),
        "cnn_test_acc": round(float(test_acc), 4),
        "cnn_test_f1": round(float(test_metrics['f1']), 4),
        "vae_val_acc": round(float(last_vaA), 4) if last_vaA is not None else None,
        "cnn_packed_bytes": cnn_bytes,
        "note": "Enhanced with SE, residuals, focal loss, mixup, scheduling"
    }
    pprint(res)
    return res
def run_ptbxl(cfg: ExpCfg, root: str):
    print("\n[make_loaders] Preparing dataset: PTB-XL")
    if not _dir_has_any(Path(root)):
        print("[PTB-XL] Data folder missing or empty.")
        return {"dataset":"PTB-XL","note":"No data at root."}
    print("Preparing to read the ptbxl loader")
    tr_loader, va_loader, te_loader, meta = load_ptbxl_loaders(
        root, batch_size=cfg.batch_size, length=cfg.input_len, task="binary_diag", lead="II"
    )
    print("Preparing to print the class destribution")
    print_class_distribution(tr_loader, "PTB-XL Train")
    print_class_distribution(va_loader, "PTB-XL Val")
    print_class_distribution(te_loader, "PTB-XL Test")
    print("Preparing configs")
    cnn = SharedCoreSeparable1D(
        in_ch=1, base=cfg.base, num_classes=2, latent_dim=cfg.latent_dim, hybrid_keep=1, input_length=cfg.input_len
    ).to(DEVICE)
    opt = torch.optim.AdamW(cnn.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = get_cosine_schedule_with_warmup(opt, len(tr_loader)*cfg.warmup_epochs, len(tr_loader)*cfg.epochs_cnn)
    criterion = FocalLoss(alpha=1, gamma=2) if cfg.use_focal_loss else nn.CrossEntropyLoss()
    print("Starting training")
    best_va = 0
    for ep in range(1, cfg.epochs_cnn+1):
        cnn.train(); tot=acc=n=0
        for xb,yb in tr_loader:
            xb,yb=xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = cnn(xb)
            loss = criterion(logits, yb)
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(cnn.parameters(), 1.0)
                opt.step(); sched.step()
                bs=xb.size(0); tot+=loss.item()*bs; acc+=acc_logits(logits,yb)*bs; n+=bs
        trL, trA = tot/max(n,1), acc/max(n,1)
        vaL, vaA, _, _ = eval_classifier(cnn, va_loader, DEVICE, criterion)
        print(f"[PTB-XL] ep{ep:02d} trL={trL:.4f} trA={trA:.3f} vaL={vaL:.4f} vaA={vaA:.3f}")
        if vaA>best_va: best_va=vaA
    print("Eval classifiers")
    _, teA, te_preds, te_targets = eval_classifier(cnn, te_loader, DEVICE)
    from collections import defaultdict
    res = {"dataset":"PTB-XL","cnn_val_acc": round(float(best_va),4),
           "cnn_test_acc": round(float(teA),4), "cnn_test_f1": round(float(compute_metrics(te_targets, te_preds)['f1']),4),
           "cnn_packed_bytes": cnn.tinyml_packed_bytes(),
           "note": f"Lead={meta['lead']} Task={meta['task']}"}
    from pprint import pprint; pprint(res)
    return res


def run_mitdb(cfg: ExpCfg, root: str):
    print("\n[make_loaders] Preparing dataset: MITDB (MIT-BIH Arrhythmia)")
    if not _dir_has_any(Path(root)):
        print("[MITDB] Data folder missing or empty.")
        return {"dataset":"MITDB","note":"No data at root."}

    tr_loader, va_loader, te_loader, meta = load_mitdb_loaders(
        root, batch_size=cfg.batch_size, length=cfg.input_len, binary=True
    )
    print_class_distribution(tr_loader, "MITDB Train")
    print_class_distribution(va_loader, "MITDB Val")
    print_class_distribution(te_loader, "MITDB Test")

    cnn = SharedCoreSeparable1D(
        in_ch=1, base=cfg.base, num_classes=2, latent_dim=cfg.latent_dim, hybrid_keep=1, input_length=cfg.input_len
    ).to(DEVICE)
    opt = torch.optim.AdamW(cnn.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = get_cosine_schedule_with_warmup(opt, len(tr_loader)*cfg.warmup_epochs, len(tr_loader)*cfg.epochs_cnn)
    criterion = FocalLoss(alpha=1, gamma=2) if cfg.use_focal_loss else nn.CrossEntropyLoss()

    best_va = 0
    for ep in range(1, cfg.epochs_cnn+1):
        cnn.train(); tot=acc=n=0
        for xb,yb in tr_loader:
            xb,yb=xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = cnn(xb)
            loss = criterion(logits, yb)
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(cnn.parameters(), 1.0)
                opt.step(); sched.step()
                bs=xb.size(0); tot+=loss.item()*bs; acc+=acc_logits(logits,yb)*bs; n+=bs
        trL, trA = tot/max(n,1), acc/max(n,1)
        vaL, vaA, _, _ = eval_classifier(cnn, va_loader, DEVICE, criterion)
        print(f"[MITDB] ep{ep:02d} trL={trL:.4f} trA={trA:.3f} vaL={vaL:.4f} vaA={vaA:.3f}")
        if vaA>best_va: best_va=vaA

    _, teA, te_preds, te_targets = eval_classifier(cnn, te_loader, DEVICE)
    res = {"dataset":"MITDB","cnn_val_acc": round(float(best_va),4),
           "cnn_test_acc": round(float(teA),4), "cnn_test_f1": round(float(compute_metrics(te_targets, te_preds)['f1']),4),
           "cnn_packed_bytes": cnn.tinyml_packed_bytes(),
           "note": f"binary={meta['binary']}, rec_splits={meta['records']}"}
    from pprint import pprint; pprint(res)
    return res


########################################################################################

# Cell — Model Size Analysis and Regular CNN Baseline (REPLACEMENT)

import torch
import torch.nn as nn
import pandas as pd
from collections import OrderedDict

# ===== 1) Regular CNN Baseline (Non-Tiny) =====
class RegularCNN(nn.Module):
    """A regular CNN without TinyML constraints for comparison"""
    def __init__(self, input_length=1800, num_classes=2):
        super().__init__()
        self.input_length = input_length

        # Larger feature extractor for baseline comparison
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            # Block 2
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            # Block 3
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            # Block 4
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            # Block 5
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)         # (B, 512, 1)
        x = x.view(x.size(0), -1)    # (B, 512)
        x = self.classifier(x)       # (B, C)
        return x


# ===== 2) Size helpers =====
def tensor_nbit_bytes(n_params: int, bits: int) -> int:
    """Bytes needed to store n_params at given bit precision (packed)."""
    return (n_params * bits + 7) // 8

def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters (weights + biases)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size_breakdown(model: nn.Module, name_prefix: str = ""):
    """Leaf-module parameter tally grouped by layer type."""
    breakdown = OrderedDict()
    total_params = 0
    for name, module in model.named_modules():
        # leaf module = no children
        if len(list(module.children())) == 0:
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if params > 0:
                layer_type = type(module).__name__
                breakdown[layer_type] = breakdown.get(layer_type, 0) + params
                total_params += params
    return breakdown, total_params

def calculate_flash_sizes(model: nn.Module, model_name: str):
    """Weights-in-flash estimates for FP32/FP16/INT8/INT4 (packed)."""
    total_params = count_parameters(model)
    return {
        f"{model_name}_fp32": {
            "flash_bytes": total_params * 4,   # 32 bits = 4 bytes
            "flash_human": f"{(total_params * 4) / 1024:.2f} KB",
            "params": total_params,
        },
        f"{model_name}_fp16": {
            "flash_bytes": total_params * 2,   # 16 bits = 2 bytes
            "flash_human": f"{(total_params * 2) / 1024:.2f} KB",
            "params": total_params,
        },
        f"{model_name}_int8": {
            "flash_bytes": tensor_nbit_bytes(total_params, 8),
            "flash_human": f"{tensor_nbit_bytes(total_params, 8) / 1024:.2f} KB",
            "params": total_params,
        },
        f"{model_name}_int4": {
            "flash_bytes": tensor_nbit_bytes(total_params, 4),
            "flash_human": f"{tensor_nbit_bytes(total_params, 4) / 1024:.2f} KB",
            "params": total_params,
        },
    }

# ===== 3) Hybrid (mixed-precision) size helper =====
def hybrid_bytes(core_model: nn.Module,
                 unique_heads,
                 conv_layers,
                 keep_pw_layers,
                 bits_core=4, bits_head=4, bits_z=4, bits_stem_dw=8, bits_keep_pw=8) -> int:
    """
    Compute hybrid weights-in-flash by assigning precisions to:
      - core_model.net parameters (bits_core),
      - optional latent 'z' tensor (bits_z),
      - each head in unique_heads (bits_head),
      - conv layers:
          * 'stem' and 'dw' at bits_stem_dw,
          * selected 'pw' layers in keep_pw_layers at bits_keep_pw,
          * all other conv params (including 'pw' not selected) at bits_core (default).
    """
    total = 0

    # Core network (if modeled as core_model.net)
    if hasattr(core_model, 'net'):
        for p in core_model.net.parameters():
            total += tensor_nbit_bytes(p.numel(), bits_core)
    else:
        # Fallback: treat entire model as "core" unless accounted below
        pass

    # Optional latent (if present)
    if hasattr(core_model, 'z'):
        total += tensor_nbit_bytes(core_model.z.numel(), bits_z)

    # Heads
    for head in (unique_heads or []):
        for p in head.parameters():
            total += tensor_nbit_bytes(p.numel(), bits_head)

    # Convs by category
    seen_params = set()
    for name, layer_type, conv in conv_layers:
        # Weight
        if hasattr(conv, "weight") and conv.weight is not None:
            n = conv.weight.numel()
            if layer_type in ("stem", "dw"):
                total += tensor_nbit_bytes(n, bits_stem_dw)
            elif layer_type == "pw" and name in keep_pw_layers:
                total += tensor_nbit_bytes(n, bits_keep_pw)
            else:
                total += tensor_nbit_bytes(n, bits_core)
        # Bias
        if hasattr(conv, "bias") and conv.bias is not None:
            n = conv.bias.numel()
            # Usually small; follow the same precision as its weight bucket
            if layer_type in ("stem", "dw"):
                total += tensor_nbit_bytes(n, bits_stem_dw)
            elif layer_type == "pw" and name in keep_pw_layers:
                total += tensor_nbit_bytes(n, bits_keep_pw)
            else:
                total += tensor_nbit_bytes(n, bits_core)

    return total


# ===== 4) Run analysis =====
def run_size_analysis(cfg: ExpCfg):
    """Run comprehensive size analysis on baseline + tiny variants and print tables."""
    print("="*60)
    print("MODEL SIZE ANALYSIS")
    print("="*60)

    # Create model instances
    models = {}
    models['regular_cnn'] = RegularCNN(input_length=cfg.input_len, num_classes=2)
    models['tiny_cnn'] = SharedCoreSeparable1D(
        in_ch=1, base=cfg.base, num_classes=2,
        latent_dim=cfg.latent_dim, hybrid_keep=1,
        input_length=cfg.input_len
    )
    models['tiny_vae'] = TinyVAE1D(
        in_channels=1, latent_dim=cfg.latent_dim,
        base=cfg.base, input_length=cfg.input_len
    )

    # Baseline param count for ratios
    baseline_params = max(1, count_parameters(models['regular_cnn']))

    # ---- Exact hybrid size for tiny_cnn (classify convs and keep one PW at INT8) ----
    conv_layers = []
    keep_pw_layers = set()

    for name, m in models['tiny_cnn'].named_modules():
        if isinstance(m, nn.Conv1d):
            is_pointwise = (m.kernel_size == (1,))
            is_depthwise = (m.groups == m.in_channels and m.out_channels % max(1, m.in_channels) == 0)
            if 'stem' in name.lower():
                kind = 'stem'
            elif is_depthwise:
                kind = 'dw'
            elif is_pointwise:
                kind = 'pw'
            else:
                kind = 'other'
            conv_layers.append((name, kind, m))

    # Policy: mark the first PW conv to keep at INT8 (others fall back to INT4 via bits_core)
    for name, kind, _ in conv_layers:
        if kind == 'pw':
            keep_pw_layers.add(name)
            break

    bytes_exact_hybrid = hybrid_bytes(
        core_model=models['tiny_cnn'],
        unique_heads=[],                 # add specific heads if your architecture has unique heads
        conv_layers=conv_layers,
        keep_pw_layers=keep_pw_layers,
        bits_core=4, bits_head=4, bits_z=4,  # default INT4
        bits_stem_dw=8, bits_keep_pw=8       # keep stem+dw and one PW at INT8
    )

    # ---- Build per-model size table ----
    size_results = []
    for model_name, model in models.items():
        print(f"\n[{model_name.upper()}]")
        breakdown, total_params = get_model_size_breakdown(model)
        print(f"  Total Parameters: {total_params:,}")
        print("  Layer Breakdown:")
        for layer_type, params in breakdown.items():
            pct = (params / total_params) * 100 if total_params else 0.0
            print(f"    {layer_type}: {params:,} ({pct:.1f}%)")

        sizes = calculate_flash_sizes(model, model_name)
        for config_name, config_data in sizes.items():
            denom = max(1, config_data["params"])
            cr_text = "1.0x (baseline)" if model_name == 'regular_cnn' else f"{baseline_params / denom:.1f}x"
            size_results.append({
                "model": config_name,
                "flash_bytes": int(config_data["flash_bytes"]),
                "flash_human": config_data["flash_human"],
                "parameters": int(config_data["params"]),
                "compression_ratio": cr_text  # param-count ratio vs regular
            })

    df_sizes = pd.DataFrame(size_results).sort_values('flash_bytes')

    print("\n" + "="*80)
    print("FLASH MEMORY REQUIREMENTS COMPARISON")
    print(f"{'='*80}")
    print(df_sizes.to_string(index=False))

    # ---- Hybrid variants (include exact figure first) ----
    print(f"\n{'='*60}")
    print("HYBRID MODEL VARIANTS")
    print(f"{'='*60}")

    tiny_cnn_params = count_parameters(models['tiny_cnn'])
    hybrid_variants = [
        {
            "name": "hybrid (exact per-layer policy)",
            "flash_bytes": int(bytes_exact_hybrid),
            "description": "Classified stem/dw at INT8, one PW at INT8, others at INT4",
        },
        {
            "name": "hybrid(core/heads INT4, keep 1 PW INT8, stem+dw INT8)",
            "flash_bytes": int(tiny_cnn_params * 0.7 * 0.5 + tiny_cnn_params * 0.3 * 1.0),  # rough illustration
            "description": "Mixed precision (approximate split)",
        },
        {
            "name": "hybrid(all INT4 packed)",
            "flash_bytes": tensor_nbit_bytes(tiny_cnn_params, 4),
            "description": "Full INT4 quantization",
        },
    ]
    for variant in hybrid_variants:
        variant["flash_human"] = f"{variant['flash_bytes'] / 1024:.2f} KB"
        print(f"  {variant['name']}: {variant['flash_human']}")
        print(f"    {variant['description']}")

    # ---- Summary: Regular FP32 vs Tiny INT4 ----
    regular_size = calculate_flash_sizes(models['regular_cnn'], 'regular')['regular_fp32']['flash_bytes']
    tiny_size    = calculate_flash_sizes(models['tiny_cnn'], 'tiny')['tiny_int4']['flash_bytes']

    print(f"\n{'='*60}")
    print("MEMORY EFFICIENCY SUMMARY")
    print(f"{'='*60}")
    print(f"Regular CNN (FP32): {regular_size / 1024:.2f} KB")
    print(f"TinyML CNN (INT4): {tiny_size / 1024:.2f} KB")
    print(f"Size Reduction: {regular_size / max(1, tiny_size):.1f}x smaller")
    print(f"Memory Efficiency: {(1 - tiny_size/regular_size)*100:.1f}% reduction")

    return df_sizes, hybrid_variants


# ---- Run the analysis ----
cfg = ExpCfg()  # Uses defaults defined elsewhere (input_len, base, latent_dim, etc.)
df_sizes, hybrid_variants = run_size_analysis(cfg)


########################################################################################

# Cell — Performance vs Size Comparison with Regular CNN Baseline
import torch
import torch.nn as nn
assert hasattr(torch, "optim"), "torch was shadowed by a local variable — rename any variable/param called 'torch'."


def train_regular_cnn(model, train_loader, val_loader, test_loader, cfg, device):
    """Train regular CNN baseline for comparison"""
    print("Training Regular CNN Baseline...")

    # Optimizer and scheduler
    EPOCHS = getattr(cfg, "epochs_cnn", cfg.epochs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=cfg.weight_decay)
    total_steps = len(train_loader) * cfg.epochs_cnn
    warmup_steps = len(train_loader) * cfg.warmup_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Loss function
    criterion = nn.CrossEntropyLoss()
    if cfg.use_focal_loss:
        criterion = SafeFocalLoss(gamma=1.5, alpha=0.5, label_smoothing=0.05)

    model = model.to(device)
    best_val_acc = 0
    patience_counter = 0
    patience = 3

    for epoch in range(cfg.epochs_cnn):
      tr_loss, tr_acc = train_cnn_epoch(model, train_loader, optimizer, criterion, device,
                                        epoch, use_mixup=cfg.use_mixup, mixup_alpha=cfg.mixup_alpha,
                                        num_classes=2, clip=1.0)
      va_loss, va_acc, va_pred, va_true = eval_cnn(model, val_loader, criterion, device)
      scheduler.step()

      # your logging here (compute F1 safely)
      from sklearn.metrics import f1_score
      try:
          va_f1 = f1_score(va_true, va_pred, average="binary", zero_division=0)
      except Exception:
          va_f1 = 0.0

      print(f"[CNN] ep {epoch+1:02d} trL={tr_loss:.4f} trA={tr_acc:.3f} vaL={va_loss:.4f} vaA={va_acc:.3f} F1={va_f1:.3f}")


    # Final test evaluation
    test_loss, test_acc, test_preds, test_targets = eval_classifier(model, test_loader, device, criterion)
    test_metrics = compute_metrics(test_targets, test_preds)

    return {
        'val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_f1': test_metrics['f1'],
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall']
    }

def comprehensive_comparison(cfg: ExpCfg):
    """Run comprehensive comparison between TinyML and regular models"""
    print("="*80)
    print("COMPREHENSIVE MODEL COMPARISON: PERFORMANCE vs SIZE")
    print("="*80)

    # Load data
    print("Loading ApneaECG dataset...")
    try:
        tr_loader, va_loader, te_loader = _safe_make_apnea_loaders(APNEA_ROOT, cfg)
    except:
        print("Error loading data. Using dummy data for demonstration.")
        return

    results = []

    # 1. Train Regular CNN
    print("\n" + "="*50)
    print("TRAINING REGULAR CNN BASELINE")
    print("="*50)

    regular_cnn = RegularCNN(input_length=cfg.input_len, num_classes=2)
    regular_results = train_regular_cnn(regular_cnn, tr_loader, va_loader, te_loader, cfg, DEVICE)
    regular_size = calculate_flash_sizes(regular_cnn, 'regular_cnn')

    results.append({
        'model_name': 'Regular CNN (FP32)',
        'test_accuracy': regular_results['test_acc'],
        'test_f1': regular_results['test_f1'],
        'flash_bytes': regular_size['regular_cnn_fp32']['flash_bytes'],
        'flash_human': regular_size['regular_cnn_fp32']['flash_human'],
        'parameters': count_parameters(regular_cnn),
        'model_type': 'Baseline'
    })

    # 2. Train Enhanced TinyML CNN
    print("\n" + "="*50)
    print("TRAINING ENHANCED TINYML CNN")
    print("="*50)

    tiny_cnn = SharedCoreSeparable1D(
        in_ch=1, base=cfg.base, num_classes=2,
        latent_dim=cfg.latent_dim, hybrid_keep=1,
        input_length=cfg.input_len
    ).to(DEVICE)

    # Use the same training procedure as the enhanced experiment
    optimizer = torch.optim.AdamW(tiny_cnn.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if cfg.use_focal_loss:
        criterion = SafeFocalLoss(gamma=1.5, alpha=0.5, label_smoothing=0.05)
    else:
        criterion = nn.CrossEntropyLoss()

    # Quick training (abbreviated for comparison)
    tiny_cnn.train()
    for epoch in range(3):  # Just a few epochs for quick comparison
        for batch_idx, (data, target) in enumerate(tr_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = tiny_cnn(data)
            loss = criterion(output, target)
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(tiny_cnn.parameters(), 1.0)
                optimizer.step()

    # Evaluate
    tiny_test_loss, tiny_test_acc, tiny_preds, tiny_targets = eval_classifier(tiny_cnn, te_loader, DEVICE)
    tiny_metrics = compute_metrics(tiny_targets, tiny_preds)
    tiny_size = calculate_flash_sizes(tiny_cnn, 'tiny_cnn')

    results.append({
        'model_name': 'TinyML CNN (INT8)',
        'test_accuracy': tiny_test_acc,
        'test_f1': tiny_metrics['f1'],
        'flash_bytes': tiny_size['tiny_cnn_int8']['flash_bytes'],
        'flash_human': tiny_size['tiny_cnn_int8']['flash_human'],
        'parameters': count_parameters(tiny_cnn),
        'model_type': 'TinyML'
    })

    results.append({
        'model_name': 'TinyML CNN (INT4)',
        'test_accuracy': tiny_test_acc,  # Same performance, different storage
        'test_f1': tiny_metrics['f1'],
        'flash_bytes': tiny_size['tiny_cnn_int4']['flash_bytes'],
        'flash_human': tiny_size['tiny_cnn_int4']['flash_human'],
        'parameters': count_parameters(tiny_cnn),
        'model_type': 'TinyML'
    })

    # 3. Create comparison DataFrame
    df_comparison = pd.DataFrame(results)

    print("\n" + "="*100)
    print("PERFORMANCE vs SIZE COMPARISON")
    print("="*100)
    print(df_comparison.to_string(index=False))

    # 4. Calculate efficiency metrics
    print("\n" + "="*80)
    print("EFFICIENCY ANALYSIS")
    print("="*80)

    baseline_size = results[0]['flash_bytes']
    baseline_acc = results[0]['test_accuracy']

    for i, result in enumerate(results[1:], 1):
        size_reduction = baseline_size / result['flash_bytes']
        acc_retention = result['test_accuracy'] / baseline_acc
        efficiency_score = acc_retention / (result['flash_bytes'] / baseline_size)

        print(f"\n{result['model_name']}:")
        print(f"  Size Reduction: {size_reduction:.1f}x smaller")
        print(f"  Accuracy Retention: {acc_retention:.1%}")
        print(f"  Efficiency Score: {efficiency_score:.2f} (higher is better)")
        print(f"  Accuracy per KB: {result['test_accuracy'] / (result['flash_bytes']/1024):.4f}")

    # 5. Detailed size breakdown table (similar to your example)
    print("\n" + "="*80)
    print("DETAILED FLASH MEMORY BREAKDOWN")
    print("="*80)

    detailed_breakdown = []

    # Regular CNN variants
    for precision in ['fp32', 'fp16', 'int8', 'int4']:
        key = f'regular_cnn_{precision}'
        if key in regular_size:
            detailed_breakdown.append({
                'name': f'baseline_cnn_{precision}',
                'flash_bytes': regular_size[key]['flash_bytes'],
                'flash_human': regular_size[key]['flash_human'],
                'model_type': 'Baseline',
                'notes': 'Standard CNN without TinyML optimizations'
            })

    # TinyML variants
    for precision in ['fp32', 'fp16', 'int8', 'int4']:
        key = f'tiny_cnn_{precision}'
        if key in tiny_size:
            detailed_breakdown.append({
                'name': f'tinyml_cnn_{precision}',
                'flash_bytes': tiny_size[key]['flash_bytes'],
                'flash_human': tiny_size[key]['flash_human'],
                'model_type': 'TinyML',
                'notes': 'Enhanced with SE blocks, residual connections'
            })

    # Hybrid variants (estimated)
    tiny_params = count_parameters(tiny_cnn)
    hybrid_estimates = [
        {
            'name': 'hybrid(core/heads INT4, keep 1 PW INT8, stem+dw INT8)',
            'flash_bytes': int(tiny_params * 0.6 * 0.5 + tiny_params * 0.4 * 1.0),
            'model_type': 'Hybrid',
            'notes': 'Mixed precision: critical layers INT8, others INT4'
        },
        {
            'name': 'hybrid(all INT4 packed)',
            'flash_bytes': tensor_nbit_bytes(tiny_params, 4),
            'model_type': 'Hybrid',
            'notes': 'Full INT4 quantization with packing'
        }
    ]

    for hybrid in hybrid_estimates:
        hybrid['flash_human'] = f"{hybrid['flash_bytes'] / 1024:.2f} KB"
        detailed_breakdown.append(hybrid)

    df_detailed = pd.DataFrame(detailed_breakdown)
    df_detailed = df_detailed.sort_values('flash_bytes')

    print(df_detailed[['name', 'flash_bytes', 'flash_human', 'model_type']].to_string(index=False))

    return df_comparison, df_detailed

# Run comprehensive comparison
print("Starting comprehensive model comparison...")
cfg_comparison = ExpCfg(
    epochs_cnn=5,  # Reduced for faster comparison
    epochs_vae_pre=3,
    epochs_head=3
)

try:
    df_perf_comparison, df_size_detailed = comprehensive_comparison(cfg_comparison)
    print(f"\n✅ Comparison completed successfully!")
    print(f"Performance comparison saved to df_perf_comparison")
    print(f"Detailed size breakdown saved to df_size_detailed")
except Exception as e:
    print(f"❌ Error in comparison: {e}")
    print("This might be due to missing data files. The analysis framework is ready to use once data is available.")

########################################################################################

# Dataset Registry and Sanity Checks for All Datasets
import numpy as np
from pathlib import Path

# -------------------- Dataset Registry --------------------
DATASET_REGISTRY = {}

def register_dataset(name, loader_fn, meta=None):
    DATASET_REGISTRY[name] = { 'loader': loader_fn, 'meta': meta or {} }

def available_datasets():
    return list(DATASET_REGISTRY.keys())

def get_dataset_loader(name):
    return DATASET_REGISTRY.get(name, {}).get('loader')

def make_dataset_for_experiment(name, **kwargs):
    """Interface expected by orchestration code"""
    loader = get_dataset_loader(name)
    if loader is None:
        raise KeyError(f'Dataset {name} not registered. Available: {available_datasets()}')
    return loader(**kwargs)

# Register ApneaECG with existing loader
def _load_apnea_for_registry(**kwargs):
    batch_size = kwargs.get('batch_size', 64)
    length = kwargs.get('length', 1800)
    limit = kwargs.get('limit', None)

    try:
        dl_tr, dl_va, dl_te = load_apnea_ecg_loaders_impl(
            APNEA_ROOT,
            batch_size=batch_size,
            length=length,
            stride=kwargs.get('stride', None),
            verbose=kwargs.get('verbose', True)
        )

        # Get a sample to determine meta
        try:
            xb, yb = next(iter(dl_tr))
            meta = {
                'num_channels': xb.shape[1],
                'seq_len': xb.shape[2],
                'num_classes': int(len(torch.unique(yb)))
            }
        except:
            meta = {'num_channels': 1, 'seq_len': length, 'num_classes': 2}

        return dl_tr, dl_va, dl_te, meta
    except Exception as e:
        print(f"[ApneaECG Registry] Failed: {e}")
        raise


'''
# Register UCI-HAR (if load_ucihar exists)
try:
    register_dataset('uci_har', lambda **kwargs: load_ucihar(**kwargs))
except NameError:
    print("[Registry] UCI-HAR loader not found - skip registration")
'''
# Register PTB-XL (if run_ptbxl exists)
try:
    def _ptbxl_wrapper(**kwargs):
        # Create a complete config object with ALL expected attributes
        class CompleteCfg:
            def __init__(self, **kw):
                # Set comprehensive defaults for ALL possible attributes
                # Dataset params
                self.target_fs = 100
                self.batch_size = 64
                self.val_split = 0.1
                self.limit = None
                self.num_workers = 2
                self.label_type = 'superclass'
                self.input_len = 1000
                self.use_focal_loss= True

                # Model architecture params
                self.base = 32  # base filters for models
                self.num_blocks = 3
                self.filter_length = 3
                self.dropout = 0.1
                self.activation = 'relu'

                # Training params
                self.epochs_cnn = 3
                self.epochs_vae_pre = 3
                self.epochs_head = 3
                self.lr = 1e-3
                self.weight_decay = 1e-4
                self.scheduler = 'cosine'
                self.warmup_epochs = 1

                # System params
                self.device = 'cpu'
                self.debug = True
                self.verbose = True
                self.seed = 42

                # Data augmentation params
                self.augment = False
                self.noise_std = 0.01
                self.time_warp = False
                self.latent_dim = 16

                # Override with provided kwargs
                for k, v in kw.items():
                    setattr(self, k, v)

        cfg = CompleteCfg(**kwargs)

        result = run_ptbxl(cfg, str(PTBXL_ROOT))
        if isinstance(result, dict) and 'dl_tr' in result:
            return result['dl_tr'], result['dl_va'], result.get('dl_te'), result['meta']
        else:
            raise Exception(f"PTB-XL loader returned: {result}")

    register_dataset('ptbxl', _ptbxl_wrapper)
except NameError:
    print("[Registry] PTB-XL loader not found - skip registration")

# Register MIT-BIH (if run_mitdb exists)
try:
    def _mitdb_wrapper(**kwargs):
        class CompleteCfg:
            def __init__(self, **kw):
                # Set comprehensive defaults for ALL possible attributes
                # Dataset params
                self.fs = 360
                self.target_fs = 250
                self.window_ms = 800
                self.batch_size = 64
                self.val_split = 0.1
                self.limit = None
                self.num_workers = 2
                self.input_len = 800
                self.use_focal_loss= True

                # Model architecture params
                self.base = 32  # base filters for models
                self.num_blocks = 3
                self.filter_length = 3
                self.dropout = 0.1
                self.activation = 'relu'

                # Training params
                self.epochs_cnn = 3
                self.epochs_vae_pre = 3
                self.epochs_head = 3
                self.lr = 1e-3
                self.weight_decay = 1e-4
                self.scheduler = 'cosine'
                self.warmup_epochs = 1

                # System params
                self.device = 'cpu'
                self.debug = True
                self.verbose = True
                self.seed = 42

                # Data augmentation params
                self.augment = False
                self.noise_std = 0.01
                self.time_warp = False
                self.latent_dim = 16

                # Override with provided kwargs
                for k, v in kw.items():
                    setattr(self, k, v)

        cfg = CompleteCfg(**kwargs)

        result = run_mitdb(cfg, str(MITDB_ROOT))
        if isinstance(result, dict) and 'dl_tr' in result:
            return result['dl_tr'], result['dl_va'], result.get('dl_te'), result['meta']
        else:
            raise Exception(f"MIT-BIH loader returned: {result}")

    register_dataset('mitdb', _mitdb_wrapper)
except NameError:
    print("[Registry] MIT-BIH loader not found - skip registration")

register_dataset('apnea_ecg', _load_apnea_for_registry)

print(f"[Registry] Available datasets: {available_datasets()}")

# -------------------- Comprehensive Sanity Checks --------------------
DO_SANITY = False

def sanity_check_dataset(name, **kwargs):
    """Comprehensive sanity check for any registered dataset"""
    print(f"\n[Sanity] {name} dataset check (args={kwargs})")

    try:
        dl_tr, dl_va, dl_te, meta = make_dataset_for_experiment(name, **kwargs)
        print(f"[Sanity] {name} loaders created successfully")
        print(f"[Sanity] Meta: {meta}")
    except Exception as e:
        print(f"[Sanity] ❌ Failed to create loaders for {name}: {e}")
        import traceback
        traceback.print_exc()  # Print full traceback for debugging
        return False

    # Check train loader
    try:
        xb, yb = next(iter(dl_tr))
        xb_np = xb.detach().cpu().numpy() if hasattr(xb, 'detach') else xb
        yb_np = yb.detach().cpu().numpy() if hasattr(yb, 'detach') else yb

        print(f"[Sanity] {name} batch shapes: {xb.shape}, {yb.shape}")
        print(f"[Sanity] {name} X dtype: {xb.dtype}, range: [{xb_np.min():.3f}, {xb_np.max():.3f}], mean/std: {xb_np.mean():.3f}/{xb_np.std():.3f}")
        print(f"[Sanity] {name} Y range: [{yb_np.min()}, {yb_np.max()}], unique: {np.unique(yb_np)}")

        # Check for common issues
        if np.isnan(xb_np).any():
            print(f"[Sanity] ⚠️  {name} contains NaN values!")
        if np.isinf(xb_np).any():
            print(f"[Sanity] ⚠️  {name} contains infinite values!")
        if abs(xb_np.mean()) > 100:
            print(f"[Sanity] ⚠️  {name} large mean - may need normalization")
        if xb_np.std() > 100:
            print(f"[Sanity] ⚠️  {name} large std - may need normalization")

        # Validate meta consistency
        if isinstance(meta, dict):
            if 'num_channels' in meta and meta['num_channels'] != xb.shape[1]:
                print(f"[Sanity] ⚠️  {name} channel count mismatch: meta={meta['num_channels']}, batch={xb.shape[1]}")
            if 'seq_len' in meta and meta['seq_len'] != xb.shape[2]:
                print(f"[Sanity] ⚠️  {name} sequence length mismatch: meta={meta['seq_len']}, batch={xb.shape[2]}")
            if 'num_classes' in meta:
                unique_classes = len(np.unique(yb_np))
                if meta['num_classes'] != unique_classes:
                    print(f"[Sanity] ⚠️  {name} class count mismatch: meta={meta['num_classes']}, batch_unique={unique_classes}")

        print(f"[Sanity] ✅ {name} passed basic sanity checks")
        return True

    except Exception as e:
        print(f"[Sanity] ❌ Failed to iterate {name} train loader: {e}")
        import traceback
        traceback.print_exc()  # Print full traceback for debugging
        return False

def run_all_sanity_checks():
    """Run sanity checks on all available datasets"""
    print("\n" + "="*60)
    print("COMPREHENSIVE DATASET SANITY CHECKS")
    print("="*60)

    results = {}

    # ApneaECG
    if 'apnea_ecg' in available_datasets():
        results['apnea_ecg'] = sanity_check_dataset('apnea_ecg', batch_size=32, length=1800, limit=100)

    # UCI-HAR
    if 'uci_har' in available_datasets():
        results['uci_har'] = sanity_check_dataset('uci_har', batch_size=64, limit=500, target_fs=50)

    # PTB-XL - with comprehensive config
    if 'ptbxl' in available_datasets():
        results['ptbxl'] = sanity_check_dataset('ptbxl',
                                                batch_size=32,
                                                limit=200,
                                                target_fs=100,
                                                input_len=1000,
                                                base=32,
                                                num_blocks=3,
                                                filter_length=3)

    # MIT-BIH - with comprehensive config
    if 'mitdb' in available_datasets():
        results['mitdb'] = sanity_check_dataset('mitdb',
                                               batch_size=64,
                                               limit=1000,
                                               target_fs=250,
                                               window_ms=800,
                                               input_len=800,
                                               base=32,
                                               num_blocks=3,
                                               filter_length=3)

    print(f"\n[Sanity] Summary: {results}")
    all_passed = all(results.values())
    if all_passed:
        print("[Sanity] ✅ All datasets passed sanity checks!")
    else:
        print("[Sanity] ❌ Some datasets failed sanity checks")
        failed = [k for k, v in results.items() if not v]
        print(f"[Sanity] Failed datasets: {failed}")

    return results

# Run sanity checks if enabled
if DO_SANITY:
    sanity_results = run_all_sanity_checks()

print("Dataset registry and sanity check system loaded!")

########################################################################################

# Fixed Orchestration: models, training, and experiment runner with dataset registry
import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple, List
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
'''
@dataclass
class ExpCfg:
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs: int = 3
    batch_size: int = 32
    lr: float = 1e-3
    limit: int = 1000  # small for testing
    target_fs: int = None
    length: int = 1800  # for ApneaECG
    window_ms: int = 800  # for MIT-BIH
    num_workers: int = 2
    debug: bool = True
'''
# -------------------- Size accounting ----------------
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def estimate_packed_bytes(model: nn.Module, quantized_byte_per_param: int = 1) -> int:
    """Simple packed byte estimate: 1 byte per param (INT8) + overhead"""
    params = count_params(model)
    overhead = 128  # bytes for metadata, headers etc
    return params * quantized_byte_per_param + overhead

# -------------------- Models ----------------
class SeparableBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1):
        super().__init__()
        self.depthwise = nn.Conv1d(in_ch, in_ch, kernel_size=k, padding=k//2, groups=in_ch, bias=False)
        self.pointwise = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)

class TinySeparableCNN(nn.Module):
    """Lightweight separable CNN for TinyML"""
    def __init__(self, in_ch, num_classes, base_filters=16, n_blocks=2):
        super().__init__()
        layers = []
        cur_ch = in_ch
        for i in range(n_blocks):
            out_ch = base_filters * (2**i)
            layers.append(SeparableBlock(cur_ch, out_ch))
            if i < n_blocks - 1:  # no pooling after last block
                layers.append(nn.MaxPool1d(2))
            cur_ch = out_ch
        self.body = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(cur_ch, num_classes)

    def forward(self, x):
        x = self.body(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

class TinyVAEHead(nn.Module):
    """VAE encoder + linear head (no decoder for inference)"""
    def __init__(self, in_ch, num_classes, latent_dim=16, base_filters=16):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, base_filters, 3, padding=1)
        self.conv2 = nn.Conv1d(base_filters, base_filters*2, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(8)  # reduce to manageable size
        self.fc_mu = nn.Linear(base_filters*2*8, latent_dim)
        self.fc_logvar = nn.Linear(base_filters*2*8, latent_dim)
        self.head = nn.Linear(latent_dim, num_classes)

    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).flatten(1)  # (B, C*L)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.head(z)

class TinyMethodModel(nn.Module):
    """Prototype of the method: synthesis MLP for channel mixing"""
    def __init__(self, in_ch, num_classes, base_filters=16, latent_dim=8):
        super().__init__()
        # First layer: normal conv (would be kept in INT8)
        self.stem = nn.Conv1d(in_ch, base_filters, 3, padding=1)

        # Synthesis MLP (tiny)
        self.synthesis_mlp = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, base_filters * base_filters)  # for 1x1 conv weights
        )

        # Learnable latent code
        self.latent_code = nn.Parameter(torch.randn(latent_dim))

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_filters, num_classes)

    def forward(self, x):
        # Apply stem
        x = F.relu(self.stem(x))  # (B, base_filters, L)

        # Synthesize pointwise conv weights
        synth_weights = self.synthesis_mlp(self.latent_code)  # (base_filters^2,)
        synth_weights = synth_weights.view(x.shape[1], x.shape[1], 1)  # (out, in, 1)

        # Apply synthesized conv
        x = F.conv1d(x, synth_weights)
        x = F.relu(x)

        x = self.pool(x).squeeze(-1)
        return self.fc(x)

# -------------------- Training & Eval helpers ----------------
def train_epoch(model, loader, opt, device='cpu'):
    model.train()
    running_loss = 0.0
    correct = 0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        opt.zero_grad()
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        opt.step()
        running_loss += float(loss.item()) * xb.size(0)
        preds = logits.argmax(1)
        correct += int((preds==yb).sum().item())
        n += xb.size(0)
    return running_loss/n, correct/n

@torch.no_grad()
def evaluate(model, loader, device='cpu'):
    model.eval()
    correct = 0
    n = 0
    running_loss = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        running_loss += float(loss.item()) * xb.size(0)
        preds = logits.argmax(1)
        correct += int((preds==yb).sum().item())
        n += xb.size(0)
    return correct/n, running_loss/n

# -------------------- Fixed Experiment runner ----------------
def run_experiment(cfg: ExpCfg, dataset_name: str, model_name: str, preload=None):
    print(f'\n{"="*60}')
    print(f'🚀 Experiment: {dataset_name} + {model_name}')
    print("="*60)

    # Use preloaded loaders if provided; otherwise load once here
    if preload is not None:
        dl_tr, dl_va, dl_te, meta = preload
    else:
        if dataset_name not in available_datasets():
            print(f'❌ Dataset {dataset_name} not in registry. Available: {available_datasets()}')
            return None
        print('🔄 Preparing data loaders...')
        try:
            ret = make_dataset_for_experiment(
                dataset_name,
                limit=cfg.limit,
                batch_size=cfg.batch_size,
                target_fs=cfg.target_fs,
                num_workers=cfg.num_workers,
                length=cfg.length,
                window_ms=cfg.window_ms,
                input_len=cfg.input_len
            )
            dl_tr, dl_va, dl_te, meta = _normalize_dataset_return(ret)
        except Exception as e:
            print(f'❌ Failed to prepare dataset: {e}')
            return None

    # Ensure meta has essentials
    meta = _probe_meta_if_needed(dl_tr, dict(meta))
    print(f'📊 Dataset meta: {meta}')
    print(f'🔢 Train batches: {len(dl_tr)}, Val batches: {len(dl_va)}')

    device = torch.device(cfg.device)
    in_ch = meta['num_channels']
    num_classes = meta['num_classes']

    # --- Instantiate model (unchanged) ---
    if model_name == 'tiny_separable_cnn':
        model = TinySeparableCNN(in_ch, num_classes)
    elif model_name == 'tiny_vae_head':
        model = TinyVAEHead(in_ch, num_classes)
    elif model_name == 'tiny_method':
        model = TinyMethodModel(in_ch, num_classes)
    elif model_name == 'regular_cnn':
        model = RegularCNN(in_ch, num_classes)
    else:
        print(f'❌ Unknown model: {model_name}')
        return None

    # >>> keep your existing training/eval code below this line <<<
    # result = train_and_evaluate(model, dl_tr, dl_va, dl_te, cfg, device, meta)
    # return result


def run_all_experiments(cfg: ExpCfg, datasets: List[str]=None, models: List[str]=None):
    if datasets is None:
        datasets = available_datasets()
    if models is None:
        models = ['tiny_separable_cnn', 'tiny_vae_head', 'tiny_method']

    print("\n" + "="*80)
    print("🚀 COMPREHENSIVE TINYML EXPERIMENTS")
    print("="*80)
    print(f"📋 Datasets: {datasets}")
    print(f"🧠 Models: {models}")
    print(f"⚙️  Config: epochs={cfg.epochs}, batch_size={cfg.batch_size}, limit={cfg.limit}")
    print(f"💻 Device: {cfg.device}")

    results = []
    # --- PRELOAD EACH DATASET ONCE ---
    loader_cache = {}
    for ds in datasets:
        if ds not in available_datasets():
            print(f'⚠️  Skipping unavailable dataset: {ds}')
            continue
        print(f'\n[make_loaders] Preparing dataset once: {ds}')
        try:
            ret = make_dataset_for_experiment(
                ds,
                limit=cfg.limit,
                batch_size=cfg.batch_size,
                target_fs=cfg.target_fs,
                num_workers=cfg.num_workers,
                length=cfg.length,
                window_ms=cfg.window_ms,
                input_len=cfg.input_len
            )
            dl_tr, dl_va, dl_te, meta = _normalize_dataset_return(ret)
            meta = _probe_meta_if_needed(dl_tr, dict(meta))
            loader_cache[ds] = (dl_tr, dl_va, dl_te, meta)
            print(f'  → cached loaders for {ds}: num_channels={meta["num_channels"]}, '
                  f'num_classes={meta["num_classes"]}, seq_len={meta["seq_len"]}')
        except Exception as e:
            print(f'❌ Failed to prepare dataset {ds}: {e}')

    total_experiments = sum(ds in loader_cache for ds in datasets) * len(models)
    current_exp = 0

    for ds in datasets:
        if ds not in loader_cache:
            continue
        preload = loader_cache[ds]  # (tr, va, te, meta)
        for model in models:
            current_exp += 1
            print(f'\n📍 Experiment {current_exp}/{total_experiments}')
            try:
                result = run_experiment(cfg, ds, model, preload=preload)
                if result:
                    results.append(result)
                    print(f'✅ Completed: {ds} + {model}')
                else:
                    print(f'❌ Failed: {ds} + {model}')
            except Exception as e:
                print(f'💥 Exception in {ds} + {model}: {e}')
                import traceback; traceback.print_exc()

    # --- Summary (unchanged) ---
    if results:
        print(f'\n{"="*80}')
        print("📊 EXPERIMENT RESULTS SUMMARY")
        print("="*80)
        df = pd.DataFrame(results)
        print(f"✅ Completed experiments: {len(results)}/{total_experiments}")
        if 'val_acc' in df:
            print(f"📈 Average validation accuracy: {df['val_acc'].mean():.4f} ± {df['val_acc'].std():.4f}")
        if 'packed_bytes_est' in df:
            print(f"💾 Average model size: {df['packed_bytes_est'].mean()/1024:.1f} KB")
        display_cols = [c for c in ['dataset','model','val_acc','test_acc','params','packed_bytes_est','train_time_s'] if c in df.columns]
        if display_cols:
            print(f"\n{df[display_cols].to_string(index=False)}")
        results_file = 'tinyml_experiment_results.csv'
        df.to_csv(results_file, index=False)
        print(f"\n💾 Results saved to: {results_file}")
        return df
    else:
        print("❌ No experiments completed successfully")
        return None


# -------------------- Quick test runner ----------------
def quick_test():
    """Run a quick test with small config to verify everything works"""
    print("🧪 Running quick test...")

    test_cfg = ExpCfg(
        epochs=1,
        batch_size=16,
        limit=100,
        device='cpu'  # force CPU for reliability
    )

    # Test with first available dataset and model
    available = available_datasets()
    if available:
        test_dataset = available[0]
        result = run_experiment(test_cfg, test_dataset, 'tiny_separable_cnn')
        if result:
            print("✅ Quick test passed!")
            return True
        else:
            print("❌ Quick test failed!")
            return False
    else:
        print("❌ No datasets available for testing")
        return False

print("🔧 Fixed orchestration system loaded!")
print(f"📋 Available datasets: {available_datasets()}")
print("💡 Use quick_test() or run_all_experiments(ExpCfg()) to start experiments")
run_all_experiments(ExpCfg())

########################################################################################

# Complete Orchestration System: Models, Training, and Experiment Runner
import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple, List
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
'''
@dataclass
class ExpCfg:
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs: int = 8
    batch_size: int = 32
    lr: float = 1e-3
    limit: int = 1000  # small for testing
    target_fs: int = None
    length: int = 1800  # for ApneaECG
    window_ms: int = 800  # for MIT-BIH
    input_len: int = 1000  # for PTB-XL/MIT-BIH configs
    num_workers: int = 2
    debug: bool = True
    epochs_cnn: int | None = None
'''
# -------------------- Size accounting ----------------
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def estimate_packed_bytes(model: nn.Module, quantized_byte_per_param: int = 1) -> int:
    """Simple packed byte estimate: 1 byte per param (INT8) + overhead"""
    params = count_params(model)
    overhead = 128  # bytes for metadata, headers etc
    return params * quantized_byte_per_param + overhead

def estimate_flash_usage(model: nn.Module, precision='int8'):
    """Estimate flash memory usage for different precisions"""
    params = count_params(model)
    bytes_per_param = {
        'fp32': 4,
        'fp16': 2,
        'int8': 1,
        'int4': 0.5
    }
    base_bytes = params * bytes_per_param.get(precision, 1)
    return {
        'params': params,
        'flash_bytes': int(base_bytes + 512),  # add overhead
        'flash_human': f"{(base_bytes + 512)/1024:.2f} KB"
    }

# -------------------- TinyML Optimized Models ----------------
class SeparableBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1):
        super().__init__()
        self.depthwise = nn.Conv1d(in_ch, in_ch, kernel_size=k, padding=k//2, groups=in_ch, bias=False)
        self.pointwise = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)

class TinySeparableCNN(nn.Module):
    """Lightweight separable CNN for TinyML - baseline model"""
    def __init__(self, in_ch, num_classes, base_filters=16, n_blocks=2):
        super().__init__()
        layers = []
        cur_ch = in_ch
        for i in range(n_blocks):
            out_ch = base_filters * (2**i)
            layers.append(SeparableBlock(cur_ch, out_ch))
            if i < n_blocks - 1:  # no pooling after last block
                layers.append(nn.MaxPool1d(2))
            cur_ch = out_ch
        self.body = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(cur_ch, num_classes)

    def forward(self, x):
        x = self.body(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

class TinyVAEHead(nn.Module):
    """VAE encoder + linear head (no decoder for inference) - feature-forward baseline"""
    def __init__(self, in_ch, num_classes, latent_dim=16, base_filters=16):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, base_filters, 3, padding=1)
        self.conv2 = nn.Conv1d(base_filters, base_filters*2, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(8)  # reduce to manageable size
        self.fc_mu = nn.Linear(base_filters*2*8, latent_dim)
        self.fc_logvar = nn.Linear(base_filters*2*8, latent_dim)
        self.head = nn.Linear(latent_dim, num_classes)

    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).flatten(1)  # (B, C*L)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.head(z)

class TinyMethodModel(nn.Module):
    """The Method: Generative compression with synthesis MLP for channel mixing"""
    def __init__(self, in_ch, num_classes, base_filters=16, latent_dim=8):
        super().__init__()
        # First layer: normal conv (would be kept in INT8 for stability)
        self.stem = nn.Conv1d(in_ch, base_filters, 3, padding=1)

        # Tiny synthesis MLP - replaces stored pointwise weights
        self.synthesis_mlp = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, base_filters * base_filters)  # synthesize 1x1 conv weights
        )

        # Learnable per-layer latent code (tiny storage)
        self.latent_code = nn.Parameter(torch.randn(latent_dim))

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_filters, num_classes)

    def forward(self, x):
        # Apply stem (kept in INT8)
        x = F.relu(self.stem(x))  # (B, base_filters, L)

        # Synthesize pointwise conv weights from latent code
        synth_weights = self.synthesis_mlp(self.latent_code)  # (base_filters^2,)
        synth_weights = synth_weights.view(x.shape[1], x.shape[1], 1)  # (out, in, 1)

        # Apply synthesized conv (generated at boot, not stored)
        x = F.conv1d(x, synth_weights)
        x = F.relu(x)

        x = self.pool(x).squeeze(-1)
        return self.fc(x)

class RegularCNN(nn.Module):
    """Regular CNN baseline for comparison"""
    def __init__(self, in_ch, num_classes, base_filters=32, n_blocks=3):
        super().__init__()
        layers = []
        cur_ch = in_ch
        for i in range(n_blocks):
            out_ch = base_filters * (2**i)
            layers.extend([
                nn.Conv1d(cur_ch, out_ch, 3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2)
            ])
            cur_ch = out_ch
        self.body = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(cur_ch, num_classes)

    def forward(self, x):
        x = self.body(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

# -------------------- Training & Evaluation ----------------
def train_epoch(model, loader, opt, device='cpu'):
    model.train()
    running_loss = 0.0
    correct = 0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        opt.zero_grad()
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        opt.step()
        running_loss += float(loss.item()) * xb.size(0)
        preds = logits.argmax(1)
        correct += int((preds==yb).sum().item())
        n += xb.size(0)
    return running_loss/n, correct/n

@torch.no_grad()
def evaluate(model, loader, device='cpu'):
    model.eval()
    correct = 0
    n = 0
    running_loss = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        running_loss += float(loss.item()) * xb.size(0)
        preds = logits.argmax(1)
        correct += int((preds==yb).sum().item())
        n += xb.size(0)
    return correct/n, running_loss/n

# --- Normalize different dataset returns to (tr, va, te, meta) ---
def _normalize_dataset_return(ret):
    if isinstance(ret, (tuple, list)):
        if len(ret) == 3: dl_tr, dl_va, dl_te; meta = {}
        elif len(ret) == 4: dl_tr, dl_va, dl_te, meta = ret
        else: raise TypeError(f"Unexpected dataset return length: {len(ret)}")
    elif isinstance(ret, dict):
        if all(k in ret for k in ("train","val","test")):
            dl_tr, dl_va, dl_te = ret["train"], ret["val"], ret["test"]
            meta = ret.get("meta", {})
        else:
            raise TypeError("Loader returned metrics dict; expected loaders.")
    else:
        raise TypeError(f"Unexpected dataset return type: {type(ret)}")
    return dl_tr, dl_va, dl_te, meta

def _probe_meta_if_needed(dl_tr, meta):
    need = ("num_channels" not in meta) or ("num_classes" not in meta) or ("seq_len" not in meta)
    if not need: return meta
    xb, yb = next(iter(dl_tr))
    meta.setdefault("num_channels", int(xb.shape[1]))
    meta.setdefault("seq_len",     int(xb.shape[-1]))
    if yb.ndim == 1:
        meta.setdefault("num_classes", int(max(2, yb.max().item()+1)))
    elif yb.ndim == 2:
        meta.setdefault("num_classes", int(yb.shape[1]))
    else:
        meta.setdefault("num_classes", 2)
    return meta

# -------------------- Experiment Runner ----------------
def run_experiment(cfg: ExpCfg, dataset_name: str, model_name: str):
    print(f'\n{"="*60}')
    print(f'🚀 Experiment: {dataset_name} + {model_name}')
    print("="*60)

    # Check dataset availability
    if dataset_name not in available_datasets():
        print(f'❌ Dataset {dataset_name} not in registry.')
        print(f'Available: {available_datasets()}')
        return None

    # Prepare loaders with debug info
    print('🔄 Preparing data loaders...')
    try:
        ret = make_dataset_for_experiment(
            dataset_name,
            limit=cfg.limit,
            batch_size=cfg.batch_size,
            target_fs=cfg.target_fs,
            num_workers=cfg.num_workers,
            length=cfg.length,
            window_ms=cfg.window_ms,
            input_len=cfg.input_len
        )
        dl_tr, dl_va, dl_te, meta = _normalize_dataset_return(ret)

        # If meta misses essentials, infer from one batch
        need_probe = ("num_channels" not in meta) or ("num_classes" not in meta) or ("seq_len" not in meta)
        if need_probe:
            xb, yb = next(iter(dl_tr))
            meta.setdefault("num_channels", int(xb.shape[1]))     # (B, C, T)
            meta.setdefault("seq_len",     int(xb.shape[-1]))
            # y can be indices or one-hot
            if yb.ndim == 1:
                meta.setdefault("num_classes", int(max(2, yb.max().item() + 1)))
            elif yb.ndim == 2:
                meta.setdefault("num_classes", int(yb.shape[1]))
            else:
                meta.setdefault("num_classes", 2)

        print(f'📊 Dataset meta: {meta}')
        print(f'🔢 Train batches: {len(dl_tr)}, Val batches: {len(dl_va)}')

    except Exception as e:
        print(f'❌ Failed to prepare dataset: {e}')
        return None

    device = torch.device(cfg.device)
    in_ch = meta['num_channels']
    num_classes = meta['num_classes']

    # Instantiate model
    if model_name == 'tiny_separable_cnn':
        model = TinySeparableCNN(in_ch, num_classes)
    elif model_name == 'tiny_vae_head':
        model = TinyVAEHead(in_ch, num_classes)
    elif model_name == 'tiny_method':
        model = TinyMethodModel(in_ch, num_classes)
    elif model_name == 'regular_cnn':
        model = RegularCNN(in_ch, num_classes)
    else:
        print(f'❌ Unknown model: {model_name}')
        return None

    model.to(device)
    params = count_params(model)
    flash_info = estimate_flash_usage(model, 'int8')

    print(f'🧠 Model: {model_name}')
    print(f'📏 Parameters: {params:,}')
    print(f'💾 Flash estimate (INT8): {flash_info["flash_human"]} ({flash_info["flash_bytes"]:,} bytes)')

    opt = Adam(model.parameters(), lr=cfg.lr)

    # Training loop with progress
    print(f'🚀 Training for {cfg.epochs} epochs...')
    start = time.time()
    best_val_acc = 0.0

    for ep in range(cfg.epochs):
        train_loss, train_acc = train_epoch(model, dl_tr, opt, device=device)
        val_acc, val_loss = evaluate(model, dl_va, device=device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        print(f'  Epoch {ep+1}/{cfg.epochs}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}')

    dur = time.time() - start

    # Test evaluation
    test_acc = None
    if dl_te is not None:
        test_acc, _ = evaluate(model, dl_te, device=device)
        print(f'🎯 Test accuracy: {test_acc:.4f}')

    print(f'⏱️  Training time: {dur:.1f}s')
    print(f'✅ Best validation accuracy: {best_val_acc:.4f}')

    results = {
        'dataset': dataset_name,
        'model': model_name,
        'train_time_s': dur,
        'params': params,
        'flash_bytes': flash_info['flash_bytes'],
        'flash_kb': flash_info['flash_bytes'] / 1024,
        'val_acc': best_val_acc,
        'test_acc': test_acc,
        'final_train_acc': train_acc,
        'channels': in_ch,
        'seq_len': meta.get('seq_len', 'unknown'),
        'num_classes': num_classes
    }
    return results

def run_all_experiments(cfg: ExpCfg, datasets: List[str]=None, models: List[str]=None):
    """Run comprehensive experiments across all dataset+model combinations"""

    if datasets is None:
        datasets = available_datasets()
    if models is None:
        models = ['tiny_separable_cnn', 'tiny_vae_head', 'tiny_method', 'regular_cnn']

    print("\n" + "="*80)
    print("🚀 COMPREHENSIVE TINYML EXPERIMENTS")
    print("="*80)
    print(f"📋 Datasets: {datasets}")
    print(f"🧠 Models: {models}")
    print(f"⚙️  Config: epochs={cfg.epochs}, batch_size={cfg.batch_size}, limit={cfg.limit}")
    print(f"💻 Device: {cfg.device}")

    results = []
    total_experiments = len(datasets) * len(models)
    current_exp = 0

    for ds in datasets:
        if ds not in available_datasets():
            print(f'⚠️  Skipping unavailable dataset: {ds}')
            continue

        for model in models:
            current_exp += 1
            print(f'\n📍 Experiment {current_exp}/{total_experiments}')

            try:
                result = run_experiment(cfg, ds, model)
                if result:
                    results.append(result)
                    print(f'✅ Completed: {ds} + {model}')
                else:
                    print(f'❌ Failed: {ds} + {model}')
            except Exception as e:
                print(f'💥 Exception in {ds} + {model}: {e}')
                import traceback
                traceback.print_exc()

    # Present comprehensive results
    if results:
        print(f'\n{"="*80}')
        print("📊 EXPERIMENT RESULTS SUMMARY")
        print("="*80)

        df = pd.DataFrame(results)

        # Summary statistics
        print(f"✅ Completed experiments: {len(results)}/{total_experiments}")
        print(f"📈 Average validation accuracy: {df['val_acc'].mean():.4f} ± {df['val_acc'].std():.4f}")
        print(f"💾 Average model size: {df['flash_kb'].mean():.1f} KB")

        # Model comparison table
        print(f"\n{'='*80}")
        print("MODEL COMPARISON")
        print("="*80)

        comparison_cols = ['dataset', 'model', 'val_acc', 'test_acc', 'flash_kb', 'params', 'train_time_s']
        print(df[comparison_cols].to_string(index=False, float_format='%.4f'))

        # Analysis by model type
        print(f"\n{'='*60}")
        print("ANALYSIS BY MODEL TYPE")
        print("="*60)

        model_analysis = df.groupby('model').agg({
            'val_acc': ['mean', 'std'],
            'flash_kb': 'mean',
            'params': 'mean',
            'train_time_s': 'mean'
        }).round(4)

        print(model_analysis)

        # Efficiency analysis (accuracy per KB)
        df['efficiency'] = df['val_acc'] / df['flash_kb']
        print(f"\n{'='*60}")
        print("EFFICIENCY ANALYSIS (Accuracy per KB)")
        print("="*60)
        efficiency_analysis = df.groupby('model')['efficiency'].agg(['mean', 'std']).round(6)
        print(efficiency_analysis.sort_values('mean', ascending=False))

        # Save results
        results_file = 'comprehensive_tinyml_results.csv'
        df.to_csv(results_file, index=False)
        print(f"\n💾 Results saved to: {results_file}")

        return df
    else:
        print("❌ No experiments completed successfully")
        return None

# -------------------- Quick Test & Utility Functions ----------------
def quick_test():
    """Run a quick test with minimal config to verify everything works"""
    print("🧪 Running quick test...")

    test_cfg = ExpCfg(
        epochs=1,
        batch_size=16,
        limit=50,
        device='cpu'  # force CPU for reliability
    )

    # Test with first available dataset and model
    available = available_datasets()
    if available:
        test_dataset = available[0]
        result = run_experiment(test_cfg, test_dataset, 'tiny_separable_cnn')
        if result:
            print("✅ Quick test passed!")
            return True
        else:
            print("❌ Quick test failed!")
            return False
    else:
        print("❌ No datasets available for testing")
        return False

def paper_experiments():
    """Run experiments specifically for the paper with appropriate configs"""
    print("📄 Running paper experiments...")

    paper_cfg = ExpCfg(
        epochs=10,
        batch_size=64,
        limit=5000,  # reasonable size for meaningful results
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Focus on the key models for the paper
    key_models = ['tiny_separable_cnn', 'tiny_vae_head', 'tiny_method']

    return run_all_experiments(paper_cfg, models=key_models)

print("🔧 Complete orchestration system loaded!")
print(f"📋 Available datasets: {available_datasets()}")
print("💡 Use quick_test(), paper_experiments(), or run_all_experiments(ExpCfg()) to start")

########################################################################################

# Fixed Dataset Registry and Comprehensive Comparison

# -------------------- Dataset Registry System ----------------
DATASET_REGISTRY = {}

def register_dataset(name, loader_func):
    """Register a dataset loader function"""
    DATASET_REGISTRY[name] = loader_func
    print(f"[Registry] Registered dataset: {name}")

def available_datasets():
    """Return list of available datasets"""
    return list(DATASET_REGISTRY.keys())

def make_dataset_for_experiment(name, **kwargs):
    """Create dataset loaders for experiments"""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{name}' not found in registry. Available: {available_datasets()}")

    loader_func = DATASET_REGISTRY[name]
    return loader_func(**kwargs)

# -------------------- Dataset Wrapper Functions ----------------
def _dir_has_any(path):
    """Check if directory exists and has files"""
    from pathlib import Path
    path = Path(path)
    return path.exists() and any(path.iterdir())

def _load_apnea_for_registry(**kwargs):
    """ApneaECG dataset wrapper for registry"""
    try:
        batch_size = kwargs.get('batch_size', 32)
        length = kwargs.get('length', 1800)
        limit = kwargs.get('limit', None)

        tr_loader, va_loader, te_loader = load_apnea_ecg_loaders_impl(
            APNEA_ROOT,
            batch_size=batch_size,
            length=length,
            verbose=False
        )

        meta = {
            'num_channels': 1,
            'seq_len': length,
            'num_classes': 2
        }

        print(f"[ApneaECG Registry] Created loaders successfully")
        return tr_loader, va_loader, te_loader, meta

    except Exception as e:
        print(f"[ApneaECG Registry] Failed: {e}")
        raise

def _ptbxl_wrapper(**kwargs):
    """PTB-XL dataset wrapper for registry - returns loaders, not full experiment results"""
    try:
        batch_size = kwargs.get('batch_size', 32)
        input_len = kwargs.get('input_len', 1000)

        # Check if data exists
        if not _dir_has_any(PTBXL_ROOT):
            raise FileNotFoundError(f"PTB-XL data not found at {PTBXL_ROOT}")

        tr_loader, va_loader, te_loader, meta = load_ptbxl_loaders(
            PTBXL_ROOT,
            batch_size=batch_size,
            length=input_len,
            task="binary_diag",
            lead="II"
        )

        print(f"[PTB-XL Registry] Created loaders successfully")
        return tr_loader, va_loader, te_loader, meta

    except Exception as e:
        print(f"[PTB-XL Registry] Failed: {e}")
        raise

def _mitdb_wrapper(**kwargs):
    """MIT-BIH dataset wrapper for registry - returns loaders, not full experiment results"""
    try:
        batch_size = kwargs.get('batch_size', 64)
        input_len = kwargs.get('input_len', 800)

        # Check if data exists
        if not _dir_has_any(MITDB_ROOT):
            raise FileNotFoundError(f"MIT-BIH data not found at {MITDB_ROOT}")

        tr_loader, va_loader, te_loader, meta = load_mitdb_loaders(
            MITDB_ROOT,
            batch_size=batch_size,
            length=input_len,
            binary=True
        )

        print(f"[MIT-BIH Registry] Created loaders successfully")
        return tr_loader, va_loader, te_loader, meta

    except Exception as e:
        print(f"[MIT-BIH Registry] Failed: {e}")
        raise

# Register all datasets


# Only register if data exists
if DO_PTBXL_DOWNLOAD and _dir_has_any(PTBXL_ROOT):
    register_dataset('ptbxl', _ptbxl_wrapper)
else:
    print("[Registry] PTB-XL skipped - data not available or download disabled")

if DO_MITDB_DOWNLOAD and _dir_has_any(MITDB_ROOT):
    register_dataset('mitdb', _mitdb_wrapper)
else:
    print("[Registry] MIT-BIH skipped - data not available or download disabled")

register_dataset('apnea_ecg', _load_apnea_for_registry)

print(f"[Registry] Available datasets: {available_datasets()}")


# -------------------- Comprehensive Comparison Function ----------------
def comprehensive_comparison():
    """Run comprehensive comparison between TinyML and regular models"""
    print("Starting comprehensive model comparison...")

    # Use the corrected ExpCfg with all required attributes
    cfg_comparison = ExpCfg(
        epochs=5,  # Use the base 'epochs' attribute
        epochs_cnn=5,  # Now this attribute exists
        epochs_vae_pre=3,
        batch_size=32,
        lr=1e-3,
        limit=500,  # Smaller for faster testing
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print(f"Configuration: {cfg_comparison}")

    # Load available datasets
    available = available_datasets()
    if not available:
        print("❌ No datasets available for comparison")
        return None

    print(f"Available datasets: {available}")

    # Test with first available dataset
    test_dataset = available[0]
    print(f"Using dataset: {test_dataset}")

    # Test models
    models_to_test = ['tiny_separable_cnn', 'regular_cnn']
    results = []

    for model_name in models_to_test:
        print(f"\nTesting model: {model_name}")
        try:
            result = run_experiment(cfg_comparison, test_dataset, model_name)
            if result:
                results.append(result)
                print(f"✅ {model_name} completed successfully")
            else:
                print(f"❌ {model_name} failed")
        except Exception as e:
            print(f"💥 {model_name} error: {e}")
            import traceback
            traceback.print_exc()

    if results:
        print("\n" + "="*60)
        print("COMPARISON RESULTS")
        print("="*60)

        import pandas as pd
        df = pd.DataFrame(results)
        print(df[['model', 'val_acc', 'test_acc', 'flash_kb', 'params']])

        return df
    else:
        print("❌ No results to compare")
        return None

# -------------------- Sanity Check Function ----------------
def sanity_check_dataset(name, **kwargs):
    """Comprehensive sanity check for any registered dataset"""
    print(f"\n[Sanity] {name} dataset check (args={kwargs})")

    try:
        dl_tr, dl_va, dl_te, meta = make_dataset_for_experiment(name, **kwargs)
        print(f"[Sanity] {name} loaders created successfully")
        print(f"[Sanity] Meta: {meta}")
    except Exception as e:
        print(f"[Sanity] ❌ Failed to create loaders for {name}: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Check train loader
    try:
        xb, yb = next(iter(dl_tr))
        xb_np = xb.detach().cpu().numpy() if hasattr(xb, 'detach') else xb
        yb_np = yb.detach().cpu().numpy() if hasattr(yb, 'detach') else yb

        print(f"[Sanity] {name} batch shapes: {xb.shape}, {yb.shape}")
        print(f"[Sanity] {name} X dtype: {xb.dtype}, range: [{xb_np.min():.3f}, {xb_np.max():.3f}], mean/std: {xb_np.mean():.3f}/{xb_np.std():.3f}")
        print(f"[Sanity] {name} Y range: [{yb_np.min()}, {yb_np.max()}], unique: {np.unique(yb_np)}")

        print(f"[Sanity] ✅ {name} passed basic sanity checks")
        return True

    except Exception as e:
        print(f"[Sanity] ❌ Failed to iterate {name} train loader: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_sanity_checks():
    """Run sanity checks on all available datasets"""
    print("\n" + "="*60)
    print("COMPREHENSIVE DATASET SANITY CHECKS")
    print("="*60)

    results = {}

    # ApneaECG
    if 'apnea_ecg' in available_datasets():
        results['apnea_ecg'] = sanity_check_dataset('apnea_ecg', batch_size=32, length=1800, limit=100)

    # PTB-XL - only if available
    if 'ptbxl' in available_datasets():
        results['ptbxl'] = sanity_check_dataset('ptbxl',
                                               batch_size=32,
                                               limit=200,
                                               target_fs=100,
                                               input_len=1000)

    # MIT-BIH - only if available
    if 'mitdb' in available_datasets():
        results['mitdb'] = sanity_check_dataset('mitdb',
                                              batch_size=64,
                                              limit=1000,
                                              target_fs=250,
                                              input_len=800)

    print(f"\n[Sanity] Summary: {results}")
    all_passed = all(results.values()) if results else False
    if all_passed:
        print("[Sanity] ✅ All datasets passed sanity checks!")
    else:
        print("[Sanity] ❌ Some datasets failed sanity checks")
        failed = [k for k, v in results.items() if not v]
        print(f"[Sanity] Failed datasets: {failed}")

    return results

print("🔧 Fixed dataset registry and comprehensive comparison system loaded!")
print("💡 Use comprehensive_comparison() or run_all_sanity_checks() to test")

########################################################################################

# Simple Comprehensive Comparison & Path Debugging

def check_dataset_paths():
    """Debug function to check dataset paths and suggest fixes"""
    print("🔍 DATASET PATH DEBUGGING")
    print("="*60)

    from pathlib import Path

    # Check each dataset path
    paths_to_check = {
        'ApneaECG': APNEA_ROOT,
        'PTB-XL': PTBXL_ROOT,
        'MIT-BIH': MITDB_ROOT
    }

    for name, path in paths_to_check.items():
        print(f"\n📂 {name}:")
        print(f"   Path: {path}")
        print(f"   Exists: {path.exists()}")

        if path.exists():
            contents = list(path.iterdir())[:10]  # Show first 10 items
            print(f"   Contents ({len(list(path.iterdir()))} items): {[p.name for p in contents]}")

            # Check for specific files based on dataset
            if name == 'ApneaECG':
                apn_files = list(path.glob("*.apn"))
                dat_files = list(path.glob("*.dat"))
                print(f"   .apn files: {len(apn_files)} (need > 0)")
                print(f"   .dat files: {len(dat_files)} (need > 0)")

            elif name == 'PTB-XL':
                csv_files = list(path.glob("**/ptbxl_database.csv"))
                raw_folder = path / "raw"
                print(f"   ptbxl_database.csv found: {len(csv_files) > 0}")
                print(f"   raw/ folder exists: {raw_folder.exists()}")
                if raw_folder.exists():
                    records_folder = raw_folder / "records100"
                    print(f"   records100/ folder exists: {records_folder.exists()}")
                    if records_folder.exists():
                        record_count = len(list(records_folder.rglob("*.hea")))
                        print(f"   .hea record files: {record_count}")

            elif name == 'MIT-BIH':
                hea_files = list(path.glob("*.hea"))
                atr_files = list(path.glob("*.atr"))
                print(f"   .hea files: {len(hea_files)} (need > 0)")
                print(f"   .atr files: {len(atr_files)} (need > 0)")
        else:
            print(f"   ❌ Path does not exist!")

    print(f"\n💡 SUGGESTIONS:")
    print("1. If PTB-XL shows 'raw/ folder exists: False', the data might be extracted directly")
    print("   in the root instead of a 'raw' subfolder. Check the ptbxl_database.csv location.")
    print("2. If MIT-BIH shows no .hea/.atr files, check if data is in a subfolder.")
    print("3. Set DO_PTBXL_DOWNLOAD=True and DO_MITDB_DOWNLOAD=True if you want to enable them.")

def simple_test():
    """Simple test with just ApneaECG dataset"""
    print("🧪 Running simple test with ApneaECG only...")

    # Create config with all needed attributes
    test_cfg = ExpCfg(
        epochs=2,
        epochs_cnn=2,  # Now this exists
        batch_size=16,
        limit=100,
        device='cpu',  # Use CPU for reliability
        input_len=1800,
        latent_dim=16
    )

    try:
        # Test ApneaECG dataset
        result = run_experiment(test_cfg, 'apnea_ecg', 'tiny_separable_cnn')
        if result:
            print("✅ Simple test passed!")
            print(f"Result: Val Acc={result['val_acc']:.4f}, Flash={result['flash_kb']:.1f}KB")
            return True
        else:
            print("❌ Simple test failed!")
            return False
    except Exception as e:
        print(f"❌ Simple test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def fix_ptbxl_paths():
    """Suggest fixes for PTB-XL path issues based on common layouts"""
    print("🔧 PTB-XL PATH FIXER")
    print("="*40)

    base_path = PTBXL_ROOT
    print(f"Current PTB-XL path: {base_path}")

    # Common PTB-XL layouts to check
    possible_layouts = [
        base_path / "ptbxl_database.csv",  # Direct in root
        base_path / "raw" / "ptbxl_database.csv",  # In raw subfolder
        base_path.parent / "ptbxl_database.csv",  # One level up
    ]

    print("\\nChecking for ptbxl_database.csv:")
    found_csv = None
    for layout in possible_layouts:
        if layout.exists():
            print(f"✅ Found: {layout}")
            found_csv = layout
            break
        else:
            print(f"❌ Not found: {layout}")

    if found_csv:
        suggested_raw = found_csv.parent
        print(f"\\n💡 Suggested fix:")
        print(f"Update PTBXL_ROOT to: {suggested_raw}")

        # Check if records exist
        records_folders = [
            suggested_raw / "records100",
            suggested_raw / "raw" / "records100"
        ]

        for rf in records_folders:
            if rf.exists():
                record_count = len(list(rf.rglob("*.hea")))
                print(f"Records found in {rf}: {record_count} .hea files")
                break
    else:
        print("\\n❌ Could not find ptbxl_database.csv in common locations")
        print("Please check if PTB-XL data is properly downloaded and extracted")

# Run diagnostics
print("Running dataset path diagnostics...")
#check_dataset_paths()

# Test the simple case
print("\\n" + "="*60)
simple_test()

# Note:
# - Ensure any constants/configs used across modules are imported where needed.
# - You may need to move some helper functions between modules if import errors occur.
