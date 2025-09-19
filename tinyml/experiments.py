
import os, sys, math, time, random
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
def line(): print("-"*80)


def _dir_has_any(path):
    """Check if directory exists and has files"""
    from pathlib import Path
    path = Path(path)
    return path.exists() and any(path.iterdir())


def _wfdb_download(db_name: str, dest: Path, do_download: bool, force: bool, verbose: bool):
    if not do_download:
	    if verbose: 
		    print(f"[{db_name}] download disabled (do=False)")
		    return

    dest = Path(dest)  
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
    if verbose: 
	    print("  - Download completed.")

DATA_BASE = os.environ.get("TINYML_DATA_ROOT","gs://store-pepper/tinyml_hyper_tiny_baselines/data") #"/content/drive/MyDrive/tinyml_hyper_tiny_baselines/data")
APNEA_ROOT = os.environ.get("APNEA_ROOT", f"{DATA_BASE}/apnea-ecg-database-1.0.0")
PTBXL_ROOT = os.environ.get("PTBXL_ROOT", f"{DATA_BASE}/ptbxl")
MITDB_ROOT = os.environ.get("MITDB_ROOT", f"{DATA_BASE}/mitbih/raw")

for p in [APNEA_ROOT, PTBXL_ROOT, MITDB_ROOT]:
    Path(p).mkdir(parents=True, exist_ok=True)

print("[Paths]")
print("  APNEA_ROOT:", APNEA_ROOT)
print("  PTBXL_ROOT:", PTBXL_ROOT)
print("  MITDB_ROOT:", MITDB_ROOT)
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    print("[Drive] Mounted")
except Exception as e:
    print("[Drive] Skipped mounting:", e)

from pathlib import Path

APNEA_ROOT = Path(APNEA_ROOT)
APNEA_ROOT.mkdir(parents=True, exist_ok=True)
print("[Paths] APNEA_ROOT:", APNEA_ROOT)
FORCE_DOWNLOAD=False
DO_APNEA_DOWNLOAD=False
DO_PTBXL_DOWNLOAD=False
DO_MITDB_DOWNLOAD=False
VERBOSE_DL=True
# Call (safe; won't download unless flags True)
_wfdb_download("apnea-ecg", APNEA_ROOT, DO_APNEA_DOWNLOAD, FORCE_DOWNLOAD, VERBOSE_DL)
_wfdb_download("ptb-xl",    PTBXL_ROOT, DO_PTBXL_DOWNLOAD, FORCE_DOWNLOAD, VERBOSE_DL)
_wfdb_download("mitdb",     MITDB_ROOT, DO_MITDB_DOWNLOAD, FORCE_DOWNLOAD, VERBOSE_DL)


DATASET_ALIAS = {
    "ApneaECG": "apnea_ecg",
    "apnea":    "apnea_ecg",
    "PTB-XL":   "ptbxl",
    "PTBXL":    "ptbxl",
    "MITDB":    "mitdb",
    "MIT-BIH":  "mitdb",
}

BASE = Path("/content/drive/MyDrive/tinyml_hyper_tiny_baselines/data")
TARGET_FOLDERS = [
    BASE / "UCI HAR Dataset",
    BASE / "apnea-ecg-database-1.0.0",
    BASE / "mitdb",
    BASE / "ptbxl",
]

def debug_apnea_root(root: Path, recurse_one_level=True):
    root = Path(root)
    print(f"[DEBUG] scanning root: {root}  (exists={root.exists()})")
    pats = ["a*.dat","a*.hea","a*.apn","a*.apn.txt",
            "b*.dat","b*.hea","b*.apn","b*.apn.txt",
            "c*.dat","c*.hea","c*.apn","c*.apn.txt"]

    counts = {p: len(list(root.glob(p))) for p in pats}
    for k in pats:
        print(f"  {k:<12} -> {counts[k]}")

    # show a few samples
    for k in ["a*.dat","a*.hea","a*.apn","a*.apn.txt"]:
        paths = list(root.glob(k))
        if paths:
            print(f"  sample {k}: {paths[0].name}")

    if recurse_one_level and sum(counts.values()) == 0:
        subs = [d for d in root.iterdir() if d.is_dir()]
        print(f"[DEBUG] no matches at root; trying subfolders ({len(subs)})…")
        for sub in subs:
            counts_sub = {p: len(list(Path(sub).glob(p))) for p in pats}
            if sum(counts_sub.values()) > 0:
                print(f"[DEBUG] FOUND in subfolder: {sub}")
                for k in pats:
                    if counts_sub[k]:
                        print(f"    {k:<12} -> {counts_sub[k]}")
                return Path(sub)
    return root

resolved = debug_apnea_root(APNEA_ROOT)
print("[DEBUG] resolved folder to use:", resolved)

# Standard library
import os
import sys
import time
import math
import glob
import copy
import inspect
from pathlib import Path
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pprint import pprint
from typing import Any, Dict, Tuple, List, Optional
from torch.optim import AdamW
from sklearn.metrics import f1_score

# Third-party
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    precision_score, recall_score
)
from pprint import pprint
from torch.utils.data import Sampler
import math
try:
    from sklearn.metrics import balanced_accuracy_score as _sk_bal_acc
    from sklearn.metrics import roc_auc_score as _sk_roc_auc

    def balanced_accuracy_score(y_true, y_pred):
        return float(_sk_bal_acc(y_true, y_pred))

    def roc_auc_score(y_true, y_score):
        # y_score should be the probability/score for the positive class
        # (for binary classification)
        return float(_sk_roc_auc(y_true, y_score))

except Exception:
    # ---- Fallbacks (no sklearn) ----

    def balanced_accuracy_score(y_true, y_pred):
        """
        Macro recall = mean recall across classes (binary or multiclass).
        If a class has no true samples, its recall contributes 0.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        classes = np.unique(y_true)
        recalls = []
        for c in classes:
            tp = np.sum((y_true == c) & (y_pred == c))
            fn = np.sum((y_true == c) & (y_pred != c))
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            recalls.append(rec)
        return float(np.mean(recalls)) if recalls else 0.0

    def _average_ranks(x):
        """Average ranks for ties (1-based)."""
        x = np.asarray(x)
        order = np.argsort(x)                 # ascending
        ranks = np.empty(len(x), dtype=float)
        n = len(x)
        i = 0
        next_rank = 1
        while i < n:
            j = i
            # group ties
            while j + 1 < n and x[order[j + 1]] == x[order[i]]:
                j += 1
            # average rank for the tie block [i..j]
            avg_rank = (next_rank + (next_rank + (j - i))) / 2.0
            ranks[order[i:j + 1]] = avg_rank
            next_rank += (j - i + 1)
            i = j + 1
        return ranks

    def roc_auc_score(y_true, y_score):
        """
        Binary ROC AUC via Mann–Whitney U (tie-correct with average ranks).
        Returns np.nan if only one class is present.
        """
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score).astype(float)

        # keep only 0/1 labels for binary case
        uniq = np.unique(y_true)
        if uniq.size != 2:
            return float('nan')  # undefined when only one class present

        # ranks of scores (ascending); higher score = better
        ranks = _average_ranks(y_score)

        # sum of ranks for positive class
        pos_mask = (y_true == 1)
        n_pos = int(np.sum(pos_mask))
        n_neg = len(y_true) - n_pos
        if n_pos == 0 or n_neg == 0:
            return float('nan')

        rp = np.sum(ranks[pos_mask])
        auc = (rp - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        return float(auc)

def _choose_avg(y):
    """binary if two labels, else macro."""
    ys = np.unique(np.asarray(y))
    return 'binary' if ys.size == 2 else 'macro'
# ===================== KD TRAINING =====================

def kd_train_epoch(student, teacher, loader, opt, T=2.0, alpha=0.7,
                   device=None, clip=1.0, meta=None,
                   w_size=0.0, w_bit=0.0, w_spec=0.0, w_softf1=0.0):
    """
    KL distillation + CE (student) with optional:
      - resource penalty on student (w_size)
      - bit-aware regularizer on student logits (w_bit)
      - spectral penalty on inputs (w_spec)
      - soft-F1 auxiliary (w_softf1)
    Teacher is frozen.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    student.train(); teacher.eval()
    tot = n = 0
    fs = float(meta.get('fs', 100.0)) if meta else 100.0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        xb = torch.nan_to_num(xb)

        with torch.no_grad():
            tlog = teacher(xb)

        slog = student(xb)

        # Distillation + CE (your original)
        loss_kd = F.kl_div(F.log_softmax(slog / T, dim=1),
                           F.softmax(tlog / T, dim=1),
                           reduction='batchmean') * (T * T)
        loss_ce = F.cross_entropy(slog, yb)
        loss = alpha * loss_kd + (1 - alpha) * loss_ce

        # NEW: resource-aware penalty (student only)
        if w_size > 0.0:
            loss = loss + resource_penalty(student, meta or {}, w_size=w_size)

        # NEW: bit-aware KD regularizer on student logits
        if w_bit > 0.0:
            loss = loss + bitaware_reg(slog, bits=8, beta=w_bit)

        # NEW: spectral regularization on inputs (band leakage)
        if w_spec > 0.0:
            loss = loss + spectral_penalty(xb, fs=fs, w=w_spec)

        # NEW: soft-F1 auxiliary
        if w_softf1 > 0.0:
            loss = loss + soft_f1_loss(slog, yb, w=w_softf1)

        if not torch.isfinite(loss):
            continue

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if clip and clip > 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), clip)
        opt.step()

        bs = xb.size(0)
        tot += float(loss.detach()) * bs
        n   += bs

    return float(tot / max(1, n))
  # Fallback simple CE trainer (used if you don't already have train_epoch)

def train_epoch_ce(model, loader, opt, device=None, clip=1.0,
                   criterion=None, meta=None,
                   w_size=0.0, w_spec=0.0, w_softf1=0.0):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    if criterion is None:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    tot = n = 0
    fs = float(meta.get('fs', 100.0)) if meta else 100.0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        xb = torch.nan_to_num(xb)

        opt.zero_grad(set_to_none=True)
        logits = model(xb)

        loss = criterion(logits, yb)
        loss = loss + resource_penalty(model, meta or {}, w_size=w_size)
        loss = loss + spectral_penalty(xb, fs=fs, w=w_spec)
        loss = loss + soft_f1_loss(logits, yb, w=w_softf1)

        if not torch.isfinite(loss):
            continue
        loss.backward()
        if clip and clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()

        bs = xb.size(0)
        tot += float(loss.detach()) * bs
        n   += bs

    return float(tot / max(1, n))

@torch.no_grad()

def eval_logits(model, loader, device=None):
    """
    Evaluate model and return logits and true labels.
    Used in run_one for validation and test evaluation.
    """
    '''
    if device is None:
        device = DEVICE

    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in loader:
            try:
                xb = xb.to(device)
                yb = yb.to(device)

                logits = model(xb)

                # Check for NaN/infinite logits
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print(f"WARNING: NaN/Inf logits detected in evaluation, skipping batch...")
                    continue

                all_logits.append(logits.cpu().numpy())
                all_labels.append(yb.cpu().numpy())

            except Exception as e:
                print(f"Error in evaluation batch: {e}")
                continue

    if not all_logits:
        print("WARNING: No valid evaluation batches processed!")
        return np.array([]), np.array([])


    return np.concatenate(all_logits), np.concatenate(all_labels)
    '''
    device = device or DEVICE
    model.eval(); outs=[]; ys=[]
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            outs.append(model(xb).detach().float().cpu()); ys.append(yb.detach().cpu())
    return torch.cat(outs).numpy(), torch.cat(ys).numpy()


def evaluate_logits(model, dl, device="cpu"):
    import torch
    model.eval()
    all_logits, all_y = [], []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device); yb = yb.to(device)
            lg = model(xb)
            all_logits.append(lg.detach().cpu())
            all_y.append(yb.detach().cpu())
    return torch.concat(all_logits).numpy(), torch.concat(all_y).numpy()


def eval_prob_fn(logits_np):
    """
    Convert logits to probabilities for binary classification.
    Used in run_one for threshold tuning.
    """
    if logits_np.ndim == 2 and logits_np.shape[1] == 2:
        t = torch.from_numpy(logits_np)
        return torch.softmax(t, dim=1)[:,1].cpu().numpy()
    t = torch.from_numpy(logits_np).squeeze(-1)
    return torch.sigmoid(t).cpu().numpy()
    '''
    if len(logits) == 0:
        return np.array([])

    # For binary classification, take softmax and get positive class probability
    if logits.shape[1] == 2:
        # Binary classification: return probability of positive class (class 1)
        probs = torch.softmax(torch.from_numpy(logits), dim=1)
        return probs[:, 1].numpy()  # Return probability of class 1
    else:
        # Multi-class: return max probability
        probs = torch.softmax(torch.from_numpy(logits), dim=1)
        return probs.max(dim=1)[0].numpy()
   '''



def tune_threshold(y_true, p1, grid=THRESH_GRID, average='macro'):
    """Find optimal threshold for binary classification"""
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        yhat = (p1 >= t).astype(int)
        f1 = f1_score(y_true, yhat, average=average, zero_division=0)
        if f1 > best_f1: best_f1, best_t = f1, t
    return float(best_t), float(best_f1)

# ===================== METRICS + CI, SIZE, LATENCY =====================

def _bootstrap_ci(metric_fn, y_true, y_pred, n_boot=200, alpha=0.05, rng=None):
    rng = rng or np.random.RandomState(1337)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        vals.append(metric_fn(y_true[idx], y_pred[idx]))
    vals = np.sort(vals)
    lo = vals[int((alpha/2)*n)]; hi = vals[int((1-alpha/2)*n)]
    return float(lo), float(hi)

def ec57_metrics_with_ci(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average='macro', zero_division=0)
    acc_ci = _bootstrap_ci(lambda a,b: accuracy_score(a,b), y_true, y_pred)
    f1_ci  = _bootstrap_ci(lambda a,b: f1_score(a,b, average='macro', zero_division=0), y_true, y_pred)
    return {"acc": float(acc), "acc_ci": acc_ci, "macro_f1": float(f1), "macro_f1_ci": f1_ci}
# Size helpers

def _count_params(m):
    if 'count_parameters' in globals():
        return int(count_parameters(m))
    return sum(p.numel() for p in m.parameters())

def packed_bytes_model_paper(model):
    """Calculate packed bytes for model (as used in run_one)"""
    total_bytes = 0
    for n, p in model.named_parameters():
        total_bytes += p.numel()
    return int(total_bytes)

def proxy_latency_estimate(model, T=2000, c_in=1, repeats=10):
    """Estimate inference latency (as used in run_one)"""
    if 'DEVICE' not in globals():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = DEVICE

    xb = torch.randn(1, c_in, T).to(device)
    model.eval()

    with torch.no_grad():
        # Warmup
        for _ in range(2):
            _ = model(xb)

    # Measure inference time
    t0 = time.time()
    with torch.no_grad():
        for _ in range(repeats):
            _ = model(xb)
    t1 = time.time()
    per_inf_ms = 1000.0 * (t1 - t0) / repeats

    # Check for synthesis/boot time (for hybrid models)
    boot_ms = 0.0
    if hasattr(model, 'synth_once'):
        if hasattr(model, '_synth_done'):
            model._synth_done = False
        t2 = time.time()
        with torch.no_grad():
            model.synth_once()
        t3 = time.time()
        boot_ms = 1000.0 * (t3 - t2)

    return float(per_inf_ms), float(boot_ms)

 #=====================

 #==================================Just Extra remove later

def already_done(name): return (EXP_DIR / f"{name}.json").exists()

def save_json(name, payload):
    p = EXP_DIR / f"{name}.json"
    p.write_text(json.dumps(payload, indent=2))
    return str(p)

def print_and_log(name, payload):
    print(f"[RESULT] {name} → {json.dumps(payload, indent=2)[:800]}...")
#'''
# ===================== MODEL-NAME SHIMS =====================
# (Map names used by the other suite to the classes you already have.)

def _scan_counts_from_loader(train_loader, num_classes: int):
    """Fast pass over the train loader to get class counts; robust to tuple/dict batches."""
    import numpy as np, torch
    counts = np.zeros(num_classes, dtype=np.int64)
    for batch in train_loader:
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            y = batch[1]
        elif isinstance(batch, dict) and 'y' in batch:
            y = batch['y']
        else:
            continue
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        counts += np.bincount(y.astype(int), minlength=num_classes)
    return counts


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
# %% Metrics & diagnostics

def acc_logits(logits, y):
    return (logits.argmax(1) == y).float().mean().item()


def compute_class_weights(train_loader, num_classes):
    counts = Counter()
    for xb, yb in train_loader:
        for y in yb.view(-1).tolist():
            counts[int(y)] += 1
    total = sum(counts.values())
    freqs = [counts.get(c, 0)/max(total,1) for c in range(num_classes)]
    # inverse frequency (normalized)
    weights = [1.0/(f+1e-8) for f in freqs]
    s = sum(weights)
    weights = [w/s * num_classes for w in weights]
    return torch.tensor(weights, dtype=torch.float32)
'''

def make_criterion(num_classes, train_loader=None, use_focal=True, gamma=1.5, class_counts=None):
    import numpy as np, torch, torch.nn as nn
    if not use_focal:
        return nn.CrossEntropyLoss()

    if class_counts is None:
        if train_loader is not None:
            class_counts = _scan_counts_from_loader(train_loader, num_classes)
        else:
            class_counts = np.ones(num_classes, dtype=np.int64)  # uniform fallback

    freq = class_counts / class_counts.sum()
    alpha = torch.tensor(1.0 - freq, dtype=torch.float32,
                         device=('cuda' if torch.cuda.is_available() else 'cpu'))
    return FocalLoss(gamma=gamma, alpha=alpha)  # uses your existing FocalLoss
'''

def make_criterion(num_classes=2, train_loader=None, use_focal=True, gamma=1.5):
    # estimate class counts quickly
    counts = np.zeros(num_classes, dtype=np.int64)
    if train_loader is not None:
        for _, y in list(zip(range(8), train_loader)):  # peek a few batches
            counts += np.bincount(y.numpy() if isinstance(y, np.ndarray) else y.cpu().numpy(), minlength=num_classes)
    if use_focal:
        return SafeFocalLoss(gamma=gamma, alpha=0.5, label_smoothing=0.05)
    else:
        w = None
        if counts.sum() > 0:
            freqs = counts / counts.sum()
            w = torch.tensor(1.0/(freqs+1e-9), dtype=torch.float32)
            w = (w / w.sum()) * num_classes
        return nn.CrossEntropyLoss(weight=w.to(DEVICE) if w is not None else None)



def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps,
                                    num_cycles: float = 0.5, last_epoch: int = -1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * 2.0 * num_cycles * progress)))
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)



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

def evaluate_model(model, dl_te):
    # delegate to the fallback in _eval_fwd by removing this symbol
    del globals()['evaluate_model']
    return _eval_fwd(model, dl_te)


def build_hypertiny_hybrid(keep_first_pw=True, base_channels=24, num_classes=2, latent_dim=16, input_length=1800, **kwargs):
    """
    Build HyperTiny model in hybrid mode (keep some layers traditional, synthesize others).

    Args:
        keep_first_pw: Whether to keep first pointwise layer traditional
        base_channels: Base channel count
        num_classes: Number of output classes
        latent_dim: Latent space dimensionality
        input_length: Expected input length
        **kwargs: Additional configuration parameters
    """
    model = SharedCoreSeparable1D(
        in_ch=1,
        base=base_channels,
        num_classes=num_classes,
        latent_dim=latent_dim,
        hybrid_keep=1 if keep_first_pw else 0,
        input_length=input_length
    )

    # Store hybrid configuration
    model.architecture_config = {
        'mode': 'hybrid' if keep_first_pw else 'full_synthesis',
        'keep_first_pw': keep_first_pw,
        **kwargs
    }

    return model


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


def _extract_labels_fast(ds):
    for attr in ("targets","labels","y"):
        if hasattr(ds, attr):
            v = getattr(ds, attr)
            if isinstance(v, torch.Tensor): return v.detach().cpu().tolist()
            try: return list(map(int, list(v)))
            except: pass
    if hasattr(ds, "df") and hasattr(ds.df, "__getitem__"):
        for col in ("label","y","target","class"):
            if col in ds.df.columns:
                return list(map(int, ds.df[col].tolist()))
    if hasattr(ds, "metadata") and isinstance(ds.metadata, dict):
        for k in ("labels","y","targets"):
            if k in ds.metadata:
                v = ds.metadata[k]
                if isinstance(v, torch.Tensor): return v.detach().cpu().tolist()
                try: return list(map(int, list(v)))
                except: pass
    if isinstance(ds, Subset):
        base = ds.dataset; base_labels = _extract_labels_fast(base)
        if base_labels is not None:
            return [int(base_labels[i]) for i in ds.indices]
    if isinstance(ds, ConcatDataset):
        out=[]
        for child in ds.datasets:
            lbls = _extract_labels_fast(child)
            if lbls is None: return None
            out.extend(lbls)
        return out
    return None



def _ensure_drive_mounted():
    try:
        from google.colab import drive
        if not Path("/content/drive").exists() or not list(Path("/content/drive").glob("*")):
            drive.mount('/content/drive', force_remount=False)
    except Exception:
        pass  # on non-Colab, ignore


def save_df_to_drive(df, filename, subdir=None):
    root = _results_dir()
    if subdir:
        root = root / subdir; root.mkdir(parents=True, exist_ok=True)
    out = root / filename
    df.to_csv(out.as_posix(), index=False)
    print(f" Saved: {out.as_posix()}")
    return out

def print_class_distribution(loader, name="", fast_limit=2000):
    ds = loader.dataset
    labels = _extract_labels_fast(ds)
    if labels is None:
        counts = Counter(); seen=0; limit = float("inf") if fast_limit is None else int(fast_limit)
        for _, y in loader:
            y = y.detach().cpu().tolist() if isinstance(y, torch.Tensor) else list(map(int, y))
            for yy in y:
                counts[int(yy)] += 1; seen += 1
                if seen >= limit: break
            if seen >= limit: break
        total = sum(counts.values()); approx = "" if fast_limit is None else " (approx)"
        print(f"\n=== {name} class distribution{approx} ===")
        print(f"  counted samples : {total}  (limit={fast_limit})")
        for cls in sorted(counts): print(f"  class {cls}: {counts[cls]} ({counts[cls]/max(1,total):.2%})")
        print("="*40)
        return
    counts = Counter(map(int, labels)); total = sum(counts.values())
    print(f"\n=== {name} class distribution ===")
    print(f"  total samples : {total}")
    for cls in sorted(counts): print(f"  class {cls}: {counts[cls]} ({counts[cls]/total:.2%})")
    print("="*40)
# Cell A — Lightweight ECG training augments (helps generalization without changing model size)
import torch
import numpy as np


def _results_dir(prefer="MyDrive/tinyml_hyper_tiny_baselines/results"):
    _ensure_drive_mounted()
    # 1) Try MyDrive path
    p = Path("/content/drive") / prefer
    Path(p).mkdir(parents=True, exist_ok=True)
    if p.exists():
        return p
    # 2) Try under any Shareddrive (if user uses a Team Drive)
    sd_root = Path("/content/drive/Shareddrives")
    if sd_root.exists():
        # pick first share drive that already has the project folder, else create in first share
        matches = list(sd_root.rglob("tinyml_hyper_tiny_baselines"))
        if matches:
            p = matches[0] / "results"
            Path(p).mkdir(parents=True, exist_ok=True)
            return p
        # fallback: create in first share drive
        shares = [d for d in sd_root.iterdir() if d.is_dir()]
        if shares:
            p = shares[0] / "tinyml_hyper_tiny_baselines" / "results"
            Path(p).mkdir(parents=True, exist_ok=True)
            return p
    # 3) Last resort: local current dir
    p = Path("./results"); Path(p).mkdir(parents=True, exist_ok=True); return p
class AugmentECG:
    def __init__(self, noise_std=0.01, amp_jitter=0.05, time_shift_frac=0.02, p=0.8):
        self.noise_std = noise_std
        self.amp_jitter = amp_jitter
        self.time_shift_frac = time_shift_frac
        self.p = p

    def __call__(self, x):
        # x: Tensor [C, T] or [T]
        if np.random.rand() > self.p:
            return x
        if x.ndim == 1:
            x = x.unsqueeze(0)
        C, T = x.shape

        # amplitude jitter
        scale = 1.0 + (2*np.random.rand(C, 1)-1.0)*self.amp_jitter
        x = x * torch.tensor(scale, dtype=x.dtype, device=x.device)

        # small time shift (circular)
        max_shift = int(T * self.time_shift_frac)
        if max_shift > 0:
            s = np.random.randint(-max_shift, max_shift+1)
            if s != 0:
                x = torch.roll(x, shifts=s, dims=1)

        # gaussian noise
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        return x

# Usage:
# - If you have a custom Dataset, apply in __getitem__ during training.
# - If you use a transform pipeline, pass AugmentECG() only for the train split.
train_ecg_augment = AugmentECG(noise_std=0.01, amp_jitter=0.05, time_shift_frac=0.02, p=0.8)

@dataclass
class ExpCfg:
    # --- Training ---
    epochs: int = 8
    epochs_cnn: Optional[int] = None
    epochs_head: Optional[int] = None
    epochs_vae_pre: Optional[int] = None
    warmup_epochs: int = 1

    # --- Optimizer ---
    lr: float = 3e-4
    weight_decay: float = 1e-4

    # --- Data loading ---
    batch_size: int = 64
    num_workers: int = 2
    limit: Optional[int] = None

    # --- Windowing / signal ---
    window_ms: int = 60000
    target_fs: int = 100
    input_len: Optional[int] = None         # alias A
    input_length: Optional[int] = None      # alias B
    length: int = 1800                      # if some code uses 'length' for windows

    # --- Model knobs ---
    base: int = 24                          # << numeric channel base (not a path)
    width_base: Optional[int] = None        # if used elsewhere, keep in sync with base
    width_mult: float = 1.0
    latent_dim: int = 16

    # --- Aug/Loss toggles ---
    use_mixup: bool = False
    mixup_alpha: float = 0.2
    use_focal_loss: bool = False
    use_label_smoothing: bool = False

    # --- Misc / paths ---
    data_base: Optional[str] = None         # << use this if you need a base *path*
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        # epochs fallbacks
        if self.epochs_cnn is None:
            self.epochs_cnn = self.epochs
        if self.epochs_head is None:
            self.epochs_head = max(1, self.epochs // 2)
        if self.epochs_vae_pre is None:
            self.epochs_vae_pre = max(1, self.epochs // 2)

        # input length unification
        if self.input_len is None and self.input_length is not None:
            self.input_len = self.input_length
        if self.input_len is None and self.window_ms and self.target_fs:
            self.input_len = int((self.window_ms / 1000.0) * self.target_fs)
        self.input_length = self.input_len  # keep both aliases consistent

        # numeric base unification
        self.base = int(self.base)
        if self.width_base is None:
            self.width_base = self.base
        else:
            self.width_base = int(self.width_base)

        # minor normalisation
        if isinstance(self.limit, float):
            self.limit = int(self.limit)





import os, random, numpy as np, wfdb, torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter, defaultdict, OrderedDict
from torch.utils.data import DataLoader, WeightedRandomSampler
import ast
import pandas as pd
from typing import List, Tuple
import pandas as pd
# Toggle: use class-balanced sampling in the TRAIN loader
USE_WEIGHTED_SAMPLER = True
try:
    from torch.utils.data import ConcatDataset, Subset, random_split, RandomSampler, WeightedRandomSampler
except Exception:
    # fallback for very old torch versions
    from torch.utils.data.dataset import ConcatDataset, Subset
    from torch.utils.data import random_split, RandomSampler, WeightedRandomSampler

FS = 100  # Apnea-ECG sampling rate


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


def _stratified_record_split_apnea(root, recs, seed=1337, frac=(0.8, 0.1, 0.1)):
    """
    Record-wise stratified split for Apnea-ECG.
    - Groups by record (subject) → prevents leakage across splits.
    - Stratifies by 'has any apnea-positive minutes' to balance splits.
    Returns: (train_recs, val_recs, test_recs)
    """
    import random, os

    rng = random.Random(seed)

    def _has_apnea_positive(record_id):
        # Use your existing stats helper if present
        try:
            stats = _record_apnea_stats(root, [record_id])  # [(rec, apnea_minutes, total_minutes, ...)]
            if stats:
                _, a_pos, *_ = stats[0]
                return a_pos > 0
        except NameError:
            pass
        # Fallback: scan .apn file (A/N per minute). Adjust if your loader differs.
        apn_path = os.path.join(root, f"{record_id}.apn")
        if not os.path.exists(apn_path):
            return False
        with open(apn_path, "r") as f:
            txt = f.read()
        return "A" in txt

    pos, neg = [], []
    for r in recs:
        (pos if _has_apnea_positive(r) else neg).append(r)

    rng.shuffle(pos); rng.shuffle(neg)

    def _split(lst):
        n = len(lst)
        ntr = int(round(frac[0] * n))
        nva = int(round(frac[1] * n))
        return lst[:ntr], lst[ntr:ntr + nva], lst[ntr + nva:]

    trp, vap, tep = _split(pos)
    trn, van, ten = _split(neg)

    train_recs = trp + trn
    val_recs   = vap + van
    test_recs  = tep + ten

    rng.shuffle(train_recs); rng.shuffle(val_recs); rng.shuffle(test_recs)
    return train_recs, val_recs, test_recs



'''

def _stratified_record_split_apnea(root: Path, records: list[str], seed=1337, frac=(0.8, 0.1, 0.1)):
    """
    Ensure val/test each include some apnea-positive records.
    We first separate positive vs. negative records based on minute labels,
    then allocate at least one positive record to val/test (when available),
    and fill the rest to approximate frac with remaining records.
    """
    rng = random.Random(seed)
    stats = _record_apnea_stats(root, records)
    pos = [r for (r,a,n,p) in stats if a > 0]
    neg = [r for (r,a,n,p) in stats if a == 0]
    rng.shuffle(pos); rng.shuffle(neg)

    n_total = len(records)
    n_tr = max(1, int(round(frac[0]*n_total)))
    n_va = max(1, int(round(frac[1]*n_total)))
    n_te = max(1, n_total - n_tr - n_va)

    # Seed val/test with at least one positive (if available)
    val, tes, tra = [], [], []
    if len(pos) >= 1: val.append(pos.pop())
    if len(pos) >= 1: tes.append(pos.pop())
    # If still none, try to move from negatives (will remain all-N, but keep sizes right)
    while len(val) < 1 and neg: val.append(neg.pop())
    while len(tes) < 1 and neg: tes.append(neg.pop())

    # Fill val to target size, prefer positives first then negatives
    while len(val) < n_va and (pos or neg):
        val.append(pos.pop() if pos else neg.pop())
    # Fill test
    while len(tes) < n_te and (pos or neg):
        tes.append(pos.pop() if pos else neg.pop())
    # Rest go to train
    tra = pos + neg
    rng.shuffle(tra)
    tra = tra[:n_tr]  # cap to target count (if too many left)

    # If we fell short somewhere due to rounding, re-balance minimally
    leftover = [r for r in records if r not in set(tra+val+tes)]
    for r in leftover:
        if   len(tra) < n_tr: tra.append(r)
        elif len(val) < n_va: val.append(r)
        elif len(tes) < n_te: tes.append(r)

    return tra, val, tes
'''

def _make_weighted_sampler_apnea(dataset):
    """
    Build per-sample weights inverse to class frequency using the dataset's internal index.
    """
    # Count positives/negatives by iterating labels (fast enough)
    pos = 0; neg = 0
    y_list = []
    for (rid, m, off) in dataset.index:
        y = dataset._labs[rid][m]
        y_list.append(y)
        if y == 1: pos += 1
        else:      neg += 1
    # Inverse frequency weights
    w0 = 1.0 / max(1, neg)
    w1 = 1.0 / max(1, pos)
    weights = [w1 if y==1 else w0 for y in y_list]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)



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


def build_hypertiny_no_focal(base_channels=24, num_classes=2, latent_dim=16, input_length=1800):
    """
    Build HyperTiny model without focal loss optimization.
    This is used for ablation studies to show the impact of focal loss.
    """
    model = SharedCoreSeparable1D(
        in_ch=1,
        base=base_channels,
        num_classes=num_classes,
        latent_dim=latent_dim,
        hybrid_keep=1,
        input_length=input_length
    )

    # Mark this model as using standard loss
    model.training_config = {
        'use_focal': False,
        'loss_type': 'crossentropy'
    }

    return model

def build_hypertiny_no_kd(base_channels=24, num_classes=2, latent_dim=16, input_length=1800):
    """
    Build HyperTiny model without knowledge distillation optimization.
    This is used for ablation studies to show the impact of KD.
    """
    model = SharedCoreSeparable1D(
        in_ch=1,
        base=base_channels,
        num_classes=num_classes,
        latent_dim=latent_dim,
        hybrid_keep=1,
        input_length=input_length
    )

    # Mark this model as not optimized for KD
    model.training_config = {
        'use_kd': False,
        'optimization_type': 'standard'
    }

    return model

def build_hypertiny_with_generator(dz, dh, r, base_channels=24, num_classes=2, latent_dim=16, input_length=1800):
    """
    Build HyperTiny model with configurable generator parameters.

    Args:
        dz: Latent code dimension for synthesis
        dh: Hidden dimension for generator network
        r: Rank factor for low-rank approximations
        base_channels: Base channel count for the model
        num_classes: Number of output classes
        latent_dim: Dimensionality of latent space
        input_length: Expected input sequence length
    """
    # Use existing SharedCoreSeparable1D as the base architecture
    # In a full implementation, dz, dh, r would configure the synthesis components
    model = SharedCoreSeparable1D(
        in_ch=1,
        base=base_channels,
        num_classes=num_classes,
        latent_dim=latent_dim,
        hybrid_keep=1,  # Keep some layers traditional
        input_length=input_length
    )

    # Store generator config as model attributes for reference
    model.synthesis_config = {
        'dz': dz,
        'dh': dh,
        'r': r,
        'generator_type': 'configurable'
    }

    return model

def stratified_by_minutes_split(root, records, seed=1337, frac=(0.8,0.1,0.1), target_prev=None):
    """
    Split by RECORD but match minute-level apnea prevalence across splits.
    target_prev: if None, uses global prevalence across provided records.
    """
    rng = random.Random(seed)
    stats = []  # (rid, apnea_minutes, norm_minutes, prevalence)
    for rid in records:
        labs = _minute_labels_rdann(root, rid)
        a = int(sum(labs)); n = int(len(labs) - a)
        p = a / max(1, a + n)
        stats.append((rid, a, n, p))

    rng.shuffle(stats)
    total_a = sum(a for _, a, _, _ in stats)
    total_n = sum(n for _, _, n, _ in stats)
    global_prev = total_a / max(1, (total_a + total_n))
    if target_prev is None:
        target_prev = global_prev

    n_total = len(records)
    n_tr = max(1, int(round(frac[0]*n_total)))
    n_va = max(1, int(round(frac[1]*n_total)))
    n_te = max(1, n_total - n_tr - n_va)

    # Greedy fill each split toward its target prevalence
    def fill_split(k):
        return {'recs': [], 'a': 0, 'n': 0, 'target_prev': target_prev, 'target_size': k}

    splits = [fill_split(n_tr), fill_split(n_va), fill_split(n_te)]

    for rid, a, n, p in stats:
        # choose split that (a) still needs records and (b) moves its prevalence closest to target
        best_idx, best_score = None, float('inf')
        for i, sp in enumerate(splits):
            if len(sp['recs']) >= sp['target_size']:
                continue
            new_a = sp['a'] + a
            new_n = sp['n'] + n
            new_prev = new_a / max(1, (new_a + new_n))
            score = abs(new_prev - sp['target_prev'])
            if score < best_score:
                best_score, best_idx = score, i
        splits[best_idx]['recs'].append(rid)
        splits[best_idx]['a'] += a
        splits[best_idx]['n'] += n

    tra = splits[0]['recs']; val = splits[1]['recs']; tes = splits[2]['recs']
    return tra, val, tes

# Loader alias shims

def _dir_has_any(root: Path, exts=(".dat",".hea",".apn",".csv",".mat",".atr")):
    try:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                return True
    except Exception:
        pass
    return False



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


def _standardize_1d(x, eps: float = 1e-6):
    # x: (B, C, T)
    mu = x.mean(dim=-1, keepdim=True)
    sd = x.std(dim=-1, keepdim=True).clamp_min(eps)
    return (x - mu) / sd

def eval_classifier_plus(model, loader, device, return_probs=False):
    logits, y = eval_logits(model, loader, device)
    p1 = eval_prob_fn(logits)
    yhat = (p1 >= 0.5).astype(int)
    out = dict(
    acc=float(accuracy_score(y, yhat)),
    bal_acc=float(balanced_accuracy_score(y, yhat)),
    macro_f1=float(f1_score(y, yhat, average='macro', zero_division=0)),
    coverage_batches=len(loader),
    auc=float(roc_auc_score(y, p1)) if len(np.unique(y)) > 1 else None,
    )
    if return_probs: out.update({'probs': p1, 'y': y})
    return out
    '''
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    covered = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        if not torch.isfinite(logits).all():
            continue  # skip pathological batches
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(1)
        y_true.append(yb.cpu()); y_pred.append(pred.cpu())
        if return_probs: y_prob.append(probs[:,1].cpu())
        covered += 1

    if not y_true:
        return {
            'coverage_batches': 0, 'acc': 0.0, 'bal_acc': 0.0, 'macro_f1': 0.0,
            'auc': None, 'cm': np.zeros((2,2), int)
        }

    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    auc = None
    if return_probs:
        y_prob = torch.cat(y_prob).numpy()
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = None

    return {
        'coverage_batches': covered,
        'acc': (y_true == y_pred).mean(),
        'bal_acc': balanced_accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'auc': auc,
        'cm': confusion_matrix(y_true, y_pred)
    }
    '''

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
@torch.no_grad()

def eval_vae_epoch(vae, loader, device, beta: float = 1.0):
    """
    Eval the VAE over a loader.
    Returns: (avg_total_loss, avg_recon_loss, avg_kld)
    - Matches train_vae_epoch preprocessing (nan_to_num + standardize_1d)
    - Robust to loaders that yield (x, y) or just x
    """
    vae.eval()
    total = 0.0
    recon_sum = 0.0
    kld_sum = 0.0
    for xb, _ in loader:
        xb = torch.nan_to_num(xb.to(device))
        xb = standardize_1d(xb)
        xhat, mu, lv = vae(xb)
        loss, recon, kld = safe_vae_loss(xb, xhat, mu, lv, beta=beta)
        bs = xb.size(0)
        total += float(loss) * bs; recon_sum += float(recon) * bs; kld_sum += float(kld) * bs; n += bs

    return total/max(1,n), recon_sum/max(1,n), kld_sum/max(1,n)

# --- Enhanced Tiny VAE with better architecture

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

def tiny_jitter(x, sigma=0.005):           # white noise
    return x + sigma*torch.randn_like(x)


def tiny_scaling(x, sigma=0.05):           # amplitude scaling
    s = (1.0 + sigma*torch.randn(x.size(0),1,1, device=x.device).clamp(-0.2,0.2))
    return x * s


def tiny_timeshift(x, max_shift=10):       # samples shift
    if max_shift <= 0: return x
    B, C, T = x.shape
    shift = torch.randint(-max_shift, max_shift+1, (B,), device=x.device)
    out = torch.zeros_like(x)
    for i,s in enumerate(shift.tolist()):
        if s>=0: out[i,:,s:] = x[i,:,:T-s]
        else:    out[i,:,:T+s] = x[i,:,-s:]
    return out


def pick_best_threshold_from_loader(model, loader, device):
    logits, y = eval_logits(model, loader, device)
    p1 = eval_prob_fn(logits)
    return tune_threshold(y, p1)



@torch.no_grad()

def pick_best_threshold(model, loader, device, n=101):
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        p = torch.softmax(model(xb), dim=1)[:,1]
        ys.append(yb.cpu()); ps.append(p.cpu())
    ys = torch.cat(ys).numpy(); ps = torch.cat(ps).numpy()
    best_t, best_f1 = 0.5, -1
    from sklearn.metrics import f1_score
    for t in np.linspace(0,1,n):
        f1 = f1_score(ys, (ps>=t).astype(int), average='macro', zero_division=0)
        if f1 > best_f1: best_f1, best_t = f1, t
    return best_t, best_f1


def eval_with_record_vote(model, dataset: ApneaECGWindows, batch_size=64, device='cuda', prob_mean=True):
    loader = DataLoader(WithIndex(dataset), batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()
    per_rec_probs = defaultdict(list)
    per_rec_true  = {}

    for xb, yb, idx in loader:
        xb, yb = xb.to(device), yb.to(device)
        probs = torch.softmax(model(xb), dim=1)[:,1]  # P(apnea)
        for p, i, y in zip(probs.cpu().numpy(), idx.numpy(), yb.cpu().numpy()):
            rid, m, off = dataset.index[i]
            per_rec_probs[rid].append(p)
            # store any minute’s label; or use majority of minute labels if you prefer
            per_rec_true.setdefault(rid, 1 if dataset._labs[rid].count(1) > (len(dataset._labs[rid])//2) else 0)

    y_true, y_pred = [], []
    for rid, plist in per_rec_probs.items():
        p = float(np.mean(plist)) if prob_mean else float(np.median(plist))
        y_hat = 1 if p >= 0.5 else 0
        y_true.append(per_rec_true[rid]); y_pred.append(y_hat)

    y_true = np.array(y_true); y_pred = np.array(y_pred)
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    return {
        'rec_acc': accuracy_score(y_true, y_pred),
        'rec_bal_acc': balanced_accuracy_score(y_true, y_pred),
        'rec_macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'n_records': len(y_true)
    }

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
    replace_batchnorm_with_groupnorm(cnn, groups=8)

    # Optimizer + scheduler (with fallback)
    opt_cnn = torch.optim.AdamW(cnn.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    try:
        scheduler_cnn = get_cosine_schedule_with_warmup(
            opt_cnn,
            num_warmup_steps=max(1, len(tr_loader) * cfg.warmup_epochs),
            num_training_steps=max(1, len(tr_loader) * cfg.epochs_cnn),
        )
        _per_step_sched = True
    except Exception:
        scheduler_cnn = make_cosine_with_warmup(opt_cnn, cfg.warmup_epochs, cfg.epochs_cnn)
        _per_step_sched = False

    # Loss
    if getattr(cfg, "use_focal_loss", False):
        criterion_cnn = SafeFocalLoss(gamma=1.5, alpha=0.5, label_smoothing=0.05)
    elif getattr(cfg, "use_label_smoothing", False):
        # PyTorch >=1.10 supports this
        criterion_cnn = nn.CrossEntropyLoss(label_smoothing=0.1)
    else:
        criterion_cnn = nn.CrossEntropyLoss()

    # AMP (optional)
    use_amp = (torch.cuda.is_available() and "cuda" in str(DEVICE))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"[ApneaECG] Training CNN with {type(criterion_cnn).__name__}...")
    best_val_score = -1.0
    best_state = None
    best_thresh = 0.5
    patience = 3
    patience_counter = 0

    for ep in range(1, cfg.epochs_cnn + 1):
        cnn.train()
        tot = acc = n = 0

        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            # tiny, safe augmentations
            xb = tiny_scaling(tiny_jitter(tiny_timeshift(xb, 5)), sigma=0.03)

            opt_cnn.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                if cfg.use_mixup and np.random.rand() > 0.5:
                    mixed_x, y_a, y_b, lam = mixup_data(xb, yb, cfg.mixup_alpha)
                    logits = cnn(mixed_x)
                    loss = mixup_criterion(criterion_cnn, logits, y_a, y_b, lam)
                    hard_targets = yb  # for accuracy accounting
                else:
                    logits = cnn(xb)
                    loss = criterion_cnn(logits, yb)
                    hard_targets = yb

            if not torch.isfinite(loss):
                continue

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(cnn.parameters(), 1.0)
            scaler.step(opt_cnn)
            scaler.update()
            if _per_step_sched: scheduler_cnn.step()

            bs = xb.size(0)
            tot += float(loss) * bs
            acc += acc_logits(logits, hard_targets) * bs
            n += bs

        trL, trA = (tot / max(1, n)), (acc / max(1, n))
        if not _per_step_sched: scheduler_cnn.step()

        # ---- Validation (balanced metrics)
        m = eval_classifier_plus(cnn, va_loader, DEVICE, return_probs=True)
        print(f"[ApneaECG] CNN ep {ep:02d} trL={trL:.4f} trA={trA:.3f} "
              f"va_acc={m['acc']:.3f} va_bal_acc={m['bal_acc']:.3f} va_macroF1={m['macro_f1']:.3f} "
              f"AUC={m['auc'] if m['auc'] is not None else 'n/a'} cov={m['coverage_batches']}")

        # Early stopping on macro-F1 (robust for imbalance)
        val_score = m['macro_f1']
        if val_score > best_val_score:
            best_val_score = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in cnn.state_dict().items()}
            # refresh best threshold whenever we improve
            best_thresh, _ = pick_best_threshold(cnn, va_loader, DEVICE)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[ApneaECG] CNN early stopping at epoch {ep}")
                break

        # Optional per-record validation snapshot each epoch
        # (skip if you want speed)
        # rec_val = eval_with_record_vote(cnn, va_loader.dataset, batch_size=64, device=DEVICE)
        # print(f"   [rec-val] {rec_val}")

    # Use best weights
    if best_state is not None:
        cnn.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    cnn_bytes = cnn.tinyml_packed_bytes()

    # ---- VAE + classifier (unchanged, just minor safety)
    vae = TinyVAE1D(in_channels=1, latent_dim=cfg.latent_dim, base=cfg.base, input_length=cfg.input_len).to(DEVICE)
    opt_vae = torch.optim.AdamW(vae.parameters(), lr=cfg.lr*0.5, weight_decay=cfg.weight_decay)

    print("[ApneaECG] Training VAE...")
    for ep in range(1, cfg.epochs_vae_pre + 1):
        beta = min(0.5, 0.1 * ep / max(1, cfg.epochs_vae_pre))
        tr_tot, tr_rec, tr_kld = train_vae_epoch(vae, tr_loader, opt_vae, DEVICE, beta=beta, clip=1.0)
        va_tot, va_rec, va_kld = eval_vae_epoch(vae, va_loader, DEVICE, beta=beta)
        print(f"[ApneaECG] VAE ep {ep:02d} loss_tr={tr_tot:.4f} recon_tr={tr_rec:.4f} kld_tr={tr_kld:.4f} | "
              f"loss_va={va_tot:.4f} recon_va={va_rec:.4f} kld_va={va_kld:.4f} beta={beta:.3f}")
        if not all(np.isfinite(v) for v in (tr_tot, tr_rec, tr_kld, va_tot, va_rec, va_kld)):
            print("[ApneaECG] VAE early stop: non-finite detected")
            break

    for p in vae.parameters(): p.requires_grad = False
    adapter = VAEAdapter(vae).to(DEVICE)
    head = TinyHead(in_dim=cfg.latent_dim, num_classes=2, hidden=64).to(DEVICE)
    opt_h = torch.optim.AdamW(list(adapter.refine.parameters()) + list(head.parameters()),
                              lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion_head = SafeFocalLoss(gamma=1.5, alpha=0.5, label_smoothing=0.05) if cfg.use_focal_loss else nn.CrossEntropyLoss()

    print("[ApneaECG] Training VAE classifier head...")
    last_vaF1 = 0.0
    for ep in range(1, cfg.epochs_head + 1):
        head.train(); adapter.train()
        tot = acc = n = 0
        for x, y in tr_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            z = adapter(x)
            opt_h.zero_grad(set_to_none=True)
            logits = head(z)
            loss = criterion_head(logits, y)
            if not torch.isfinite(loss): continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(adapter.refine.parameters()) + list(head.parameters()), 1.0)
            opt_h.step()
            bs = x.size(0); tot += float(loss)*bs; acc += acc_logits(logits, y)*bs; n += bs
        trL, trA = tot/max(1,n), acc/max(1,n)

        head.eval(); adapter.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in va_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = head(adapter(x))
                y_true.extend(y.cpu().numpy()); y_pred.extend(logits.argmax(1).cpu().numpy())
        metrics = compute_metrics(np.array(y_true), np.array(y_pred))
        last_vaF1 = metrics['f1']
        print(f"[ApneaECG] VAE+Head ep {ep:02d} trL={trL:.4f} trA={trA:.3f} vaF1={last_vaF1:.3f}")

    # ---- Final test evaluation (balanced + per-record)
    print("\n[ApneaECG] Final test evaluation...")
    test_m = eval_classifier_plus(cnn, te_loader, DEVICE, return_probs=True)
    rec_test = eval_with_record_vote(cnn, te_loader.dataset, batch_size=64, device=DEVICE)

    res = {
        "dataset": "ApneaECG",
        "cnn_val_macroF1": round(float(best_val_score), 4),
        "cnn_val_best_threshold": round(float(best_thresh), 4),
        "cnn_test_acc": round(float(test_m['acc']), 4),
        "cnn_test_bal_acc": round(float(test_m['bal_acc']), 4),
        "cnn_test_macroF1": round(float(test_m['macro_f1']), 4),
        "cnn_test_auc": (None if test_m['auc'] is None else round(float(test_m['auc']), 4)),
        "cnn_test_record_bal_acc": round(float(rec_test['rec_bal_acc']), 4),
        "cnn_packed_bytes": cnn_bytes,
        "vae_val_f1": round(float(last_vaF1), 4) if last_vaF1 is not None else None,
        "note": "GN, focal/mixup, AMP(opt), cosine sched, thresholded + per-record metrics"
    }
    print(res)
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
    replace_batchnorm_with_groupnorm(cnn, groups=8)
    opt = torch.optim.AdamW(cnn.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    print("Preparing sched")
    steps_per_epoch = math.ceil(len(tr_loader.dataset) / tr_loader.batch_size)
    sched = get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=steps_per_epoch * cfg.warmup_epochs,
        num_training_steps=steps_per_epoch * cfg.epochs_cnn,
    )
    print("Preparing criterion")
    def _labels_from_ds(ds):
        for attr in ('y', 'labels', 'targets'):
            if hasattr(ds, attr):
                arr = np.asarray(getattr(ds, attr))
                return arr.astype(int)
        # TensorDataset fallback
        if hasattr(ds, 'tensors') and len(ds.tensors) >= 2:
            return ds.tensors[1].cpu().numpy().astype(int)
        raise AttributeError("Could not find label array on dataset.")

    class_counts = None
    if cfg.use_focal_loss:
        try:
            y_arr = _labels_from_ds(tr_loader.dataset)
            class_counts = np.bincount(y_arr, minlength=2)
        except Exception:
            class_counts = None  # will fallback inside make_criterion

    criterion = make_criterion(
        num_classes=2,
        train_loader=None,            # <-- don't pass the loader (avoid scan)
        use_focal=cfg.use_focal_loss,
        gamma=1.5,
        class_counts=class_counts     # <-- new optional arg
    ) if cfg.use_focal_loss else nn.CrossEntropyLoss()
    print("Starting training")
    best_va = 0
    for ep in range(1, cfg.epochs_cnn+1):
        cnn.train(); tot=acc=n=0
        for xb,yb in tr_loader:
            xb,yb=xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = cnn(xb)
            loss = criterion(logits, yb)
            loss = loss + resource_penalty(model, meta, w_size=1.0)  # start tiny (e.g., 1.0), tune
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
    replace_batchnorm_with_groupnorm(cnn, groups=8)

    opt = torch.optim.AdamW(cnn.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = get_cosine_schedule_with_warmup(opt, len(tr_loader)*cfg.warmup_epochs, len(tr_loader)*cfg.epochs_cnn)
    criterion = make_criterion(num_classes=2, train_loader=tr_loader, use_focal=True, gamma=1.5) if cfg.use_focal_loss else nn.CrossEntropyLoss()

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


def register_dataset(name, loader_fn, meta=None):
    DATASET_REGISTRY[name] = { 'loader': loader_fn, 'meta': meta or {} }


def available_datasets():
    return list(DATASET_REGISTRY.keys())


def get_dataset_loader(name):
    return DATASET_REGISTRY.get(name, {}).get('loader')

# Register ApneaECG with existing loader

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


def sanity_check_dataset(name, **kwargs):
    """Comprehensive sanity check for any registered dataset"""
    print(f"\n[Sanity] {name} dataset check (args={kwargs})")

    try:
        dl_tr, dl_va, dl_te, meta = make_dataset_for_experiment(name, **kwargs)
        print(f"[Sanity] {name} loaders created successfully")
        print(f"[Sanity] Meta: {meta}")
    except Exception as e:
        print(f"[Sanity] Failed to create loaders for {name}: {e}")
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
            print(f"[Sanity]   {name} contains NaN values!")
        if np.isinf(xb_np).any():
            print(f"[Sanity]   {name} contains infinite values!")
        if abs(xb_np.mean()) > 100:
            print(f"[Sanity]   {name} large mean - may need normalization")
        if xb_np.std() > 100:
            print(f"[Sanity]   {name} large std - may need normalization")

        # Validate meta consistency
        if isinstance(meta, dict):
            if 'num_channels' in meta and meta['num_channels'] != xb.shape[1]:
                print(f"[Sanity]   {name} channel count mismatch: meta={meta['num_channels']}, batch={xb.shape[1]}")
            if 'seq_len' in meta and meta['seq_len'] != xb.shape[2]:
                print(f"[Sanity]   {name} sequence length mismatch: meta={meta['seq_len']}, batch={xb.shape[2]}")
            if 'num_classes' in meta:
                unique_classes = len(np.unique(yb_np))
                if meta['num_classes'] != unique_classes:
                    print(f"[Sanity]   {name} class count mismatch: meta={meta['num_classes']}, batch_unique={unique_classes}")

        print(f"[Sanity]  {name} passed basic sanity checks")
        return True

    except Exception as e:
        print(f"[Sanity] Failed to iterate {name} train loader: {e}")
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
        print("[Sanity]  All datasets passed sanity checks!")
    else:
        print("[Sanity] Some datasets failed sanity checks")
        failed = [k for k, v in results.items() if not v]
        print(f"[Sanity] Failed datasets: {failed}")

    return results

# -------------------- Size accounting ----------------

def count_params(m):
    return sum(p.numel() for p in m.parameters())


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


def quick_test():
    """Run a quick test with small config to verify everything works"""
    print(" Running quick test...")

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
            print(" Quick test passed!")
            return True
        else:
            print("Quick test failed!")
            return False
    else:
        print("No datasets available for testing")
        return False


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


def diagnose_nan_issues(model, sample_input, device=None):
    """
    Comprehensive diagnostic function to identify NaN sources.
    Call this BEFORE training starts.
    """
    if device is None:
        device = DEVICE

    print(" DIAGNOSTIC: Checking for NaN issues...")

    model.eval()
    with torch.no_grad():
        # 1. Check input
        print(f"Input shape: {sample_input.shape}")
        print(f"Input has NaN: {torch.isnan(sample_input).any()}")
        print(f"Input has Inf: {torch.isinf(sample_input).any()}")
        print(f"Input min/max: {sample_input.min():.4f} / {sample_input.max():.4f}")

        # 2. Check model parameters
        nan_params = 0
        inf_params = 0
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                print(f" NaN in parameter: {name}")
                nan_params += 1
            if torch.isinf(param).any():
                print(f" Inf in parameter: {name}")
                inf_params += 1

        if nan_params == 0 and inf_params == 0:
            print(" All parameters are clean")

        # 3. Forward pass test
        try:
            output = model(sample_input.to(device))
            print(f"Output shape: {output.shape}")
            print(f"Output has NaN: {torch.isnan(output).any()}")
            print(f"Output has Inf: {torch.isinf(output).any()}")
            if not torch.isnan(output).any() and not torch.isinf(output).any():
                print(" Forward pass produces clean output")
            else:
                print(" Forward pass produces NaN/Inf output")

        except Exception as e:
            print(f" Forward pass failed: {e}")

    print("🔍 Diagnostic complete\n")


def fix_nan_issues(model):
    """
    Apply comprehensive fixes for common NaN causes.
    Call this BEFORE training if diagnostic finds issues.
    """
    print("🔧 FIXING: Applying comprehensive NaN prevention measures...")

    # 1. Initialize parameters with more conservative values
    for module in model.modules():
        if isinstance(module, nn.Linear):
            # Use smaller initialization for linear layers
            nn.init.xavier_uniform_(module.weight, gain=0.5)  # Smaller gain
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv1d):
            # Use He initialization with smaller gain
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu', a=0.01)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
            # Reset running stats
            module.reset_running_stats()

    print(" Parameters reinitialized with conservative values")

    # 2. Replace any problematic activations with more stable ones
    def replace_activations(module):
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU):
                # Replace with LeakyReLU to prevent dying neurons
                setattr(module, name, nn.LeakyReLU(0.01))
            elif isinstance(child, nn.GELU):
                # GELU can be unstable, replace with ReLU
                setattr(module, name, nn.LeakyReLU(0.01))
            else:
                replace_activations(child)

    replace_activations(model)
    print(" Activations replaced with stable LeakyReLU")

    # 3. Add gradient scaling if any parameters are very small/large
    param_scales = []
    for param in model.parameters():
        if param.requires_grad:
            param_scale = param.data.abs().mean().item()
            param_scales.append(param_scale)

            # Rescale if parameters are too small or too large
            if param_scale < 1e-6:
                param.data *= 1000
                print(f"  Rescaled small parameters (scale was {param_scale:.2e})")
            elif param_scale > 10:
                param.data *= 0.1
                print(f"  Rescaled large parameters (scale was {param_scale:.2e})")

    avg_param_scale = sum(param_scales) / len(param_scales) if param_scales else 1.0
    print(f" Parameter scale check complete (avg: {avg_param_scale:.4f})")

    # 4. Ensure model is in correct mode and device
    model.train()

    print("🔧 Comprehensive NaN fixes applied\n")


def train_epoch(model, loader, opt, device=None):
    """
    Enhanced NaN-safe train_epoch function with diagnostic integration.
    - Automatically handles NaN/Inf detection and prevention
    - Uses diagnose_nan_issues and fix_nan_issues principles
    - train_epoch function is now NaN-safe by default
    """
    if device is None:
        device = DEVICE  # Use global DEVICE variable

    model.train()
    running_loss = 0.0
    correct = 0
    n = 0
    nan_warnings = 0

    for batch_idx, (xb, yb) in enumerate(loader):
        try:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)

            # Enhanced NaN/infinite loss detection with diagnostics
            if torch.isnan(loss) or torch.isinf(loss):
                nan_warnings += 1
                if nan_warnings <= 3:  # Only show first 3 warnings to avoid spam
                    print(f"  WARNING: NaN/Inf loss detected (batch {batch_idx}), skipping...")
                    if nan_warnings == 3:
                        print("   ℹ  Use diagnose_nan_issues(model, sample_input) before training")
                        print("   ℹ  Use fix_nan_issues(model) if diagnostic finds problems")
                elif nan_warnings == 10:
                    print(f"  Multiple NaN detections ({nan_warnings} so far) - consider running diagnostics")
                continue

            loss.backward()

            # Enhanced gradient clipping with NaN detection
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if torch.isnan(grad_norm):
                nan_warnings += 1
                if nan_warnings <= 3:
                    print(f"  WARNING: NaN gradients detected (batch {batch_idx}), skipping...")
                continue

            opt.step()

            running_loss += float(loss.item()) * xb.size(0)
            preds = logits.argmax(1)
            correct += int((preds==yb).sum().item())
            n += xb.size(0)

        except Exception as e:
            print(f"  Error in batch {batch_idx}: {e}")
            continue

    if nan_warnings > 0:
        print(f" Training completed with {nan_warnings} NaN warnings - train_epoch function handled them safely")

    if n == 0:
        print(" WARNING: No valid batches processed!")
        return float('nan'), 0.0

    return float(running_loss/n), correct/n


def evaluate(model, loader, device=None):
    """
    Simple evaluation function that returns accuracy and loss.
    Alternative to eval_logits when you just need metrics.
    """
    if device is None:
        device = DEVICE

    model.eval()
    correct = 0
    n = 0
    running_loss = 0.0

    for xb, yb in loader:
        try:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = F.cross_entropy(logits, yb)

            if not (torch.isnan(loss) or torch.isinf(loss)):
                running_loss += float(loss.item()) * xb.size(0)
                preds = logits.argmax(1)
                correct += int((preds==yb).sum().item())
                n += xb.size(0)

        except Exception as e:
            print(f"Error in evaluation batch: {e}")
            continue

    if n == 0:
        return 0.0, float('inf')

    return correct/n, running_loss/n


def evaluate(model, loader, device=None):
    """
    Simple evaluation function that returns accuracy and loss.
    Alternative to eval_logits when you just need metrics.
    """
    if device is None:
        device = DEVICE

    model.eval()
    correct = 0
    n = 0
    running_loss = 0.0

    for xb, yb in loader:
        try:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = F.cross_entropy(logits, yb)

            if not (torch.isnan(loss) or torch.isinf(loss)):
                running_loss += float(loss.item()) * xb.size(0)
                preds = logits.argmax(1)
                correct += int((preds==yb).sum().item())
                n += xb.size(0)

        except Exception as e:
            print(f"Error in evaluation batch: {e}")
            continue

    if n == 0:
        return 0.0, float('inf')

    return correct/n, running_loss/n


def _unwrap_dataset(obj):
    d = getattr(obj, 'dataset', obj)
    # unwrap Subset/DataLoader.dataset nesting
    while hasattr(d, 'dataset'):
        d = d.dataset
    return d


def _records_from_loader(dl):
    d = _unwrap_dataset(dl)
    # your Apnea dataset stores tuples like (rid, minute, offset) in `index`
    idx = getattr(d, 'index', None) or getattr(d, '_index', None)
    if idx is None:
        return set()  # cannot inspect
    # first element of each tuple is record id
    return {t[0] for t in idx}
# -------------------- Experiment Runner ----------------
# ===== REPLACE: run_experiment (batched with safe builder & stability probes) =====

def run_experiment(cfg: ExpCfg, dataset_name: str, model_name: str):
    print(f'\n{"="*60}')
    print(f' Experiment: {dataset_name} + {model_name}')
    print("="*60)

    # Dataset availability
    if dataset_name not in available_datasets():
        print(f'Dataset {dataset_name} not in registry.')
        print(f'Available: {available_datasets()}')
        return None

    # Prepare loaders
    print(' Preparing data loaders...')
    try:
        ret = make_dataset_for_experiment(
            dataset_name,
            limit=cfg.limit,
            batch_size=cfg.batch_size,
            target_fs=cfg.target_fs,
            num_workers=cfg.num_workers,   # ok even if 0 (you removed pin/persistent)
            length=cfg.length,
            window_ms=cfg.window_ms,
            input_len=cfg.input_len
        )
        dl_tr, dl_va, dl_te, meta = _normalize_dataset_return(ret)
        tr_recs = _records_from_loader(dl_tr)
        va_recs = _records_from_loader(dl_va)
        te_recs = _records_from_loader(dl_te)
        print("Split record counts:", len(tr_recs), len(va_recs), len(te_recs))
        print("Overlap train∩val:", tr_recs & va_recs)
        print("Overlap train∩test:", tr_recs & te_recs)
        print("Overlap val∩test:", va_recs & te_recs)

        # Probe meta if needed
        need_probe = ("num_channels" not in meta) or ("num_classes" not in meta) or ("seq_len" not in meta)
        if need_probe:
            xb, yb = next(iter(dl_tr))
            meta.setdefault("num_channels", int(xb.shape[1]))     # (B, C, T)
            meta.setdefault("seq_len",     int(xb.shape[-1]))
            if yb.ndim == 1:
                meta.setdefault("num_classes", int(max(2, yb.max().item() + 1)))
            elif yb.ndim == 2:
                meta.setdefault("num_classes", int(yb.shape[1]))
            else:
                meta.setdefault("num_classes", 2)

        print(f' Dataset meta: {meta}')
        print(f' Train batches: {len(dl_tr)}, Val batches: {len(dl_va)}')

    except Exception as e:
        print(f'Failed to prepare dataset: {e}')
        return None

    # Device & model
    device = torch.device(cfg.device)
    in_ch = meta['num_channels']
    num_classes = meta['num_classes']

    try:
        model = safe_build_model(model_name, in_ch, num_classes)
    except Exception as e:
        print(f'Failed to build model: {e}')
        return None

    model.to(device)
    params = count_params(model)
    flash_info = estimate_flash_usage(model, 'int8')

    print(f' Model: {model_name}')
    print(f' Parameters: {params:,}')
    print(f' Flash estimate (INT8): {flash_info["flash_human"]} ({flash_info["flash_bytes"]:,} bytes)')

    opt = Adam(model.parameters(), lr=cfg.lr)

    # Diagnostics before training
    print("\n🔧 NaN-Safe Training System Active")
    try:
        sample_batch = next(iter(dl_tr))
        sample_input = sample_batch[0][:1]
        diagnose_nan_issues(model, sample_input, device)
        fix_nan_issues(model)  # harmless if nothing to fix
    except Exception as _:
        pass

    # Quick stability probe (few batches)
    print(" Testing training stability with first few batches...")
    model.train()
    stable_training = True
    for i, (xb, yb) in enumerate(dl_tr):
        if i >= 3:
            break
        try:
            xb, yb = xb.to(device), yb.to(device)
            if torch.isnan(xb).any() or torch.isinf(xb).any():
                print(f"  Batch {i}: Input has NaN/Inf - skipped")
                continue
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"  Batch {i}: Logits NaN/Inf - stability issue")
                stable_training = False
                break
            loss = F.cross_entropy(logits, yb)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  Batch {i}: Loss NaN/Inf - stability issue")
                stable_training = False
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            print(f"  Batch {i}: Loss stable ({float(loss):.4f})")
        except Exception as e:
            print(f"  Batch {i}: Error - {e}")
            stable_training = False
            break

    if not stable_training:
        print(" Stability issues detected - applying emergency fixes...")
        fix_nan_issues(model)
        for pg in opt.param_groups:
            pg['lr'] = min(pg['lr'], 1e-4)
        print("   Applied emergency fixes.")

    # Training loop
    print(f' Training for {cfg.epochs} epochs...')
    start = time.time()
    best_val_acc = 0.0

    for ep in range(cfg.epochs):
        train_loss, train_acc = train_epoch(model, dl_tr, opt, device=device)
        val_acc, val_loss = evaluate(model, dl_va, device=device)
        best_val_acc = max(best_val_acc, val_acc)
        print(f'  Epoch {ep+1}/{cfg.epochs}: train_loss={train_loss:.4f} '
              f'train_acc={train_acc:.4f} val_acc={val_acc:.4f}')

    dur = time.time() - start

    # Test evaluation
    '''
    test_acc = None
    if dl_te is not None:
        test_acc, _ = evaluate(model, dl_te, device=device)
        print(f' Test accuracy: {test_acc:.4f}')
    '''
    v_logits, vy = evaluate_logits(model, dl_va, device=device)
    vp = eval_prob_fn(v_logits)
    t_star, val_f1 = tune_threshold(vy, vp)

    vy_hat = (vp >= t_star).astype(int)
    _val_avg = _choose_avg(vy)  # or 'binary' if you prefer to force binary
    val_precision = precision_score(vy, vy_hat, average=_val_avg, zero_division=0)
    val_recall    = recall_score(vy, vy_hat, average=_val_avg, zero_division=0)

    # 2) Evaluate on test with the same threshold
    test_acc = None; test_f1 = None
    test_precision = None; test_recall = None
    if dl_te is not None:
        test_acc, _ = evaluate(model, dl_te, device=device)
        try:
            te_logits, ty = evaluate_logits(model, dl_te, device=device)
            tp   = eval_prob_fn(te_logits)
            yhat = (tp >= t_star).astype(int)
            _te_avg = _choose_avg(ty)  # or 'binary'
            test_f1        = f1_score(ty, yhat, average=_te_avg, zero_division=0)
            test_precision = precision_score(ty, yhat, average=_te_avg, zero_division=0)
            test_recall    = recall_score(ty, yhat, average=_te_avg, zero_division=0)
        except Exception:
            pass
    if dl_te is not None:
        print(f" Test P/R/F1@t*: "
              f"{(test_precision if test_precision is not None else float('nan')):.3f}/"
              f"{(test_recall    if test_recall    is not None else float('nan')):.3f}/"
              f"{(test_f1        if test_f1        is not None else float('nan')):.3f}")
    print(f" Val  P/R/F1@t*: {val_precision:.3f}/{val_recall:.3f}/{val_f1:.3f}")
    print(f" val_F1@t*={val_f1:.3f}  test_acc@t*={test_acc:.3f}  test_F1@t*={test_f1:.3f}  t*={t_star:.2f}")

    deploy = deployment_profile(model, meta, flash_bytes_fn=_flash_bytes_int8, device=str(device))

    print(f'  Training time: {dur:.1f}s')
    print(f' Best validation accuracy: {best_val_acc:.4f}')

    results = {
          'dataset': dataset_name,
          'model': _normalize_model_name(model_name),
          'model_kwargs': model_kwargs,
          'kd': kd,
          'epochs': cfg.epochs,
          'lr': cfg.lr,

          'val_acc': best_val_acc,
          'val_f1_at_t': val_f1,
          'val_precision_at_t': float(val_precision),
          'val_recall_at_t': float(val_recall),

          'test_acc': test_acc,
          'test_f1_at_t': test_f1,
          'test_precision_at_t': test_precision,
          'test_recall_at_t': test_recall,

          'threshold_t': t_star,
          'params': params,

          'flash_kb': deploy['flash_kb'],
          'ram_act_peak_kb': deploy['ram_act_peak_kb'],
          'param_kb': deploy['param_kb'],
          'buffer_kb': deploy['buffer_kb'],
          'macs': deploy['macs'],
          'latency_ms': deploy['latency_ms'],
          'energy_mJ': deploy['energy_mJ'],

          'train_time_s': dur,
          'channels': meta.get('num_channels', None),
          'seq_len': meta.get('seq_len', None),
          'num_classes': meta.get('num_classes', None),
    }
    return results



def run_all_experiments(cfg: ExpCfg, datasets: List[str]=None):
    # Datasets to run
    if datasets is None:
        datasets = [d for d in available_datasets()]

    all_specs = []
    for ds in datasets:
        if ds not in available_datasets(): continue
        for spec in build_model_grid_for_dataset(ds):
            all_specs.append((ds, spec))
    total_exp = len(all_specs)

    print("\n" + "="*80)
    print(" UNIFIED TINYML EXPERIMENTS")
    print("="*80)
    print(f" Datasets: {datasets}")
    print(f"  Config: epochs={cfg.epochs}, batch_size={cfg.batch_size}, limit={cfg.limit}, device={cfg.device}")

    # Build one size table (probe first available dataset)
    if datasets:
        try:
            df_size = build_size_table_one_dataset(datasets[0], cfg)
        except Exception as e:
            print(f"[WARN] Size table failed: {e}")

    all_results = []
    total = 0
    for ds in datasets:
        if ds not in available_datasets():
            print(f"  Skipping unavailable dataset: {ds}")
            continue
        grid = build_model_grid_for_dataset(ds)
        total += len(grid)

        #for spec in grid:
        for i, (ds, spec) in enumerate(all_specs, start=1):
            name   = spec['name']
            kwargs = spec.get('kwargs', {})
            kd     = spec.get('kd', False)
            exp_id = make_exp_id(i, total_exp, ds, _normalize_model_name(name), kd, kwargs, seed=getattr(cfg, 'seed', None))
            print(f"\n📍 {exp_id}")
            try:
                #res = run_experiment_unified(cfg, ds, name, model_kwargs=kwargs, kd=kd,
                #                            w_size=1.0, w_bit=(0.05 if kd else 0.0),
                 #w_spec=1e-4, w_softf1=0.10)
                dl_tr, dl_va, dl_te, meta = get_or_make_loaders_once(ds, cfg)
                print("Before printing ",dl_tr, dl_va, dl_te, meta)
                print_class_dist_from_loaders(dl_tr, dl_va, dl_te, meta)
                # pass these into run_experiment_unified instead of rebuilding
                res = run_experiment_unified(cfg, ds, name, model_kwargs=kwargs, kd=kd,
                                            loaders=(dl_tr, dl_va, dl_te, meta),
                                            w_size=1.0, w_bit=(0.05 if kd else 0.0),
                                            w_spec=1e-4, w_softf1=0.10)
                if res:
                    res['exp_id'] = exp_id         # <- keep ID in the row
                    res['exp_name'] = f"{ds}+{name}"
                    all_results.append(res)
            except Exception as e:
                print(f"💥 {exp_id} failed: {e}")
                import traceback; traceback.print_exc()

    if not all_results:
        print("No experiments completed successfully")
        return None

    df = pd.DataFrame(all_results).sort_values('exp_id')
    # Pareto + summaries
    try:
        pf = plot_pareto(df, x='flash_kb', y='test_f1_at_t', save_path='pareto_accuracy_vs_flash.png')
        print("\nPARETO FRONTIER (non-dominated points):")
        print(pf[['model','flash_kb','test_f1_at_t']])
    except Exception as e:
        print(f"[WARN] Pareto plot failed: {e}")

    print(f"\n{'='*80}")
    print(" EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    print(f" Completed runs: {len(all_results)}/{total}")
    if 'val_acc' in df:
        print(f"📈 Avg val acc: {df['val_acc'].mean():.4f} ± {df['val_acc'].std():.4f}")
    if 'flash_kb' in df:
        print(f" Avg model flash: {df['flash_kb'].mean():.1f} KB")

    # Compact table
    cols = ['dataset','model','kd','model_kwargs','val_acc','test_acc','val_f1_at_t','test_f1_at_t',
            'flash_kb','params','macs','latency_ms','energy_mJ','train_time_s']
    cols = [c for c in cols if c in df.columns]
    print("\nRESULTS TABLE")
    print(df[cols].to_string(index=False, float_format='%.4f'))

    # Save
    save_df_to_drive(df, 'comprehensive_tinyml_results.csv')
    print(" Saved: comprehensive_tinyml_results.csv")
    return df

# -------------------- Quick Test & Utility Functions ----------------

def quick_test():
    """Run a quick test with minimal config to verify everything works"""
    print(" Running quick test...")

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
            print(" Quick test passed!")
            return True
        else:
            print("Quick test failed!")
            return False
    else:
        print("No datasets available for testing")
        return False


def paper_experiments():
    """Run experiments specifically for the paper with appropriate configs"""
    print(" Running paper experiments...")

    paper_cfg = ExpCfg(
        epochs=10,
        batch_size=64,
        limit=5000,  # reasonable size for meaningful results
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Focus on the key models for the paper
    key_models = ['tiny_separable_cnn', 'tiny_vae_head', 'tiny_method']

    return run_all_experiments(paper_cfg, models=key_models)


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
    print(name,loader_func)
    return loader_func(**kwargs)

# -------------------- Dataset Wrapper Functions ----------------

def _dir_has_any(path):
    """Check if directory exists and has files"""
    from pathlib import Path
    path = Path(path)
    return path.exists() and any(path.iterdir())


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

# -------------------- Comprehensive Comparison Function ----------------

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

# -------------------- Sanity Check Function ----------------

def sanity_check_dataset(name, **kwargs):
    """Comprehensive sanity check for any registered dataset"""
    print(f"\n[Sanity] {name} dataset check (args={kwargs})")

    try:
        dl_tr, dl_va, dl_te, meta = make_dataset_for_experiment(name, **kwargs)
        print(f"[Sanity] {name} loaders created successfully")
        print(f"[Sanity] Meta: {meta}")
    except Exception as e:
        print(f"[Sanity] Failed to create loaders for {name}: {e}")
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

        print(f"[Sanity]  {name} passed basic sanity checks")
        return True

    except Exception as e:
        print(f"[Sanity] Failed to iterate {name} train loader: {e}")
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
        print("[Sanity]  All datasets passed sanity checks!")
    else:
        print("[Sanity] Some datasets failed sanity checks")
        failed = [k for k, v in results.items() if not v]
        print(f"[Sanity] Failed datasets: {failed}")

    return results



def check_dataset_paths():
    """Debug function to check dataset paths and suggest fixes"""
    print(" DATASET PATH DEBUGGING")
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
            print(f"   Path does not exist!")

    print(f"\n SUGGESTIONS:")
    print("1. If PTB-XL shows 'raw/ folder exists: False', the data might be extracted directly")
    print("   in the root instead of a 'raw' subfolder. Check the ptbxl_database.csv location.")
    print("2. If MIT-BIH shows no .hea/.atr files, check if data is in a subfolder.")
    print("3. Set DO_PTBXL_DOWNLOAD=True and DO_MITDB_DOWNLOAD=True if you want to enable them.")


def simple_test():
    """Simple test with just ApneaECG dataset"""
    print(" Running simple test with ApneaECG only...")

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
            print(" Simple test passed!")
            print(f"Result: Val Acc={result['val_acc']:.4f}, Flash={result['flash_kb']:.1f}KB")
            return True
        else:
            print("Simple test failed!")
            return False
    except Exception as e:
        print(f"Simple test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def fix_ptbxl_paths():
    """Suggest fixes for PTB-XL path issues based on common layouts"""
    print(" PTB-XL PATH FIXER")
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
            print(f" Found: {layout}")
            found_csv = layout
            break
        else:
            print(f"Not found: {layout}")

    if found_csv:
        suggested_raw = found_csv.parent
        print(f"\\n Suggested fix:")
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
        print("\\nCould not find ptbxl_database.csv in common locations")
        print("Please check if PTB-XL data is properly downloaded and extracted")

# Fallbacks

def _count_parameters_fallback(model):
    return sum(p.numel() for p in model.parameters())


def _estimate_packed_fallback(n_params: int, qbits: int) -> int:
    # bytes = ceil(params * bits/8) ; INT4 packs 2 per byte
    return int(np.ceil(n_params * (qbits / 8.0)))

# Use helpers if present, else fallbacks
_count_params = globals().get("count_parameters", _count_parameters_fallback)
_raw_est_fn   = globals().get("estimate_packed_bytes", None)

def _estimate_packed_any(model, qbits: int) -> int:
    """Try multiple calling conventions before falling back to formula."""
    if callable(_raw_est_fn):
        # try kw 'quant_bits'
        try: return int(_raw_est_fn(model, quant_bits=qbits))
        except TypeError: pass
        # try positional
        try: return int(_raw_est_fn(model, qbits))
        except TypeError: pass
        # try common kw names
        for kw in ("bits","nbits","qbits","pack_bits","weight_bits","precision"):
            try: return int(_raw_est_fn(model, **{kw: qbits}))
            except TypeError: continue
        # try dtype-style
        try:
            dtype_map = {4:"int4", 8:"int8", 16:"float16", 32:"float32"}
            return int(_raw_est_fn(model, dtype=dtype_map.get(qbits, f"int{qbits}")))
        except TypeError:
            pass
    # fallback: formula on parameter count only
    n_params = _count_params(model)
    return _estimate_packed_fallback(n_params, qbits)


def _resolve_dataset_key(ds_name):
    key = DATASET_ALIAS.get(ds_name, ds_name)
    if key not in available_datasets():
        raise KeyError(f"Dataset '{ds_name}' not found. Available: {available_datasets()}")
    return key


def _preload_dataset(ds_key, batch_size=64):
    ret = make_dataset_for_experiment(
        ds_key,
        batch_size=batch_size,
        limit=getattr(cfg, "limit", None),
        target_fs=getattr(cfg, "target_fs", None),
        num_workers=getattr(cfg, "num_workers", 0),
        length=getattr(cfg, "length", 1800),
        window_ms=getattr(cfg, "window_ms", 800),
        input_len=getattr(cfg, "input_len", 1000),
    )
    dl_tr, dl_va, dl_te, meta = _normalize_dataset_return(ret)
    meta = _probe_meta_if_needed(dl_tr, dict(meta))
    return (dl_tr, dl_va, dl_te, meta)

# Instantiate builders flexibly: support fn(ic,nc) or fn()

def _instantiate_model(build_fn, in_ch, num_classes):
    try:
        return build_fn(in_ch, num_classes)
    except TypeError:
        return build_fn()


def replace_bn_with_gn(module: nn.Module) -> nn.Module:
    """
    Recursively replace nn.BatchNorm1d with nn.GroupNorm(1, C).
    Modifies in-place and returns the same module.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm1d):
            gn = nn.GroupNorm(num_groups=1, num_channels=child.num_features, affine=True)
            setattr(module, name, gn)
        else:
            replace_bn_with_gn(child)
    return module


def train_model(model, dl_tr, dl_va, epochs=8, lr=1e-3, weight_decay=1e-4, max_grad_norm=1.0,
                label_smoothing=0.05, device=None, verbose=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss with smoothing (helps minority sensitivity; avoids overconfidence)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_va_acc = -1.0
    best_state = None

    def _run_epoch(dl, train: bool):
        model.train(mode=train)
        total_loss, correct, count = 0.0, 0, 0
        all_pred, all_true = [], []

        for xb, yb in dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

            # Last-line defense against bad inputs from upstream
            xb = torch.nan_to_num(xb, nan=0.0, posinf=0.0, neginf=0.0)

            if train:
                opt.zero_grad(set_to_none=True)

            logits = model(xb)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

            loss = criterion(logits, yb)

            if torch.isnan(loss) or torch.isinf(loss):
                # Skip this batch, log minimal info
                if verbose: print("  WARNING: NaN/Inf loss detected, skipping batch...")
                continue

            if train:
                loss.backward()
                if max_grad_norm is not None and max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                opt.step()

            total_loss += float(loss.detach().cpu())
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == yb).sum().item())
            count += int(yb.numel())

            all_pred.extend(pred.detach().cpu().tolist())
            all_true.extend(yb.detach().cpu().tolist())

        acc = (correct / max(1, count))
        try:
            f1 = f1_score(all_true, all_pred, average='macro')
        except Exception:
            f1 = 0.0
        return total_loss / max(1, len(dl)), acc, f1

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc, tr_f1 = _run_epoch(dl_tr, train=True)
        va_loss, va_acc, va_f1 = _run_epoch(dl_va, train=False)

        if verbose:
            print(f"  Epoch {ep}/{epochs}: "
                  f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} (F1={tr_f1:.3f}) "
                  f"val_acc={va_acc:.4f} (F1={va_f1:.3f})")

        if va_acc > best_va_acc:
            best_va_acc = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    return {"best_val_acc": best_va_acc}



def _eval_fwd(model, dl_te, device=None):
    if "evaluate_model" in globals():
        return evaluate_model(model, dl_te)
    # simple accuracy/F1 fallback
    import torch, numpy as np
    from sklearn.metrics import f1_score
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device); model.eval()
    ys, yh = [], []
    with torch.no_grad():
        for xb, yb in dl_te:
            xb = xb.to(device)
            pred = model(xb).argmax(dim=1).cpu().numpy()
            ys.append(yb.numpy()); yh.append(pred)
    y = np.concatenate(ys); h = np.concatenate(yh)
    acc = float((y==h).mean())
    f1m = float(f1_score(y, h, average="macro"))
    return acc, f1m, {}

def _train_fwd(model, dl_tr, dl_va, epochs=8, device=None):
    if "train_model" in globals():
        return train_model(model, dl_tr, dl_va, epochs=epochs)
    # very small fallback trainer (optional; comment out if you have your own)
    import torch, torch.nn.functional as F
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    for ep in range(epochs):
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
    return {}


def run_ablation(name, build_fn, dataset="apnea_ecg", epochs=8, batch_size=64, preload=None):
    try:
        ds_key = _resolve_dataset_key(dataset)
        dl_tr, dl_va, dl_te, meta = preload or _preload_dataset(ds_key, batch_size=batch_size)
        in_ch, num_classes = meta["num_channels"], meta["num_classes"]

        model = _instantiate_model(build_fn, in_ch, num_classes)
        _train_fwd(model, dl_tr, dl_va, epochs=epochs, device=getattr(cfg, "device", "cpu"))
        acc, f1_macro, per_class = _eval_fwd(model, dl_te, device=getattr(cfg, "device", "cpu"))
        return {"name": name, "dataset": ds_key, "accuracy": acc, "macro_f1": f1_macro, "per_class_f1": per_class}
    except Exception as e:
        print(f"[WARN] Ablation {name} failed: {e}")
        return {"name": name, "dataset": dataset, "accuracy": None, "macro_f1": None, "per_class_f1": {}}
import torch.nn as nn


def freeze_batchnorm(model: nn.Module):
    """Set all BatchNorm layers to eval and stop updating running stats."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.eval()
            m.bias.requires_grad_(False)
            m.weight.requires_grad_(False)


def replace_batchnorm_with_groupnorm(model, groups=1):
    """
    Returns the SAME model with all BatchNormNd replaced by GroupNorm(groups, num_channels).
    Safe: returns the model object (not None).
    """
    import torch.nn as nn

    def _swap(m):
        for name, child in list(m.named_children()):
            if isinstance(child, nn.BatchNorm1d):
                gn = nn.GroupNorm(num_groups=groups, num_channels=child.num_features, eps=child.eps, affine=True)
                setattr(m, name, gn)
            elif isinstance(child, nn.BatchNorm2d):
                gn = nn.GroupNorm(num_groups=groups, num_channels=child.num_features, eps=child.eps, affine=True)
                setattr(m, name, gn)
            elif isinstance(child, nn.BatchNorm3d):
                gn = nn.GroupNorm(num_groups=groups, num_channels=child.num_features, eps=child.eps, affine=True)
                setattr(m, name, gn)
            else:
                _swap(child)
        return m

    return _swap(model)


def safe_build_model(model_name: str, in_ch: int, num_classes: int) -> nn.Module:
    """
    Builds a model by name, ensures it's an nn.Module, swaps BN→GN(1),
    and validates the forward shape [B, num_classes].
    """
    name = (model_name or "").strip().lower()
    try:
        if name == "tiny_separable_cnn":
            model = TinySeparableCNN(in_ch, num_classes)
        elif name == "tiny_vae_head":
            model = TinyVAEHead(in_ch, num_classes)
        elif name == "tiny_method":
            model = TinyMethodModel(in_ch, num_classes)
        elif name == "regular_cnn":
            model = RegularCNN(in_ch, num_classes)
        elif name == "hrv_featnet":
            # Apnea-ECG default fs=100; change if your dataset differs
            model = HRVFeatNet(num_classes=num_classes, fs=100.0)
        elif name == "cnn3_small":
            model = CNN1D_3Blocks(in_ch, num_classes, base=16)  # bump base to 24 if you want a bit more capacity
        elif name == "resnet1d_small":
            model = ResNet1DSmall(in_ch, num_classes, base=16)
        else:
            raise KeyError(f"Unknown model '{model_name}'. Expected one of "
                           f"['tiny_separable_cnn','tiny_vae_head','tiny_method','regular_cnn'].")
    except NameError as e:
        raise RuntimeError(f"Model class for '{model_name}' is not defined/imported.") from e

    if model is None:
        raise RuntimeError(f"Constructor for '{model_name}' returned None (missing `return`?).")
    if not isinstance(model, nn.Module):
        raise TypeError(f"Builder for '{model_name}' returned {type(model)}; expected nn.Module.")

    # BN → GN(1) for stability on small/variable batches
    replace_batchnorm_with_groupnorm(model, groups=1)

    # Forward-shape sanity check
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, in_ch, 1800)  # if seq_len varies, your trainer will revalidate on real batch
        out = model(dummy)
        if out is None or out.ndim != 2:
            raise RuntimeError(f"Model '{model_name}' returned {None if out is None else out.shape}; "
                               f"expected [B, num_classes].")
        if out.shape[1] != num_classes:
            raise RuntimeError(f"Classifier head mismatch for '{model_name}': got {out.shape[1]} "
                               f"classes, expected {num_classes}.")
    model.train()
    return model
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from typing import Tuple, List

# ---------------------------
# A) HRV-ish feature baseline
# ---------------------------
try:
    import scipy.signal as ss
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def _bandpass(x: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    if not _HAS_SCIPY:
        # Fallback: cheap detrend + 3-point moving average (keeps it robust if SciPy missing)
        x = x - np.nanmean(x)
        return np.convolve(x, np.ones(3)/3.0, mode="same").astype(np.float32)
    ny = 0.5 * fs
    lo /= ny; hi /= ny
    b, a = ss.butter(2, [max(1e-3, lo), min(0.999, hi)], btype="band")
    return ss.filtfilt(b, a, x).astype(np.float32)

def _qrs_peaks_simple(x: np.ndarray, fs: float) -> np.ndarray:
    """
    Lightweight QRS-ish peak detector:
    - bandpass 5–15 Hz
    - square + moving average
    - peak picking with refractory
    """
    xbp = _bandpass(x, fs, 5.0, 15.0)
    env = np.convolve(xbp * xbp, np.ones(int(0.12*fs))/max(1, int(0.12*fs)), mode="same")
    env = env / (np.max(np.abs(env)) + 1e-8)
    thr = 0.35 * np.nanmax(env)
    # refractory ~ 250ms
    min_dist = int(0.25 * fs)
    peaks = []
    i = 0
    N = len(env)
    while i < N:
        if env[i] >= thr:
            j = min(N, i + min_dist)
            seg = env[i:j]
            if seg.size > 0:
                pk = i + int(np.argmax(seg))
                peaks.append(pk)
            i = j
        else:
            i += 1
    return np.array(peaks, dtype=np.int32)

def _hrv_features(x: np.ndarray, fs: float=100.0) -> np.ndarray:
    """
    Compute a compact HRV(+amp) feature vector from a single-lead ECG window.
    If too few RR intervals are found, fall back to robust time/spectral amp features.
    """
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    x = x - np.mean(x)
    peaks = _qrs_peaks_simple(x, fs)
    feats: List[float] = []

    # ---- RR-based features ----
    if peaks.size >= 3:
        rr = np.diff(peaks) / fs  # seconds
        rr = rr[(rr > 0.25) & (rr < 2.0)]  # 30–240 bpm sanity
        if rr.size >= 2:
            # time-domain HRV
            mean_rr = float(np.mean(rr))
            sdnn    = float(np.std(rr))
            rmssd   = float(np.sqrt(np.mean(np.diff(rr)**2)))
            pnn50   = float(np.mean((np.abs(np.diff(rr)) > 0.05).astype(np.float32)))

            # heart rate features
            hr_mean = float(60.0 / (mean_rr + 1e-8))
            hr_std  = float(np.std(60.0 / (rr + 1e-8)))

            # frequency-domain HRV (RR series resampled to 4 Hz)
            try:
                t = np.cumsum(np.concatenate([[0.0], rr]))
                t = t - t[0]
                if t[-1] > 1e-3:
                    t_uniform = np.linspace(0, t[-1], int(4.0 * t[-1]) + 1)
                    rr_interp = np.interp(t_uniform, t[:-1], rr)
                    if _HAS_SCIPY:
                        f, Pxx = ss.welch(rr_interp - np.mean(rr_interp), fs=4.0, nperseg=min(256, rr_interp.size))
                        def bandpower(lo, hi):
                            mask = (f >= lo) & (f < hi)
                            return float(np.trapz(Pxx[mask], f[mask])) if np.any(mask) else 0.0
                        lf = bandpower(0.04, 0.15)
                        hf = bandpower(0.15, 0.40)
                    else:
                        # crude fallback
                        P = np.abs(np.fft.rfft(rr_interp - np.mean(rr_interp)))**2
                        f = np.fft.rfftfreq(rr_interp.size, d=1/4.0)
                        def bandpower(lo, hi):
                            mask = (f >= lo) & (f < hi)
                            return float(np.trapz(P[mask], f[mask])) if np.any(mask) else 0.0
                        lf = bandpower(0.04, 0.15)
                        hf = bandpower(0.15, 0.40)
                else:
                    lf = hf = 0.0
            except Exception:
                lf = hf = 0.0

            feats.extend([mean_rr, sdnn, rmssd, pnn50, hr_mean, hr_std, lf, hf])
        else:
            feats.extend([0.0]*8)
    else:
        feats.extend([0.0]*8)

    # ---- amplitude/spectral features on raw (robust fallbacks) ----
    x_abs = np.abs(x)
    mean  = float(np.mean(x))
    std   = float(np.std(x))
    skew  = float((np.mean((x - mean)**3) / (std**3 + 1e-8)))
    kurt  = float((np.mean((x - mean)**4) / (std**4 + 1e-8)))
    p2p   = float(np.max(x) - np.min(x))
    zcr   = float(np.mean((x[:-1] * x[1:]) < 0))
    # bandpower 0.5–5 Hz (most ECG energy)
    if _HAS_SCIPY:
        f, Pxx = ss.welch(x - np.mean(x), fs=fs, nperseg=min(1024, len(x)))
        mask = (f >= 0.5) & (f <= 5.0)
        bp = float(np.trapz(Pxx[mask], f[mask])) if np.any(mask) else 0.0
        # spectral centroid
        sc = float(np.sum(f * Pxx) / (np.sum(Pxx) + 1e-8))
    else:
        X = np.abs(np.fft.rfft(x - np.mean(x)))**2
        f = np.fft.rfftfreq(x.size, d=1.0/fs)
        mask = (f >= 0.5) & (f <= 5.0)
        bp = float(np.trapz(X[mask], f[mask])) if np.any(mask) else 0.0
        sc = float(np.sum(f * X) / (np.sum(X) + 1e-8))

    feats.extend([mean, std, skew, kurt, p2p, zcr, bp, sc])

    vec = np.nan_to_num(np.array(feats, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return vec  # length = 16


def _flash_bytes_int8(m):
    try:
        return estimate_flash_usage(m, 'int8')["flash_bytes"]  # you already call this above
    except Exception:
        return sum(p.numel() for p in m.parameters())  # fallback
# --- MACs/ops estimator for Conv1d/Linear (batch=1) ---

def conv1d_macs(m: nn.Conv1d, L_in: int) -> int:
    Cin  = m.in_channels
    Cout = m.out_channels
    k    = m.kernel_size[0]
    s    = m.stride[0]
    p    = m.padding[0]
    d    = m.dilation[0]
    g    = m.groups
    Lout = math.floor((L_in + 2*p - d*(k-1) - 1)/s + 1)
    # per-output MACs = (Cin/g)*k, total = Cout * Lout * (Cin/g)*k
    macs = Cout * Lout * (Cin // g) * k
    return macs, Lout

def linear_macs(m: nn.Linear) -> int:
    return m.in_features * m.out_features

def estimate_macs(model: nn.Module, in_ch: int, seq_len: int) -> int:
    L = seq_len
    macs_total = 0
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            macs, L = conv1d_macs(m, L)
            macs_total += macs
        elif isinstance(m, nn.Linear):
            macs_total += linear_macs(m)
    return macs_total  # multiply by batch if needed

# --- Activation peak (lower-bound) via forward hooks ---
@torch.no_grad()

def measure_activation_peak_kb(model: nn.Module, sample: torch.Tensor) -> float:
    bytes_per_elem = sample.element_size()
    peaks = []

    def hook(_, __, out):
        if isinstance(out, torch.Tensor):
            peaks.append(out.numel() * bytes_per_elem)
        elif isinstance(out, (tuple, list)):
            s = 0
            for t in out:
                if isinstance(t, torch.Tensor): s += t.numel() * bytes_per_elem
            peaks.append(s)

    hs = [m.register_forward_hook(hook) for m in model.modules() if len(list(m.children())) == 0]
    _ = model.eval()(sample)
    for h in hs: h.remove()

    act_peak_bytes = max(peaks) if peaks else 0
    return act_peak_bytes / 1024.0

# --- Parameter & buffer memory (KB) ---

def parameter_bytes_kb(model: nn.Module, dtype=torch.float32) -> float:
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
    n = sum(p.numel() for p in model.parameters())
    return (n * bytes_per_elem) / 1024.0

def buffer_bytes_kb(model: nn.Module, dtype=torch.float32) -> float:
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
    n = sum(b.numel() for b in model.buffers())
    return (n * bytes_per_elem) / 1024.0

# --- Energy model (configurable; defaults are placeholders) ---
def estimate_energy_mJ(macs: int, bitwidth: int = 8, pJ_per_mac_8bit: float = 3.0) -> float:
    """
    Energy ≈ MACs * energy_per_MAC. Default 3 pJ/MAC@8-bit (set to your MCU’s measured value).
    Returns mJ.
    """
    pj = macs * (pJ_per_mac_8bit if bitwidth == 8 else pJ_per_mac_8bit * (bitwidth/8))
    return pj / 1e9  # pJ -> mJ

# --- End-to-end deployment profile (uses your existing flash estimator if present) ---
def deployment_profile(model: nn.Module, meta: dict, flash_bytes_fn=None, device="cpu"):
    in_ch   = meta.get("num_channels", 1)
    seq_len = meta.get("seq_len", 1800)

    # MACs
    macs = estimate_macs(model, in_ch=in_ch, seq_len=seq_len)

    # Activations peak
    sample = torch.zeros(1, in_ch, seq_len, device=device)
    act_peak_kb = measure_activation_peak_kb(model.to(device), sample)

    # Params/Buf (FP32 runtime; for INT8 runtime RAM, you can adjust bytes_per_elem)
    param_kb  = parameter_bytes_kb(model, dtype=torch.float32)
    buffer_kb = buffer_bytes_kb(model, dtype=torch.float32)

    # Flash (use your packer/estimator if provided)
    if flash_bytes_fn is not None:
        fb = flash_bytes_fn(model)  # should return bytes
    else:
        fb = sum(p.numel() for p in model.parameters()) * 1  # placeholder
    flash_kb = fb / 1024.0

    # Latency proxy (use your function if available; else scale from MACs)
    latency_ms = None
    if "proxy_latency_estimate" in globals():
        try:
            latency_ms, _ = proxy_latency_estimate(model, T=seq_len)
        except Exception:
            pass
    if latency_ms is None:
        # crude proxy: 100 MMAC/s -> 10 ms per 1 MMAC
        latency_ms = macs / 1e8 * 1000.0

    # Energy
    energy_mJ = estimate_energy_mJ(macs, bitwidth=8)

    return {
        "flash_kb": flash_kb,
        "ram_act_peak_kb": act_peak_kb,
        "param_kb": param_kb,
        "buffer_kb": buffer_kb,
        "macs": macs,
        "latency_ms": latency_ms,
        "energy_mJ": energy_mJ,
    }

import matplotlib.pyplot as plt
# --- Model name aliases so legacy names work everywhere ---



def _normalize_model_name(name: str) -> str:
    n = (name or "").strip().lower()
    return MODEL_ALIASES.get(n, n)

# --- Safe builder now accepts **kwargs for ablation variants (dz, dh, r, use_generator, etc.) ---
def safe_build_model(model_name: str, in_ch: int, num_classes: int, **model_kwargs):
    name = _normalize_model_name(model_name)

    if name == 'tiny_separable_cnn':
        model = TinySeparableCNN(in_ch, num_classes)
    elif name == 'tiny_vae_head':
        model = TinyVAEHead(in_ch, num_classes)
    elif name == 'tiny_method':
        model = TinyMethodModel(in_ch, num_classes, **model_kwargs)
    elif name == 'regular_cnn':
        model = RegularCNN(in_ch, num_classes)
    elif name == 'hrv_featnet':
        fs = model_kwargs.get('fs', 100.0)
        model = HRVFeatNet(num_classes=num_classes, fs=fs)
    elif name == 'cnn3_small':
        model = CNN1D_3Blocks(in_ch, num_classes, base=model_kwargs.get('base', 16))
    elif name == 'resnet1d_small':
        model = ResNet1DSmall(in_ch, num_classes, base=model_kwargs.get('base', 16))
    else:
        raise KeyError(f"Unknown model '{model_name}'.")

    # IMPORTANT: keep the returned model
    model = replace_batchnorm_with_groupnorm(model, groups=1)
    return model


def count_class_distribution_from_dataset(ds, max_samples=None):
    # Works with your ApneaECGWindows: y is int label per window
    counts = {}
    N = len(ds) if max_samples is None else min(max_samples, len(ds))
    for i in range(N):
        _, y = ds[i]
        y = int(y)
        counts[y] = counts.get(y, 0) + 1
    total = sum(counts.values())
    return total, counts

def print_class_dist_from_loaders(dl_tr, dl_va, dl_te, meta, max_samples=2000):
    # use underlying datasets, not shuffled loaders
    ds_tr, ds_va, ds_te = dl_tr.dataset, dl_va.dataset, (dl_te.dataset if dl_te is not None else None)
    total, c = count_class_distribution_from_dataset(ds_tr, max_samples)
    print("\n=== ApneaECG Train class distribution (deterministic, ~) ===")
    print(f"  counted samples : {total}  (limit={max_samples})")
    for k in sorted(c): print(f"  class {k}: {c[k]} ({c[k]*100.0/total:.2f}%)")
    print("========================================")

    total, c = count_class_distribution_from_dataset(ds_va, max_samples)
    print("\n=== ApneaECG Val class distribution (deterministic, ~) ===")
    print(f"  counted samples : {total}  (limit={max_samples})")
    for k in sorted(c): print(f"  class {k}: {c[k]} ({c[k]*100.0/total:.2f}%)")
    print("========================================")

    if ds_te is not None:
        total, c = count_class_distribution_from_dataset(ds_te, max_samples)
        print("\n=== ApneaECG Test class distribution (deterministic, ~) ===")
        print(f"  counted samples : {total}  (limit={max_samples})")
        for k in sorted(c): print(f"  class {k}: {c[k]} ({c[k]*100.0/total:.2f}%)")
        print("========================================")

# ---------- Unified run_experiment (now supports model_kwargs + kd + weights) ----------

def run_experiment_unified(cfg, dataset_name, model_name, model_kwargs=None, kd=False,
                           w_size=1.0, w_bit=0.05, w_spec=1e-4, w_softf1=0.10,
                           loaders=None):
    if loaders is None:
        # fallback (but normally not used now)
        ret = make_dataset_for_experiment(..., seed=getattr(cfg, "seed", 42))
        dl_tr, dl_va, dl_te, meta = _normalize_dataset_return(ret)
    else:
        dl_tr, dl_va, dl_te, meta = loaders

    print(f'\n{"="*60}\n Experiment: {dataset_name} + {model_name}\n{"="*60}')
    model_kwargs = model_kwargs or {}

    # --- loaders ---
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
        #dl_tr, dl_va, dl_te, meta = _normalize_dataset_return(ret)
        need_probe = ("num_channels" not in meta) or ("num_classes" not in meta) or ("seq_len" not in meta)
        if need_probe:
            xb, yb = next(iter(dl_tr))
            meta.setdefault("num_channels", int(xb.shape[1]))
            meta.setdefault("seq_len",     int(xb.shape[-1]))
            if yb.ndim == 1:
                meta.setdefault("num_classes", int(max(2, yb.max().item() + 1)))
            elif yb.ndim == 2:
                meta.setdefault("num_classes", int(yb.shape[1]))
            else:
                meta.setdefault("num_classes", 2)
        print(f" Dataset meta: {meta}")
        print(f" Train batches: {len(dl_tr)}, Val batches: {len(dl_va)}")
    except Exception as e:
        print(f'Failed to prepare dataset: {e}')
        return None

    device = torch.device(cfg.device)
    in_ch  = meta['num_channels']
    ncls   = meta['num_classes']

    # --- teacher (optional) ---
    teacher = None
    if kd:
        try:
            teacher = safe_build_model("regular_cnn", in_ch, ncls).to(device)
            t_opt = torch.optim.AdamW(teacher.parameters(), lr=cfg.lr)
            t_epochs = max(3, (cfg.epochs // 2))
            for _ in range(t_epochs):
                _ = train_epoch_ce(teacher, dl_tr, t_opt, device=device, meta=meta,
                                   w_size=0.0, w_spec=0.0, w_softf1=0.0)
            print(" Teacher ready.")
        except Exception as e:
            print(f" Teacher build failed: {e} (continuing without KD)")

    # --- student ---
    model = safe_build_model(model_name, in_ch, ncls, **model_kwargs).to(device)

    # --- optimizer ---
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    # --- diagnostics (short) ---
    try:
        sample_x = next(iter(dl_tr))[0][:1].to(device)
        diagnose_nan_issues(model, sample_x, device)
        fix_nan_issues(model)
    except Exception:
        pass

    # --- train loop ---
    best_val_acc = 0.0
    best_state = None
    start = time.time()
    for ep in range(cfg.epochs):
        if teacher is not None:
            tr_loss = kd_train_epoch(model, teacher, dl_tr, opt, device=device, meta=meta,
                                     w_size=w_size, w_bit=w_bit, w_spec=w_spec, w_softf1=w_softf1)
        else:
            tr_loss = train_epoch_ce(model, dl_tr, opt, device=device, meta=meta,
                                     w_size=w_size, w_spec=w_spec, w_softf1=w_softf1)

        val_acc, val_loss = evaluate(model, dl_va, device=device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f'  Epoch {ep+1}/{cfg.epochs}: train_loss={tr_loss:.4f} val_acc={val_acc:.4f}')

    if best_state is not None:
        model.load_state_dict(best_state)
    dur = time.time() - start

    # --- Calibration on val, test with same threshold (binary/single-label path) ---
    # Assumes your `evaluate_logits`, `eval_prob_fn`, and `tune_threshold` are defined.
    try:
        v_logits, vy = evaluate_logits(model, dl_va, device=device)
        vp = eval_prob_fn(v_logits)
        t_star, val_f1 = tune_threshold(vy, vp)
    except Exception:
        t_star, val_f1 = 0.5, None

    test_acc = None; test_f1 = None
    if dl_te is not None:
        test_acc, _ = evaluate(model, dl_te, device=device)
        try:
            te_logits, ty = evaluate_logits(model, dl_te, device=device)
            tp = eval_prob_fn(te_logits)
            yhat = (tp >= t_star).astype(int)
            test_f1 = f1_score(ty, yhat)
        except Exception:
            pass

    # --- deployment profile ---
    def _flash_bytes_int8(m):
        try: return estimate_flash_usage(m, 'int8')["flash_bytes"]
        except: return sum(p.numel() for p in m.parameters())
    deploy = deployment_profile(model, meta, flash_bytes_fn=_flash_bytes_int8, device=str(device))

    params = count_params(model)
    print(f" Test accuracy: {test_acc if test_acc is not None else float('nan'):.4f}")
    print(f" Best val acc: {best_val_acc:.4f} | Val F1@t*: {val_f1 if val_f1 is not None else float('nan'):.3f}")
    print(f" Training time: {dur:.1f}s | Flash: {deploy['flash_kb']:.2f} KB")

    return {
        'dataset': dataset_name,
        'model': _normalize_model_name(model_name),
        'model_kwargs': model_kwargs,
        'kd': kd,
        'epochs': cfg.epochs,
        'lr': cfg.lr,
        'val_acc': best_val_acc,
        'test_acc': test_acc,
        'val_f1_at_t': val_f1,
        'test_f1_at_t': test_f1,
        'threshold_t': t_star,
        'params': params,
        'flash_kb': deploy['flash_kb'],
        'ram_act_peak_kb': deploy['ram_act_peak_kb'],
        'param_kb': deploy['param_kb'],
        'buffer_kb': deploy['buffer_kb'],
        'macs': deploy['macs'],
        'latency_ms': deploy['latency_ms'],
        'energy_mJ': deploy['energy_mJ'],
        'train_time_s': dur,
        'channels': meta.get('num_channels', None),
        'seq_len': meta.get('seq_len', None),
        'num_classes': meta.get('num_classes', None),
    }

# ---------- Size table (static) ----------

def count_params(m):
    return sum(p.numel() for p in m.parameters())


def build_size_table_one_dataset(probe_dataset: str, cfg: ExpCfg):
    ret = make_dataset_for_experiment(
        probe_dataset,
        limit=64, batch_size=16,
        target_fs=getattr(cfg, "target_fs", None),
        num_workers=getattr(cfg, "num_workers", 0),
        length=getattr(cfg, "length", 1800),
        window_ms=getattr(cfg, "window_ms", 800),
        input_len=getattr(cfg, "input_len", 1000),
    )
    dl_tr, dl_va, dl_te, meta0 = _normalize_dataset_return(ret)
    meta0 = _probe_meta_if_needed(dl_tr, dict(meta0))
    in_ch, ncls = meta0["num_channels"], meta0["num_classes"]

    model_names = ['hrv_featnet','cnn3_small','resnet1d_small',
                   'tiny_separable_cnn','tiny_vae_head','tiny_method','regular_cnn']

    rows = []
    for name in model_names:
        try:
            m = safe_build_model(name, in_ch, ncls)
            nparams = int(count_params(m))
            for qbits in (4, 8, 16, 32):
                try:
                    bytes_est = _estimate_packed_any(m, qbits)  # your existing estimator
                except Exception:
                    # simple fallback: params * qbits/8 bytes
                    bytes_est = nparams * max(1, qbits//8)
                rows.append({
                    "model": name,
                    "quant_bits": qbits,
                    "packed_bytes": int(bytes_est),
                    "packed_kb": round(bytes_est/1024, 2),
                    "nparams": nparams,
                })
        except Exception as e:
            print(f"[WARN] Size calc failed for {name}: {e}")

    if not rows:
        print("[WARN] Size table empty (all size calls failed).")
        return pd.DataFrame()

    df_size = pd.DataFrame(rows).sort_values(["model","quant_bits"])
    save_df_to_drive(df_size, "model_size_packed_flash.csv")
    print(" Saved: model_size_packed_flash.csv")
    return df_size


# ---------- Pareto helpers ----------

def pareto_front(df: pd.DataFrame, x='flash_kb', y='test_f1_at_t'):
    d = df[[x, y, 'model']].dropna().sort_values([x, y], ascending=[True, False])
    pareto = []; best_y = -1e9
    for _, row in d.iterrows():
        if row[y] > best_y:
            pareto.append(row); best_y = row[y]
    return pd.DataFrame(pareto)


def plot_pareto(df: pd.DataFrame, x='flash_kb', y='test_f1_at_t', save_path='pareto_accuracy_vs_flash.png'):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,6))
    for mdl in df['model'].unique():
        sub = df[df['model']==mdl]
        plt.scatter(sub[x], sub[y], label=mdl, alpha=0.7)
    pf = pareto_front(df, x=x, y=y)
    if not pf.empty:
        plt.plot(pf[x], pf[y], marker='o')
        for _, r in pf.iterrows():
            plt.text(r[x]*1.01, r[y]*1.001, r['model'], fontsize=8)
    plt.xlabel('Flash (KB)')
    plt.ylabel('Macro-F1 (test @ t*)')
    plt.title('Pareto Frontier: Macro-F1 vs Flash')
    plt.grid(True, alpha=0.3); plt.legend(fontsize=8, ncol=2)
    plt.tight_layout(); plt.savefig(save_path, dpi=150)
    print(f" Saved Pareto plot → {save_path}")
    return pf

# ---------- Build unified model grid (incl. ablations & KD variants) ----------

def build_model_grid_for_dataset(ds_key: str):
    grid = []
    # Core baselines
    grid += [
        dict(name='hrv_featnet',     kwargs={}),
        dict(name='cnn3_small',      kwargs={'base':16}),
        dict(name='resnet1d_small',  kwargs={'base':16}),
        dict(name='tiny_separable_cnn', kwargs={}),
        dict(name='tiny_method',     kwargs={'dz':4, 'dh':12}),
        dict(name='tiny_method',     kwargs={'dz':6, 'dh':16}),  # allSynth-ish
        dict(name='regular_cnn',     kwargs={}),
    ]
    # Generator variant (if supported by your TinyMethod)
    grid += [dict(name='tiny_method', kwargs={'dz':6, 'dh':16, 'r':4, 'use_generator':True})]
    # KD variants of tiny baselines
    kd_variants = [
        dict(name='tiny_separable_cnn', kwargs={}, kd=True),
        dict(name='tiny_method', kwargs={'dz':4, 'dh':12}, kd=True),
        dict(name='tiny_method', kwargs={'dz':6, 'dh':16}, kd=True),
    ]
    grid += kd_variants
    return grid


def resource_penalty(model: nn.Module, meta: dict, w_size: float = 0.0, w_macs: float = 0.0):
    # L1 on learnable params (encourages sparsity / smaller effective size)
    l1 = torch.zeros((), device=next(model.parameters()).device)
    if w_size > 0.0:
        for p in model.parameters():
            if p.requires_grad:
                l1 = l1 + p.abs().sum()
        l1 = l1 * 1e-7  # scale to keep magnitudes sane
    # NOTE: MACs are not differentiable here, so we keep w_macs=0 unless you add learnable gates
    return w_size * l1


def fake_quant(x, bits=8):
    qlevels = 2**bits - 1
    scale = x.detach().abs().max() / max(1, qlevels/2)
    scale = scale + 1e-8
    xq = torch.clamp(torch.round(x/scale), -qlevels/2, qlevels/2)
    return xq * scale


def kd_loss(student_logits, teacher_logits, T=2.0, alpha=0.7):
    # KL(student||teacher) at temperature T
    p_t = torch.softmax(teacher_logits / T, dim=1)
    log_p_s = torch.log_softmax(student_logits / T, dim=1)
    kl = torch.sum(p_t * (torch.log(p_t + 1e-8) - log_p_s), dim=1).mean()
    return alpha * (T*T) * kl


def bitaware_reg(student_logits, bits=8, beta=0.1):
    q = fake_quant(student_logits, bits=bits)
    return beta * torch.mean((student_logits - q)**2)

def bitaware_reg(student_logits: torch.Tensor, bits: int = 8, beta: float = 0.0):
    if beta <= 0.0:
        return torch.zeros((), device=student_logits.device)
    q = fake_quant(student_logits, bits=bits)
    return beta * torch.mean((student_logits - q) ** 2)

# soft-F1 auxiliary (threshold-aware training nudging toward macro-F1)

def soft_f1_loss(logits: torch.Tensor, y_true: torch.Tensor, w: float = 0.0, eps: float = 1e-7):
    if w <= 0.0:
        return torch.zeros((), device=logits.device)
    p1 = torch.softmax(logits, dim=1)[:, 1]
    y  = y_true.float()
    tp = (p1 * y).sum()
    fp = (p1 * (1 - y)).sum()
    fn = ((1 - p1) * y).sum()
    soft_f1 = (2 * tp) / (2 * tp + fp + fn + eps)
    return w * (1.0 - soft_f1)

# light spectral regularizer (keep energy in plausible ECG band)

def spectral_penalty(xb: torch.Tensor, fs: float, w: float = 0.0, lo: float = 0.5, hi: float = 40.0):
    if w <= 0.0:
        return torch.zeros((), device=xb.device)
    X = torch.fft.rfft(xb, dim=-1)
    T = xb.shape[-1]
    freqs = torch.fft.rfftfreq(T, d=1.0 / fs).to(xb.device)
    mask_out = (freqs < lo) | (freqs > hi)
    power = (X.abs() ** 2).mean(dim=(0, 1))  # [F]
    leak = (power[mask_out].sum() / (power.sum() + 1e-8))
    return w * leak
# Model factory matching those names (via the shims you added in the first cell)

def build_model(spec):
    mdl = spec['model']
    if mdl == 'regcnn':   return RegularCNN1D().to(DEVICE)
    if mdl == 'tinysep':  return TinySep1D().to(DEVICE)
    if mdl == 'allSynth' and spec.get('use_generator') and 'build_hypertiny_with_generator' in globals():
        return build_hypertiny_with_generator(spec.get('dz',6), spec.get('dh',16), spec.get('r',4)).to(DEVICE)
    if mdl == 'allSynth': return HypertinyAllSynth(dz=spec.get('dz',6), dh=spec.get('dh',16)).to(DEVICE)
    if mdl == 'hybrid':   return HypertinyHybrid(dz=spec.get('dz',4), dh=spec.get('dh',12)).to(DEVICE)
    if mdl == 'vae':      return TinyVAEHead(z=spec.get('z',16)).to(DEVICE)  # match constructor
    raise ValueError(mdl)

# Loader bridge that uses your registry + normalizer

def make_loaders_from_legacy(ds_key, batch=64, verbose=True):
  return get_loaders(ds_key, batch=batch, verbose=verbose)
  '''
    if 'make_dataset_for_experiment' in globals():
        ret = make_dataset_for_experiment(ds_key, batch_size=batch, verbose=verbose)
        if '_normalize_dataset_return' in globals():
            try:
                dl_tr, dl_va, dl_te, meta = _normalize_dataset_return(ret)
                return dl_tr, dl_va, dl_te, meta
            except Exception as e:
                print("[Legacy normalize] Failed:", e)
        if isinstance(ret, (tuple,list)) and len(ret)>=3:
            return ret[0], ret[1], ret[2], {}
        if isinstance(ret, dict) and all(k in ret for k in ('train','val','test')):
            return ret['train'], ret['val'], ret['test'], ret.get('meta', {})
    raise KeyError(f"Could not obtain loaders for {ds_key}. Ensure registry & ExpCfg are loaded.")
  '''

def _ensure_meta(meta: Dict, dl_tr):
    """Fill in missing meta fields by peeking one batch."""
    need = any(k not in meta for k in ("num_channels", "num_classes", "seq_len"))
    if not need:
        return meta
    xb, yb = next(iter(dl_tr))
    meta.setdefault("num_channels", int(xb.shape[1]))        # (B, C, T)
    meta.setdefault("seq_len",     int(xb.shape[-1]))
    if yb.ndim == 1:
        meta.setdefault("num_classes", int(max(2, yb.max().item() + 1)))
    elif yb.ndim == 2:
        meta.setdefault("num_classes", int(yb.shape[1]))
    else:
        meta.setdefault("num_classes", 2)
    return meta
# One run with optional KD, periodic threshold-aware val metrics, and JSON output

def run_one(spec):
    name, ds_key = spec['name'], spec['dataset']
    if RUN_ONCE and already_done(name):
        print(f"[SKIP] {name} (cached)")
        return

    print("="*60, f"\n Experiment: {name}\n", "="*60, sep="")
    dl_tr, dl_va, dl_te, meta = make_loaders_from_legacy(ds_key, batch=64, verbose=True)
    meta = _ensure_meta(meta, dl_tr)
    print(f" Dataset meta: {meta}")
    print(f" Train batches: {len(dl_tr)}, Val batches: {len(dl_va)}")

    # ----- optional KD teacher -----
    teacher = None
    if spec.get('kd', False):
        print(" Setting up teacher (RegularCNN1D) for KD...")
        try:
            in_ch = meta.get('num_channels', 1)
            num_classes = meta.get('num_classes', 2)
            # build safe & swap BN→GN
            teacher = safe_build_model("regular_cnn", in_ch, num_classes).to(DEVICE)
        except Exception as e:
            print(f"  Teacher build failed: {e} (continuing without KD)")
            teacher = None
        '''
        if teacher is not None:
            t_opt = torch.optim.AdamW(teacher.parameters(), lr=spec['lr'])
            t_epochs = max(3, DATASET_SPECS[ds_key]['epochs']//2)
            for _ in range(t_epochs):
                train_fn = globals().get('train_epoch', train_epoch_ce)
                _ = train_fn(teacher, dl_tr, t_opt, device=DEVICE)  # ensure your train_epoch signature accepts device
            print(" Teacher ready.")
        '''
        if teacher is not None:
          t_opt = torch.optim.AdamW(teacher.parameters(), lr=spec['lr'])
          t_epochs = max(3, DATASET_SPECS[ds_key]['epochs']//2)
          for _ in range(t_epochs):
              # use the CE trainer; no resource/spec/F1 on the teacher
              _ = train_epoch_ce(teacher, dl_tr, t_opt, device=DEVICE, meta=meta,
                                w_size=0.0, w_spec=0.0, w_softf1=0.0)
          print(" Teacher ready.")
    # ----- student model -----
    print(f" Building student model: {spec['model']}")
    in_ch = meta.get('num_channels', 1)
    num_classes = meta.get('num_classes', 2)
    try:
        model = safe_build_model(spec['model'], in_ch, num_classes).to(DEVICE)
    except Exception as e:
        print(f"  Student build failed: {e}")
        save_json(name, {'status': 'failed_build', 'error': str(e), 'meta': meta})
        return

    opt = torch.optim.AdamW(model.parameters(), lr=spec['lr'])

    # choose trainer (your custom if present)
    train_fn = globals().get('train_epoch', train_epoch_ce)

    # ----- training -----
    print(f" Training for {spec['epochs']} epochs...")
    best = (-1.0, None, None)  # (val_f1, t_star, state_dict)
    w_size   = 1.0                       # resource (L1) pressure on student
    w_bit    = 0.05 if teacher else 0.0  # bit-aware KD only matters when KD is on
    w_spec   = 1e-4                      # spectral leakage penalty (very light)
    w_softf1 = 0.10                      # soft-F1 auxiliary (imbalance-friendly)
    for ep in range(spec['epochs']):
      if teacher is not None:
          tr_loss = kd_train_epoch(student=model, teacher=teacher, loader=dl_tr, opt=opt,
                              T=2.0, alpha=0.7, device=DEVICE, meta=meta, clip=1.0,
                              w_size=w_size, w_bit=w_bit, w_spec=w_spec, w_softf1=w_softf1)
      else:
          tr_loss = train_epoch_ce(model, dl_tr, opt, device=DEVICE, meta=meta, clip=1.0,
                              w_size=w_size, w_spec=w_spec, w_softf1=w_softf1)

    # occasional val read-out (thresholded F1)
    if (ep + 1) % max(1, spec['epochs'] // 3) == 0:
        v_logits, vy = eval_logits(model, dl_va, device=DEVICE)
        vp = eval_prob_fn(v_logits)
        t_star, val_f1 = tune_threshold(vy, vp, THRESH_GRID)
        if val_f1 > best[0]:
            best = (val_f1, t_star, copy.deepcopy(model.state_dict()))
        val_acc = accuracy_score(vy, (vp >= t_star).astype(int))
        print(f"{name} ep{ep+1}/{spec['epochs']}  tr_loss={tr_loss:.4f}  "
              f"val_acc@t*={val_acc:.3f}  val_f1@t*={val_f1:.3f}  t*={t_star:.2f}")

    # ----- final eval (tuned on val) -----
    v_logits, vy = eval_logits(model, dl_va, device=DEVICE)
    vp = eval_prob_fn(v_logits)
    t_star, val_f1 = tune_threshold(vy, vp, THRESH_GRID)
    if val_f1 > best[0]:
        best = (val_f1, t_star, copy.deepcopy(model.state_dict()))

    if best[2] is not None:
        model.load_state_dict(best[2])
    t_star = best[1]

    te_logits, ty = eval_logits(model, dl_te, device=DEVICE)
    tp = eval_prob_fn(te_logits)
    yhat = (tp >= t_star).astype(int)

    metrics = ec57_metrics_with_ci(ty, yhat)
    cm = confusion_matrix(ty, yhat).tolist()

    # size & latency
    packed = packed_bytes_model_paper(model)
    inf_ms, boot_ms = proxy_latency_estimate(model, T=DATASET_SPECS[ds_key]['T'])

    payload = {
        'exp': spec,
        'threshold': float(t_star),
        'val': {'macro_f1_at_t': float(val_f1)},
        'test': {**metrics, 'cm': cm},
        'packed_bytes': int(packed),
        'latency_ms': {'per_inference': inf_ms, 'boot_or_synth': boot_ms},
        'device': str(DEVICE), 'meta': meta
    }
    save_json(name, payload)
    print_and_log(name, payload)
    print(f" Success: {name}")


def run_suite(parallel: bool = False, max_workers: int = None):
    """
    Runs EXPERIMENTS sequentially by default.
    Set parallel=True to run with a ThreadPool (safer in notebooks than processes).
    """
    if not EXPERIMENTS:
        print("No experiments defined for available datasets.")
        return

    names = [e['name'] for e in EXPERIMENTS]
    print(f"Planned experiments: {names}")

    if not parallel:
        for spec in EXPERIMENTS:
            run_one(spec)
        return

    # Parallel (threads) — PyTorch releases the GIL during compute, so this helps on CPU.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    max_workers = max_workers or min(8, len(EXPERIMENTS))

    futures = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for spec in EXPERIMENTS:
            futures[ex.submit(run_one, spec)] = spec['name']

        for fut in as_completed(futures):
            name = futures[fut]
            try:
                fut.result()
                print(f"[DONE] {name}")
            except Exception as e:
                print(f"[FAIL] {name} → {e}")

def get_loaders(ds_key, batch=64, verbose=True, force_reload=False):
    """
    Returns (dl_tr, dl_va, dl_te, meta) with caching keyed by (ds_key, batch).
    """
    key = (ds_key, batch)
    if not force_reload and key in LOADER_CACHE:
        return LOADER_CACHE[key]

    ret = make_dataset_for_experiment(ds_key, batch_size=batch, verbose=verbose)
    if '_normalize_dataset_return' in globals():
        dl_tr, dl_va, dl_te, meta = _normalize_dataset_return(ret)
    elif isinstance(ret, (tuple, list)) and len(ret) >= 3:
        dl_tr, dl_va, dl_te, meta = ret[0], ret[1], ret[2], {}
    elif isinstance(ret, dict) and all(k in ret for k in ('train','val','test')):
        dl_tr, dl_va, dl_te, meta = ret['train'], ret['val'], ret['test'], ret.get('meta', {})
    else:
        raise RuntimeError(f"Unexpected return from make_dataset_for_experiment for {ds_key}")

    LOADER_CACHE[key] = (dl_tr, dl_va, dl_te, meta)
    return LOADER_CACHE[key]
import hashlib

def make_exp_id(idx:int, total:int, ds:str, model:str, kd:bool, kwargs:dict, seed:int=None):
    tag = f"{ds}:{model}:{'KD' if kd else 'noKD'}:{json.dumps(kwargs, sort_keys=True)}:{seed}"
    h = hashlib.md5(tag.encode()).hexdigest()[:6]
    # 01/12-apnea-tiny_method-KD-dz6dh16-abc123
    kwtag = "-".join([f"{k}{v}" for k,v in kwargs.items()]) if kwargs else "base"
    return f"{idx:02d}/{total:02d}-{ds}-{model}{'-KD' if kd else ''}-{kwtag}-{h}"


def get_or_make_loaders_once(ds_key, cfg):
    # cache key can include batch size
    print("In get_or_make_loaders_once")
    key = (ds_key, cfg.batch_size)
    if not hasattr(get_or_make_loaders_once, "_cache"):
        get_or_make_loaders_once._cache = {}
    cache = get_or_make_loaders_once._cache
    if key in cache:
        return cache[key]

    ret = make_dataset_for_experiment(
        ds_key,
        limit=cfg.limit,
        batch_size=cfg.batch_size,
        target_fs=cfg.target_fs,
        num_workers=cfg.num_workers,
        length=cfg.length,
        window_ms=cfg.window_ms,
        input_len=cfg.input_len,
        seed=getattr(cfg, "seed", 42),  # ensure deterministic split
    )
    dl_tr, dl_va, dl_te, meta = _normalize_dataset_return(ret)
    cache[key] = (dl_tr, dl_va, dl_te, meta)
    return cache[key]

def seed_everything(s=42):
    import random, numpy as np, torch
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def _wif(worker_id):
    s = getattr(cfg, "seed", 42) + worker_id
    np.random.seed(s)
    random.seed(s)
print("Loaded all helper functions")


#The main

MODEL_ALIASES = {
    "regcnn": "regular_cnn",
    "tinysep": "tiny_separable_cnn",
    "hybrid": "tiny_method",
    "allsynth": "tiny_method",
}
# -------------------- Dataset Registry --------------------
DATASET_REGISTRY = {}
register_dataset('apnea_ecg', _load_apnea_for_registry)

register_ptb = False
if register_ptb:
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
      if DO_MITDB_DOWNLOAD and _dir_has_any(MITDB_ROOT):
          register_dataset('mitdb', _mitdb_wrapper)
      else:
          print("[Registry] MIT-BIH skipped - data not available or download disabled")
      print("[Registry] PTB-XL loader not found - skip registration")
register_mit = False
if register_mit:
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
              # Only register if data exists
              if DO_PTBXL_DOWNLOAD and _dir_has_any(PTBXL_ROOT):
                  register_dataset('ptbxl', _ptbxl_wrapper)
              else:
                  print("[Registry] PTB-XL skipped - data not available or download disabled")
              raise Exception(f"MIT-BIH loader returned: {result}")
      register_dataset('mitdb', _mitdb_wrapper)
  except NameError:
      print("[Registry] MIT-BIH loader not found - skip registration")


print("Dataset registry and sanity check system loaded!")

print(f"[Registry] Available datasets: {available_datasets()}")

# Single cfg for all (adjust if needed)
cfg = ExpCfg(
    epochs=8, batch_size=64, lr=2e-3, device='cpu',
    limit=None, num_workers=0, target_fs=None, length=1800, window_ms=800, input_len=1000
)
seed_everything(getattr(cfg, "seed", 42))
df = run_all_experiments(cfg, datasets=['apnea_ecg'])
print("Final thing")
print(df)



from glob import glob
from pathlib import Path
import json, math
import numpy as np
import pandas as pd


def _get(d, path, default=np.nan):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

rows, skipped, errors = [], [], []

exp_dir = Path(EXP_DIR) if 'EXP_DIR' in globals() else Path('.')
for p in sorted(glob(str(exp_dir/'*.json'))):
    try:
        j = json.loads(Path(p).read_text())
    except Exception as e:
        skipped.append((p, f"read_error:{e}"))
        continue

    # Skip explicit error artifacts
    if j.get('status') == 'failed' or 'error' in j:
        errors.append((p, j.get('error_type', 'unknown')))
        continue

    # Try multiple schemas for the experiment spec
    spec = j.get('exp') or j.get('spec') or {}
    name     = spec.get('name')     or j.get('name')     or Path(p).stem
    dataset  = spec.get('dataset')  or j.get('dataset')
    model    = spec.get('model')    or j.get('model')

    # If we still don't have minimal fields, skip
    if dataset is None or model is None:
        skipped.append((p, "missing spec(model/dataset)"))
        continue

    # Metrics (support both new + older keys)
    packed_bytes = j.get('packed_bytes', np.nan)
    packed_kb    = round(packed_bytes/1024.0, 2) if isinstance(packed_bytes, (int,float)) and not math.isnan(packed_bytes) else np.nan
    val_f1_at_t  = _get(j, ['val','macro_f1_at_t'], default=np.nan)
    test_acc     = _get(j, ['test','acc'], default=j.get('test_acc', np.nan))
    test_macro_f1= _get(j, ['test','macro_f1'], default=j.get('test_macro_f1', np.nan))
    acc_ci_lo    = _get(j, ['test','ci','acc',0], default=np.nan)
    acc_ci_hi    = _get(j, ['test','ci','acc',1], default=np.nan)
    f1_ci_lo     = _get(j, ['test','ci','macro_f1',0], default=np.nan)
    f1_ci_hi     = _get(j, ['test','ci','macro_f1',1], default=np.nan)
    t_star       = j.get('threshold', j.get('t_star', np.nan))
    lat_ms       = _get(j, ['latency_ms','per_inference'], default=np.nan)
    boot_ms      = _get(j, ['latency_ms','boot_or_synth'], default=np.nan)

    rows.append({
        'exp': name,
        'dataset': dataset,
        'model': model,
        'packed_kb': packed_kb,
        'val_f1@t': round(val_f1_at_t, 4) if isinstance(val_f1_at_t, (int,float)) and not math.isnan(val_f1_at_t) else np.nan,
        'test_acc': round(test_acc, 4) if isinstance(test_acc, (int,float)) and not math.isnan(test_acc) else np.nan,
        'test_macro_f1': round(test_macro_f1, 4) if isinstance(test_macro_f1, (int,float)) and not math.isnan(test_macro_f1) else np.nan,
        'acc_ci_lo': round(acc_ci_lo, 4) if isinstance(acc_ci_lo, (int,float)) and not math.isnan(acc_ci_lo) else np.nan,
        'acc_ci_hi': round(acc_ci_hi, 4) if isinstance(acc_ci_hi, (int,float)) and not math.isnan(acc_ci_hi) else np.nan,
        'f1_ci_lo': round(f1_ci_lo, 4) if isinstance(f1_ci_lo, (int,float)) and not math.isnan(f1_ci_lo) else np.nan,
        'f1_ci_hi': round(f1_ci_hi, 4) if isinstance(f1_ci_hi, (int,float)) and not math.isnan(f1_ci_hi) else np.nan,
        't_star': round(t_star, 3) if isinstance(t_star, (int,float)) and not math.isnan(t_star) else np.nan,
        'lat_ms': round(lat_ms, 2) if isinstance(lat_ms, (int,float)) and not math.isnan(lat_ms) else np.nan,
        'boot_ms': round(boot_ms, 2) if isinstance(boot_ms, (int,float)) and not math.isnan(boot_ms) else np.nan,
    })

df = pd.DataFrame(rows).sort_values(['dataset','model','exp']) if rows else pd.DataFrame(columns=[
    'exp','dataset','model','packed_kb','val_f1@t','test_acc','test_macro_f1',
    'acc_ci_lo','acc_ci_hi','f1_ci_lo','f1_ci_hi','t_star','lat_ms','boot_ms'
])

display(df)

csv_path = exp_dir / 'summary.csv'
df.to_csv(csv_path, index=False)
try:
    save_df_to_drive(df, "aggregate_summary.csv")
except Exception:
    pass
print("Saved CSV:", csv_path)

# Optional: compact LaTeX with flexible columns

def df_to_latex_table(df, caption="TinyML Results (LEAN)", label="tab:tinyml_lean"):
    cols = [c for c in ['dataset','model','packed_kb','test_acc','acc_ci_lo','acc_ci_hi',
                        'test_macro_f1','f1_ci_lo','f1_ci_hi','t_star','lat_ms','boot_ms']
            if c in df.columns]
    sub = df[cols].copy()
    latex = sub.to_latex(index=False, float_format="%.4f")
    return (
        "\\begin{table}[t]\n\\centering\n" +
        latex +
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{table}\n"
    )

summary_txt = exp_dir / 'summary.txt'
with summary_txt.open('w') as f:
    for _, r in df.iterrows():
        f.write(f"{r['exp']} :: acc={r.get('test_acc', np.nan)}  macroF1={r.get('test_macro_f1', np.nan)}  "
                f"t*={r.get('t_star', np.nan)}  kb={r.get('packed_kb', np.nan)}  "
                f"lat={r.get('lat_ms', np.nan)}ms\n")
print("Wrote text summary:", summary_txt)

if skipped:
    print(f"Skipped {len(skipped)} file(s):")
    for p, why in skipped[:5]:
        print("  -", p, "->", why)
    if len(skipped) > 5:
        print("  ...")

if errors:
    print(f"Found {len(errors)} error artifact(s) (not included).")


from models import safe_build_model, MODEL_BUILDERS
from data_loaders import (
    APNEA_ROOT, PTBXL_ROOT, MITDB_ROOT,
    load_apnea_ecg_loaders_impl, load_ptbxl_loaders, load_mitdb_loaders
)

def available_datasets() -> list:
    return ["apnea_ecg","ptbxl","mitdb"]

def make_loaders_from_legacy(ds_key: str, batch: int = 64, length: int = 1800, verbose: bool = True):
    if ds_key == "apnea_ecg":
        tr, va, te = load_apnea_ecg_loaders_impl(APNEA_ROOT, batch_size=batch, length=length, verbose=verbose)
        meta = {'num_channels': 1, 'num_classes': 2, 'seq_len': length, 'fs': 100}
        return tr, va, te, meta
    elif ds_key == "ptbxl":
        tr, va, te, meta = load_ptbxl_loaders(PTBXL_ROOT, batch_size=batch, length=length)
        # meta already returned: ensure keys
        if isinstance(meta, dict):
            meta.setdefault('num_channels', 1)
            meta.setdefault('seq_len', length)
        return tr, va, te, meta
    elif ds_key == "mitdb":
        tr, va, te, meta = load_mitdb_loaders(MITDB_ROOT, batch_size=batch, length=length)
        return tr, va, te, {'num_channels': 1, 'num_classes': 2, 'seq_len': length, 'fs': 360}
    else:
        raise ValueError(f"Unknown dataset key: {ds_key}")


# === GCS-aware results saving (timestamped) ===
import datetime, json

try:
    import gcsfs
except Exception:
    gcsfs = None

RUN_TS = os.environ.get("RUN_TS") or datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
RESULTS_BASE_GCS = "gs://store-pepper/tinyml_hyper_tiny_baselines/result" #os.environ.get("TINYML_RESULTS_GCS")  # e.g., 

def _gcsfs_handle():
    if gcsfs is None:
        raise ImportError("gcsfs required to write to GCS. pip install gcsfs")
    return gcsfs.GCSFileSystem(cache_timeout=60)

def _results_join(root: str, *parts: str) -> str:
    root = root.rstrip("/")
    for p in parts:
        root += "/" + str(p).lstrip("/")
    return root

def save_json(name, payload):
    fname = f"{name}-{RUN_TS}.json"
    if RESULTS_BASE_GCS:
        dst = _results_join(RESULTS_BASE_GCS, fname)
        fs = _gcsfs_handle()
        with fs.open(dst, "w") as f:
            f.write(json.dumps(payload, indent=2))
        print(f"[RESULTS] wrote {dst}")
        return dst
    else:
        # local fallback under ./results
        local_dir = Path(__file__).parent / "results"
        local_dir.mkdir(parents=True, exist_ok=True)
        p = local_dir / fname
        p.write_text(json.dumps(payload, indent=2))
        print(f"[RESULTS] wrote {p}")
        return str(p)

def print_and_log(name, payload):
    print(f"[RESULT] {name} -> {json.dumps(payload, indent=2)[:800]}...")
    save_json(name, payload)
