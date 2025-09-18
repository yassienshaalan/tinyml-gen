
import os, sys, json, math, time, random, inspect
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path

from data_loaders import (
    APNEA_ROOT, PTBXL_ROOT, MITDB_ROOT, EXP_DIR,
    load_apnea_ecg_loaders_impl, load_ptbxl_loaders, load_mitdb_loaders
)
from models import MODEL_BUILDERS


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



def kd_loss(student_logits, teacher_logits, T=2.0, alpha=0.7):
    # KL(student||teacher) at temperature T
    p_t = torch.softmax(teacher_logits / T, dim=1)
    log_p_s = torch.log_softmax(student_logits / T, dim=1)
    kl = torch.sum(p_t * (torch.log(p_t + 1e-8) - log_p_s), dim=1).mean()
    return alpha * (T*T) * kl



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



def safe_build_model(spec):
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


def estimate_packed_bytes(model: nn.Module, quantized_byte_per_param: int = 1) -> int:
    """Simple packed byte estimate: 1 byte per param (INT8) + overhead"""
    params = count_params(model)
    overhead = 128  # bytes for metadata, headers etc
    return params * quantized_byte_per_param + overhead

# -------------------- Models ----------------


def packed_bytes_model_paper(model):
    """Calculate packed bytes for model (as used in run_one)"""
    total_bytes = 0
    for n, p in model.named_parameters():
        total_bytes += p.numel()
    return int(total_bytes)


def count_params(m):
    return sum(p.numel() for p in m.parameters())



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
            teacher = safe_safe_build_model("regular_cnn", in_ch, ncls).to(device)
            t_opt = torch.optim.AdamW(teacher.parameters(), lr=cfg.lr)
            t_epochs = max(3, (cfg.epochs // 2))
            for _ in range(t_epochs):
                _ = train_epoch_ce(teacher, dl_tr, t_opt, device=device, meta=meta,
                                   w_size=0.0, w_spec=0.0, w_softf1=0.0)
            print(" Teacher ready.")
        except Exception as e:
            print(f" Teacher build failed: {e} (continuing without KD)")

    # --- student ---
    model = safe_safe_build_model(model_name, in_ch, ncls, **model_kwargs).to(device)

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
        model = safe_safe_build_model(model_name, in_ch, num_classes)
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
            teacher = safe_safe_build_model("regular_cnn", in_ch, num_classes).to(DEVICE)
        except Exception as e:
            print(f"  Teacher build failed: {e} (continuing without KD)")
            teacher = None
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
        model = safe_safe_build_model(spec['model'], in_ch, num_classes).to(DEVICE)
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



def run_all_experiments(cfg: ExpCfg, datasets: List[str]=None, models: List[str]=None):
    """Run comprehensive experiments across all dataset+model combinations"""
    if datasets is None:
        datasets = available_datasets()
    if models is None:
        models = [
              'hrv_featnet',
              'cnn3_small',
              'resnet1d_small',
              'tiny_separable_cnn',
              'tiny_vae_head',
              'tiny_method',
              'regular_cnn',
          ]

    print("\n" + "="*80)
    print(" COMPREHENSIVE TINYML EXPERIMENTS")
    print("="*80)
    print(f" Datasets: {datasets}")
    print(f" Models: {models}")
    print(f"  Config: epochs={cfg.epochs}, batch_size={cfg.batch_size}, limit={cfg.limit}")
    print(f" Device: {cfg.device}")

    results = []
    total_experiments = len(datasets) * len(models)
    current_exp = 0

    for ds in datasets:
        if ds not in available_datasets():
            print(f'  Skipping unavailable dataset: {ds}')
            continue

        for model in models:
            current_exp += 1
            print(f'\n📍 Experiment {current_exp}/{total_experiments}')

            try:
                result = run_experiment(cfg, ds, model)
                if result:
                    results.append(result)
                    print(f' Completed: {ds} + {model}')
                else:
                    print(f'Failed: {ds} + {model}')
            except Exception as e:
                print(f'💥 Exception in {ds} + {model}: {e}')
                import traceback
                traceback.print_exc()

    if results:
        print(f'\n{"="*80}')
        print(" EXPERIMENT RESULTS SUMMARY")
        print("="*80)

        df = pd.DataFrame(results)
        pf = plot_pareto(df, x='flash_kb', y='test_f1_at_t')
        print("\nPARETO FRONTIER (non-dominated points):")
        print(pf[['model','flash_kb','test_f1_at_t']])

        print(f" Completed experiments: {len(results)}/{total_experiments}")
        print(f"📈 Average validation accuracy: {df['val_acc'].mean():.4f} ± {df['val_acc'].std():.4f}")
        print(f" Average model size: {df['flash_kb'].mean():.1f} KB")

        print(f"\n{'='*80}")
        print("MODEL COMPARISON")
        print("="*80)
        comparison_cols = ['dataset', 'model', 'val_acc', 'test_acc', 'flash_kb', 'params', 'train_time_s']
        print(df[comparison_cols].to_string(index=False, float_format='%.4f'))

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

        df['efficiency'] = df['val_acc'] / df['flash_kb']
        print(f"\n{'='*60}")
        print("EFFICIENCY ANALYSIS (Accuracy per KB)")
        print("="*60)
        efficiency_analysis = df.groupby('model')['efficiency'].agg(['mean', 'std']).round(6)
        print(efficiency_analysis.sort_values('mean', ascending=False))

        results_file = 'comprehensive_tinyml_results.csv'
        df.to_csv(results_file, index=False)
        print(f"\n Results saved to: {results_file}")
        return df
    else:
        print("No experiments completed successfully")
        return None




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


def available_datasets():
    return list(DATASET_REGISTRY.keys())



def register_dataset(name, loader_fn, meta=None):
    DATASET_REGISTRY[name] = { 'loader': loader_fn, 'meta': meta or {} }



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


def save_json(name, payload):
    p = EXP_DIR / f"{name}.json"
    p.write_text(json.dumps(payload, indent=2))
    return str(p)


def print_and_log(name, payload):
    print(f"[RESULT] {name} → {json.dumps(payload, indent=2)[:800]}...")
#'''
# ===================== MODEL-NAME SHIMS =====================
# (Map names used by the other suite to the classes you already have.)



def available_datasets() -> List[str]:
    return ['apnea_ecg', 'ptbxl', 'mitdb']

def make_loaders_from_legacy(ds_key: str, batch: int = 64, length: int = 1800, verbose: bool = True):
    if ds_key == 'apnea_ecg':
        tr, va, te = load_apnea_ecg_loaders_impl(APNEA_ROOT, batch_size=batch, length=length, verbose=verbose)
        meta = {'num_channels': 1, 'num_classes': 2, 'seq_len': length, 'fs': 100}
        return tr, va, te, meta
    elif ds_key == 'ptbxl':
        tr, va, te, classes = load_ptbxl_loaders(PTBXL_ROOT, batch_size=batch, length=length)
        meta = {'num_channels': 1, 'num_classes': len(classes), 'seq_len': length, 'fs': 100}
        return tr, va, te, meta
    elif ds_key == 'mitdb':
        tr, va, te = load_mitdb_loaders(MITDB_ROOT, batch_size=batch, length=length)
        meta = {'num_channels': 1, 'num_classes': 2, 'seq_len': length, 'fs': 360}
        return tr, va, te, meta
    else:
        raise ValueError(f"Unknown dataset key: {ds_key}")

def _filter_kwargs_for_ctor(ctor, kwargs):
    sig = inspect.signature(ctor)
    allowed = set(sig.parameters.keys())
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if has_varkw:
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in allowed}

def safe_build_model(model_name: str, in_ch: int, num_classes: int, **model_kwargs):
    if model_name not in MODEL_BUILDERS:
        raise KeyError(f"Model '{model_name}' is not registered. Available: {list(MODEL_BUILDERS.keys())}")
    builder = MODEL_BUILDERS[model_name]
    # we cannot always introspect builder target; rely on **kwargs forwarding and filtering below
    try:
        return builder(in_ch, num_classes, **model_kwargs)
    except TypeError as e:
        # As a fallback, drop unknown kwargs aggressively
        try:
            # Try calling with no kwargs
            return builder(in_ch, num_classes)
        except Exception:
            # If builder wraps a class, try to infer allowed keys by trial
            filtered = {}
            for k, v in list(model_kwargs.items()):
                try:
                    _ = builder(in_ch, num_classes, **filtered, **{k: v})
                    filtered[k] = v
                except TypeError:
                    pass
            return builder(in_ch, num_classes, **filtered)

def save_json(name, payload):
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    p = EXP_DIR / f"{name}.json"
    p.write_text(json.dumps(payload, indent=2))
    return str(p)

def print_and_log(name, payload):
    print(f"[RESULT] {name} -> {json.dumps(payload, indent=2)[:800]}...")



def run_suite(datasets: List[str] = None, models: List[str] = None, cfg: Optional['ExpCfg'] = None):
    datasets = datasets or available_datasets()
    models = models or list(MODEL_BUILDERS.keys())
    cfg = cfg or ExpCfg()
    for ds in datasets:
        for m in models:
            try:
                spec = {'name': f'{ds}__{m}', 'dataset': ds, 'model': m, 'lr': cfg.lr}
                run_one(spec)  # relies on make_loaders_from_legacy + safe_build_model internally
            except Exception as e:
                print(f"[ERROR] {ds} / {m}: {e}")

