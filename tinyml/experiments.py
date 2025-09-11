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

from .training import *
from .models import *
from .data import *


# === Builder Functions Verification ===

def verify_builder_functions():
    """Verify that all required builder functions are properly defined and callable."""
    print("🔍 Verifying HyperTiny builder functions...")

    # List of all required builder functions
    required_functions = [
        'build_hypertiny_all_synth',
        'build_hypertiny_hybrid',
        'build_hypertiny_with_generator',
        'build_hypertiny_no_kd',
        'build_hypertiny_no_focal',
        'build_baseline_cnn',
        'build_tiny_method_variant'
    ]

    verification_results = {}

    for func_name in required_functions:
        if func_name in globals():
            func = globals()[func_name]
            if callable(func):
                try:
                    # Test function signature (try with different parameter sets)
                    if func_name == 'build_hypertiny_with_generator':
                        # This function requires dz, dh, r parameters
                        test_model = func(6, 16, 4, base_channels=16, num_classes=2, input_length=1000)
                    elif func_name == 'build_hypertiny_hybrid':
                        # This function has keep_first_pw parameter
                        test_model = func(keep_first_pw=True, base_channels=16, num_classes=2, input_length=1000)
                    else:
                        # Standard parameters
                        test_model = func(base_channels=16, num_classes=2, input_length=1000)

                    param_count = sum(p.numel() for p in test_model.parameters())
                    verification_results[func_name] = {
                        'status': 'OK',
                        'callable': True,
                        'test_params': param_count
                    }
                    print(f"   {func_name}: {param_count:,} parameters")

                except Exception as e:
                    verification_results[func_name] = {
                        'status': ' ERROR',
                        'callable': True,
                        'error': str(e)
                    }
                    print(f"    {func_name}: Error during test - {e}")
            else:
                verification_results[func_name] = {
                    'status': 'NOT CALLABLE',
                    'callable': False
                }
                print(f"   {func_name}: Exists but not callable")
        else:
            verification_results[func_name] = {
                'status': 'NOT FOUND',
                'callable': False
            }
            print(f"   {func_name}: Not found in globals()")

    # Summary
    total_functions = len(required_functions)
    ok_functions = sum(1 for r in verification_results.values() if r['status'] == 'OK')

    print(f"\nVerification Summary:")
    print(f"   Total functions: {total_functions}")
    print(f"   Working functions: {ok_functions}")
    print(f"   Success rate: {ok_functions/total_functions*100:.1f}%")

    if ok_functions == total_functions:
        print(" All builder functions are properly defined and working!")
    else:
        print(" Some builder functions need attention")

    return verification_results

# Run verification
builder_verification = verify_builder_functions()

# Also verify that ablation builders can access these functions
print("\n🧪 Verifying ablation builder access...")
ablation_accessible = {}

test_builders = [
    ('build_hypertiny_hybrid', 'build_hypertiny_hybrid'),
    ('build_hypertiny_all_synth', 'build_hypertiny_all_synth'),
    ('build_hypertiny_no_kd', 'build_hypertiny_no_kd'),
    ('build_hypertiny_no_focal', 'build_hypertiny_no_focal'),
    ('build_hypertiny_with_generator', 'build_hypertiny_with_generator')
]

for name, func_name in test_builders:
    if func_name in globals():
        ablation_accessible[name] = True
        print(f"   {name}: Accessible for ablation studies")
    else:
        ablation_accessible[name] = False
        print(f"   {name}: Not accessible for ablation studies")

print(f"\n🎯 Ready for ablation studies: {sum(ablation_accessible.values())}/{len(ablation_accessible)} builders accessible")
print("Builder function verification complete!")


# === TinyML V8 Experimental Suite: Comprehensive Ablation Studies ===
# This section contains the advanced experiments from TinyML_New_ExperimentsV8.ipynb

from pathlib import Path
import numpy as np
import time
import csv
import json
from typing import Dict, Tuple, Any, List
from math import ceil
import itertools

# Create output directory for experiment results - SAVE TO GOOGLE DRIVE
try:
    # Check if running in Google Colab
    if 'google.colab' in str(get_ipython()):
        # Mount Google Drive if not already mounted
        from google.colab import drive
        try:
            drive.mount('/content/drive')
            print("Google Drive mounted successfully")
        except Exception as e:
            print(f" Drive mount warning: {e}")

        # Use Google Drive path for persistent storage
        OUT_DIR = Path('/content/drive/MyDrive/TinyML_V8_Results')
        print("🔗 Using Google Drive for persistent results storage")
    else:
        # Local development - still save to a reasonable location
        OUT_DIR = Path('./TinyML_V8_Results')
        print("💻 Using local directory for results storage")

except Exception as e:
    # Fallback to basic path if detection fails
    OUT_DIR = Path('/content/TinyML_V8_Results')
    print(f" Fallback path used: {e}")

# Ensure directory exists
OUT_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED = 42
np.random.seed(RNG_SEED)

print(f"V8 Experimental Suite initialized")
print(f"Results will be saved to: {OUT_DIR}")
print(f"Full path: {OUT_DIR.absolute()}")

# === Proxy Timing Utilities ===

def _time_forward_cpu(fn, iters=1):
    """Time function execution on CPU."""
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0  # ms

def _time_forward_cuda(fn, iters=1):
    """Time function execution on GPU with CUDA events."""
    import torch
    if not torch.cuda.is_available():
        return _time_forward_cpu(fn, iters)

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    starter.record()
    for _ in range(iters):
        fn()
    ender.record()
    torch.cuda.synchronize()
    return starter.elapsed_time(ender)  # ms

def proxy_time_loader(model, loader, device, warmup_batches=3, measure_batches=20, iters_per_batch=1):
    """
    Measure proxy inference timing on host (CPU/GPU) as estimate for MCU latency.

    Returns:
        Dict with timing statistics (mean, p95, std)
    """
    model.eval()
    times = []
    n_measured = 0

    def _forward_once(xb):
        def _fn():
            with torch.no_grad():
                _ = model(xb)
        return _fn

    with torch.no_grad():
        for i, (xb, yb) in enumerate(loader):
            xb = xb.to(device, non_blocking=True)

            # Warmup phase
            if i < warmup_batches:
                if device.type == "cuda":
                    _ = _time_forward_cuda(_forward_once(xb), iters=iters_per_batch)
                else:
                    _ = _time_forward_cpu(_forward_once(xb), iters=iters_per_batch)
                continue

            # Measurement phase
            if device.type == "cuda":
                ms = _time_forward_cuda(_forward_once(xb), iters=iters_per_batch)
            else:
                ms = _time_forward_cpu(_forward_once(xb), iters=iters_per_batch)

            times.append(ms / max(1, iters_per_batch))
            n_measured += 1
            if n_measured >= measure_batches:
                break

    if len(times) == 0:
        return {"proxy_ms_mean": None, "proxy_ms_p95": None, "proxy_ms_std": None, "n_batches": 0}

    arr = np.array(times, dtype=np.float64)
    return {
        "proxy_ms_mean": float(arr.mean()),
        "proxy_ms_p95": float(np.percentile(arr, 95)),
        "proxy_ms_std": float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
        "n_batches": int(len(arr))
    }

# === Synthesis Timing Analysis ===

def synthesize_pw_layers(g_phi, Z_list, H_list):
    """Host proxy to estimate synthesis time & peak SRAM usage."""
    start = time.perf_counter()
    peak_bytes = 0
    for z, H in zip(Z_list, H_list):
        h = g_phi(z)
        # Assuming H is a matrix, h is a vector -> matrix-vector multiplication
        w = H @ h
        peak_bytes = max(peak_bytes, w.nbytes)
    synth_ms = (time.perf_counter() - start) * 1000.0
    return synth_ms, peak_bytes

def report_boot_vs_lazy(g_phi, Z_list, H_list, runs=3):
    """
    Compare boot-time vs lazy synthesis strategies.

    Boot: Synthesize all layers once at startup
    Lazy: Synthesize each layer when first needed
    """
    # Boot: all layers once
    boot_times = []
    boot_peaks = []
    for _ in range(runs):
        ms, peak = synthesize_pw_layers(g_phi, Z_list, H_list)
        boot_times.append(ms)
        boot_peaks.append(peak)

    # Lazy: synthesize each layer when first used
    lazy_times = []
    lazy_peaks = []
    for _ in range(runs):
        total_ms, peak = 0.0, 0
        for i in range(len(Z_list)):
            ms, p = synthesize_pw_layers(g_phi, [Z_list[i]], [H_list[i]])
            total_ms += ms
            peak = max(peak, p)
        lazy_times.append(total_ms)
        lazy_peaks.append(peak)

    return {
        "boot_ms_mean": float(np.mean(boot_times)),
        "lazy_ms_mean": float(np.mean(lazy_times)),
        "boot_peak_bytes": int(np.max(boot_peaks)),
        "lazy_peak_bytes": int(np.max(lazy_peaks))
    }

# === Ablation Grid Runner ===

def run_variant(train_fn, eval_fn, config: Dict[str, Any], labels=None) -> Dict[str, Any]:
    """
    Run a single experimental variant with comprehensive metrics collection.

    Args:
        train_fn: Function that trains model given config
        eval_fn: Function that evaluates model and returns (y_true, y_pred, components)
        config: Configuration dictionary
        labels: Optional class labels for per-class metrics

    Returns:
        Dictionary with all metrics including timing, accuracy, and memory
    """
    t0 = time.time()

    try:
        model, aux = train_fn(config)
        y_true, y_pred, comp = eval_fn(model, config)
        mets = ec57_metrics(y_true, y_pred, labels=labels)

        # Proxy timing if model supports it
        try:
            if hasattr(model, 'eval'):
                # Create a dummy loader for timing (use validation data if available)
                timing_config = {**config, "proxy_warmup": 2, "proxy_batches": 10, "proxy_iters": 1}
                proxy_results = {"proxy_ms_mean": None, "proxy_ms_p95": None, "proxy_ms_std": None}
                mets.update(proxy_results)
        except Exception as e:
            print(f" Proxy timing failed: {e}")
            mets.update({"proxy_ms_mean": None, "proxy_ms_p95": None, "proxy_ms_std": None})

        # Flash memory calculation
        total_bytes, br = packed_flash_bytes(comp)
        mets.update({
            "flash_kb": to_kb(total_bytes),
            "breakdown_bytes": br,
            "train_secs": round(time.time() - t0, 1),
            "variant": config.get("name", "unnamed")
        })

        return mets

    except Exception as e:
        print(f"Variant failed: {config.get('name', 'unnamed')} - {e}")
        return {
            "variant": config.get("name", "failed"),
            "error": str(e),
            "flash_kb": None,
            "acc": None,
            "train_secs": round(time.time() - t0, 1)
        }

def ablation_grid(grid_spec: Dict[str, List[Any]], base_config: Dict[str, Any],
                  train_fn, eval_fn, labels=None, out_csv: str = None):
    """
    Run comprehensive ablation study across parameter grid.

    Args:
        grid_spec: Dictionary mapping parameter names to lists of values
        base_config: Base configuration dictionary
        train_fn: Training function
        eval_fn: Evaluation function
        labels: Optional class labels
        out_csv: Optional CSV filename for results

    Returns:
        List of result dictionaries
    """
    keys = list(grid_spec.keys())
    results = []

    total_variants = np.prod([len(grid_spec[k]) for k in keys])
    print(f"🧪 Running ablation grid: {total_variants} variants")

    for i, values in enumerate(itertools.product(*[grid_spec[k] for k in keys])):
        cfg = dict(base_config)
        for k, v in zip(keys, values):
            cfg[k] = v

        # Generate descriptive variant name
        cfg["name"] = (
            f"hyb={cfg.get('keep_pw1', cfg.get('hybrid_keep', 1))}_"
            f"dz{cfg.get('dz', 6)}_dh{cfg.get('dh', 16)}_r{cfg.get('r', 4)}_"
            f"bits({cfg.get('code_bits', 6)},{cfg.get('head_bits', 6)},{cfg.get('phi_bits', 6)})_"
            f"kd{int(bool(cfg.get('use_kd', True)))}_focal{int(bool(cfg.get('use_focal', True)))}"
        )

        print(f"📋 [{i+1}/{total_variants}] Running: {cfg['name']}")
        res = run_variant(train_fn, eval_fn, cfg, labels=labels)
        results.append(res)

        # Print progress
        if res.get("acc") is not None:
            print(f"   ✓ Acc: {res['acc']:.3f}, F1: {res.get('macro_f1', 0):.3f}, "
                  f"Flash: {res.get('flash_kb', 0):.1f}KB, Time: {res.get('train_secs', 0):.1f}s")

    # Save results to CSV
    if out_csv:
        csv_path = OUT_DIR / out_csv
        fieldnames = ["variant", "flash_kb", "acc", "acc_ci", "macro_f1", "macro_f1_ci",
                     "proxy_ms_mean", "proxy_ms_p95", "proxy_ms_std", "train_secs"]

        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                row = {k: r.get(k) for k in fieldnames}
                writer.writerow(row)
        print(f"💾 Results saved to: {csv_path}")

    return results

# === Leakage-Safe Split Validation ===

def assert_no_leakage(train_ids, val_ids, test_ids):
    """Verify no overlap between train/validation/test splits."""
    a = set(train_ids) & set(val_ids)
    b = set(train_ids) & set(test_ids)
    c = set(val_ids) & set(test_ids)
    if a or b or c:
        raise ValueError(f"Data leakage detected between splits: train∩val={len(a)}, train∩test={len(b)}, val∩test={len(c)}")
    return True

# === Google Drive Connectivity Helper ===

def verify_google_drive_access():
    """Verify Google Drive is properly mounted and accessible."""
    try:
        if 'google.colab' in str(get_ipython()):
            drive_path = Path('/content/drive/MyDrive')
            if drive_path.exists():
                print("Google Drive is mounted and accessible")

                # Test write access
                test_file = drive_path / 'tinyml_test.txt'
                try:
                    test_file.write_text("TinyML V8 Test")
                    test_file.unlink()  # Delete test file
                    print("Google Drive write access confirmed")
                    return True
                except Exception as e:
                    print(f" Google Drive write test failed: {e}")
                    return False
            else:
                print("Google Drive not mounted. Please run: drive.mount('/content/drive')")
                return False
        else:
            print("💻 Running locally - using local storage")
            return True
    except Exception as e:
        print(f" Drive verification failed: {e}")
        return False

def ensure_drive_setup():
    """Ensure Google Drive is properly set up for results storage."""
    if not verify_google_drive_access():
        print("\n🔧 Setting up Google Drive access...")
        try:
            from google.colab import drive
            drive.mount('/content/drive', force_remount=True)
            if verify_google_drive_access():
                print("Google Drive setup complete!")
                return True
            else:
                print("Could not establish Google Drive access")
                return False
        except ImportError:
            print("💻 Not in Colab environment - using local storage")
            return True
        except Exception as e:
            print(f"Drive setup failed: {e}")
            return False
    return True

print("V8 Experimental Suite utilities loaded!")
print("🔧 Use ensure_drive_setup() to verify Google Drive connectivity before experiments")


# ==== Save helpers (Colab/Drive-safe) ====
import os, glob
from pathlib import Path

def _ensure_drive_mounted():
    try:
        from google.colab import drive
        if not Path("/content/drive").exists() or not list(Path("/content/drive").glob("*")):
            drive.mount('/content/drive', force_remount=False)
    except Exception:
        pass  # on non-Colab, ignore

def _results_dir(prefer="MyDrive/tinyml_hyper_tiny_baselines/results"):
    _ensure_drive_mounted()
    # 1) Try MyDrive path
    p = Path("/content/drive") / prefer
    p.mkdir(parents=True, exist_ok=True)
    if p.exists():
        return p
    # 2) Try under any Shareddrive (if user uses a Team Drive)
    sd_root = Path("/content/drive/Shareddrives")
    if sd_root.exists():
        # pick first share drive that already has the project folder, else create in first share
        matches = list(sd_root.rglob("tinyml_hyper_tiny_baselines"))
        if matches:
            p = matches[0] / "results"
            p.mkdir(parents=True, exist_ok=True)
            return p
        # fallback: create in first share drive
        shares = [d for d in sd_root.iterdir() if d.is_dir()]
        if shares:
            p = shares[0] / "tinyml_hyper_tiny_baselines" / "results"
            p.mkdir(parents=True, exist_ok=True)
            return p
    # 3) Last resort: local current dir
    p = Path("./results"); p.mkdir(parents=True, exist_ok=True); return p

def save_df_to_drive(df, filename, subdir=None):
    root = _results_dir()
    if subdir:
        root = root / subdir; root.mkdir(parents=True, exist_ok=True)
    out = root / filename
    df.to_csv(out.as_posix(), index=False)
    print(f"💾 Saved: {out.as_posix()}")
    return out



# Pre-run helper

# ==== Pre-run helper: set a global CURRENT_CFG so model builders/classes can consult it ====
def set_current_cfg(cfg):
    globals()["CURRENT_CFG"] = cfg
    return cfg

print("Pre-run helper ready: call set_current_cfg(cfg) after creating cfg.")



# Export/profile stubs

# (Optional) Hardware Export Hooks: TFLM / CMSIS-NN
# Provide stubs to export the best model and compile / profile on MCU toolchains.
# Fill these with your actual export paths & kernels.

def export_to_tflm(model, export_dir):
    os.makedirs(export_dir, exist_ok=True)
    # TODO: convert and save TFLite, int8; include cmsis-nn compatible ops
    # e.g., via tf.lite.TFLiteConverter (if using TF), or ONNX -> TFLite paths if starting in PyTorch.
    print(f"[STUB] Exported (stub) to {export_dir}")

def profile_on_mcu(binary_path):
    # TODO: integrate with your measurement harness (pin toggling, power measurement scripts)
    print(f"[STUB] Profile {binary_path} on device")

print("Export/Profile stubs ready.")

# Note:
# - Ensure any constants/configs used across modules are imported where needed.
# - You may need to move some helper functions between modules if import errors occur.
