# datasets.py
import os, glob, math
from pathlib import Path
from typing import Dict, Tuple, Optional
import torch
from torch.utils.data import DataLoader

# ---------------- Registry ----------------
_DATASET_REGISTRY: Dict[str, callable] = {}
_LOADER_CACHE: Dict[Tuple[str, int], tuple] = {}

def register_dataset(name: str, fn):
    _DATASET_REGISTRY[name] = fn

def available_datasets():
    return list(_DATASET_REGISTRY.keys())

def get_or_make_loaders_once(ds_key: str, batch_size: int, **kwargs):
    """
    Returns (dl_tr, dl_va, dl_te, meta). Caches by (ds_key, batch_size).
    Any kwargs are forwarded to the dataset factory.
    """
    key = (ds_key, batch_size)
    if key in _LOADER_CACHE:
        return _LOADER_CACHE[key]

    if ds_key not in _DATASET_REGISTRY:
        raise RuntimeError(f"Dataset '{ds_key}' not registered. Registered: {available_datasets()}")

    dl_tr, dl_va, dl_te, meta = _DATASET_REGISTRY[ds_key](batch_size=batch_size, **kwargs)
    _LOADER_CACHE[key] = (dl_tr, dl_va, dl_te, meta)
    return _LOADER_CACHE[key]

# ---------------- Utilities ----------------
def _require_dir(p: str):
    if not p: 
        raise RuntimeError("Dataset root path is empty / None.")
    root = Path(p)
    if not root.exists():
        raise RuntimeError(f"Dataset root does not exist: {root}")
    if not root.is_dir():
        raise RuntimeError(f"Dataset root is not a directory: {root}")
    return root

def _debug_scan_apnea(root: Path) -> Dict[str,int]:
    pats = ["a*.dat","a*.hea","a*.apn","a*.apn.txt",
            "b*.dat","b*.hea","b*.apn","b*.apn.txt",
            "c*.dat","c*.hea","c*.apn","c*.apn.txt"]
    return {pat: len(glob.glob(str(root/pat))) for pat in pats}

def _assert_apnea_usable(root: Path):
    counts = _debug_scan_apnea(root)
    # consider .apn OR .apn.txt as label files
    has_a = any(glob.glob(str(root/"a*.dat"))) and any(glob.glob(str(root/"a*.hea")))
    has_b = any(glob.glob(str(root/"b*.dat"))) and any(glob.glob(str(root/"b*.hea")))
    has_c = any(glob.glob(str(root/"c*.dat"))) and any(glob.glob(str(root/"c*.hea")))
    has_any_apn = any(glob.glob(str(root/"*.apn"))) or any(glob.glob(str(root/"*.apn.txt")))
    if not (has_a or has_b or has_c) or not has_any_apn:
        dbg = "\n".join([f"  {k:<11} -> {v}" for k,v in counts.items()])
        raise RuntimeError("No usable records (need a**, b**, c** with .apn/.dat/.hea).\nScan:\n"+dbg)

# ---------------- Adapters for your loaders ----------------
def load_apnea_ecg_loaders_impl(root: str,
                                length: int = 1800,
                                stride: Optional[int] = None,
                                batch_size: int = 64,
                                num_workers: int = 0,
                                seed: int = 42,
                                verbose: bool = True):
    """
    Adapter to your Apnea-ECG dataset code. Points to the root containing a**/b**/c** .dat/.hea and .apn(.txt).
    Replace with your real dataset builder if you prefer.
    """
    rootp = _require_dir(root)
    _assert_apnea_usable(rootp)

    # Try to call your existing function if it's importable in this environment
    # Expected signature: (root, length, stride, batch_size, num_workers, seed, verbose) -> (tr, va, te, meta) or (tr,va,te) + we add meta
    try:
        from your_existing_module import load_apnea_ecg_loaders_impl as _impl  # <- replace if needed
        out = _impl(root=str(rootp), length=length, stride=stride,
                    batch_size=batch_size, num_workers=num_workers,
                    seed=seed, verbose=verbose)
        if isinstance(out, (tuple, list)) and len(out) == 4:
            return out
        elif isinstance(out, (tuple, list)) and len(out) == 3:
            tr, va, te = out
            meta = {'num_channels': 1, 'seq_len': length, 'num_classes': 2}
            return tr, va, te, meta
        else:
            raise RuntimeError("Unexpected return from user apnea loader.")
    except Exception:
        # Fallback: simple torch.utils.data.DataLoader stubs — replace with your true dataset.
        raise RuntimeError(
            "Could not import your Apnea-ECG loader.\n"
            "Please wire your real loader by replacing the import above, or drop your loader code into this project "
            "and import it here."
        )

# ---- PTB-XL / MITDB wrappers (call your own loaders if you have them) ----
def load_ptbxl_loaders_impl(root: str,
                            batch_size: int = 64,
                            num_workers: int = 0,
                            **kwargs):
    rootp = _require_dir(root)
    try:
        from your_existing_module import run_ptbxl as _ptb  # expects a function that returns dict w/ loaders
        cfg = kwargs.copy()
        cfg.setdefault('batch_size', batch_size)
        cfg.setdefault('num_workers', num_workers)
        result = _ptb(cfg, str(rootp))
        dl_tr, dl_va, dl_te, meta = result['dl_tr'], result['dl_va'], result.get('dl_te'), result['meta']
        return dl_tr, dl_va, dl_te, meta
    except Exception:
        raise RuntimeError("PTB-XL loader not wired. Replace import above with your loader or add one.")

def load_mitdb_loaders_impl(root: str,
                            batch_size: int = 64,
                            num_workers: int = 0,
                            **kwargs):
    rootp = _require_dir(root)
    try:
        from your_existing_module import run_mitdb as _mit
        cfg = kwargs.copy()
        cfg.setdefault('batch_size', batch_size)
        cfg.setdefault('num_workers', num_workers)
        result = _mit(cfg, str(rootp))
        dl_tr, dl_va, dl_te, meta = result['dl_tr'], result['dl_va'], result.get('dl_te'), result['meta']
        return dl_tr, dl_va, dl_te, meta
    except Exception:
        raise RuntimeError("MIT-BIH loader not wired. Replace import above with your loader or add one.")

# --------------- Registration helpers ---------------
def register_apnea(root: str):
    register_dataset("apnea_ecg", lambda **kw: load_apnea_ecg_loaders_impl(root=root, **kw))

def register_ptbxl(root: str):
    register_dataset("ptbxl", lambda **kw: load_ptbxl_loaders_impl(root=root, **kw))

def register_mitdb(root: str):
    register_dataset("mitdb", lambda **kw: load_mitdb_loaders_impl(root=root, **kw))
