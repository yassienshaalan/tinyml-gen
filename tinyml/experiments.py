
import os, sys, json, math, time, random, inspect, datetime
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path

from data_loaders import (
    APNEA_ROOT, PTBXL_ROOT, MITDB_ROOT,
    load_apnea_ecg_loaders_impl, load_ptbxl_loaders, load_mitdb_loaders
)
from models import MODEL_BUILDERS

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
        tr, va, te, meta = load_mitdb_loaders(MITDB_ROOT, batch_size=batch, length=length)
        return tr, va, te, {'num_channels': 1, 'num_classes': 2, 'seq_len': length, 'fs': 360}
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
    try:
        return builder(in_ch, num_classes, **model_kwargs)
    except TypeError:
        filtered = {}
        for k, v in list(model_kwargs.items()):
            try:
                _ = builder(in_ch, num_classes, **filtered, **{k: v})
                filtered[k] = v
            except TypeError:
                pass
        return builder(in_ch, num_classes, **filtered)

def run_suite(datasets: List[str] = None, models: List[str] = None, cfg: Optional['ExpCfg'] = None):
    datasets = datasets or available_datasets()
    models = models or list(MODEL_BUILDERS.keys())
    cfg = cfg or ExpCfg()
    for ds in datasets:
        for m in models:
            try:
                spec = {'name': f'{ds}__{m}', 'dataset': ds, 'model': m, 'lr': cfg.lr}
                run_one(spec)
            except Exception as e:
                print(f"[ERROR] {ds} / {m}: {e}")

try:
    import gcsfs
except Exception:
    gcsfs = None

RUN_TS = os.environ.get("RUN_TS") or datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
RESULTS_BASE_GCS = os.environ.get("TINYML_RESULTS_GCS")  # e.g., gs://my-bucket/tinyml/results

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
        local_dir = Path(__file__).parent / "results"
        local_dir.mkdir(parents=True, exist_ok=True)
        p = local_dir / fname
        p.write_text(json.dumps(payload, indent=2))
        print(f"[RESULTS] wrote {p}")
        return str(p)

def print_and_log(name, payload):
    print(f"[RESULT] {name} -> {json.dumps(payload, indent=2)[:800]}...")
    save_json(name, payload)
