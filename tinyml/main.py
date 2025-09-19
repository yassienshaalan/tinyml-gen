# main.py — compat wrapper that preserves your old flow but uses GCS-aware loaders

import os, sys, logging
from pathlib import Path

# --- logging: screen + run.log next to main ---
LOG_PATH = Path(__file__).parent / "run.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")],
)
logging.info("Starting TinyML compat main")

# --- experiments: keep your original registry-driven runner ---
from experiments import (
    ExpCfg,
    register_dataset,        # <-- must exist in experiments.py
    available_datasets,
    run_all_experiments,     # <-- your original entrypoint
)

# --- use the new GCS-aware loaders directly ---
from data_loaders import (
    APNEA_ROOT, PTBXL_ROOT, MITDB_ROOT,
    load_apnea_ecg_loaders_impl as gcs_load_apnea,
    load_ptbxl_loaders          as gcs_load_ptbxl,
    load_mitdb_loaders          as gcs_load_mitdb,
)

# --- models: optional aliases + list available (if needed) ---
try:
    from models import MODEL_BUILDERS
    _MODEL_KEYS = list(MODEL_BUILDERS.keys())
except Exception:
    MODEL_BUILDERS, _MODEL_KEYS = {}, []

MODEL_ALIASES = {
    "regcnn":   "regular_cnn",
    "tinysep":  "tiny_separable_cnn",
    "hybrid":   "tiny_method",
    "allsynth": "tiny_method",
    # If your registry key is the class name, you can also alias to it:
    # "tiny_method": "tinymethodmodel",
}

def _resolve_model_key(name: str) -> str:
    k = name.lower().strip()
    k = MODEL_ALIASES.get(k, k)
    return "".join(ch for ch in k if ch.isalnum() or ch == "_")

# -------------------- Dataset Registry (GCS) --------------------
# We override the dataset handlers here so your run uses the new loaders.

def _apnea_gcs_wrapper(batch_size=64, length=1800, stride=None, **_):
    """Returns (tr, va, te, meta) to match your original registry contract."""
    logging.info("[apnea_ecg] root=%s", APNEA_ROOT)
    tr, va, te = gcs_load_apnea(APNEA_ROOT, batch_size=batch_size, length=length, stride=stride, verbose=True)
    meta = {'num_channels': 1, 'seq_len': length, 'num_classes': 2, 'fs': 100}
    return tr, va, te, meta

def _ptbxl_gcs_wrapper(batch_size=64, length=1800, **_):
    logging.info("[ptbxl] root=%s", PTBXL_ROOT)
    tr, va, te, meta = gcs_load_ptbxl(PTBXL_ROOT, batch_size=batch_size, length=length)
    if isinstance(meta, dict):
        meta.setdefault('num_channels', 1)
        meta.setdefault('seq_len', length)
    return tr, va, te, meta

def _mitdb_gcs_wrapper(batch_size=64, length=1800, binary=True, **_):
    logging.info("[mitdb] root=%s", MITDB_ROOT)
    tr, va, te, info = gcs_load_mitdb(MITDB_ROOT, batch_size=batch_size, length=length, binary=binary)
    meta = {'num_channels': 1, 'seq_len': length, 'num_classes': 2, 'fs': info.get('fs', 360)}
    return tr, va, te, meta

# Register (overrides any earlier registration inside experiments.py)
register_dataset('apnea_ecg', _apnea_gcs_wrapper)

# Flip these to True if you want them in this run (kept False to match your old main)
REGISTER_PTB = False
REGISTER_MIT = False

if REGISTER_PTB:
    register_dataset('ptbxl', _ptbxl_gcs_wrapper)

if REGISTER_MIT:
    register_dataset('mitdb', _mitdb_gcs_wrapper)

logging.info("[Registry] Available datasets: %s", available_datasets())

# -------------------- Config (kept close to your old main) --------------------
# NOTE: input_len=1800 matches your Apnea windows; adjust per dataset if needed.
cfg = ExpCfg(
    epochs=8, batch_size=64, lr=2e-3, device=('cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'),
    limit=None, num_workers=0, target_fs=None, length=1800, window_ms=800, input_len=1800
)

# Seed (fallback if experiments doesn’t expose seed_everything)
try:
    from experiments import seed_everything
except Exception:
    import random, numpy as np, torch
    def seed_everything(seed: int = 42):
        random.seed(seed); np.random.seed(seed)
        try:
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass

seed_everything(getattr(cfg, "seed", 42))

# -------------------- Run (like your old main) --------------------
# Start with apnea_ecg only, exactly like you had:
df = run_all_experiments(cfg, datasets=['apnea_ecg'])
print("Final thing")
print(df)
