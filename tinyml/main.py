# main.py — clean entrypoint using GCS-aware loaders + normalized gs:// roots

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
logging.info("Starting TinyML main")

# --- framework hooks from experiments (no side-effect registration here) ---
from experiments import (
    ExpCfg, seed_everything, run_all_experiments,
    register_dataset, available_datasets,
)

# --- GCS-aware loaders + URI normalizer ---
from data_loaders import (
    APNEA_ROOT as DL_APNEA_ROOT,
    PTBXL_ROOT as DL_PTBXL_ROOT,
    MITDB_ROOT as DL_MITDB_ROOT,
    load_apnea_ecg_loaders_impl as gcs_load_apnea,
    load_ptbxl_loaders          as gcs_load_ptbxl,
    load_mitdb_loaders          as gcs_load_mitdb,
    _normalize_gs_uri,
)

# --- small helper to pick env > defaults and normalize gs:// ---
def _pick_root(env_key: str, fallback: str) -> str:
    raw = os.environ.get(env_key, fallback)
    norm = _normalize_gs_uri(raw)
    return norm

# -------------------- Dataset Registry (GCS) --------------------
# We override the dataset handlers here so your run uses the new loaders.

def _apnea_gcs_wrapper(batch_size=64, length=1800, stride=None, **_):
    """Returns (tr, va, te, meta) to match your registry contract."""
    root = _pick_root("APNEA_ROOT", DL_APNEA_ROOT)
    logging.info("[apnea_ecg] root=%s", root)
    assert root.startswith("gs://") or Path(root).exists(), f"APNEA_ROOT invalid or not found: {root}"

    tr, va, te = gcs_load_apnea(root, batch_size=batch_size, length=length, stride=stride, verbose=True)
    meta = {'num_channels': 1, 'seq_len': length, 'num_classes': 2, 'fs': 100}
    return tr, va, te, meta

def _ptbxl_gcs_wrapper(batch_size=64, length=1800, **_):
    root = _pick_root("PTBXL_ROOT", DL_PTBXL_ROOT)
    logging.info("[ptbxl] root=%s", root)
    tr, va, te, meta = gcs_load_ptbxl(root, batch_size=batch_size, length=length)
    if isinstance(meta, dict):
        meta.setdefault('num_channels', 1)
        meta.setdefault('seq_len', length)
    return tr, va, te, meta

def _mitdb_gcs_wrapper(batch_size=64, length=1800, binary=True, **_):
    root = _pick_root("MITDB_ROOT", DL_MITDB_ROOT)
    logging.info("[mitdb] root=%s", root)
    tr, va, te, info = gcs_load_mitdb(root, batch_size=batch_size, length=length, binary=binary)
    meta = {'num_channels': 1, 'seq_len': length, 'num_classes': 2, 'fs': info.get('fs', 360)}
    return tr, va, te, meta

# Register (overrides anything experiments.py might have done)
register_dataset('apnea_ecg', _apnea_gcs_wrapper)

# Flip these to True if you want them in this run
REGISTER_PTB = False
REGISTER_MIT = False
if REGISTER_PTB:
    register_dataset('ptbxl', _ptbxl_gcs_wrapper)
if REGISTER_MIT:
    register_dataset('mitdb', _mitdb_gcs_wrapper)

# -------------------- Config --------------------
cfg = ExpCfg(
    epochs=8,
    batch_size=64,
    lr=2e-3,
    device=('cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'),
    limit=None,
    num_workers=0,
    target_fs=None,
    length=1800,
    window_ms=800,
    input_len=1800,   # 1800 to match Apnea windows
)

seed_everything(getattr(cfg, "seed", 42))

def main():
    # Show normalized roots up front + guard gs:// typos
    apnea_root = _pick_root("APNEA_ROOT", DL_APNEA_ROOT)
    ptbxl_root = _pick_root("PTBXL_ROOT", DL_PTBXL_ROOT)
    mitdb_root = _pick_root("MITDB_ROOT", DL_MITDB_ROOT)
    print("[Roots]", apnea_root, ptbxl_root, mitdb_root)
    #assert not apnea_root.startswith("gs:/"), f"Fix APNEA_ROOT: {apnea_root} -> must be gs://..."

    logging.info("[Registry] Available datasets: %s", available_datasets())
    # Start with apnea_ecg only, exactly like your old main
    df = run_all_experiments(cfg, datasets=['apnea_ecg'])
    print("Final thing")
    print(df)

if __name__ == "__main__":
    main()
