#!/usr/bin/env python3
"""
Download ECG datasets from GCP bucket to local data folder.
Supports two backends:
  1. gsutil  (Google Cloud SDK)   — fastest for large syncs
  2. gcsfs   (pip install gcsfs)  — pure-Python fallback

Usage:
    python download_ecg_data.py --dataset apnea          # Download Apnea ECG only
    python download_ecg_data.py --dataset all             # Download all datasets
    python download_ecg_data.py --target-dir ./data       # Custom target directory
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path

# ── GCP bucket layout ──────────────────────────────────────────────────────────
GCP_BASE = "gs://hypertinypw"

# Maps dataset key → (GCS path, *local subdir*, check file).
# Local subdirs match what data_loaders.py expects by default.
DATASETS = {
    'apnea': {
        'gcs': f"{GCP_BASE}/apnea-ecg-database-1.0.0",
        'local_dir': 'apnea-ecg-database-1.0.0',
        'check_file': 'a01.dat',
        'env_var': 'APNEA_ROOT',
    },
    'ptbxl': {
        'gcs': f"{GCP_BASE}/ptbxl",
        'local_dir': 'ptbxl',
        'check_file': 'ptbxl_database.csv',
        'env_var': 'PTBXL_ROOT',
    },
    'mitbih': {
        'gcs': f"{GCP_BASE}/mitbih",
        'local_dir': 'mitbih',
        'check_file': '100.dat',
        'env_var': 'MITDB_ROOT',
    },
}


# ── Backend helpers ────────────────────────────────────────────────────────────
def _has_gsutil():
    try:
        subprocess.run(['gsutil', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _has_gcsfs():
    try:
        import gcsfs  # noqa: F401
        return True
    except ImportError:
        return False


def _download_gsutil(gcs_path, local_path):
    """Download via gsutil -m rsync (parallel, resumable)."""
    cmd = ['gsutil', '-m', 'rsync', '-r', gcs_path, str(local_path)]
    subprocess.run(cmd, check=True)


def _download_gcsfs(gcs_path, local_path):
    """Download via gcsfs (pure-Python, no Cloud SDK needed). Uses anonymous access for public buckets."""
    import gcsfs
    # Try ADC first, fall back to anonymous for public buckets
    try:
        fs = gcsfs.GCSFileSystem()
        fs.info(gcs_path.replace("gs://", "").split("/")[0])
    except Exception:
        fs = gcsfs.GCSFileSystem(token="anon")
    # Strip gs:// for gcsfs
    bucket_path = gcs_path.replace("gs://", "")
    remote_files = fs.find(bucket_path)
    total = len(remote_files)
    print(f"  Found {total} files to download")
    for i, rpath in enumerate(remote_files, 1):
        rel = rpath[len(bucket_path):].lstrip("/")
        dst = local_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.stat().st_size > 0:
            continue
        fs.get(rpath, str(dst))
        if i % 50 == 0 or i == total:
            print(f"  [{i}/{total}] downloaded")


def download_dataset(ds_name, ds_info, data_root, *, backend, force=False):
    """Download a single dataset, returns True/False/'skipped'."""
    local_path = data_root / ds_info['local_dir']
    check = local_path / ds_info['check_file']

    if not force and check.exists():
        print(f"  [{ds_name}] Already present ({check}), skipping.")
        return 'skipped'

    local_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"  Downloading {ds_name}")
    print(f"    From: {ds_info['gcs']}")
    print(f"    To:   {local_path}")
    print(f"    Via:  {backend}")
    print(f"{'=' * 70}")

    try:
        if backend == 'gsutil':
            _download_gsutil(ds_info['gcs'], local_path)
        else:
            _download_gcsfs(ds_info['gcs'], local_path)
        print(f"  [{ds_name}] Download complete.")
        return True
    except Exception as e:
        print(f"  [{ds_name}] Download FAILED: {e}")
        return False


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Download ECG datasets from GCS bucket for experiments',
    )
    parser.add_argument(
        '--dataset', type=str, default='all',
        choices=['apnea', 'ptbxl', 'mitbih', 'all'],
        help='Which dataset to download (default: all)',
    )
    parser.add_argument(
        '--target-dir', type=str, default='./data',
        help='Local directory to download into (default: ./data)',
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Force re-download even if data exists',
    )
    args = parser.parse_args()

    # Pick backend
    if _has_gsutil():
        backend = 'gsutil'
    elif _has_gcsfs():
        backend = 'gcsfs'
    else:
        print("ERROR: Neither gsutil nor gcsfs is available.")
        print("  Option A: pip install gcsfs")
        print("  Option B: Install Google Cloud SDK (https://cloud.google.com/sdk/docs/install)")
        sys.exit(1)

    data_root = Path(args.target_dir).resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ECG DATASET DOWNLOADER")
    print("=" * 70)
    print(f"  Backend:    {backend}")
    print(f"  Target dir: {data_root}")

    datasets = list(DATASETS.keys()) if args.dataset == 'all' else [args.dataset]
    results = {}
    for name in datasets:
        results[name] = download_dataset(name, DATASETS[name], data_root,
                                         backend=backend, force=args.force)

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    for name, status in results.items():
        info = DATASETS[name]
        local = data_root / info['local_dir']
        tag = {'skipped': 'PRESENT', True: 'OK', False: 'FAILED'}.get(status, '??')
        print(f"  {name:<10} [{tag}]  {local}")

    # Environment variable hints
    ok = [n for n, s in results.items() if s in (True, 'skipped')]
    if ok:
        print(f"\nSet environment variables before running experiments:")
        for name in ok:
            info = DATASETS[name]
            local = data_root / info['local_dir']
            print(f'  export {info["env_var"]}="{local}"')
        print(f'\n  # Or set the common root (data_loaders.py resolves sub-paths):')
        print(f'  export TINYML_DATA_ROOT="{data_root}"')
        print(f"\nThen run experiments:")
        print(f"  cd tinyml && python run_experiments.py --experiments all")


if __name__ == '__main__':
    main()
