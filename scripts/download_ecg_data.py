#!/usr/bin/env python3
"""
Download ECG datasets from GCP bucket to local data folder.
This allows running experiments on real data instead of synthetic data.

Usage:
    python download_ecg_data.py --dataset apnea          # Download Apnea ECG only
    python download_ecg_data.py --dataset all            # Download all datasets
    python download_ecg_data.py --target-dir ./data      # Custom target directory
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path

# GCP bucket paths (from your data_loaders.py)
GCP_BASE = "gs://store-pepper/tinyml_hyper_tiny_baselines/data"
DATASETS = {
    'apnea': {
        'gcs': f"{GCP_BASE}/apnea-ecg-database-1.0.0",
        'check_file': 'a01.dat',  # File that should exist if downloaded
    },
    'ptbxl': {
        'gcs': f"{GCP_BASE}/ptbxl",
        'check_file': 'raw/ptbxl_database.csv',
    },
    'mitbih': {
        'gcs': f"{GCP_BASE}/mitbih/raw",
        'check_file': '100.dat',
    },
}


def check_gsutil():
    """Check if gsutil is installed"""
    try:
        subprocess.run(['gsutil', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def is_already_downloaded(local_path, check_file):
    """Check if dataset already exists locally"""
    check_path = Path(local_path) / check_file
    exists = check_path.exists()
    if exists:
        print(f"  ✓ Already downloaded: {check_path} exists")
    return exists


def download_dataset(dataset_name, gcs_path, local_path, check_file):
    """Download dataset from GCS to local path"""
    local_path = Path(local_path)
    
    # Check if already downloaded
    if is_already_downloaded(local_path, check_file):
        print(f"  → Skipping {dataset_name} (already downloaded)")
        return 'skipped'
    
    local_path.mkdir(parents=True, exist_ok=True)
    local_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Downloading {dataset_name}")
    print(f"  From: {gcs_path}")
    print(f"  To:   {local_path}")
    print(f"{'='*80}\n")
    
    # Use gsutil to sync (only downloads new/changed files)
    cmd = ['gsutil', '-m', 'rsync', '-r', gcs_path, str(local_path)]
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ {dataset_name} downloaded successfully to {local_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed to download {dataset_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Download ECG datasets from GCS bucket for experiments'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='all',
        choices=['apnea', 'ptbxl', 'mitbih', 'all'],
        help='Which dataset to download (default: all)'
    )
    parser.add_argument(
        '--target-dir',
        type=str,
        default='./data',
        help='Local directory to download to (default: ./data)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if data exists'
    )
    
    args = parser.parse_args()
    
    # Check prerequisites
    if not check_gsutil():
        print("ERROR: gsutil not found!")
        print("\nTo install gsutil:")
        print("  1. Install Google Cloud SDK: https://cloud.google.com/sdk/docs/install")
        print("  2. Authenticate: gcloud auth login")
        print("  3. Configure: gcloud config set project YOUR_PROJECT_ID")
        sys.exit(1)
    
    print("=" * 80)
    print("ECG DATASET DOWNLOADER")
    print("=" * 80)
    print(f"\nTarget directory: {args.target_dir}")
    print(f"Dataset(s): {args.dataset}")
    
    # Determine which datasets to download
    if args.dataset == 'all':
        datasets_to_download = list(DATASETS.keys())
    else:
        datasets_to_download = [args.dataset]
    
    print(f"Datasets to process: {', '.join(datasets_to_download)}")
    print(f"Force re-download: {args.force}")
    
    # Download each dataset
    results = {}
    for ds_name in datasets_to_download:
        ds_info = DATASETS[ds_name]
        local_path = Path(args.target_dir) / ds_name
        
        if args.force or not is_already_downloaded(local_path, ds_info['check_file']):
            status = download_dataset(ds_name.upper(), ds_info['gcs'], local_path, ds_info['check_file'])
            results[ds_name] = status if status != 'skipped' else True
        else:
            results[ds_name] = 'skipped'
    
    # Summary
    print("\n" + "=" * 80)
    print("Download Summary:")
    print("=" * 80)
    for ds_name, status in results.items():
        if status == 'skipped':
            print(f"  {ds_name.upper():<15} ✓ Already downloaded")
        elif status:
            print(f"  {ds_name.upper():<15} ✓ Downloaded successfully")
        else:
            print(f"  {ds_name.upper():<15} ✗ Download failed")
    
    success_count = sum(1 for s in results.values() if s in [True, 'skipped'])
    print(f"\n{success_count}/{len(datasets_to_download)} datasets ready")
    print("=" * 80)
    
    if success_count > 0:
        print("\nEnvironment Variables (set these before running experiments):")
        for ds_name in datasets_to_download:
            local_path = Path(args.target_dir).absolute() / ds_name
            env_var = f"{ds_name.upper()}_ROOT" if ds_name != 'mitbih' else 'MITDB_ROOT'
            print(f'export {env_var}="{local_path}"')
        
        print("\nOr add to ~/.bashrc:")
        for ds_name in datasets_to_download:
            local_path = Path(args.target_dir).absolute() / ds_name
            env_var = f"{ds_name.upper()}_ROOT" if ds_name != 'mitbih' else 'MITDB_ROOT'
            print(f'echo \'export {env_var}="{local_path}"\' >> ~/.bashrc')
        print("\nThen run experiments:")
        print("  cd tinyml")
        print("  python run_experiments.py --experiments ternary,synthesis,multi_scale")


if __name__ == '__main__':
    main()
