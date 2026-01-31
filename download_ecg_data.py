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
    'apnea': f"{GCP_BASE}/apnea-ecg-database-1.0.0",
    'ptbxl': f"{GCP_BASE}/ptbxl",
    'mitdb': f"{GCP_BASE}/mitbih/raw",
}


def check_gsutil():
    """Check if gsutil is installed"""
    try:
        subprocess.run(['gsutil', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def download_dataset(dataset_name, gcs_path, local_path):
    """Download dataset from GCS to local path"""
    local_path = Path(local_path)
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
        description='Download ECG datasets from GCP bucket for rebuttal experiments'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='apnea',
        choices=['apnea', 'ptbxl', 'mitdb', 'all'],
        help='Which dataset to download (default: apnea)'
    )
    parser.add_argument(
        '--target-dir',
        type=str,
        default='./data',
        help='Local directory to download to (default: ./data)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be downloaded without actually downloading'
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
    
    if args.dry_run:
        print("\n[DRY RUN MODE - No files will be downloaded]")
    
    # Determine which datasets to download
    if args.dataset == 'all':
        datasets_to_download = DATASETS.items()
    else:
        datasets_to_download = [(args.dataset, DATASETS[args.dataset])]
    
    # Download each dataset
    success_count = 0
    for name, gcs_path in datasets_to_download:
        local_path = Path(args.target_dir) / name
        
        if args.dry_run:
            print(f"\nWould download:")
            print(f"  {name}: {gcs_path} -> {local_path}")
        else:
            if download_dataset(name, gcs_path, local_path):
                success_count += 1
    
    # Summary
    print("\n" + "=" * 80)
    if args.dry_run:
        print("DRY RUN COMPLETE")
    else:
        print("DOWNLOAD COMPLETE")
        print(f"  {success_count}/{len(list(datasets_to_download))} dataset(s) downloaded successfully")
    print("=" * 80)
    
    if not args.dry_run and success_count > 0:
        print("\nTo use the downloaded data:")
        print(f"  export APNEA_ROOT={Path(args.target_dir).absolute() / 'apnea'}")
        print(f"  export PTBXL_ROOT={Path(args.target_dir).absolute() / 'ptbxl'}")
        print(f"  export MITDB_ROOT={Path(args.target_dir).absolute() / 'mitdb'}")
        print("\nOr on Windows:")
        print(f"  set APNEA_ROOT={Path(args.target_dir).absolute() / 'apnea'}")
        print(f"  set PTBXL_ROOT={Path(args.target_dir).absolute() / 'ptbxl'}")
        print(f"  set MITDB_ROOT={Path(args.target_dir).absolute() / 'mitdb'}")
        print("\nThen run experiments:")
        print("  cd tinyml")
        print("  python run_rebuttal_experiments.py --experiments ternary,synthesis,multi_scale")


if __name__ == '__main__':
    main()
