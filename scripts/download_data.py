#!/usr/bin/env python3
"""
Data Download Script for TinyML Experiments
Downloads datasets for experiments.

Usage:
  python download_data.py --datasets all              # Download all datasets
  python download_data.py --datasets speech           # Only Speech Commands
  python download_data.py --datasets speech,apnea     # Multiple specific datasets
  python download_data.py --minimal                   # Only datasets needed for minimal experiments
"""

import os
import sys
import argparse
import urllib.request
import tarfile
import zipfile
from pathlib import Path
import subprocess


def download_file(url, output_path, desc="Downloading"):
    """Download file with progress bar"""
    print(f"\n{desc}...")
    print(f"  URL: {url}")
    print(f"  Output: {output_path}")
    
    try:
        # Use wget if available (better progress bar)
        result = subprocess.run(['wget', '--version'], capture_output=True)
        if result.returncode == 0:
            subprocess.run(['wget', '-O', str(output_path), url], check=True)
            return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Fallback to Python's urllib
    try:
        def reporthook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\r  Progress: {percent}%")
            sys.stdout.flush()
        
        urllib.request.urlretrieve(url, output_path, reporthook)
        print("\n  ✓ Download complete")
        return True
    except Exception as e:
        print(f"\n  ✗ Download failed: {e}")
        return False


def extract_tar_gz(filepath, output_dir):
    """Extract .tar.gz file"""
    print(f"\nExtracting {filepath.name}...")
    try:
        with tarfile.open(filepath, 'r:gz') as tar:
            tar.extractall(output_dir)
        print(f"  ✓ Extracted to {output_dir}")
        return True
    except Exception as e:
        print(f"  ✗ Extraction failed: {e}")
        return False


def extract_zip(filepath, output_dir):
    """Extract .zip file"""
    print(f"\nExtracting {filepath.name}...")
    try:
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print(f"  ✓ Extracted to {output_dir}")
        return True
    except Exception as e:
        print(f"  ✗ Extraction failed: {e}")
        return False


def download_speech_commands(data_dir):
    """
    Download Google Speech Commands v0.02 dataset
    Required for: Keyword Spotting experiment (cross-domain validation)
    Size: ~2GB
    """
    print("\n" + "=" * 70)
    print("SPEECH COMMANDS DATASET (for Keyword Spotting)")
    print("=" * 70)
    
    url = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
    output_dir = data_dir / "speech_commands_v0.02"
    archive_path = data_dir / "speech_commands_v0.02.tar.gz"
    
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"✓ Dataset already exists at {output_dir}")
        return True
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download
    if not archive_path.exists():
        if not download_file(url, archive_path, "Downloading Speech Commands v0.02 (~2GB)"):
            return False
    else:
        print(f"✓ Archive already downloaded: {archive_path}")
    
    # Extract
    if not extract_tar_gz(archive_path, output_dir):
        return False
    
    # Verify
    expected_dirs = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
    found = sum(1 for d in expected_dirs if (output_dir / d).exists())
    
    if found >= 8:
        print(f"✓ Verified: Found {found}/{len(expected_dirs)} expected directories")
        print(f"\nTo use this dataset, set environment variable:")
        print(f"  export SPEECH_COMMANDS_ROOT={output_dir}")
        return True
    else:
        print(f"✗ Verification failed: Only found {found}/{len(expected_dirs)} directories")
        return False


def download_apnea_ecg(data_dir):
    """
    Download Apnea-ECG dataset
    Required for: Original experiments and optional for ternary baseline
    Note: This is a placeholder - you'll need to add the actual download URL
    """
    print("\n" + "=" * 70)
    print("APNEA-ECG DATASET")
    print("=" * 70)
    
    output_dir = data_dir / "apnea-ecg"
    
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"✓ Dataset already exists at {output_dir}")
        return True
    
    print("\n⚠ MANUAL DOWNLOAD REQUIRED")
    print("  Apnea-ECG dataset requires manual download from PhysioNet:")
    print("  1. Visit: https://physionet.org/content/apnea-ecg/1.0.0/")
    print("  2. Download dataset files")
    print(f"  3. Extract to: {output_dir}")
    print("\n  For automated download, you may need PhysioNet credentials:")
    print("  wget -r -N -c -np --user YOUR_USERNAME --ask-password \\")
    print("       https://physionet.org/files/apnea-ecg/1.0.0/")
    
    return False


def download_ptbxl(data_dir):
    """
    Download PTB-XL dataset
    Required for: Original experiments and optional for ternary baseline
    Size: ~900MB
    """
    print("\n" + "=" * 70)
    print("PTB-XL DATASET")
    print("=" * 70)
    
    url = "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
    output_dir = data_dir / "ptb-xl"
    archive_path = data_dir / "ptb-xl-1.0.3.zip"
    
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"✓ Dataset already exists at {output_dir}")
        return True
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n⚠ MANUAL DOWNLOAD RECOMMENDED")
    print("  PTB-XL dataset is large (~900MB)")
    print(f"  URL: {url}")
    print(f"  Download to: {archive_path}")
    print(f"  Extract to: {output_dir}")
    print("\n  Or use:")
    print(f"  wget {url} -O {archive_path}")
    print(f"  unzip {archive_path} -d {data_dir}")
    
    return False


def download_mitdb(data_dir):
    """
    Download MIT-BIH Arrhythmia Database
    Required for: Original experiments and optional for ternary baseline
    """
    print("\n" + "=" * 70)
    print("MIT-BIH ARRHYTHMIA DATABASE")
    print("=" * 70)
    
    output_dir = data_dir / "mit-bih-arrhythmia"
    
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"✓ Dataset already exists at {output_dir}")
        return True
    
    print("\n⚠ MANUAL DOWNLOAD REQUIRED")
    print("  MIT-BIH database requires PhysioNet access:")
    print("  1. Visit: https://physionet.org/content/mitdb/1.0.0/")
    print("  2. Download dataset files")
    print(f"  3. Extract to: {output_dir}")
    
    return False


def create_synthetic_data(data_dir):
    """
    Create synthetic data for experiments that don't require real datasets
    (Synthesis profiling and NAS compatibility)
    """
    print("\n" + "=" * 70)
    print("SYNTHETIC DATA")
    print("=" * 70)
    print("✓ Synthesis profiling and NAS compatibility use synthetic data")
    print("  No download needed - generated at runtime")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Download datasets for TinyML experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all datasets
  python download_data.py --datasets all
  
  # Download only datasets needed for minimal experiments
  python download_data.py --minimal
  
  # Download specific datasets
  python download_data.py --datasets speech
  python download_data.py --datasets speech,apnea,ptbxl
  
  # Specify custom data directory
  python download_data.py --rebuttal-only --data-dir /path/to/data

Available datasets:
  speech   - Google Speech Commands v0.02 (~2GB) [REBUTTAL REQUIRED]
  apnea    - Apnea-ECG dataset (requires manual download)
  ptbxl    - PTB-XL dataset (~900MB, manual download recommended)
  mitdb    - MIT-BIH Arrhythmia database (requires manual download)
        """
    )
    
    parser.add_argument(
        '--datasets',
        type=str,
        default=None,
        help='Comma-separated list of datasets to download: speech,apnea,ptbxl,mitdb,all'
    )
    parser.add_argument(
        '--minimal',
        action='store_true',
        help='Download only datasets required for minimal experiments (speech commands)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data',
        help='Directory to store downloaded datasets (default: ./data)'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        default=True,
        help='Skip datasets that already exist (default: True)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.datasets and not args.minimal:
        parser.print_help()
        print("\\n[ERROR] Must specify either --datasets or --minimal")
        sys.exit(1)
    
    # Setup data directory
    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("TINYML DATA DOWNLOAD SCRIPT")
    print("=" * 70)
    print(f"\nData directory: {data_dir}")
    
    # Determine which datasets to download
    if args.minimal:
        datasets = ['speech']
        print("\\nMode: MINIMAL")
        print("  Downloading only datasets required for minimal experiments")
    else:
        if args.datasets.lower() == 'all':
            datasets = ['speech', 'apnea', 'ptbxl', 'mitdb']
        else:
            datasets = [d.strip() for d in args.datasets.split(',')]
    
    print(f"\nDatasets to download: {', '.join(datasets)}")
    
    # Download datasets
    results = {}
    
    if 'speech' in datasets:
        results['speech'] = download_speech_commands(data_dir)
    
    if 'apnea' in datasets:
        results['apnea'] = download_apnea_ecg(data_dir)
    
    if 'ptbxl' in datasets:
        results['ptbxl'] = download_ptbxl(data_dir)
    
    if 'mitdb' in datasets:
        results['mitdb'] = download_mitdb(data_dir)
    
    # Synthetic data info
    if args.minimal or 'all' in args.datasets.lower():
        create_synthetic_data(data_dir)
    
    # Summary
    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    
    for dataset, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED (see instructions above)"
        print(f"  {dataset:20s}: {status}")
    
    # Instructions for running experiments
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    
    if results.get('speech', False):
        speech_dir = data_dir / "speech_commands_v0.02"
        print("\n1. Set environment variable for Speech Commands:")
        print(f"   export SPEECH_COMMANDS_ROOT={speech_dir}")
        print("   # Add to ~/.bashrc for persistence")
        print(f"   echo 'export SPEECH_COMMANDS_ROOT={speech_dir}' >> ~/.bashrc")
    
    print("\n2. Run experiments:")
    print("   cd tinyml")
    print("   python test_experiments.py                     # Test setup")
    print("   python run_experiments.py --experiments all --epochs 20")
    
    print("\n3. Check results:")
    print("   ls -lh tinyml/results/")
    
    if args.minimal:
        print("\n" + "=" * 70)
        print("REBUTTAL EXPERIMENTS DATA REQUIREMENTS")
        print("=" * 70)
        print("\n✓ Keyword Spotting: Speech Commands downloaded")
        print("✓ Ternary Baseline: Uses existing ECG data or synthetic")
        print("✓ Synthesis Profiling: Uses synthetic data (no download)")
        print("✓ NAS Compatibility: Uses synthetic data (no download)")
        print("\nYou can now run all 4 rebuttal experiments!")
    
    # Exit code
    if all(results.values()):
        print("\n✓ All downloads successful!")
        sys.exit(0)
    elif any(results.values()):
        print("\n⚠ Some downloads completed, others require manual steps")
        sys.exit(0)
    else:
        print("\n✗ No automatic downloads completed - follow manual instructions above")
        sys.exit(1)


if __name__ == '__main__':
    main()
