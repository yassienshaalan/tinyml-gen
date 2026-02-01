#!/usr/bin/env python3
"""
Run ternary comparison on ALL available datasets to ensure robust results.
This prevents "cherry-picking" by testing on Apnea, PTB-XL, and MIT-BIH.
"""
import os
import sys
import subprocess
import json
from pathlib import Path

def run_experiment_on_dataset(dataset_name, env_vars):
    """Run ternary experiment with specific dataset environment"""
    print(f"\n{'='*80}")
    print(f"RUNNING TERNARY EXPERIMENT ON: {dataset_name}")
    print(f"{'='*80}\n")
    
    # Set environment variables
    env = os.environ.copy()
    env.update(env_vars)
    
    # Run experiment
    cmd = [sys.executable, 'run_rebuttal_experiments.py', '--experiments', 'ternary']
    result = subprocess.run(cmd, env=env, capture_output=False)
    
    return result.returncode == 0


def main():
    # Check what datasets are available
    data_dir = Path('../data')
    
    datasets = []
    
    # Check Apnea
    apnea_path = data_dir / 'apnea'
    if (apnea_path / 'a01.dat').exists():
        datasets.append(('Apnea-ECG', {'APNEA_ROOT': str(apnea_path.absolute())}))
        print(f"✓ Found Apnea-ECG at {apnea_path}")
    
    # Check PTB-XL
    ptbxl_path = data_dir / 'ptbxl'
    if (ptbxl_path / 'raw/ptbxl_database.csv').exists():
        datasets.append(('PTB-XL', {'PTBXL_ROOT': str(ptbxl_path.absolute())}))
        print(f"✓ Found PTB-XL at {ptbxl_path}")
    
    # Check MIT-BIH
    mitbih_path = data_dir / 'mitbih'
    if (mitbih_path / '100.dat').exists():
        datasets.append(('MIT-BIH', {'MITDB_ROOT': str(mitbih_path.absolute())}))
        print(f"✓ Found MIT-BIH at {mitbih_path}")
    
    if not datasets:
        print("\n✗ No datasets found!")
        print("  Run: python ../download_ecg_data.py --dataset all")
        sys.exit(1)
    
    print(f"\nWill test on {len(datasets)} dataset(s): {', '.join(d[0] for d in datasets)}")
    
    # Run on each dataset
    results = {}
    for ds_name, env_vars in datasets:
        success = run_experiment_on_dataset(ds_name, env_vars)
        results[ds_name] = 'SUCCESS' if success else 'FAILED'
    
    # Summary
    print(f"\n{'='*80}")
    print("MULTI-DATASET TERNARY COMPARISON SUMMARY")
    print(f"{'='*80}")
    for ds_name, status in results.items():
        symbol = '✓' if status == 'SUCCESS' else '✗'
        print(f"  {symbol} {ds_name}: {status}")
    
    # Check results
    print(f"\nResults saved in: rebuttal_results/ternary_comparison.json")
    print("Check balanced accuracy to ensure HyperTinyPW > Ternary on most datasets")
    
    success_count = sum(1 for s in results.values() if s == 'SUCCESS')
    print(f"\n{success_count}/{len(datasets)} datasets tested successfully")
    
    if success_count < len(datasets):
        sys.exit(1)


if __name__ == '__main__':
    main()
