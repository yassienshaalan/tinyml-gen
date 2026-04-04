# HyperTinyPW: Generative Compression for TinyML

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

Code for reproducing results from **"Once-for-All Channel Mixers (HYPERTINYPW): Generative Compression for TinyML"**  
**Paper**: [arXiv:2603.24916](https://arxiv.org/abs/2603.24916) — final camera-ready to appear in **MLSys 2026**

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download data + run all experiments (one command)
cd scripts
./run_all_experiments.sh

# 3. Check results
ls tinyml/results/
```

Or step-by-step:

```bash
# Download ECG datasets from GCP bucket (uses gcsfs or gsutil)
python scripts/download_ecg_data.py --dataset all --target-dir ./data

# Download Speech Commands v0.02 for keyword-spotting (~2 GB)
python scripts/download_data.py --datasets speech --data-dir ./data

# Set data paths and run
export TINYML_DATA_ROOT=./data
export SPEECH_COMMANDS_ROOT=./data/speech_commands_v0.02
cd tinyml
python run_experiments.py --experiments all --epochs 20
```

All experiments log to files automatically.  
Results saved to `tinyml/results/` with JSON outputs and detailed logs.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Running Experiments](#running-experiments)
- [Experiment Details](#experiment-details)
- [Results & Outputs](#results--outputs)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

---

## Overview

**HyperTinyPW** achieves **maximum compression while preserving accuracy** for TinyML models:

- **12.5x compression ratio** (e.g., 903 KB → 72 KB)
- **Clinical-grade accuracy** (79.4% balanced accuracy on PTB-XL ECG)
- **Optimal balance**: 72 KB vs. ternary's 6.7 KB with 24% accuracy loss
- **Cross-domain**: Works on ECG, speech, and time-series data
- **100K-500K parameter sweet spot** with stable compression

### Key Innovation

HyperTinyPW uses a **lightweight generator** to synthesize full-precision weights at boot time, achieving optimal compression-accuracy balance:

| Method | Size | Balanced Accuracy | Status |
|--------|------|-------------------|--------|
| **Ternary (2-bit)** | 6.7 KB | 55.3% | Over-compressed |
| **HyperTinyPW** | 72 KB | 79.4% | Optimally compressed |
| **Full Precision** | 903 KB | 80.3% | Too large for MCU |

---

## Repository Structure

```
tinyml-gen/
├── tinyml/                          # Main source code
│   ├── models.py                    # HyperTinyPW architecture
│   ├── experiments.py               # Experiment framework
│   ├── datasets.py                  # ECG dataset loaders
│   ├── data_loaders.py              # Data loading utilities
│   ├── ternary_baseline.py          # Ternary quantization
│   ├── synthesis_profiler.py        # Boot-time profiling
│   ├── speech_dataset.py            # Audio dataset (KWS)
│   ├── run_experiments.py           # Main experiment runner
│   ├── test_experiments.py          # Comprehensive unit tests
│   └── main.py                      # Original experiments
│
├── scripts/                         # Utility scripts
│   ├── download_data.sh             # Download datasets
│   ├── run_all_experiments.sh       # Full pipeline
│   ├── download_ecg_data.py         # ECG downloader
│   └── fix_dependencies.sh          # Dependency fixer
│
├── docs/                            # Documentation
│   ├── COMPLETE_EXPERIMENTAL_REPORT.md  # Technical report
│   └── SETUP_LINUX.md                   # Linux setup
│
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── LICENSE                          # MIT License
```
[Data on GCP Bucket](https://console.cloud.google.com/storage/browser/hypertinypw)
---

## Installation

### Prerequisites

- **Python**: 3.8+
- **PyTorch**: 1.9+
- **pip**: Latest version

### Step 1: Clone Repository

```bash
git clone https://github.com/yassienshaalan/tinyml-gen.git
cd tinyml-gen
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Core Dependencies**:
- `torch` - Deep learning framework
- `numpy`, `pandas` - Data handling
- `scikit-learn` - Metrics
- `wfdb` - ECG data
- `gcsfs` - Cloud storage (optional)

### Step 3: Verify Installation

```bash
cd tinyml

# Run comprehensive unit tests
python test_experiments.py

# Or run quick VM tests (fastest)
python ../scripts/quick_test.py
```

**Expected Output**:
```
[OK] All modules imported successfully!
[OK] Models can be instantiated
[OK] Ready to run experiments
```

---

## Running Tests

### Quick VM Tests (Recommended First)

Fast validation without training - tests all code paths in under 30 seconds:

```bash
# Option 1: Python script (works on Windows/Linux)
cd scripts
python quick_test.py

# Option 2: Bash script (Linux/Mac)
./test_vm.sh
```

**Tests performed**:
1. Module imports
2. Model instantiation and forward/backward pass
3. Ternary quantization
4. Synthesis profiling
5. Data generation
6. Experiment configuration

### Comprehensive Unit Tests

Full test suite covering all functionality:

```bash
cd tinyml

# Run all tests
python test_experiments.py

# Run only quick tests
python test_experiments.py --quick
```

**Test coverage**:
- Module imports (6 modules)
- Model architectures (HyperTinyPW, ternary)
- Data loading (synthetic, real datasets)
- Experiment framework (training, evaluation)
- Synthesis profiling
- Utility functions

---

## Running Experiments

All experiments are consolidated in **one file**: `tinyml/run_experiments.py`

### Option 1: Run All Experiments

```bash
cd tinyml
python run_experiments.py --experiments all
```

**Runs**: keyword_spotting, ternary, multi_scale, synthesis, 8bit, kws_perclass  
**Output**: `results/` directory with JSON files + logs

### Option 2: Run Specific Experiments

```bash
# Single experiment
python run_experiments.py --experiments ternary

# Multiple experiments (any combination)
python run_experiments.py --experiments ternary,multi_scale,synthesis

# Custom epochs
python run_experiments.py --experiments all --epochs 30
```

### Option 3: Quick Test (No Data Download)

```bash
# Uses synthetic data fallback
python run_experiments.py --experiments synthesis,ternary
```

### Option 4: Run Unit Tests

```bash
# Test all code paths and modules
python test_experiments.py

# Quick VM tests (fast, no training)
python test_experiments.py --quick
```

### Available Experiments

| Experiment | Flag | Data | Purpose |
|------------|------|------|---------|
| **Keyword Spotting** | `keyword_spotting` | Speech Commands | Cross-domain validation |
| **Ternary Comparison** | `ternary` | ECG | Quantization trade-off |
| **8-bit Quantization** | `8bit` | ECG | INT8 baseline |
| **Multi-Scale** | `multi_scale` | ECG | Scalability (100K-500K) |
| **Synthesis Profiling** | `synthesis` | Synthetic | Boot-time overhead |
| **KWS Per-Class** | `kws_perclass` | Speech | Class balance analysis |

**Special Flags**:
- `all` - Run all experiments
- `--epochs N` - Training epochs (default: 20)
- `--batch-size N` - Batch size (default: 32/64)

### Using Complete Pipeline Script

```bash
# Run everything with one command (downloads data + runs all experiments)
cd scripts
./run_all_experiments.sh

# Or customize
./run_all_experiments.sh --experiments ternary,synthesis --epochs 10
./run_all_experiments.sh --skip-download --cpu          # data already local, CPU only
./run_all_experiments.sh --help                         # see all options
```

This script:
1. Downloads ECG data from GCP + Speech Commands via wget
2. Sets environment variables
3. Verifies setup (unit tests)
4. Runs selected experiments
5. Prints results summary

---

## Experiment Details

### 1. Keyword Spotting (Cross-Domain)

**Purpose**: Prove HyperTinyPW works beyond ECG  
**Dataset**: Google Speech Commands v0.02  
**Task**: 12-class audio classification

```bash
# Download dataset (~2GB)
cd scripts
./download_data.sh speech_commands

# Run experiment
cd ../tinyml
python run_experiments.py --experiments keyword_spotting
```

**Expected Results**:
- Test Accuracy: **96.2%**
- Model: 234K params → 73 KB compressed
- Proves: Cross-domain applicability

---

### 2. Ternary Comparison (Accuracy vs. Size)

**Purpose**: Compare against extreme quantization  
**Dataset**: PTB-XL ECG (21,799 records)  
**Task**: Binary ECG classification

```bash
cd tinyml
python run_experiments.py --experiments ternary
```

**Results**:
- **HyperTinyPW**: 72 KB, 79.4% balanced accuracy
- **Ternary (2-bit)**: 6.7 KB, 55.3% balanced accuracy
- **Trade-off**: 10.8x smaller but 24% accuracy loss

**Key Finding**: Ternary over-compresses (unusable), HyperTinyPW optimally compresses

---

### 3. Multi-Scale Validation (Scalability)

**Purpose**: Validate 100K-500K parameter range  
**Dataset**: Apnea-ECG  
**Task**: Binary apnea detection

```bash
cd tinyml
python run_experiments.py --experiments multi_scale
```

**Configurations**:
- **Small**: 231K params → 72 KB (82% accuracy)
- **Medium**: 347K params → 109 KB (80% accuracy)
- **Large**: 489K params → 153 KB (85% accuracy)

**Key Finding**: Stable 12.5x compression + 80-85% accuracy across scales

---

### 4. Synthesis Profiling (Boot-Time)

**Purpose**: Quantify boot-time overhead  
**Dataset**: Synthetic (no download needed)

```bash
cd tinyml
python run_experiments.py --experiments synthesis
```

**Expected Results**:
- Synthesis time: ~12 ms (one-time)
- Inference time: ~0.8 ms/sample
- Amortization: 15 inferences

---

## Results & Outputs

### Output Structure

All results saved to `tinyml/results/`:

```
results/
├── keyword_spotting_results.json      # Audio validation
├── ternary_comparison.json            # Accuracy vs. size
├── multi_scale_validation.json        # Scalability
├── synthesis_profile.json             # Profiling
├── 8bit_quantization_ptbxl.json       # INT8 baseline
├── kws_perclass_analysis.json         # Per-class analysis
├── experiment_full.log                # Combined log
└── experiment_summary.json            # Summary
```

### Example Output (`ternary_comparison.json`)

```json
{
  "hypertiny": {
    "size_kb": 72.29,
    "test_acc": 78.75,
    "balanced_acc": 79.36
  },
  "ternary": {
    "size_kb": 6.70,
    "test_acc": 60.96,
    "balanced_acc": 55.32
  },
  "comparison": {
    "accuracy_advantage": 24.04,
    "size_ratio": 10.8,
    "trade_off": "72 KB (clinical-grade) vs 6.7 KB (over-compressed)"
  }
}
```

### Quick Results Summary

| Metric | HyperTinyPW | Ternary | Full Precision |
|--------|-------------|---------|----------------|
| **PTB-XL Accuracy** | 79.4% | 55.3% | 80.3% |
| **Compressed Size** | 72 KB | 6.7 KB | 903 KB |
| **Compression** | 12.5x | 138x | 1x |
| **MCU Feasible** | Yes | Yes | No |
| **Clinical Grade** | Yes | No | Yes |

---

## Documentation

- **[docs/COMPLETE_EXPERIMENTAL_REPORT.md](docs/COMPLETE_EXPERIMENTAL_REPORT.md)** - Full technical report
- **[docs/SETUP_LINUX.md](docs/SETUP_LINUX.md)** - Linux-specific instructions

---

## Troubleshooting

### Common Issues

#### 1. Module Not Found

```bash
pip install -r requirements.txt
```

#### 2. Dataset Not Found

```bash
# Download ECG datasets from GCP bucket
python scripts/download_ecg_data.py --dataset all --target-dir ./data

# Download Speech Commands for keyword spotting
python scripts/download_data.py --datasets speech --data-dir ./data

# Set data root (sub-paths resolved automatically)
export TINYML_DATA_ROOT=./data
export SPEECH_COMMANDS_ROOT=./data/speech_commands_v0.02
```

#### 3. CUDA/GPU Issues

```bash
# Force CPU (TinyML targets CPU anyway)
export CUDA_VISIBLE_DEVICES=""
python run_experiments.py --experiments all
```

#### 4. Check Logs

```bash
tail -f tinyml/results/experiment_full.log
```

#### 5. Permission Errors (Linux)

```bash
chmod +x scripts/*.sh
```

---

## Quick Tests for Your VM

Here are actual fast tests you can run on your VM to validate everything works:

### Test 1: Quick Python Test (30 seconds)

```bash
cd scripts
python quick_test.py
```

This tests:
- All module imports
- Model creation and forward/backward pass
- Quantization
- Profiling
- Data generation

### Test 2: Individual Component Tests

```bash
cd tinyml

# Test model instantiation
python -c "from models import HyperTinyPW; import torch; m = HyperTinyPW(2,1,8,2,32,100); print('Model OK')"

# Test quantization
python -c "from ternary_baseline import TernaryQuantizer; import torch; q = TernaryQuantizer(); print('Quantization OK')"

# Test data generation
python -c "from datasets import create_synthetic_ecg_data; X,y = create_synthetic_ecg_data(50, 100); print('Data OK')"
```

### Test 3: Run Synthesis Experiment (Fast)

```bash
cd tinyml
python run_experiments.py --experiments synthesis
```

This runs the full synthesis profiling experiment with real model - takes about 1-2 minutes.

### Test 4: Unit Tests

```bash
cd tinyml

# Quick tests only (imports + basic functionality)
python test_experiments.py --quick

# All tests (comprehensive coverage)
python test_experiments.py
```

### Expected Results

All tests should show `[OK]` for each component. If you see `[FAIL]`, check the error message for missing dependencies or import issues.

---

## Citation

**Preprint**: [arXiv:2603.24916](https://arxiv.org/abs/2603.24916)  
**Camera-ready**: to appear in MLSys 2026

```bibtex
@inproceedings{shaalan2026hypertinypw,
  title={Once-for-All Channel Mixers (HYPERTINYPW): Generative Compression for TinyML},
  author={Shaalan, Yassien},
  booktitle={Proceedings of MLSys 2026},
  year={2026}
}
```

---

## License

Apache License - see [LICENSE](LICENSE)

---

## Acknowledgments

- **PTB-XL**: Wagner et al., Scientific Data, 2020
- **Apnea-ECG**: Penzel et al., Computers in Cardiology, 2000
- **Speech Commands**: Warden, 2018
