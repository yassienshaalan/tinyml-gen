# HyperTinyPW: Generative Compression for TinyML

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

Code for reproducing results from **"Once-for-All Channel Mixers (HYPERTINYPW): Generative Compression for TinyML"**  
Accepted at **MLSys 2026** (Conference on Machine Learning and Systems)

---

## 🎯 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test your setup (30 seconds)
cd tinyml
python test_rebuttal_modules.py

# 3. Run all experiments (10-15 min with existing data)
python run_rebuttal_experiments.py --experiments all

# 4. Check results
ls rebuttal_results/
```

**✅ All experiments log to files automatically!**  
**✅ Results saved to `tinyml/rebuttal_results/` with JSON outputs and detailed logs**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Running Experiments](#running-experiments)
- [Experiment Details](#experiment-details)
- [Results & Outputs](#results--outputs)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

---

## 🔬 Overview

**HyperTinyPW** achieves **maximum compression while preserving accuracy** for TinyML models:

- **12.5× compression ratio** (e.g., 903 KB → 72 KB)
- **Clinical-grade accuracy** (79.4% balanced accuracy on PTB-XL ECG)
- **Optimal balance**: 72 KB vs. ternary's 6.7 KB with 24% accuracy loss
- **Cross-domain**: Works on ECG, speech, and time-series data
- **100K-500K parameter sweet spot** with stable compression

### Key Innovation

HyperTinyPW uses a **lightweight generator** to synthesize full-precision weights at boot time, achieving optimal compression-accuracy balance:

| Method | Size | Balanced Accuracy | Status |
|--------|------|-------------------|--------|
| **Ternary (2-bit)** | 6.7 KB | 55.3% | ❌ Over-compressed |
| **HyperTinyPW** | 72 KB | 79.4% | ✅ Optimally compressed |
| **Full Precision** | 903 KB | 80.3% | ❌ Too large for MCU |

---

## 📁 Repository Structure

```
tinyml-gen/
├── tinyml/                          # 📦 Main source code
│   ├── models.py                    # HyperTinyPW architecture
│   ├── experiments.py               # Experiment framework
│   ├── datasets.py                  # ECG dataset loaders
│   ├── data_loaders.py              # Data loading utilities
│   ├── ternary_baseline.py          # Ternary quantization
│   ├── synthesis_profiler.py        # Boot-time profiling
│   ├── speech_dataset.py            # Audio dataset (KWS)
│   ├── run_rebuttal_experiments.py  # 🚀 Main runner
│   ├── test_rebuttal_modules.py     # Setup verification
│   └── main.py                      # Original experiments
│
├── scripts/                         # 🛠 Utility scripts
│   ├── download_data.sh             # Download datasets
│   ├── run_all_experiments.sh       # Full pipeline
│   ├── download_ecg_data.py         # ECG downloader
│   └── fix_dependencies.sh          # Dependency fixer
│
├── docs/                            # 📚 Documentation
│   ├── COMPLETE_EXPERIMENTAL_REPORT.md  # Technical report
│   ├── REBUTTAL_GUIDE.md                # Experiment guide
│   ├── COMPLETE_REBUTTAL_GUIDE.md       # Full guide
│   └── SETUP_LINUX.md                   # Linux setup
│
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── LICENSE                          # MIT License
```

---

## 🛠 Installation

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
python test_rebuttal_modules.py
```

**Expected Output**:
```
✅ All rebuttal modules imported successfully!
✅ Models can be instantiated
✅ Ready to run experiments
```

---

## 🚀 Running Experiments

All experiments are consolidated in **one file**: `tinyml/run_rebuttal_experiments.py`

### Option 1: Run All Experiments

```bash
cd tinyml
python run_rebuttal_experiments.py --experiments all
```

**Runs**: keyword_spotting, ternary, multi_scale, synthesis  
**Duration**: 10-60 minutes (depends on dataset availability)  
**Output**: `rebuttal_results/` directory with JSON files + logs

### Option 2: Run Specific Experiments

```bash
# Single experiment
python run_rebuttal_experiments.py --experiments ternary

# Multiple experiments (any combination)
python run_rebuttal_experiments.py --experiments ternary,multi_scale,synthesis

# Custom epochs
python run_rebuttal_experiments.py --experiments all --epochs 30
```

### Option 3: Quick Test (No Data Download)

```bash
# Uses synthetic data fallback (2-5 min)
python run_rebuttal_experiments.py --experiments synthesis,ternary
```

### Available Experiments

| Experiment | Flag | Duration | Data | Purpose |
|------------|------|----------|------|---------|
| **Keyword Spotting** | `keyword_spotting` | 30-60 min | Speech Commands | Cross-domain validation |
| **Ternary Comparison** | `ternary` | 5-10 min | ECG | Quantization trade-off |
| **Multi-Scale** | `multi_scale` | 5-10 min | ECG | Scalability (100K-500K) |
| **Synthesis Profiling** | `synthesis` | 2-5 min | Synthetic | Boot-time overhead |

**Special Flags**:
- `all` - Run all four experiments
- `--epochs N` - Training epochs (default: 20)
- `--batch-size N` - Batch size (default: 32)

### Using Complete Pipeline Script

```bash
# Run everything with one command
cd scripts
./run_all_experiments.sh
```

This script:
1. ✅ Tests setup
2. ✅ Runs all experiments
3. ✅ Generates summary
4. ✅ Shows results

---

## 📊 Experiment Details

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
python run_rebuttal_experiments.py --experiments keyword_spotting
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
python run_rebuttal_experiments.py --experiments ternary
```

**Results**:
- **HyperTinyPW**: 72 KB, 79.4% balanced accuracy
- **Ternary (2-bit)**: 6.7 KB, 55.3% balanced accuracy
- **Trade-off**: 10.8× smaller but 24% accuracy loss

**Key Finding**: Ternary over-compresses (unusable), HyperTinyPW optimally compresses

---

### 3. Multi-Scale Validation (Scalability)

**Purpose**: Validate 100K-500K parameter range  
**Dataset**: Apnea-ECG  
**Task**: Binary apnea detection

```bash
cd tinyml
python run_rebuttal_experiments.py --experiments multi_scale
```

**Configurations**:
- **Small**: 231K params → 72 KB (82% accuracy)
- **Medium**: 347K params → 109 KB (80% accuracy)
- **Large**: 489K params → 153 KB (85% accuracy)

**Key Finding**: Stable 12.5× compression + 80-85% accuracy across scales

---

### 4. Synthesis Profiling (Boot-Time)

**Purpose**: Quantify boot-time overhead  
**Dataset**: Synthetic (no download needed)

```bash
cd tinyml
python run_rebuttal_experiments.py --experiments synthesis
```

**Expected Results**:
- Synthesis time: ~12 ms (one-time)
- Inference time: ~0.8 ms/sample
- Amortization: 15 inferences

---

## 📈 Results & Outputs

### Output Structure

All results saved to `tinyml/rebuttal_results/`:

```
rebuttal_results/
├── keyword_spotting_results.json      # Audio validation
├── ternary_comparison.json            # Accuracy vs. size
├── multi_scale_validation.json        # Scalability
├── synthesis_profile.json             # Profiling
├── experiment_full.log                # Combined log
└── rebuttal_summary.json              # Summary
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
| **Compression** | 12.5× | 138× | 1× |
| **MCU Feasible** | ✅ | ✅ | ❌ |
| **Clinical Grade** | ✅ | ❌ | ✅ |

---

## 📖 Documentation

- **[docs/COMPLETE_EXPERIMENTAL_REPORT.md](docs/COMPLETE_EXPERIMENTAL_REPORT.md)** - Full technical report (50+ pages)
- **[docs/REBUTTAL_GUIDE.md](docs/REBUTTAL_GUIDE.md)** - Step-by-step experiment guide
- **[docs/SETUP_LINUX.md](docs/SETUP_LINUX.md)** - Linux-specific instructions

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Module Not Found

```bash
pip install -r requirements.txt
```

#### 2. Dataset Not Found

```bash
# Download datasets
cd scripts
./download_data.sh all

# Or set existing data paths
export PTBXL_ROOT=/path/to/ptbxl
export APNEA_ROOT=/path/to/apnea
```

#### 3. CUDA/GPU Issues

```bash
# Force CPU (TinyML targets CPU anyway)
export CUDA_VISIBLE_DEVICES=""
python run_rebuttal_experiments.py --experiments all
```

#### 4. Check Logs

```bash
tail -f tinyml/rebuttal_results/experiment_full.log
```

#### 5. Permission Errors (Linux)

```bash
chmod +x scripts/*.sh
```

---

## 🎓 Citation

```bibtex
@inproceedings{shaalan2026hypertinypw,
  title={Once-for-All Channel Mixers (HYPERTINYPW): Generative Compression for TinyML},
  author={Shaalan, Yassien and [Co-authors]},
  booktitle={Proceedings of MLSys},
  year={2026}
}
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE)

---

## 🙏 Acknowledgments

- **PTB-XL**: Wagner et al., Scientific Data, 2020
- **Apnea-ECG**: Penzel et al., Computers in Cardiology, 2000
- **Speech Commands**: Warden, 2018
- **MLSys Community**: For valuable feedback

---

## 📧 Contact

- **GitHub**: https://github.com/yassienshaalan/tinyml-gen
- **Conference**: MLSys 2026

---

**Last Updated**: February 10, 2026  
**Version**: 2.0 (Reorganized - All experiments consolidated)
