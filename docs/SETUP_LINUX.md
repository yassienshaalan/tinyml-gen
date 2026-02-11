# Setup Guide for Linux Machine

Complete setup instructions for running experiments on a fresh Linux machine.

## 1. System Requirements

- **OS**: Ubuntu 18.04+ (or similar Linux distribution)
- **Python**: 3.7 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Disk Space**: 5GB (2GB for Speech Commands + 3GB for dependencies)
- **Internet**: Required for downloading data

## 2. Initial Setup

### Clone Repository
```bash
git clone https://github.com/yassienshaalan/tinyml-gen.git
cd tinyml-gen
```

### Install System Dependencies (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install -y python3 python3-pip wget curl
```

### Install Python Dependencies
```bash
pip3 install -r requirements.txt
```

**Required packages** (from requirements.txt):
```
torch>=2.2
torchaudio>=2.2
numpy
scikit-learn
pandas>=2.0.0,<3.0.0
tqdm
pydrive2
wfdb>=4.1.0  # For ECG data loading
scipy
soundfile>=0.12.0  # For audio loading
```

If requirements.txt is missing or you get compatibility errors, install manually:
```bash
pip3 install torch>=2.2 torchaudio>=2.2 numpy scikit-learn pandas tqdm wfdb>=4.1.0 scipy soundfile>=0.12.0
```

**Common Issue - pandas/wfdb Compatibility:**
If you see `TypeError: unhashable type: 'StringArray'`, update wfdb:
```bash
pip3 install --upgrade "wfdb>=4.1.0"
# Or use the fix script:
bash fix_dependencies.sh
```

**Common Issue - Audio Loading (torchcodec):**
If you see `ModuleNotFoundError: No module named 'torchcodec'`, install soundfile:
```bash
pip3 install --upgrade "soundfile>=0.12.0"
# Or use the fix script:
bash fix_dependencies.sh
```

## 3. Download Data

### Option A: Download Only Minimal Data (Recommended)
Downloads only Speech Commands dataset (~2GB) needed for keyword spotting.

```bash
# Make download script executable
chmod +x download_data.py

# Download minimal datasets
python3 download_data.py --minimal
```

**What this downloads:**
- Speech Commands v0.02 (~2GB) - For keyword spotting experiment
- Note: Other experiments use synthetic data or existing ECG data

### Option B: Download All Datasets
Downloads all datasets including ECG datasets for original experiments.

```bash
python3 download_data.py --datasets all
```

**Note**: ECG datasets (apnea, ptbxl, mitdb) require manual download from PhysioNet. The script will show instructions.

### Option C: Download Specific Datasets
```bash
# Only Speech Commands
python3 download_data.py --datasets speech

# Speech + specific ECG datasets
python3 download_data.py --datasets speech,apnea
```

## 4. Set Environment Variables

After downloading Speech Commands, set the environment variable:

```bash
# For current session
export SPEECH_COMMANDS_ROOT=/path/to/tinyml-gen/data/speech_commands_v0.02

# Make permanent (add to ~/.bashrc)
echo "export SPEECH_COMMANDS_ROOT=$(pwd)/data/speech_commands_v0.02" >> ~/.bashrc
source ~/.bashrc
```

Verify:
```bash
echo $SPEECH_COMMANDS_ROOT
ls $SPEECH_COMMANDS_ROOT  # Should show directories: yes, no, up, down, etc.
```

## 5. Test Setup

```bash
cd tinyml
python3 test_experiments.py
```

**Expected output:**
```
Testing speech_dataset module...
Testing ternary_baseline module...
Testing synthesis_profiler module...
Testing nas_compatibility module...

All tests passed! Ready to run experiments.
```

## 6. Run Experiments

### Quick Run (Without Keyword Spotting, 5-10 min)
For fast testing without downloading data:

```bash
python3 run_experiments.py --experiments ternary,synthesis,nas
```

### Full Run (All 4 Experiments, ~60 min)
Includes keyword spotting with Speech Commands:

```bash
python3 run_experiments.py --experiments all --epochs 20
```

### Custom Options
```bash
# Smaller batch size for limited RAM
python3 run_experiments.py --experiments all --batch-size 32 --epochs 10

# Force CPU (if no GPU)
python3 run_experiments.py --experiments all --cpu

# Custom output directory
python3 run_experiments.py --experiments all --output-dir /path/to/results
```

## 7. Check Results

All results are saved in `tinyml/results/`:

```bash
cd tinyml/results

# List all output files
ls -lh

# View summary
cat experiment_summary.json

# View specific results
cat ternary_comparison.json
cat synthesis_profile.json
cat nas_compatibility.json
cat keyword_spotting_results.json  # If you ran keyword spotting

# View complete log
less experiment_full.log
```

## 8. Output Files Explained

```
results/
├── experiment_full.log              # Complete console output
├── experiment_summary.json          # Summary of all experiments
├── keyword_spotting_results.json    # Speech experiment results
├── keyword_spotting.log             # Speech training logs
├── ternary_comparison.json          # Quantization comparison
├── synthesis_profile.json           # Boot-time profiling data
└── nas_compatibility.json           # NAS compatibility results
```

## 9. Data Requirements by Experiment

| Experiment | Data Needed | Download? | Runtime |
|------------|-------------|-----------|---------|
| Ternary Baseline | Existing ECG or synthetic | NO | 1 min |
| Synthesis Profiling | Synthetic | NO | 2-5 min |
| NAS Compatibility | Synthetic | NO | 2 min |
| Keyword Spotting | Speech Commands v0.02 | YES (~2GB) | 30-60 min |

## 10. Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"
```bash
pip3 install torch torchaudio
```

### "TypeError: unhashable type: 'StringArray'" (pandas/wfdb issue)
```bash
# Update wfdb to version compatible with pandas 2.x
pip3 install --upgrade "wfdb>=4.1.0"

# Or use the fix script
bash fix_dependencies.sh
```

### "ModuleNotFoundError: No module named 'torchcodec'" (audio loading)
```bash
# Install soundfile for audio loading (avoids torchcodec)
pip3 install --upgrade "soundfile>=0.12.0"

# Or use the fix script
bash fix_dependencies.sh
```

### "SPEECH_COMMANDS_ROOT not set"
```bash
export SPEECH_COMMANDS_ROOT=$(pwd)/data/speech_commands_v0.02
```

### "Dataset not found"
```bash
# Verify download
ls data/speech_commands_v0.02/

# Re-download if needed
python3 download_data.py --minimal
```

### Out of Memory
```bash
# Use smaller batch size
python3 run_experiments.py --experiments all --batch-size 16 --cpu
```

### CUDA Errors
```bash
# Force CPU mode
python3 run_experiments.py --experiments all --cpu
```

### Permission Denied on download_data.py
```bash
chmod +x download_data.py
```

## 11. Quick Start Commands (Copy-Paste)

For a completely fresh Linux machine:

```bash
# 1. Install dependencies
sudo apt-get update && sudo apt-get install -y python3 python3-pip wget
pip3 install -r requirements.txt

# Fix pandas/wfdb compatibility if needed
bash fix_dependencies.sh

# 2. Clone and enter repo (if not already cloned)
git clone https://github.com/yassienshaalan/tinyml-gen.git
cd tinyml-gen

# 3. Download minimal data
python3 download_data.py --minimal

# 4. Set environment variable
export SPEECH_COMMANDS_ROOT=$(pwd)/data/speech_commands_v0.02
echo "export SPEECH_COMMANDS_ROOT=$(pwd)/data/speech_commands_v0.02" >> ~/.bashrc

# 5. Test setup
cd tinyml
python3 test_experiments.py

# 6. Run all experiments
python3 run_experiments.py --experiments all --epochs 20

# 7. Check results
ls -lh results/
cat results/experiment_summary.json
```

## 12. Expected Timeline

- **Setup** (steps 1-4): 10-20 minutes
- **Data Download** (step 3): 5-15 minutes (depending on connection)
- **Testing** (step 5): 30 seconds
- **Running Experiments**:
  - Without keyword spotting: 5-10 minutes
  - With keyword spotting: 60-90 minutes total
- **Total Time**: ~1.5-2 hours for complete setup and all experiments

## 13. Verification Checklist

Before running experiments, verify:

- [ ] Python 3.7+ installed: `python3 --version`
- [ ] PyTorch installed: `python3 -c "import torch; print(torch.__version__)"`
- [ ] Speech Commands downloaded: `ls data/speech_commands_v0.02/`
- [ ] Environment variable set: `echo $SPEECH_COMMANDS_ROOT`
- [ ] Tests pass: `python3 test_experiments.py`
- [ ] In correct directory: `pwd` should end with `/tinyml`

## 14. Support

If you encounter issues:

1. Check `results/experiment_full.log` for detailed error messages
2. Verify all dependencies: `pip3 list | grep torch`
3. Check disk space: `df -h`
4. Check memory: `free -h`

For detailed documentation, see:
- [COMPLETE_EXPERIMENTAL_REPORT.md](COMPLETE_EXPERIMENTAL_REPORT.md) - Complete experiment guide
- [README.md](README.md) - Project overview
