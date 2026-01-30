# Once-for-All Channel Mixers (HYPERTINYPW): Generative Compression for TinyML

Code for reproducing results from paper "Once-for-All Channel Mixers (HYPERTINYPW): Generative Compression for TinyML". Accepted at MlSys26 (Conference on Machine Learning and Systems). 

## Rebuttal Experiments (NEW)

**All experiments log results to files automatically!**

### Quick Start

```bash
cd tinyml

# Test setup (30 seconds)
python test_rebuttal_modules.py

# Run experiments WITHOUT needing to download new data (5-10 min)
python run_rebuttal_experiments.py --experiments ternary,synthesis,nas

# OR run all experiments including keyword spotting (needs dataset, ~60 min)
python run_rebuttal_experiments.py --experiments all --epochs 20
```

**Results**: All outputs saved to `rebuttal_results/` with comprehensive logs

### What's Included

| Experiment | Addresses | Data Needed | Time |
|------------|-----------|-------------|------|
| **Ternary Baseline** | "Compare to quantization" | Uses existing ECG data | 1 min |
| **Synthesis Profiling** | "Boot-time cost?" | Synthetic (no data) | 2-5 min |
| **NAS Compatibility** | "Relation to NAS" | Synthetic (no data) | 2 min |
| **Keyword Spotting** | "Limited to ECG" | Needs Speech Commands (~2GB) | 30-60 min |

### Complete Documentation

**See [REBUTTAL_GUIDE.md](REBUTTAL_GUIDE.md)** for:
- Detailed instructions
- Data sources (most use existing data!)
- Expected outputs
- Integration with paper
- Troubleshooting

## Structure

```
tinyml-gen/
  tinyml/
    # Core files
    main.py                        # Original experiments
    models.py                      # Model architectures
    experiments.py                 # Experiment framework
    datasets.py, data_loaders.py   # ECG datasets
    
    # NEW: Rebuttal experiments (all with file logging)
    run_rebuttal_experiments.py   # Main runner (logs to files)
    ternary_baseline.py           # Ternary quantization baseline
    synthesis_profiler.py         # Boot-time profiling
    nas_compatibility.py          # NAS compatibility
    speech_dataset.py             # Keyword spotting (optional)
    test_rebuttal_modules.py      # Test suite
    
  REBUTTAL_GUIDE.md               # Complete guide (READ THIS)
  colab_version/                  # Original notebook
```

## Original Experiments

Run original ECG experiments:
```bash
cd tinyml
python main.py --datasets apnea_ecg,ptbxl,mitdb
```
