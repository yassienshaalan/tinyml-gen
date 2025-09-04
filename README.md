# TinyML Generative Compression for Biosignals
Edge health monitoring on microcontrollers with only tens of KBs of flash and a few KBs of SRAM. This repo contains the paper code for a generative channel-mixing compressor and a VAE+linear-head baseline that deliver clinically useful predictions under strict TinyML budgets.


## Why this matters

On MCU-class devices, channel-mixing weights dominate storage even after quantization. By turning storage into generation exactly at the mixing bottleneck, \method{} frees flash while preserving accuracy.

## Datasets (supported in this repo)

* Apnea-ECG (PhysioNet): apnea vs normal minute-level labels.

* PTB-XL (PhysioNet): 12-lead ECG diagnostics (proxy subsets).

* UCI-HAR: human activity recognition (as a non-ECG proxy).

## Repo layout (typical)
```bash
experiments/
  run_apnea.py         # Enhanced experiment runner (CNN, VAE+Head, metrics)
  run_ptbxl.py         # PTB-XL proxy
  run_ucihar.py        # UCI-HAR proxy
models/
  separable_cnn.py     # SharedCoreSeparable1D + tinyml_packed_bytes()
  generative_mix.py     # Synthesizer MLP + latent codes
  vae_1d.py            # Tiny VAE encoder/decoder + adapter/head
utils/
  data_apnea.py        # loaders, caching, class distribution
  schedulers.py        # cosine warmup, etc.
  losses.py            # focal loss, label smoothing, mixup
  metrics.py           # accuracy, F1, reports
tools/
  report_packed_bytes.py
notebooks/
  TinyML_Experiments.ipynb
'''

## Repro tips

Use cosine warmup + AdamW (weight decay ~1e-4).

For imbalanced data, enable FocalLoss and mixup (α≈0.2).

Early stop on val accuracy; track F1 for apnea balance.

Always report packed bytes (flash) and peak SRAM alongside accuracy.

## TinyML accounting

### We report:

Packed flash bytes for: synthesizer MLP, latent codes, INT8 first-layer weights.

Peak SRAM use during inference.

Optional: boot-time synthesis cost (one-off) vs lazy per-layer synthesis.

##  Ethical & clinical use

This code is for research. Not a medical device. Do not use for diagnosis without regulatory approval and clinical validation.
