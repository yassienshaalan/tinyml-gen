
# TinyML Experiments (Modularized, v2 with explicit TinyMethodModel dz/dh)

- `data_loaders.py` — dataset loaders for Apnea-ECG, PTB-XL, MIT-BIH (hardcoded Drive paths).
- `models.py` — includes all models TinyMethodModel(in_ch, num_classes, base_filters=16, dz, dh, **kwargs).
  Registry forwards **kwargs.
- `experiments.py` — uses `safe_build_model` (and falls back to aggressive kwarg filtering).
- `main.py` — CLI + logging to `run.log` next to main.

Run:
  python3 main.py
  python3 main.py --dataset ptbxl
  python3 main.py --dataset apnea_ecg --models tiny_method --epochs 10

Drive paths are the same as your Colab; adjust constants in `data_loaders.py` if needed.
