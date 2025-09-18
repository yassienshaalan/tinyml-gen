# main.py
from __future__ import annotations
import os, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.optim import Adam

from data import (
    register_apnea, available_datasets, load_apnea_ecg_loaders_impl,
    print_class_dist_from_loaders
)
from models_and_train import (
    safe_build_model, count_params, estimate_flash_usage, deployment_profile,
    diagnose_nan_issues, fix_nan_issues, train_epoch_ce, evaluate, evaluate_logits,
    eval_prob_fn, tune_threshold
)

def get_or_make_loaders_once(ds_key, root: str, batch_size=64, length=1800, num_workers=0, seed=42, verbose=True):
    if ds_key != 'apnea_ecg':
        raise NotImplementedError("Only apnea_ecg is wired here, extend as needed.")
    return load_apnea_ecg_loaders_impl(root=root, batch_size=batch_size, length=length,
                                       num_workers=num_workers, seed=seed, verbose=verbose)

def run_experiment(
    dataset_name: str, model_name: str, root: str,
    epochs=8, batch_size=64, length=1800, device='cpu', lr=1e-3, num_workers=0, seed=42
):
    torch.manual_seed(seed); np.random.seed(seed)
    dl_tr, dl_va, dl_te, meta = get_or_make_loaders_once(dataset_name, root, batch_size, length, num_workers, seed, True)
    print_class_dist_from_loaders(dl_tr, dl_va, dl_te, meta)

    in_ch = meta['num_channels']; num_classes = meta['num_classes']
    model = safe_build_model(model_name, in_ch, num_classes)
    model.to(device)

    params = count_params(model)
    flash_info = estimate_flash_usage(model, 'int8')
    print(f" Model: {model_name} | Params: {params:,} | Flash (INT8): {flash_info['flash_human']}")

    opt = Adam(model.parameters(), lr=lr)

    # Pre-training diagnostics
    xb1, yb1 = next(iter(dl_tr))
    diagnose_nan_issues(model, xb1[:1], device=device)
    fix_nan_issues(model)

    best_val_acc = 0.0; best_state=None; best_t=0.5; val_f1=0.0; val_precision=0.0; val_recall=0.0
    print(f" Training for {epochs} epochs...")
    t0 = time.time()
    for ep in range(epochs):
        tr_loss = train_epoch_ce(model, dl_tr, opt, device=device, w_size=1.0)
        va_acc, _ = evaluate(model, dl_va, device=device)

        # Calibrate threshold on val
        v_logits, vy = evaluate_logits(model, dl_va, device=device)
        vp = eval_prob_fn(v_logits)
        t_star, val_f1 = tune_threshold(vy, vp)
        vy_hat = (vp >= t_star).astype(int)
        _avg = 'binary' if np.unique(vy).size == 2 else 'macro'
        val_precision = float((__import__("sklearn.metrics").metrics.precision_score)(vy, vy_hat, average=_avg, zero_division=0))
        val_recall    = float((__import__("sklearn.metrics").metrics.recall_score)(vy, vy_hat, average=_avg, zero_division=0))

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k:v.cpu() for k,v in model.state_dict().items()}
            best_t = t_star

        print(f"  Epoch {ep+1}/{epochs}: train_loss={tr_loss:.4f} val_acc={va_acc:.4f} | Val P/R/F1@t*: {val_precision:.3f}/{val_recall:.3f}/{val_f1:.3f}")

    dur = time.time()-t0
    if best_state is not None:
        model.load_state_dict(best_state)

    # Test
    test_acc=test_f1=test_precision=test_recall=None
    te_logits, ty = evaluate_logits(model, dl_te, device=device)
    tp = eval_prob_fn(te_logits)
    yhat = (tp >= best_t).astype(int)
    _avg_te = 'binary' if np.unique(ty).size == 2 else 'macro'
    test_acc, _ = evaluate(model, dl_te, device=device)
    from sklearn.metrics import precision_score, recall_score, f1_score
    test_precision = float(precision_score(ty, yhat, average=_avg_te, zero_division=0))
    test_recall    = float(recall_score(ty, yhat, average=_avg_te, zero_division=0))
    test_f1        = float(f1_score(ty, yhat, average=_avg_te, zero_division=0))

    # Deployment profile
    def _flash_bytes_int8(m): 
        try: return estimate_flash_usage(m, 'int8')["flash_bytes"]
        except: return count_params(m)
    deploy = deployment_profile(model, meta, flash_bytes_fn=_flash_bytes_int8, device=str(device))

    print(f" Test accuracy: {test_acc:.4f}")
    print(f" Test P/R/F1@t*: {test_precision:.3f}/{test_recall:.3f}/{test_f1:.3f}")
    print(f" Best val acc: {best_val_acc:.4f} | Val P/R/F1@t*: {val_precision:.3f}/{val_recall:.3f}/{val_f1:.3f}")
    print(f" Training time: {dur:.1f}s | Flash: {deploy['flash_kb']:.2f} KB")

    return {
        'dataset': dataset_name,
        'model': model_name,
        'epochs': epochs,
        'lr': lr,
        'val_acc': best_val_acc,
        'val_f1_at_t': float(val_f1),
        'val_precision_at_t': float(val_precision),
        'val_recall_at_t': float(val_recall),
        'test_acc': float(test_acc),
        'test_f1_at_t': float(test_f1),
        'test_precision_at_t': float(test_precision),
        'test_recall_at_t': float(test_recall),
        'threshold_t': float(best_t),
        'params': int(sum(p.numel() for p in model.parameters())),
        'flash_kb': float(deploy['flash_kb']),
        'ram_act_peak_kb': float(deploy['ram_act_peak_kb']),
        'param_kb': float(deploy['param_kb']),
        'buffer_kb': float(deploy['buffer_kb']),
        'macs': float(deploy['macs']),
        'latency_ms': float(deploy['latency_ms']),
        'energy_mJ': float(deploy['energy_mJ']),
        'train_time_s': float(dur),
        'channels': int(meta.get('num_channels', 1)),
        'seq_len': int(meta.get('seq_len', 1800)),
        'num_classes': int(meta.get('num_classes', 2)),
    }

def plot_pareto(df: pd.DataFrame, x='flash_kb', y='test_f1_at_t') -> pd.DataFrame:
    # return non-dominated points
    D = df[[x,y]].to_numpy()
    keep = []
    for i in range(len(D)):
        xi, yi = D[i]
        dominated = False
        for j in range(len(D)):
            if j==i: continue
            xj,yj = D[j]
            if (xj <= xi and yj >= yi) and (xj < xi or yj > yi):
                dominated = True; break
        if not dominated: keep.append(i)
    return df.iloc[keep].sort_values([x, y])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=str, required=True, help="Path containing a**/b**/c** .dat/.hea/.apn[.txt]")
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--length', type=int, default=1800)
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--models', type=str, nargs='+',
                    default=['tiny_separable_cnn','regular_cnn'])
    ap.add_argument('--out_csv', type=str, default='results.csv')
    args = ap.parse_args()

    # Register dataset
    register_apnea(args.root)

    results = []
    for m in args.models:
        print("\n" + "="*60)
        print(f" Experiment: apnea_ecg + {m}")
        print("="*60)
        res = run_experiment(
            'apnea_ecg', m, root=args.root,
            epochs=args.epochs, batch_size=args.batch_size,
            length=args.length, device=args.device, lr=args.lr,
            num_workers=0
        )
        results.append(res)

    df = pd.DataFrame(results)
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved results to {args.out_csv}")

    try:
        pf = plot_pareto(df, x='flash_kb', y='test_f1_at_t')
        print("\nPARETO FRONTIER (non-dominated):")
        print(pf[['model','flash_kb','test_f1_at_t']])
        pf.to_csv('pareto.csv', index=False)
        print("Saved pareto.csv")
    except Exception as e:
        print(f"[WARN] Pareto failed: {e}")

if __name__ == '__main__':
    main()
