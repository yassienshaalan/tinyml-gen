# data.py
from __future__ import annotations
import os, glob, re, math, random, time
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score

# ----------------------- Google Drive helpers (optional) -----------------------
# If you will mount Drive yourself (gcsfuse, or manual sync), you can ignore these.
def drive_download_folder(folder_id: str, local_dir: str):
    """
    OPTIONAL: Download every file in a Drive folder to local_dir using pydrive2.
    Requires a OAuth setup or a Service Account with the folder shared to it.
    """
    try:
        from pydrive2.auth import GoogleAuth
        from pydrive2.drive import GoogleDrive
    except ImportError:
        print("[DriveSync] pydrive2 not installed. Skipping.")
        return
    os.makedirs(local_dir, exist_ok=True)
    gauth = GoogleAuth()
    # For headless VMs: use service account JSON via env GOOGLE_APPLICATION_CREDENTIALS
    gauth.LocalWebserverAuth()  # or gauth.ServiceAuth() if configured
    drive = GoogleDrive(gauth)
    file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
    for f in file_list:
        dst = Path(local_dir) / f['title']
        print(f"[DriveSync] downloading {f['title']} -> {dst}")
        f.GetContentFile(str(dst))

# ---------------------------- Apnea ECG loader --------------------------------
FS = 100  # assume 100 Hz after any resample

def _list_trainable_records(root: str):
    """
    Find record ids like a01,b02,c05 that have the full triplet (.dat,.hea,.apn).
    Accepts .apn or .apn.txt. If nothing at root, auto-try one subfolder.
    Ignores Google-Drive duplicate suffixes like 'a01 (1).apn'.
    """
    root = os.path.abspath(root)

    def clean_base(bn):
        return re.sub(r"\s+\(\d+\)$", "", bn)

    def collect_ids(folder):
        dats = {clean_base(os.path.splitext(os.path.basename(p))[0])
                for p in glob.glob(os.path.join(folder, "[abc][0-9][0-9].dat"))}
        heas = {clean_base(os.path.splitext(os.path.basename(p))[0])
                for p in glob.glob(os.path.join(folder, "[abc][0-9][0-9].hea"))}
        apn_raw = set()
        for p in glob.glob(os.path.join(folder, "[abc][0-9][0-9].apn")) + \
                  glob.glob(os.path.join(folder, "[abc][0-9][0-9].apn.txt")):
            bn = os.path.basename(p)
            bn = bn.replace(".apn.txt","").replace(".apn","")
            apn_raw.add(clean_base(bn))
        ids = sorted(list(dats & heas & apn_raw))
        return ids

    ids = collect_ids(root)
    if ids:
        return ids

    # try one subfolder automatically
    try:
        for d in os.listdir(root):
            sub = os.path.join(root, d)
            if os.path.isdir(sub):
                ids_sub = collect_ids(sub)
                if ids_sub:
                    print(f"[ApneaECG] Auto-selected subfolder with records: {sub}")
                    # NOTE: upstream callers must pass the sub path as root to use it
                    # Here we just return ids; root must be corrected by caller if needed.
                    return ids_sub
    except FileNotFoundError:
        pass

    raise RuntimeError("No usable records (need a**, b**, c** with .apn/.dat/.hea).")

def _read_sig(root: str, rid: str) -> np.ndarray:
    """
    Very lightweight WFDB-ish reader for (rid).dat/.hea.
    Expects a 1-channel record; adjust if yours differs.
    """
    hea = Path(root) / f"{rid}.hea"
    dat = Path(root) / f"{rid}.dat"
    if not hea.exists() or not dat.exists():
        raise FileNotFoundError(rid)
    # Parse .hea for signal length (assume '# samples' field or last line)
    with open(hea, "r") as f:
        head = f.read().strip().splitlines()
    # heuristic to find length from header
    nsamp = None
    for line in head:
        m = re.search(r"(\d+)\s+samples", line)
        if m:
            nsamp = int(m.group(1)); break
    if nsamp is None:
        # fallback: assume 30 minutes * 60 * FS
        nsamp = 30 * 60 * FS

    # raw 16-bit signed ints
    data = np.fromfile(dat, dtype=np.int16, count=nsamp)
    # scale roughly into [-5, 5] range (typical ECG gain), adjust as needed
    data = data.astype(np.float32) / 200.0
    return data

def _read_apn_labels(root: str, rid: str) -> np.ndarray:
    """
    Read apnea annotations minute-wise: vector of 0/1 per minute.
    Accepts .apn or .apn.txt
    """
    p1 = Path(root) / f"{rid}.apn"
    p2 = Path(root) / f"{rid}.apn.txt"
    p = p1 if p1.exists() else p2
    if not p.exists():
        raise FileNotFoundError(f"No .apn for {rid}")
    labels = []
    with open(p, "r") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            # allow formats like '0'/'1' or 'A'/'N'
            if line in ("0","1"):
                labels.append(int(line))
            elif line in ("A","N"):
                labels.append(1 if line=="A" else 0)
    return np.array(labels, dtype=np.int64)

def _record_apnea_stats(root: str, rids):
    stats=[]
    for rid in rids:
        labs=_read_apn_labels(root, rid)
        stats.append((rid, labs.sum(), labs.size, float(labs.mean())))
    return stats

def stratified_by_minutes_split(root: str, rids, seed=42, frac=(0.8,0.1,0.1)):
    """
    Build splits by minute-level labels pooled across records (stratified),
    then map back to record IDs with balanced positives presence in each split.
    Simple heuristic: shuffle records but greedily assign to reach label fractions.
    """
    rng = random.Random(seed)
    pos_map = {rid:int(_read_apn_labels(root,rid).sum()>0) for rid in rids}
    rpos = [r for r in rids if pos_map[r]==1]
    rneg = [r for r in rids if pos_map[r]==0]
    rng.shuffle(rpos); rng.shuffle(rneg)

    n = len(rids)
    n_tr, n_va = int(frac[0]*n), int(frac[1]*n)
    def take(lst, k): return lst[:k], lst[k:]
    tr_p, rpos = take(rpos, int(frac[0]*len(rpos)))
    va_p, rpos = take(rpos, int(frac[1]*len(rpos)))
    te_p, rpos = rpos, []

    tr_n, rneg = take(rneg, n_tr-len(tr_p))
    va_n, rneg = take(rneg, n_va-len(va_p))
    te_n, rneg = rneg, []

    return tr_p+tr_n, va_p+va_n, te_p+te_n

class ApneaECGWindows(Dataset):
    def __init__(self, root: str, rids, length=1800, stride=None, normalize="per_window", verbose=False):
        self.root = root
        self.length = int(length)    # samples (100 Hz * 60 * 0.3 min = 1800 for 18 sec; adapt)
        self.normalize = normalize

        self._sig_cache: Dict[str, np.ndarray] = {}
        self._labs: Dict[str, np.ndarray] = {rid: _read_apn_labels(root, rid) for rid in rids}
        self.index = []  # (rid, minute, offset)
        for rid in rids:
            sig = self._sig(rid)
            nmin = len(self._labs[rid])
            # 1 window per minute, starting at minute*FS*60
            for m in range(nmin):
                self.index.append((rid, m, 0))
        if verbose:
            print(f"[ApneaECGWindows] windows={len(self.index)} length={self.length}")

    def _sig(self, rid):
        if rid not in self._sig_cache:
            self._sig_cache[rid] = _read_sig(self.root, rid)
        return self._sig_cache[rid]

    def __len__(self): return len(self.index)

    def __getitem__(self, i: int):
        rid, m, off = self.index[i]
        sig  = self._sig(rid)
        labs = self._labs[rid]

        start = m*FS*60 + off
        end   = start + self.length

        if start >= len(sig):
            chunk = np.zeros((self.length,), dtype=np.float32)
        else:
            if end > len(sig):
                pad = end - len(sig)
                chunk = np.pad(sig[start:], (0, pad), mode='edge')
            else:
                chunk = sig[start:end]

        if self.normalize == "per_window":
            mu = float(chunk.mean())
            sd = float(chunk.std() + 1e-6)
            chunk = (chunk - mu) / sd

        x = torch.from_numpy(chunk.astype(np.float32))[None, :]
        y = torch.tensor(labs[m], dtype=torch.long)
        return x, y

# --------------------------- Registry + builders -------------------------------
DATASET_REGISTRY = {}

def register_dataset(name: str, fn):
    DATASET_REGISTRY[name] = fn

def available_datasets():
    return list(DATASET_REGISTRY.keys())

def load_apnea_ecg_loaders_impl(root: str, batch_size=64, length=1800, seed=42, num_workers=0, verbose=True):
    rids = _list_trainable_records(root)
    if verbose:
        print(f"[ApneaECG] root={root} | length={length}")
        print(f"[ApneaECG] usable records={len(rids)} → {rids[:10]}{' ...' if len(rids)>10 else ''}")
    if not rids:
        raise RuntimeError("No usable records (need a**, b**, c** with .apn/.dat/.hea).")

    tr, va, te = stratified_by_minutes_split(root, rids, seed=seed, frac=(0.8,0.1,0.1))
    ds_tr = ApneaECGWindows(root, tr, length=length, normalize="per_window", verbose=False)
    ds_va = ApneaECGWindows(root, va, length=length, normalize="per_window", verbose=False)
    ds_te = ApneaECGWindows(root, te, length=length, normalize="per_window", verbose=False)

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    meta = {'num_channels': 1, 'seq_len': length, 'num_classes': 2, 'fs': FS}
    return dl_tr, dl_va, dl_te, meta

def register_apnea(root: str):
    def _wrapper(**kwargs):
        return load_apnea_ecg_loaders_impl(root=root, **kwargs)
    register_dataset('apnea_ecg', _wrapper)

# --------------------------- Class-distribution print --------------------------
def print_class_dist_from_loaders(dl_tr, dl_va, dl_te, meta, max_samples=2000):
    def _approx(dl):
        n0=n1=0; c=0
        for xb,yb in dl:
            yb = yb.numpy() if isinstance(yb, torch.Tensor) else yb
            z0 = int((yb==0).sum()); z1 = int((yb==1).sum())
            n0 += z0; n1 += z1; c += len(yb)
            if c >= max_samples: break
        tot = n0+n1 if (n0+n1)>0 else 1
        p0 = 100.0*n0/tot; p1=100.0*n1/tot
        print(f"  counted samples : {min(c,max_samples)}  (limit={max_samples})")
        print(f"  class 0: {n0} ({p0:.2f}%)")
        print(f"  class 1: {n1} ({p1:.2f}%)")
        print("========================================")
    print("\n=== ApneaECG Train class distribution (approx) ===")
    _approx(dl_tr)
    print("\n=== ApneaECG Val class distribution (approx) ===")
    _approx(dl_va)
    print("\n=== ApneaECG Test class distribution (approx) ===")
    _approx(dl_te)
