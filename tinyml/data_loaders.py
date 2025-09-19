
import math, os, json, random
from pathlib import Path
from typing import Tuple, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import wfdb

# =============================
# GCS support (optional)
# =============================
try:
    import gcsfs  # pip install gcsfs
except Exception:
    gcsfs = None

def _is_gcs_path(p: Union[str, Path]) -> bool:
    return str(p).startswith("gs://")

def _gcsfs():
    if gcsfs is None:
        raise ImportError("gcsfs is required for gs:// paths. Install with: pip install gcsfs")
    # Uses Application Default Credentials on GCP VM
    return gcsfs.GCSFileSystem(cache_timeout=60)

# Local cache for WFDB files fetched from GCS
DATA_GCS_CACHE = Path(os.environ.get("DATA_GCS_CACHE", "/tmp/data_gcs_cache"))
DATA_GCS_CACHE.mkdir(parents=True, exist_ok=True)

def _gcs_join(*parts: str) -> str:
    out = str(parts[0]).rstrip("/")
    for p in parts[1:]:
        out += "/" + str(p).lstrip("/")
    return out

def _gcs_ls(prefix: str) -> List[str]:
    fs = _gcsfs()
    try:
        return fs.ls(prefix)
    except FileNotFoundError:
        return []
    except Exception:
        return []

def _gcs_exists(path: str) -> bool:
    fs = _gcsfs()
    try:
        return fs.exists(path)
    except Exception:
        return False

def _gcs_get_file(src: str, dst: Path):
    """Download a single file from GCS to local dst if not present."""
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    fs = _gcsfs()
    with fs.open(src, "rb") as fsrc, open(dst, "wb") as fdst:
        fdst.write(fsrc.read())

def _ensure_local_record(gcs_dir: str, rid: str, exts: List[str]) -> Path:
    """
    Ensure local copies of WFDB record files (e.g., .hea/.dat/.apn) for record 'rid' under gcs_dir.
    Returns the local directory containing the files.
    """
    if not _is_gcs_path(gcs_dir):
        return Path(gcs_dir)
    local_dir = DATA_GCS_CACHE / Path(gcs_dir.replace("gs://", "")).as_posix().replace("/", "_") / rid
    for ext in exts:
        _gcs_get_file(_gcs_join(gcs_dir, f"{rid}.{ext}"), local_dir / f"{rid}.{ext}")
    return local_dir

def _record_base_paths(root: Union[str, Path], rid: str, needed_exts: List[str]) -> str:
    """
    Return a base path (without extension) that wfdb.* can use.
    If root is gs://, ensure files are cached locally first.
    """
    root = str(root)
    if _is_gcs_path(root):
        local_dir = _ensure_local_record(root, rid, needed_exts)
        return str(local_dir / rid)
    else:
        return str(Path(root) / rid)

# =============================
# Config
# =============================
# Data roots can be either local or gs://; defaults can be overridden via env vars.
DATA_BASE = os.environ.get("TINYML_DATA_ROOT", "/content/drive/MyDrive/tinyml_hyper_tiny_baselines/data")
APNEA_ROOT = os.environ.get("APNEA_ROOT", f"{DATA_BASE}/apnea-ecg-database-1.0.0")
PTBXL_ROOT = os.environ.get("PTBXL_ROOT", f"{DATA_BASE}/ptbxl")
MITDB_ROOT = os.environ.get("MITDB_ROOT", f"{DATA_BASE}/mitbih/raw")

FS = 100  # Apnea-ECG sampling rate

# =============================
# Apnea-ECG (GCS-aware)
# =============================
def _list_trainable_records(root: Union[str, Path]):
    """
    Use only learning-set records that truly have .apn: a**, b**, c**.
    Exclude x** (no labels) and *er variants.
    Require .dat, .hea, .apn to exist.
    """
    root = str(root)
    recs = []
    if _is_gcs_path(root):
        for f in _gcs_ls(root):
            p = Path(f)
            if p.suffix == ".hea":
                rid = p.stem
                if not rid.startswith(("a", "b", "c")):  # drop x**
                    continue
                if rid.endswith("er"):
                    continue
                dat_ok = _gcs_exists(_gcs_join(root, f"{rid}.dat"))
                apn_ok = _gcs_exists(_gcs_join(root, f"{rid}.apn"))
                if dat_ok and apn_ok:
                    recs.append(rid)
    else:
        for hea in Path(root).glob("*.hea"):
            rid = hea.stem
            if not rid.startswith(("a","b","c")):
                continue
            if rid.endswith("er"):
                continue
            if (Path(root)/f"{rid}.dat").exists() and (Path(root)/f"{rid}.apn").exists():
                recs.append(rid)
    return sorted(set(recs))

def _minute_labels_rdann(root: Union[str, Path], rid: str):
    base = _record_base_paths(root, rid, needed_exts=["apn", "hea", "dat"])
    ann = wfdb.rdann(base, 'apn')
    return [1 if s.upper() == 'A' else 0 for s in ann.symbol]

def _load_signal(root: Union[str, Path], rid: str) -> np.ndarray:
    base = _record_base_paths(root, rid, needed_exts=["hea", "dat"])
    try:
        sig, fields = wfdb.rdsamp(base)
        idx = 0
        try:
            names = fields.get('sig_name', None)
            if names and isinstance(names, (list, tuple)):
                match_idx = None
                for i, nm in enumerate(names):
                    if str(nm).lower() == 'ecg' or 'ecg' in str(nm).lower():
                        match_idx = i
                        break
                if match_idx is not None:
                    idx = match_idx
        except Exception:
            pass
        sig_arr = sig[:, idx] if sig.ndim > 1 else sig
        return sig_arr.astype(np.float32)
    except Exception:
        rec = wfdb.rdrecord(base)
        if rec.p_signal is not None:
            sig = rec.p_signal
        else:
            sig = rec.d_signal.astype(np.float32)
        idx = 0
        try:
            if hasattr(rec, 'sig_name') and rec.sig_name:
                match_idx = None
                for i, nm in enumerate(rec.sig_name):
                    if str(nm).lower() == 'ecg' or 'ecg' in str(nm).lower():
                        match_idx = i
                        break
                if match_idx is not None:
                    idx = match_idx
        except Exception:
            pass
        sig_arr = sig[:, idx] if sig.ndim > 1 else sig
        return sig_arr.astype(np.float32)

def _sanitize_and_standardize_window(x: np.ndarray, clip_val: float = 10.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32, order="C")
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    m = float(x.mean())
    v = float(x.var())
    if not np.isfinite(m):
        m = 0.0
    if (not np.isfinite(v)) or (v < 1e-4):
        x = x - m
    else:
        x = (x - m) / np.sqrt(v + 1e-6)
    if clip_val is not None:
        x = np.clip(x, -clip_val, clip_val)
    return x.astype(np.float32, copy=False)

class ApneaECGWindows(Dataset):
    def __init__(self, root: Union[str, Path], records, length: int, stride: int|None=None, normalize="per_window", verbose=True):
        self.root = root if isinstance(root, str) else str(root)
        self.records = list(records)
        self.length  = int(length)
        self.stride  = int(length) if stride is None else int(stride)
        self.normalize = normalize
        self.verbose = verbose

        assert self.length > 0 and self.stride > 0
        per_min_len = FS * 60  # 6000
        max_start   = per_min_len - self.length
        if max_start < 0:
            raise ValueError(f"length={self.length} exceeds 60s (6000 samples). Use <=6000.")

        self._sig_cache = {}
        self._labs = {}
        self.index = []

        offsets = list(range(0, max_start+1, self.stride)) or [0]

        for rid in self.records:
            base = _record_base_paths(self.root, rid, needed_exts=["hea"])
            hdr = wfdb.rdheader(base)
            if int(round(hdr.fs)) != FS:
                if self.verbose: print(f"[ApneaECGWindows] Skip {rid}: fs={hdr.fs} != 100Hz")
                continue

            try:
                labs = _minute_labels_rdann(self.root, rid)
            except Exception:
                labs = []
            if not labs:
                if self.verbose: print(f"[ApneaECGWindows] Skip {rid}: no A/N labels in .apn")
                continue
            self._labs[rid] = labs

            for m in range(len(labs)):
                for off in offsets:
                    self.index.append((rid, m, off))

        if self.verbose:
            print(f"[ApneaECGWindows] Built {len(self.index)} windows from {len(self._labs)} records.")

    def __len__(self): return len(self.index)

    def _sig(self, rid: str):
        if rid not in self._sig_cache:
            self._sig_cache[rid] = _load_signal(self.root, rid)
        return self._sig_cache[rid]

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
              chunk = np.pad(sig[start:], (0, pad), mode="constant", constant_values=0.0)
          else:
              chunk = sig[start:end]

      if getattr(self, "normalize", None) == "per_window":
          chunk = _sanitize_and_standardize_window(chunk)
      else:
          chunk = np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)

      raw_y = labs[m]
      if isinstance(raw_y, (str, bytes)):
          y = 1 if (raw_y == 'A' or raw_y == b'A' or raw_y == '1' or raw_y == b'1') else 0
      else:
          try:
              yi = int(raw_y)
              y = 1 if yi == 1 else 0
          except Exception:
              y = 0

      if not np.isfinite(chunk).all():
          chunk = np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)

      x = torch.from_numpy(chunk.astype(np.float32)).unsqueeze(0)  # [1, T]
      y = torch.tensor(y, dtype=torch.long)
      return x, y

def _record_apnea_stats(root: Union[str, Path], records: list[str]):
    stats = []
    for rid in records:
        labs = _minute_labels_rdann(root, rid)
        a = int(sum(labs))
        n = int(len(labs) - a)
        prev = a / max(1, a + n)
        stats.append((rid, a, n, prev))
    return stats

def _records_from_index(ds):
    return sorted({rid for (rid, _, _) in ds.dataset.index})

# These helpers must exist elsewhere in your project (unchanged):
# _stratified_record_split_apnea, USE_WEIGHTED_SAMPLER, _make_weighted_sampler_apnea, _wif, print_class_distribution
def load_apnea_ecg_loaders_impl(root, batch_size=64, length=1800, stride=None, verbose=True, seed=1337):
    root = root if isinstance(root, str) else str(root)
    if verbose:
        print(f"[ApneaECG] root={root} | length={length} | stride={stride}")

    recs = _list_trainable_records(root)
    if verbose:
        print(f"[ApneaECG] usable records={len(recs)} → {recs[:10]}{' ...' if len(recs)>10 else ''}")
    if not recs:
        raise RuntimeError("No usable records (need a**, b**, c** with .apn/.dat/.hea).")

    train_recs, val_recs, test_recs = _stratified_record_split_apnea(root, recs, seed=seed, frac=(0.8,0.1,0.1))

    if verbose:
        tr_stats = _record_apnea_stats(root, train_recs)
        va_stats = _record_apnea_stats(root, val_recs)
        te_stats = _record_apnea_stats(root, test_recs)
        print(f"[Split] train|val|test records: {len(train_recs)}|{len(val_recs)}|{len(test_recs)}")
        print(f"  positives per split (records w/ any apnea): "
              f"{sum(1 for _,a,_,_ in tr_stats if a>0)} | "
              f"{sum(1 for _,a,_,_ in va_stats if a>0)} | "
              f"{sum(1 for _,a,_,_ in te_stats if a>0)}")

    ds_tr = ApneaECGWindows(root, train_recs, length=length, stride=stride, verbose=verbose)
    ds_va = ApneaECGWindows(root, val_recs,   length=length, stride=stride, verbose=verbose)
    ds_te = ApneaECGWindows(root, test_recs,  length=length, stride=stride, verbose=verbose)

    num_workers = 2 if torch.cuda.is_available() else 0
    if USE_WEIGHTED_SAMPLER:
        sampler = _make_weighted_sampler_apnea(ds_tr)
        dl_tr = DataLoader(ds_tr, batch_size=batch_size, sampler=sampler, shuffle=False,num_workers=num_workers, drop_last=True, worker_init_fn=_wif)
    else:
        dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,num_workers=num_workers, drop_last=True, worker_init_fn=_wif)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False, worker_init_fn=_wif)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False, worker_init_fn=_wif)

    print("\\n=== ApneaECG Train class distribution (approx) ===")
    print_class_distribution(dl_tr, "ApneaECG Train")
    print("========================================")
    print("\\n=== ApneaECG Val class distribution (approx) ===")
    print_class_distribution(dl_va, "ApneaECG Val")
    print("========================================")
    print("\\n=== ApneaECG Test class distribution (approx) ===")
    print_class_distribution(dl_te, "ApneaECG Test")
    print("========================================")

    if len(ds_tr) == 0 or len(ds_va) == 0:
        raise RuntimeError("No windows built. Check .apn presence and that length < 6000.")

    return dl_tr, dl_va, dl_te

# =============================
# MIT-BIH (GCS-aware)
# =============================
AAMI_MAP = {
    'N':'N', 'L':'N', 'R':'N', 'e':'N', 'j':'N',
    'A':'S','a':'S','J':'S','S':'S',
    'V':'V','E':'V',
    'F':'F',
    '/':'Q','f':'Q','Q':'Q','|':'Q','~':'Q','!':'Q','x':'Q','t':'Q','p':'Q','u':'Q'
}

def _read_signal_record(root: Union[str, Path], rec: str, prefer_lead_idx=0):
    base = _record_base_paths(root, rec, needed_exts=["hea", "dat"])
    record = wfdb.rdrecord(base)
    X = np.asarray(record.p_signal, dtype=np.float32)  # (T, L)
    fs = int(record.fs)
    L = X.shape[1]
    idx = prefer_lead_idx if prefer_lead_idx < L else 0
    x = X[:, idx]
    return x, fs

def _read_beats(root: Union[str, Path], rec: str):
    base = _record_base_paths(root, rec, needed_exts=["hea", "dat", "atr"])
    ann = wfdb.rdann(base, 'atr')
    return np.asarray(ann.sample), ann.symbol  # sample indices, symbols

def _window_around(center: int, T: int, L: int):
    s = max(0, center - L//2)
    e = s + L
    if e > T:
        e = T
        s = max(0, e - L)
    return s, e

class MITBIHBeats(Dataset):
    def __init__(self, root: Union[str, Path], records: list[str], length: int = 1800, binary=True, zscore=True):
        self.root = root if isinstance(root, str) else str(root)
        self.length = int(length)
        self.binary = binary
        self.zscore = zscore
        self.items = []  # (rec, start, end, label)

        for rec in records:
            try:
                x, fs = _read_signal_record(self.root, rec)
                rpeaks, symbols = _read_beats(self.root, rec)
            except Exception:
                continue
            for s, sym in zip(rpeaks, symbols):
                aami = AAMI_MAP.get(sym, 'Q')
                if self.binary:
                    y = 1 if aami == 'V' else 0
                else:
                    cls = {'N':0,'S':1,'V':2,'F':3,'Q':4}
                    y = cls.get(aami, 4)
                st, en = _window_around(s, len(x), self.length)
                if en - st != self.length or st < 0:
                    continue
                self.items.append((rec, st, en, y))

        random.Random(123).shuffle(self.items)

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        rec, st, en, y = self.items[i]
        x, fs = _read_signal_record(self.root, rec)
        seg = x[st:en]
        if self.zscore and np.std(seg) > 1e-6:
            seg = (seg - np.mean(seg)) / (np.std(seg) + 1e-8)
        seg = np.expand_dims(seg.astype(np.float32), 0)
        return torch.from_numpy(seg), torch.tensor(int(y), dtype=torch.long)

def _mitbih_records(root: Union[str, Path]) -> List[str]:
    root = str(root)
    recs = []
    if _is_gcs_path(root):
        for f in _gcs_ls(root):
            if f.endswith(".hea"):
                recs.append(Path(f).stem)
    else:
        for p in Path(root).glob("*.hea"):
            recs.append(p.stem)
    return sorted(set(recs))

def load_mitdb_loaders(root: Union[str, Path], batch_size=64, length=1800, binary=True):
    root = root if isinstance(root, str) else str(root)
    if not _is_gcs_path(root) and not Path(root).exists():
        raise FileNotFoundError(f"MITDB root not found: {root}")
    recs = _mitbih_records(root)
    if not recs:
        raise RuntimeError(f"No .hea records found under {root}")

    recs = sorted(recs)
    n = len(recs)
    tr_recs = recs[: int(0.8*n)]
    va_recs = recs[int(0.8*n): int(0.9*n)]
    te_recs = recs[int(0.9*n):]

    tr_ds = MITBIHBeats(root, tr_recs, length=length, binary=binary)
    va_ds = MITBIHBeats(root, va_recs, length=length, binary=binary)
    te_ds = MITBIHBeats(root, te_recs, length=length, binary=binary)

    tr = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2, worker_init_fn=_wif)
    va = DataLoader(va_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=2, worker_init_fn=_wif)
    te = DataLoader(te_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=2, worker_init_fn=_wif)
    return tr, va, te, {"binary": binary, "records": {"train": len(tr_recs), "val": len(va_recs), "test": len(te_recs)}}

# =============================
# PTB-XL (GCS-aware)
# =============================
def _ptbxl_paths(root: Union[str, Path]):
    root = str(root)
    if _is_gcs_path(root):
        raw = _gcs_join(root, "raw")
        csv = _gcs_join(raw, "ptbxl_database.csv")
        scp = _gcs_join(raw, "scp_statements.csv")
        if not _gcs_exists(csv) or not _gcs_exists(scp):
            raise FileNotFoundError(f"Could not find ptbxl_database.csv / scp_statements.csv under: {root}")
        return csv, scp, raw
    else:
        rootp = Path(root)
        raw = rootp / "raw"
        if (raw / "ptbxl_database.csv").exists():
            return raw / "ptbxl_database.csv", raw / "scp_statements.csv", raw
        for p in [rootp, rootp.parent]:
            csv = list(p.rglob("ptbxl_database.csv"))
            scp = list(p.rglob("scp_statements.csv"))
            if csv and scp:
                raw = Path(csv[0]).parent
                return csv[0], scp[0], raw
        raise FileNotFoundError("Could not find ptbxl_database.csv / scp_statements.csv under: " + str(root))

def _wfdb_read_lead(base: Union[str, Path], prefer_lead="II") -> Tuple[np.ndarray, int]:
    base = str(base)
    if _is_gcs_path(base):
        gp = Path(base.replace("gs://", ""))
        rid = gp.name
        gdir = "gs://" + str(gp.parent).strip("/")
        local_dir = _ensure_local_record(gdir, rid, exts=["hea", "dat"])
        base_local = str(local_dir / rid)
    else:
        base_local = str(Path(base))

    rec = wfdb.rdrecord(base_local)
    fs = int(rec.fs)
    X = np.asarray(rec.p_signal, dtype=np.float32) if rec.p_signal is not None else np.asarray(rec.d_signal, dtype=np.float32)
    if isinstance(prefer_lead, int):
        idx = prefer_lead if prefer_lead < X.shape[1] else 0
    else:
        idx = 0
        try:
            names = getattr(rec, 'sig_name', []) or []
            for i, nm in enumerate(names):
                if str(nm).strip().upper() == str(prefer_lead).strip().upper():
                    idx = i; break
        except Exception:
            pass
    x = X[:, idx] if X.ndim > 1 else X
    return x.astype(np.float32), fs

def _ptbxl_labelize(
    df: pd.DataFrame,
    scp_df: pd.DataFrame,
    task: str = "binary_diag",
    debug: bool = True
) -> Tuple[pd.DataFrame, List[int]]:
    import ast
    if task not in ("binary_diag", "superclass"):
        raise ValueError("task must be 'binary_diag' or 'superclass'")
    df = df.copy()

    def parse_codes(s):
        if isinstance(s, dict):
            return s
        if isinstance(s, str):
            try:
                return ast.literal_eval(s)
            except Exception:
                return {}
        if pd.isna(s):
            return {}
        return {}

    df["scp_codes_dict"] = df["scp_codes"].apply(parse_codes)

    scp_df = scp_df.copy()
    idx_upper = set(scp_df.index.astype(str).str.strip().str.upper())
    if "NORM" not in idx_upper and len(scp_df.columns) > 0:
        code_col = scp_df.columns[0]
        scp_df = scp_df.set_index(code_col, drop=True)
    scp_df.index = scp_df.index.astype(str).str.strip().str.upper()

    super_col = "superclass" if "superclass" in scp_df.columns else "diagnostic_class"
    code2super = scp_df[super_col].astype(str).str.upper().to_dict()

    if "diagnostic" in scp_df.columns:
        diag_raw = scp_df["diagnostic"]
    else:
        diag_raw = pd.Series(0, index=scp_df.index)

    diag_mask = (
        diag_raw.astype(str).str.lower().isin({"1", "1.0", "true", "t", "yes"})
        | (pd.to_numeric(diag_raw, errors="coerce").fillna(0).astype(float) > 0)
    )

    diag_codes = {c for c in scp_df.index[diag_mask] if code2super.get(c, "NORM") != "NORM"}

    order = ["NORM", "MI", "STTC", "HYP", "CD"]
    cls_map = {c: i for i, c in enumerate(order)}
    y = []

    def _score(v):
        try:
            return float(v)
        except Exception:
            return 0.0

    for d in df["scp_codes_dict"]:
        dk = {str(k).strip().upper(): v for k, v in (d or {}).items()}
        diag_subset = {k: v for k, v in dk.items() if k in diag_codes}
        if diag_subset:
            top_code = max(diag_subset.items(), key=lambda kv: _score(kv[1]))[0]
            superc = code2super.get(top_code, "NORM")
            if task == "binary_diag":
                y.append(1 if superc != "NORM" else 0)
            else:
                y.append(cls_map.get(superc, 0))
        else:
            y.append(0 if task == "binary_diag" else cls_map["NORM"])

    df["y"] = pd.Series(y, index=df.index).astype("int64")

    if debug:
        used_super_col = super_col
        print(f"[PTB-XL] using '{used_super_col}' | diag usable codes: {len(diag_codes)}")
        print(df["y"].value_counts().sort_index())

    classes = [0, 1] if task == "binary_diag" else list(range(len(order)))
    return df, classes

def _pad_crop(x: np.ndarray, L: int):
    if len(x) == L:
        return x
    if len(x) > L:
        s = (len(x) - L)//2
        return x[s:s+L]
    out = np.zeros(L, dtype=np.float32)
    s = (L - len(x))//2
    out[s:s+len(x)] = x
    return out

class PTBXLWindows(Dataset):
    def __init__(self, df: pd.DataFrame, raw_root: Union[str, Path], length: int, lead="II", zscore=True):
        self.df = df.reset_index(drop=True)
        self.raw_root = raw_root if isinstance(raw_root, str) else str(raw_root)
        self.length = int(length)
        self.lead = lead
        self.zscore = zscore

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        base_rel = r["filename_lr"]
        if _is_gcs_path(self.raw_root):
            base = _gcs_join(self.raw_root, str(base_rel).lstrip("/"))
        else:
            base = str((Path(self.raw_root) / base_rel).with_suffix(""))
        x, fs = _wfdb_read_lead(base, prefer_lead=self.lead)
        if self.zscore and np.std(x) > 1e-6:
            x = (x - np.mean(x)) / (np.std(x) + 1e-8)
        x = _pad_crop(x, self.length)
        x = np.expand_dims(x.astype(np.float32), 0)  # (1, L)
        y = int(r["y"])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

def load_ptbxl_loaders(
    root: Union[str, Path],
    batch_size: int = 64,
    length: int = 1800,
    task: str = "binary_diag",
    lead: str = "II",
    folds_train=tuple(range(1,9)), fold_val=(9,), fold_test=(10,)
):
    db_csv, scp_csv, raw_root = _ptbxl_paths(root)
    df = pd.read_csv(db_csv)          # gcsfs enables gs:// here
    scp_df = pd.read_csv(scp_csv, index_col=0)

    df, classes = _ptbxl_labelize(df, scp_df, task=task)

    tr = df[df["strat_fold"].isin(folds_train)]
    va = df[df["strat_fold"].isin(fold_val)]
    te = df[df["strat_fold"].isin(fold_test)]

    tr_ds = PTBXLWindows(tr, raw_root, length=length, lead=lead)
    va_ds = PTBXLWindows(va, raw_root, length=length, lead=lead)
    te_ds = PTBXLWindows(te, raw_root, length=length, lead=lead)

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2, worker_init_fn=_wif)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=2, worker_init_fn=_wif)
    te_loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=2, worker_init_fn=_wif)
    return tr_loader, va_loader, te_loader, {"n_classes": len(set(classes)), "task": task, "lead": lead}
