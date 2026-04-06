import os, sys
import gcsfs

APNEA_ROOT = os.environ.get("APNEA_ROOT", "")
print("APNEA_ROOT =", APNEA_ROOT)
assert APNEA_ROOT.startswith("gs://"), "APNEA_ROOT must start with gs://"

fs = gcsfs.GCSFileSystem()  # uses ADC (service account or 'gcloud auth application-default')

# Recursively find .hea files under APNEA_ROOT
hea_files = [p for p in fs.find(APNEA_ROOT.rstrip("/")) if p.endswith(".hea")]
print("Found .hea files:", len(hea_files))
print("First few .hea:", hea_files[:5])

def has_triplet(p):
    rid = p.rsplit("/", 1)[-1].split(".")[0]
    parent = p.rsplit("/", 1)[0]
    return fs.exists(f"{parent}/{rid}.dat") and fs.exists(f"{parent}/{rid}.apn"), rid, parent

ok = 0
for p in hea_files[:200]:
    ok_triplet, rid, parent = has_triplet(p)
    if ok_triplet:
        print("OK triplet:", rid, "in", parent)
        ok += 1
        break

if ok == 0:
    print("No complete .hea/.dat/.apn triplets found; wrong path or files not collocated.")


import os
from data_loaders import _list_trainable_records, load_apnea_ecg_loaders_impl

os.environ["APNEA_ROOT"] = "gs://hypertinypw/apnea-ecg-database-1.0.0"
root = os.environ["APNEA_ROOT"]

recs = _list_trainable_records(root)
print("Apnea usable records:", len(recs), recs[:10])
dl_tr, dl_va, dl_te = load_apnea_ecg_loaders_impl(root, batch_size=8, length=1800, stride=None, verbose=True)
print("batches:", len(dl_tr), len(dl_va), len(dl_te))

