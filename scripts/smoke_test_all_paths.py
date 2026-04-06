#!/usr/bin/env python3
"""
Smoke Test: validates every experiment path in < 5 minutes.

Runs each experiment for 1 epoch on synthetic data to prove there are no
runtime errors, import failures, or API mismatches — without needing any
real datasets downloaded.

Usage:
    python scripts/smoke_test_all_paths.py           # from repo root
    cd tinyml && python ../scripts/smoke_test_all_paths.py
"""
import os
import sys
import time
import shutil
import tempfile
import traceback
from pathlib import Path

# ── Setup ───────────────────────────────────────────────────────────────────
TINYML_DIR = Path(__file__).resolve().parent.parent / "tinyml"
sys.path.insert(0, str(TINYML_DIR))
os.chdir(str(TINYML_DIR))

# Force synthetic fallback for all datasets
os.environ["TINYML_DATA_ROOT"] = "/tmp/_smoke_test_no_data_"
os.environ.pop("APNEA_ROOT", None)
os.environ.pop("PTBXL_ROOT", None)
os.environ.pop("MITDB_ROOT", None)
os.environ.pop("SPEECH_COMMANDS_ROOT", None)

import argparse

TMPDIR = tempfile.mkdtemp(prefix="tinyml_smoke_")
ARGS = argparse.Namespace(output_dir=TMPDIR, epochs=1, batch_size=16, cpu=True)

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

results = {}


def _safe_print(*args, **kwargs):
    """Print that recovers if sys.stdout was replaced by a closed TeeLogger."""
    try:
        print(*args, **kwargs)
    except ValueError:
        sys.stdout = sys.__stdout__
        print(*args, **kwargs)


def run_test(name, func):
    _safe_print(f"\n{'=' * 60}")
    _safe_print(f"  {name}")
    _safe_print(f"{'=' * 60}")
    t0 = time.time()
    try:
        ret = func()
        elapsed = time.time() - t0
        tag = f"{GREEN}PASS{RESET}"
        results[name] = ("PASS", elapsed)
        _safe_print(f"\n  [{tag}] {name} ({elapsed:.1f}s)")
        return ret
    except Exception as e:
        elapsed = time.time() - t0
        tag = f"{RED}FAIL{RESET}"
        results[name] = ("FAIL", elapsed, str(e))
        _safe_print(f"\n  [{tag}] {name}: {e}")
        traceback.print_exc(file=sys.__stdout__)
        return None


# ── Tests ───────────────────────────────────────────────────────────────────
def test_imports():
    """Import all core modules and check expected symbols."""
    from models import HyperTinyPW, safe_build_model
    from ternary_baseline import TernaryQuantizer
    from datasets import create_synthetic_ecg_data
    from synthesis_profiler import SynthesisProfiler
    import experiments
    return True


def test_unit_tests():
    """Run the existing unittest suite."""
    import unittest
    original_stdout = sys.__stdout__
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName("test_experiments")
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)
    # Restore stdout in case TeeLogger hijacked it
    sys.stdout = original_stdout
    if not result.wasSuccessful():
        fails = [str(f[0]) for f in result.failures + result.errors]
        raise RuntimeError(f"{len(fails)} test(s) failed: {fails}")
    return True


def test_ternary():
    import run_experiments
    run_experiments.setup_paths()
    return run_experiments.run_ternary_baseline_comparison(ARGS)


def test_multi_scale():
    import run_experiments
    run_experiments.setup_paths()
    return run_experiments.run_multi_scale_validation(ARGS)


def test_synthesis():
    import run_experiments
    run_experiments.setup_paths()
    return run_experiments.run_synthesis_profiling(ARGS)


def test_8bit():
    import run_experiments
    run_experiments.setup_paths()
    # 8bit may return None (no data) — that's fine, just shouldn't crash
    return run_experiments.run_8bit_quantization_baseline(ARGS)


def test_gcs_bucket_config():
    """Verify bucket URIs point to gs://hypertinypw."""
    src_dl = (TINYML_DIR / "data_loaders.py").read_text()
    assert "gs://hypertinypw" in src_dl, "data_loaders.py missing gs://hypertinypw"

    sys.path.insert(0, str(TINYML_DIR.parent / "scripts"))
    import download_ecg_data
    assert download_ecg_data.GCP_BASE == "gs://hypertinypw", \
        f"download_ecg_data.GCP_BASE = {download_ecg_data.GCP_BASE}"
    return True


def test_results_written():
    """Check that experiments wrote their JSON outputs."""
    expected = ["ternary_comparison.json", "multi_scale_validation.json"]
    missing = [f for f in expected if not (Path(TMPDIR) / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing result files: {missing}")
    # Validate JSON is parseable
    for f in expected:
        fp = Path(TMPDIR) / f
        if fp.exists():
            data = __import__("json").loads(fp.read_text())
            assert isinstance(data, dict), f"{f} is not a dict"
    return True


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  HyperTinyPW - Smoke Test (All Experiment Paths)")
    print("=" * 60)
    print(f"  Output dir: {TMPDIR}")
    print(f"  Data: synthetic only (no downloads)")
    print()

    t_start = time.time()

    run_test("1. Core imports", test_imports)
    run_test("2. Unit test suite", test_unit_tests)
    run_test("3. GCS bucket config", test_gcs_bucket_config)
    run_test("4. Ternary experiment", test_ternary)
    run_test("5. Multi-scale experiment", test_multi_scale)
    run_test("6. Synthesis profiling", test_synthesis)
    run_test("7. 8-bit quantization", test_8bit)
    run_test("8. Result files written", test_results_written)

    total = time.time() - t_start

    # Ensure stdout is clean for summary
    sys.stdout = sys.__stdout__

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    passes = sum(1 for v in results.values() if v[0] == "PASS")
    fails = sum(1 for v in results.values() if v[0] == "FAIL")
    for name, info in results.items():
        tag = f"{GREEN}PASS{RESET}" if info[0] == "PASS" else f"{RED}FAIL{RESET}"
        t = info[1]
        extra = f"  ({info[2]})" if len(info) > 2 else ""
        print(f"  [{tag}] {name} ({t:.1f}s){extra}")

    print(f"\n  {passes}/{passes + fails} passed in {total:.0f}s")
    print("=" * 60)

    # Cleanup
    shutil.rmtree(TMPDIR, ignore_errors=True)

    sys.exit(0 if fails == 0 else 1)
