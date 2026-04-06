#!/usr/bin/env python3
"""
Data Readiness & Experiment Path Smoke Tests
=============================================
Validates that:
  1. GCS bucket is accessible (public, anonymous)
  2. ECG datasets can be listed / downloaded
  3. All experiment paths execute without runtime errors (using synthetic data)
  4. Results JSON files are written correctly

Usage:
  python test_data_readiness.py                  # run all tests
  python test_data_readiness.py TestGCSAccess     # run only GCS tests
  python test_data_readiness.py TestExperimentPaths  # run only experiment smoke tests
"""
import os
import sys
import json
import shutil
import tempfile
import unittest
from pathlib import Path

# Ensure tinyml/ is on sys.path
_HERE = Path(__file__).resolve().parent
_TINYML = _HERE.parent / "tinyml" if _HERE.name == "scripts" else _HERE
sys.path.insert(0, str(_TINYML))

import torch


# ---------------------------------------------------------------------------
# 1. GCS Bucket Access Tests
# ---------------------------------------------------------------------------
class TestGCSAccess(unittest.TestCase):
    """Verify the public GCS bucket gs://hypertinypw is reachable."""

    def _get_fs(self):
        import gcsfs
        return gcsfs.GCSFileSystem(token="anon")

    def test_gcsfs_installed(self):
        """gcsfs must be importable"""
        import gcsfs  # noqa: F401

    def test_bucket_listable(self):
        """Top-level ls on hypertinypw should return folders"""
        fs = self._get_fs()
        entries = fs.ls("hypertinypw")
        names = [e.split("/")[-1] for e in entries]
        self.assertTrue(len(names) > 0, "Bucket listing returned nothing")
        # At least apnea and ptbxl should be present
        self.assertTrue(
            any("apnea" in n for n in names),
            f"No apnea folder found. Got: {names}",
        )

    def test_apnea_data_exists(self):
        """Apnea-ECG folder should contain .dat/.hea files"""
        fs = self._get_fs()
        files = fs.ls("hypertinypw/apnea-ecg-database-1.0.0")
        dat_files = [f for f in files if f.endswith(".dat")]
        self.assertGreater(len(dat_files), 0, "No .dat files in apnea folder")

    def test_ptbxl_data_exists(self):
        """PTB-XL folder should contain records"""
        fs = self._get_fs()
        entries = fs.ls("hypertinypw/ptbxl")
        self.assertGreater(len(entries), 0, "ptbxl folder is empty")

    def test_mitbih_data_exists(self):
        """MIT-BIH folder should contain records"""
        fs = self._get_fs()
        entries = fs.ls("hypertinypw/mitbih")
        self.assertGreater(len(entries), 0, "mitbih folder is empty")

    def test_anonymous_read_file(self):
        """Should be able to read a small file anonymously"""
        fs = self._get_fs()
        # Try reading the first .hea file from apnea
        files = fs.ls("hypertinypw/apnea-ecg-database-1.0.0")
        hea_files = [f for f in files if f.endswith(".hea")]
        if not hea_files:
            self.skipTest("No .hea files found to test read")
        with fs.open(hea_files[0], "r") as f:
            content = f.read()
        self.assertGreater(len(content), 0, "Read returned empty content")


# ---------------------------------------------------------------------------
# 2. Local Data Readiness (after download)
# ---------------------------------------------------------------------------
class TestLocalDataReadiness(unittest.TestCase):
    """Check that local data directories look correct (skip if not downloaded)."""

    def _root(self, env_var, default_subdir):
        from data_loaders import DATA_BASE
        root = os.environ.get(env_var, f"{DATA_BASE}/{default_subdir}")
        if root.startswith("gs://"):
            self.skipTest(f"{env_var} points to GCS ({root}); set a local path to test")
        if not Path(root).exists():
            self.skipTest(f"{env_var}={root} does not exist locally")
        return Path(root)

    def test_apnea_local(self):
        root = self._root("APNEA_ROOT", "apnea-ecg-database-1.0.0")
        dat = list(root.glob("a*.dat"))
        self.assertGreater(len(dat), 0, f"No a*.dat in {root}")

    def test_ptbxl_local(self):
        root = self._root("PTBXL_ROOT", "ptbxl")
        csvs = list(root.rglob("*.csv"))
        self.assertGreater(len(csvs), 0, f"No CSV files in {root}")

    def test_mitbih_local(self):
        root = self._root("MITDB_ROOT", "mitbih")
        dat = list(root.rglob("*.dat"))
        self.assertGreater(len(dat), 0, f"No .dat files in {root}")

    def test_speech_commands_local(self):
        root = os.environ.get("SPEECH_COMMANDS_ROOT", "./data/speech_commands_v0.02")
        root = Path(root)
        if not root.exists():
            self.skipTest(f"Speech Commands not found at {root}")
        expected = ["yes", "no", "up", "down", "left", "right"]
        found = [d for d in expected if (root / d).is_dir()]
        self.assertGreaterEqual(len(found), 4, f"Only found dirs: {found}")


# ---------------------------------------------------------------------------
# 3. Experiment Path Smoke Tests (synthetic data, 1 epoch)
# ---------------------------------------------------------------------------
class TestExperimentPaths(unittest.TestCase):
    """Run each experiment for 1 epoch on synthetic data to catch runtime errors."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix="tinyml_smoke_")
        # Force synthetic fallback
        os.environ["TINYML_DATA_ROOT"] = "/tmp/_nonexistent_data_"
        os.environ.pop("APNEA_ROOT", None)
        os.environ.pop("PTBXL_ROOT", None)
        os.environ.pop("MITDB_ROOT", None)
        os.environ.pop("SPEECH_COMMANDS_ROOT", None)

        import importlib
        import run_experiments
        importlib.reload(run_experiments)
        run_experiments.setup_paths()
        cls.re = run_experiments

        import argparse
        cls.args = argparse.Namespace(
            output_dir=cls.tmpdir, epochs=1, batch_size=16, cpu=True,
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_ternary_path(self):
        """Ternary experiment runs without crashing"""
        result = self.re.run_ternary_baseline_comparison(self.args)
        self.assertIsNotNone(result)
        out = Path(self.tmpdir) / "ternary_comparison.json"
        self.assertTrue(out.exists(), "ternary_comparison.json not written")

    def test_multi_scale_path(self):
        """Multi-scale experiment runs without crashing"""
        result = self.re.run_multi_scale_validation(self.args)
        # may return None on size-only, but shouldn't crash
        out = Path(self.tmpdir) / "multi_scale_validation.json"
        self.assertTrue(out.exists(), "multi_scale_validation.json not written")

    def test_synthesis_path(self):
        """Synthesis profiling runs without crashing"""
        result = self.re.run_synthesis_profiling(self.args)
        self.assertIsNotNone(result)

    def test_8bit_path(self):
        """8-bit quantization experiment doesn't crash (may return None without data)"""
        try:
            result = self.re.run_8bit_quantization_baseline(self.args)
            # result may be None if dataset unavailable — that's OK
        except Exception as e:
            self.fail(f"8bit experiment crashed: {e}")


# ---------------------------------------------------------------------------
# 4. Download Script Validation
# ---------------------------------------------------------------------------
class TestDownloadScript(unittest.TestCase):
    """Verify download_ecg_data.py is importable and has correct config."""

    def test_bucket_uri(self):
        """download_ecg_data.py should point to gs://hypertinypw"""
        sys.path.insert(0, str(_HERE.parent / "scripts"))
        import download_ecg_data
        self.assertTrue(
            download_ecg_data.GCP_BASE.startswith("gs://hypertinypw"),
            f"GCP_BASE is {download_ecg_data.GCP_BASE}, expected gs://hypertinypw",
        )

    def test_data_loaders_default_bucket(self):
        """data_loaders.py source code should default to gs://hypertinypw"""
        dl_path = _TINYML / "data_loaders.py"
        src = dl_path.read_text()
        self.assertIn("gs://hypertinypw", src,
                       "data_loaders.py does not contain gs://hypertinypw as default")


if __name__ == "__main__":
    unittest.main(verbosity=2)
