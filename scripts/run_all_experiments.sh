#!/bin/bash
# ============================================================================
# HyperTinyPW — Complete Artifact Evaluation Script
# ============================================================================
#
# One-command pipeline: download data → verify setup → run experiments → report
#
# Usage:
#   ./run_all_experiments.sh                       # all experiments, 20 epochs
#   ./run_all_experiments.sh --experiments ternary  # single experiment
#   ./run_all_experiments.sh --epochs 5             # fewer epochs (faster)
#   ./run_all_experiments.sh --skip-download        # data already local
#   ./run_all_experiments.sh --cpu                  # force CPU
#
# Datasets:
#   ECG datasets (apnea, ptbxl, mitbih) are downloaded from a GCP bucket
#   using gcsfs (pure Python) or gsutil (Cloud SDK).
#   Speech Commands v0.02 (~2 GB) is downloaded separately via wget.
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$REPO_DIR/data"
TINYML_DIR="$REPO_DIR/tinyml"

# ── Parse arguments ─────────────────────────────────────────────────────────
EXPERIMENTS="all"
EPOCHS=20
BATCH_SIZE=64
SKIP_DOWNLOAD=false
EXTRA_FLAGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --experiments)  EXPERIMENTS="$2"; shift 2 ;;
        --epochs)       EPOCHS="$2"; shift 2 ;;
        --batch-size)   BATCH_SIZE="$2"; shift 2 ;;
        --skip-download) SKIP_DOWNLOAD=true; shift ;;
        --cpu)          EXTRA_FLAGS="$EXTRA_FLAGS --cpu"; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --experiments LIST  Comma-separated: keyword_spotting,ternary,8bit,"
            echo "                      multi_scale,synthesis,kws_perclass  (default: all)"
            echo "  --epochs N          Training epochs (default: 20)"
            echo "  --batch-size N      Batch size (default: 64)"
            echo "  --skip-download     Skip dataset download (use existing local data)"
            echo "  --cpu               Force CPU (disable CUDA)"
            echo "  -h, --help          Show this help"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "========================================================================"
echo "  HyperTinyPW: Complete Artifact Evaluation"
echo "========================================================================"
echo ""
echo "  Experiments : $EXPERIMENTS"
echo "  Epochs      : $EPOCHS"
echo "  Batch size  : $BATCH_SIZE"
echo "  Data dir    : $DATA_DIR"
echo ""

# ── Step 1: Download datasets ──────────────────────────────────────────────
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo "[1/4] Downloading datasets..."
    echo "--------------------------------------------------------------"

    # ECG datasets from GCP bucket
    cd "$SCRIPT_DIR"
    python3 download_ecg_data.py --dataset all --target-dir "$DATA_DIR" || \
        echo "  [WARN] ECG download had issues; experiments will use synthetic fallback"

    # Speech Commands (only if keyword_spotting is in the experiment list)
    if [[ "$EXPERIMENTS" == "all" || "$EXPERIMENTS" == *"keyword_spotting"* ]]; then
        mkdir -p "$DATA_DIR"
        cd "$DATA_DIR"
        if [ ! -d "speech_commands_v0.02/yes" ]; then
            echo ""
            echo "  Downloading Speech Commands v0.02 (~2 GB)..."
            if [ ! -f "speech_commands_v0.02.tar.gz" ]; then
                wget -q --show-progress \
                    http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz \
                    || echo "  [WARN] wget failed; keyword_spotting will be skipped"
            fi
            if [ -f "speech_commands_v0.02.tar.gz" ]; then
                mkdir -p speech_commands_v0.02
                tar -xzf speech_commands_v0.02.tar.gz -C speech_commands_v0.02
                echo "  [OK] Speech Commands extracted"
            fi
        else
            echo "  [OK] Speech Commands already present"
        fi
    fi
    echo ""
else
    echo "[1/4] Skipping download (--skip-download)"
    echo ""
fi

# ── Step 2: Set environment variables ──────────────────────────────────────
echo "[2/4] Setting environment variables..."
export TINYML_DATA_ROOT="$DATA_DIR"
export APNEA_ROOT="$DATA_DIR/apnea-ecg-database-1.0.0"
export PTBXL_ROOT="$DATA_DIR/ptbxl"
export MITDB_ROOT="$DATA_DIR/mitbih"
export SPEECH_COMMANDS_ROOT="$DATA_DIR/speech_commands_v0.02"

echo "  TINYML_DATA_ROOT   = $TINYML_DATA_ROOT"
echo "  APNEA_ROOT         = $APNEA_ROOT"
echo "  PTBXL_ROOT         = $PTBXL_ROOT"
echo "  MITDB_ROOT         = $MITDB_ROOT"
echo "  SPEECH_COMMANDS_ROOT = $SPEECH_COMMANDS_ROOT"
echo ""

# ── Step 3: Verify setup ──────────────────────────────────────────────────
echo "[3/4] Verifying setup..."
cd "$TINYML_DIR"
python3 -m unittest test_experiments -v 2>&1 | tail -5
echo ""

# ── Step 4: Run experiments ────────────────────────────────────────────────
echo "[4/4] Running experiments: $EXPERIMENTS  (epochs=$EPOCHS)"
echo "--------------------------------------------------------------"
echo ""

python3 run_experiments.py \
    --experiments "$EXPERIMENTS" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --output-dir ./results \
    $EXTRA_FLAGS

echo ""
echo "========================================================================"
echo "  COMPLETE"
echo "========================================================================"
echo ""

if [ -d "$TINYML_DIR/results" ]; then
    echo "Result files:"
    ls -lh "$TINYML_DIR/results/"*.json 2>/dev/null || echo "  (no JSON results)"
    echo ""
    echo "Summary:"
    if [ -f "$TINYML_DIR/results/experiment_summary.json" ]; then
        python3 -m json.tool "$TINYML_DIR/results/experiment_summary.json" 2>/dev/null \
            || cat "$TINYML_DIR/results/experiment_summary.json"
    fi
fi
echo ""
