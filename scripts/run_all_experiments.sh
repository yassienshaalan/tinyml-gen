#!/bin/bash
# Complete Experiment Pipeline for HyperTinyPW
# Runs all rebuttal experiments with proper logging

set -e  # Exit on error

echo "======================================"
echo "HyperTinyPW: Complete Experiment Suite"
echo "======================================"
echo ""

# Navigate to tinyml directory
cd "$(dirname "$0")/../tinyml"

# Test setup
echo "[1/5] Testing setup..."
python test_rebuttal_modules.py
echo "✅ Setup verified"
echo ""

# Run all experiments
echo "[2/5] Running all experiments..."
echo "This will take 10-60 minutes depending on data availability"
echo ""

python run_rebuttal_experiments.py --experiments all --epochs 20

echo ""
echo "[3/5] Generating summary..."
echo ""

# Check results
if [ -d "rebuttal_results" ]; then
    echo "✅ Results saved to: tinyml/rebuttal_results/"
    echo ""
    echo "Generated files:"
    ls -lh rebuttal_results/*.json
    echo ""
fi

echo "[4/5] Experiment log:"
if [ -f "rebuttal_results/experiment_full.log" ]; then
    tail -n 20 rebuttal_results/experiment_full.log
fi

echo ""
echo "======================================"
echo "[5/5] ✅ COMPLETE!"
echo "======================================"
echo ""
echo "Next steps:"
echo "  1. Review results: cd tinyml/rebuttal_results"
echo "  2. Read report: cat ../docs/COMPLETE_EXPERIMENTAL_REPORT.md"
echo "  3. Check summary: cat rebuttal_results/rebuttal_summary.json"
echo ""
