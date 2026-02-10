#!/bin/bash
# Complete rebuttal experiments runner - clean and comprehensive

set -e  # Exit on error

echo "================================================================================"
echo "COMPLETE REBUTTAL EXPERIMENTS - CLEAN RUN"
echo "================================================================================"
echo ""

# Check environment
echo "1. Checking environment..."
cd ~/tinyml-gen

# Clean previous results
echo ""
echo "2. Cleaning previous results..."
if [ -d "tinyml/rebuttal_results" ]; then
    echo "   Backing up previous results to rebuttal_results_backup_$(date +%Y%m%d_%H%M%S)"
    mv tinyml/rebuttal_results tinyml/rebuttal_results_backup_$(date +%Y%m%d_%H%M%S)
fi

# Pull latest code
echo ""
echo "3. Pulling latest code..."
git pull origin main

# Check datasets
echo ""
echo "4. Checking datasets..."
echo "   Apnea: $([ -f data/apnea/a01.dat ] && echo '✓ Found' || echo '✗ Not found')"
echo "   PTB-XL: $([ -f data/ptbxl/raw/ptbxl_database.csv ] && echo '✓ Found' || echo '✗ Not found')"
echo "   MITBIH: $([ -f data/mitbih/100.dat ] && echo '✓ Found' || echo '✗ Not found')"

# Set environment variables
echo ""
echo "5. Setting environment variables..."
export APNEA_ROOT="$(pwd)/data/apnea"
export PTBXL_ROOT="$(pwd)/data/ptbxl"
export MITDB_ROOT="$(pwd)/data/mitbih"
echo "   APNEA_ROOT=$APNEA_ROOT"
echo "   PTBXL_ROOT=$PTBXL_ROOT"
echo "   MITDB_ROOT=$MITDB_ROOT"

# Run complete experiments
echo ""
echo "6. Running COMPLETE experiment suite..."
echo "   - Keyword Spotting (audio validation)"
echo "   - Ternary Comparison (ECG with error analysis)"
echo "   - Multi-Scale Validation (150K-500K params)"
echo "   - Synthesis Profiling (boot-time metrics)"
echo ""
echo "   Expected runtime: 2-3 hours"
echo "   Progress will be logged to: tinyml/rebuttal_results/experiment_full.log"
echo ""
read -p "Press Enter to start (or Ctrl+C to cancel)..."

cd tinyml
python run_rebuttal_experiments.py --experiments keyword_spotting,ternary,multi_scale,synthesis

# Analyze results
echo ""
echo "================================================================================"
echo "RESULTS SUMMARY"
echo "================================================================================"
echo ""

if [ -f "rebuttal_results/rebuttal_summary.json" ]; then
    echo "✓ Experiments completed successfully!"
    echo ""
    echo "Key Results:"
    echo "------------"
    
    # Keyword spotting
    if [ -f "rebuttal_results/keyword_spotting_results.json" ]; then
        KW_ACC=$(python3 -c "import json; print(json.load(open('rebuttal_results/keyword_spotting_results.json'))['test_accuracy'])")
        echo "  Keyword Spotting: ${KW_ACC}% test accuracy"
    fi
    
    # Ternary comparison
    if [ -f "rebuttal_results/ternary_comparison.json" ]; then
        DATASET=$(python3 -c "import json; d=json.load(open('rebuttal_results/ternary_comparison.json')); print(d.get('dataset', 'Unknown'))" 2>/dev/null || echo "N/A")
        HYPER_ACC=$(python3 -c "import json; d=json.load(open('rebuttal_results/ternary_comparison.json')); print(f\"{d['hypertiny']['balanced_acc']:.2f}\")" 2>/dev/null || echo "N/A")
        TERN_ACC=$(python3 -c "import json; d=json.load(open('rebuttal_results/ternary_comparison.json')); print(f\"{d['ternary']['balanced_acc']:.2f}\")" 2>/dev/null || echo "N/A")
        
        echo "  Ternary Comparison (Dataset: ${DATASET}):"
        echo "    HyperTinyPW: ${HYPER_ACC}% balanced accuracy"
        echo "    Ternary:     ${TERN_ACC}% balanced accuracy"
        
        if [ "$HYPER_ACC" != "N/A" ] && [ "$TERN_ACC" != "N/A" ]; then
            DIFF=$(python3 -c "print(f'{float('$HYPER_ACC') - float('$TERN_ACC'):.2f}')")
            if (( $(echo "$DIFF > 0" | bc -l) )); then
                echo "    → HyperTinyPW wins by +${DIFF}% ✓"
            else
                echo "    → Warning: Ternary winning (check results)"
            fi
        fi
    fi
    
    # Multi-scale
    if [ -f "rebuttal_results/multi_scale_validation.json" ]; then
        echo "  Multi-Scale Validation:"
        python3 -c "
import json
data = json.load(open('rebuttal_results/multi_scale_validation.json'))
for config in data['configs']:
    print(f\"    {config['name']}: {config['test_accuracy']:.2f}% ({config['params']:,} params)\")
" 2>/dev/null || echo "    (see multi_scale_validation.json)"
    fi
    
    echo ""
    echo "Full results in: tinyml/rebuttal_results/"
    echo "  - rebuttal_summary.json (high-level overview)"
    echo "  - experiment_full.log (detailed training logs)"
    echo "  - keyword_spotting_results.json"
    echo "  - ternary_comparison.json"
    echo "  - multi_scale_validation.json"
    echo "  - synthesis_profiling.json"
    
else
    echo "✗ Experiments did not complete successfully"
    echo "Check: tinyml/rebuttal_results/experiment_full.log for errors"
    exit 1
fi

echo ""
echo "================================================================================"
echo "Next Steps:"
echo "================================================================================"
echo "1. Review results: cat tinyml/rebuttal_results/rebuttal_summary.json | python -m json.tool"
echo "2. Check error analysis: grep -A 10 'Confusion Matrix' tinyml/rebuttal_results/experiment_full.log"
echo "3. Verify balanced accuracy: grep 'Balanced Accuracy' tinyml/rebuttal_results/experiment_full.log"
echo "4. Write rebuttal using results from rebuttal_results/*.json"
echo ""
