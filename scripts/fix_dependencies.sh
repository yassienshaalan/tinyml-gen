#!/bin/bash
# Quick fix script for pandas/wfdb compatibility issue
# Run this on Linux machine if you get StringArray error

echo "Fixing pandas/wfdb compatibility issue..."

# Update wfdb to version compatible with pandas 2.x
pip install --upgrade "wfdb>=4.1.0"

# Install soundfile for audio loading (avoids torchcodec dependency)
pip install --upgrade "soundfile>=0.12.0"

# Verify installation
python3 -c "import wfdb; print(f'wfdb version: {wfdb.__version__}')"
python3 -c "import pandas; print(f'pandas version: {pandas.__version__}')"
python3 -c "import soundfile; print(f'soundfile version: {soundfile.__version__}')"

echo ""
echo "✓ Dependencies updated"
echo ""
echo "Now you can run:"
echo "  cd tinyml"
echo "  python3 run_experiments.py --experiments all --epochs 20"
