#!/bin/bash
# Download datasets for HyperTinyPW experiments
# Usage: ./download_data.sh [all|speech_commands|ecg|apnea|ptbxl|mitdb]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$(dirname "$SCRIPT_DIR")/data"

mkdir -p "$DATA_DIR"

download_speech_commands() {
    echo "Downloading Google Speech Commands v0.02 (~2GB)..."
    cd "$DATA_DIR"
    
    if [ ! -f "speech_commands_v0.02.tar.gz" ]; then
        wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
    fi
    
    if [ ! -d "speech_commands_v0.02" ]; then
        mkdir -p speech_commands_v0.02
        tar -xzf speech_commands_v0.02.tar.gz -C speech_commands_v0.02
    fi
    
    export SPEECH_COMMANDS_ROOT="$DATA_DIR/speech_commands_v0.02"
    echo "✅ Speech Commands downloaded to: $SPEECH_COMMANDS_ROOT"
}

download_ecg() {
    echo "Downloading ECG datasets using download_ecg_data.py..."
    cd "$SCRIPT_DIR"
    
    # Download all ECG datasets
    python download_ecg_data.py --dataset all --target-dir "$DATA_DIR"
    
    echo "✅ ECG datasets downloaded to: $DATA_DIR"
}

download_apnea() {
    echo "Downloading Apnea-ECG dataset..."
    cd "$SCRIPT_DIR"
    python download_ecg_data.py --dataset apnea --target-dir "$DATA_DIR"
    echo "✅ Apnea-ECG downloaded"
}

download_ptbxl() {
    echo "Downloading PTB-XL dataset..."
    cd "$SCRIPT_DIR"
    python download_ecg_data.py --dataset ptbxl --target-dir "$DATA_DIR"
    echo "✅ PTB-XL downloaded"
}

download_mitdb() {
    echo "Downloading MIT-BIH dataset..."
    cd "$SCRIPT_DIR"
    python download_ecg_data.py --dataset mitdb --target-dir "$DATA_DIR"
    echo "✅ MIT-BIH downloaded"
}

# Parse command line argument
case "${1:-all}" in
    all)
        download_speech_commands
        download_ecg
        ;;
    speech_commands)
        download_speech_commands
        ;;
    ecg)
        download_ecg
        ;;
    apnea)
        download_apnea
        ;;
    ptbxl)
        download_ptbxl
        ;;
    mitdb)
        download_mitdb
        ;;
    *)
        echo "Usage: $0 [all|speech_commands|ecg|apnea|ptbxl|mitdb]"
        exit 1
        ;;
esac

echo ""
echo "======================================"
echo "✅ Data download complete!"
echo "======================================"
echo ""
echo "Set environment variables:"
echo "  export SPEECH_COMMANDS_ROOT=\"$DATA_DIR/speech_commands_v0.02\""
echo "  export APNEA_ROOT=\"$DATA_DIR/apnea\""
echo "  export PTBXL_ROOT=\"$DATA_DIR/ptbxl\""
echo "  export MITDB_ROOT=\"$DATA_DIR/mitdb\""
echo ""
