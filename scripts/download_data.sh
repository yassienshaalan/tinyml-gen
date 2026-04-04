#!/bin/bash
# Download datasets for HyperTinyPW experiments
# Usage: ./download_data.sh [all|speech_commands|ecg|apnea|ptbxl|mitdb]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$REPO_DIR/data"

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
    echo "[OK] Speech Commands downloaded to: $SPEECH_COMMANDS_ROOT"
}

download_ecg() {
    echo "Downloading ECG datasets from GCP bucket..."
    cd "$SCRIPT_DIR"
    python3 download_ecg_data.py --dataset all --target-dir "$DATA_DIR"
    echo "[OK] ECG datasets downloaded to: $DATA_DIR"
}

download_apnea() {
    cd "$SCRIPT_DIR"
    python3 download_ecg_data.py --dataset apnea --target-dir "$DATA_DIR"
}

download_ptbxl() {
    cd "$SCRIPT_DIR"
    python3 download_ecg_data.py --dataset ptbxl --target-dir "$DATA_DIR"
}

download_mitdb() {
    cd "$SCRIPT_DIR"
    python3 download_ecg_data.py --dataset mitbih --target-dir "$DATA_DIR"
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
echo "[OK] Data download complete!"
echo "======================================"
echo ""
echo "Set environment variables:"
echo "  export TINYML_DATA_ROOT=\"$DATA_DIR\""
echo "  export SPEECH_COMMANDS_ROOT=\"$DATA_DIR/speech_commands_v0.02\""
echo ""
echo "Then run experiments:"
echo "  cd tinyml"
echo "  python run_experiments.py --experiments all"
echo ""
