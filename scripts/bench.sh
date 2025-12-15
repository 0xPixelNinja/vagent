#!/bin/bash

set -euo pipefail

# Add NVIDIA DLL folders (cuDNN, cuBLAS, etc.) to PATH
# This is required on Windows so Python extensions can locate required DLLs.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/.venv"
NVIDIA_DIR="$VENV_DIR/Lib/site-packages/nvidia"

if [ -d "$NVIDIA_DIR" ]; then
	while IFS= read -r -d '' bin_dir; do
		PATH="$bin_dir:$PATH"
	done < <(find "$NVIDIA_DIR" -maxdepth 2 -type d -name bin -print0)
fi

export PATH

# Enable latency timing logs by default (override by exporting VAGENT_LATENCY beforehand)
: "${VAGENT_LATENCY:=1}"
export VAGENT_LATENCY

# Activate venv and run benchmark
source "$VENV_DIR/Scripts/activate"
python "$PROJECT_DIR/tools/bench_services.py" --stt-from-tts
