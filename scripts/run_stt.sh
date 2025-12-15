#!/bin/bash
set -euo pipefail

# Add NVIDIA DLLs to PATH (Windows support)
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

source "$VENV_DIR/Scripts/activate"
python -m vagent.stt_server
