#!/bin/bash
# start_videoanalyse_emb.sh

VENV_DIR="$HOME/videoanalyse_venv"
SCRIPT_DIR="$(cd "$(dirname "$0")"; pwd)"

source "$VENV_DIR/bin/activate"
python "$SCRIPT_DIR/videoanalyse_emb.py"
deactivate
