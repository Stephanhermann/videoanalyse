#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="$HOME/videoanalyse_venv/bin/python"
GUI_SCRIPT="$SCRIPT_DIR/videoanalyse_mac_gui.py"

if [ ! -x "$VENV_PYTHON" ]; then
  echo "Fehler: Python in der virtuellen Umgebung nicht gefunden:"
  echo "$VENV_PYTHON"
  read -p "Enter zum Beenden..."
  exit 1
fi

if [ ! -f "$GUI_SCRIPT" ]; then
  echo "Fehler: GUI-Script nicht gefunden:"
  echo "$GUI_SCRIPT"
  read -p "Enter zum Beenden..."
  exit 1
fi

exec "$VENV_PYTHON" "$GUI_SCRIPT"
