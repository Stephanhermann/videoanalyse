#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIDEO_ENV="$HOME/videoanalyse_venv/bin/python3"
MOVIEMETA_DIR="$HOME/MoviemetaDb"

if [ ! -x "$VIDEO_ENV" ]; then
  echo "Error: Python env not found at $VIDEO_ENV"
  echo "Create it first: python3 -m venv ~/videoanalyse_venv"
  exit 1
fi

if [ ! -d "$MOVIEMETA_DIR" ]; then
  echo "Error: MoviemetaDb repo not found at $MOVIEMETA_DIR"
  echo "Clone first: git clone https://github.com/Stephanhermann/MoviemetaDb.git ~/MoviemetaDb"
  exit 1
fi

# Load shared env from videoanalyse project.
if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a
  source "$SCRIPT_DIR/.env"
  set +a
fi

export MOVIEMETADB_DATABASE_URL="${MOVIEMETADB_DATABASE_URL:-$MOVIEMETA_DIR/moviemetadb.db}"
export MOVIEMETADB_API_KEY="${MOVIEMETADB_API_KEY:-}"

cd "$MOVIEMETA_DIR"

# Ensure required runtime dependencies are available.
"$VIDEO_ENV" -m pip install -q fastapi "uvicorn[standard]" sqlalchemy

echo "Starting MoviemetaDb API on http://127.0.0.1:8001"
export PYTHONPATH="$MOVIEMETA_DIR:${PYTHONPATH:-}"
exec "$VIDEO_ENV" -m uvicorn moviemetadb.web:app --host 127.0.0.1 --port 8001
