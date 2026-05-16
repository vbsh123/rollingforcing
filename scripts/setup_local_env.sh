#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3.10}"
VENV_DIR="${VENV_DIR:-.venv}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "$PYTHON_BIN was not found. Rolling Forcing recommends Python 3.10."
  echo "Install python3.10 in WSL or set PYTHON_BIN=/path/to/python before running this script."
  exit 1
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e ../TokenTrim

cat <<'EOF'

Environment created.

Next checks:
  source .venv/bin/activate
  python scripts/check_local_gpu.py

Do not run inference until CUDA is visible and checkpoints are downloaded.
EOF
