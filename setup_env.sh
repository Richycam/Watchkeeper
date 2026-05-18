#!/usr/bin/env bash
set -euo pipefail
VENV_DIR="venv"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet
pip install -r requirements.txt
echo ""
echo "[OK] Environment ready."
echo "[*]  Activate : source $VENV_DIR/bin/activate"
echo "[*]  Run      : python run.py"
