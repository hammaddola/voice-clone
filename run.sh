#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  # CPU wheels; for an NVIDIA GPU use --index-url https://download.pytorch.org/whl/cu128
  .venv/bin/pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
  .venv/bin/pip install -r requirements.txt
fi
# Setting this means YOU accept the Coqui Public Model License (non-commercial) for XTTS-v2 weights.
export COQUI_TOS_AGREED=1
exec .venv/bin/python app.py "$@"
