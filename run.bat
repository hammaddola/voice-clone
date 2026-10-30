@echo off
cd /d "%~dp0"
if not exist .venv (
  python -m venv .venv
  .venv\Scripts\pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
  .venv\Scripts\pip install -r requirements.txt
)
rem Setting this means YOU accept the Coqui Public Model License (non-commercial) for XTTS-v2 weights.
set COQUI_TOS_AGREED=1
.venv\Scripts\python app.py %*
