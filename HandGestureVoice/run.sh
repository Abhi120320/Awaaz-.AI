#!/bin/bash
set -e
cd "$(dirname "$0")"

export GLOG_minloglevel=3
export TF_CPP_MIN_LOG_LEVEL=3

PYTHON3="${PYTHON3:-python3}"
if ! command -v "$PYTHON3" >/dev/null 2>&1; then
  PYTHON3="/usr/bin/python3"
fi

if [[ ! -d ".venv" ]]; then
  echo "Creating virtual environment…"
  "$PYTHON3" -m venv .venv
fi

if [[ ! -f ".venv/lib/python3.9/site-packages/mediapipe/__init__.py" ]] \
   && [[ ! -f ".venv/lib/python3.12/site-packages/mediapipe/__init__.py" ]]; then
  echo "Installing dependencies…"
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/pip install --no-compile -r requirements.txt
fi

if [[ ! -f gesture_recognizer.task ]]; then
  echo "Downloading gesture model (~7MB)…"
  curl -L -o gesture_recognizer.task \
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
fi

exec .venv/bin/python main.py
