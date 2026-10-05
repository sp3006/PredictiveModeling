#!/usr/bin/env bash
set -euo pipefail

if command -v dnf >/dev/null 2>&1; then
  sudo dnf install -y python3 python3-pip git
elif command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y python3 python3-venv python3-pip git
else
  echo "Unsupported package manager. Install Python 3.10+, pip, venv, and git manually." >&2
  exit 1
fi

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
pip install -e .

python - <<'PY'
import boto3, pandas, sklearn, catboost
print("Environment ready")
print("boto3", boto3.__version__)
print("pandas", pandas.__version__)
print("sklearn", sklearn.__version__)
print("catboost", catboost.__version__)
PY

echo
echo "Next: source .venv/bin/activate"
echo "Jupyter: jupyter lab --no-browser --ip=127.0.0.1 --port=8888"
