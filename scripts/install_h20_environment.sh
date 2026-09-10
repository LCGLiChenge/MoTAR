#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ "${1:-}" != --active-env ]]; then
  if ! command -v conda >/dev/null; then
    echo 'Install Miniforge/Conda first, or create a Python 3.10 venv and run this script with --active-env.' >&2
    exit 2
  fi
  conda env create -f environment-h20.yml
  exec conda run --no-capture-output -n motar-h20 bash scripts/install_h20_environment.sh --active-env
fi
python -c 'import sys; assert sys.version_info[:2] == (3,10), sys.version'
python -m pip install torch==2.10.0 torchvision==0.25.0 -c constraints-h20.txt --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-h20.txt -c constraints-h20.txt
python -m pip check
python -m h20.preflight --environment-only
echo 'Environment ready. Activate motar-h20 before downloading assets or launching.'
