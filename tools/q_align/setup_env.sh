#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ENV_NAME="${ENV_NAME:-qalign-artifacts}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Install Miniconda/Anaconda first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda env ${ENV_NAME} already exists."
else
  conda create -n "${ENV_NAME}" python=3.10 pip -y
fi

conda activate "${ENV_NAME}"
python -m pip install --upgrade pip
python -m pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r tools/q_align/requirements.txt

python - << 'PY'
import torch
import transformers
import peft
import decord
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
print("transformers", transformers.__version__)
print("peft", peft.__version__)
print("decord", decord.__version__)
PY

echo "Environment ready. Activate with: conda activate ${ENV_NAME}"
