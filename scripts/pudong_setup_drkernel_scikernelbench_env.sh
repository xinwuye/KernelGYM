#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-drkernel-scikernelbench}"
KERNELGYM_ROOT="${KERNELGYM_ROOT:-/public/home/xinwuye/KernelGYM}"
SCIKERNELBENCH_ROOT="${SCIKERNELBENCH_ROOT:-/public/home/xinwuye/SciKernelBench}"

if ! command -v conda >/dev/null 2>&1; then
  if [ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda/etc/profile.d/conda.sh"
  elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
  else
    echo "conda not found; source your conda init before running this script" >&2
    exit 1
  fi
fi

if [ ! -d "$KERNELGYM_ROOT" ]; then
  echo "KernelGYM root not found: $KERNELGYM_ROOT" >&2
  exit 1
fi
if [ ! -d "$SCIKERNELBENCH_ROOT" ]; then
  echo "SciKernelBench root not found: $SCIKERNELBENCH_ROOT" >&2
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" python=3.10
fi

conda activate "$ENV_NAME"
python -m pip install --upgrade pip setuptools wheel

# PyTorch first, then project dependencies. Override TORCH_INDEX_URL if Pudong
# needs a site-local or domestic mirror.
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
python -m pip install --index-url "$TORCH_INDEX_URL" torch torchvision torchaudio

python -m pip install \
  "transformers>=4.51.0" \
  "accelerate>=0.34.0" \
  "huggingface_hub>=0.24.0" \
  safetensors \
  sentencepiece \
  protobuf \
  ninja \
  pydantic \
  requests \
  tqdm \
  numpy \
  pandas \
  pyarrow \
  packaging \
  setuptools \
  einops \
  "triton>=3.0.0" \
  python-dotenv \
  openai \
  litellm \
  pydra-config \
  tomli \
  tabulate \
  vllm

python -m pip install --no-deps -e "$SCIKERNELBENCH_ROOT"

python - <<'PY'
import torch
import transformers
import triton
import vllm
import kernelbench
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("triton", triton.__version__)
print("vllm", vllm.__version__)
print("kernelbench", getattr(kernelbench, "__file__", "ok"))
PY
