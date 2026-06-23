#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-drkernel-scikernelbench}"
KERNELGYM_ROOT="${KERNELGYM_ROOT:-/public/home/xinwuye/KernelGYM}"
SCIKERNELBENCH_ROOT="${SCIKERNELBENCH_ROOT:-/public/home/xinwuye/SciKernelBench}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"
export PIP_DISABLE_PIP_VERSION_CHECK=1

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

configure_nvidia_python_libs() {
  local lib_paths
  lib_paths="$(python - <<'PY'
import site
from pathlib import Path

paths = []
root = Path(site.getsitepackages()[0]) / "nvidia"
if root.is_dir():
    for path in sorted(root.glob("**/lib*")):
        if path.is_dir():
            paths.append(str(path))
print(":".join(paths))
PY
)"
  if [ -z "$lib_paths" ]; then
    echo "No Python NVIDIA library paths found under the active conda env" >&2
    exit 1
  fi
  export LD_LIBRARY_PATH="${lib_paths}:${LD_LIBRARY_PATH:-}"
}

python -m pip install --upgrade pip wheel
python -m pip install "setuptools<82"

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
  tabulate

python -m pip install \
  "vllm==0.23.0" \
  "transformers==5.12.1" \
  "tokenizers==0.22.2" \
  "opentelemetry-api==1.42.1" \
  "opentelemetry-sdk==1.42.1" \
  "opentelemetry-exporter-otlp==1.42.1" \
  "fastapi==0.136.3"
python -m pip install --force-reinstall --no-cache-dir "PyYAML==6.0.3"

python -m pip install --no-deps -e "$SCIKERNELBENCH_ROOT"

configure_nvidia_python_libs

python - <<'PY'
import torch
import transformers
import triton
import vllm
from vllm import LLM
import kernelbench
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("triton", triton.__version__)
print("vllm", vllm.__version__)
print("vllm_llm", LLM)
print("kernelbench", getattr(kernelbench, "__file__", "ok"))
PY
