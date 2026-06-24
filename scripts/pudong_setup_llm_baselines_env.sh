#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-kernel-baselines-scikernelbench}"
SOURCE_ENV="${SOURCE_ENV:-drkernel-scikernelbench}"
KERNELGYM_ROOT="${KERNELGYM_ROOT:-/public/home/xinwuye/KernelGYM}"
SCIKERNELBENCH_ROOT="${SCIKERNELBENCH_ROOT:-/public/home/xinwuye/SciKernelBench}"
DICE_ROOT="${DICE_ROOT:-/public/home/xinwuye/DICE}"
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

for path in "$KERNELGYM_ROOT" "$SCIKERNELBENCH_ROOT" "$DICE_ROOT"; do
  if [ ! -d "$path" ]; then
    echo "Required path not found: $path" >&2
    exit 1
  fi
done

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  if ! conda env list | awk '{print $1}' | grep -qx "$SOURCE_ENV"; then
    echo "Source env not found: $SOURCE_ENV. Create it first or set SOURCE_ENV to an existing env." >&2
    exit 1
  fi
  conda create -y --clone "$SOURCE_ENV" -n "$ENV_NAME"
fi

conda activate "$ENV_NAME"
python -m pip install --no-deps -e "$SCIKERNELBENCH_ROOT"

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

configure_nvidia_python_libs

PYTHONPATH="$KERNELGYM_ROOT/scripts:${SCIKERNELBENCH_ROOT}/src:${DICE_ROOT}/evaluation:${PYTHONPATH:-}" python - <<'PY'
import sys
import torch
import transformers
import vllm
import kernelbench
from scripts.sdar_utils import block_diffusion_generate
import llm_baselines_scikernelbench

print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("vllm", vllm.__version__)
print("kernelbench", getattr(kernelbench, "__file__", "ok"))
print("dice_sdar", block_diffusion_generate)
print("baseline_harness", llm_baselines_scikernelbench.__file__)
PY
