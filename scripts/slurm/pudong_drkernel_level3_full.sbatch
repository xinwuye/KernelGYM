#!/usr/bin/env bash
#SBATCH --job-name=drk-l3-full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=18:00:00
#SBATCH --array=1-50%32
#SBATCH --output=/public/home/xinwuye/drkernel_scikernelbench_runs/logs/full_%A_%a.out
#SBATCH --error=/public/home/xinwuye/drkernel_scikernelbench_runs/logs/full_%A_%a.err

set -euo pipefail

ENV_NAME="${ENV_NAME:-drkernel-scikernelbench}"
KERNELGYM_ROOT="${KERNELGYM_ROOT:-/public/home/xinwuye/KernelGYM}"
SCIKERNELBENCH_ROOT="${SCIKERNELBENCH_ROOT:-/public/home/xinwuye/SciKernelBench}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/public/home/xinwuye/drkernel_scikernelbench_runs/full}"
MODEL_PATH="${MODEL_PATH:-/public/home/xinwuye/models/drkernel-14b}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"

mkdir -p /public/home/xinwuye/drkernel_scikernelbench_runs/logs

if [ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
conda activate "$ENV_NAME"

NVIDIA_LIB_PATHS="$(python - <<'PY'
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
if [ -z "$NVIDIA_LIB_PATHS" ]; then
  echo "No Python NVIDIA library paths found under the active conda env" >&2
  exit 1
fi
export LD_LIBRARY_PATH="${NVIDIA_LIB_PATHS}:${LD_LIBRARY_PATH:-}"

export PYTHONPATH="$KERNELGYM_ROOT:${SCIKERNELBENCH_ROOT}/src:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

CACHE_BASE="${SLURM_TMPDIR:-/tmp}/xinwuye_drkernel_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p \
  "$CACHE_BASE/flashinfer" \
  "$CACHE_BASE/vllm" \
  "$CACHE_BASE/xdg" \
  "$CACHE_BASE/torchinductor" \
  "$CACHE_BASE/triton" \
  "$CACHE_BASE/cuda" \
  "$CACHE_BASE/tmp"
if [ ! -w "$CACHE_BASE" ]; then
  echo "Cache base is not writable: $CACHE_BASE" >&2
  exit 1
fi
export FLASHINFER_WORKSPACE_BASE="$CACHE_BASE/flashinfer"
export VLLM_CACHE_ROOT="$CACHE_BASE/vllm"
export XDG_CACHE_HOME="$CACHE_BASE/xdg"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_BASE/torchinductor"
export TRITON_CACHE_DIR="$CACHE_BASE/triton"
export CUDA_CACHE_PATH="$CACHE_BASE/cuda"
export TMPDIR="$CACHE_BASE/tmp"

VLLM_PORT_BASE="${VLLM_PORT_BASE:-20000}"
VLLM_PORT_STRIDE="${VLLM_PORT_STRIDE:-20}"
VLLM_PORT_RANGE="${VLLM_PORT_RANGE:-2000}"
VLLM_PORT_VALUE=$((VLLM_PORT_BASE + (SLURM_JOB_ID % VLLM_PORT_RANGE) * VLLM_PORT_STRIDE))
if [ "$VLLM_PORT_VALUE" -gt 65515 ]; then
  echo "Computed VLLM_PORT is too high: $VLLM_PORT_VALUE" >&2
  exit 1
fi
export VLLM_PORT="$VLLM_PORT_VALUE"
export MASTER_PORT="$VLLM_PORT_VALUE"

cd "$KERNELGYM_ROOT"

python scripts/drkernel_scikernelbench_agent.py run-agent \
  --scikernelbench-root "$SCIKERNELBENCH_ROOT" \
  --model "$MODEL_PATH" \
  --generation-backend vllm \
  --level 3 \
  --problem-id "$SLURM_ARRAY_TASK_ID" \
  --samples 8 \
  --max-turns 3 \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --output-dir "$OUTPUT_ROOT" \
  --model-device cuda:0 \
  --eval-visible-index 1 \
  --eval-device cuda:0 \
  --vllm-tensor-parallel-size 1 \
  --vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 28672 \
  --backend triton \
  --precision fp32 \
  --num-correct-trials 5 \
  --num-perf-trials 10 \
  --eval-timeout 1200
