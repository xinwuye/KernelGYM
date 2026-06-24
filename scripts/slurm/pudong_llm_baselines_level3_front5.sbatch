#!/usr/bin/env bash
#SBATCH --job-name=llm-l3-front5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --array=1-20%8
#SBATCH --output=/public/home/xinwuye/baseline_scikernelbench_runs/logs/front5_%A_%a.out
#SBATCH --error=/public/home/xinwuye/baseline_scikernelbench_runs/logs/front5_%A_%a.err

set -euo pipefail

ENV_NAME="${ENV_NAME:-kernel-baselines-scikernelbench}"
KERNELGYM_ROOT="${KERNELGYM_ROOT:-/public/home/xinwuye/KernelGYM}"
SCIKERNELBENCH_ROOT="${SCIKERNELBENCH_ROOT:-/public/home/xinwuye/SciKernelBench}"
DICE_ROOT="${DICE_ROOT:-/public/home/xinwuye/DICE}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/public/home/xinwuye/baseline_scikernelbench_runs/level3_front5_official_llm_baselines}"

mkdir -p /public/home/xinwuye/baseline_scikernelbench_runs/logs

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

export PYTHONPATH="$KERNELGYM_ROOT/scripts:$KERNELGYM_ROOT:${SCIKERNELBENCH_ROOT}/src:${DICE_ROOT}/evaluation:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/public/home/xinwuye/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export PYTHONUNBUFFERED=1

CACHE_BASE="${SLURM_TMPDIR:-/tmp}/xinwuye_llm_baselines_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
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

VLLM_PORT_BASE="${VLLM_PORT_BASE:-22000}"
VLLM_PORT_STRIDE="${VLLM_PORT_STRIDE:-20}"
VLLM_PORT_RANGE="${VLLM_PORT_RANGE:-2000}"
TASK_OFFSET="${SLURM_ARRAY_TASK_ID:-1}"
VLLM_PORT_VALUE=$((VLLM_PORT_BASE + ((SLURM_JOB_ID + TASK_OFFSET) % VLLM_PORT_RANGE) * VLLM_PORT_STRIDE))
if [ "$VLLM_PORT_VALUE" -gt 65515 ]; then
  echo "Computed VLLM_PORT is too high: $VLLM_PORT_VALUE" >&2
  exit 1
fi
export VLLM_PORT="$VLLM_PORT_VALUE"
export MASTER_PORT="$VLLM_PORT_VALUE"

BASELINES=(kernelcoder autotriton kernelllm dice)
ARRAY_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
BASELINE_INDEX=$((ARRAY_INDEX / 5))
PROBLEM_ID=$((ARRAY_INDEX % 5 + 1))
BASELINE="${BASELINES[$BASELINE_INDEX]}"

case "$BASELINE" in
  kernelcoder)
    MODEL_PATH="${KERNELCODER_MODEL_PATH:-lkongam/KernelCoder}"
    ;;
  autotriton)
    MODEL_PATH="${AUTOTRITON_MODEL_PATH:-ai9stars/AutoTriton}"
    ;;
  kernelllm)
    MODEL_PATH="${KERNELLLM_MODEL_PATH:-facebook/KernelLLM}"
    ;;
  dice)
    MODEL_PATH="${DICE_MODEL_PATH:-deadlykitten4/DICE-8B}"
    ;;
  *)
    echo "Unknown baseline: $BASELINE" >&2
    exit 1
    ;;
esac

echo "baseline=$BASELINE problem_id=$PROBLEM_ID output_root=$OUTPUT_ROOT model=$MODEL_PATH"

cd "$KERNELGYM_ROOT"

python scripts/llm_baselines_scikernelbench.py run-problem \
  --scikernelbench-root "$SCIKERNELBENCH_ROOT" \
  --dice-root "$DICE_ROOT" \
  --baseline "$BASELINE" \
  --model "$MODEL_PATH" \
  --level 3 \
  --problem-id "$PROBLEM_ID" \
  --output-dir "$OUTPUT_ROOT" \
  --model-device cuda:0 \
  --model-dtype auto \
  --eval-visible-index 1 \
  --eval-device cuda:0 \
  --vllm-tensor-parallel-size 1 \
  --precision fp32 \
  --num-correct-trials 5 \
  --num-perf-trials 100 \
  --eval-timeout 1800
