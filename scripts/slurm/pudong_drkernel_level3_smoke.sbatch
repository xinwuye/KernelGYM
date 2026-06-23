#!/usr/bin/env bash
#SBATCH --job-name=drk-l3-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --array=1-5
#SBATCH --output=/public/home/xinwuye/drkernel_scikernelbench_runs/logs/smoke_%A_%a.out
#SBATCH --error=/public/home/xinwuye/drkernel_scikernelbench_runs/logs/smoke_%A_%a.err

set -euo pipefail

ENV_NAME="${ENV_NAME:-drkernel-scikernelbench}"
KERNELGYM_ROOT="${KERNELGYM_ROOT:-/public/home/xinwuye/KernelGYM}"
SCIKERNELBENCH_ROOT="${SCIKERNELBENCH_ROOT:-/public/home/xinwuye/SciKernelBench}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/public/home/xinwuye/drkernel_scikernelbench_runs/smoke}"
MODEL_PATH="${MODEL_PATH:-/public/home/xinwuye/models/drkernel-14b}"

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

cd "$KERNELGYM_ROOT"

python scripts/drkernel_scikernelbench_agent.py run-agent \
  --scikernelbench-root "$SCIKERNELBENCH_ROOT" \
  --model "$MODEL_PATH" \
  --generation-backend vllm \
  --level 3 \
  --problem-id "$SLURM_ARRAY_TASK_ID" \
  --samples 1 \
  --max-turns 3 \
  --max-new-tokens 2048 \
  --output-dir "$OUTPUT_ROOT" \
  --model-device cuda:0 \
  --eval-visible-index 1 \
  --eval-device cuda:0 \
  --vllm-tensor-parallel-size 1 \
  --vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 28672 \
  --backend triton \
  --precision fp32 \
  --num-correct-trials 2 \
  --num-perf-trials 3 \
  --eval-timeout 900
