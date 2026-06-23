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
export PYTHONPATH="$KERNELGYM_ROOT:${SCIKERNELBENCH_ROOT}/src:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

cd "$KERNELGYM_ROOT"

python scripts/drkernel_scikernelbench_agent.py run-agent \
  --scikernelbench-root "$SCIKERNELBENCH_ROOT" \
  --model "$MODEL_PATH" \
  --level 3 \
  --problem-id "$SLURM_ARRAY_TASK_ID" \
  --samples 8 \
  --max-turns 3 \
  --output-dir "$OUTPUT_ROOT" \
  --model-device cuda:0 \
  --eval-visible-index 1 \
  --eval-device cuda:0 \
  --backend triton \
  --precision fp32 \
  --num-correct-trials 5 \
  --num-perf-trials 10 \
  --eval-timeout 1200
