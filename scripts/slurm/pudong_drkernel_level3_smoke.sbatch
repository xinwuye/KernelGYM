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

cd "$KERNELGYM_ROOT"

python scripts/drkernel_scikernelbench_agent.py run-agent \
  --scikernelbench-root "$SCIKERNELBENCH_ROOT" \
  --model hkust-nlp/drkernel-14b \
  --level 3 \
  --problem-id "$SLURM_ARRAY_TASK_ID" \
  --samples 1 \
  --max-turns 3 \
  --output-dir "$OUTPUT_ROOT" \
  --model-device cuda:0 \
  --eval-visible-index 1 \
  --eval-device cuda:0 \
  --backend triton \
  --precision fp32 \
  --num-correct-trials 2 \
  --num-perf-trials 3 \
  --eval-timeout 900
