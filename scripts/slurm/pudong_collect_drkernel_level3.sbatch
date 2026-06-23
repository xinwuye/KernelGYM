#!/usr/bin/env bash
#SBATCH --job-name=drk-l3-collect
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/public/home/xinwuye/drkernel_scikernelbench_runs/logs/collect_%j.out
#SBATCH --error=/public/home/xinwuye/drkernel_scikernelbench_runs/logs/collect_%j.err

set -euo pipefail

ENV_NAME="${ENV_NAME:-drkernel-scikernelbench}"
KERNELGYM_ROOT="${KERNELGYM_ROOT:-/public/home/xinwuye/KernelGYM}"
SCIKERNELBENCH_ROOT="${SCIKERNELBENCH_ROOT:-/public/home/xinwuye/SciKernelBench}"
AGENT_OUTPUT_DIR="${AGENT_OUTPUT_DIR:-/public/home/xinwuye/drkernel_scikernelbench_runs/full}"
RUN_NAME="${RUN_NAME:-drkernel_14b_agent_level3_scikernelbench}"
HARDWARE_NAME="${HARDWARE_NAME:-Pudong_DrKernel_SciKernelBench}"

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
python scripts/drkernel_scikernelbench_agent.py collect \
  --scikernelbench-root "$SCIKERNELBENCH_ROOT" \
  --agent-output-dir "$AGENT_OUTPUT_DIR" \
  --level 3 \
  --run-name "$RUN_NAME" \
  --hardware-name "$HARDWARE_NAME" \
  --baseline-name baseline_time_torch

cd "$SCIKERNELBENCH_ROOT"
python scripts/benchmark_eval_analysis.py \
  run_name="$RUN_NAME" \
  level=3 \
  hardware="$HARDWARE_NAME" \
  baseline=baseline_time_torch \
  output_file="runs/${RUN_NAME}/analysis.json"
