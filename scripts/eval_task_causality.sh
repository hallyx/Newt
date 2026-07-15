#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
cd "${REPO_ROOT}"

PYTHON=${PYTHON:-/home/gpuserver/miniconda3/envs/isaac51/bin/python}
CHECKPOINT=${CHECKPOINT:-}
REPLAY=${REPLAY:-}
ANCHOR_REPLAY=${ANCHOR_REPLAY:-}
CONFIG=${CONFIG:-configs/train/srsa_01125_imitation_relaxed.yaml}
OUTPUT=${OUTPUT:-logs/task_vec_sensitivity/task_causality_report.json}
GPU_ID=${GPU_ID:-0}
BATCH_SIZE=${BATCH_SIZE:-512}
SEED=${SEED:-1}
ASSEMBLY_ID=${ASSEMBLY_ID:-00256}
EVAL_TASK_ID=${EVAL_TASK_ID:-2}

if [[ -z "${CHECKPOINT}" ]]; then
  echo "[task-causality] CHECKPOINT is required" >&2
  exit 2
fi
if [[ -z "${REPLAY}" ]]; then
  echo "[task-causality] REPLAY is required" >&2
  exit 2
fi

cmd=(
  "${PYTHON}" tdmpc2/scripts/task_vec_sensitivity_report.py
  --checkpoint "${CHECKPOINT}"
  --replay "${REPLAY}"
  --config "${CONFIG}"
  --output "${OUTPUT}"
  --gpu-id "${GPU_ID}"
  --batch-size "${BATCH_SIZE}"
  --seed "${SEED}"
  --assembly-id "${ASSEMBLY_ID}"
  --eval-task-id "${EVAL_TASK_ID}"
  --include-zero
  --include-random
  --include-extreme
)

if [[ -n "${ANCHOR_REPLAY}" ]]; then
  cmd+=(--anchor-replay "${ANCHOR_REPLAY}")
fi

if [[ -n "${WRONG_TEMPLATE_TASK_VEC:-}" ]]; then
  cmd+=(--task-vec "wrong_template=${WRONG_TEMPLATE_TASK_VEC}")
fi
if [[ -n "${WRONG_ASSEMBLY_TASK_VEC:-}" ]]; then
  cmd+=(--task-vec "wrong_assembly=${WRONG_ASSEMBLY_TASK_VEC}")
fi

printf '[task-causality]'
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
