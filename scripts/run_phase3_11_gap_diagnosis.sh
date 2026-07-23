#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
cd "${REPO_ROOT}"

# Physical CUDA0 is reserved for another project.  Expose only physical CUDA1;
# inside the process it is intentionally addressed as logical cuda:0.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
if [[ "${CUDA_VISIBLE_DEVICES}" != "1" ]]; then
  echo "FAIL: Phase 3.11 is pinned to physical CUDA1; CUDA_VISIBLE_DEVICES must equal 1." >&2
  exit 2
fi

PYTHON=${PYTHON:-/home/gpuserver/miniconda3/envs/isaac51/bin/python}

cmd=(
  "${PYTHON}"
  tdmpc2/scripts/phase3_three_task_pilot/offline_closed_loop_gap_diagnosis.py
  --gpu-id 0
  --intervention-episodes "${INTERVENTION_EPISODES:-8}"
  --distribution-episodes "${DISTRIBUTION_EPISODES:-8}"
  --old-task-episodes "${OLD_TASK_EPISODES:-20}"
  --candidate-base-states "${CANDIDATE_BASE_STATES:-4}"
)

if [[ -n "${OUTPUT_JSON:-}" ]]; then
  cmd+=(--output-json "${OUTPUT_JSON}")
fi
if [[ -n "${OUTPUT_MD:-}" ]]; then
  cmd+=(--output-md "${OUTPUT_MD}")
fi
cmd+=(--partial-dir "${PARTIAL_DIR:-reports/phase3_three_task_pilot/phase3_11_parts}")

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  cmd+=(--dry-run)
  exec "${cmd[@]}"
elif [[ "${CLONE_SMOKE:-0}" == "1" ]]; then
  cmd+=(--clone-smoke)
  exec "${cmd[@]}"
fi

if [[ -n "${SECTION:-}" ]]; then
  exec "${cmd[@]}" --section "${SECTION}"
fi

for section in intervention prediction_reality distribution_shift anchor_01125 anchor_00256; do
  "${cmd[@]}" --section "${section}"
done
exec "${cmd[@]}" --section aggregate
