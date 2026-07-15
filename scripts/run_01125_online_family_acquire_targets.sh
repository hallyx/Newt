#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
cd "${REPO_ROOT}"

PYTHON=${PYTHON:-/home/gpuserver/miniconda3/envs/isaac51/bin/python}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
SEED=${SEED:-1}

# Acquisition-first continuation from the validated 01125 online-family stage.
# Keep CUDA_VISIBLE_DEVICES external: for physical GPU 1 use
# CUDA_VISIBLE_DEVICES=1 GPU_ID=0 scripts/run_01125_online_family_acquire_targets.sh

DEFAULT_STAGE1_ROOT=${REPO_ROOT}/logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_replay_from_01125
DEFAULT_STAGE1_STAMP=20260615_202326
DEFAULT_STAGE1_CHECKPOINT=${DEFAULT_STAGE1_ROOT}/${DEFAULT_STAGE1_STAMP}_stage-1_asm-01125/models/latest.pt
DEFAULT_STAGE1_REPLAY=${DEFAULT_STAGE1_ROOT}/${DEFAULT_STAGE1_STAMP}_launcher/replay/01125.pt

export TARGETS=${TARGETS:-"00256"}
export RESUME_COMPLETED_TARGETS=${RESUME_COMPLETED_TARGETS:-"01125"}
export STAGE_OFFSET=${STAGE_OFFSET:-1}
export SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT:-${DEFAULT_STAGE1_CHECKPOINT}}
export ANCHOR_REPLAY_FP=${ANCHOR_REPLAY_FP:-${DEFAULT_STAGE1_REPLAY}}

export EXP_NAME=${EXP_NAME:-srsa_axial_online_family_acquire_from_01125}
WORK_BASE=${REPO_ROOT}/logs/isaaclab-srsa-assembly/${SEED}/${EXP_NAME}
export LOG_ROOT=${LOG_ROOT:-${WORK_BASE}/${RUN_STAMP}_launcher}
export MANIFEST_FP=${MANIFEST_FP:-${LOG_ROOT}/online_family_replay_manifest.json}
export REPLAY_DIR=${REPLAY_DIR:-${LOG_ROOT}/replay}

export STEPS_PER_TASK=${STEPS_PER_TASK:-300000}
export NUM_ENVS=${NUM_ENVS:-256}
export RETENTION_NUM_ENVS=${RETENTION_NUM_ENVS:-256}
export MULTIPROC=${MULTIPROC:-false}
export NUM_GPUS=${NUM_GPUS:-1}
export GPU_ID=${GPU_ID:-0}
export EVAL_FREQ=${EVAL_FREQ:-49920}
export SAVE_FREQ=${SAVE_FREQ:-${EVAL_FREQ}}
export EVAL_EPISODES=${EVAL_EPISODES:-1}
export RETENTION_EVAL_EPISODES=${RETENTION_EVAL_EPISODES:-20}

export CURRENT_RATIO=${CURRENT_RATIO:-0.80}
export ANCHOR_RATIO=${ANCHOR_RATIO:-0.20}
export HISTORY_RATIO=${HISTORY_RATIO:-0.0}
export MIN_CURRENT_EPISODES=${MIN_CURRENT_EPISODES:-5}

export ACQUISITION_STOP_ENABLED=${ACQUISITION_STOP_ENABLED:-true}
export ACQUISITION_REQUIRE_SUCCESS=${ACQUISITION_REQUIRE_SUCCESS:-true}
export ACQUISITION_SUCCESS_THRESHOLD=${ACQUISITION_SUCCESS_THRESHOLD:-0.80}
export ACQUISITION_MIN_STEPS=${ACQUISITION_MIN_STEPS:-150000}
export ACQUISITION_METRIC=${ACQUISITION_METRIC:-episode_success}
export UPDATE_PROGRESS_LOG_EVERY=${UPDATE_PROGRESS_LOG_EVERY:-50}
export RETENTION_REQUIRE_GATE=${RETENTION_REQUIRE_GATE:-true}
export RETENTION_MIN_FAMILY=${RETENTION_MIN_FAMILY:-0.80}
export RETENTION_MIN_ANCHOR=${RETENTION_MIN_ANCHOR:-0.75}
export RETENTION_MIN_CURRENT=${RETENTION_MIN_CURRENT:-0.80}
export RETENTION_GATE_METRIC=${RETENTION_GATE_METRIC:-episode_success}

export CONFIG_NAME=${CONFIG_NAME:-srsa_01125_imitation_relaxed}
export EVAL_SUCCESS_METRIC=${EVAL_SUCCESS_METRIC:-relaxed}
export ENABLE_WANDB=${ENABLE_WANDB:-false}

print_command() {
  printf ' '
  printf '%q ' "$@"
  printf '\n'
}

seed_manifest_cmd=(
  "${PYTHON}" scripts/update_online_family_replay_manifest.py
  --manifest "${MANIFEST_FP}"
  --task-id "01125"
  --assembly-id "01125"
  --template-id "${SRSA_PARAM_TEMPLATE_ID:-2}"
  --condition-id "01125|tid-${SRSA_PARAM_TEMPLATE_ID:-2}"
  --replay-fp "${ANCHOR_REPLAY_FP}"
  --checkpoint "${SOURCE_CHECKPOINT}"
  --stage-index 1
  --anchor-task-id "01125"
  --role anchor
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[dry-run] seed manifest command:"
  print_command "${seed_manifest_cmd[@]}"
else
  if [[ ! -f "${SOURCE_CHECKPOINT}" ]]; then
    echo "[acquire] source checkpoint not found: ${SOURCE_CHECKPOINT}" >&2
    exit 1
  fi
  if [[ ! -f "${ANCHOR_REPLAY_FP}" ]]; then
    echo "[acquire] anchor replay not found: ${ANCHOR_REPLAY_FP}" >&2
    exit 1
  fi
  mkdir -p "${LOG_ROOT}" "${REPLAY_DIR}"
  "${seed_manifest_cmd[@]}"
fi

exec "${SCRIPT_DIR}/run_01125_online_family_replay_targets.sh"
