#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
cd "${REPO_ROOT}"

PYTHON=${PYTHON:-/home/gpuserver/miniconda3/envs/isaac51/bin/python}
export RUN_STAMP=${RUN_STAMP:-20260712_phase3_1_00186}
SEED=${SEED:-1}

export EXP_NAME=${EXP_NAME:-srsa_axial_online_family_taskctx_repair_01125_00256_00186}
export SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT:-${REPO_ROOT}/logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_taskctx_repair_01125_00256/20260708_taskctx_repair_phase2_stage-2_asm-00256/models/best_step-99840_s-0p9961.pt}
export ANCHOR_REPLAY_FP=${ANCHOR_REPLAY_FP:-${REPO_ROOT}/logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_replay_from_01125/20260615_202326_launcher/replay/01125.pt}
export HISTORY_REPLAY_FP=${HISTORY_REPLAY_FP:-${REPO_ROOT}/logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_taskctx_repair_01125_00256/20260708_taskctx_repair_phase2_launcher/replay/00256.pt}

export TARGETS=${TARGETS:-00186}
export RESUME_COMPLETED_TARGETS=${RESUME_COMPLETED_TARGETS:-"01125 00256"}
export STAGE_OFFSET=${STAGE_OFFSET:-2}
export ANCHOR_TASK_ID=${ANCHOR_TASK_ID:-01125}

export CURRENT_RATIO=${CURRENT_RATIO:-0.60}
export ANCHOR_RATIO=${ANCHOR_RATIO:-0.20}
export HISTORY_RATIO=${HISTORY_RATIO:-0.20}
export MIN_CURRENT_EPISODES=${MIN_CURRENT_EPISODES:-5}
export ONLINE_FAMILY_SAMPLE_BALANCE=${ONLINE_FAMILY_SAMPLE_BALANCE:-ratio}

export STEPS_PER_TASK=${STEPS_PER_TASK:-100000}
export NUM_ENVS=${NUM_ENVS:-256}
export RETENTION_NUM_ENVS=${RETENTION_NUM_ENVS:-256}
export EVAL_FREQ=${EVAL_FREQ:-49920}
export SAVE_FREQ=${SAVE_FREQ:-49920}
export EVAL_EPISODES=${EVAL_EPISODES:-1}
export RETENTION_EVAL_EPISODES=${RETENTION_EVAL_EPISODES:-20}
export ACQUISITION_STOP_ENABLED=${ACQUISITION_STOP_ENABLED:-false}
export ACQUISITION_REQUIRE_SUCCESS=${ACQUISITION_REQUIRE_SUCCESS:-false}
export RETENTION_REQUIRE_GATE=${RETENTION_REQUIRE_GATE:-false}

export TASK_CONTEXT_ADAPTER_ENABLED=${TASK_CONTEXT_ADAPTER_ENABLED:-true}
export TASK_CONTEXT_ADAPTER_SOURCE=${TASK_CONTEXT_ADAPTER_SOURCE:-raw_task_vec}
export TASK_CONTEXT_ADAPTER_ALPHA=${TASK_CONTEXT_ADAPTER_ALPHA:-0.01}
export TASK_CONTEXT_ADAPTER_HIDDEN_DIM=${TASK_CONTEXT_ADAPTER_HIDDEN_DIM:-128}
export TASK_CONTEXT_ADAPTER_APPLY_ENCODER=${TASK_CONTEXT_ADAPTER_APPLY_ENCODER:-true}
export TASK_CONTEXT_ADAPTER_APPLY_DYNAMICS=${TASK_CONTEXT_ADAPTER_APPLY_DYNAMICS:-true}
export TASK_CONTEXT_ADAPTER_APPLY_POLICY=${TASK_CONTEXT_ADAPTER_APPLY_POLICY:-false}
export TASK_CONTEXT_ADAPTER_APPLY_REWARD=${TASK_CONTEXT_ADAPTER_APPLY_REWARD:-false}
export TASK_CONTEXT_ADAPTER_APPLY_Q=${TASK_CONTEXT_ADAPTER_APPLY_Q:-false}
export TASK_CONTEXT_ADAPTER_LR_SCALE=${TASK_CONTEXT_ADAPTER_LR_SCALE:-0.1}

export TASK_CONTEXT_REPAIR_ENABLED=${TASK_CONTEXT_REPAIR_ENABLED:-true}
export TASK_RECON_COEF=${TASK_RECON_COEF:-0.1}
export TASK_SPREAD_COEF=${TASK_SPREAD_COEF:-0.01}
export TASK_RAW_RESIDUAL_SCALE=${TASK_RAW_RESIDUAL_SCALE:-0.1}
export TASK_SPREAD_NEAR_THRESHOLD=${TASK_SPREAD_NEAR_THRESHOLD:-0.3}
export TASK_SPREAD_FAR_THRESHOLD=${TASK_SPREAD_FAR_THRESHOLD:-1.0}
export TASK_SPREAD_MARGIN=${TASK_SPREAD_MARGIN:-0.5}

WORK_BASE=${REPO_ROOT}/logs/isaaclab-srsa-assembly/${SEED}/${EXP_NAME}
export LOG_ROOT=${LOG_ROOT:-${WORK_BASE}/${RUN_STAMP}_launcher}
export MANIFEST_FP=${MANIFEST_FP:-${LOG_ROOT}/online_family_replay_manifest.json}
export REPLAY_DIR=${REPLAY_DIR:-${LOG_ROOT}/replay}

seed_anchor=(
  "${PYTHON}" scripts/update_online_family_replay_manifest.py
  --manifest "${MANIFEST_FP}"
  --task-id 01125
  --assembly-id 01125
  --template-id 2
  --condition-id "01125|tid-2"
  --replay-fp "${ANCHOR_REPLAY_FP}"
  --checkpoint "${SOURCE_CHECKPOINT}"
  --stage-index 1
  --anchor-task-id 01125
  --role anchor
)
seed_history=(
  "${PYTHON}" scripts/update_online_family_replay_manifest.py
  --manifest "${MANIFEST_FP}"
  --task-id 00256
  --assembly-id 00256
  --template-id 2
  --condition-id "00256|tid-2"
  --replay-fp "${HISTORY_REPLAY_FP}"
  --checkpoint "${SOURCE_CHECKPOINT}"
  --stage-index 2
  --anchor-task-id 01125
  --role history
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '[dry-run] '
  printf '%q ' "${seed_anchor[@]}"
  printf '\n[dry-run] '
  printf '%q ' "${seed_history[@]}"
  printf '\n'
else
  mkdir -p "${LOG_ROOT}" "${REPLAY_DIR}"
  "${seed_anchor[@]}"
  "${seed_history[@]}"
fi

exec "${SCRIPT_DIR}/run_01125_online_family_replay_targets.sh"
