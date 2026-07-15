#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
cd "${REPO_ROOT}"

PYTHON=${PYTHON:-/home/gpuserver/miniconda3/envs/isaac51/bin/python}
SEED=${SEED:-1}
EXP_NAME=${EXP_NAME:-srsa_multitask_conditional_roundrobin}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
BASE_RUN_STAMP=${RUN_STAMP}
WORK_BASE=${REPO_ROOT}/logs/isaaclab-srsa-assembly/${SEED}/${EXP_NAME}
LOG_ROOT=${LOG_ROOT:-${WORK_BASE}/${RUN_STAMP}_launcher}
MANIFEST_FP=${MANIFEST_FP:-${LOG_ROOT}/online_family_replay_manifest.json}
REPLAY_DIR=${REPLAY_DIR:-${LOG_ROOT}/replay}

SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT:-${REPO_ROOT}/logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_replay_from_01125/20260615_202326_stage-1_asm-01125/models/latest.pt}
ANCHOR_REPLAY_FP=${ANCHOR_REPLAY_FP:-${REPO_ROOT}/logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_replay_from_01125/20260615_202326_launcher/replay/01125.pt}
ANCHOR_TASK_ID=${ANCHOR_TASK_ID:-01125}

# Format: "assembly:template_id assembly:template_id ..."
CONDITIONS=${CONDITIONS:-"01125:2 00256:2"}
CYCLES=${CYCLES:-1}
BLOCK_STEPS=${BLOCK_STEPS:-50000}
RESUME_COMPLETED_TARGETS=${RESUME_COMPLETED_TARGETS:-${ANCHOR_TASK_ID}}
STAGE_OFFSET=${STAGE_OFFSET:-1}
DRY_RUN=${DRY_RUN:-0}

export EXP_NAME RUN_STAMP LOG_ROOT MANIFEST_FP REPLAY_DIR
export SOURCE_CHECKPOINT ANCHOR_TASK_ID
export STEPS_PER_TASK=${STEPS_PER_TASK:-${BLOCK_STEPS}}
export ONLINE_FAMILY_SAMPLE_BALANCE=${ONLINE_FAMILY_SAMPLE_BALANCE:-condition_uniform}
export REPLAY_CONDITION_FILENAME=${REPLAY_CONDITION_FILENAME:-true}
export ACQUISITION_STOP_ENABLED=${ACQUISITION_STOP_ENABLED:-false}
export ACQUISITION_REQUIRE_SUCCESS=${ACQUISITION_REQUIRE_SUCCESS:-false}
export RETENTION_REQUIRE_GATE=${RETENTION_REQUIRE_GATE:-false}
export CURRENT_RATIO=${CURRENT_RATIO:-0.50}
export ANCHOR_RATIO=${ANCHOR_RATIO:-0.50}
export HISTORY_RATIO=${HISTORY_RATIO:-0.0}
export MIN_CURRENT_EPISODES=${MIN_CURRENT_EPISODES:-5}
export MULTI_TASK_BOOTSTRAP_CURRENT_ONLY=${MULTI_TASK_BOOTSTRAP_CURRENT_ONLY:-true}
export TASK_CONTEXT_ADAPTER_ENABLED=${TASK_CONTEXT_ADAPTER_ENABLED:-false}
export TASK_CONTEXT_ADAPTER_APPLY_ENCODER=${TASK_CONTEXT_ADAPTER_APPLY_ENCODER:-true}
export TASK_CONTEXT_ADAPTER_APPLY_DYNAMICS=${TASK_CONTEXT_ADAPTER_APPLY_DYNAMICS:-true}
export TASK_CONTEXT_ADAPTER_APPLY_POLICY=${TASK_CONTEXT_ADAPTER_APPLY_POLICY:-false}
export TASK_CONTEXT_ADAPTER_APPLY_REWARD=${TASK_CONTEXT_ADAPTER_APPLY_REWARD:-false}
export TASK_CONTEXT_ADAPTER_APPLY_Q=${TASK_CONTEXT_ADAPTER_APPLY_Q:-false}
export TASK_CONTEXT_ADAPTER_LR_SCALE=${TASK_CONTEXT_ADAPTER_LR_SCALE:-0.1}

print_command() {
  printf ' '
  printf '%q ' "$@"
  printf '\n'
}

template_pair_for_id() {
  case "$1" in
    0) printf '0.5:0.5\n' ;;
    1) printf '0.5:1.0\n' ;;
    2) printf '1.0:1.0\n' ;;
    3) printf '2.0:1.5\n' ;;
    4) printf '4.0:2.0\n' ;;
    *) printf '1.0:1.0\n' ;;
  esac
}

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${LOG_ROOT}" "${REPLAY_DIR}"
fi

seed_manifest_cmd=(
  "${PYTHON}" scripts/update_online_family_replay_manifest.py
  --manifest "${MANIFEST_FP}"
  --task-id "${ANCHOR_TASK_ID}"
  --assembly-id "${ANCHOR_TASK_ID}"
  --template-id "${SRSA_PARAM_TEMPLATE_ID:-2}"
  --condition-id "${ANCHOR_TASK_ID}|tid-${SRSA_PARAM_TEMPLATE_ID:-2}"
  --replay-fp "${ANCHOR_REPLAY_FP}"
  --checkpoint "${SOURCE_CHECKPOINT}"
  --stage-index 1
  --anchor-task-id "${ANCHOR_TASK_ID}"
  --role anchor
)

if [[ -n "${ANCHOR_REPLAY_FP}" ]]; then
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] seed manifest command:"
    print_command "${seed_manifest_cmd[@]}"
  else
    "${seed_manifest_cmd[@]}"
  fi
fi

current_checkpoint=${SOURCE_CHECKPOINT}
completed_targets=(${RESUME_COMPLETED_TARGETS})
stage_idx=${STAGE_OFFSET}

echo "[roundrobin] conditions=${CONDITIONS}"
echo "[roundrobin] cycles=${CYCLES} block_steps=${BLOCK_STEPS} sample_balance=${ONLINE_FAMILY_SAMPLE_BALANCE}"
echo "[roundrobin] manifest=${MANIFEST_FP}"

for ((cycle = 1; cycle <= CYCLES; cycle++)); do
  for condition in ${CONDITIONS}; do
    asm=${condition%%:*}
    template_id=${condition#*:}
    if [[ -z "${asm}" || -z "${template_id}" || "${asm}" == "${template_id}" ]]; then
      echo "[roundrobin] invalid condition '${condition}', expected assembly:template_id" >&2
      exit 2
    fi
    stage_idx=$((stage_idx + 1))
    block_stamp="${BASE_RUN_STAMP}_cycle-${cycle}_stage-${stage_idx}_asm-${asm}_tid-${template_id}"
    export RUN_STAMP="${block_stamp}"
    export SOURCE_CHECKPOINT="${current_checkpoint}"
    export TARGETS="${asm}"
    export RESUME_COMPLETED_TARGETS="${completed_targets[*]}"
    export STAGE_OFFSET=$((stage_idx - 1))
    export SRSA_PARAM_TEMPLATE_ID="${template_id}"
    export SRSA_CLEARANCE_DEPTH_TEMPLATES
    SRSA_CLEARANCE_DEPTH_TEMPLATES=$(template_pair_for_id "${template_id}")
    export STEPS_PER_TASK="${BLOCK_STEPS}"

    echo "[roundrobin] cycle=${cycle} stage=${stage_idx} condition=${asm}|tid-${template_id}"
    if [[ "${DRY_RUN}" == "1" ]]; then
      DRY_RUN=1 "${SCRIPT_DIR}/run_01125_online_family_replay_targets.sh"
    else
      "${SCRIPT_DIR}/run_01125_online_family_replay_targets.sh"
    fi

    current_checkpoint="${WORK_BASE}/${block_stamp}_stage-${stage_idx}_asm-${asm}/models/latest.pt"
    completed_targets+=("${asm}")
  done
done

echo "[roundrobin] final_checkpoint=${current_checkpoint}"
echo "[roundrobin] manifest=${MANIFEST_FP}"
