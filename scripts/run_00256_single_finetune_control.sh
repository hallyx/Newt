#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}

# Single-task control for assembly 00256.
# This intentionally disables online-family replay and target handoff by
# delegating to the direct fine-tune launcher with TARGETS fixed to 00256.

RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}

export TARGETS=${TARGETS:-"00256"}
export STEPS=${STEPS:-600000}
export NUM_ENVS=${NUM_ENVS:-256}
export MULTIPROC=${MULTIPROC:-false}
export NUM_GPUS=${NUM_GPUS:-1}
export GPU_ID=${GPU_ID:-0}
export EVAL_FREQ=${EVAL_FREQ:-49920}
export SAVE_FREQ=${SAVE_FREQ:-${EVAL_FREQ}}
export EVAL_EPISODES=${EVAL_EPISODES:-1}
export EVAL_SUCCESS_METRIC=${EVAL_SUCCESS_METRIC:-relaxed}
export EXP_NAME=${EXP_NAME:-srsa_axial_00256_single_finetune_from_01125}
export RUN_STAMP
export LOG_ROOT=${LOG_ROOT:-${REPO_ROOT}/logs/single_00256_finetune_control/${RUN_STAMP}}

exec "${SCRIPT_DIR}/run_01125_direct_finetune_targets.sh"
