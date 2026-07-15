#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export EXP_NAME=${EXP_NAME:-mt_m1_01125_00256_asmembed_fixed_template}
export CONDITIONS=${CONDITIONS:-"01125:2 00256:2"}
export CYCLES=${CYCLES:-1}
export BLOCK_STEPS=${BLOCK_STEPS:-50000}
export ONLINE_FAMILY_SAMPLE_BALANCE=${ONLINE_FAMILY_SAMPLE_BALANCE:-condition_uniform}
export TASK_CONTEXT_ADAPTER_ENABLED=${TASK_CONTEXT_ADAPTER_ENABLED:-false}
export RETENTION_REQUIRE_GATE=${RETENTION_REQUIRE_GATE:-false}

exec "${SCRIPT_DIR}/run_srsa_multitask_conditional_roundrobin.sh"
