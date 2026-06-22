#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
cd "${REPO_ROOT}"

PYTHON=${PYTHON:-/home/gpuserver/miniconda3/envs/isaac51/bin/python}
CONFIG_DIR=${CONFIG_DIR:-configs/train}
CONFIG_NAME=${CONFIG_NAME:-srsa_01125_imitation_relaxed}
ISAACLAB_DIR=${ISAACLAB_DIR:-/home/gpuserver/IsaacLab}
SRSA_DIR=${SRSA_DIR:-/home/gpuserver/hx/github/srsa}

TARGET_ASSEMBLY=${TARGET_ASSEMBLY:-00256}
CHECKPOINT=${CHECKPOINT:-${REPO_ROOT}/logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_acquire_from_01125/20260618_001734_stage-2_asm-00256/models/latest.pt}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT_ROOT=${OUT_ROOT:-${REPO_ROOT}/logs/task_vec_swap_eval/${RUN_STAMP}}

NUM_ENVS=${NUM_ENVS:-256}
EPISODES=${EPISODES:-20}
GPU_ID=${GPU_ID:-0}
EVAL_SUCCESS_METRIC=${EVAL_SUCCESS_METRIC:-relaxed}
SRSA_CLEARANCE_DEPTH_TEMPLATES=${SRSA_CLEARANCE_DEPTH_TEMPLATES:-1.0:1.0}

VEC_01125=${VEC_01125:-[0,-0.155195,0.145688,0.165645,1,0]}
VEC_ZERO=${VEC_ZERO:-[0,0,0,0,0,0]}
DRY_RUN=${DRY_RUN:-0}

print_command() {
  printf ' '
  printf '%q ' "$@"
  printf '\n'
}

run_eval() {
  local label=$1
  local force_vec=${2:-}
  local output_dir="${OUT_ROOT}/${label}"
  local summary_fp="${output_dir}/batch_eval_summary.json"
  local cmd=(
    "${PYTHON}" tdmpc2/batch_eval_tasks.py
    --config-dir "${CONFIG_DIR}"
    --config-name "${CONFIG_NAME}"
    checkpoint="${CHECKPOINT}"
    eval_assembly_ids="[${TARGET_ASSEMBLY}]"
    isaaclab_backend=srsa
    task=isaaclab-srsa-assembly
    isaaclab_dir="${ISAACLAB_DIR}"
    srsa_dir="${SRSA_DIR}"
    srsa_task_template_fp="${REPO_ROOT}/data/srsa_axial_task_templates.json"
    srsa_mesh_geometry_fp="${REPO_ROOT}/data/srsa_mesh_geometry_params.csv"
    srsa_param_template_id=2
    eval_task_template_exact=true
    srsa_axial_reference_anchor_assembly_id=01125
    srsa_axial_reference_anchor_task_type_id=0
    srsa_axial_recompute_manifest_task_vecs=true
    "srsa_axial_clearance_depth_templates='${SRSA_CLEARANCE_DEPTH_TEMPLATES}'"
    num_envs="${NUM_ENVS}"
    gpu_id="${GPU_ID}"
    model_size=S
    horizon=3
    compile=false
    mpc=true
    isaaclab_headless=true
    isaaclab_use_canonical_obs=true
    srsa_task_family_name=normal_fit
    srsa_task_param_obs=false
    srsa_task_param_obs_mode=task_vec
    srsa_enable_axial_task_param_sampler=true
    srsa_axial_fixed_plug_scale=true
    srsa_axial_clearance_base=0.000114
    srsa_axial_clearance_jitter_ratio=0.10
    srsa_axial_depth_base=0.015
    srsa_axial_depth_jitter_ratio=0.10
    'srsa_axial_init_error_xy_range="0.009,0.0010"'
    'srsa_axial_init_error_z_range="0.0010,0.0020"'
    'srsa_axial_init_error_yaw_range="-0.0872665,0.0872665"'
    'srsa_axial_visual_noise_xy_range="0.0,0.0"'
    'srsa_axial_visual_noise_z_range="0.0,0.0"'
    srsa_enable_flange_force_sensor=true
    isaaclab_canonical_append_force=true
    isaaclab_canonical_append_task_params=false
    srsa_vision_noise_xy_std=0.0
    srsa_vision_noise_xy_jitter_std=0.0
    srsa_vision_noise_z_std=0.0
    srsa_vision_noise_z_jitter_std=0.0
    isaaclab_canonical_use_visual_noise=false
    task_conditioning=axial_params
    contact_history_enabled=true
    contact_history_len=4
    contact_context_dim=64
    contact_history_hidden_dim=128
    contact_history_layers=2
    contact_force_dim=6
    contact_action_dim=3
    contact_ee_delta_dim=3
    contact_history_use_ee_delta=true
    eval_success_metric="${EVAL_SUCCESS_METRIC}"
    srsa_eval_success_metric="${EVAL_SUCCESS_METRIC}"
    batch_eval_episodes_per_task="${EPISODES}"
    batch_eval_spawn_per_assembly=false
    batch_eval_overwrite=true
    batch_eval_output_dir="${output_dir}"
    batch_eval_summary_fp="${summary_fp}"
    enable_wandb=false
    exp_name=srsa_axial_task_vec_swap_eval
    run_id="${RUN_STAMP}_${label}"
    seed=1
  )
  if [[ -n "${force_vec}" ]]; then
    cmd+=(
      batch_eval_force_task_vec_label="${label}"
      batch_eval_force_task_vec_6="${force_vec}"
    )
  fi

  echo "[swap-eval] label=${label} output_dir=${output_dir}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    print_command "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
}

if [[ ! -f "${CHECKPOINT}" && "${DRY_RUN}" != "1" ]]; then
  echo "[swap-eval] checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

run_eval normal ""
run_eval force_01125_vec "${VEC_01125}"
run_eval force_zero_vec "${VEC_ZERO}"

if [[ "${DRY_RUN}" != "1" ]]; then
  "${PYTHON}" - "${OUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for label in ["normal", "force_01125_vec", "force_zero_vec"]:
    fp = root / label / "batch_eval_summary.json"
    if not fp.exists():
        print(f"{label}: missing {fp}")
        continue
    payload = json.loads(fp.read_text(encoding="utf-8"))
    task = payload["tasks"][0]
    print(
        f"{label}: success={task['episode_success']:.4f} "
        f"strict={task.get('episode_strict_success_stable', float('nan')):.4f} "
        f"reward={task['episode_reward']:.3f}"
    )
PY
fi
