#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
cd "${REPO_ROOT}"

PYTHON=${PYTHON:-/home/gpuserver/miniconda3/envs/isaac51/bin/python}
CHECKPOINT=${CHECKPOINT:?Set CHECKPOINT to the direct or multitask checkpoint.}
OUTPUT_DIR=${OUTPUT_DIR:?Set OUTPUT_DIR for the collected rollout dataset.}
PROFILE=${PROFILE:-default}
GPU_ID=${GPU_ID:-0}
NUM_ENVS=${NUM_ENVS:-24}
EPISODES=${EPISODES:-24}
SEED=${SEED:-1}

case "${PROFILE}" in
  easy)
    INIT_ERROR_XY_RANGE=${INIT_ERROR_XY_RANGE:-0.001,0.004}
    ;;
  default)
    INIT_ERROR_XY_RANGE=${INIT_ERROR_XY_RANGE:-0.001,0.009}
    ;;
  hard)
    INIT_ERROR_XY_RANGE=${INIT_ERROR_XY_RANGE:-0.009,0.015}
    ;;
  *)
    echo "FAIL: PROFILE must be easy, default, or hard; got ${PROFILE}." >&2
    exit 2
    ;;
esac

INIT_ERROR_Z_RANGE=${INIT_ERROR_Z_RANGE:-0.001,0.002}
INIT_ERROR_YAW_RANGE=${INIT_ERROR_YAW_RANGE:--0.0872665,0.0872665}
RUN_ID=${RUN_ID:-phase3_3_${PROFILE}_rollout_collect}

cmd=(
  "${PYTHON}" tdmpc2/collect_eval_rollouts.py
  --config-dir configs/train
  --config-name srsa_01125_imitation_relaxed
  checkpoint="${CHECKPOINT}"
  assembly_id=00186
  collect_assembly_ids='[00186]'
  collect_spawn_per_assembly=false
  collect_output_dir="${OUTPUT_DIR}"
  collect_manifest_fp="${OUTPUT_DIR}/offline_manifest.json"
  collect_episodes_per_task="${EPISODES}"
  collect_overwrite=true
  collect_match_checkpoint=true
  collect_mpc=true
  collect_skip_deferred_tasks=false
  isaaclab_backend=srsa
  task=isaaclab-srsa-assembly
  isaaclab_dir=/home/gpuserver/IsaacLab
  srsa_dir=/home/gpuserver/hx/github/srsa
  srsa_task_template_fp=data/srsa_axial_task_templates.json
  srsa_mesh_geometry_fp=data/srsa_mesh_geometry_params.csv
  srsa_param_template_id=2
  eval_task_template_exact=true
  srsa_axial_reference_anchor_assembly_id=01125
  srsa_axial_reference_anchor_task_type_id=0
  srsa_axial_recompute_manifest_task_vecs=true
  "srsa_axial_clearance_depth_templates='1.0:1.0'"
  num_envs="${NUM_ENVS}"
  gpu_id="${GPU_ID}"
  model_size=S
  horizon=3
  compile=false
  mpc=true
  eval_terminate_on_success=false
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
  "srsa_axial_init_error_xy_range=\"${INIT_ERROR_XY_RANGE}\""
  "srsa_axial_init_error_z_range=\"${INIT_ERROR_Z_RANGE}\""
  "srsa_axial_init_error_yaw_range=\"${INIT_ERROR_YAW_RANGE}\""
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
  enable_wandb=false
  save_agent=false
  exp_name=srsa_phase3_3_attribution_rollout
  run_id="${RUN_ID}"
  seed="${SEED}"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'PASS dry-run: '
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "${OUTPUT_DIR}"
exec "${cmd[@]}"
