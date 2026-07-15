#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
cd "${REPO_ROOT}"

PYTHON=${PYTHON:-/home/gpuserver/miniconda3/envs/isaac51/bin/python}
CHECKPOINT=${CHECKPOINT:?Set CHECKPOINT to the checkpoint to evaluate.}
OUTPUT_DIR=${OUTPUT_DIR:?Set OUTPUT_DIR for the batch evaluation outputs.}
GPU_ID=${GPU_ID:-0}
NUM_ENVS=${NUM_ENVS:-256}
EPISODES=${EPISODES:-20}
RUN_ID=${RUN_ID:-phase3_three_task_checkpoint_eval}
EVAL_ASSEMBLY_IDS=${EVAL_ASSEMBLY_IDS:-'[01125,00256,00186]'}
TASK_CONTEXT_ADAPTER_ENABLED=${TASK_CONTEXT_ADAPTER_ENABLED:-true}
TASK_CONTEXT_REPAIR_ENABLED=${TASK_CONTEXT_REPAIR_ENABLED:-true}
INIT_ERROR_XY_RANGE=${INIT_ERROR_XY_RANGE:-0.009,0.0010}
INIT_ERROR_Z_RANGE=${INIT_ERROR_Z_RANGE:-0.0010,0.0020}
INIT_ERROR_YAW_RANGE=${INIT_ERROR_YAW_RANGE:--0.0872665,0.0872665}

cmd=(
  "${PYTHON}" tdmpc2/batch_eval_tasks.py
  --config-dir configs/train
  --config-name srsa_01125_imitation_relaxed
  checkpoint="${CHECKPOINT}"
  eval_assembly_ids="${EVAL_ASSEMBLY_IDS}"
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
  task_context_adapter_enabled="${TASK_CONTEXT_ADAPTER_ENABLED}"
  task_context_adapter_hidden_dim=128
  task_context_adapter_alpha=0.01
  task_context_adapter_source=raw_task_vec
  task_context_adapter_apply_encoder=true
  task_context_adapter_apply_dynamics=true
  task_context_adapter_apply_policy=false
  task_context_adapter_apply_reward=false
  task_context_adapter_apply_q=false
  task_context_adapter_lr_scale=0.1
  task_context_repair_enabled="${TASK_CONTEXT_REPAIR_ENABLED}"
  task_recon_coef=0.1
  task_spread_coef=0.01
  task_raw_residual_scale=0.1
  task_spread_near_threshold=0.3
  task_spread_far_threshold=1.0
  task_spread_margin=0.5
  eval_success_metric=relaxed
  srsa_eval_success_metric=relaxed
  batch_eval_episodes_per_task="${EPISODES}"
  batch_eval_spawn_per_assembly=true
  batch_eval_overwrite=true
  batch_eval_output_dir="${OUTPUT_DIR}"
  batch_eval_summary_fp="${OUTPUT_DIR}/batch_eval_summary.json"
  enable_wandb=false
  exp_name=srsa_phase3_three_task_checkpoint_eval
  run_id="${RUN_ID}"
  seed=1
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '[dry-run] '
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "${OUTPUT_DIR}"
exec "${cmd[@]}"
