#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
cd "${REPO_ROOT}"

PYTHON=${PYTHON:-/home/robot2/miniconda3/envs/env_isaaclab/bin/python}
ISAACLAB_DIR=${ISAACLAB_DIR:-/home/robot2/isaaclab/IsaacLab}
SRSA_DIR=${SRSA_DIR:-/home/robot2/hx/github/srsa}
SOURCE_ID=${SOURCE_ID:-00186}
TARGETS=${TARGETS:-"${SOURCE_ID}"}
STEPS=${STEPS:-10000000}
NUM_ENVS=${NUM_ENVS:-350}
MULTIPROC=${MULTIPROC:-false}
NUM_GPUS=${NUM_GPUS:-1}
GPU_ID=${GPU_ID:-0}
EVAL_FREQ=${EVAL_FREQ:-100000}
SAVE_FREQ=${SAVE_FREQ:-${EVAL_FREQ}}
EXP_NAME=${EXP_NAME:-srsa_axial_imitation_relaxed}
SRSA_TASK_TEMPLATE_FP=${SRSA_TASK_TEMPLATE_FP:-data/srsa_axial_task_templates.json}
SRSA_MESH_GEOMETRY_FP=${SRSA_MESH_GEOMETRY_FP:-docs/srsa_mesh_geometry_params.csv}
SRSA_PARAM_TEMPLATE_ID=${SRSA_PARAM_TEMPLATE_ID:-2}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
LOG_ROOT=${LOG_ROOT:-${REPO_ROOT}/logs/train_${SOURCE_ID}_axial_hole/${RUN_STAMP}}

make_abs_path() {
  local path=$1
  if [[ "${path}" == "~" ]]; then
    path=${HOME}
  elif [[ "${path}" == "~/"* ]]; then
    path="${HOME}/${path#~/}"
  fi
  if [[ "${path}" == /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s/%s\n' "${REPO_ROOT}" "${path}"
  fi
}

SRSA_TASK_TEMPLATE_FP=$(make_abs_path "${SRSA_TASK_TEMPLATE_FP}")
SRSA_MESH_GEOMETRY_FP=$(make_abs_path "${SRSA_MESH_GEOMETRY_FP}")

if [[ ! -x "${PYTHON}" ]]; then
  echo "[launcher] python not found or not executable: ${PYTHON}" >&2
  exit 1
fi
if [[ ! -d "${ISAACLAB_DIR}" ]]; then
  echo "[launcher] IsaacLab dir not found: ${ISAACLAB_DIR}" >&2
  exit 1
fi
if [[ ! -d "${SRSA_DIR}" ]]; then
  echo "[launcher] SRSA dir not found: ${SRSA_DIR}" >&2
  exit 1
fi
if [[ ! -f "${SRSA_TASK_TEMPLATE_FP}" ]]; then
  echo "[launcher] SRSA task template not found: ${SRSA_TASK_TEMPLATE_FP}" >&2
  exit 1
fi
if [[ ! -f "${SRSA_MESH_GEOMETRY_FP}" ]]; then
  echo "[launcher] SRSA mesh geometry CSV not found: ${SRSA_MESH_GEOMETRY_FP}" >&2
  exit 1
fi
if [[ "${CHECK_CUDA:-1}" == "1" ]]; then
  AVAILABLE_GPUS=$(PYTHONWARNINGS=ignore "${PYTHON}" -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)')
  if [[ "${MULTIPROC}" == "true" ]]; then
    REQUIRED_GPUS=${NUM_GPUS}
  else
    REQUIRED_GPUS=1
  fi
  if (( AVAILABLE_GPUS - GPU_ID < REQUIRED_GPUS )); then
    echo "[launcher] CUDA not ready: available_gpus=${AVAILABLE_GPUS}, gpu_id=${GPU_ID}, required_gpus=${REQUIRED_GPUS}" >&2
    echo "[launcher] Fix the NVIDIA driver/CUDA visibility, or set CHECK_CUDA=0 to skip this preflight." >&2
    exit 1
  fi
fi

mkdir -p "${LOG_ROOT}"

echo "[launcher] log_root=${LOG_ROOT}"
echo "[launcher] repo_root=${REPO_ROOT}"
echo "[launcher] python=${PYTHON}"
echo "[launcher] isaaclab_dir=${ISAACLAB_DIR}"
echo "[launcher] srsa_dir=${SRSA_DIR}"
echo "[launcher] targets=${TARGETS}"
echo "[launcher] train_mode=retrain"
echo "[launcher] steps=${STEPS} num_envs=${NUM_ENVS} multiproc=${MULTIPROC} num_gpus=${NUM_GPUS} gpu_id=${GPU_ID}"
echo "[launcher] eval_freq=${EVAL_FREQ} save_freq=${SAVE_FREQ}"
echo "[launcher] exp_name=${EXP_NAME}"
echo "[launcher] srsa_task_template_fp=${SRSA_TASK_TEMPLATE_FP}"
echo "[launcher] srsa_mesh_geometry_fp=${SRSA_MESH_GEOMETRY_FP}"
echo "[launcher] srsa_param_template_id=${SRSA_PARAM_TEMPLATE_ID} eval_success_metric=relaxed"

for ASM in ${TARGETS}; do
  ASM_LOG="${LOG_ROOT}/asm-${ASM}.train.log"
  echo "[launcher] $(date --iso-8601=seconds) start assembly_id=${ASM} log=${ASM_LOG}"
  "${PYTHON}" tdmpc2/train.py \
    isaaclab_backend=srsa \
    task=isaaclab-srsa-assembly \
    assembly_id="${ASM}" \
    isaaclab_dir="${ISAACLAB_DIR}" \
    srsa_dir="${SRSA_DIR}" \
    srsa_task_template_fp="${SRSA_TASK_TEMPLATE_FP}" \
    srsa_mesh_geometry_fp="${SRSA_MESH_GEOMETRY_FP}" \
    srsa_param_template_id="${SRSA_PARAM_TEMPLATE_ID}" \
    eval_task_template_exact=true \
    eval_task_template_print=true \
    srsa_sparse_reward=false \
    isaaclab_disable_imitation_reward=false \
    srsa_align_direct_reward_success=true \
    srsa_if_sbc=false \
    num_envs="${NUM_ENVS}" \
    isaaclab_gpu_collision_stack_size=268435456 \
    gpu_id="${GPU_ID}" \
    multiproc="${MULTIPROC}" \
    num_gpus="${NUM_GPUS}" \
    steps="${STEPS}" \
    model_size=S \
    batch_size=1024 \
    buffer_size=10000000 \
    horizon=3 \
    utd=0.075 \
    use_demos=false \
    compile=false \
    enable_wandb=false \
    save_agent=true \
    save_best=true \
    save_best_metric=episode_success \
    mpc=true \
    isaaclab_headless=true \
    isaaclab_use_canonical_obs=true \
    srsa_task_family_name=normal_fit \
    srsa_task_param_obs=false \
    srsa_task_param_obs_mode=task_vec \
    srsa_enable_axial_task_param_sampler=true \
    srsa_axial_fixed_plug_scale=true \
    srsa_axial_clearance_base=0.000114 \
    'srsa_axial_clearance_depth_templates="0.5:0.5;0.5:1.0;1.0:1.0;2.0:1.5;4.0:2.0"' \
    srsa_axial_clearance_jitter_ratio=0.10 \
    srsa_axial_depth_base=0.015 \
    srsa_axial_depth_jitter_ratio=0.10 \
    'srsa_axial_init_error_xy_range="0.009,0.0010"' \
    'srsa_axial_init_error_z_range="0.0010,0.0020"' \
    'srsa_axial_init_error_yaw_range="-0.0872665,0.0872665"' \
    'srsa_axial_visual_noise_xy_range="0.0,0.0"' \
    'srsa_axial_visual_noise_z_range="0.0,0.0"' \
    srsa_enable_flange_force_sensor=true \
    isaaclab_canonical_append_force=true \
    isaaclab_canonical_append_task_params=false \
    srsa_vision_noise_xy_std=0.0 \
    srsa_vision_noise_xy_jitter_std=0.0 \
    srsa_vision_noise_z_std=0.0 \
    srsa_vision_noise_z_jitter_std=0.0 \
    isaaclab_canonical_use_visual_noise=false \
    task_conditioning=axial_params \
    eval_success_metric=relaxed \
    srsa_eval_success_metric=relaxed \
    strict_depth_fraction=0.90 \
    strict_success_steps=10 \
    strict_lateral_tol_min=0.0005 \
    strict_lateral_tol_max=0.0020 \
    strict_keypoint_tol_min=0.0010 \
    strict_keypoint_tol_max=0.0030 \
    strict_angle_tol_deg=3.0 \
    progress_log_interval_sec=30 \
    skip_initial_eval=true \
    eval_episodes=1 \
    eval_freq="${EVAL_FREQ}" \
    save_freq="${SAVE_FREQ}" \
    exp_name="${EXP_NAME}" \
    contact_history_enabled=true \
    contact_history_len=4 \
    contact_context_dim=64 \
    contact_history_hidden_dim=128 \
    contact_history_layers=2 \
    contact_force_dim=6 \
    contact_action_dim=3 \
    contact_ee_delta_dim=3 \
    contact_history_use_ee_delta=true \
    > "${ASM_LOG}" 2>&1
  echo "[launcher] $(date --iso-8601=seconds) done assembly_id=${ASM}"
done

echo "[launcher] $(date --iso-8601=seconds) all train jobs completed"
