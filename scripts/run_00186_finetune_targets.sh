#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
cd "${REPO_ROOT}"

PYTHON=${PYTHON:-/home/robot2/miniconda3/envs/env_isaaclab/bin/python}
ISAACLAB_DIR=${ISAACLAB_DIR:-/home/robot2/isaaclab/IsaacLab}
SRSA_DIR=${SRSA_DIR:-/home/robot2/hx/github/srsa}
SOURCE_ID=${SOURCE_ID:-00186}
CHECKPOINT=${CHECKPOINT:-${REPO_ROOT}/logs/isaaclab-srsa-assembly/1/srsa_axial_online/20260523_214912_asm-${SOURCE_ID}/models/best.pt}
TARGETS=${TARGETS:-"00308 00581 00190 00422"}
STEPS=${STEPS:-600000}
NUM_ENVS=${NUM_ENVS:-400}
MULTIPROC=${MULTIPROC:-false}
NUM_GPUS=${NUM_GPUS:-1}
GPU_ID=${GPU_ID:-0}
EVAL_FREQ=${EVAL_FREQ:-150000}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
LOG_ROOT=${LOG_ROOT:-${REPO_ROOT}/logs/finetune_${SOURCE_ID}_axial_hole/${RUN_STAMP}}

if [[ ! -x "${PYTHON}" ]]; then
  echo "[launcher] python not found or not executable: ${PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[launcher] checkpoint not found: ${CHECKPOINT}" >&2
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
echo "[launcher] checkpoint=${CHECKPOINT}"
echo "[launcher] targets=${TARGETS}"
echo "[launcher] steps=${STEPS} num_envs=${NUM_ENVS} multiproc=${MULTIPROC} num_gpus=${NUM_GPUS} gpu_id=${GPU_ID}"

for ASM in ${TARGETS}; do
  ASM_LOG="${LOG_ROOT}/asm-${ASM}.log"
  echo "[launcher] $(date --iso-8601=seconds) start assembly_id=${ASM} log=${ASM_LOG}"
  "${PYTHON}" tdmpc2/train.py \
    checkpoint="${CHECKPOINT}" \
    finetune=true \
    seeding_coef=1 \
    isaaclab_backend=srsa \
    task=isaaclab-srsa-assembly \
    assembly_id="${ASM}" \
    isaaclab_dir="${ISAACLAB_DIR}" \
    srsa_dir="${SRSA_DIR}" \
    srsa_sparse_reward=false \
    isaaclab_disable_imitation_reward=false \
    srsa_if_sbc=false \
    num_envs="${NUM_ENVS}" \
    isaaclab_gpu_collision_stack_size=268435456 \
    gpu_id="${GPU_ID}" \
    multiproc="${MULTIPROC}" \
    num_gpus="${NUM_GPUS}" \
    steps="${STEPS}" \
    model_size=S \
    batch_size=1024 \
    buffer_size=6000000 \
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
    eval_success_metric=strict \
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
    save_freq="${EVAL_FREQ}" \
    exp_name=srsa_axial_finetune_from_${SOURCE_ID} \
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

echo "[launcher] $(date --iso-8601=seconds) all finetune jobs completed"
