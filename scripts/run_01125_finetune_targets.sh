#!/usr/bin/env bash
set -euo pipefail

cd /home/gpuserver/hx/github/Newt

PYTHON=${PYTHON:-/home/gpuserver/miniconda3/envs/isaac51/bin/python}
CHECKPOINT=${CHECKPOINT:-/home/gpuserver/hx/github/Newt/logs/isaaclab-srsa-assembly/1/srsa_axial_online/20260523_163332_asm-01125/models/best.pt}
TARGETS=${TARGETS:-"00004 00014 00062 00271"}
STEPS=${STEPS:-600000}
NUM_ENVS=${NUM_ENVS:-300}
NUM_GPUS=${NUM_GPUS:-2}
GPU_ID=${GPU_ID:-0}
EVAL_FREQ=${EVAL_FREQ:-150000}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
LOG_ROOT=${LOG_ROOT:-/home/gpuserver/hx/github/Newt/logs/finetune_01125_axial_hole/${RUN_STAMP}}

mkdir -p "${LOG_ROOT}"

echo "[launcher] log_root=${LOG_ROOT}"
echo "[launcher] checkpoint=${CHECKPOINT}"
echo "[launcher] targets=${TARGETS}"
echo "[launcher] steps=${STEPS} num_envs=${NUM_ENVS} num_gpus=${NUM_GPUS} gpu_id=${GPU_ID}"

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
    srsa_dir=/home/gpuserver/hx/github/srsa \
    srsa_sparse_reward=false \
    isaaclab_disable_imitation_reward=false \
    srsa_if_sbc=false \
    num_envs="${NUM_ENVS}" \
    isaaclab_gpu_collision_stack_size=268435456 \
    gpu_id="${GPU_ID}" \
    multiproc=true \
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
    exp_name=srsa_axial_finetune_from_01125 \
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
