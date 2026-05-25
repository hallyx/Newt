#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
cd "${REPO_ROOT}"

PYTHON=${PYTHON:-/home/gpuserver/miniconda3/envs/isaac51/bin/python}
ISAACLAB_DIR=${ISAACLAB_DIR:-/home/gpuserver/IsaacLab}
SRSA_DIR=${SRSA_DIR:-/home/gpuserver/hx/github/srsa}
SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT:-${REPO_ROOT}/logs/isaaclab-srsa-assembly/1/srsa_axial_online/20260523_163332_asm-01125/models/best.pt}
TARGETS=${TARGETS:-"00004 00014 00062 00271"}
STEPS_PER_TASK=${STEPS_PER_TASK:-1000000}
HANDOFF=${HANDOFF:-latest}
RETENTION_EVAL_EPISODES=${RETENTION_EVAL_EPISODES:-200}
NUM_ENVS=${NUM_ENVS:-300}
RETENTION_NUM_ENVS=${RETENTION_NUM_ENVS:-${NUM_ENVS}}
NUM_GPUS=${NUM_GPUS:-2}
GPU_ID=${GPU_ID:-0}
MULTIPROC=${MULTIPROC:-true}
SEED=${SEED:-1}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
EXP_NAME=${EXP_NAME:-srsa_axial_continual_from_01125}
WORK_BASE=${REPO_ROOT}/logs/isaaclab-srsa-assembly/${SEED}/${EXP_NAME}
LOG_ROOT=${LOG_ROOT:-${WORK_BASE}/${RUN_STAMP}_launcher}
DRY_RUN=${DRY_RUN:-0}
CHECK_CUDA=${CHECK_CUDA:-1}

if [[ "${HANDOFF}" != "latest" ]]; then
  echo "[launcher] unsupported HANDOFF=${HANDOFF}; continual finetune handoff is fixed to latest" >&2
  exit 1
fi

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
if [[ ! -f "${SOURCE_CHECKPOINT}" ]]; then
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[launcher] dry-run warning: source checkpoint not found: ${SOURCE_CHECKPOINT}" >&2
  else
    echo "[launcher] source checkpoint not found: ${SOURCE_CHECKPOINT}" >&2
    exit 1
  fi
fi
if [[ "${DRY_RUN}" != "1" && "${CHECK_CUDA}" == "1" ]]; then
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

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${LOG_ROOT}"
fi

if [[ "${MULTIPROC}" == "true" ]]; then
  step_stride=$((NUM_ENVS * NUM_GPUS))
else
  step_stride=${NUM_ENVS}
fi
if (( step_stride <= 0 )); then
  echo "[launcher] invalid step stride: num_envs=${NUM_ENVS}, num_gpus=${NUM_GPUS}" >&2
  exit 1
fi
default_eval_freq=150000
if (( STEPS_PER_TASK < default_eval_freq )); then
  default_eval_freq=${STEPS_PER_TASK}
fi
if (( default_eval_freq < step_stride )); then
  default_eval_freq=${step_stride}
fi
if (( default_eval_freq % step_stride != 0 )); then
  default_eval_freq=$(((default_eval_freq / step_stride) * step_stride))
fi
if (( default_eval_freq <= 0 )); then
  default_eval_freq=${step_stride}
fi
EVAL_FREQ=${EVAL_FREQ:-${default_eval_freq}}
SAVE_FREQ=${SAVE_FREQ:-${EVAL_FREQ}}

print_command() {
  printf ' '
  printf '%q ' "$@"
  printf '\n'
}

join_eval_ids() {
  local joined="01125"
  local item
  for item in "$@"; do
    joined="${joined},${item}"
  done
  printf '[%s]' "${joined}"
}

check_anchor_retention() {
  local csv_fp=$1
  "${PYTHON}" - "${csv_fp}" <<'PY'
import csv
import sys

csv_fp = sys.argv[1]
with open(csv_fp, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

anchor = next((row for row in rows if str(row.get("assembly_id", "")).zfill(5) == "01125"), None)
if anchor is None:
    print(f"[launcher] retention warning: 01125 row not found in {csv_fp}")
    sys.exit(0)

strict = float(anchor.get("strict_success") or 0.0)
gap = anchor.get("official_strict_gap", "")
if strict < 0.65:
    print(
        "[launcher] forgetting risk: 01125 strict_success="
        f"{strict:.4f} < 0.65; official_strict_gap={gap}"
    )
else:
    print(
        "[launcher] anchor ok: 01125 strict_success="
        f"{strict:.4f}; official_strict_gap={gap}"
    )
PY
}

echo "[launcher] repo_root=${REPO_ROOT}"
echo "[launcher] python=${PYTHON}"
echo "[launcher] isaaclab_dir=${ISAACLAB_DIR}"
echo "[launcher] srsa_dir=${SRSA_DIR}"
echo "[launcher] source_checkpoint=${SOURCE_CHECKPOINT}"
echo "[launcher] targets=${TARGETS}"
echo "[launcher] steps_per_task=${STEPS_PER_TASK}"
echo "[launcher] handoff=${HANDOFF}"
echo "[launcher] retention_eval_episodes=${RETENTION_EVAL_EPISODES}"
echo "[launcher] num_envs=${NUM_ENVS} retention_num_envs=${RETENTION_NUM_ENVS} multiproc=${MULTIPROC} num_gpus=${NUM_GPUS} gpu_id=${GPU_ID}"
echo "[launcher] eval_freq=${EVAL_FREQ} save_freq=${SAVE_FREQ}"
echo "[launcher] exp_name=${EXP_NAME}"
echo "[launcher] work_base=${WORK_BASE}"
echo "[launcher] log_root=${LOG_ROOT}"
echo "[launcher] dry_run=${DRY_RUN}"

current_checkpoint=${SOURCE_CHECKPOINT}
completed_targets=()
stage_idx=0

for ASM in ${TARGETS}; do
  stage_idx=$((stage_idx + 1))
  RUN_ID="${RUN_STAMP}_stage-${stage_idx}_asm-${ASM}"
  stage_work_dir="${WORK_BASE}/${RUN_ID}"
  stage_log="${LOG_ROOT}/stage-${stage_idx}_asm-${ASM}.train.log"

  echo "[launcher] $(date --iso-8601=seconds) start stage=${stage_idx} assembly_id=${ASM}"
  echo "[launcher] stage_run_id=${RUN_ID}"
  echo "[launcher] stage_work_dir=${stage_work_dir}"
  echo "[launcher] stage_checkpoint_in=${current_checkpoint}"
  echo "[launcher] stage_log=${stage_log}"

  train_cmd=(
    "${PYTHON}" tdmpc2/train.py
    checkpoint="${current_checkpoint}"
    finetune=true
    seeding_coef=1
    isaaclab_backend=srsa
    task=isaaclab-srsa-assembly
    assembly_id="${ASM}"
    isaaclab_dir="${ISAACLAB_DIR}"
    srsa_dir="${SRSA_DIR}"
    srsa_sparse_reward=false
    isaaclab_disable_imitation_reward=false
    srsa_if_sbc=false
    num_envs="${NUM_ENVS}"
    isaaclab_gpu_collision_stack_size=268435456
    gpu_id="${GPU_ID}"
    multiproc="${MULTIPROC}"
    num_gpus="${NUM_GPUS}"
    steps="${STEPS_PER_TASK}"
    model_size=S
    batch_size=1024
    buffer_size=6000000
    horizon=3
    utd=0.075
    use_demos=false
    compile=false
    enable_wandb=false
    save_agent=true
    save_best=true
    save_best_metric=episode_success
    mpc=true
    isaaclab_headless=true
    isaaclab_use_canonical_obs=true
    srsa_task_family_name=normal_fit
    srsa_task_param_obs=false
    srsa_task_param_obs_mode=task_vec
    srsa_enable_axial_task_param_sampler=true
    srsa_axial_fixed_plug_scale=true
    srsa_axial_clearance_base=0.000114
    'srsa_axial_clearance_depth_templates="0.5:0.5;0.5:1.0;1.0:1.0;2.0:1.5;4.0:2.0"'
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
    eval_success_metric=strict
    strict_depth_fraction=0.90
    strict_success_steps=10
    strict_lateral_tol_min=0.0005
    strict_lateral_tol_max=0.0020
    strict_keypoint_tol_min=0.0010
    strict_keypoint_tol_max=0.0030
    strict_angle_tol_deg=3.0
    progress_log_interval_sec=30
    skip_initial_eval=true
    eval_episodes=1
    eval_freq="${EVAL_FREQ}"
    save_freq="${SAVE_FREQ}"
    exp_name="${EXP_NAME}"
    run_id="${RUN_ID}"
    seed="${SEED}"
    contact_history_enabled=true
    contact_history_len=4
    contact_context_dim=64
    contact_history_hidden_dim=128
    contact_history_layers=2
    contact_force_dim=6
    contact_action_dim=3
    contact_ee_delta_dim=3
    contact_history_use_ee_delta=true
  )

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] train command:"
    print_command "${train_cmd[@]}"
  else
    "${train_cmd[@]}" > "${stage_log}" 2>&1
  fi

  next_checkpoint="${stage_work_dir}/models/latest.pt"
  if [[ "${DRY_RUN}" != "1" && ! -f "${next_checkpoint}" ]]; then
    echo "[launcher] expected handoff checkpoint not found: ${next_checkpoint}" >&2
    echo "[launcher] Check ${stage_log}; if training finished before an eval point, lower EVAL_FREQ." >&2
    exit 1
  fi
  current_checkpoint=${next_checkpoint}
  completed_targets+=("${ASM}")

  eval_ids=$(join_eval_ids "${completed_targets[@]}")
  retention_dir="${LOG_ROOT}/retention_after_${ASM}"
  retention_log="${retention_dir}/batch_eval.log"
  retention_summary="${retention_dir}/batch_eval_summary.json"
  if [[ "${DRY_RUN}" != "1" ]]; then
    mkdir -p "${retention_dir}"
  fi

  echo "[launcher] $(date --iso-8601=seconds) stage=${stage_idx} complete; handoff_checkpoint=${current_checkpoint}"
  echo "[launcher] retention_eval_ids=${eval_ids}"
  echo "[launcher] retention_dir=${retention_dir}"

  eval_cmd=(
    "${PYTHON}" tdmpc2/batch_eval_tasks.py
    checkpoint="${current_checkpoint}"
    eval_assembly_ids="${eval_ids}"
    isaaclab_backend=srsa
    task=isaaclab-srsa-assembly
    isaaclab_dir="${ISAACLAB_DIR}"
    srsa_dir="${SRSA_DIR}"
    srsa_sparse_reward=false
    isaaclab_disable_imitation_reward=false
    srsa_if_sbc=false
    num_envs="${RETENTION_NUM_ENVS}"
    isaaclab_gpu_collision_stack_size=268435456
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
    'srsa_axial_clearance_depth_templates="0.5:0.5;0.5:1.0;1.0:1.0;2.0:1.5;4.0:2.0"'
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
    eval_success_metric=strict
    strict_depth_fraction=0.90
    strict_success_steps=10
    strict_lateral_tol_min=0.0005
    strict_lateral_tol_max=0.0020
    strict_keypoint_tol_min=0.0010
    strict_keypoint_tol_max=0.0030
    strict_angle_tol_deg=3.0
    batch_eval_episodes_per_task="${RETENTION_EVAL_EPISODES}"
    batch_eval_spawn_per_assembly=true
    batch_eval_overwrite=true
    batch_eval_output_dir="${retention_dir}"
    batch_eval_summary_fp="${retention_summary}"
    enable_wandb=false
    exp_name="${EXP_NAME}"
    run_id="${RUN_ID}_retention_after_${ASM}"
    seed="${SEED}"
    contact_history_enabled=true
    contact_history_len=4
    contact_context_dim=64
    contact_history_hidden_dim=128
    contact_history_layers=2
    contact_force_dim=6
    contact_action_dim=3
    contact_ee_delta_dim=3
    contact_history_use_ee_delta=true
  )

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] batch eval command:"
    print_command "${eval_cmd[@]}"
  else
    "${eval_cmd[@]}" > "${retention_log}" 2>&1
    if [[ ! -f "${retention_summary}" || ! -f "${retention_summary%.json}.csv" ]]; then
      echo "[launcher] retention eval output missing under ${retention_dir}" >&2
      echo "[launcher] Check ${retention_log}" >&2
      exit 1
    fi
    check_anchor_retention "${retention_summary%.json}.csv"
  fi
done

echo "[launcher] $(date --iso-8601=seconds) all continual finetune stages completed"
echo "[launcher] final_checkpoint=${current_checkpoint}"
