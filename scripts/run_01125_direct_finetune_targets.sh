#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
cd "${REPO_ROOT}"

PYTHON=${PYTHON:-/home/gpuserver/miniconda3/envs/isaac51/bin/python}
ISAACLAB_DIR=${ISAACLAB_DIR:-/home/gpuserver/IsaacLab}
SRSA_DIR=${SRSA_DIR:-/home/gpuserver/hx/github/srsa}
CONFIG_NAME=${CONFIG_NAME:-srsa_01125_imitation_relaxed}
CONFIG_DIR=${CONFIG_DIR:-configs/train}
CHECKPOINT=${CHECKPOINT:-${REPO_ROOT}/logs/isaaclab-srsa-assembly/1/srsa_axial_imitation_relaxed/20260525_233657_asm-01125_tid-2/models/best.pt}

# Independent direct online fine-tune jobs. Each target starts from CHECKPOINT.
TARGETS=${TARGETS:-"00186 00256 00062 00271 00726 01079 01029 01092 01102"}
STEPS=${STEPS:-600000}
NUM_ENVS=${NUM_ENVS:-300}
NUM_GPUS=${NUM_GPUS:-2}
GPU_ID=${GPU_ID:-0}
MULTIPROC=${MULTIPROC:-true}
SEED=${SEED:-1}
BATCH_SIZE=${BATCH_SIZE:-1024}
BUFFER_SIZE=${BUFFER_SIZE:-6000000}
HORIZON=${HORIZON:-3}
UTD=${UTD:-0.075}
EVAL_FREQ=${EVAL_FREQ:-150000}
SAVE_FREQ=${SAVE_FREQ:-${EVAL_FREQ}}
EVAL_EPISODES=${EVAL_EPISODES:-1}
EVAL_SUCCESS_METRIC=${EVAL_SUCCESS_METRIC:-relaxed}
SAVE_BEST_METRIC=${SAVE_BEST_METRIC:-episode_success}

SRSA_TASK_TEMPLATE_FP=${SRSA_TASK_TEMPLATE_FP:-data/srsa_axial_task_templates.json}
SRSA_MESH_GEOMETRY_FP=${SRSA_MESH_GEOMETRY_FP:-data/srsa_mesh_geometry_params.csv}
SRSA_PARAM_TEMPLATE_ID=${SRSA_PARAM_TEMPLATE_ID:-2}
REFERENCE_ANCHOR_ID=${REFERENCE_ANCHOR_ID:-01125}
REFERENCE_ANCHOR_TYPE_ID=${REFERENCE_ANCHOR_TYPE_ID:-0}

EXP_NAME=${EXP_NAME:-srsa_axial_direct_finetune_from_01125}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
LOG_ROOT=${LOG_ROOT:-${REPO_ROOT}/logs/direct_finetune_01125_axial_hole/${RUN_STAMP}}
ENABLE_WANDB=${ENABLE_WANDB:-false}
DRY_RUN=${DRY_RUN:-0}
CHECK_CUDA=${CHECK_CUDA:-1}

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

print_command() {
  printf ' '
  printf '%q ' "$@"
  printf '\n'
}

CHECKPOINT=$(make_abs_path "${CHECKPOINT}")
CONFIG_DIR=$(make_abs_path "${CONFIG_DIR}")
SRSA_TASK_TEMPLATE_FP=$(make_abs_path "${SRSA_TASK_TEMPLATE_FP}")
SRSA_MESH_GEOMETRY_FP=$(make_abs_path "${SRSA_MESH_GEOMETRY_FP}")

if [[ ! -x "${PYTHON}" ]]; then
  echo "[launcher] python not found or not executable: ${PYTHON}" >&2
  exit 1
fi
if [[ ! -d "${CONFIG_DIR}" ]]; then
  echo "[launcher] config dir not found: ${CONFIG_DIR}" >&2
  exit 1
fi
if [[ ! -f "${CONFIG_DIR}/${CONFIG_NAME}.yaml" && ! -f "${CONFIG_DIR}/${CONFIG_NAME}.yml" ]]; then
  echo "[launcher] config not found: ${CONFIG_DIR}/${CONFIG_NAME}.{yaml,yml}" >&2
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
if [[ ! -f "${CHECKPOINT}" ]]; then
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[launcher] dry-run warning: checkpoint not found: ${CHECKPOINT}" >&2
  else
    echo "[launcher] checkpoint not found: ${CHECKPOINT}" >&2
    exit 1
  fi
fi
if [[ ! -f "${SRSA_TASK_TEMPLATE_FP}" ]]; then
  echo "[launcher] SRSA task template not found: ${SRSA_TASK_TEMPLATE_FP}" >&2
  exit 1
fi
if [[ ! -f "${SRSA_MESH_GEOMETRY_FP}" ]]; then
  echo "[launcher] SRSA mesh geometry CSV not found: ${SRSA_MESH_GEOMETRY_FP}" >&2
  exit 1
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
    echo "[launcher] Fix CUDA visibility, or set CHECK_CUDA=0 to skip this preflight." >&2
    exit 1
  fi
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${LOG_ROOT}"
fi

echo "[launcher] mode=direct-online-finetune"
echo "[launcher] repo_root=${REPO_ROOT}"
echo "[launcher] python=${PYTHON}"
echo "[launcher] config=${CONFIG_DIR}/${CONFIG_NAME}"
echo "[launcher] checkpoint=${CHECKPOINT}"
echo "[launcher] targets=${TARGETS}"
echo "[launcher] steps=${STEPS} num_envs=${NUM_ENVS} multiproc=${MULTIPROC} num_gpus=${NUM_GPUS} gpu_id=${GPU_ID}"
echo "[launcher] eval_freq=${EVAL_FREQ} save_freq=${SAVE_FREQ} eval_success_metric=${EVAL_SUCCESS_METRIC}"
echo "[launcher] exp_name=${EXP_NAME} log_root=${LOG_ROOT}"
echo "[launcher] dry_run=${DRY_RUN}"

for ASM in ${TARGETS}; do
  RUN_ID="${RUN_STAMP}_asm-${ASM}"
  ASM_LOG="${LOG_ROOT}/asm-${ASM}.train.log"
  echo "[launcher] $(date --iso-8601=seconds) start direct finetune assembly_id=${ASM}"
  echo "[launcher] run_id=${RUN_ID}"
  echo "[launcher] log=${ASM_LOG}"

  train_cmd=(
    "${PYTHON}" tdmpc2/train.py
    --config-dir "${CONFIG_DIR}"
    --config-name "${CONFIG_NAME}"
    checkpoint="${CHECKPOINT}"
    finetune=true
    seeding_coef=1
    assembly_id="${ASM}"
    isaaclab_dir="${ISAACLAB_DIR}"
    srsa_dir="${SRSA_DIR}"
    srsa_task_template_fp="${SRSA_TASK_TEMPLATE_FP}"
    srsa_mesh_geometry_fp="${SRSA_MESH_GEOMETRY_FP}"
    srsa_param_template_id="${SRSA_PARAM_TEMPLATE_ID}"
    srsa_axial_reference_anchor_assembly_id="${REFERENCE_ANCHOR_ID}"
    srsa_axial_reference_anchor_task_type_id="${REFERENCE_ANCHOR_TYPE_ID}"
    srsa_axial_recompute_manifest_task_vecs=true
    steps="${STEPS}"
    num_envs="${NUM_ENVS}"
    multiproc="${MULTIPROC}"
    num_gpus="${NUM_GPUS}"
    gpu_id="${GPU_ID}"
    seed="${SEED}"
    batch_size="${BATCH_SIZE}"
    buffer_size="${BUFFER_SIZE}"
    horizon="${HORIZON}"
    utd="${UTD}"
    eval_freq="${EVAL_FREQ}"
    save_freq="${SAVE_FREQ}"
    eval_episodes="${EVAL_EPISODES}"
    eval_success_metric="${EVAL_SUCCESS_METRIC}"
    srsa_eval_success_metric="${EVAL_SUCCESS_METRIC}"
    enable_wandb="${ENABLE_WANDB}"
    save_agent=true
    save_best=true
    save_best_metric="${SAVE_BEST_METRIC}"
    exp_name="${EXP_NAME}"
    run_id="${RUN_ID}"
  )

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] train command:"
    print_command "${train_cmd[@]}"
  else
    "${train_cmd[@]}" > "${ASM_LOG}" 2>&1
    echo "[launcher] $(date --iso-8601=seconds) done direct finetune assembly_id=${ASM}"
  fi
done

echo "[launcher] $(date --iso-8601=seconds) all direct finetune jobs completed"
