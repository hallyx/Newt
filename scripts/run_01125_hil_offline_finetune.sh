#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
cd "${REPO_ROOT}"

PYTHON=${PYTHON:-/home/gpuserver/miniconda3/envs/isaac51/bin/python}
CHECKPOINT=${CHECKPOINT:-${REPO_ROOT}/logs/isaaclab-srsa-assembly/1/srsa_axial_online/20260523_163332_asm-01125/models/best.pt}
OFFLINE_DATASET_FP=${OFFLINE_DATASET_FP:-data/real_hil_01125_clean.pt}
BC_STEPS=${BC_STEPS:-20000}
WM_STEPS=${WM_STEPS:-10000}
RL_STEPS=${RL_STEPS:-0}
BATCH_SIZE=${BATCH_SIZE:-256}
HORIZON=${HORIZON:-3}
GPU_ID=${GPU_ID:-0}
OFFLINE_GPU_ID=${OFFLINE_GPU_ID:-${GPU_ID}}
ENABLE_WANDB=${ENABLE_WANDB:-false}
EXP_NAME=${EXP_NAME:-real_01125_hil_offline_ft}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
LOG_ROOT=${LOG_ROOT:-${REPO_ROOT}/logs/real_hil_01125_offline_ft/${RUN_STAMP}}
TRAIN_LOG=${TRAIN_LOG:-${LOG_ROOT}/offline_train.log}

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

CHECKPOINT=$(make_abs_path "${CHECKPOINT}")
OFFLINE_DATASET_FP=$(make_abs_path "${OFFLINE_DATASET_FP}")

if [[ ! -x "${PYTHON}" ]]; then
  echo "[launcher] python not found or not executable: ${PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[launcher] checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${OFFLINE_DATASET_FP}" ]]; then
  echo "[launcher] clean HIL dataset not found: ${OFFLINE_DATASET_FP}" >&2
  echo "[launcher] Build it first with tdmpc2/scripts/clean_real_hil_dataset.py." >&2
  exit 1
fi

mkdir -p "${LOG_ROOT}"

echo "[launcher] log_root=${LOG_ROOT}"
echo "[launcher] checkpoint=${CHECKPOINT}"
echo "[launcher] offline_dataset_fp=${OFFLINE_DATASET_FP}"
echo "[launcher] bc_steps=${BC_STEPS} wm_steps=${WM_STEPS} rl_steps=${RL_STEPS}"
echo "[launcher] batch_size=${BATCH_SIZE} horizon=${HORIZON} gpu_id=${GPU_ID} offline_gpu_id=${OFFLINE_GPU_ID}"
echo "[launcher] train_log=${TRAIN_LOG}"

train_cmd=(
  "${PYTHON}" tdmpc2/offline_train.py
  --config-dir configs/train
  --config-name srsa_01125_imitation_relaxed
  checkpoint="${CHECKPOINT}"
  offline_dataset_fp="${OFFLINE_DATASET_FP}"
  offline_bc_steps="${BC_STEPS}"
  offline_wm_steps="${WM_STEPS}"
  offline_rl_steps="${RL_STEPS}"
  offline_bc_filter_mode=success_or_high_depth
  offline_wm_filter_mode=all
  offline_rl_filter_mode=all
  batch_size="${BATCH_SIZE}"
  horizon="${HORIZON}"
  gpu_id="${GPU_ID}"
  offline_gpu_id="${OFFLINE_GPU_ID}"
  task_balanced_sampling=false
  compile=false
  enable_wandb="${ENABLE_WANDB}"
  save_agent=true
  save_best=false
  exp_name="${EXP_NAME}"
)

printf '[launcher] command:'
printf ' %q' "${train_cmd[@]}"
printf '\n'

echo "[launcher] $(date --iso-8601=seconds) start conservative 01125 HIL offline fine-tune"
"${train_cmd[@]}" > "${TRAIN_LOG}" 2>&1
echo "[launcher] $(date --iso-8601=seconds) done conservative 01125 HIL offline fine-tune"
echo "[launcher] log saved to ${TRAIN_LOG}"
