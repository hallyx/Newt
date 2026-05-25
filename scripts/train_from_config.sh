#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
cd "${REPO_ROOT}"

PYTHON=${PYTHON:-/home/gpuserver/miniconda3/envs/isaac51/bin/python}
CONFIG_FP=${1:-configs/train/srsa_01125_imitation_relaxed.yaml}
if [[ $# -gt 0 ]]; then
  shift
fi

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

CONFIG_FP=$(make_abs_path "${CONFIG_FP}")
if [[ ! -f "${CONFIG_FP}" ]]; then
  echo "[train_from_config] config not found: ${CONFIG_FP}" >&2
  exit 1
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "[train_from_config] python not found or not executable: ${PYTHON}" >&2
  exit 1
fi

CONFIG_DIR=$(dirname "${CONFIG_FP}")
CONFIG_NAME=$(basename "${CONFIG_FP}")
CONFIG_NAME=${CONFIG_NAME%.yaml}
CONFIG_NAME=${CONFIG_NAME%.yml}

exec "${PYTHON}" tdmpc2/train.py \
  --config-dir "${CONFIG_DIR}" \
  --config-name "${CONFIG_NAME}" \
  "$@"
