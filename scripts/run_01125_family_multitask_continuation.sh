#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
cd "${REPO_ROOT}"

PYTHON=${PYTHON:-/home/gpuserver/miniconda3/envs/isaac51/bin/python}
ISAACLAB_DIR=${ISAACLAB_DIR:-/home/gpuserver/IsaacLab}
SRSA_DIR=${SRSA_DIR:-/home/gpuserver/hx/github/srsa}
CHECKPOINT=${CHECKPOINT:-${REPO_ROOT}/logs/isaaclab-srsa-assembly/1/srsa_axial_online/20260523_163332_asm-01125/models/best.pt}

# Scheme A trains one shared model. Include the anchor task in TASK_IDS.
TASK_IDS=${TASK_IDS:-"01125 00004 00014 00062 00271"}
EVAL_TASK_IDS=${EVAL_TASK_IDS:-${TASK_IDS}}
ANCHOR_TASK_ID=${ANCHOR_TASK_ID:-01125}

# checkpoint is only the warm-start model .pt. The replay manifest is a data .json.
# OFFLINE_MANIFEST_FP is kept as a backwards-compatible alias for the clearer name.
OFFLINE_MANIFEST_FP=${OFFLINE_MANIFEST_FP:-}
MULTITASK_REPLAY_MANIFEST_FP=${MULTITASK_REPLAY_MANIFEST_FP:-${OFFLINE_MANIFEST_FP}}
MULTITASK_AUTO_COLLECT_REPLAY=${MULTITASK_AUTO_COLLECT_REPLAY:-false}
OFFLINE_DATASET_FP=${OFFLINE_DATASET_FP:-}
OFFLINE_SOURCE_FP=${OFFLINE_SOURCE_FP:-}
OFFLINE_EXPORT_FP=${OFFLINE_EXPORT_FP:-}
OFFLINE_EXPORT_OVERWRITE=${OFFLINE_EXPORT_OVERWRITE:-false}
OFFLINE_FILTER_MODE=${OFFLINE_FILTER_MODE:-all}

TOTAL_STEPS=${TOTAL_STEPS:-5000000}
STAGE_STEPS=${STAGE_STEPS:-1000000}
CURRICULUM_MODE=${CURRICULUM_MODE:-progressive}
SAMPLING_MODE=${SAMPLING_MODE:-balanced}
TASK_SAMPLING_WEIGHTS=${TASK_SAMPLING_WEIGHTS:-}
ANCHOR_MIN_RATIO=${ANCHOR_MIN_RATIO:-0.2}
NEW_TASK_MIN_RATIO=${NEW_TASK_MIN_RATIO:-0.2}
HARD_CASE_RATIO=${HARD_CASE_RATIO:-0.2}
MULTITASK_EVAL_INTERVAL=${MULTITASK_EVAL_INTERVAL:-50000}
MULTITASK_NO_FORGETTING_MAX_FORGETTING=${MULTITASK_NO_FORGETTING_MAX_FORGETTING:-0.05}
MULTITASK_PROX_REG_ENABLED=${MULTITASK_PROX_REG_ENABLED:-false}
MULTITASK_PROX_REG_COEF=${MULTITASK_PROX_REG_COEF:-1e-4}

SRSA_TASK_TEMPLATE_FP=${SRSA_TASK_TEMPLATE_FP:-data/srsa_axial_task_templates.json}
SRSA_MESH_GEOMETRY_FP=${SRSA_MESH_GEOMETRY_FP:-data/srsa_mesh_geometry_params.csv}
SRSA_PARAM_TEMPLATE_ID=${SRSA_PARAM_TEMPLATE_ID:-2}
REFERENCE_ANCHOR_ID=${REFERENCE_ANCHOR_ID:-01125}
REFERENCE_ANCHOR_TYPE_ID=${REFERENCE_ANCHOR_TYPE_ID:-0}
EVAL_SUCCESS_METRIC=${EVAL_SUCCESS_METRIC:-strict}
BATCH_EVAL_EPISODES_PER_TASK=${BATCH_EVAL_EPISODES_PER_TASK:-100}

GPU_ID=${GPU_ID:-0}
NUM_GPUS=${NUM_GPUS:-1}
NUM_ENVS=${NUM_ENVS:-200}
MODEL_SIZE=${MODEL_SIZE:-S}
BATCH_SIZE=${BATCH_SIZE:-1024}
HORIZON=${HORIZON:-3}
COLLECT_EPISODES_PER_TASK=${COLLECT_EPISODES_PER_TASK:-300}
COLLECT_PARALLEL_WORKERS=${COLLECT_PARALLEL_WORKERS:-1}
COLLECT_PARALLEL_GPU_IDS=${COLLECT_PARALLEL_GPU_IDS:-}
COLLECT_MAX_ENV_STEPS=${COLLECT_MAX_ENV_STEPS:-}
OFFLINE_LOG_FREQ=${OFFLINE_LOG_FREQ:-200}
OFFLINE_SAVE_FREQ=${OFFLINE_SAVE_FREQ:-5000}
EXP_NAME=${EXP_NAME:-srsa_axial_family_continuation}
RUN_ID=${RUN_ID:-}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
LOG_ROOT=${LOG_ROOT:-${REPO_ROOT}/logs/family_multitask_01125_axial_hole/${RUN_STAMP}}
DRY_RUN=${DRY_RUN:-0}
CHECK_CUDA=${CHECK_CUDA:-1}
ENABLE_WANDB=${ENABLE_WANDB:-false}

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

json_list() {
  local out="["
  local sep=""
  local item
  for item in "$@"; do
    out="${out}${sep}\"${item}\""
    sep=","
  done
  out="${out}]"
  printf '%s\n' "${out}"
}

print_command() {
  printf ' '
  printf '%q ' "$@"
  printf '\n'
}

CHECKPOINT=$(make_abs_path "${CHECKPOINT}")
SRSA_TASK_TEMPLATE_FP=$(make_abs_path "${SRSA_TASK_TEMPLATE_FP}")
SRSA_MESH_GEOMETRY_FP=$(make_abs_path "${SRSA_MESH_GEOMETRY_FP}")
if [[ -n "${MULTITASK_REPLAY_MANIFEST_FP}" ]]; then
  MULTITASK_REPLAY_MANIFEST_FP=$(make_abs_path "${MULTITASK_REPLAY_MANIFEST_FP}")
  OFFLINE_MANIFEST_FP="${MULTITASK_REPLAY_MANIFEST_FP}"
fi
if [[ -n "${OFFLINE_DATASET_FP}" ]]; then
  OFFLINE_DATASET_FP=$(make_abs_path "${OFFLINE_DATASET_FP}")
fi
if [[ -n "${OFFLINE_SOURCE_FP}" ]]; then
  OFFLINE_SOURCE_FP=$(make_abs_path "${OFFLINE_SOURCE_FP}")
fi
if [[ -n "${OFFLINE_EXPORT_FP}" ]]; then
  OFFLINE_EXPORT_FP=$(make_abs_path "${OFFLINE_EXPORT_FP}")
fi

read -r -a TASK_ID_ARRAY <<< "${TASK_IDS}"
read -r -a EVAL_TASK_ID_ARRAY <<< "${EVAL_TASK_IDS}"
TASK_IDS_JSON=$(json_list "${TASK_ID_ARRAY[@]}")
EVAL_TASK_IDS_JSON=$(json_list "${EVAL_TASK_ID_ARRAY[@]}")

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
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[launcher] checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi
if [[ "${CHECKPOINT}" != *.pt ]]; then
  echo "[launcher] checkpoint must point to an initialization model .pt file, got: ${CHECKPOINT}" >&2
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
if [[ -z "${MULTITASK_REPLAY_MANIFEST_FP}" && "${MULTITASK_AUTO_COLLECT_REPLAY}" != "true" ]]; then
  echo "[launcher] Offline multitask continuation requires a replay manifest. To train without an existing manifest, enable online rollout collection or run the replay collection script first." >&2
  echo "[launcher] To build a manifest from existing per-task rollouts, run:" >&2
  echo "[launcher]   ${PYTHON} tdmpc2/scripts/build_family_offline_manifest.py --assembly-ids ${TASK_IDS} --source-template '/path/to/policy_rollouts/{assembly_id}/policy_eval_rollouts.pt' --output-manifest-fp data/offline_manifest_01125_family_multitask.json --srsa-mesh-geometry-fp docs/srsa_mesh_geometry_params.csv --expected-obs-dim 17 --expected-action-dim 3 --overwrite" >&2
  exit 1
fi
if [[ -n "${MULTITASK_REPLAY_MANIFEST_FP}" && "${MULTITASK_REPLAY_MANIFEST_FP}" != *.json ]]; then
  echo "[launcher] manifest should point to a replay/data manifest json, not a model checkpoint: ${MULTITASK_REPLAY_MANIFEST_FP}" >&2
  echo "[launcher] Use checkpoint=/.../models/best.pt and MULTITASK_REPLAY_MANIFEST_FP=/.../data/offline_manifest_family.json." >&2
  exit 1
fi
if [[ -n "${MULTITASK_REPLAY_MANIFEST_FP}" && "${MULTITASK_REPLAY_MANIFEST_FP}" == */models/* ]]; then
  echo "[launcher] manifest should point to a replay/data manifest json, not a model checkpoint: ${MULTITASK_REPLAY_MANIFEST_FP}" >&2
  echo "[launcher] Use checkpoint=/.../models/best.pt and MULTITASK_REPLAY_MANIFEST_FP=/.../data/offline_manifest_family.json." >&2
  exit 1
fi
if [[ -n "${MULTITASK_REPLAY_MANIFEST_FP}" && ! -f "${MULTITASK_REPLAY_MANIFEST_FP}" ]]; then
  echo "[launcher] multitask replay manifest not found: ${MULTITASK_REPLAY_MANIFEST_FP}" >&2
  exit 1
fi
if [[ -n "${OFFLINE_DATASET_FP}" && ! -f "${OFFLINE_DATASET_FP}" ]]; then
  echo "[launcher] offline compact dataset not found: ${OFFLINE_DATASET_FP}" >&2
  exit 1
fi
if [[ -n "${OFFLINE_SOURCE_FP}" && ! -f "${OFFLINE_SOURCE_FP}" ]]; then
  echo "[launcher] offline source dataset not found: ${OFFLINE_SOURCE_FP}" >&2
  exit 1
fi
if [[ "${DRY_RUN}" != "1" && "${CHECK_CUDA}" == "1" ]]; then
  AVAILABLE_GPUS=$(PYTHONWARNINGS=ignore "${PYTHON}" -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)')
  REQUIRED_GPUS=${NUM_GPUS}
  if (( REQUIRED_GPUS < 1 )); then
    REQUIRED_GPUS=1
  fi
  if (( AVAILABLE_GPUS - GPU_ID < REQUIRED_GPUS )); then
    echo "[launcher] CUDA not ready: available_gpus=${AVAILABLE_GPUS}, gpu_id=${GPU_ID}, required_gpus=${REQUIRED_GPUS}" >&2
    echo "[launcher] Fix CUDA visibility, or set CHECK_CUDA=0 to skip this preflight." >&2
    exit 1
  fi
fi

mkdir -p "${LOG_ROOT}"
TRAIN_LOG="${LOG_ROOT}/family_multitask_continuation.log"

echo "[launcher] log_root=${LOG_ROOT}"
echo "[launcher] repo_root=${REPO_ROOT}"
echo "[launcher] python=${PYTHON}"
echo "[launcher] isaaclab_dir=${ISAACLAB_DIR}"
echo "[launcher] srsa_dir=${SRSA_DIR}"
echo "[launcher] checkpoint=${CHECKPOINT}"
echo "[launcher] multitask_replay_manifest_fp=${MULTITASK_REPLAY_MANIFEST_FP}"
echo "[launcher] multitask_auto_collect_replay=${MULTITASK_AUTO_COLLECT_REPLAY}"
echo "[launcher] offline_dataset_fp=${OFFLINE_DATASET_FP}"
echo "[launcher] offline_source_fp=${OFFLINE_SOURCE_FP}"
echo "[launcher] task_ids=${TASK_IDS_JSON}"
echo "[launcher] eval_task_ids=${EVAL_TASK_IDS_JSON}"
echo "[launcher] anchor_task_id=${ANCHOR_TASK_ID}"
echo "[launcher] curriculum=${CURRICULUM_MODE} stage_steps=${STAGE_STEPS} total_steps=${TOTAL_STEPS}"
echo "[launcher] sampling=${SAMPLING_MODE} anchor_min=${ANCHOR_MIN_RATIO} new_min=${NEW_TASK_MIN_RATIO} hard_case=${HARD_CASE_RATIO}"
echo "[launcher] eval_interval=${MULTITASK_EVAL_INTERVAL} eval_episodes_per_task=${BATCH_EVAL_EPISODES_PER_TASK}"
echo "[launcher] num_envs=${NUM_ENVS} collect_episodes_per_task=${COLLECT_EPISODES_PER_TASK}"
echo "[launcher] collect_parallel_workers=${COLLECT_PARALLEL_WORKERS} collect_parallel_gpu_ids=${COLLECT_PARALLEL_GPU_IDS:-auto-from-gpu_id-num_gpus} num_gpus=${NUM_GPUS} gpu_id=${GPU_ID}"
echo "[launcher] srsa_task_template_fp=${SRSA_TASK_TEMPLATE_FP}"
echo "[launcher] srsa_mesh_geometry_fp=${SRSA_MESH_GEOMETRY_FP}"
echo "[launcher] srsa_param_template_id=${SRSA_PARAM_TEMPLATE_ID}"
echo "[launcher] reference_anchor=${REFERENCE_ANCHOR_ID} type=${REFERENCE_ANCHOR_TYPE_ID}"
echo "[launcher] train_log=${TRAIN_LOG}"
echo "[launcher] dry_run=${DRY_RUN}"

train_cmd=(
  "${PYTHON}" tdmpc2/train.py
  checkpoint="${CHECKPOINT}"
  multitask_reference_checkpoint_path="${CHECKPOINT}"
  multitask_continuation_enabled=true
  multitask_auto_collect_replay="${MULTITASK_AUTO_COLLECT_REPLAY}"
  multitask_task_ids="${TASK_IDS_JSON}"
  multitask_anchor_task_id="${ANCHOR_TASK_ID}"
  multitask_curriculum_mode="${CURRICULUM_MODE}"
  multitask_stage_steps="${STAGE_STEPS}"
  multitask_total_steps="${TOTAL_STEPS}"
  multitask_sampling_mode="${SAMPLING_MODE}"
  multitask_anchor_min_ratio="${ANCHOR_MIN_RATIO}"
  multitask_new_task_min_ratio="${NEW_TASK_MIN_RATIO}"
  multitask_hard_case_ratio="${HARD_CASE_RATIO}"
  multitask_eval_task_ids="${EVAL_TASK_IDS_JSON}"
  multitask_eval_interval="${MULTITASK_EVAL_INTERVAL}"
  multitask_save_per_task_metrics=true
  multitask_forgetting_metric_enabled=true
  multitask_no_forgetting_max_forgetting="${MULTITASK_NO_FORGETTING_MAX_FORGETTING}"
  multitask_prox_reg_enabled="${MULTITASK_PROX_REG_ENABLED}"
  multitask_prox_reg_coef="${MULTITASK_PROX_REG_COEF}"
  multitask_distill_old_policy_enabled=false
  isaaclab_backend=srsa
  task=isaaclab-srsa-assembly
  assembly_id="${ANCHOR_TASK_ID}"
  isaaclab_dir="${ISAACLAB_DIR}"
  srsa_dir="${SRSA_DIR}"
  srsa_task_template_fp="${SRSA_TASK_TEMPLATE_FP}"
  srsa_mesh_geometry_fp="${SRSA_MESH_GEOMETRY_FP}"
  srsa_param_template_id="${SRSA_PARAM_TEMPLATE_ID}"
  eval_task_template_exact=true
  eval_task_template_print=true
  srsa_sparse_reward=false
  isaaclab_disable_imitation_reward=false
  srsa_align_direct_reward_success=true
  srsa_if_sbc=false
  gpu_id="${GPU_ID}"
  num_gpus="${NUM_GPUS}"
  num_envs="${NUM_ENVS}"
  multiproc=false
  steps="${TOTAL_STEPS}"
  model_size="${MODEL_SIZE}"
  batch_size="${BATCH_SIZE}"
  horizon="${HORIZON}"
  offline_filter_mode="${OFFLINE_FILTER_MODE}"
  offline_log_freq="${OFFLINE_LOG_FREQ}"
  offline_save_freq="${OFFLINE_SAVE_FREQ}"
  task_balanced_sampling=false
  finetune=true
  use_demos=false
  compile=false
  enable_wandb="${ENABLE_WANDB}"
  save_agent=true
  save_best=false
  mpc=true
  isaaclab_headless=true
  isaaclab_use_canonical_obs=true
  srsa_task_family_name=normal_fit
  srsa_task_param_obs=false
  srsa_task_param_obs_mode=task_vec
  srsa_enable_axial_task_param_sampler=true
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
  srsa_axial_reference_anchor_assembly_id="${REFERENCE_ANCHOR_ID}"
  srsa_axial_reference_anchor_task_type_id="${REFERENCE_ANCHOR_TYPE_ID}"
  srsa_axial_recompute_manifest_task_vecs=true
  eval_success_metric="${EVAL_SUCCESS_METRIC}"
  srsa_eval_success_metric="${EVAL_SUCCESS_METRIC}"
  strict_depth_fraction=0.90
  strict_success_steps=10
  strict_lateral_tol_min=0.0005
  strict_lateral_tol_max=0.0020
  strict_keypoint_tol_min=0.0010
  strict_keypoint_tol_max=0.0030
  strict_angle_tol_deg=3.0
  batch_eval_episodes_per_task="${BATCH_EVAL_EPISODES_PER_TASK}"
  batch_eval_overwrite=true
  collect_episodes_per_task="${COLLECT_EPISODES_PER_TASK}"
  collect_spawn_per_assembly=true
  collect_parallel_workers="${COLLECT_PARALLEL_WORKERS}"
  collect_match_checkpoint=true
  progress_log_interval_sec=30
  exp_name="${EXP_NAME}"
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

if [[ -n "${RUN_ID}" ]]; then
  train_cmd+=(run_id="${RUN_ID}")
fi
if [[ -n "${MULTITASK_REPLAY_MANIFEST_FP}" ]]; then
  train_cmd+=(multitask_replay_manifest_fp="${MULTITASK_REPLAY_MANIFEST_FP}")
fi
if [[ -n "${OFFLINE_DATASET_FP}" ]]; then
  train_cmd+=(offline_dataset_fp="${OFFLINE_DATASET_FP}")
fi
if [[ -n "${OFFLINE_SOURCE_FP}" ]]; then
  train_cmd+=(offline_source_fp="${OFFLINE_SOURCE_FP}")
fi
if [[ -n "${OFFLINE_EXPORT_FP}" ]]; then
  train_cmd+=(offline_export_fp="${OFFLINE_EXPORT_FP}")
fi
if [[ -n "${TASK_SAMPLING_WEIGHTS}" ]]; then
  train_cmd+=(multitask_task_sampling_weights="${TASK_SAMPLING_WEIGHTS}")
fi
if [[ -n "${COLLECT_MAX_ENV_STEPS}" ]]; then
  train_cmd+=(collect_max_env_steps="${COLLECT_MAX_ENV_STEPS}")
fi
if [[ -n "${COLLECT_PARALLEL_GPU_IDS}" ]]; then
  train_cmd+=(collect_parallel_gpu_ids="${COLLECT_PARALLEL_GPU_IDS}")
fi
train_cmd+=(offline_export_overwrite="${OFFLINE_EXPORT_OVERWRITE}")

echo "[launcher] command:"
print_command "${train_cmd[@]}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[launcher] dry run complete; command was not executed."
  exit 0
fi

echo "[launcher] $(date --iso-8601=seconds) start shared family multitask continuation"
"${train_cmd[@]}" > "${TRAIN_LOG}" 2>&1
echo "[launcher] $(date --iso-8601=seconds) done shared family multitask continuation"
echo "[launcher] log saved to ${TRAIN_LOG}"
