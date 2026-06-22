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
SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT:-${REPO_ROOT}/logs/isaaclab-srsa-assembly/1/srsa_axial_imitation_relaxed/20260525_233657_asm-01125_tid-2/models/best.pt}

TARGETS=${TARGETS:-"01125 00256 00186"}
RESUME_COMPLETED_TARGETS=${RESUME_COMPLETED_TARGETS:-}
ANCHOR_TASK_ID=${ANCHOR_TASK_ID:-01125}
STEPS_PER_TASK=${STEPS_PER_TASK:-50000}
NUM_ENVS=${NUM_ENVS:-64}
RETENTION_NUM_ENVS=${RETENTION_NUM_ENVS:-${NUM_ENVS}}
NUM_GPUS=${NUM_GPUS:-1}
GPU_ID=${GPU_ID:-0}
MULTIPROC=${MULTIPROC:-false}
SEED=${SEED:-1}
BATCH_SIZE=${BATCH_SIZE:-1024}
BUFFER_SIZE=${BUFFER_SIZE:-6000000}
HORIZON=${HORIZON:-3}
UTD=${UTD:-0.075}
EVAL_EPISODES=${EVAL_EPISODES:-1}
RETENTION_EVAL_EPISODES=${RETENTION_EVAL_EPISODES:-20}
EVAL_SUCCESS_METRIC=${EVAL_SUCCESS_METRIC:-relaxed}
UPDATE_PROGRESS_LOG_EVERY=${UPDATE_PROGRESS_LOG_EVERY:-50}
ACQUISITION_STOP_ENABLED=${ACQUISITION_STOP_ENABLED:-false}
ACQUISITION_REQUIRE_SUCCESS=${ACQUISITION_REQUIRE_SUCCESS:-false}
ACQUISITION_SUCCESS_THRESHOLD=${ACQUISITION_SUCCESS_THRESHOLD:-0.80}
ACQUISITION_MIN_STEPS=${ACQUISITION_MIN_STEPS:-150000}
ACQUISITION_METRIC=${ACQUISITION_METRIC:-episode_success}
RETENTION_REQUIRE_GATE=${RETENTION_REQUIRE_GATE:-false}
RETENTION_MIN_FAMILY=${RETENTION_MIN_FAMILY:-0.80}
RETENTION_MIN_ANCHOR=${RETENTION_MIN_ANCHOR:-0.75}
RETENTION_MIN_CURRENT=${RETENTION_MIN_CURRENT:-0.80}
RETENTION_GATE_METRIC=${RETENTION_GATE_METRIC:-episode_success}
TASK_CONTEXT_ADAPTER_ENABLED=${TASK_CONTEXT_ADAPTER_ENABLED:-false}
TASK_CONTEXT_ADAPTER_HIDDEN_DIM=${TASK_CONTEXT_ADAPTER_HIDDEN_DIM:-128}
TASK_CONTEXT_ADAPTER_ALPHA=${TASK_CONTEXT_ADAPTER_ALPHA:-1.0}
TASK_CONTEXT_ADAPTER_SOURCE=${TASK_CONTEXT_ADAPTER_SOURCE:-task_context}

CURRENT_RATIO=${CURRENT_RATIO:-0.50}
ANCHOR_RATIO=${ANCHOR_RATIO:-0.20}
HISTORY_RATIO=${HISTORY_RATIO:-0.30}
MIN_CURRENT_EPISODES=${MIN_CURRENT_EPISODES:-5}
REPLAY_STORAGE_DEVICE=${REPLAY_STORAGE_DEVICE:-cpu}
SRSA_CLEARANCE_DEPTH_TEMPLATES=${SRSA_CLEARANCE_DEPTH_TEMPLATES:-1.0:1.0}

SRSA_TASK_TEMPLATE_FP=${SRSA_TASK_TEMPLATE_FP:-data/srsa_axial_task_templates.json}
SRSA_MESH_GEOMETRY_FP=${SRSA_MESH_GEOMETRY_FP:-data/srsa_mesh_geometry_params.csv}
SRSA_PARAM_TEMPLATE_ID=${SRSA_PARAM_TEMPLATE_ID:-2}
REFERENCE_ANCHOR_TYPE_ID=${REFERENCE_ANCHOR_TYPE_ID:-0}
EVAL_TASK_TEMPLATE_EXACT=${EVAL_TASK_TEMPLATE_EXACT:-true}

EXP_NAME=${EXP_NAME:-srsa_axial_online_family_replay_from_01125}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
WORK_BASE=${REPO_ROOT}/logs/isaaclab-srsa-assembly/${SEED}/${EXP_NAME}
LOG_ROOT=${LOG_ROOT:-${WORK_BASE}/${RUN_STAMP}_launcher}
MANIFEST_FP=${MANIFEST_FP:-${LOG_ROOT}/online_family_replay_manifest.json}
REPLAY_DIR=${REPLAY_DIR:-${LOG_ROOT}/replay}
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

join_eval_ids() {
  local joined=""
  local item
  for item in "$@"; do
    if [[ -z "${joined}" ]]; then
      joined="${item}"
    elif [[ ",${joined}," != *",${item},"* ]]; then
      joined="${joined},${item}"
    fi
  done
  printf '[%s]' "${joined}"
}

score_family_csv() {
  local csv_fp=$1
  "${PYTHON}" - "${csv_fp}" <<'PY'
import csv
import math
import sys

csv_fp = sys.argv[1]
with open(csv_fp, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
if not rows:
    print("nan")
    raise SystemExit(0)

keys = ["relaxed_success", "episode_success", "process_success", "strict_success", "official_success_terminal"]
values = []
for row in rows:
    value = None
    for key in keys:
        raw = row.get(key, "")
        if raw not in {"", None}:
            try:
                value = float(raw)
                break
            except ValueError:
                pass
    if value is not None and math.isfinite(value):
        values.append(value)
if not values:
    print("nan")
else:
    mean_success = sum(values) / len(values)
    min_success = min(values)
    print(f"{0.7 * mean_success + 0.3 * min_success:.6f}")
PY
}

check_retention_gate() {
  local summary_fp=$1
  local anchor_id=$2
  local current_id=$3
  local min_family=$4
  local min_anchor=$5
  local min_current=$6
  local metric=$7
  "${PYTHON}" - "${summary_fp}" "${anchor_id}" "${current_id}" "${min_family}" "${min_anchor}" "${min_current}" "${metric}" <<'PY'
import json
import math
import sys

summary_fp, anchor_id, current_id, min_family, min_anchor, min_current, metric = sys.argv[1:]
min_family = float(min_family)
min_anchor = float(min_anchor)
min_current = float(min_current)
payload = json.load(open(summary_fp, encoding="utf-8"))

def value_from(item, key):
    raw = item.get(key)
    if raw is None and key == "episode_success":
        raw = item.get("relaxed_success")
    if raw is None:
        return None
    value = float(raw)
    return value if math.isfinite(value) else None

family = value_from(payload, metric)
tasks = {
    str(item.get("assembly_id")): item
    for item in payload.get("tasks", [])
    if isinstance(item, dict)
}
anchor = value_from(tasks.get(str(anchor_id), {}), metric)
current = value_from(tasks.get(str(current_id), {}), metric)
problems = []
if family is None or family < min_family:
    problems.append(f"family {metric}={family} < {min_family}")
if anchor is None or anchor < min_anchor:
    problems.append(f"anchor {anchor_id} {metric}={anchor} < {min_anchor}")
if current is None or current < min_current:
    problems.append(f"current {current_id} {metric}={current} < {min_current}")

print(
    f"[retention-gate] metric={metric} family={family} anchor:{anchor_id}={anchor} "
    f"current:{current_id}={current} thresholds=family:{min_family},anchor:{min_anchor},current:{min_current}"
)
if problems:
    print("[retention-gate] FAIL: " + "; ".join(problems), file=sys.stderr)
    raise SystemExit(3)
print("[retention-gate] PASS")
PY
}

SOURCE_CHECKPOINT=$(make_abs_path "${SOURCE_CHECKPOINT}")
CONFIG_DIR=$(make_abs_path "${CONFIG_DIR}")
SRSA_TASK_TEMPLATE_FP=$(make_abs_path "${SRSA_TASK_TEMPLATE_FP}")
SRSA_MESH_GEOMETRY_FP=$(make_abs_path "${SRSA_MESH_GEOMETRY_FP}")
MANIFEST_FP=$(make_abs_path "${MANIFEST_FP}")
REPLAY_DIR=$(make_abs_path "${REPLAY_DIR}")

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
if [[ ! -f "${SOURCE_CHECKPOINT}" ]]; then
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[launcher] dry-run warning: source checkpoint not found: ${SOURCE_CHECKPOINT}" >&2
  else
    echo "[launcher] source checkpoint not found: ${SOURCE_CHECKPOINT}" >&2
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

if [[ "${MULTIPROC}" == "true" ]]; then
  step_stride=$((NUM_ENVS * NUM_GPUS))
else
  step_stride=${NUM_ENVS}
fi
if (( step_stride <= 0 )); then
  echo "[launcher] invalid step stride: num_envs=${NUM_ENVS}, num_gpus=${NUM_GPUS}" >&2
  exit 1
fi
default_eval_freq=${STEPS_PER_TASK}
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

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${LOG_ROOT}" "${REPLAY_DIR}"
fi

echo "[launcher] mode=online-family-replay"
echo "[launcher] repo_root=${REPO_ROOT}"
echo "[launcher] python=${PYTHON}"
echo "[launcher] config=${CONFIG_DIR}/${CONFIG_NAME}"
echo "[launcher] source_checkpoint=${SOURCE_CHECKPOINT}"
echo "[launcher] targets=${TARGETS}"
echo "[launcher] resume_completed_targets=${RESUME_COMPLETED_TARGETS}"
echo "[launcher] steps_per_task=${STEPS_PER_TASK} num_envs=${NUM_ENVS} multiproc=${MULTIPROC} num_gpus=${NUM_GPUS} gpu_id=${GPU_ID}"
echo "[launcher] eval_freq=${EVAL_FREQ} save_freq=${SAVE_FREQ} eval_success_metric=${EVAL_SUCCESS_METRIC}"
echo "[launcher] update_progress_log_every=${UPDATE_PROGRESS_LOG_EVERY}"
echo "[launcher] acquisition_stop=${ACQUISITION_STOP_ENABLED} require_success=${ACQUISITION_REQUIRE_SUCCESS} metric=${ACQUISITION_METRIC} threshold=${ACQUISITION_SUCCESS_THRESHOLD} min_steps=${ACQUISITION_MIN_STEPS}"
echo "[launcher] retention_gate=${RETENTION_REQUIRE_GATE} metric=${RETENTION_GATE_METRIC} family>=${RETENTION_MIN_FAMILY} anchor>=${RETENTION_MIN_ANCHOR} current>=${RETENTION_MIN_CURRENT}"
echo "[launcher] replay_mix=current:${CURRENT_RATIO} anchor:${ANCHOR_RATIO} history:${HISTORY_RATIO} min_current_eps=${MIN_CURRENT_EPISODES}"
echo "[launcher] task_context_adapter enabled=${TASK_CONTEXT_ADAPTER_ENABLED} source=${TASK_CONTEXT_ADAPTER_SOURCE} hidden=${TASK_CONTEXT_ADAPTER_HIDDEN_DIM} alpha=${TASK_CONTEXT_ADAPTER_ALPHA}"
echo "[launcher] size_templates=${SRSA_CLEARANCE_DEPTH_TEMPLATES} eval_task_template_exact=${EVAL_TASK_TEMPLATE_EXACT}"
echo "[launcher] manifest=${MANIFEST_FP}"
echo "[launcher] replay_dir=${REPLAY_DIR}"
echo "[launcher] log_root=${LOG_ROOT}"
echo "[launcher] dry_run=${DRY_RUN}"

current_checkpoint=${SOURCE_CHECKPOINT}
completed_targets=()
for ASM in ${RESUME_COMPLETED_TARGETS}; do
  completed_targets+=("${ASM}")
done
stage_idx=${STAGE_OFFSET:-${#completed_targets[@]}}

for ASM in ${TARGETS}; do
  stage_idx=$((stage_idx + 1))
  RUN_ID="${RUN_STAMP}_stage-${stage_idx}_asm-${ASM}"
  stage_work_dir="${WORK_BASE}/${RUN_ID}"
  stage_log="${LOG_ROOT}/stage-${stage_idx}_asm-${ASM}.train.log"
  replay_fp="${REPLAY_DIR}/${ASM}.pt"
  acquisition_status_fp="${LOG_ROOT}/stage-${stage_idx}_asm-${ASM}.acquisition_status.json"

  echo "[launcher] $(date --iso-8601=seconds) start stage=${stage_idx} assembly_id=${ASM}"
  echo "[launcher] stage_run_id=${RUN_ID}"
  echo "[launcher] stage_checkpoint_in=${current_checkpoint}"
  echo "[launcher] stage_replay_out=${replay_fp}"
  echo "[launcher] stage_acquisition_status=${acquisition_status_fp}"
  echo "[launcher] stage_log=${stage_log}"

  train_cmd=(
    "${PYTHON}" tdmpc2/train.py
    --config-dir "${CONFIG_DIR}"
    --config-name "${CONFIG_NAME}"
    checkpoint="${current_checkpoint}"
    finetune=true
    seeding_coef=1
    assembly_id="${ASM}"
    isaaclab_dir="${ISAACLAB_DIR}"
    srsa_dir="${SRSA_DIR}"
    srsa_task_template_fp="${SRSA_TASK_TEMPLATE_FP}"
    srsa_mesh_geometry_fp="${SRSA_MESH_GEOMETRY_FP}"
    srsa_param_template_id="${SRSA_PARAM_TEMPLATE_ID}"
    eval_task_template_exact="${EVAL_TASK_TEMPLATE_EXACT}"
    srsa_axial_reference_anchor_assembly_id="${ANCHOR_TASK_ID}"
    srsa_axial_reference_anchor_task_type_id="${REFERENCE_ANCHOR_TYPE_ID}"
    srsa_axial_recompute_manifest_task_vecs=true
    "srsa_axial_clearance_depth_templates='${SRSA_CLEARANCE_DEPTH_TEMPLATES}'"
    online_family_replay_enabled=true
    online_family_replay_manifest_fp="${MANIFEST_FP}"
    online_family_replay_save_fp="${replay_fp}"
    online_family_current_task_id="${ASM}"
    online_family_anchor_task_id="${ANCHOR_TASK_ID}"
    online_family_current_ratio="${CURRENT_RATIO}"
    online_family_anchor_ratio="${ANCHOR_RATIO}"
    online_family_history_ratio="${HISTORY_RATIO}"
    online_family_min_current_episodes="${MIN_CURRENT_EPISODES}"
    online_family_replay_storage_device="${REPLAY_STORAGE_DEVICE}"
    online_family_acquisition_stop_enabled="${ACQUISITION_STOP_ENABLED}"
    online_family_acquisition_success_threshold="${ACQUISITION_SUCCESS_THRESHOLD}"
    online_family_acquisition_min_steps="${ACQUISITION_MIN_STEPS}"
    online_family_acquisition_metric="${ACQUISITION_METRIC}"
    online_family_acquisition_status_fp="${acquisition_status_fp}"
    steps="${STEPS_PER_TASK}"
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
    skip_initial_eval=true
    eval_episodes="${EVAL_EPISODES}"
    eval_success_metric="${EVAL_SUCCESS_METRIC}"
    srsa_eval_success_metric="${EVAL_SUCCESS_METRIC}"
    update_progress_log_every="${UPDATE_PROGRESS_LOG_EVERY}"
    enable_wandb="${ENABLE_WANDB}"
    save_agent=true
    save_best=true
    save_best_metric=episode_success
    task_context_adapter_enabled="${TASK_CONTEXT_ADAPTER_ENABLED}"
    task_context_adapter_hidden_dim="${TASK_CONTEXT_ADAPTER_HIDDEN_DIM}"
    task_context_adapter_alpha="${TASK_CONTEXT_ADAPTER_ALPHA}"
    task_context_adapter_source="${TASK_CONTEXT_ADAPTER_SOURCE}"
    exp_name="${EXP_NAME}"
    run_id="${RUN_ID}"
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
  if [[ "${DRY_RUN}" != "1" && ! -f "${replay_fp}" ]]; then
    echo "[launcher] expected replay snapshot not found: ${replay_fp}" >&2
    echo "[launcher] Check ${stage_log}; replay is saved by Trainer at eval/finish." >&2
    exit 1
  fi
  require_acquisition_success=false
  case "${ACQUISITION_REQUIRE_SUCCESS,,}" in
    1|true|yes) require_acquisition_success=true ;;
  esac
  if [[ "${DRY_RUN}" != "1" && "${require_acquisition_success}" == "true" ]]; then
    if [[ ! -f "${acquisition_status_fp}" ]]; then
      echo "[launcher] acquisition status missing: ${acquisition_status_fp}" >&2
      echo "[launcher] Check ${stage_log}; ensure eval_freq triggers at least one eval." >&2
      exit 2
    fi
    acquisition_reached=$("${PYTHON}" -c 'import json, sys; print("true" if bool(json.load(open(sys.argv[1], encoding="utf-8")).get("reached", False)) else "false")' "${acquisition_status_fp}")
    if [[ "${acquisition_reached}" != "true" ]]; then
      echo "[launcher] acquisition gate not reached for assembly_id=${ASM}; skipping manifest update and retention eval." >&2
      echo "[launcher] status=${acquisition_status_fp}" >&2
      exit 2
    fi
  fi

  completed_targets+=("${ASM}")
  current_checkpoint=${next_checkpoint}

  update_manifest_cmd=(
    "${PYTHON}" scripts/update_online_family_replay_manifest.py
    --manifest "${MANIFEST_FP}"
    --task-id "${ASM}"
    --assembly-id "${ASM}"
    --replay-fp "${replay_fp}"
    --checkpoint "${current_checkpoint}"
    --stage-index "${stage_idx}"
    --anchor-task-id "${ANCHOR_TASK_ID}"
  )
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] manifest update command:"
    print_command "${update_manifest_cmd[@]}"
  else
    "${update_manifest_cmd[@]}"
  fi

  eval_ids=$(join_eval_ids "${completed_targets[@]}")
  retention_dir="${LOG_ROOT}/family_eval_after_${ASM}"
  retention_log="${retention_dir}/batch_eval.log"
  retention_summary="${retention_dir}/batch_eval_summary.json"
  if [[ "${DRY_RUN}" != "1" ]]; then
    mkdir -p "${retention_dir}"
  fi

  echo "[launcher] family_eval_ids=${eval_ids}"
  echo "[launcher] family_eval_dir=${retention_dir}"

  eval_cmd=(
    "${PYTHON}" tdmpc2/batch_eval_tasks.py
    --config-dir "${CONFIG_DIR}"
    --config-name "${CONFIG_NAME}"
    checkpoint="${current_checkpoint}"
    eval_assembly_ids="${eval_ids}"
    isaaclab_backend=srsa
    task=isaaclab-srsa-assembly
    isaaclab_dir="${ISAACLAB_DIR}"
    srsa_dir="${SRSA_DIR}"
    srsa_task_template_fp="${SRSA_TASK_TEMPLATE_FP}"
    srsa_mesh_geometry_fp="${SRSA_MESH_GEOMETRY_FP}"
    srsa_param_template_id="${SRSA_PARAM_TEMPLATE_ID}"
    eval_task_template_exact="${EVAL_TASK_TEMPLATE_EXACT}"
    srsa_axial_reference_anchor_assembly_id="${ANCHOR_TASK_ID}"
    srsa_axial_reference_anchor_task_type_id="${REFERENCE_ANCHOR_TYPE_ID}"
    srsa_axial_recompute_manifest_task_vecs=true
    "srsa_axial_clearance_depth_templates='${SRSA_CLEARANCE_DEPTH_TEMPLATES}'"
    num_envs="${RETENTION_NUM_ENVS}"
    gpu_id="${GPU_ID}"
    model_size=S
    horizon="${HORIZON}"
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
    task_context_adapter_hidden_dim="${TASK_CONTEXT_ADAPTER_HIDDEN_DIM}"
    task_context_adapter_alpha="${TASK_CONTEXT_ADAPTER_ALPHA}"
    task_context_adapter_source="${TASK_CONTEXT_ADAPTER_SOURCE}"
    eval_success_metric="${EVAL_SUCCESS_METRIC}"
    srsa_eval_success_metric="${EVAL_SUCCESS_METRIC}"
    batch_eval_episodes_per_task="${RETENTION_EVAL_EPISODES}"
    batch_eval_spawn_per_assembly=true
    batch_eval_overwrite=true
    batch_eval_output_dir="${retention_dir}"
    batch_eval_summary_fp="${retention_summary}"
    enable_wandb=false
    exp_name="${EXP_NAME}"
    run_id="${RUN_ID}_family_eval_after_${ASM}"
    seed="${SEED}"
  )

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] batch eval command:"
    print_command "${eval_cmd[@]}"
  else
    "${eval_cmd[@]}" > "${retention_log}" 2>&1
    if [[ ! -f "${retention_summary%.json}.csv" ]]; then
      echo "[launcher] family eval CSV missing under ${retention_dir}" >&2
      echo "[launcher] Check ${retention_log}" >&2
      exit 1
    fi
    family_score=$(score_family_csv "${retention_summary%.json}.csv")
    echo "[launcher] family_score=${family_score} checkpoint=${current_checkpoint}"
    require_retention_gate=false
    case "${RETENTION_REQUIRE_GATE,,}" in
      1|true|yes) require_retention_gate=true ;;
    esac
    if [[ "${require_retention_gate}" == "true" ]]; then
      check_retention_gate \
        "${retention_summary}" \
        "${ANCHOR_TASK_ID}" \
        "${ASM}" \
        "${RETENTION_MIN_FAMILY}" \
        "${RETENTION_MIN_ANCHOR}" \
        "${RETENTION_MIN_CURRENT}" \
        "${RETENTION_GATE_METRIC}"
    fi
  fi
done

echo "[launcher] $(date --iso-8601=seconds) all online family replay stages completed"
echo "[launcher] final_checkpoint=${current_checkpoint}"
echo "[launcher] final_manifest=${MANIFEST_FP}"
