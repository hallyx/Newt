#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
cd "${REPO_ROOT}"

PYTHON=${PYTHON:-/home/gpuserver/miniconda3/envs/isaac51/bin/python}
MODE=${MODE:-all}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
GPU_ID=${GPU_ID:-0}
ROOT=${ROOT:-reports/phase4_2_parametric_pretraining}
if [[ "${ROOT}" != /* ]]; then
  ROOT="${REPO_ROOT}/${ROOT}"
fi
PRETRAIN_STEPS=${PRETRAIN_STEPS:-299520}
PRETRAIN_EVAL_FREQ=${PRETRAIN_EVAL_FREQ:-49920}
EXPANSION_UPDATES=${EXPANSION_UPDATES:-8447}
BATCH_SIZE=${BATCH_SIZE:-1024}
NUM_ENVS=${NUM_ENVS:-256}
PARAM_EVAL_NUM_ENVS=${PARAM_EVAL_NUM_ENVS:-64}
THREE_TASK_NUM_ENVS=${THREE_TASK_NUM_ENVS:-64}
EVAL_EPISODES=${EVAL_EPISODES:-20}

STAGED_A=${STAGED_A:-logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_taskctx_repair_01125_00256_00186_rescue/20260713_phase3_2_rescue_00186_stage-3_asm-00186/models/best_step-50176_s-0p2461.pt}
FIXED_RUN_DIR=${FIXED_RUN_DIR:-logs/isaaclab-srsa-assembly/1/srsa_phase4_2_single_family/fixed_from_start}
PARAM_RUN_DIR=${PARAM_RUN_DIR:-logs/isaaclab-srsa-assembly/1/srsa_phase4_2_single_family/parametric_from_start}
FIXED_CKPT=${FIXED_CKPT:-${FIXED_RUN_DIR}/models/latest.pt}
PARAM_CKPT=${PARAM_CKPT:-${PARAM_RUN_DIR}/models/latest.pt}
EXPANDED_CKPT=${EXPANDED_CKPT:-${ROOT}/checkpoints/parametric_then_three_task.pt}
FIXED_REPLAY=${FIXED_REPLAY:-${ROOT}/replays/fixed_from_start.pt}
PARAM_REPLAY=${PARAM_REPLAY:-${ROOT}/replays/parametric_from_start.pt}

SEEN_TEMPLATES="0.5:0.5;0.5:1.0;1.0:1.0;2.0:1.5;4.0:2.0"

export CUDA_VISIBLE_DEVICES
export TERM=${TERM:-xterm}

if [[ "${CUDA_VISIBLE_DEVICES}" != "1" || "${GPU_ID}" != "0" ]]; then
  echo "Phase 4.2 requires physical CUDA1 via CUDA_VISIBLE_DEVICES=1 and logical GPU_ID=0." >&2
  exit 2
fi

case "${MODE}" in
  all|runtime-gate|pretrain|expansion|offline|ranking|param-eval|three-task|report|dry-run) ;;
  *) echo "Unknown MODE=${MODE}" >&2; exit 2 ;;
esac

run_runtime_gate() {
  "${PYTHON}" tdmpc2/scripts/phase4_parametric_runtime_gate.py \
    --output "${ROOT}/runtime_gate/parametric_runtime_gate.json"
}

train_single_family() {
  local arm=$1
  local run_dir=$2
  local replay=$3
  local templates=$4
  local scale_range=$5
  local fixed_scale=$6
  local clearance_jitter=$7
  local depth_jitter=$8
  if [[ -f "${run_dir}/models/latest.pt" && -f "${replay}" ]]; then
    echo "[phase4.2] ${arm} pretrain outputs exist, skipping."
    return
  fi
  "${PYTHON}" tdmpc2/train.py \
    --config-dir configs/train \
    --config-name srsa_01125_imitation_relaxed \
    assembly_id=01125 \
    num_envs="${NUM_ENVS}" \
    gpu_id="${GPU_ID}" \
    multiproc=false \
    num_gpus=1 \
    steps="${PRETRAIN_STEPS}" \
    eval_freq="${PRETRAIN_EVAL_FREQ}" \
    save_freq="${PRETRAIN_EVAL_FREQ}" \
    eval_episodes=1 \
    batch_size="${BATCH_SIZE}" \
    buffer_size=500000 \
    horizon=3 \
    utd=0.075 \
    seeding_coef=1 \
    checkpoint=null \
    finetune=false \
    skip_initial_eval=true \
    use_demos=false \
    compile=false \
    enable_wandb=false \
    save_agent=true \
    save_best=true \
    save_best_metric=episode_success \
    parametric_phase_replay_enabled=true \
    "parametric_phase_replay_anchor_templates='${templates}'" \
    parametric_phase_replay_seed=4201 \
    online_family_replay_save_fp="${replay}" \
    online_family_replay_save_every_eval=false \
    online_family_replay_save_on_finish=true \
    eval_task_id=null \
    srsa_param_template_id=null \
    srsa_task_template_id=null \
    eval_task_template_exact=false \
    eval_task_template_apply_geometry=false \
    eval_task_template_apply_sampler=false \
    srsa_axial_recompute_manifest_task_vecs=false \
    srsa_enable_axial_task_param_sampler=true \
    srsa_axial_fixed_plug_scale="${fixed_scale}" \
    "srsa_axial_scale_range='${scale_range}'" \
    srsa_axial_clearance_base=0.000114 \
    srsa_axial_depth_base=0.015 \
    "srsa_axial_clearance_depth_templates='${templates}'" \
    srsa_axial_clearance_jitter_ratio="${clearance_jitter}" \
    srsa_axial_depth_jitter_ratio="${depth_jitter}" \
    srsa_axial_reference_radius=0.003993 \
    srsa_axial_reference_depth=0.015 \
    srsa_axial_yaw_requirement=false \
    task_context_adapter_enabled=true \
    task_context_adapter_hidden_dim=128 \
    task_context_adapter_alpha=0.01 \
    task_context_adapter_source=raw_task_vec \
    task_context_adapter_apply_encoder=true \
    task_context_adapter_apply_dynamics=true \
    task_context_adapter_apply_policy=false \
    task_context_adapter_apply_reward=false \
    task_context_adapter_apply_q=false \
    task_context_adapter_lr_scale=0.1 \
    task_context_repair_enabled=true \
    task_recon_coef=0.1 \
    task_spread_coef=0.01 \
    task_raw_residual_scale=0.1 \
    task_spread_near_threshold=0.3 \
    task_spread_far_threshold=1.0 \
    task_spread_margin=0.5 \
    latent_residual_enabled=false \
    exp_name=srsa_phase4_2_single_family \
    run_id="${arm}_from_start" \
    seed=1
}

run_pretrain() {
  mkdir -p "${ROOT}/replays"
  train_single_family fixed "${FIXED_RUN_DIR}" "${FIXED_REPLAY}" "1.0:1.0" "1.0,1.0" true 0.0 0.0
  train_single_family parametric "${PARAM_RUN_DIR}" "${PARAM_REPLAY}" "${SEEN_TEMPLATES}" "0.85,1.15" false 0.10 0.10
}

run_expansion() {
  "${PYTHON}" tdmpc2/scripts/phase4_parametric_expansion_train.py \
    --pretrain-checkpoint "${PARAM_CKPT}" \
    --output "${EXPANDED_CKPT}" \
    --report "${ROOT}/checkpoints/parametric_then_three_task_train.json" \
    --run-label phase4_2_expansion_v2 \
    --updates "${EXPANSION_UPDATES}" \
    --batch-size "${BATCH_SIZE}"
}

run_offline() {
  mkdir -p "${ROOT}/offline"
  "${PYTHON}" tdmpc2/scripts/phase4_parametric_offline_eval.py \
    --arm fixed_from_start --checkpoint "${FIXED_CKPT}" --replay "${FIXED_REPLAY}" \
    --anchor-templates "1.0:1.0" --depth-base 0.015 \
    --output "${ROOT}/offline/fixed_from_start.json"
  "${PYTHON}" tdmpc2/scripts/phase4_parametric_offline_eval.py \
    --arm parametric_from_start --checkpoint "${PARAM_CKPT}" --replay "${PARAM_REPLAY}" \
    --anchor-templates "${SEEN_TEMPLATES}" --depth-base 0.015 \
    --output "${ROOT}/offline/parametric_from_start.json"
  "${PYTHON}" tdmpc2/scripts/phase4_multitask_origin_offline_eval.py \
    --variant staged_A --checkpoint "${STAGED_A}" --output-dir "${ROOT}/offline"
  "${PYTHON}" tdmpc2/scripts/phase4_multitask_origin_offline_eval.py \
    --variant parametric_then_three_task --checkpoint "${EXPANDED_CKPT}" --output-dir "${ROOT}/offline"
}

run_ranking_one() {
  local label=$1 checkpoint=$2 assembly=$3 scale=$4 template=$5 output=$6
  "${PYTHON}" tdmpc2/scripts/phase4_multitask_origin_ranking_eval.py \
    --variant "${label}" --checkpoint "${checkpoint}" --output "${output}" \
    --assembly-id "${assembly}" --param-scale "${scale}" --param-template "${template}" \
    --param-clearance-base 0.000114 --param-depth-base 0.015 --gpu-id "${GPU_ID}"
}

run_ranking() {
  mkdir -p "${ROOT}/ranking"
  run_ranking_one fixed_nominal "${FIXED_CKPT}" 01125 1.0 "1.0:1.0" "${ROOT}/ranking/fixed_nominal.json"
  run_ranking_one param_seen_nominal "${PARAM_CKPT}" 01125 1.0 "1.0:1.0" "${ROOT}/ranking/param_seen_nominal.json"
  run_ranking_one param_heldout_interpolation "${PARAM_CKPT}" 01125 1.05 "1.5:1.25" "${ROOT}/ranking/param_heldout_interpolation.json"
  run_ranking_one param_heldout_extrapolation "${PARAM_CKPT}" 01125 1.20 "4.5:2.2" "${ROOT}/ranking/param_heldout_extrapolation.json"
  "${PYTHON}" tdmpc2/scripts/phase4_multitask_origin_ranking_eval.py \
    --variant staged_A --checkpoint "${STAGED_A}" --output "${ROOT}/ranking/staged_A_00186.json" --gpu-id "${GPU_ID}"
  "${PYTHON}" tdmpc2/scripts/phase4_multitask_origin_ranking_eval.py \
    --variant parametric_then_three_task --checkpoint "${EXPANDED_CKPT}" \
    --output "${ROOT}/ranking/parametric_then_three_task_00186.json" --gpu-id "${GPU_ID}"
}

eval_param_condition() {
  local arm=$1 checkpoint=$2 name=$3 scale=$4 template=$5 split=$6
  local output="${ROOT}/closed_loop_param/${arm}/${name}"
  "${PYTHON}" tdmpc2/batch_eval_tasks.py \
    --config-dir configs/train --config-name srsa_01125_imitation_relaxed \
    checkpoint="${checkpoint}" \
    eval_assembly_ids='[01125]' \
    assembly_id=01125 \
    isaaclab_backend=srsa \
    task=isaaclab-srsa-assembly \
    num_envs="${PARAM_EVAL_NUM_ENVS}" \
    gpu_id="${GPU_ID}" \
    multiproc=false \
    compile=false \
    mpc=true \
    eval_task_template_exact=false \
    eval_task_template_apply_geometry=false \
    eval_task_template_apply_sampler=false \
    eval_task_template_print=false \
    srsa_axial_recompute_manifest_task_vecs=false \
    srsa_enable_axial_task_param_sampler=true \
    srsa_axial_fixed_plug_scale=false \
    "srsa_axial_scale_range='${scale},${scale}'" \
    srsa_axial_clearance_base=0.000114 \
    srsa_axial_depth_base=0.015 \
    "srsa_axial_clearance_depth_templates='${template}'" \
    srsa_axial_clearance_jitter_ratio=0.0 \
    srsa_axial_depth_jitter_ratio=0.0 \
    srsa_axial_reference_radius=0.003993 \
    srsa_axial_reference_depth=0.015 \
    srsa_axial_yaw_requirement=false \
    batch_eval_episodes_per_task="${EVAL_EPISODES}" \
    batch_eval_spawn_per_assembly=true \
    batch_eval_overwrite=true \
    batch_eval_output_dir="${output}" \
    batch_eval_summary_fp="${output}/batch_eval_summary.json" \
    enable_wandb=false \
    run_id="phase4_2_${arm}_${name}_${split}" \
    seed=1
}

run_param_eval() {
  local arms=(fixed parametric staged_A parametric_then_three_task)
  local checkpoints=("${FIXED_CKPT}" "${PARAM_CKPT}" "${STAGED_A}" "${EXPANDED_CKPT}")
  local names=(seen_low seen_nominal seen_high heldout_interpolation heldout_extrapolation)
  local scales=(0.90 1.00 1.10 1.05 1.20)
  local templates=("0.5:0.5" "1.0:1.0" "4.0:2.0" "1.5:1.25" "4.5:2.2")
  local splits=(seen seen seen interpolation extrapolation)
  local i j
  for i in "${!arms[@]}"; do
    for j in "${!names[@]}"; do
      eval_param_condition "${arms[$i]}" "${checkpoints[$i]}" "${names[$j]}" "${scales[$j]}" "${templates[$j]}" "${splits[$j]}"
    done
  done
}

eval_three_task_arm() {
  local arm=$1 checkpoint=$2
  CUDA_VISIBLE_DEVICES=1 GPU_ID="${GPU_ID}" CHECKPOINT="${checkpoint}" \
    OUTPUT_DIR="${ROOT}/closed_loop_three_task/${arm}" NUM_ENVS="${THREE_TASK_NUM_ENVS}" EPISODES="${EVAL_EPISODES}" \
    RUN_ID="phase4_2_${arm}_three_task" bash scripts/eval_phase3_three_task_checkpoint.sh
}

run_three_task() {
  eval_three_task_arm fixed "${FIXED_CKPT}"
  eval_three_task_arm parametric "${PARAM_CKPT}"
  eval_three_task_arm staged_A "${STAGED_A}"
  eval_three_task_arm parametric_then_three_task "${EXPANDED_CKPT}"
}

run_report() {
  "${PYTHON}" tdmpc2/scripts/build_phase4_parametric_pretraining_report.py \
    --root "${ROOT}" \
    --output-json reports/phase4_2_parametric_pretraining_ablation.json \
    --output-md reports/phase4_2_parametric_pretraining_ablation.md
}

if [[ "${MODE}" == "dry-run" ]]; then
  "${PYTHON}" tdmpc2/scripts/phase4_parametric_runtime_gate.py --dry-run
  DRY_RUN=1 MODE=pretrain bash "${BASH_SOURCE[0]}"
  "${PYTHON}" tdmpc2/scripts/phase4_parametric_expansion_train.py \
    --pretrain-checkpoint "${PARAM_CKPT}" --dry-run
  "${PYTHON}" tdmpc2/scripts/phase4_multitask_origin_ranking_eval.py \
    --variant staged_A --checkpoint "${STAGED_A}" --output "${ROOT}/ranking/staged_A_00186.json" --dry-run
  exit 0
fi

if [[ "${DRY_RUN:-0}" == "1" && "${MODE}" == "pretrain" ]]; then
  echo "[dry-run] fixed pretrain -> ${FIXED_CKPT}, replay -> ${FIXED_REPLAY}"
  echo "[dry-run] parametric pretrain -> ${PARAM_CKPT}, replay -> ${PARAM_REPLAY}"
  exit 0
fi

if [[ "${MODE}" == "all" || "${MODE}" == "runtime-gate" ]]; then run_runtime_gate; fi
if [[ "${MODE}" == "all" || "${MODE}" == "pretrain" ]]; then run_pretrain; fi
if [[ "${MODE}" == "all" || "${MODE}" == "expansion" ]]; then run_expansion; fi
if [[ "${MODE}" == "all" || "${MODE}" == "offline" ]]; then run_offline; fi
if [[ "${MODE}" == "all" || "${MODE}" == "ranking" ]]; then run_ranking; fi
if [[ "${MODE}" == "all" || "${MODE}" == "param-eval" ]]; then run_param_eval; fi
if [[ "${MODE}" == "all" || "${MODE}" == "three-task" ]]; then run_three_task; fi
if [[ "${MODE}" == "all" || "${MODE}" == "report" ]]; then run_report; fi
