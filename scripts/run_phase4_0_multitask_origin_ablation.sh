#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
cd "${REPO_ROOT}"

PYTHON=${PYTHON:-/home/gpuserver/miniconda3/envs/isaac51/bin/python}
MODE=${MODE:-all}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
GPU_ID=${GPU_ID:-0}
OUTPUT_ROOT=${OUTPUT_ROOT:-reports/phase4_0_multitask_origin}
UPDATES=${UPDATES:-8447}
BATCH_SIZE=${BATCH_SIZE:-1024}
EVAL_EPISODES=${EVAL_EPISODES:-20}
EVAL_NUM_ENVS=${EVAL_NUM_ENVS:-256}
SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT:-logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_taskctx_repair_01125_00256_00186_rescue/20260713_phase3_2_rescue_00186_stage-3_asm-00186/models/best_step-50176_s-0p2461.pt}

export CUDA_VISIBLE_DEVICES
export TERM=${TERM:-xterm}

if [[ "${CUDA_VISIBLE_DEVICES}" != "1" || "${GPU_ID}" != "0" ]]; then
  echo "Phase 4.0 requires physical CUDA1 via CUDA_VISIBLE_DEVICES=1 and logical GPU_ID=0." >&2
  exit 2
fi

case "${MODE}" in
  all|train|offline|ranking|closed-loop|report|dry-run) ;;
  *) echo "Unknown MODE=${MODE}" >&2; exit 2 ;;
esac

run_train() {
  "${PYTHON}" tdmpc2/scripts/phase4_multitask_origin_train.py \
    --source-checkpoint "${SOURCE_CHECKPOINT}" \
    --output-dir "${OUTPUT_ROOT}/checkpoints" \
    --updates "${UPDATES}" \
    --batch-size "${BATCH_SIZE}"
}

run_offline() {
  "${PYTHON}" tdmpc2/scripts/phase4_multitask_origin_offline_eval.py \
    --source-checkpoint "${SOURCE_CHECKPOINT}" \
    --checkpoint-dir "${OUTPUT_ROOT}/checkpoints" \
    --output-dir "${OUTPUT_ROOT}/offline"
}

variant_checkpoint() {
  local variant=$1
  if [[ "${variant}" == "A" ]]; then
    printf '%s\n' "${SOURCE_CHECKPOINT}"
  else
    printf '%s\n' "${OUTPUT_ROOT}/checkpoints/variant_${variant}.pt"
  fi
}

run_ranking() {
  mkdir -p "${OUTPUT_ROOT}/ranking"
  local variant checkpoint
  for variant in A B C D; do
    checkpoint=$(variant_checkpoint "${variant}")
    "${PYTHON}" tdmpc2/scripts/phase4_multitask_origin_ranking_eval.py \
      --variant "${variant}" \
      --checkpoint "${checkpoint}" \
      --output "${OUTPUT_ROOT}/ranking/variant_${variant}.json" \
      --gpu-id "${GPU_ID}"
  done
}

run_closed_loop() {
  local variant checkpoint
  for variant in A B C D; do
    checkpoint=$(variant_checkpoint "${variant}")
    CUDA_VISIBLE_DEVICES=1 \
    GPU_ID="${GPU_ID}" \
    CHECKPOINT="${checkpoint}" \
    OUTPUT_DIR="${OUTPUT_ROOT}/closed_loop/${variant}" \
    NUM_ENVS="${EVAL_NUM_ENVS}" \
    EPISODES="${EVAL_EPISODES}" \
    RUN_ID="phase4_0_variant_${variant}_three_task_eval" \
      bash scripts/eval_phase3_three_task_checkpoint.sh
  done
}

run_report() {
  "${PYTHON}" tdmpc2/scripts/build_phase4_multitask_origin_report.py \
    --root "${OUTPUT_ROOT}" \
    --output-json reports/phase4_0_multitask_origin_ablation.json \
    --output-md reports/phase4_0_multitask_origin_ablation.md
}

if [[ "${MODE}" == "dry-run" ]]; then
  CUDA_VISIBLE_DEVICES=1 "${PYTHON}" tdmpc2/scripts/phase4_multitask_origin_train.py --dry-run
  CUDA_VISIBLE_DEVICES=1 "${PYTHON}" tdmpc2/scripts/phase4_multitask_origin_offline_eval.py --variant A --dry-run
  CUDA_VISIBLE_DEVICES=1 "${PYTHON}" tdmpc2/scripts/phase4_multitask_origin_ranking_eval.py \
    --variant A --checkpoint "${SOURCE_CHECKPOINT}" --output "${OUTPUT_ROOT}/ranking/variant_A.json" --dry-run
  DRY_RUN=1 CHECKPOINT="${SOURCE_CHECKPOINT}" OUTPUT_DIR="${OUTPUT_ROOT}/closed_loop/A" \
    bash scripts/eval_phase3_three_task_checkpoint.sh
  exit 0
fi

if [[ "${MODE}" == "all" || "${MODE}" == "train" ]]; then
  run_train
fi
if [[ "${MODE}" == "all" || "${MODE}" == "offline" ]]; then
  run_offline
fi
if [[ "${MODE}" == "all" || "${MODE}" == "ranking" ]]; then
  run_ranking
fi
if [[ "${MODE}" == "all" || "${MODE}" == "closed-loop" ]]; then
  run_closed_loop
fi
if [[ "${MODE}" == "all" || "${MODE}" == "report" ]]; then
  run_report
fi
