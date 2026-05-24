# AGENT.md

## 2026-05-24 Recovery Note: 01125 Axial-Hole Consolidation

The active task has shifted from the older single-task 14D Phase 1 note below to
multi-assembly consolidation for the same axial shaft-in-hole task family.

### Current Goal

Use this unchanged source checkpoint:

- `/home/gpuserver/hx/github/Newt/logs/isaaclab-srsa-assembly/1/srsa_axial_online/20260523_163332_asm-01125/models/best.pt`

Keep the checkpoint architecture and I/O contract unchanged:

- `model_size=S`
- `horizon=3`
- 17D canonical observation with force appended
- 3D position-control action
- `isaaclab_use_canonical_obs=true`
- `srsa_enable_flange_force_sensor=true`
- `isaaclab_canonical_append_force=true`
- `task_conditioning=axial_params`
- `contact_history_enabled=true`
- `contact_action_dim=3`
- `contact_ee_delta_dim=3`
- `srsa_position_control_only=true`
- `compile=false`

First-round target assemblies:

- collect/train: `00004,00014,00062,00271`
- eval/retention: `01125,00004,00014,00062,00271`
- `01125` is the source checkpoint and retention/eval anchor; do not mix its
  rollouts into round-1 training unless `include_source_anchor_rollouts=true`.

### Required Workflow

1. Zero-shot screen target tasks with the 01125 checkpoint.
2. Collect policy rollouts for accepted target tasks.
3. Validate the manifest immediately after collection.
4. Run offline consolidation smoke test.
5. Run full offline consolidation.
6. Batch eval on `01125,00004,00014,00062,00271`.
7. For weak tasks with strict/process success `<0.45`, run short online boost
   only to collect better rollouts.
8. Merge boost manifests and rerun offline consolidation; the boosted online
   checkpoint is not the final unified model.

Primary success metric is strict/process success, not IsaacLab/AutoMate
latched `env.ep_succeeded`.

Acceptance targets:

- target mean strict/process success `>=0.70`
- each hard target strict/process success `>=0.45`
- `01125` anchor strict/process success `>=0.65`
- always report `official_success_latched - strict_success` gap

### Implemented Changes In This Working Tree

`tdmpc2/config.py`

- Added `eval_assembly_ids`, `include_source_anchor_rollouts=false`,
  `eval_success_metric=strict`, and `srsa_eval_success_metric=strict`.
- Added strict success thresholds:
  `strict_depth_fraction=0.90`, `strict_success_steps=10`,
  `strict_lateral_tol_min=0.0005`, `strict_lateral_tol_max=0.0020`,
  `strict_keypoint_tol_min=0.0010`, `strict_keypoint_tol_max=0.0030`,
  `strict_angle_tol_deg=3.0`.
- Added collection defaults for source `01125`, target assemblies
  `00004,00014,00062,00271`, 300 episodes per task, and weak-task 600 episode
  escalation.
- Added offline stage filter defaults:
  `offline_wm_filter_mode=all`,
  `offline_bc_filter_mode=success_or_high_depth`,
  `offline_rl_filter_mode=all`,
  `task_balanced_sampling=true`,
  `offline_high_depth_threshold=0.75`,
  `offline_high_depth_lateral_tol_m=0.0020`.

`tdmpc2/envs/isaaclab.py`

- Default SRSA success metric now resolves to strict stable process success.
- `final_info.success` no longer defaults to latched `env.ep_succeeded`.
- Added and preserved success/geometry fields:
  `official_success_latched`, `official_success_terminal`,
  `process_success_terminal`, `strict_success_stable`,
  `strict_success_episode`, `depth_fraction`, `lateral_error`,
  `angle_error`, `keypoint_error`.
- Preserved old aliases:
  `official_success`, `current_official_success`, `process_success`,
  `terminal_process_success`.

`tdmpc2/eval.py` and `tdmpc2/batch_eval_tasks.py`

- Added grouped JSON/CSV summaries with fixed columns:
  `assembly_id`, `official_success_latched`,
  `official_success_terminal`, `strict_success`, `process_success`,
  `mean_depth_fraction`, `mean_lateral_error_mm`,
  `mean_angle_error_deg`, `mean_keypoint_error_mm`,
  `episode_len_mean`, `official_strict_gap`.
- `batch_eval_tasks.py` can evaluate `01125` anchor even when it is not in the
  offline manifest by using runtime axial params / `task_vec_6` fallback.

`tdmpc2/collect_eval_rollouts.py`

- Defaults collection to source `01125` and targets
  `00004,00014,00062,00271`.
- Added screening-file support: strict success `<0.10` escalates to weak-task
  episode count; strict `<0.03` plus low depth can be skipped and marked
  `weak_task_requires_online_boost`.
- Manifest default:
  `data/offline_manifest_policy_rollouts_from_01125_axial_hole_3d.json`.
- Manifest and rollout datasets now include `assembly_id`, consecutive
  `task_id`, `task_param_vec/task_vec_6`, `action_dim=3`, `obs_shape=[17]`,
  and episode-level official/process/strict/depth/lateral/angle/keypoint
  metadata.
- Manifest validation checks task-id continuity, source file existence, dims,
  and episode counts.

`tdmpc2/offline_io.py`, `tdmpc2/offline_dataset.py`,
`tdmpc2/offline_train.py`

- Added optional episode metadata preservation for strict/process/official
  success and geometry stats.
- Added `success_or_high_depth` filtering for BC.
- Added stage-specific datasets for BC, WM, and RL.
- Added task-balanced sequence sampling and per-task sample fraction logging.

New scripts:

- `tdmpc2/scripts/screen_tasks.py`
  - zero-shot screening wrapper around batch eval
  - writes `data/task_screening_01125_axial_hole.csv/json`
  - labels tasks as `hard_target`, `hard_target_extra_episodes`,
    `official_gap_target`, `easy_anchor`, or `defer_online_boost`
- `tdmpc2/scripts/merge_offline_manifests.py`
  - merges base and boost manifests
  - offsets episode ids
  - rewrites consecutive task ids

### Commands To Resume

Zero-shot screening:

```bash
/home/gpuserver/miniconda3/envs/isaac51/bin/python tdmpc2/scripts/screen_tasks.py \
  checkpoint=/home/gpuserver/hx/github/Newt/logs/isaaclab-srsa-assembly/1/srsa_axial_online/20260523_163332_asm-01125/models/best.pt \
  eval_assembly_ids="[00004,00014,00062,00271]" \
  screen_trials=200 \
  eval_success_metric=strict \
  compile=false
```

Rollout collection:

```bash
/home/gpuserver/miniconda3/envs/isaac51/bin/python tdmpc2/collect_eval_rollouts.py \
  checkpoint=/home/gpuserver/hx/github/Newt/logs/isaaclab-srsa-assembly/1/srsa_axial_online/20260523_163332_asm-01125/models/best.pt \
  collect_source_assembly_id=01125 \
  collect_assembly_ids="[00004,00014,00062,00271]" \
  collect_screening_fp=data/task_screening_01125_axial_hole.csv \
  collect_episodes_per_task=300 \
  collect_weak_task_episodes=600 \
  collect_manifest_fp=data/offline_manifest_policy_rollouts_from_01125_axial_hole_3d.json \
  num_envs=200 \
  compile=false
```

Offline smoke:

```bash
/home/gpuserver/miniconda3/envs/isaac51/bin/python tdmpc2/offline_train.py \
  checkpoint=/home/gpuserver/hx/github/Newt/logs/isaaclab-srsa-assembly/1/srsa_axial_online/20260523_163332_asm-01125/models/best.pt \
  offline_manifest_fp=data/offline_manifest_policy_rollouts_from_01125_axial_hole_3d.json \
  offline_bc_steps=10 \
  offline_wm_steps=10 \
  offline_rl_steps=0 \
  task_balanced_sampling=true \
  compile=false
```

Batch eval:

```bash
/home/gpuserver/miniconda3/envs/isaac51/bin/python tdmpc2/batch_eval_tasks.py \
  checkpoint=<offline_final.pt> \
  offline_manifest_fp=data/offline_manifest_policy_rollouts_from_01125_axial_hole_3d.json \
  eval_assembly_ids="[01125,00004,00014,00062,00271]" \
  batch_eval_episodes_per_task=200 \
  eval_success_metric=strict \
  compile=false
```

### Verification Already Done

- `py_compile` passed for the touched Python files.
- Synthetic `OfflineSequenceDataset` test passed for `success_or_high_depth`
  and task-balanced sampling.
- `screen_tasks.py --help` and `merge_offline_manifests.py --help` both load.
- Recovery continuation on 2026-05-24 fixed batch-eval worker routing for the
  `01125` anchor when it is absent from the offline manifest, kept
  `screen_decision` within the documented enum, and corrected single-eval
  `episode_len_mean` summary output.

### Still Not Run

- Real IsaacLab/SRSA zero-shot screening.
- Real rollout collection.
- Real manifest validation on generated rollout files.
- Offline smoke/full consolidation with the 01125 checkpoint.
- Batch eval on the final offline checkpoint.

Recovery continuation added `screen_assembly_ids` with default
`[00004,00014,00062,00271]`, so screening no longer accidentally includes the
`01125` anchor from the broader eval-retention default.

## Purpose

This repository is being extended toward a contact-rich, canonical-state offline-to-online world-model pipeline for assembly.

The immediate development target is **Phase 1 offline training**:

- single task
- state-only
- 14D canonical observation
- offline expert data from SRSA teacher rollouts

## Current Priority

Do not optimize for full Newt reproduction.

Do optimize for:

- a clean offline data contract
- a minimal offline training entrypoint
- a canonical 14D evaluation path
- checkpoints that can later initialize online finetuning

## Canonical State Definition

All TCP quantities are expressed in the target socket frame.

Phase 1 state:

- `tcp_pos_socket[3]`
- `tcp_quat_socket[4]`
- `tcp_linvel_socket[3]`
- `tcp_angvel_socket[3]`
- `gripper_width[1]`

Total: `14D`

This state is intentionally robot-agnostic. Do not reintroduce joint-space inputs into Phase 1 unless there is a hard blocker.

## Offline Dataset Source

Current source:

- `/home/gpuserver/hx/github/SRSA/rollout_out/debug_00783/teacher_rollouts_newt.pt`

Observed statistics from the original debug export:

- transitions: `22200`
- episodes: `300`
- fixed horizon per episode: `74`

Important implication:

- older exports were near-expert-only
- newer exports may now include both successful and failed trajectories
- treat the dataset as offline pretraining data first, then exploit explicit failure labels in later phases

## Phase 1 Dataset Contract

Preferred compact dataset fields:

- `obs`
- `next_obs`
- `action`
- `reward`
- `done`
- `terminated`
- `truncated`
- `episode`
- `step_id`
- `success_episode`

If present in the source dataset, keep these episode-level supervision fields too:

- `episode_return_running`
- `episode_return_final`
- `episode_success_final`
- `episode_failure_final`
- `terminal_success`
- `terminal_failure`

Phase 1 mapping:

- `obs <- canonical_obs`
- `next_obs <- next_canonical_obs`
- `action <- teacher_action`

Trajectory input entry for Phase 1:

- `offline_source_fp`
  - path to the raw SRSA rollout `.pt`
- `offline_dataset_fp`
  - path to an already-exported compact offline dataset

If only `offline_source_fp` is provided, `offline_train.py` should auto-export a compact dataset before training starts.

Phase 1 training defaults:

- BC sanity check may start with all transitions
- dataset loader should support `all`, `success_only`, and `failure_only` episode filtering
- explicit terminal success/failure labels should be preserved even if the first training loop does not consume all of them
- offline rollout tensors may be interleaved by global step rather than stored as contiguous episode blocks; any sampler must reconstruct episodes using `episode` plus `step_id`

## Development Constraints

### Keep Phase 1 Narrow

Do not add:

- image training
- depth training
- multi-task data mixing
- sim2real adapter code
- action-frame canonicalization
- conservative offline RL penalties

These belong to later phases.

### Preserve The Existing Online Path

Avoid destabilizing the current Isaac Lab online training path.

If possible:

- add `offline_train.py`
- add a separate offline dataset loader
- do not heavily refactor the existing online replay buffer unless needed

## Required Files For Phase 1

Expected additions:

- `scripts/export_offline_newt_dataset.py`
- `tdmpc2/offline_dataset.py`
- `tdmpc2/offline_train.py`

Expected modifications:

- `tdmpc2/config.py`
- `tdmpc2/envs/isaaclab.py`

Relevant Phase 1 flag:

- `isaaclab_use_canonical_obs=true`
  - when evaluating offline 14D checkpoints in Isaac Lab
  - makes the wrapper emit socket-frame canonical observations instead of the default 24D policy observation

## Recommended Implementation Order

1. Export a compact state-only offline dataset.
2. Implement BC-only offline training.
3. Add a canonical 14D evaluation observation path.
4. Verify offline BC checkpoint in Isaac Lab.
5. Add world-model pretraining losses.

## Success Criteria For Phase 1

### BC Sanity Check

- action loss decreases
- checkpoint can run in env
- success rate is above random baseline

### WM Pretraining

- dynamics loss is stable
- reward prediction loss is stable
- eval success is competitive with or better than BC-only

## Known Risks

### Observation Contract Mismatch

The largest risk is mismatch between:

- offline `obs`
- evaluation-time env `obs`

Any agent working on this code should verify that offline training and env evaluation use the same 14D canonical state definition.

### Action Semantics

Phase 1 uses `teacher_action` as-is. This is acceptable for the first milestone, but later phases should revisit action-frame canonicalization.

### Dataset Bias

Some exports may still be strongly expert-skewed, but the collector can now preserve failed trajectories and explicit terminal outcome labels.

Prefer:

- preserving all failure annotations during export
- keeping episode-level return fields
- postponing aggressive success-only filtering until BC sanity checks are in place

## Non-Goals For Phase 1

- proving strong offline RL performance
- proving sim2real transfer
- proving cross-robot generalization

Those are later milestones.

## Reference Plan

Detailed execution plan:

- `docs/offline_rl_project_plan.md`
