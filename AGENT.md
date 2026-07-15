# AGENT.md

## 2026-06-18 Active Direction: SRSA Online Family Replay V2

The active multi-task training direction is staged online family replay V2
(`acquisition-first`), not offline-first family continuation and not a true
mixed-assembly IsaacLab env.

Primary implementation rule:

- Keep each online train job single-assembly.
- Learn the current target before handing off to the next stage.
- During acquisition, use current-heavy replay:
  - current task replay: 80%
  - `01125` anchor replay: 20%
  - previous task history replay: 0%
- Gate handoff on eval success instead of fixed 50k-step stages.
- Save the current stage's online replay only after the acquisition stage has
  produced useful current-task data.
- Run retention eval only after the acquisition gate is reached.
- Treat single-task success as evidence that the target is learnable, not as
  proof that `AxialTaskEncoder` / `task_vec_6` conditioning is being used.
- Do not reuse `multitask_continuation_*` for this path. Those names remain
  reserved for the existing offline continuation route.
- Use new config names under `online_family_replay_*`.

Current validated V2 result:

- Run:
  `logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_acquire_from_01125/20260618_001734_launcher`
- Stage:
  `01125 -> 00256`
- Checkpoint in:
  `logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_replay_from_01125/20260615_202326_stage-1_asm-01125/models/latest.pt`
- Stage-2 checkpoint out:
  `logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_acquire_from_01125/20260618_001734_stage-2_asm-00256/models/latest.pt`
- Acquisition gate:
  `00256 episode_success=0.9023` at 299,520 env steps.
- Retention after 00256:
  `01125 relaxed_success=0.75`, `00256 relaxed_success=0.90`,
  family average `0.825`.
- Task-vector swap diagnostic on the same 00256 checkpoint:
  - normal 00256 task vector: `episode_success=0.90`
  - forced 01125 task vector in the 00256 environment:
    `episode_success=0.90`
  - forced zero task vector in the 00256 environment:
    `episode_success=0.75`

Interpretation:

- `00256` is learnable from the online-family stage-1 checkpoint.
- The older 50k online-family failure was mainly premature handoff.
- `01125` retention is acceptable but weakened; do not add many tasks before
  checking task-vector usage and adding retention gates.
- The current issue is not simply "`task_vec_6` is disconnected"; the more
  precise diagnosis is that `task_vec_6` is not yet an irreplaceable information
  source. The model can still use state/contact feedback, MPC correction, and
  single-task bias paths to solve relaxed success.
- The current policy is weakly sensitive to `task_vec_6`; the 01125 vector swap
  does not hurt 00256 at all. Do not assume the task encoder is carrying
  multi-task routing yet.
- Strict/process success remains weak, so report strict/process/lateral
  diagnostics alongside relaxed success.

2026-06-19 replay/task tensor check:

- Checked replay snapshots:
  - `20260615_202326_launcher/replay/01125.pt`
  - `20260618_001734_launcher/replay/00256.pt`
- Both snapshots store a per-transition `task` tensor with shape `[N, 6]`.
- `01125` replay has one unique vector:
  `[0, -0.155195, 0.145688, 0.165645, 1, 0]`.
- `00256` replay has one unique vector:
  `[0, -0.178871, 0.099081, 0.115353, 1.2189, 0]`.
- A mixed `OnlineFamilyReplayBuffer` sample with 50/50 current/anchor replay
  returned `task.shape == [3, 64, 6]`, task counts
  `{'01125': 32, '00256': 32}`, and nonzero task-vector std on the differing
  dimensions.
- `tdmpc2/tdmpc2.py:update()` consumes `task` from `buffer.sample()` and passes
  it into `_update()` / `_loss_fn()`; the model-update path is not replacing the
  sampled batch task tensor with the current env task vector.
- Therefore replay save/load and online-family mixed sampling are not currently
  showing a global `env.current_task_vec` broadcast bug. The next problem to
  test is whether the trained model actually changes action/value/reward/model
  predictions when only `task_vec_6` changes.

Recommended next work order:

- validated acquisition: `01125 -> 00256`
- completed diagnostic: offline paired sensitivity report on identical replay
  states with only `task_vec_6` changed. Report:
  `logs/task_vec_sensitivity/20260619_00256_v2_offline_report.json`.
  Result: action/Q/reward/next-latent deltas are near numerical noise
  (`action_l2` about `4e-7` for 01125/00186/zero/extreme swaps; random about
  `3e-6`) while correct action norm is about `0.8`. This strongly supports
  that the learned model is currently almost invariant to task-vector changes.
- next training step: retention polish on `01125,00256` with guaranteed mixed
  batches, ideally 50/50 for the diagnostic polish run. Batch task/source
  counts and task entropy are now logged from `TDMPC2.update()` when using
  `OnlineFamilyReplayBuffer`; per-task losses remain the next deeper logging
  item if needed.
- Recommended polish command is recorded in
  `docs/multitask_status_brief.md` section 13. It starts from the validated V2
  00256 checkpoint, runs `TARGETS=00256`, `CURRENT_RATIO=0.50`,
  `ANCHOR_RATIO=0.50`, `HISTORY_RATIO=0.0`, and is meant to be launched on
  physical GPU 1 with `CUDA_VISIBLE_DEVICES=1 GPU_ID=0`.
- Active polish run launched on physical GPU 1:
  `logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_polish_01125_00256/20260619_0135_retention_polish_launcher`.
  Early verification: update batches are exactly 50/50
  (`01125=512`, `00256=512`, entropy norm `1.0`), and first eval at
  49,920 env steps reached `00256 episode_success=0.8555`. Let the run finish
  to 100k and retention eval before judging retention/task-vector sensitivity.
- The polish run finished and passed retention, but the paired sensitivity
  report still showed near-zero task-vector influence. A zero-init task context
  FiLM adapter is now implemented behind `task_context_adapter_enabled=true`.
  It inserts task-conditioned residual modulation at encoder latent, dynamics,
  policy, reward, and Q sites; old checkpoints load into an initially equivalent
  model because adapter final projections are reset to zero.
- next architecture experiment: start from the 20260619 polish checkpoint with
  `TASK_CONTEXT_ADAPTER_ENABLED=true`, keep 01125/00256 replay at 50/50, then
  rerun the paired sensitivity report before considering `00186`.
- 2026-06-22 adapter run diagnosis: the first adapter run completed but degraded
  (`00256` eval ended around `0.578`), and family eval initially reported 0/20
  because batch-eval workers did not inherit `task_context_adapter_*` settings.
  That eval compatibility bug is fixed. The deeper model issue is that the
  learned `AxialTaskEncoder` context is nearly collapsed for realistic SRSA
  vectors, so a new `task_context_adapter_source=raw_task_vec` option was added.
  Prefer the next experiment with `TASK_CONTEXT_ADAPTER_SOURCE=raw_task_vec` and
  smaller `TASK_CONTEXT_ADAPTER_ALPHA=0.05`.
- 2026-06-22 raw-task adapter result: the `01125 + 00256` diagnostic with
  `TASK_CONTEXT_ADAPTER_SOURCE=raw_task_vec` and `TASK_CONTEXT_ADAPTER_ALPHA=0.05`
  completed, but degraded badly. `00256` eval fell from the source checkpoint's
  strong pre-update rollout to `0.1445` at 49,920 steps and `0.0` at 99,840
  steps; post-run family eval was `01125=0.0`, `00256=0.0` relaxed success.
  Mixed replay was correct (`01125=512`, `00256=512`, entropy norm `1.0`).
- The same raw-task run did make `task_vec_6` influential: after fixing the
  sensitivity loader to restore adapter alpha from checkpoint metadata, paired
  sensitivity was no longer numerical noise (`anchor action_l2=0.059`,
  `zero action_l2=0.089`, `random action_l2=0.102`). Interpret this as
  "raw task input can move the model, but the full-site adapter is too
  destructive at alpha 0.05", not as a reason to add tasks.
- do not add `00186` until a two-task run keeps `00256` success acceptable,
  keeps `01125` retention acceptable, logs near-50/50 batch task counts, and
  produces paired action/Q/reward/next-latent sensitivity above numerical noise.
  The next two-task attempt should lower adapter strength first, e.g. try
  `TASK_CONTEXT_ADAPTER_ALPHA=0.005` from the same clean polish checkpoint.
- next acquisition step: cautious `00186` only after 01125/00256 retention and
  task-conditioning sensitivity are acceptable
- medium family: `01125 -> 00256 -> 00186 -> 00004 -> 00014`
- hard cases only after the baseline is stable: `00062`, then `00271`

Curriculum rule:

- First fix `srsa_axial_clearance_depth_templates` to `"1.0:1.0"` and solve
  assembly transfer.
- Add size generalization only after retention is stable:
  `"1.0:1.0;0.5:1.0;2.0:1.5"`, then the full five-template set.

Checkpoint/eval rule:

- Prefer `relaxed_success` and family retention metrics over
  `official_success_latched`.
- For acquisition gates, start with `episode_success >= 0.80` after at least
  150k env steps.
- For retention gates before adding another task, require both the new task and
  the anchor/family to remain acceptable, e.g. current `>=0.80`, anchor
  `>=0.75`, family mean `>=0.80`.
- Run task-vector swap diagnostics before changing model architecture:
  evaluate the same environment/checkpoint with the correct task vector, an
  anchor task vector, and a zero/random vector. If success barely changes, the
  policy may be ignoring task conditioning.
- With the current V2 00256 checkpoint, wrong-vector eval confirms weak
  dependence on `task_vec_6`; prefer task-conditioning fixes or diagnostics
  before scaling to many more assemblies.
- The current mainline is two-task diagnosis first, three-task expansion second:
  `01125 + 00256` must pass retention and paired sensitivity before running
  `01125 + 00256 + 00186`.
- If the three-task ablation runs, keep it to `01125:0.34`, `00256:0.33`,
  `00186:0.33` replay pressure, keep
  `srsa_axial_clearance_depth_templates=1.0:1.0`, and do not add `00062` or
  `00271`.
- Do not treat success swap alone as decisive. For the next diagnostic, keep the
  initial state paired and compute `delta_action`, `delta_Q`,
  `delta_reward_pred`, and `delta_next_latent` under correct, anchor, zero,
  random, and extreme task vectors.
- The first family score can be:
  `0.7 * mean_relaxed_success + 0.3 * min_relaxed_success`.
- Continue reporting strict/process/lateral/force diagnostics; do not let
  official-latched success hide insertion failures.

Implementation surface for this route:

- `tdmpc2/common/buffer.py`
- `tdmpc2/common/online_family_replay.py`
- `tdmpc2/config.py`
- `tdmpc2/train.py`
- `tdmpc2/trainer.py`
- `scripts/run_01125_online_family_replay_targets.sh`
- `scripts/run_01125_online_family_acquire_targets.sh`
- `scripts/run_01125_00256_rawtask_adapter_diagnostic.sh`
- `scripts/run_00256_task_vec_swap_eval.sh`
- `scripts/update_online_family_replay_manifest.py`
- `tdmpc2/scripts/task_vec_sensitivity_report.py`

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
