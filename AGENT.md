# AGENT.md

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
