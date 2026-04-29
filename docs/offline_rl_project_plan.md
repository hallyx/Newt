# Offline RL Project Plan

## Goal

Build a first offline RL / world-model training path for contact-rich assembly using a 14D canonical state:

- `tcp_pos_socket[3]`
- `tcp_quat_socket[4]`
- `tcp_linvel_socket[3]`
- `tcp_angvel_socket[3]`
- `gripper_width[1]`

The state is task-centric and robot-agnostic: all TCP quantities are expressed relative to the target socket frame.

This phase does **not** aim to reproduce full Newt behavior. It aims to deliver a minimal, stable offline pretraining path that can later support:

- world-model pretraining from demonstrations
- online finetuning in Isaac Lab
- sim2real residual adaptation

## Scope Of Phase 1

Phase 1 is intentionally narrow:

- single task
- state-only
- 14D canonical input
- offline dataset from SRSA teacher rollouts
- BC sanity check first
- then world-model pretraining

Out of scope for Phase 1:

- raw image / depth training
- multi-task training
- action canonicalization to socket frame
- real robot execution
- advanced offline RL penalties such as CQL / IQL

## Source Dataset

Current source dataset:

- `/home/gpuserver/hx/github/SRSA/rollout_out/debug_00783/teacher_rollouts_newt.pt`

Observed properties from the original debug inspection:

- transitions: `22200`
- episodes: `300`
- episode length: fixed `74`
- `obs`: `(22200, 14)`
- `next_obs`: `(22200, 14)`
- `action`: `(22200, 6)`
- `reward`: `(22200, 1)`
- `done`: `(22200, 1)`
- `episode`: `(22200, 1)`
This older export looked like a strong expert dataset and should be treated as **offline expert pretraining data**.

However, the collector has since been updated to preserve:

- successful trajectories
- failed trajectories
- explicit final episode outcome labels
- explicit terminal success/failure markers
- running and final episode return

That means Phase 1 should preserve the richer labels even if the first BC/WM loop only consumes a subset of them.

## Canonical Data Contract

Phase 1 training should use a compact state-only dataset with:

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

If available in the source data, also preserve:

- `episode_return_running`
- `episode_return_final`
- `episode_success_final`
- `episode_failure_final`
- `terminal_success`
- `terminal_failure`

The first exported dataset version should map:

- `obs <- canonical_obs`
- `next_obs <- next_canonical_obs`
- `action <- teacher_action`

This keeps the state definition aligned with the canonical task-space formulation while avoiding action-frame refactoring in Phase 1.

The offline dataset loader should also support episode filtering modes:

- `all`
- `success_only`
- `failure_only`

Implementation note:

- do not assume transitions from the same episode are contiguous on disk
- reconstruct per-episode sequences from `episode` and `step_id`

## Training Stages

### Stage 1: BC Sanity Check

Train only the policy prior:

- input: `obs (14D)`
- target: `action (6D)`

Objective:

- verify state and action field alignment
- verify the 14D canonical state is sufficient to recover teacher actions
- verify a checkpoint trained on this dataset can execute in the environment

Primary metric:

- action MSE

Secondary metrics:

- Isaac Lab eval success rate
- episode reward

### Stage 2: World-Model Pretraining

Train:

- state encoder
- latent dynamics
- reward head
- value head
- policy prior

Initial objective:

- `L_bc`
- `L_dyn`
- `L_reward`
- `0.1 * L_value`

Recommended initial modeling choices:

- latent dim: `128` or `256`
- horizon: `3` or `5`
- state-only MLP encoder

## Required New Files

### 1. `scripts/export_offline_newt_dataset.py`

Purpose:

- load `teacher_rollouts_newt.pt`
- drop unused image-heavy fields for Phase 1
- write a compact offline dataset

Suggested output:

- `data/offline_canonical14_v0.pt`

The exporter should preserve explicit success/failure episode labels when present, even if the first BC/WM training loop only consumes a subset of them.

### Phase 1 trajectory input entry

Offline training should accept either:

- `offline_source_fp`
  - raw SRSA rollout file such as `teacher_rollouts_newt.pt`
- `offline_dataset_fp`
  - pre-exported compact dataset

If only `offline_source_fp` is given, `offline_train.py` should auto-export:

- `work_dir/data/<source_stem>_compact.pt`

### 2. `tdmpc2/offline_dataset.py`

Purpose:

- load compact offline dataset
- sample by episode
- provide fixed-horizon subsequences for model training

### 3. `tdmpc2/offline_train.py`

Purpose:

- run offline-only BC sanity check
- run offline world-model pretraining
- periodically save checkpoints
- optionally call evaluation on Isaac Lab

## Required Modifications

### `tdmpc2/config.py`

Add offline-specific configuration:

- `offline_only`
- `offline_dataset_fp`
- `offline_bc_steps`
- `offline_wm_steps`
- `offline_obs_dim`
- `offline_eval_freq`

### `tdmpc2/envs/isaaclab.py`

Add a 14D canonical evaluation observation path so that offline checkpoints can be evaluated against the same observation definition used during offline training.

Current implementation flag:

- `isaaclab_use_canonical_obs=true`

This should switch the wrapper from the default AutoMate 24D policy observation to:

- `tcp_pos_socket[3]`
- `tcp_quat_socket[4]`
- `tcp_linvel_socket[3]`
- `tcp_angvel_socket[3]`
- `gripper_width[1]`

### `tdmpc2/common/buffer.py`

Either:

- add a dedicated offline path that accepts already-exported episode transitions

Or:

- keep current replay buffer untouched and build a dedicated offline dataset sampler in `offline_dataset.py`

Preferred choice for Phase 1: dedicated offline dataset sampler, to avoid destabilizing the online training path.

## Milestones

### M1: Dataset Export

Deliver:

- compact `offline_canonical14_v0.pt`

Success criteria:

- fields match the Phase 1 data contract
- no NaNs
- episode segmentation preserved

### M2: BC Checkpoint

Deliver:

- `bc_best.pt`

Success criteria:

- action loss decreases
- eval runs in Isaac Lab
- success rate above random baseline

Suggested eval command pattern:

```bash
python tdmpc2/eval.py \
  task=isaaclab-automate-assembly \
  assembly_id=00783 \
  checkpoint=/path/to/bc_best.pt \
  isaaclab_use_canonical_obs=true \
  obs=state
```

### M3: WM Checkpoint

Deliver:

- `wm_best.pt`

Success criteria:

- dynamics loss and reward loss are stable
- eval success improves or remains competitive with BC-only

## Risks

### Risk 1: Observation Mismatch

The largest risk is not algorithmic. It is an observation contract mismatch between:

- offline dataset `obs`
- evaluation-time environment observation

Mitigation:

- explicitly evaluate on 14D canonical observations
- do not reuse the current 24D Isaac Lab wrapper path for offline checkpoints

### Risk 2: Action Semantics

Phase 1 uses `teacher_action` directly. If those actions are not in the ideal canonical frame, BC may still work but world-model portability may be limited.

Mitigation:

- defer action-frame canonicalization to Phase 2

### Risk 3: Dataset Bias

Some exports may still be heavily expert-skewed.

Mitigation:

- treat early Phase 1 as expert-leaning pretraining
- do not over-claim robustness or failure recovery from BC-only results

The updated collector reduces this risk by preserving failed trajectories plus explicit terminal labels. Preserve those labels in the exported offline dataset so later phases can support:

- success-only filtering
- failure-only filtering
- return-based ranking
- failure recovery and risk prediction

## Phase 2 Preview

After Phase 1 works:

- export `action_socket`
- compare world-frame vs socket-frame action
- add online finetuning with demo buffer + online buffer
- add contact-aware state extensions
- add sim2real adapter and residual action correction

## Immediate Next Steps

1. Implement `scripts/export_offline_newt_dataset.py`
2. Implement `tdmpc2/offline_dataset.py` with episode filtering and fixed-horizon sampling
3. Implement `tdmpc2/offline_train.py` with BC-only mode
4. Add a canonical 14D eval observation path in `tdmpc2/envs/isaaclab.py`
5. Run BC sanity check on assembly `00783`
