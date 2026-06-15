# Newt/SRSA Multi-Task Training Status Brief

Date: 2026-06-15

This brief summarizes the current Newt multi-task SRSA assembly setup, the code paths already present, observed experiment results, and the main design questions for the next multi-task revision.

## 1. Current Project Goal

The current SRSA/Newt direction is no longer the older language-conditioned multi-task setup. The active method is:

- TD-MPC2 world model / policy for SRSA axial insertion.
- 3D translation action policy.
- Canonical 17D observation:
  - 14D canonical SRSA state.
  - force observation appended.
- Structured axial task conditioning:
  - `task_conditioning=axial_params`.
  - `task_vec_6 = [task_type, log_scale, clearance_abs_norm, clearance_rel_norm, depth_abs_norm, yaw_requirement]`.
  - AxialTaskEncoder maps 6D params to task embedding.
- Contact dynamics conditioning:
  - `contact_history_enabled=true`.
  - `contact_history_len=4`.
  - force/action/end-effector-delta history is used by dynamics.

The practical goal is to train one family policy that can work across multiple `assembly_id`s and task sizes, without losing the original 01125 capability.

## 2. Main Code Paths

### 2.1 Direct Online Fine-Tune

Entry:

- `scripts/run_01125_direct_finetune_targets.sh`

Behavior:

- Each target assembly starts independently from the same 01125 checkpoint.
- No replay is shared between targets.
- This is currently the cleanest and most reliable path for learning a single target.

Default important settings:

- checkpoint: `logs/isaaclab-srsa-assembly/1/srsa_axial_imitation_relaxed/20260525_233657_asm-01125_tid-2/models/best.pt`
- targets: `00186 00256 00062 00271 00726 01079 01029 01092 01102`
- steps: `600000`
- `num_envs=300`
- `multiproc=true`
- `num_gpus=2`
- `model_size=S`
- `horizon=3`
- `utd=0.075`
- `seeding_coef=1`
- `eval_success_metric=relaxed`
- `task_conditioning=axial_params`
- `contact_history_enabled=true`

### 2.2 Serial Continual Fine-Tune

Entry:

- `scripts/run_01125_continual_finetune_targets.sh`

Behavior:

- Trains one target at a time.
- Passes `models/latest.pt` from one stage to the next.
- Runs batch eval after each stage on 01125 plus completed targets.

Important limitation:

- This is not true multi-task training.
- It does not preserve old online replay buffers.
- It only transfers weights forward, so it is vulnerable to catastrophic forgetting.

### 2.3 Offline Family Multi-Task Continuation

Entries:

- `scripts/run_01125_family_multitask_continuation.sh`
- `tdmpc2/train.py` with `multitask_continuation_enabled=true`
- `tdmpc2/offline_train.py`
- `tdmpc2/common/multitask_replay.py`

Behavior:

- Loads a replay manifest / compact offline dataset.
- Builds one shared task-conditioned model.
- Uses `MultiTaskReplayPool` to sample task-balanced offline batches.
- Supports:
  - `multitask_task_ids`
  - `multitask_anchor_task_id`
  - `multitask_curriculum_mode`
  - `multitask_anchor_min_ratio`
  - `multitask_new_task_min_ratio`
  - `multitask_hard_case_ratio`
  - optional proximal regularization.

Important limitation:

- This path is still offline replay based.
- It does not collect fresh online experience during the actual TD-MPC2 update loop unless using the separate auto-collect path.
- Current datasets are very small and contain many failed terminal rollouts.

### 2.4 Existing Generic Multi-Task Wrapper

File:

- `tdmpc2/envs/wrappers/vectorized_multitask.py`

Behavior:

- Generic Gym vectorized multi-task wrapper.
- Used when not IsaacLab and not `child_env`.

Important limitation:

- It is not currently the SRSA/IsaacLab multi-assembly online solution.
- `tdmpc2/envs/__init__.py` routes IsaacLab tasks directly into `envs.isaaclab.make_env`, so SRSA training effectively uses one `assembly_id` per launched train job.

## 3. Core Current Training Config

Primary config:

- `configs/train/srsa_01125_imitation_relaxed.yaml`

Key settings:

```yaml
isaaclab_backend: srsa
task: isaaclab-srsa-assembly
assembly_id: "01125"
srsa_task_template_fp: data/srsa_axial_task_templates.json
srsa_param_template_id: 2

num_envs: 350
multiproc: true
num_gpus: 2
model_size: S
batch_size: 1024
buffer_size: 10000000
horizon: 3
utd: 0.075
mpc: true

isaaclab_use_canonical_obs: true
srsa_task_family_name: normal_fit
srsa_task_param_obs: false
srsa_task_param_obs_mode: task_vec
srsa_enable_axial_task_param_sampler: true
srsa_axial_fixed_plug_scale: true
srsa_axial_clearance_base: 0.000114
srsa_axial_clearance_depth_templates: "0.5:0.5;0.5:1.0;1.0:1.0;2.0:1.5;4.0:2.0"
srsa_axial_clearance_jitter_ratio: 0.10
srsa_axial_depth_base: 0.015
srsa_axial_depth_jitter_ratio: 0.10
srsa_axial_init_error_xy_range: "0.009,0.0010"
srsa_axial_init_error_z_range: "0.0010,0.0020"

srsa_enable_flange_force_sensor: true
isaaclab_canonical_append_force: true
isaaclab_canonical_append_task_params: false

task_conditioning: axial_params
srsa_axial_reference_anchor_assembly_id: "01125"
srsa_axial_recompute_manifest_task_vecs: true
eval_success_metric: relaxed

contact_history_enabled: true
contact_history_len: 4
contact_context_dim: 64
contact_history_hidden_dim: 128
contact_history_layers: 2
contact_force_dim: 6
contact_action_dim: 3
contact_ee_delta_dim: 3
contact_history_use_ee_delta: true
```

Important observation:

- Size templates are already part of normal training config.
- This mixes assembly generalization and size generalization if used directly in every multi-task stage.

## 4. Existing Experiment Results

### 4.1 Direct Online Fine-Tune Results

Source logs:

- `logs/finetune_01125_axial_hole/20260525_003125/asm-*.train.log`

Best saved checkpoint metrics:

| Target | Best `episode_success` | Step |
| --- | ---: | ---: |
| 00186 | 0.4133 | 600k |
| 00256 | 0.9556 | 600k |
| 00062 | 0.0000 | 150k |
| 00271 | 0.0000 | 150k |

Interpretation:

- Direct online fine-tune can work very well for some targets, especially 00256.
- 00186 partially learns.
- 00062 and 00271 are currently hard/failing cases under the same settings.
- This means the model and SRSA bridge are not fundamentally broken; the multi-task strategy and task-specific stability are the likely bottlenecks.

### 4.2 Serial Continual Results

Source logs:

- `logs/finetune_01125_axial_hole/20260525_003125/stage-*.train.log`
- `logs/finetune_01125_axial_hole/20260525_003125/retention_after_*/batch_eval_summary.csv`

Best per-stage train metrics:

| Stage | Target | Best `episode_success` | Step |
| --- | --- | ---: | ---: |
| stage-1 | 00004 | 0.2567 | 600k |
| stage-2 | 00014 | 0.1278 | 600k |
| stage-3 | 00062 | 0.1156 | 300k |
| stage-4 | 00271 | 0.0000 | 150k |

Retention snapshots:

| Retention point | Assembly | official terminal | strict/process |
| --- | --- | ---: | --- |
| after 00004 | 01125 | 0.005 | strict 0.0 / process 0.0 |
| after 00004 | 00004 | 0.595 | strict 0.245 / process 0.25 |
| after 00014 | 01125 | 0.49 | strict 0.0 / process 0.0 |
| after 00014 | 00004 | 0.065 | strict 0.0 / process 0.0 |
| after 00014 | 00014 | 0.88 | strict 0.045 / process 0.065 |
| after 00062 | 01125 | 0.01 | strict 0.0 / process 0.0 |
| after 00062 | 00004 | 0.0 | strict 0.0 / process 0.0 |
| after 00062 | 00014 | 0.07 | strict 0.0 / process 0.0 |
| after 00062 | 00062 | 0.65 | strict 0.1 / process 0.13 |
| after 00271 | 01125 | 0.04 | strict 0.0 / process 0.0 |
| after 00271 | 00062 | 0.025 | strict 0.01 / process 0.01 |
| after 00271 | 00271 | 0.0 | strict 0.0 / process 0.0 |

Interpretation:

- Serial checkpoint handoff learns the current or recent target somewhat, but old targets collapse.
- This is classic forgetting.
- The current `run_01125_continual_finetune_targets.sh` is not sufficient as a family learning algorithm because it does not mix old replay during updates.

### 4.3 Offline Family Data

Important data files:

- `data/offline_manifest_01125_family.json`
- `data/rollouts_01125_family/*/policy_eval_rollouts.pt`
- `data/rollouts_01125_family/*/policy_eval_rollouts.pt.json`

Family continuation log:

- `logs/finetune_01125_axial_hole/20260525_003125/family_multitask_continuation.log`

Observed dataset scale:

- 10 tasks.
- 10 episodes per task.
- 740 transitions per task.
- 7,400 transitions total.

Example from `data/rollouts_01125_family/00256/policy_eval_rollouts.pt.json`:

- `episodes_collected=10`
- `num_transitions=740`
- `episode_success_mean=0.0`
- `official_success_terminal=1.0`
- `process_success_terminal=0.0`
- `relaxed_success_stable=0.0`
- `terminal_success_count=0`

Interpretation:

- A lot of collected data has contradictory-looking success indicators:
  - official/latched success may be high.
  - process/relaxed/strict terminal success can still be zero.
- For insertion policy learning, terminal/process/relaxed/strict metrics are more informative than latched official success.
- 7,400 transitions across 10 tasks is too small for robust multi-task TD-MPC2 family training.
- If many trajectories are failure-terminal rollouts, offline training is likely to imitate or reinforce bad behavior.

## 5. Recent Evaluation/Video Tooling State

There are uncommitted changes related to video recording and force traces:

- `docs/eval.md`
- `scripts/batch_record_task_sizes.py`
- `scripts/batch_record_ids_task_sizes.py`
- `tdmpc2/common/logger.py`
- `tdmpc2/config.py`
- `tdmpc2/envs/isaaclab.py`
- `tdmpc2/eval.py`

Purpose of those changes:

- Batch record different task sizes.
- Batch record different assembly/checkpoint pairs.
- Support socket camera following via SRSA camera profile.
- Save force trace CSV aligned to recorded video.
- Add child-process/artifact guard for IsaacSim teardown hangs.

This tooling is useful for diagnosis but should be kept separate from the core multi-task training redesign.

## 6. Current Main Problems

### Problem A: Offline-first multi-task data is too weak

The current offline family dataset is too small and too failure-heavy. It is useful for debugging format compatibility, but not enough as the primary learning signal.

### Problem B: Serial continual training has no replay protection

The continual script only passes checkpoint weights. It does not preserve old task samples during new task updates. This explains the severe retention collapse.

### Problem C: Success metrics are confusing

`official_success_latched` can be high while terminal/process success is zero. For training/eval decisions, the family pipeline should prioritize:

- `relaxed_success`
- `terminal_process_success`
- `process_success`
- `strict_success`
- mean lateral error
- depth fraction
- force/jam diagnostics

and should not select checkpoints using official-latched success alone.

### Problem D: Multi-task and multi-size difficulty are entangled

Current configs often include five size templates:

```text
0.5:0.5;0.5:1.0;1.0:1.0;2.0:1.5;4.0:2.0
```

This multiplies the task space. If assembly generalization is still unstable, adding size generalization simultaneously makes the failure mode hard to identify.

### Problem E: SRSA online training is effectively single assembly per process

The current stable SRSA path launches one `assembly_id` per train job. A true mixed-assembly online environment is not currently implemented in the stable training loop.

## 7. Recommended Next Design Direction

The next robust design should not be "collect first, offline train later" as the main path.

Recommended main path:

```text
online staged family replay
```

Meaning:

1. Keep the stable one-assembly online training job.
2. After each stage, save that stage's online replay.
3. During later stages, train on:
   - current online replay,
   - anchor replay,
   - history replay from previous tasks.
4. Evaluate all joined tasks after every stage.
5. Choose checkpoints using no-forgetting/family metrics.

Initial replay mix proposal:

```text
current task replay: 50%
anchor 01125 replay: 20%
previous task replay: 30%
```

This is lower risk than immediately implementing one giant mixed-assembly IsaacLab environment.

## 8. Proposed Implementation Plan

### Step 1: Make `Buffer` load/save and partial sampling reusable

File:

- `tdmpc2/common/buffer.py`

Needed changes:

- Allow `Buffer.sample(device, batch_size=None)`.
- Keep default behavior unchanged.
- Add robust save/load for replay buffers with metadata:
  - task id / assembly id.
  - number of episodes.
  - horizon.
  - obs/action/task shapes.
  - storage contents.

Reason:

- Later family stages need to load previous task replay and sample a controlled number of sequences.

### Step 2: Add an online family replay wrapper

New file:

- `tdmpc2/common/online_family_replay.py`

Main class:

```python
class OnlineFamilyReplayBuffer:
    def __init__(self, current_buffer, replay_buffers, cfg):
        ...

    def add(self, td, world_size=1, rank=0):
        return self.current_buffer.add(td, world_size, rank)

    def sample(self, device):
        # sample current / anchor / history by ratio
        ...
```

Important behavior:

- `add()` always writes only to the current task buffer.
- `sample()` mixes current + old buffers.
- Before current buffer has enough episodes, sample only current buffer to avoid old replay dominating.
- Return exactly the same tuple expected by TD-MPC2:
  - `obs`
  - `action`
  - `reward`
  - `task`

### Step 3: Add clean config fields

File:

- `tdmpc2/config.py`

Suggested fields:

```python
online_family_replay_enabled: bool = False
online_family_replay_manifest_fp: Optional[str] = None
online_family_replay_save_fp: Optional[str] = None
online_family_current_task_id: Optional[str] = None
online_family_anchor_task_id: str = "01125"
online_family_current_ratio: float = 0.50
online_family_anchor_ratio: float = 0.20
online_family_history_ratio: float = 0.30
online_family_min_current_episodes: int = 5
online_family_replay_max_episodes_per_task: Optional[int] = None
online_family_replay_save_every_eval: bool = True
```

Do not reuse `multitask_continuation_*` for this. That name should remain tied to the offline continuation path.

### Step 4: Wire the wrapper into `tdmpc2/train.py`

File:

- `tdmpc2/train.py`

After creating the normal online `Buffer`, wrap it when enabled:

```python
if cfg.online_family_replay_enabled:
    buffer = OnlineFamilyReplayBuffer.from_manifest(buffer, cfg)
```

The trainer should not need major changes because `OnlineFamilyReplayBuffer` will keep the same `add()` and `sample()` interface.

### Step 5: Save current replay during training

File:

- `tdmpc2/trainer.py`

Add optional save points:

- after eval checkpoint save.
- at training finish.

Example behavior:

```python
if cfg.online_family_replay_save_fp:
    self.buffer.save_current(cfg.online_family_replay_save_fp)
```

Need to handle both plain `Buffer` and wrapped `OnlineFamilyReplayBuffer`.

### Step 6: Add a new launcher

New file:

- `scripts/run_01125_online_family_replay_targets.sh`

Recommended semantics:

- This becomes the recommended family training entry.
- It should:
  1. train current target;
  2. save current replay;
  3. append/update replay manifest;
  4. batch eval all joined tasks;
  5. choose next checkpoint.

Example target order:

```text
01125 -> 00256 -> 00186 -> 00004 -> 00014 -> 00062 -> 00271
```

Suggested first smoke run:

```text
01125 -> 00256 -> 00186
steps_per_task=50000
num_envs=64
eval_episodes=20
multiproc=false
```

Only after this works should the full setting return to:

```text
steps_per_task=600000
num_envs=300
multiproc=true
num_gpus=2
```

### Step 7: Separate size curriculum from assembly curriculum

Recommended sequence:

1. Train family only on `1.0:1.0`.
2. Add moderate sizes:
   - `0.5:1.0`
   - `2.0:1.5`
3. Add extremes:
   - `0.5:0.5`
   - `4.0:2.0`

This makes failure attribution easier.

## 9. Questions for Pro Review

Please analyze these specific questions:

1. Should the next implementation prioritize staged online family replay, or true mixed-assembly online environments?
2. Is 50/20/30 current-anchor-history replay a reasonable starting ratio for TD-MPC2 here?
3. Should old replay be sampled uniformly by task, proportional to buffer size, or weighted by recent retention failure?
4. Which checkpoint metric should gate progression:
   - `relaxed_success`,
   - `terminal_process_success`,
   - `strict_success`,
   - or a composite including force/jam/lateral error?
5. Should 00062/00271 be included early, or held out as hard cases until the family baseline stabilizes?
6. Should the action history / force history context be frozen during multi-task replay, or trained normally?
7. Is offline data useful only as replay regularization, or should it still be part of world-model pretraining?
8. Should size templates be represented as separate task entries in replay manifests, or kept as continuous task vectors within each assembly?

## 10. Current Working-Tree Notes

Current uncommitted files include video/eval tooling changes:

```text
M docs/eval.md
M scripts/batch_record_task_sizes.py
M tdmpc2/common/logger.py
M tdmpc2/config.py
M tdmpc2/envs/isaaclab.py
M tdmpc2/eval.py
?? scripts/batch_record_ids_task_sizes.py
?? data/video_eval_ids_task_sizes/
```

These are not directly part of the multi-task training redesign, but they are useful for diagnostics and should not be accidentally reverted.

