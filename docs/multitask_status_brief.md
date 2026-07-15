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

Implementation note:

- The current smoke launcher keeps `srsa_param_template_id=2`, whose template is `c1.0-d1.0`.
- With `eval_task_template_exact=true`, this exact template is what fixes Phase A size.
- Later size curriculum runs should deliberately revisit `eval_task_template_exact` / sampler-template behavior instead of assuming the semicolon template string alone controls the active distribution.

## 9. Adopted V1 Decisions

The next implementation should prioritize staged online family replay.

Resolved choices:

- Do not build a true mixed-assembly IsaacLab environment for V1.
- Use `online_family_replay_*` config names; keep `multitask_continuation_*` for the existing offline continuation route.
- Start with 50/20/30 current-anchor-history replay.
- Sample history replay uniformly by task for V1, not proportional to buffer size.
- Train contact/action history normally; do not freeze it in the first implementation.
- Keep size/clearance/depth as continuous `task_vec_6` values inside each assembly replay, not as separate manifest task ids.
- Treat offline family data as smoke/format/debug data for now, not as the main learning signal.
- Hold out `00062` and `00271` until the 5-task baseline has meaningful retention.

Initial family checkpoint score:

```text
family_score = 0.7 * mean_relaxed_success + 0.3 * min_relaxed_success
```

This is intentionally simple. Later versions can add terminal/process success, jam rate, force, and lateral error.

## 10. Implemented Scaffold

The online family replay scaffold now has these project surfaces:

- `tdmpc2/common/buffer.py`
  - `Buffer.sample(device, batch_size=None)`
  - replay snapshot `save()` / `load()`
  - `save_current()` for trainer integration
- `tdmpc2/common/online_family_replay.py`
  - `OnlineFamilyReplayBuffer`
  - current-only warmup until `online_family_min_current_episodes`
  - current / anchor / uniform-history mixed sampling
- `tdmpc2/config.py`
  - `online_family_replay_*` fields and validation
- `tdmpc2/train.py`
  - wraps the normal online buffer when `online_family_replay_enabled=true`
- `tdmpc2/trainer.py`
  - saves current replay at eval and finish when `online_family_replay_save_fp` is set
- `scripts/update_online_family_replay_manifest.py`
  - updates the manifest with real replay snapshot metadata
- `scripts/run_01125_online_family_replay_targets.sh`
  - staged training launcher
  - replay save / manifest update / joined-task eval
- `configs/train/srsa_01125_online_family_replay.yaml`
  - smoke-default online family replay config
- `data/replay_manifests/01125_online_family.json`
  - empty manifest template

## 11. First Commands

Dry-run the V1 smoke route:

```bash
DRY_RUN=1 \
CONFIG_NAME=srsa_01125_online_family_replay \
TARGETS="01125 00256 00186" \
STEPS_PER_TASK=50000 \
NUM_ENVS=64 \
MULTIPROC=false \
NUM_GPUS=1 \
scripts/run_01125_online_family_replay_targets.sh
```

Then run the smoke route:

```bash
CONFIG_NAME=srsa_01125_online_family_replay \
TARGETS="01125 00256 00186" \
STEPS_PER_TASK=50000 \
NUM_ENVS=64 \
MULTIPROC=false \
NUM_GPUS=1 \
scripts/run_01125_online_family_replay_targets.sh
```

After smoke passes, scale in this order:

1. `01125 -> 00256 -> 00186 -> 00004 -> 00014`, still fixed at `1.0:1.0`.
2. Full family with `00062` and `00271` added last.
3. Size curriculum after assembly retention is stable.

## 12. 2026-06-17 Update: Acquisition-First V2

The first online-family 00256 stage should not be interpreted as a task or
architecture failure. The failed online-family stage trained 00256 for only
about 50k env steps and stopped at `episode_success=0.0`.

Follow-up single-target controls showed the missing piece was training time:

- `00256` from the original 01125 checkpoint reached high success after longer
  direct fine-tuning, with best observed eval success around `0.97`.
- `00256` from the online-family stage-1 01125 checkpoint reached
  `episode_success=0.8672` at 249,600 env steps and `0.8867` at 299,520 env
  steps.
- Therefore the stage-1 checkpoint is still a viable transfer source, and
  00256 is learnable. The 50k online-family failure mainly implicates stage
  scheduling and premature handoff.

New default direction:

- Use acquisition-first stages instead of fixed 50k stages.
- Train the current target until it reaches a success gate or the full budget.
- Keep replay current-heavy during acquisition, e.g. current/anchor/history
  `0.80/0.20/0.0`.
- Update the online-family replay manifest and run retention eval only after
  the acquisition gate is reached.
- Do not use single-task success alone to conclude that `AxialTaskEncoder` is
  effective; a task-vector swap diagnostic is still needed.

Implemented V2 surfaces:

- `online_family_acquisition_*` config fields in `tdmpc2/config.py`.
- Eval-time acquisition status and early stop in `tdmpc2/trainer.py`.
- Gate-aware staged launcher behavior in
  `scripts/run_01125_online_family_replay_targets.sh`.
- Dedicated 00256 acquisition entry:
  `scripts/run_01125_online_family_acquire_targets.sh`.
- Eval-only task-vector override in `tdmpc2/batch_eval_tasks.py`:
  `batch_eval_force_task_vec_6` and `batch_eval_force_task_vec_label`.

Recommended next command on physical GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 \
RUN_STAMP=$(date +%Y%m%d_%H%M%S) \
scripts/run_01125_online_family_acquire_targets.sh
```

Inside that process, `GPU_ID=0` is correct because `CUDA_VISIBLE_DEVICES=1`
makes the physical GPU 1 appear as local CUDA device 0.

Task-vector diagnostic after a strong 00256 checkpoint:

```bash
CUDA_VISIBLE_DEVICES=1 \
/home/gpuserver/miniconda3/envs/isaac51/bin/python tdmpc2/batch_eval_tasks.py \
  --config-dir configs/train \
  --config-name srsa_01125_imitation_relaxed \
  checkpoint=/path/to/00256/models/latest.pt \
  eval_assembly_ids="[00256]" \
  batch_eval_episodes_per_task=20 \
  batch_eval_force_task_vec_label=01125_vec \
  'batch_eval_force_task_vec_6="[0,-0.155195,0.145688,0.165645,1,0]"' \
  batch_eval_output_dir=logs/task_vec_swap_eval/00256_with_01125_vec
```

Compare that with a normal 00256 eval and a zero-vector eval. If the success
rate barely changes, the policy is likely ignoring task conditioning. If the
wrong vector sharply degrades success, the task encoder is influencing control.

Observed 2026-06-18 swap result on
`20260618_001734_stage-2_asm-00256/models/latest.pt`:

| Eval | 00256 env model input | relaxed success | strict success |
| --- | --- | ---: | ---: |
| normal | 00256 task vector | 0.90 | 0.00 |
| forced anchor | 01125 task vector | 0.90 | 0.00 |
| forced zero | zero vector | 0.75 | 0.00 |

Interpretation:

- The current V2 checkpoint is only weakly sensitive to `task_vec_6`.
- The policy can still solve 00256 when the model receives the 01125 vector.
- Before scaling to many more assemblies, add retention gates and either
  strengthen task conditioning or run a more discriminative mixed-task
  diagnostic.

## 13. 2026-06-19 Update: Task-Vector Indispensability Plan

The refined diagnosis is:

```text
The current problem is not simply that task_vec_6 is disconnected.
The problem is that task_vec_6 is not yet an irreplaceable information source.
```

The model can still solve relaxed 00256 with weak or wrong task-vector input
because state, contact feedback, MPC correction, and single-task bias paths can
carry much of the control signal. The next revision should therefore create and
measure pressure where the same class of state requires different predictions,
values, or actions under different `task_vec_6` values.

### Step 1 Completed: Replay and Mixed-Batch Task Tensor Check

Checked replay snapshots:

- `logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_replay_from_01125/20260615_202326_launcher/replay/01125.pt`
- `logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_acquire_from_01125/20260618_001734_launcher/replay/00256.pt`

Observed storage:

| Replay | stored task shape | unique task vectors | expected vector match |
| --- | --- | ---: | --- |
| `01125.pt` | `[38400, 6]` | 1 | yes, max abs diff about `3.9e-7` |
| `00256.pt` | `[230400, 6]` | 1 | yes, max abs diff about `1.4e-6` |

Stored task vectors:

- 01125: `[0, -0.155195, 0.145688, 0.165645, 1, 0]`
- 00256: `[0, -0.178871, 0.099081, 0.115353, 1.2189, 0]`

Mixed-sampling check:

- Current buffer: `00256.pt`.
- Manifest: `20260618_001734_launcher/online_family_replay_manifest.json`.
- Ratios for the diagnostic sample: current/anchor/history `0.50/0.50/0.0`.
- Sampled batch shapes:
  - `obs`: `[4, 64, 17]`
  - `action`: `[3, 64, 3]`
  - `reward`: `[3, 64, 1]`
  - `task`: `[3, 64, 6]`
- `last_batch_task_counts`: `{'01125': 32, '00256': 32}`.
- `task_vec_6` std was nonzero on dimensions that differ between the tasks.

Training update path:

- `Trainer.to_td()` uses the runtime SRSA task vector when writing current
  online rollouts into replay.
- `TDMPC2.update()` calls `buffer.sample(device=self.device)`, receives
  `obs, action, reward, task`, and passes that sampled `task` into
  `_update()` / `_loss_fn()`.
- The update path is therefore not replacing a mixed replay batch's `task`
  tensor with a single current-env task vector.

Conclusion:

- Replay save/load and `OnlineFamilyReplayBuffer` mixed sampling are preserving
  per-transition task tensors.
- The current evidence does not show a global `env.current_task_vec` broadcast
  overwrite at the replay/wrapper layer.
- The remaining diagnosis should focus on whether the trained model changes
  behavior and predictions when only `task_vec_6` changes.

### Step 2: Paired Sensitivity Diagnostic

Do not rely on success-rate swap alone. The next diagnostic should use paired
states:

```text
same seed
same initial state
same 00256 environment
only task_vec_6 changes
```

Compare at least these task-vector conditions:

- correct 00256 vector
- anchor 01125 vector
- 00186 vector
- zero vector
- random vector
- extreme vector

Record paired deltas:

- `delta_action = ||pi(s, task_i) - pi(s, task_j)||`
- `delta_Q`
- `delta_reward_pred`
- `delta_next_latent`
- return / relaxed success / strict success delta
- MPC selected-plan delta, if easy to expose

If action/Q/reward/dynamics deltas are near zero, the task-conditioning path is
being ignored by the learned model. If deltas are visible but success is stable,
the vector is affecting decisions, but the tasks may be behaviorally compatible
or MPC/contact feedback is correcting the wrong prior.

Implemented offline paired report:

- Script: `tdmpc2/scripts/task_vec_sensitivity_report.py`
- Report:
  `logs/task_vec_sensitivity/20260619_00256_v2_offline_report.json`
- Checkpoint:
  `logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_acquire_from_01125/20260618_001734_stage-2_asm-00256/models/latest.pt`
- Replay states/actions:
  `logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_acquire_from_01125/20260618_001734_launcher/replay/00256.pt`
- Sample size: 512 finite replay transitions.

Command used:

```bash
/home/gpuserver/miniconda3/envs/isaac51/bin/python \
  tdmpc2/scripts/task_vec_sensitivity_report.py \
  --cpu \
  --batch-size 512 \
  --output logs/task_vec_sensitivity/20260619_00256_v2_offline_report.json \
  --condition-from-replay old_00186=logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_replay_from_01125/20260615_202326_launcher/replay/00186.pt
```

Observed deltas versus the correct 00256 vector:

| Swapped task vector | action L2 mean | Q abs mean | reward abs mean | next-latent L2 mean |
| --- | ---: | ---: | ---: | ---: |
| 01125 replay vector | `4.43e-7` | `2.09e-5` | `9.81e-7` | `1.84e-6` |
| old 00186 replay vector | `4.76e-7` | `2.18e-5` | `9.42e-7` | `1.92e-6` |
| zero vector | `4.79e-7` | `2.11e-5` | `9.78e-7` | `1.91e-6` |
| random vector | `3.44e-6` | `3.79e-5` | `3.58e-6` | `1.17e-5` |
| extreme vector | `4.51e-7` | `2.12e-5` | `9.99e-7` | `1.86e-6` |

Scale reference under the correct vector:

- action mean norm: about `0.80`
- latent norm: about `4.20`
- predicted Q scalar mean: about `132.3`
- predicted reward scalar mean: about `6.40`

Interpretation:

- The current model is almost invariant to `task_vec_6` in this offline paired
  probe.
- The result is stronger than the closed-loop success swap result because the
  same replay states/actions were reused and only `task_vec_6` changed.
- The next useful training experiment should force mixed 01125/00256 replay
  pressure and log batch task entropy/per-task behavior. If sensitivity remains
  near zero after that, a zero-init FiLM/residual adapter is justified.

### Step 3: Retention Polish With Real Mixed-Task Pressure

Before adding another assembly, run an 01125/00256 retention-polish stage whose
training batches are forced to contain both tasks, ideally 50/50 for the
diagnostic polish run.

Required logging:

- batch task counts and batch task entropy
- per-task world-model loss
- per-task reward loss
- per-task value/policy loss when available
- per-task eval success, including relaxed/process/strict/lateral metrics

The purpose is not only to improve average success. The purpose is to make it
hard for one task to overwrite the other while giving the model repeated
examples where similar states have different task-conditioned value/action
targets.

Implemented logging support:

- `tdmpc2/tdmpc2.py:TDMPC2.update()` now reads
  `OnlineFamilyReplayBuffer.last_batch_task_counts` and
  `last_batch_source_counts` after each sample.
- Logged metrics include:
  - `online_family_batch_task_count_<task_id>`
  - `online_family_batch_task_frac_<task_id>`
  - `online_family_batch_task_entropy`
  - `online_family_batch_task_entropy_norm`
  - `online_family_batch_num_tasks`
  - `online_family_batch_source_count_<source>`
  - `online_family_batch_source_frac_<source>`
- This verifies whether retention-polish updates are actually mixed, e.g.
  01125/00256 near 50/50 with entropy norm near 1.0.

Still pending for a deeper retention-polish run:

- per-task world-model/reward/value/policy losses
- paired sensitivity report after polish
- closed-loop paired eval with identical initial states, if practical

Recommended 01125/00256 retention-polish command on physical GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 \
GPU_ID=0 \
RUN_STAMP=20260619_00256_retention_polish \
EXP_NAME=srsa_axial_online_family_polish_01125_00256 \
SOURCE_CHECKPOINT=logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_acquire_from_01125/20260618_001734_stage-2_asm-00256/models/latest.pt \
TARGETS=00256 \
RESUME_COMPLETED_TARGETS=01125 \
CURRENT_RATIO=0.50 \
ANCHOR_RATIO=0.50 \
HISTORY_RATIO=0.0 \
MIN_CURRENT_EPISODES=5 \
STEPS_PER_TASK=100000 \
NUM_ENVS=256 \
RETENTION_NUM_ENVS=256 \
ACQUISITION_STOP_ENABLED=false \
ACQUISITION_REQUIRE_SUCCESS=false \
RETENTION_REQUIRE_GATE=true \
UPDATE_PROGRESS_LOG_EVERY=50 \
scripts/run_01125_online_family_acquire_targets.sh
```

Dry-run validation passed with the same command shape and `DRY_RUN=1`.

### Step 4: Zero-Init Task Context Adapter

The 01125/00256 retention-polish run passed retention gates, but the paired
sensitivity report still showed near-zero action/Q/reward/dynamics deltas when
only `task_vec_6` changed. The low-risk architecture change has therefore been
implemented, but remains disabled by default:

- `task_vec_6 -> AxialTaskEncoder -> task_ctx -> zero-init FiLM residual`
- implementation: `tdmpc2/models/task_context_adapter.py`
- insertion sites: encoder latent, dynamics output, policy prior, reward head,
  and Q head
- config switch: `task_context_adapter_enabled=true`
- old checkpoints load into an initially equivalent model because each adapter's
  final projection is reset to zero after global model initialization
- metrics log adapter parameter norms such as
  `task_context_adapter_encoder_final_weight_norm`

Smoke validation passed:

- `py_compile` on modified modules
- `git diff --check`
- launcher dry-run with `TASK_CONTEXT_ADAPTER_ENABLED=true`
- strict load of the 20260619 retention-polish checkpoint into an adapter-enabled
  model, with all keys matched and adapter final weight norm `0.0`

Recommended first adapter experiment on physical GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 \
GPU_ID=0 \
RUN_STAMP=20260622_taskctx_adapter_00256 \
EXP_NAME=srsa_axial_online_family_taskctx_adapter_01125_00256 \
SOURCE_CHECKPOINT=logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_polish_01125_00256/20260619_0135_retention_polish_stage-2_asm-00256/models/latest.pt \
TARGETS=00256 \
RESUME_COMPLETED_TARGETS=01125 \
CURRENT_RATIO=0.50 \
ANCHOR_RATIO=0.50 \
HISTORY_RATIO=0.0 \
MIN_CURRENT_EPISODES=5 \
STEPS_PER_TASK=100000 \
NUM_ENVS=256 \
RETENTION_NUM_ENVS=256 \
ACQUISITION_STOP_ENABLED=false \
ACQUISITION_REQUIRE_SUCCESS=false \
RETENTION_REQUIRE_GATE=true \
TASK_CONTEXT_ADAPTER_ENABLED=true \
TASK_CONTEXT_ADAPTER_HIDDEN_DIM=128 \
TASK_CONTEXT_ADAPTER_ALPHA=1.0 \
UPDATE_PROGRESS_LOG_EVERY=50 \
scripts/run_01125_online_family_acquire_targets.sh
```

After this run, rerun `tdmpc2/scripts/task_vec_sensitivity_report.py` against
the new checkpoint. The acceptance signal is not just high relaxed success; the
adapter route should produce clearly larger paired deltas than the previous
`1e-6`-scale task-vector swaps while preserving 01125/00256 retention.

### 2026-06-22 Adapter Run Diagnosis

Run inspected:

- `logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_taskctx_adapter_01125_00256/20260622_taskctx_adapter_00256_launcher`

Observed:

- No training process remained; GPU1 was idle after the run.
- Training completed, but degraded from the source polish checkpoint:
  - pre-update train rollout: `episode_success=0.949`
  - 49,920 eval: `episode_success=0.250`
  - 99,840 eval: `episode_success=0.578`
- Adapter weights moved quickly:
  - final `task_context_adapter_pi_final_weight_norm=11.93`
  - final `task_context_adapter_q_final_weight_norm=16.47`
- Family eval initially reported 0/20 on both 01125 and 00256, but that eval
  path had a bug: the batch-eval worker did not inherit
  `task_context_adapter_*` overrides, and checkpoint compatibility did not infer
  adapter-enabled checkpoints. This has been fixed in:
  - `tdmpc2/batch_eval_tasks.py`
  - `tdmpc2/collect_eval_rollouts.py`
- Even with adapter-aware sensitivity loading, realistic task-vector swaps
  remained near numerical noise:
  - `anchor_from_replay action_l2=1.727e-07`
  - `old_00186 action_l2=1.820e-07`
  - `zero action_l2=1.711e-07`

Deeper cause:

- The adapter was using learned `task_ctx` from `AxialTaskEncoder`, but that
  context has collapsed for realistic SRSA vectors:
  - 00256 vs 01125 task-context L2: about `1.8e-08`
  - 00256 vs 00186 task-context L2: about `4.2e-08`
- Therefore the adapter learned a mostly task-independent modulation and damaged
  the policy/value landscape without making `task_vec_6` indispensable.

Follow-up implementation:

- `task_context_adapter_source` was added.
- Supported values:
  - `task_context`: old adapter source, kept as default for compatibility
  - `raw_task_vec`: feed raw 6D `task_vec_6` directly to the adapter
  - `both`: concatenate learned task context and raw task vector
- For the next experiment, use `raw_task_vec` and a smaller alpha:

```bash
CUDA_VISIBLE_DEVICES=1 \
GPU_ID=0 \
RUN_STAMP=20260622_rawtask_adapter_00256 \
EXP_NAME=srsa_axial_online_family_rawtask_adapter_01125_00256 \
SOURCE_CHECKPOINT=logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_polish_01125_00256/20260619_0135_retention_polish_stage-2_asm-00256/models/latest.pt \
TARGETS=00256 \
RESUME_COMPLETED_TARGETS=01125 \
CURRENT_RATIO=0.50 \
ANCHOR_RATIO=0.50 \
HISTORY_RATIO=0.0 \
STEPS_PER_TASK=100000 \
NUM_ENVS=256 \
RETENTION_NUM_ENVS=256 \
ACQUISITION_STOP_ENABLED=false \
ACQUISITION_REQUIRE_SUCCESS=false \
RETENTION_REQUIRE_GATE=true \
TASK_CONTEXT_ADAPTER_ENABLED=true \
TASK_CONTEXT_ADAPTER_SOURCE=raw_task_vec \
TASK_CONTEXT_ADAPTER_ALPHA=0.05 \
UPDATE_PROGRESS_LOG_EVERY=50 \
scripts/run_01125_online_family_acquire_targets.sh
```

### 2026-06-22 Route Lock: Two Tasks Before Three

Do not make three-task early mixed training the next mainline experiment. It is
theoretically attractive because it would pressure the task representation from
the start, but it would also entangle too many variables in the current state:
`00186` is harder than `00256`, the online path is still single-assembly env plus
family replay rather than a true mixed-assembly IsaacLab env, and the adapter
strength/source questions are not resolved.

The active mainline is therefore:

```text
01125 + 00256 diagnostic first
00186 expansion only after the diagnostic passes
```

Default next launcher:

```bash
CUDA_VISIBLE_DEVICES=1 \
GPU_ID=0 \
scripts/run_01125_00256_rawtask_adapter_diagnostic.sh
```

This wrapper fixes the run shape to:

- source checkpoint:
  `logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_polish_01125_00256/20260619_0135_retention_polish_stage-2_asm-00256/models/latest.pt`
- tasks: `01125 + 00256`
- replay mix: `CURRENT_RATIO=0.50`, `ANCHOR_RATIO=0.50`,
  `HISTORY_RATIO=0.0`
- adapter: `TASK_CONTEXT_ADAPTER_ENABLED=true`,
  `TASK_CONTEXT_ADAPTER_SOURCE=raw_task_vec`,
  `TASK_CONTEXT_ADAPTER_ALPHA=0.005`
- retention gate: enabled before any next-task handoff

Two-task pass criteria:

- `00256` relaxed success stays near the known acceptable level.
- `01125` retention does not clearly regress.
- update logs show real 50/50 task batches for `01125` and `00256`.
- paired `task_vec_6` sensitivity is no longer at `1e-7` numerical-noise scale
  for action/Q/reward/next-latent deltas.

Only after those pass, run the cautious three-task ablation:

```text
01125 + 00256 + 00186
```

Use the online-family ratios to approximate:

```text
01125: 0.34
00256: 0.33
00186: 0.33
```

In the current launcher shape, that means an `00186` acquisition stage whose
manifest already contains `01125` and `00256`, with `01125` as anchor, `00256`
as history, and `00186` as current. Keep
`srsa_axial_clearance_depth_templates=1.0:1.0`; do not add `00062`, `00271`, or
size-generalization templates in the same experiment.

If the three-task ablation fails, do not immediately reject the multi-task route.
First check whether paired sensitivity collapsed back to noise, then whether
`00186` is still weak under single-task/direct fine-tune, and only then decide
whether `AxialTaskEncoder` needs a raw residual path, reconstruction loss, or
spread loss.

### 2026-06-23 Raw-Task Adapter Result

Run inspected:

- `logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_rawtask_adapter_01125_00256/20260622_194546_launcher`

Training completed, but the run failed the two-task gate:

- pre-update rollout on `00256`: `episode_success=0.949`
- 49,920-step eval: `episode_success=0.1445`
- 99,840-step eval: `episode_success=0.0`
- best checkpoint: `best_step-49920_s-0p1445.pt`
- latest checkpoint: `99,840` steps with `episode_success=0.0`

Family eval after the run:

| Assembly | relaxed success | strict success | official latched | mean lateral error |
| --- | ---: | ---: | ---: | ---: |
| `01125` | `0.00` | `0.00` | `0.85` | `103.1 mm` |
| `00256` | `0.00` | `0.00` | `0.25` | `98.4 mm` |

Sampling was not the problem. Update logs show exact 50/50 mixed batches:

- `online_family_batch_task_count_01125=512`
- `online_family_batch_task_count_00256=512`
- `online_family_batch_task_entropy_norm=1.0`

Adapter weights still moved quickly even with `alpha=0.05`:

- final encoder adapter weight norm: `7.92`
- final dynamics adapter weight norm: `8.15`
- final policy adapter weight norm: `7.63`
- final reward adapter weight norm: `5.91`
- final Q adapter weight norm: `54.21`

Paired sensitivity was rerun with
`tdmpc2/scripts/task_vec_sensitivity_report.py` after fixing the loader to read
`task_context_adapter_alpha` from checkpoint metadata. The fixed report is:

- `logs/task_vec_sensitivity/20260622_rawtask_adapter_00256_report.json`

Mean deltas versus the correct `00256` task vector:

| Swap | action L2 | Q abs | reward abs | next-latent L2 |
| --- | ---: | ---: | ---: | ---: |
| `01125` replay vector | `0.0595` | `1.64` | `0.227` | `0.562` |
| old `00186` replay vector | `0.0752` | `22.21` | `0.393` | `1.170` |
| zero vector | `0.0888` | `5.33` | `0.437` | `1.257` |
| random vector | `0.1025` | `28.96` | `0.410` | `1.532` |

Interpretation:

- `raw_task_vec` fixed the invariance symptom: task-vector swaps now produce
  large action/Q/reward/dynamics deltas rather than `1e-7` numerical noise.
- The full-site adapter at `alpha=0.05` is too destructive for the current
  two-task policy; it turns task conditioning into a strong perturbation before
  the policy/value landscape can absorb it.
- This is not a green light for `00186`. The two-task gate failed on retention
  and current-task success.

Next recommendation:

- stay on `01125 + 00256`;
- restart from the clean 20260619 polish checkpoint;
- try a weaker raw-task adapter first, e.g.
  `TASK_CONTEXT_ADAPTER_ALPHA=0.005`;
- keep the same 50/50 replay and retention gates;
- rerun paired sensitivity and family eval before any three-task ablation.
