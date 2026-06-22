# AGENTS.md

Use `AGENT.md` as the canonical project guidance file for this repository.

The current active direction is SRSA staged online family replay V2
(`acquisition-first`):

- keep each online train job single-assembly;
- learn the current target before handing off to the next stage;
- use current-heavy replay during acquisition, with no history replay by default;
- gate new-task handoff on eval success instead of fixed 50k-step stages;
- save current replay after each acquisition stage;
- run retention eval only after the acquisition gate is reached;
- treat single-task success as evidence that the target is learnable, not as proof
  that task conditioning is used;
- keep `online_family_replay_*` separate from offline `multitask_continuation_*`.

Current validated result:

- `01125 -> 00256` acquisition reached `00256 episode_success=0.9023`
  at 299,520 env steps;
- retention after 00256: `01125=0.75`, `00256=0.90`, family mean `0.825`.
- task-vector swap on 00256: normal `0.90`, forced 01125 vector `0.90`,
  forced zero vector `0.75`; current policy is only weakly task-vector sensitive.
- 2026-06-19 replay/task check passed at the storage and sampling layer:
  `01125.pt` and `00256.pt` store per-transition `[N, 6]` task tensors, and a
  50/50 online-family mixed sample returned `task.shape=[3,64,6]` with both
  task vectors present.
- 2026-06-19 offline paired sensitivity report:
  `logs/task_vec_sensitivity/20260619_00256_v2_offline_report.json`; action,
  Q, reward, and next-latent deltas are near numerical noise when only
  `task_vec_6` changes.
- A zero-init task context FiLM adapter is now available behind
  `task_context_adapter_enabled=true`; it is disabled by default and old
  checkpoints load into an initially equivalent model.

Default next experiment:

- keep retention gates enabled before any next task;
- treat the current issue as "`task_vec_6` is not yet indispensable", not as a
  simple wiring failure;
- start from the 20260619 polish checkpoint, run 01125/00256 at 50/50 replay
  with `TASK_CONTEXT_ADAPTER_ENABLED=true`,
  `TASK_CONTEXT_ADAPTER_SOURCE=raw_task_vec`, and
  `TASK_CONTEXT_ADAPTER_ALPHA=0.05`; rerun paired sensitivity before cautious
  `00186` acquisition.
