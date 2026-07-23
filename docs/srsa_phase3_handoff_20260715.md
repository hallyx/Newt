# SRSA Phase 3 Handoff - 2026-07-15

## Scope And Current State

Repository: `/home/gpuserver/hx/github/Newt`

This handoff covers the completed SRSA task-conditioning investigation from Phase 0.5 through Phase 3.10. The active result is not a new training recipe: it is a negative transfer result for policy-only elite adaptation on the three-task pilot.

Current decision: `DO_NOT_INTEGRATE_POLICY_ONLY_ELITE_ADAPTATION`.

- `00186` is a medium-distance third-task pilot, not an easy task.
- Do not start three-task consolidation.
- Do not add a fourth task or use `00062`/`00271`.
- Do not enable counterfactual reward, reward residual, task-specific policy heads, PCGrad, or MPPI sampling changes from the existing audits.
- Do not modify TD-MPC2 reward, Q, dynamics, task_context, MPPI, or replay sampler without a new diagnostic decision.

## Evidence Chain

### Task And Environment Validity

- Phase 0.5: replay task tensors are per-transition `[N, 6]`; task id maps to a unique task-vector hash; `01125` and `00256` hashes differ; mixed replay label/hash counts match.
- Phase 0.7: `01125` and `00256` differ in runtime asset/mesh/collision, target pose/depth, thresholds, and contact dynamics. Size parameters are `GEOMETRY_OBJECTIVE_REWARD_CONTACT_EFFECTIVE`, not label-only.
- Phase 1.0: the model is downstream task-sensitive, but `AxialTaskEncoder` task context was collapsed.

### Task Context Repair

- Phase 2 repair added normalized task-vector support, zero-init raw residual, decoder/reconstruction, and auxiliary losses behind disabled-by-default configuration.
- Phase 2.1: context L2 for `01125` vs `00256` improved from `1.36e-08` to `0.01298`; real task reconstruction R2 was `0.983`.
- Phase 2.2: zero/random vectors changed strict/process/keypoint/lateral/reward while relaxed success was retained for the near pair. Phase 2 was accepted as `PASS_WITH_CAVEAT`.
- `01125` and `00256` remain a near pair. Their relaxed swap invariance is not evidence that task vectors are ignored.

### Three-Task Pilot

- Phase 3.0 selected `00186`; `00004` and `00014` are backups. `00062` and `00271` are hard canaries.
- Phase 3.1 representation passed: context pairwise L2 `01125/00256=0.01293`, versus `00186` distances about `1.29`; reconstruction R2 `0.999905`; context/task-distance correlation `0.997172`.
- Phase 3.1 acquisition failed: `00186` relaxed `0.15`, jam `0.70`, lateral/keypoint `24.149/28.815 mm`; old-task relaxed retention passed but old strict/process quality did not.
- Phase 3.2 current-heavy rescue (`0.75/0.125/0.125`) produced non-monotonic `00186` acquisition. Do not add more generic acquisition steps before diagnosing the mismatch.
- Phase 3.3 main root cause: `POLICY_PROPOSAL_FAILURE` versus a comparable direct fine-tune checkpoint.
- Phase 3.4: more policy proposals, wider proposal std, and random candidates did not rescue contact regret. Direct proposal oracle helped only `32.2%`, below the required `50%`; do not change MPPI candidate composition.
- Phase 3.5 found policy-gradient conflict, but Phase 3.6 one-step virtual updates were too small to decide a remedy.
- Phase 3.7: even `00186`-only policy imitation reduced BC loss without improving proposal regret: `POLICY_OBJECTIVE_MISALIGNMENT`.
- Phase 3.8: frozen-world-model MPPI elite distillation strongly improved held-out proposal regret (`+0.972` at 100 updates) but caused old-task action drift around `0.30`. The target has signal, but the new-task/retention tradeoff is real.
- Phase 3.9: elite distillation plus frozen old-policy behavior anchor, lambda `3`, 100 in-memory policy updates passed the offline gate: proposal-regret reduction `0.976`, contact/jam action-L2 reduction `0.367`, old drifts `0.0355/0.0279`.

## Phase 3.10 - Closed-Loop Falsification Of The Offline Result

Source checkpoint, never modified:

```text
logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_taskctx_repair_01125_00256_00186_rescue/20260713_phase3_2_rescue_00186_stage-3_asm-00186/models/best_step-50176_s-0p2461.pt
```

Script:

```text
tdmpc2/scripts/phase3_three_task_pilot/policy_only_anchored_adaptation_ablation.py
```

It generated independently saved full checkpoint payloads, with every non-`_pi` model tensor exact-equal to the source:

```text
reports/phase3_three_task_pilot/phase3_10_policy_only_checkpoints/elite_only_100.pt
reports/phase3_three_task_pilot/phase3_10_policy_only_checkpoints/elite_behavior_anchor_l3_100.pt
reports/phase3_three_task_pilot/phase3_10_policy_only_checkpoints/elite_behavior_anchor_l10_100.pt
```

Protocol:

- only `WorldModel._pi` updated for 100 Adam steps;
- encoder, dynamics, reward, Q, task encoder/context, MPPI, and sampler frozen;
- `00186` target: frozen multitask-world-model top-8 elite action target from 3 policy + 61 Gaussian candidates at horizon 3;
- `01125`/`00256` anchors: frozen pre-update policy output MSE;
- no direct policy/rollout enters target construction;
- all variants closed-loop evaluated with the same exact-template, headless, `num_envs=256`, `20` episodes/task contract.

Result: `RETENTION_FAILURE`.

| Variant | 00186 relaxed | 00186 lateral/keypoint mm | 00186 jam | 01125 relaxed/process | 00256 relaxed/process |
| --- | ---: | ---: | ---: | ---: | ---: |
| source | 0.40 | 2.504 / 5.683 | 0.30 | 0.90 / 0.30 | 1.00 / 0.10 |
| elite-only | 0.20 | 4.126 / 6.742 | 0.25 | 0.90 / 0.10 | 1.00 / 0.10 |
| elite + behavior lambda=3 | 0.25 | 9.936 / 13.051 | 0.25 | 0.90 / 0.20 | 0.95 / 0.05 |
| elite + behavior lambda=10 | 0.10 | 3.665 / 6.582 | 0.25 | 0.80 / 0.30 | 1.00 / 0.05 |

Lambda 3 retained offline benefits but failed closed-loop transfer:

- proposal-regret reduction `0.950` and contact/jam action-L2 reduction `0.337`;
- old task action drift `01125=0.0359`, `00256=0.0269`, both within the `0.05` offline gate;
- `00186` relaxed decreased `0.40 -> 0.25`, missing both `+0.15` and `>=0.55` gates;
- `00186` jam fell only `0.30 -> 0.25`, while lateral/keypoint deteriorated sharply;
- old relaxed stayed above `0.90`, but old strict/process retention failed: `01125` process `0.30 -> 0.20`, `00256` process `0.10 -> 0.05`.

This falsifies the claim that the existing offline proposal-regret metric is sufficient for safe closed-loop policy-only adaptation. Do not promote any Phase 3.10 checkpoint.

Primary reports:

- `reports/phase3_three_task_pilot/phase3_10_policy_only_anchored_adaptation.md`
- `reports/phase3_three_task_pilot/phase3_10_policy_only_anchored_adaptation.json`
- `reports/phase3_three_task_pilot/phase3_10_closed_loop/<variant>/batch_eval_summary.json`

`max_force` remains `UNKNOWN_WITH_REASON`: `batch_eval_tasks.py` does not export force maxima, and no evaluation-trunk change was made.

## Inputs And Anchors

Comparable direct checkpoint, evaluation only:

```text
logs/isaaclab-srsa-assembly/1/srsa_axial_direct_finetune_from_01125/20260525_112528_asm-00186/models/best_step-600000_s-0p4133.pt
```

Key replay files:

```text
logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_replay_from_01125/20260615_202326_launcher/replay/01125.pt
logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_taskctx_repair_01125_00256/20260708_taskctx_repair_phase2_launcher/replay/00256.pt
logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_taskctx_repair_01125_00256_00186_rescue/20260713_phase3_2_rescue_00186_launcher/replay/00186.pt
```

## Next Conversation Starting Point

Do not continue policy-only training or add generic replay steps. The next diagnostic should explain why frozen-model elite ranking is highly predictive under the direct-reference offline metric but harmful in the actual environment. A narrow next step is to audit the temporal/return alignment between frozen MPPI elite targets and environment outcomes by phase (pre-contact, contact, insertion), including whether short-horizon elite actions improve local score while pushing later lateral/keypoint error outside recovery. Keep this diagnostic read-only and preserve the existing exact-template evaluation contract.

Before any new method proposal, read these in order:

1. `reports/phase3_three_task_pilot/phase3_10_policy_only_anchored_adaptation.md`
2. `reports/phase3_three_task_pilot/phase3_9_behavior_anchoring_audit.md`
3. `reports/phase3_three_task_pilot/phase3_8_policy_objective_audit.md`
4. `reports/phase3_three_task_pilot/phase3_3_planner_action_diagnosis.md`
5. `reports/phase3_three_task_pilot/phase3_2_acquisition_rescue_summary.md`

## Working Tree Note

At handoff time, the following audit scripts are untracked and should be reviewed/committed intentionally if they are to become project artifacts:

```text
tdmpc2/scripts/phase3_three_task_pilot/policy_old_task_behavior_anchoring_audit.py
tdmpc2/scripts/phase3_three_task_pilot/policy_only_anchored_adaptation_ablation.py
```
