#!/usr/bin/env python3
"""Phase 3.10 policy-only anchored adaptation and three-task closed-loop ablation.

The source rescue checkpoint is never changed.  Each adapted checkpoint retains
the complete source payload and changes only ``WorldModel._pi`` state tensors.
The frozen direct checkpoint is evaluation-only and never supplies a target.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_AUDIT_DIR = SCRIPT_DIR.parent / "model_task_sensitivity"
if str(MODEL_AUDIT_DIR) not in sys.path:
	sys.path.insert(0, str(MODEL_AUDIT_DIR))

import closed_loop_task_swap_eval as phase2_eval  # noqa: E402
import planner_action_attribution_diagnosis as attribution  # noqa: E402
import policy_conflict_mitigation_audit as mitigation  # noqa: E402
import policy_multistep_trainability_audit as phase37  # noqa: E402
import policy_old_task_behavior_anchoring_audit as phase39  # noqa: E402
import policy_prior_supervision_audit as prior_audit  # noqa: E402
import policy_proposal_quality_objective_audit as phase38  # noqa: E402
from _common import resolve, write_json, write_text  # noqa: E402


DEFAULT_PHASE32_DIAGNOSIS = (
	"reports/phase3_three_task_pilot/phase3_2_diagnosis/"
	"standalone_vs_multitask_diagnosis.json"
)
DEFAULT_ROLLOUT_ROOT = "reports/phase3_three_task_pilot/phase3_3_rollouts"
DEFAULT_REPORT_DIR = Path("reports/phase3_three_task_pilot")
DEFAULT_OUTPUT_JSON = DEFAULT_REPORT_DIR / "phase3_10_policy_only_anchored_adaptation.json"
DEFAULT_OUTPUT_MD = DEFAULT_REPORT_DIR / "phase3_10_policy_only_anchored_adaptation.md"
DEFAULT_CHECKPOINT_DIR = DEFAULT_REPORT_DIR / "phase3_10_policy_only_checkpoints"
DEFAULT_CLOSED_LOOP_ROOT = DEFAULT_REPORT_DIR / "phase3_10_closed_loop"
TASKS = ("01125", "00256", "00186")
OLD_TASKS = ("01125", "00256")


def _load_json(value: str | Path) -> dict[str, Any]:
	path = resolve(value)
	if not path.exists():
		raise FileNotFoundError(path)
	return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as handle:
		for block in iter(lambda: handle.read(1024 * 1024), b""):
			digest.update(block)
	return digest.hexdigest()


def _is_pi_key(key: str) -> bool:
	return key.removeprefix("module.").startswith("_pi.")


def _checkpoint_model_state(payload: Any) -> dict[str, torch.Tensor]:
	if not isinstance(payload, dict):
		raise TypeError("Expected checkpoint payload to be a dict.")
	state = payload.get("model", payload)
	if not isinstance(state, dict):
		raise TypeError("Checkpoint model state must be a dict.")
	return state


def _assert_checkpoint_delta(
	source_state: dict[str, torch.Tensor],
	adapted_state: dict[str, torch.Tensor],
	*,
	require_policy_change: bool,
) -> dict[str, int]:
	if tuple(source_state) != tuple(adapted_state):
		raise RuntimeError("Adapted checkpoint changed model-state key structure.")
	pi_tensors = 0
	changed_pi_tensors = 0
	non_pi_tensors = 0
	for key, source_value in source_state.items():
		adapted_value = adapted_state[key]
		if not torch.is_tensor(source_value) or not torch.is_tensor(adapted_value):
			if source_value != adapted_value:
				raise RuntimeError(f"Non-tensor checkpoint entry changed: {key}")
			continue
		if _is_pi_key(key):
			pi_tensors += 1
			if not torch.equal(source_value, adapted_value):
				changed_pi_tensors += 1
		elif not torch.equal(source_value, adapted_value):
			raise RuntimeError(f"Non-policy model tensor changed: {key}")
		else:
			non_pi_tensors += 1
	if pi_tensors == 0:
		raise RuntimeError("Source checkpoint has no policy-prior state tensors.")
	if require_policy_change and changed_pi_tensors == 0:
		raise RuntimeError("Policy-only adaptation did not change any _pi tensor.")
	return {
		"pi_tensors": pi_tensors,
		"changed_pi_tensors": changed_pi_tensors,
		"unchanged_non_pi_tensors": non_pi_tensors,
	}


def _write_policy_checkpoint(
	source_payload: dict[str, Any],
	source_state: dict[str, torch.Tensor],
	model,
	output_path: Path,
) -> dict[str, Any]:
	"""Save source payload with only the model _pi tensors replaced."""
	model_state = model.state_dict()
	payload = copy.deepcopy(source_payload)
	adapted_state = _checkpoint_model_state(payload)
	for source_key in source_state:
		if not _is_pi_key(source_key):
			continue
		model_key = source_key.removeprefix("module.")
		if model_key not in model_state:
			raise KeyError(f"Adapted model has no checkpoint policy key {model_key}.")
		if tuple(source_state[source_key].shape) != tuple(model_state[model_key].shape):
			raise RuntimeError(
			f"Policy tensor shape mismatch for {source_key}: "
			f"source={tuple(source_state[source_key].shape)} adapted={tuple(model_state[model_key].shape)}."
		)
		adapted_state[source_key] = model_state[model_key].detach().cpu().clone()
	check = _assert_checkpoint_delta(source_state, adapted_state, require_policy_change=True)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	torch.save(payload, output_path)
	reloaded = torch.load(output_path, map_location="cpu", weights_only=False)
	reload_check = _assert_checkpoint_delta(
		source_state,
		_checkpoint_model_state(reloaded),
		require_policy_change=True,
	)
	if check != reload_check:
		raise RuntimeError("Checkpoint validation changed after round-trip serialization.")
	return {
		"path": str(output_path),
		"sha256": _sha256(output_path),
		"non_policy_state_exact": True,
		"validation": check,
	}


def _policy_update_variant(
	base_model,
	base_cfg,
	new_rows: dict[str, torch.Tensor],
	old_rows: OrderedDict[str, dict[str, torch.Tensor]],
	*,
	anchor_lambda: float,
	args: argparse.Namespace,
	device: torch.device,
	seed_offset: int,
):
	model = copy.deepcopy(base_model)
	phase37._freeze_world_model_except_policy(model)
	policy_lr = float(args.policy_lr) if args.policy_lr is not None else float(base_cfg.lr)
	optimizer = torch.optim.Adam(phase37._policy_parameters(model), lr=policy_lr)
	base_policy_device = phase39._flat_policy_device(model).detach().clone()
	new_generator = torch.Generator().manual_seed(int(args.seed) + seed_offset)
	old_generators = OrderedDict((
		task_id,
		torch.Generator().manual_seed(int(args.seed) + seed_offset + 1_000 + task_index),
	) for task_index, task_id in enumerate(OLD_TASKS))
	final_losses: dict[str, float] = {}
	for update in range(1, int(args.updates) + 1):
		new_batch = phase38._sample_batch(new_rows, int(args.new_batch_size), new_generator)
		optimizer.zero_grad(set_to_none=True)
		new_loss = phase39._new_loss(model, new_batch, "elite", device, int(args.seed) + seed_offset + update)
		if anchor_lambda > 0.0:
			old_batches = OrderedDict((
				task_id,
				phase38._sample_batch(old_rows[task_id], int(args.old_anchor_batch_size), old_generators[task_id]),
			) for task_id in OLD_TASKS)
			anchor_loss = phase39._behavior_anchor_loss(model, base_model, old_batches, device)
		else:
			anchor_loss = torch.zeros((), device=device)
		loss = new_loss + float(anchor_lambda) * anchor_loss
		loss.backward()
		optimizer.step()
		phase38._project_policy_delta(model, base_policy_device, float(args.policy_delta_budget))
		final_losses = {
			"new_elite_loss": float(new_loss.detach().item()),
			"old_behavior_anchor_loss": float(anchor_loss.detach().item()),
			"combined_loss": float(loss.detach().item()),
		}
	return model, {
		"updates": int(args.updates),
		"anchor_lambda": float(anchor_lambda),
		"policy_lr": policy_lr,
		"policy_parameter_delta_l2": float((phase39._flat_policy_device(model) - base_policy_device).norm().item()),
		"final_losses": final_losses,
	}


def _offline_metrics(
	model,
	base_model,
	base_cfg,
	direct_model,
	direct_cfg,
	objective_rows: dict[str, dict[str, torch.Tensor]],
	eval_rows: dict[str, torch.Tensor],
	eval_elite: dict[str, torch.Tensor],
	old_val_rows: OrderedDict[str, dict[str, torch.Tensor]],
	base_policy_device: torch.Tensor,
	*,
	device: torch.device,
	args: argparse.Namespace,
):
	stage = phase39._stage_metrics(
		model,
		base_model,
		base_model,
		base_cfg,
		direct_model,
		direct_cfg,
		objective_rows,
		"elite",
		eval_rows,
		eval_elite,
		old_val_rows,
		base_policy_device,
		device,
		args,
		0,
	)
	return stage


def _closed_loop_command(args: argparse.Namespace, checkpoint: Path, label: str, output_dir: Path) -> list[str]:
	"""Reuse the established Phase 2.2 exact-template batch-eval contract."""
	summary_fp = output_dir / "batch_eval_summary.json"
	eval_args = copy.copy(args)
	eval_args.checkpoint = str(checkpoint)
	overrides = phase2_eval._base_eval_overrides(
		eval_args,
		assembly_id="00186",
		label=label,
		output_dir=output_dir,
		summary_fp=summary_fp,
	)
	overrides = [
		value for value in overrides
		if not value.startswith(("checkpoint=", "eval_assembly_ids=", "batch_eval_spawn_per_assembly=", "run_id=", "exp_name="))
	]
	overrides.extend((
		f"checkpoint={checkpoint}",
		"eval_assembly_ids=[01125,00256,00186]",
		"batch_eval_spawn_per_assembly=true",
		f"batch_eval_episodes_per_task={int(args.episodes)}",
		f"batch_eval_output_dir={output_dir}",
		f"batch_eval_summary_fp={summary_fp}",
		"batch_eval_overwrite=true",
		"enable_wandb=false",
		"exp_name=srsa_phase3_10_policy_only_adaptation",
		f"run_id=phase3_10_{label}",
	))
	return [str(resolve(args.python)), "tdmpc2/batch_eval_tasks.py", *overrides]


def _metric(item: dict[str, Any], *keys: str, default: float = 0.0) -> float:
	for key in keys:
		for candidate in (key, f"episode_{key}"):
			if item.get(candidate) is not None:
				return float(item[candidate])
	return float(default)


def _closed_loop_metrics(item: dict[str, Any]) -> dict[str, Any]:
	max_force = _metric(item, "max_force", "flange_force_norm", "flange_force_max", "contact_force_max", default=float("nan"))
	return {
		"relaxed_success": _metric(item, "relaxed_success_stable", "relaxed_success_episode", "success"),
		"strict_success": _metric(item, "strict_success_stable", "strict_success_episode"),
		"process_success": _metric(item, "process_success_terminal", "process_success"),
		"reward": _metric(item, "episode_reward"),
		"lateral_error_mm": 1000.0 * _metric(item, "lateral_error"),
		"keypoint_error_mm": 1000.0 * _metric(item, "keypoint_error"),
		"jamming_rate": _metric(item, "jam"),
		"max_force": max_force if torch.isfinite(torch.tensor(max_force)) else {
			"status": "UNKNOWN_WITH_REASON",
			"reason": "batch_eval_tasks.py does not export a force maximum in eval_metrics.json.",
		},
		"episode_length": _metric(item, "episode_length"),
		"raw_metrics": item,
	}


def _run_closed_loop(args: argparse.Namespace, checkpoint: Path, label: str) -> dict[str, Any]:
	output_dir = resolve(args.closed_loop_root) / label
	summary_fp = output_dir / "batch_eval_summary.json"
	command = _closed_loop_command(args, checkpoint, label, output_dir)
	if args.skip_closed_loop:
		return {
			"status": "SKIPPED",
			"reason": "--skip-closed-loop was requested.",
			"command": command,
			"output_dir": str(output_dir),
		}
	if args.reuse_existing and summary_fp.exists():
		pass
	else:
		output_dir.mkdir(parents=True, exist_ok=True)
		log_fp = output_dir / "batch_eval.log"
		start = time.monotonic()
		try:
			with log_fp.open("w", encoding="utf-8") as log_handle:
				result = subprocess.run(
					command,
					cwd=str(Path(__file__).resolve().parents[3]),
					stdout=log_handle,
					stderr=subprocess.STDOUT,
					text=True,
					check=False,
					timeout=float(args.subprocess_timeout),
				)
		except subprocess.TimeoutExpired:
			return {
				"status": "FAILED_TIMEOUT",
				"command": command,
				"output_dir": str(output_dir),
				"log_fp": str(log_fp),
				"elapsed_sec": time.monotonic() - start,
			}
		if result.returncode != 0:
			return {
				"status": "FAILED",
				"returncode": int(result.returncode),
				"command": command,
				"output_dir": str(output_dir),
				"log_fp": str(log_fp),
				"elapsed_sec": time.monotonic() - start,
			}
	if not summary_fp.exists():
		return {
			"status": "FAILED",
			"error": "batch_eval_summary.json is missing after evaluation.",
			"command": command,
			"output_dir": str(output_dir),
		}
	summary = _load_json(summary_fp)
	tasks = {str(item.get("assembly_id")).zfill(5): _closed_loop_metrics(item) for item in summary.get("tasks") or []}
	missing = set(TASKS) - set(tasks)
	if missing:
		return {
			"status": "FAILED",
			"error": f"Batch eval omitted tasks: {sorted(missing)}",
			"command": command,
			"output_dir": str(output_dir),
			"summary_fp": str(summary_fp),
		}
	return {
		"status": "DONE",
		"command": command,
		"output_dir": str(output_dir),
		"summary_fp": str(summary_fp),
		"tasks": tasks,
	}


def _closed_loop_deltas(tasks: dict[str, dict[str, Any]], baseline: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
	keys = ("relaxed_success", "strict_success", "process_success", "reward", "lateral_error_mm", "keypoint_error_mm", "jamming_rate")
	return {
		task_id: {key: float(tasks[task_id][key]) - float(baseline[task_id][key]) for key in keys}
		for task_id in TASKS
	}


def _classify(report: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
	closed_loop = report["closed_loop"]
	base = closed_loop["original_checkpoint"]
	anchored = closed_loop["elite_behavior_anchor_l3"]
	if base.get("status") != "DONE" or anchored.get("status") != "DONE":
		return {
			"classification": "MIXED_RESULT",
			"reason": "The required original or lambda=3 closed-loop evaluation did not complete; inspect the saved Isaac logs.",
		}
	base_tasks = base["tasks"]
	anchor_tasks = anchored["tasks"]
	old_relaxed = all(float(anchor_tasks[task]["relaxed_success"]) >= float(args.old_relaxed_min) for task in OLD_TASKS)
	old_quality = all(
		float(anchor_tasks[task]["strict_success"]) >= float(base_tasks[task]["strict_success"]) - float(args.old_quality_drop_tol)
		and float(anchor_tasks[task]["process_success"]) >= float(base_tasks[task]["process_success"]) - float(args.old_quality_drop_tol)
		for task in OLD_TASKS
	)
	offline = report["variants"]["elite_behavior_anchor_l3"]["offline"]
	drift_ok = max(float(value["mean"]) for value in offline["old_task_action_drift"].values()) <= float(args.max_old_action_drift)
	new_base = base_tasks["00186"]
	new_anchor = anchor_tasks["00186"]
	new_relaxed = (
		float(new_anchor["relaxed_success"]) >= float(new_base["relaxed_success"]) + float(args.new_relaxed_gain)
		or float(new_anchor["relaxed_success"]) >= float(args.new_relaxed_absolute)
	)
	quality_improved = (
		float(new_anchor["jamming_rate"]) <= float(new_base["jamming_rate"]) - float(args.jam_improvement)
		or float(new_anchor["lateral_error_mm"]) <= float(new_base["lateral_error_mm"]) - float(args.lateral_improvement_mm)
		or float(new_anchor["keypoint_error_mm"]) <= float(new_base["keypoint_error_mm"]) - float(args.keypoint_improvement_mm)
	)
	offline_gain = (
		float(offline["reductions"]["proposal_regret_reduction_fraction"]) >= float(args.min_proposal_regret_improvement)
		and float(offline["reductions"]["contact_jam_action_l2_reduction_fraction"]) > 0.0
	)
	if new_relaxed and quality_improved and old_relaxed and old_quality and drift_ok:
		return {
			"classification": "ANCHORED_POLICY_ADAPTATION_CLOSED_LOOP_PASS",
			"reason": "Lambda=3 converted offline proposal-quality gains into the required 00186 improvement while preserving both old-task behavior and closed-loop quality.",
			"gates": {"new_relaxed": new_relaxed, "new_quality": quality_improved, "old_relaxed": old_relaxed, "old_quality": old_quality, "old_drift": drift_ok, "offline_gain": offline_gain},
		}
	if not old_relaxed or not old_quality or not drift_ok:
		return {
			"classification": "RETENTION_FAILURE",
			"reason": "The anchored variant did not preserve required old-task behavior or closed-loop retention.",
			"gates": {"new_relaxed": new_relaxed, "new_quality": quality_improved, "old_relaxed": old_relaxed, "old_quality": old_quality, "old_drift": drift_ok, "offline_gain": offline_gain},
		}
	if offline_gain and (not new_relaxed or not quality_improved):
		return {
			"classification": "OFFLINE_GAIN_NOT_TRANSFERRED",
			"reason": "The anchored clone kept old-task behavior and improved offline proposal quality, but that gain did not meet the 00186 closed-loop acquisition/quality gate.",
			"gates": {"new_relaxed": new_relaxed, "new_quality": quality_improved, "old_relaxed": old_relaxed, "old_quality": old_quality, "old_drift": drift_ok, "offline_gain": offline_gain},
		}
	return {
		"classification": "MIXED_RESULT",
		"reason": "The lambda=3 clone produced an incomplete or internally mixed offline/closed-loop signal.",
		"gates": {"new_relaxed": new_relaxed, "new_quality": quality_improved, "old_relaxed": old_relaxed, "old_quality": old_quality, "old_drift": drift_ok, "offline_gain": offline_gain},
	}


def _markdown(report: dict[str, Any]) -> str:
	classification = report["classification"]
	lines = [
		"# SRSA Phase 3.10 Policy-Only Anchored Adaptation Ablation",
		"",
		"本报告只更新独立 checkpoint 的 `WorldModel._pi`；encoder、dynamics、reward、Q、task_context、MPPI 和 replay sampler 均冻结。direct checkpoint 仅用于离线评估参考，从未参与训练 target。",
		"",
		f"Status: `{report['status']}`",
		f"Final classification: `{classification['classification']}`",
		"",
		"## Conclusion",
		"",
		classification["reason"],
		"",
		"## Policy-Only Checkpoints",
		"",
		"| Variant | Anchor lambda | Updates | Policy delta L2 | Proposal-regret reduction | Contact/jam L2 reduction | 01125 drift | 00256 drift | Checkpoint |",
		"| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
	]
	for name, item in report["variants"].items():
		offline = item["offline"]
		checkpoint = item.get("checkpoint", {}).get("path", report["inputs"]["source_checkpoint"])
		lines.append(
			f"| `{name}` | {item['training']['anchor_lambda']:g} | {item['training']['updates']} | "
			f"{item['training']['policy_parameter_delta_l2']:.4f} | "
			f"{offline['reductions']['proposal_regret_reduction_fraction']:+.3f} | "
			f"{offline['reductions']['contact_jam_action_l2_reduction_fraction']:+.3f} | "
			f"{offline['old_task_action_drift']['01125']['mean']:.4f} | "
			f"{offline['old_task_action_drift']['00256']['mean']:.4f} | `{checkpoint}` |"
		)
	lines.extend((
		"",
		"## Three-Task Closed Loop",
		"",
		"| Variant | Task | Relaxed | Strict | Process | Reward | Lateral mm | Keypoint mm | Jam | Episode len |",
		"| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
	))
	for name, item in report["closed_loop"].items():
		if item.get("status") != "DONE":
			lines.append(f"| `{name}` | all | NA | NA | NA | NA | NA | NA | NA | NA |")
			continue
		for task_id in TASKS:
			metrics = item["tasks"][task_id]
			lines.append(
				f"| `{name}` | `{task_id}` | {metrics['relaxed_success']:.3f} | {metrics['strict_success']:.3f} | "
				f"{metrics['process_success']:.3f} | {metrics['reward']:.2f} | {metrics['lateral_error_mm']:.3f} | "
				f"{metrics['keypoint_error_mm']:.3f} | {metrics['jamming_rate']:.3f} | {metrics['episode_length']:.1f} |"
			)
	lines.extend((
		"",
		"## Safety Checks",
		"",
		f"- Source checkpoint SHA256 before/after: `{report['source_checkpoint_unchanged']}`.",
		"- Each adapted checkpoint passed an exact tensor check: all non-`_pi` model tensors equal the source checkpoint.",
		"- Elite targets come from the frozen multitask world model; old-task anchors use the frozen source policy on old-task state/task-vector pairs.",
		"- Closed-loop `max_force` remains `UNKNOWN_WITH_REASON` because the shared batch evaluator does not export force maxima; no eval-trunk change was made.",
	))
	gates = classification.get("gates")
	if gates:
		lines.extend((
			"",
			"## Lambda=3 Gate",
			"",
			f"- `00186 relaxed acquisition`: `{'PASS' if gates['new_relaxed'] else 'FAIL'}`.",
			f"- `00186 jam/lateral/keypoint quality`: `{'PASS' if gates['new_quality'] else 'FAIL'}`.",
			f"- `01125/00256 relaxed retention`: `{'PASS' if gates['old_relaxed'] else 'FAIL'}`.",
			f"- `01125/00256 strict/process retention`: `{'PASS' if gates['old_quality'] else 'FAIL'}`.",
			f"- `01125/00256 policy action drift <= threshold`: `{'PASS' if gates['old_drift'] else 'FAIL'}`.",
			f"- `offline proposal/action improvement`: `{'PASS' if gates['offline_gain'] else 'FAIL'}`.",
		))
	return "\n".join(lines) + "\n"


def build_report(args: argparse.Namespace) -> dict[str, Any]:
	diagnosis = _load_json(args.phase32_diagnosis)
	if diagnosis.get("status") != "PASS":
		raise RuntimeError(f"Unexpected Phase 3.2 diagnosis status: {diagnosis.get('status')}")
	checkpoints = diagnosis.get("checkpoints") or {}
	source_checkpoint = resolve(checkpoints.get("multitask_rescue_best", ""))
	direct_checkpoint = resolve(checkpoints.get("direct_finetune", ""))
	replay_paths = OrderedDict(prior_audit.DEFAULT_REPLAYS)
	replay_paths["00186"] = resolve(diagnosis.get("replay", ""))
	replay_paths = OrderedDict((task_id, resolve(path)) for task_id, path in replay_paths.items())
	for path in (source_checkpoint, direct_checkpoint, *replay_paths.values(), resolve(args.rollout_root)):
		if not path.exists():
			raise FileNotFoundError(path)

	device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() and not args.cpu else "cpu")
	if device.type == "cuda":
		torch.cuda.set_device(device)
	source_sha_before = _sha256(source_checkpoint)
	source_payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
	source_state = _checkpoint_model_state(source_payload)
	base_model, cfg, multitask_compat = attribution._load_model(source_checkpoint, args, device)
	direct_model, direct_cfg, direct_compat = attribution._load_model(direct_checkpoint, args, device)
	base_policy_cpu = phase37._flatten_policy(base_model).clone()
	base_policy_device = phase39._flat_policy_device(base_model).detach().clone()

	new_train_all, new_val_all, _ = phase38._replay_rows(replay_paths["00186"], int(args.seed) + 1)
	new_train = phase38._rows_fixed(new_train_all, int(args.new_replay_train_rows), int(args.seed) + 10)
	new_val = phase38._rows_fixed(new_val_all, int(args.new_replay_val_rows), int(args.seed) + 11)
	new_train = phase38._attach_cache(
		new_train,
		phase38._build_elite_cache(base_model, cfg, new_train, device, args, int(args.seed) + 500),
	)
	new_val = phase38._attach_cache(
		new_val,
		phase38._build_elite_cache(base_model, cfg, new_val, device, args, int(args.seed) + 600),
	)
	quality_train_all, quality_val_all = phase38._quality_rollout_rows(args)
	contact_jam_idx = torch.nonzero((quality_val_all["phase"] == 1) & (quality_val_all["outcome"] == 1), as_tuple=False).reshape(-1)
	if contact_jam_idx.numel() == 0:
		raise RuntimeError("Phase 3.3 held-out rollout pool has no contact/jam states.")
	eval_rows = phase38._rows_fixed(
		phase38._rows_select(quality_val_all, contact_jam_idx),
		int(args.eval_contact_jam_rows),
		int(args.seed) + 900,
	)
	eval_elite = phase38._build_elite_cache(base_model, cfg, eval_rows, device, args, int(args.seed) + 1000)

	old_train_rows: OrderedDict[str, dict[str, torch.Tensor]] = OrderedDict()
	old_val_rows: OrderedDict[str, dict[str, torch.Tensor]] = OrderedDict()
	for task_index, task_id in enumerate(OLD_TASKS):
		train_all, val_all, _ = phase38._replay_rows(replay_paths[task_id], int(args.seed) + 97 * task_index)
		old_train_rows[task_id] = phase38._rows_fixed(train_all, int(args.old_anchor_train_rows), int(args.seed) + 1100 + task_index)
		old_val_rows[task_id] = phase38._rows_fixed(val_all, int(args.old_task_val_rows), int(args.seed) + 1200 + task_index)
	objective_rows = {"train": new_train, "val": new_val, "replay_val": new_val}

	baseline_offline = _offline_metrics(
		base_model, base_model, cfg, direct_model, direct_cfg, objective_rows, eval_rows, eval_elite,
		old_val_rows, base_policy_device, device=device, args=args,
	)
	baseline_offline["reductions"] = {"contact_jam_action_l2_reduction_fraction": 0.0, "proposal_regret_reduction_fraction": 0.0}
	variants: OrderedDict[str, dict[str, Any]] = OrderedDict()
	variants["original_checkpoint"] = {
		"training": {"updates": 0, "anchor_lambda": 0.0, "policy_parameter_delta_l2": 0.0, "final_losses": {}},
		"offline": baseline_offline,
		"checkpoint": {"path": str(source_checkpoint), "sha256": source_sha_before, "source_checkpoint": True},
	}
	variant_specs = (
		("elite_only", 0.0),
		("elite_behavior_anchor_l3", 3.0),
		("elite_behavior_anchor_l10", 10.0),
	)
	for variant_index, (name, anchor_lambda) in enumerate(variant_specs):
		model, training = _policy_update_variant(
			base_model,
			cfg,
			new_train,
			old_train_rows,
			anchor_lambda=anchor_lambda,
			args=args,
			device=device,
			seed_offset=10_000 + 10_000 * variant_index,
		)
		offline = _offline_metrics(
			model, base_model, cfg, direct_model, direct_cfg, objective_rows, eval_rows, eval_elite,
			old_val_rows, base_policy_device, device=device, args=args,
		)
		offline["reductions"] = phase39._reductions(offline, baseline_offline)
		checkpoint = _write_policy_checkpoint(
			source_payload,
			source_state,
			model,
			resolve(args.checkpoint_dir) / f"{name}_{int(args.updates)}.pt",
		)
		# A loader round trip guards the external batch-eval contract, not just tensor equality.
		loaded_model, _, _ = attribution._load_model(Path(checkpoint["path"]), args, device)
		if not torch.equal(phase37._flatten_policy(model), phase37._flatten_policy(loaded_model)):
			raise RuntimeError(f"Saved policy checkpoint did not reload exactly: {name}")
		variants[name] = {"training": training, "offline": offline, "checkpoint": checkpoint}
		del model, loaded_model
		if device.type == "cuda":
			torch.cuda.empty_cache()

	if not torch.equal(base_policy_cpu, phase37._flatten_policy(base_model)):
		raise RuntimeError("Source in-memory policy changed during Phase 3.10 adaptation.")
	if source_sha_before != _sha256(source_checkpoint):
		raise RuntimeError("Source checkpoint bytes changed during Phase 3.10 adaptation.")
	closed_loop: OrderedDict[str, dict[str, Any]] = OrderedDict()
	for name, item in variants.items():
		closed_loop[name] = _run_closed_loop(args, Path(item["checkpoint"]["path"]), name)
	if closed_loop.get("original_checkpoint", {}).get("status") == "DONE":
		base_tasks = closed_loop["original_checkpoint"]["tasks"]
		for name, item in closed_loop.items():
			if name != "original_checkpoint" and item.get("status") == "DONE":
				item["delta_vs_original"] = _closed_loop_deltas(item["tasks"], base_tasks)
	report = {
		"status": "PENDING",
		"inputs": {
			"source_checkpoint": str(source_checkpoint),
			"direct_checkpoint_evaluation_only": str(direct_checkpoint),
			"replays": {task_id: str(path) for task_id, path in replay_paths.items()},
			"rollout_root": str(resolve(args.rollout_root)),
		},
		"device": str(device),
		"checkpoint_compatibility": {"multitask": multitask_compat, "direct": direct_compat},
		"source_checkpoint_sha256": source_sha_before,
		"source_checkpoint_unchanged": True,
		"frozen_modules": ["_encoder", "_dynamics", "_reward", "_Qs", "_task_encoder", "_task_context_adapters"],
		"policy_only": True,
		"elite_target": {
			"scorer": "frozen_multitask_world_model",
			"direct_checkpoint_used_for_targets": False,
			"num_candidates": int(args.num_candidates),
			"num_policy_candidates": int(args.num_policy_candidates),
			"num_elites": int(args.num_elites),
			"horizon": int(args.horizon),
		},
		"old_task_anchor": {"reference": "frozen_pre_update_source_policy", "tasks": list(OLD_TASKS), "lambda_recommended": 3.0},
		"variants": variants,
		"closed_loop": closed_loop,
		"gates": {
			"new_relaxed_gain": float(args.new_relaxed_gain),
			"new_relaxed_absolute": float(args.new_relaxed_absolute),
			"old_relaxed_min": float(args.old_relaxed_min),
			"old_quality_drop_tol": float(args.old_quality_drop_tol),
			"max_old_action_drift": float(args.max_old_action_drift),
			"min_proposal_regret_improvement": float(args.min_proposal_regret_improvement),
		},
		"limitations": [
			"Closed-loop max_force is explicit UNKNOWN_WITH_REASON because batch_eval_tasks.py does not export force maxima.",
			"This is an independent policy-only checkpoint ablation, not a TD-MPC2 joint training run or a consolidation decision.",
		],
	}
	report["classification"] = _classify(report, args)
	report["status"] = "PASS" if report["classification"]["classification"] == "ANCHORED_POLICY_ADAPTATION_CLOSED_LOOP_PASS" else "WARNING"
	return report


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--phase32-diagnosis", default=DEFAULT_PHASE32_DIAGNOSIS)
	parser.add_argument("--rollout-root", default=DEFAULT_ROLLOUT_ROOT)
	parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
	parser.add_argument("--closed-loop-root", default=str(DEFAULT_CLOSED_LOOP_ROOT))
	parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
	parser.add_argument("--output-md", default=str(DEFAULT_OUTPUT_MD))
	parser.add_argument("--python", default=phase2_eval.DEFAULT_PYTHON)
	parser.add_argument("--config-dir", default=phase2_eval.DEFAULT_CONFIG_DIR)
	parser.add_argument("--config-name", default=phase2_eval.DEFAULT_CONFIG_NAME)
	parser.add_argument("--config", default="configs/train/srsa_01125_imitation_relaxed.yaml")
	parser.add_argument("--isaaclab-dir", default="/home/gpuserver/IsaacLab")
	parser.add_argument("--srsa-dir", default="/home/gpuserver/hx/github/srsa")
	parser.add_argument("--gpu-id", type=int, default=0)
	parser.add_argument("--cpu", action="store_true")
	parser.add_argument("--num-envs", type=int, default=256)
	parser.add_argument("--episodes", type=int, default=20)
	parser.add_argument("--subprocess-timeout", type=float, default=1200.0)
	parser.add_argument("--reuse-existing", action="store_true")
	parser.add_argument("--skip-closed-loop", action="store_true")
	parser.add_argument("--batch-size", type=int, default=256, help="Checkpoint config-loader batch size.")
	parser.add_argument("--updates", type=int, default=100)
	parser.add_argument("--new-batch-size", type=int, default=128)
	parser.add_argument("--old-anchor-batch-size", type=int, default=128)
	parser.add_argument("--eval-batch-size", type=int, default=1024)
	parser.add_argument("--elite-batch-size", type=int, default=64)
	parser.add_argument("--proposal-batch-size", type=int, default=32)
	parser.add_argument("--new-replay-train-rows", type=int, default=4096)
	parser.add_argument("--new-replay-val-rows", type=int, default=2048)
	parser.add_argument("--old-anchor-train-rows", type=int, default=4096)
	parser.add_argument("--old-task-val-rows", type=int, default=2048)
	parser.add_argument("--eval-contact-jam-rows", type=int, default=256)
	parser.add_argument("--policy-lr", type=float, default=None)
	parser.add_argument("--policy-delta-budget", type=float, default=3.5)
	parser.add_argument("--num-candidates", type=int, default=64)
	parser.add_argument("--num-policy-candidates", type=int, default=3)
	parser.add_argument("--num-elites", type=int, default=8)
	parser.add_argument("--horizon", type=int, default=3)
	parser.add_argument("--jam-lateral-threshold", type=float, default=0.008)
	parser.add_argument("--jam-keypoint-threshold", type=float, default=0.012)
	parser.add_argument("--jam-force-excursion-threshold", type=float, default=2.0)
	parser.add_argument("--new-relaxed-gain", type=float, default=0.15)
	parser.add_argument("--new-relaxed-absolute", type=float, default=0.55)
	parser.add_argument("--jam-improvement", type=float, default=0.05)
	parser.add_argument("--lateral-improvement-mm", type=float, default=0.10)
	parser.add_argument("--keypoint-improvement-mm", type=float, default=0.25)
	parser.add_argument("--old-relaxed-min", type=float, default=0.90)
	parser.add_argument("--old-quality-drop-tol", type=float, default=0.05)
	parser.add_argument("--max-old-action-drift", type=float, default=0.05)
	parser.add_argument("--min-proposal-regret-improvement", type=float, default=0.20)
	parser.add_argument("--seed", type=int, default=1)
	parser.add_argument("--dry-run", action="store_true")
	args = parser.parse_args()
	if args.updates <= 0 or args.new_batch_size <= 0 or args.old_anchor_batch_size <= 0:
		raise ValueError("updates and policy batch sizes must be positive.")
	if args.num_elites <= 0 or args.num_elites > args.num_candidates:
		raise ValueError("num-elites must be in [1, num-candidates].")
	if args.num_policy_candidates <= 0 or args.num_policy_candidates >= args.num_candidates:
		raise ValueError("num-policy-candidates must be in [1, num-candidates - 1].")
	if args.dry_run:
		print(
			"PASS dry-run: would create independent policy-only checkpoints under "
			f"{resolve(args.checkpoint_dir)} and write {resolve(args.output_json)} / {resolve(args.output_md)}."
		)
		return 0
	report = build_report(args)
	write_json(report, args.output_json)
	write_text(_markdown(report), args.output_md)
	print(report["status"])
	print(f"Final classification: {report['classification']['classification']}")
	print(f"Source checkpoint unchanged: {report['source_checkpoint_unchanged']}")
	return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
	raise SystemExit(main())
