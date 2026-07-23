#!/usr/bin/env python3
"""Read-only Phase 3.9 audit of old-task behavior anchors for policy adaptation."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_AUDIT_DIR = SCRIPT_DIR.parent / "model_task_sensitivity"
if str(MODEL_AUDIT_DIR) not in sys.path:
	sys.path.insert(0, str(MODEL_AUDIT_DIR))

import planner_action_attribution_diagnosis as attribution  # noqa: E402
import policy_conflict_mitigation_audit as mitigation  # noqa: E402
import policy_multistep_trainability_audit as phase37  # noqa: E402
import policy_prior_supervision_audit as prior_audit  # noqa: E402
import policy_proposal_quality_objective_audit as phase38  # noqa: E402
from _common import resolve, td_math, write_json, write_text  # noqa: E402


DEFAULT_PHASE32_DIAGNOSIS = (
	"reports/phase3_three_task_pilot/phase3_2_diagnosis/"
	"standalone_vs_multitask_diagnosis.json"
)
DEFAULT_ROLLOUT_ROOT = "reports/phase3_three_task_pilot/phase3_3_rollouts"
DEFAULT_OUTPUT_JSON = "reports/phase3_three_task_pilot/phase3_9_behavior_anchoring_audit.json"
DEFAULT_OUTPUT_MD = "reports/phase3_three_task_pilot/phase3_9_behavior_anchoring_audit.md"


def _load_json(path_value: str | Path) -> dict[str, Any]:
	path = resolve(path_value)
	if not path.exists():
		raise FileNotFoundError(path)
	return json.loads(path.read_text(encoding="utf-8"))


def _parse_lambdas(value: str) -> tuple[float, ...]:
	result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
	if not result or any(item < 0.0 for item in result):
		raise ValueError("anchor-lambdas must contain one or more non-negative floats.")
	return result


def _policy_mean_with_grad(model, obs: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
	z = model.encode(obs, task)
	_, info = model.pi(z, task)
	return info["mean"]


@torch.no_grad()
def _reference_policy_mean(reference_model, obs: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
	z = reference_model.encode(obs, task)
	_, info = reference_model.pi(z, task)
	return info["mean"]


def _behavior_anchor_loss(model, reference_model, old_batches: dict[str, dict[str, torch.Tensor]], device: torch.device) -> torch.Tensor:
	"""Anchor output behavior on recorded old-task state/task-vector pairs only."""
	values = []
	for batch in old_batches.values():
		obs = batch["obs"].to(device)
		task = batch["task"].to(device)
		current = _policy_mean_with_grad(model, obs, task)
		with torch.no_grad():
			reference = _reference_policy_mean(reference_model, obs, task)
		values.append(torch.nn.functional.mse_loss(current, reference))
	return torch.stack(values).mean()


def _parameter_l2_anchor(model, base_policy_device: torch.Tensor) -> torch.Tensor:
	return (_flat_policy_device(model) - base_policy_device).pow(2).sum()


def _flat_policy_device(model) -> torch.Tensor:
	return torch.cat([parameter.reshape(-1) for parameter in model._pi.parameters()])


def _old_anchor_loss_eval(model, reference_model, rows: OrderedDict[str, dict[str, torch.Tensor]], device: torch.device, batch_size: int) -> float:
	values = []
	for task_index, data in enumerate(rows.values()):
		for start in range(0, int(data["obs"].shape[0]), int(batch_size)):
			stop = min(start + int(batch_size), int(data["obs"].shape[0]))
			obs = data["obs"][start:stop].to(device)
			task = data["task"][start:stop].to(device)
			with torch.no_grad():
				current = _reference_policy_mean(model, obs, task)
				reference = _reference_policy_mean(reference_model, obs, task)
				values.append(float(torch.nn.functional.mse_loss(current, reference).item()))
	return float(sum(values) / len(values))


def _standard_bc_loss(model, rows: dict[str, torch.Tensor], device: torch.device, args, seed: int) -> float:
	values = []
	for start in range(0, int(rows["obs"].shape[0]), int(args.eval_batch_size)):
		stop = min(start + int(args.eval_batch_size), int(rows["obs"].shape[0]))
		batch = {key: value[start:stop] for key, value in rows.items()}
		with torch.no_grad():
			per_row = phase38._per_row_bc(model, batch, batch["action"], device, int(seed) + start)
			values.append(float(per_row.mean().item()))
	return float(sum(values) / len(values))


def _new_objective_name(family: str) -> str:
	if family in ("elite", "elite_behavior", "elite_parameter"):
		return "mppi_elite_distillation"
	if family == "quality_behavior":
		return "contact_jam_quality_weighted_bc"
	raise ValueError(family)


def _make_specs(lambdas: tuple[float, ...]) -> tuple[dict[str, Any], ...]:
	specs = [{"name": "elite_distillation", "family": "elite", "lambda": 0.0, "anchor": "none"}]
	for value in lambdas:
		label = f"{value:g}".replace(".", "p")
		specs.extend((
			{"name": f"elite_behavior_anchor_l{label}", "family": "elite_behavior", "lambda": value, "anchor": "behavior"},
			{"name": f"quality_behavior_anchor_l{label}", "family": "quality_behavior", "lambda": value, "anchor": "behavior"},
			{"name": f"elite_parameter_anchor_l{label}", "family": "elite_parameter", "lambda": value, "anchor": "parameter"},
		))
	return tuple(specs)


def _new_loss(model, batch: dict[str, torch.Tensor], family: str, device: torch.device, seed: int) -> torch.Tensor:
	return phase38._objective_loss(model, batch, _new_objective_name(family), device, seed)


def _stage_metrics(
	model,
	reference_model,
	base_model,
	multitask_cfg,
	direct_model,
	direct_cfg,
	objective_rows,
	family: str,
	eval_rows,
	eval_elite,
	old_val_rows,
	base_policy_device: torch.Tensor,
	device: torch.device,
	args,
	updates: int,
):
	objective = _new_objective_name(family)
	train_loss = phase38._evaluate_objective(model, objective_rows["train"], objective, device, args, int(args.seed) + 100)
	val_loss = phase38._evaluate_objective(model, objective_rows["val"], objective, device, args, int(args.seed) + 200)
	state_bank = {"obs": eval_rows["obs"], "task_vec": eval_rows["task"][0]}
	action_l2 = mitigation._action_l2_to_direct(direct_model, model, state_bank, device)
	proposal_regret = mitigation._proposal_regret(direct_model, direct_cfg, model, state_bank, device, args)
	elite_coverage = phase38._elite_score_coverage(model, base_model, multitask_cfg, eval_rows, eval_elite, device, args)
	drift = mitigation._old_task_drifts(base_model, model, old_val_rows, device)
	policy_loss = OrderedDict((
		task_id,
		_standard_bc_loss(model, rows, device, args, int(args.seed) + 500 + task_index),
	) for task_index, (task_id, rows) in enumerate(old_val_rows.items()))
	policy_loss["00186"] = _standard_bc_loss(model, objective_rows["replay_val"], device, args, int(args.seed) + 503)
	return {
		"updates": int(updates),
		"train_objective_loss": train_loss,
		"val_objective_loss": val_loss,
		"three_task_replay_bc_loss": policy_loss,
		"heldout_contact_jam_action_l2": action_l2,
		"heldout_proposal_regret": proposal_regret,
		"elite_score_coverage": elite_coverage,
		"old_task_action_drift": drift,
		"old_task_behavior_anchor_mse": _old_anchor_loss_eval(model, reference_model, old_val_rows, device, int(args.eval_batch_size)),
		"policy_parameter_delta_l2": float((_flat_policy_device(model) - base_policy_device).norm().item()),
	}


def _reductions(stage: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
	return {
		"contact_jam_action_l2_reduction_fraction": 1.0 - float(stage["heldout_contact_jam_action_l2"]["mean"]) / max(float(baseline["heldout_contact_jam_action_l2"]["mean"]), 1.0e-8),
		"proposal_regret_reduction_fraction": 1.0 - float(stage["heldout_proposal_regret"]["mean"]) / max(float(baseline["heldout_proposal_regret"]["mean"]), 1.0e-8),
	}


def _max_drift(stage: dict[str, Any]) -> float:
	return max(float(item["mean"]) for item in stage["old_task_action_drift"].values())


def _passes(stage: dict[str, Any], args) -> bool:
	return (
		float(stage["reductions"]["proposal_regret_reduction_fraction"]) >= float(args.min_proposal_regret_improvement)
		and float(stage["reductions"]["contact_jam_action_l2_reduction_fraction"]) > 0.0
		and _max_drift(stage) <= float(args.max_old_action_drift)
	)


def _pareto_frontier(methods: OrderedDict[str, dict[str, Any]]) -> list[dict[str, Any]]:
	points = []
	for name, item in methods.items():
		for stage in item["stages"].values():
			points.append({
				"method": name,
				"family": item["spec"]["family"],
				"lambda": float(item["spec"]["lambda"]),
				"updates": int(stage["updates"]),
				"proposal_regret_reduction": float(stage["reductions"]["proposal_regret_reduction_fraction"]),
				"contact_jam_l2_reduction": float(stage["reductions"]["contact_jam_action_l2_reduction_fraction"]),
				"max_old_task_drift": _max_drift(stage),
			})
	frontier = []
	for point in points:
		dominated = any(
			other is not point
			and other["proposal_regret_reduction"] >= point["proposal_regret_reduction"]
			and other["contact_jam_l2_reduction"] >= point["contact_jam_l2_reduction"]
			and other["max_old_task_drift"] <= point["max_old_task_drift"]
			and (
				other["proposal_regret_reduction"] > point["proposal_regret_reduction"]
				or other["contact_jam_l2_reduction"] > point["contact_jam_l2_reduction"]
				or other["max_old_task_drift"] < point["max_old_task_drift"]
			)
			for other in points
		)
		if not dominated:
			frontier.append(point)
	return sorted(frontier, key=lambda item: (-item["proposal_regret_reduction"], item["max_old_task_drift"]))


def _classify(methods: OrderedDict[str, dict[str, Any]], args) -> dict[str, Any]:
	labels = (
		("elite_behavior", "ELITE_WITH_BEHAVIOR_ANCHOR_EFFECTIVE"),
		("quality_behavior", "QUALITY_BC_WITH_BEHAVIOR_ANCHOR_EFFECTIVE"),
		("elite_parameter", "PARAMETER_ANCHOR_EFFECTIVE"),
	)
	for family, label in labels:
		candidates = [
			(name, stage)
			for name, item in methods.items() if item["spec"]["family"] == family
			for stage in item["stages"].values() if _passes(stage, args)
		]
		if candidates:
			name, stage = max(candidates, key=lambda item: item[1]["reductions"]["proposal_regret_reduction_fraction"])
			return {
				"classification": label,
				"method": name,
				"updates": int(stage["updates"]),
				"reason": "The anchored policy objective cleared the new-task proposal, contact/jam action, and old-task behavior gates without direct-policy supervision.",
			}
	strongest = max(
		((name, stage) for name, item in methods.items() for stage in item["stages"].values()),
		key=lambda item: item[1]["reductions"]["proposal_regret_reduction_fraction"],
	)
	return {
		"classification": "NO_SAFE_POLICY_ADAPTATION",
		"method": None,
		"updates": None,
		"reason": (
			"No anchoring setting cleared proposal-quality and old-task behavior gates together. The strongest unsafe signal is "
			f"{strongest[0]} at {strongest[1]['updates']} updates: proposal-regret reduction="
			f"{strongest[1]['reductions']['proposal_regret_reduction_fraction']:.3f}, max old-task drift={_max_drift(strongest[1]):.3f}."
		),
		"strongest_unsafe_signal": {
			"method": strongest[0],
			"updates": int(strongest[1]["updates"]),
			"proposal_regret_reduction_fraction": float(strongest[1]["reductions"]["proposal_regret_reduction_fraction"]),
			"contact_jam_action_l2_reduction_fraction": float(strongest[1]["reductions"]["contact_jam_action_l2_reduction_fraction"]),
			"max_old_task_drift": _max_drift(strongest[1]),
		},
	}


def _markdown(report: dict[str, Any]) -> str:
	classification = report["classification"]
	lines = [
		"# SRSA Phase 3.9 Old-Task Behavior Anchoring Audit",
		"",
		"本报告冻结 world model，仅训练内存 policy clone。行为锚定的 reference 是更新前 policy；direct checkpoint 只用于 held-out evaluation，绝不参与训练 target。",
		"",
		f"Status: `{report['status']}`",
		f"Final classification: `{classification['classification']}`",
		"",
		"## Conclusion",
		"",
		classification["reason"],
		"",
		"## Anchoring Sweep",
		"",
		"正值 reduction 表示相对 elite baseline update=0 的改善。three-task policy loss 是每个任务 held-out replay-action BC，供观察，不是 gate。",
		"",
		"| Objective | Lambda | Updates | New train/val loss | 00186 BC | 01125 BC | 00256 BC | L2 red. | Regret red. | 01125 drift | 00256 drift | Behavior MSE | Param delta |",
		"| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
	]
	for name, item in report["methods"].items():
		for stage in item["stages"].values():
			loss = stage["three_task_replay_bc_loss"]
			lines.append(
				f"| `{name}` | {item['spec']['lambda']:g} | {stage['updates']} | {stage['train_objective_loss']:.6f}/{stage['val_objective_loss']:.6f} | "
				f"{loss['00186']:.6f} | {loss['01125']:.6f} | {loss['00256']:.6f} | "
				f"{stage['reductions']['contact_jam_action_l2_reduction_fraction']:+.3f} | "
				f"{stage['reductions']['proposal_regret_reduction_fraction']:+.3f} | "
				f"{stage['old_task_action_drift']['01125']['mean']:.4f} | {stage['old_task_action_drift']['00256']['mean']:.4f} | "
				f"{stage['old_task_behavior_anchor_mse']:.6f} | {stage['policy_parameter_delta_l2']:.4f} |"
			)
	lines.extend([
		"",
		"## Proposal/Retention Pareto Frontier",
		"",
		"| Method | Lambda | Updates | Proposal-regret reduction | Contact/jam L2 reduction | Max old-task drift |",
		"| --- | ---: | ---: | ---: | ---: | ---: |",
	])
	for item in report["pareto_frontier"]:
		lines.append(
			f"| `{item['method']}` | {item['lambda']:g} | {item['updates']} | {item['proposal_regret_reduction']:+.3f} | "
			f"{item['contact_jam_l2_reduction']:+.3f} | {item['max_old_task_drift']:.4f} |"
		)
	lines.extend([
		"",
		"## Protocol",
		"",
		f"- Lambda scan: `{report['anchor_lambdas']}`; update checkpoints: `{report['update_checkpoints']}`; policy delta budget: `L2 <= {report['policy_delta_budget']}`.",
		f"- Old-task behavior anchor uses equal-size recorded 01125/00256 state-task-vector batches and a frozen pre-update policy reference. It has no task-id-specific model path.",
		f"- New-task elite targets use frozen multitask world-model scoring: `{report['elite_target']['num_policy_candidates']}` policy + `{report['elite_target']['num_gaussian_candidates']}` Gaussian candidates, top-`{report['elite_target']['num_elites']}` at horizon `{report['elite_target']['horizon']}`.",
		"- No direct policy action or direct rollout is used to construct an objective. No checkpoint, replay, sampler, MPPI, or non-policy module is written.",
	])
	return "\n".join(lines) + "\n"


def build_report(args: argparse.Namespace) -> dict[str, Any]:
	diagnosis = _load_json(args.phase32_diagnosis)
	if diagnosis.get("status") != "PASS":
		raise RuntimeError(f"Unexpected Phase 3.2 diagnosis status: {diagnosis.get('status')}")
	checkpoints = diagnosis.get("checkpoints") or {}
	multitask_checkpoint = resolve(checkpoints.get("multitask_rescue_best", ""))
	direct_checkpoint = resolve(checkpoints.get("direct_finetune", ""))
	replay_paths = OrderedDict(prior_audit.DEFAULT_REPLAYS)
	replay_paths["00186"] = resolve(diagnosis.get("replay", ""))
	replay_paths = OrderedDict((task_id, resolve(path)) for task_id, path in replay_paths.items())
	for path in (multitask_checkpoint, direct_checkpoint, *replay_paths.values(), resolve(args.rollout_root)):
		if not path.exists():
			raise FileNotFoundError(path)

	device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() and not args.cpu else "cpu")
	if device.type == "cuda":
		torch.cuda.set_device(device)
	base_model, cfg, multitask_compat = attribution._load_model(multitask_checkpoint, args, device)
	direct_model, direct_cfg, direct_compat = attribution._load_model(direct_checkpoint, args, device)
	base_policy_cpu = phase37._flatten_policy(base_model).clone()
	policy_lr = float(args.policy_lr) if args.policy_lr is not None else float(cfg.lr)

	new_replay_train_all, new_replay_val_all, _ = phase38._replay_rows(replay_paths["00186"], int(args.seed) + 1)
	new_replay_train = phase38._rows_fixed(new_replay_train_all, int(args.new_replay_train_rows), int(args.seed) + 10)
	new_replay_val = phase38._rows_fixed(new_replay_val_all, int(args.new_replay_val_rows), int(args.seed) + 11)
	new_elite_train = phase38._build_elite_cache(base_model, cfg, new_replay_train, device, args, int(args.seed) + 500)
	new_elite_val = phase38._build_elite_cache(base_model, cfg, new_replay_val, device, args, int(args.seed) + 600)
	new_replay_train = phase38._attach_cache(new_replay_train, new_elite_train)
	new_replay_val = phase38._attach_cache(new_replay_val, new_elite_val)

	quality_train_all, quality_val_all = phase38._quality_rollout_rows(args)
	quality_train = phase38._rows_fixed(quality_train_all, int(args.quality_train_rows), int(args.seed) + 12)
	quality_val = phase38._rows_fixed(quality_val_all, int(args.quality_val_rows), int(args.seed) + 13)
	quality_train_weight, quality_train_counts = phase38._quality_weights(quality_train, args)
	quality_val_weight, quality_val_counts = phase38._quality_weights(quality_val, args)
	quality_train["quality_weight"] = quality_train_weight
	quality_val["quality_weight"] = quality_val_weight
	contact_jam_idx = torch.nonzero((quality_val["phase"] == 1) & (quality_val["outcome"] == 1), as_tuple=False).reshape(-1)
	if contact_jam_idx.numel() == 0:
		raise RuntimeError("Held-out rollout split has no contact/jam states.")
	eval_rows = phase38._rows_fixed(phase38._rows_select(quality_val, contact_jam_idx), int(args.eval_contact_jam_rows), int(args.seed) + 900)
	eval_elite = phase38._build_elite_cache(base_model, cfg, eval_rows, device, args, int(args.seed) + 1000)

	old_train_rows = OrderedDict()
	old_val_rows = OrderedDict()
	for task_index, task_id in enumerate(("01125", "00256")):
		train_all, val_all, _ = phase38._replay_rows(replay_paths[task_id], int(args.seed) + 97 * task_index)
		old_train_rows[task_id] = phase38._rows_fixed(train_all, int(args.old_anchor_train_rows), int(args.seed) + 1100 + task_index)
		old_val_rows[task_id] = phase38._rows_fixed(val_all, int(args.old_task_val_rows), int(args.seed) + 1200 + task_index)

	objective_rows = {
		"elite": {"train": new_replay_train, "val": new_replay_val, "replay_val": new_replay_val},
		"elite_behavior": {"train": new_replay_train, "val": new_replay_val, "replay_val": new_replay_val},
		"elite_parameter": {"train": new_replay_train, "val": new_replay_val, "replay_val": new_replay_val},
		"quality_behavior": {"train": quality_train, "val": quality_val, "replay_val": new_replay_val},
	}
	anchor_lambdas = _parse_lambdas(args.anchor_lambdas)
	specs = _make_specs(anchor_lambdas)
	checkpoints_updates = tuple(sorted(set((100, 500, 2000))))
	methods: OrderedDict[str, dict[str, Any]] = OrderedDict()
	for spec_index, spec in enumerate(specs):
		model = copy.deepcopy(base_model)
		phase37._freeze_world_model_except_policy(model)
		optimizer = torch.optim.Adam(phase37._policy_parameters(model), lr=policy_lr)
		base_policy_device = _flat_policy_device(model).detach().clone()
		new_generator = torch.Generator().manual_seed(int(args.seed) + 10_000 * spec_index)
		old_generators = OrderedDict((
			task_id,
			torch.Generator().manual_seed(int(args.seed) + 20_000 * spec_index + task_index),
		) for task_index, task_id in enumerate(old_train_rows))
		rows = objective_rows[spec["family"]]
		baseline = _stage_metrics(
			model, base_model, base_model, cfg, direct_model, direct_cfg, rows, spec["family"], eval_rows, eval_elite,
			old_val_rows, base_policy_device, device, args, 0,
		)
		stages = OrderedDict()
		for update in range(1, max(checkpoints_updates) + 1):
			new_batch = phase38._sample_batch(rows["train"], int(args.new_batch_size), new_generator)
			optimizer.zero_grad(set_to_none=True)
			new_loss = _new_loss(model, new_batch, spec["family"], device, int(args.seed) + spec_index * 100_000 + update)
			if spec["anchor"] == "behavior":
				old_batches = OrderedDict((
					task_id,
					phase38._sample_batch(old_train_rows[task_id], int(args.old_anchor_batch_size), old_generators[task_id]),
				) for task_id in old_train_rows)
				anchor_loss = _behavior_anchor_loss(model, base_model, old_batches, device)
			elif spec["anchor"] == "parameter":
				anchor_loss = _parameter_l2_anchor(model, base_policy_device)
			else:
				anchor_loss = torch.zeros((), device=device)
			loss = new_loss + float(spec["lambda"]) * anchor_loss
			loss.backward()
			optimizer.step()
			phase38._project_policy_delta(model, base_policy_device, float(args.policy_delta_budget))
			if update in checkpoints_updates:
				stage = _stage_metrics(
					model, base_model, base_model, cfg, direct_model, direct_cfg, rows, spec["family"], eval_rows, eval_elite,
					old_val_rows, base_policy_device, device, args, update,
				)
				stage["reductions"] = _reductions(stage, baseline)
				stage["train_anchor_loss"] = float(anchor_loss.detach().item())
				stages[str(update)] = stage
		methods[spec["name"]] = {"spec": spec, "baseline": baseline, "stages": stages}
		del model, optimizer
		if device.type == "cuda":
			torch.cuda.empty_cache()

	if not torch.equal(base_policy_cpu, phase37._flatten_policy(base_model)):
		raise RuntimeError("Base checkpoint policy prior changed during Phase 3.9 audit.")
	classification = _classify(methods, args)
	status = "PASS" if classification["classification"] != "NO_SAFE_POLICY_ADAPTATION" else "WARNING"
	return {
		"status": status,
		"classification": classification,
		"base_policy_unchanged": True,
		"device": str(device),
		"inputs": {
			"multitask_checkpoint": str(multitask_checkpoint),
			"direct_checkpoint_evaluation_only": str(direct_checkpoint),
			"replays": {task_id: str(path) for task_id, path in replay_paths.items()},
			"rollout_root": str(resolve(args.rollout_root)),
		},
		"checkpoint_compatibility": {"multitask": multitask_compat, "direct": direct_compat},
		"policy_lr": policy_lr,
		"anchor_lambdas": list(anchor_lambdas),
		"policy_delta_budget": float(args.policy_delta_budget),
		"update_checkpoints": list(checkpoints_updates),
		"elite_target": {
			"num_policy_candidates": int(args.num_policy_candidates),
			"num_gaussian_candidates": int(args.num_candidates) - int(args.num_policy_candidates),
			"num_elites": int(args.num_elites),
			"horizon": int(args.horizon),
			"scorer": "frozen_multitask_world_model",
		},
		"split": {
			"new_replay_train_rows": int(new_replay_train["obs"].shape[0]),
			"new_replay_val_rows": int(new_replay_val["obs"].shape[0]),
			"quality_train_rows": int(quality_train["obs"].shape[0]),
			"quality_val_rows": int(quality_val["obs"].shape[0]),
			"heldout_contact_jam_rows": int(eval_rows["obs"].shape[0]),
			"old_anchor_train_rows_per_task": int(args.old_anchor_train_rows),
			"old_task_val_rows_per_task": int(args.old_task_val_rows),
			"quality_train_label_counts": quality_train_counts,
			"quality_val_label_counts": quality_val_counts,
		},
		"gates": {
			"min_proposal_regret_improvement": float(args.min_proposal_regret_improvement),
			"max_old_action_drift": float(args.max_old_action_drift),
		},
		"methods": methods,
		"pareto_frontier": _pareto_frontier(methods),
		"limitations": [
			"Direct policy actions and direct rollout data are evaluation-only and never enter an objective or anchor target.",
			"Behavior anchoring checks recorded old-task state/task-vector pairs; it is a local retention proxy, not a closed-loop retention evaluation.",
			"No original checkpoint, replay, sampler, MPPI, or non-policy module is written or modified.",
		],
	}


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--phase32-diagnosis", default=DEFAULT_PHASE32_DIAGNOSIS)
	parser.add_argument("--rollout-root", default=DEFAULT_ROLLOUT_ROOT)
	parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
	parser.add_argument("--output-md", default=DEFAULT_OUTPUT_MD)
	parser.add_argument("--config", default="configs/train/srsa_01125_imitation_relaxed.yaml")
	parser.add_argument("--gpu-id", type=int, default=0)
	parser.add_argument("--cpu", action="store_true")
	parser.add_argument("--batch-size", type=int, default=256, help="Checkpoint config-loader batch size; audit updates use --new-batch-size.")
	parser.add_argument("--new-batch-size", type=int, default=128)
	parser.add_argument("--old-anchor-batch-size", type=int, default=128)
	parser.add_argument("--eval-batch-size", type=int, default=1024)
	parser.add_argument("--elite-batch-size", type=int, default=64)
	parser.add_argument("--proposal-batch-size", type=int, default=32)
	parser.add_argument("--new-replay-train-rows", type=int, default=4096)
	parser.add_argument("--new-replay-val-rows", type=int, default=2048)
	parser.add_argument("--quality-train-rows", type=int, default=4096)
	parser.add_argument("--quality-val-rows", type=int, default=2048)
	parser.add_argument("--old-anchor-train-rows", type=int, default=4096)
	parser.add_argument("--old-task-val-rows", type=int, default=2048)
	parser.add_argument("--eval-contact-jam-rows", type=int, default=256)
	parser.add_argument("--policy-lr", type=float, default=None)
	parser.add_argument("--policy-delta-budget", type=float, default=3.5)
	parser.add_argument("--anchor-lambdas", default="0.1,0.3,1.0,3.0,10.0")
	parser.add_argument("--quality-success-weight", type=float, default=4.0)
	parser.add_argument("--quality-jam-weight", type=float, default=0.25)
	parser.add_argument("--quality-failure-weight", type=float, default=0.5)
	parser.add_argument("--num-candidates", type=int, default=64)
	parser.add_argument("--num-policy-candidates", type=int, default=3)
	parser.add_argument("--num-elites", type=int, default=8)
	parser.add_argument("--horizon", type=int, default=3)
	parser.add_argument("--min-proposal-regret-improvement", type=float, default=0.20)
	parser.add_argument("--max-old-action-drift", type=float, default=0.05)
	parser.add_argument("--jam-lateral-threshold", type=float, default=0.008)
	parser.add_argument("--jam-keypoint-threshold", type=float, default=0.012)
	parser.add_argument("--jam-force-excursion-threshold", type=float, default=2.0)
	parser.add_argument("--seed", type=int, default=1)
	parser.add_argument("--dry-run", action="store_true")
	args = parser.parse_args()
	if args.num_elites <= 0 or args.num_elites > args.num_candidates:
		raise ValueError("num_elites must be in [1, num_candidates].")
	if args.new_batch_size <= 0 or args.old_anchor_batch_size <= 0:
		raise ValueError("batch sizes must be positive.")
	if args.dry_run:
		print(f"PASS dry-run: would write {resolve(args.output_json)} and {resolve(args.output_md)} without checkpoint mutation.")
		return 0
	report = build_report(args)
	write_json(report, args.output_json)
	write_text(_markdown(report), args.output_md)
	print(report["status"])
	print(f"Final classification: {report['classification']['classification']}")
	print(f"Base policy unchanged: {report['base_policy_unchanged']}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
