#!/usr/bin/env python3
"""Read-only Phase 3.7 multi-step trainability audit for the policy prior alone."""

from __future__ import annotations

import argparse
import copy
import json
import math
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
import policy_prior_supervision_audit as prior_audit  # noqa: E402
from _common import resolve, summarize_tensor, td_math, write_json, write_text  # noqa: E402


DEFAULT_PHASE32_DIAGNOSIS = (
	"reports/phase3_three_task_pilot/phase3_2_diagnosis/"
	"standalone_vs_multitask_diagnosis.json"
)
DEFAULT_ROLLOUT_ROOT = "reports/phase3_three_task_pilot/phase3_3_rollouts"
DEFAULT_OUTPUT_JSON = "reports/phase3_three_task_pilot/phase3_7_policy_trainability_audit.json"
DEFAULT_OUTPUT_MD = "reports/phase3_three_task_pilot/phase3_7_policy_trainability_audit.md"
TASKS = prior_audit.TASKS
CURRENT_WEIGHTS = mitigation.CURRENT_WEIGHTS


def _load_json(path_value: str | Path) -> dict[str, Any]:
	path = resolve(path_value)
	if not path.exists():
		raise FileNotFoundError(path)
	return json.loads(path.read_text(encoding="utf-8"))


def _policy_parameters(model):
	return tuple(parameter for parameter in model._pi.parameters() if parameter.requires_grad)


def _flatten_policy(model) -> torch.Tensor:
	return torch.cat([parameter.detach().reshape(-1).cpu() for parameter in model._pi.parameters()])


def _freeze_world_model_except_policy(model) -> None:
	for parameter in model.parameters():
		parameter.requires_grad_(False)
	for parameter in model._pi.parameters():
		parameter.requires_grad_(True)


def _episode_split(td, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
	"""Deterministic 70/30 split that keeps a full episode on one side."""
	valid = prior_audit._finite_transition_mask(td)
	episodes = td["episode"].detach().long().reshape(-1)
	train = torch.zeros_like(valid)
	for episode in torch.unique(episodes, sorted=True):
		stable_hash = 37 * int(episode.item()) + int(seed)
		if stable_hash % 10 < 7:
			train |= episodes == episode
	train &= valid
	val = valid & ~train
	if not train.any() or not val.any():
		raise RuntimeError("Episode-wise replay split is degenerate.")
	return train, val


def _rows_from_indices(td, indices: torch.Tensor):
	return {
		"obs": td["obs"][indices].detach().float(),
		"action": td["action"][indices].detach().float(),
		"task": td["task"][indices].detach().float(),
		"episode": td["episode"][indices].detach().long(),
	}


def _sample_batch(rows, size: int, generator: torch.Generator):
	indices = torch.randint(int(rows["obs"].shape[0]), (int(size),), generator=generator)
	return {key: value[indices] for key, value in rows.items() if key != "episode"}


def _fixed_metric_rows(rows, size: int, seed: int):
	count = min(int(size), int(rows["obs"].shape[0]))
	generator = torch.Generator().manual_seed(int(seed))
	indices = torch.randperm(int(rows["obs"].shape[0]), generator=generator)[:count]
	return {key: value[indices] for key, value in rows.items()}


def _policy_loss_tensor(model, batch, device: torch.device, seed: int) -> torch.Tensor:
	obs = batch["obs"].to(device)
	action = batch["action"].to(device)
	task = batch["task"].to(device)
	devices = [device.index] if device.type == "cuda" else []
	with torch.random.fork_rng(devices=devices):
		torch.manual_seed(int(seed))
		if device.type == "cuda":
			torch.cuda.manual_seed_all(int(seed))
		z = model.encode(obs, task)
		pi_action, _ = model.pi(z, task)
		return td_math.masked_bc_per_timestep(
			pi_action.unsqueeze(0), action.unsqueeze(0), task.unsqueeze(0), model._action_masks
		).mean()


def _evaluate_loss(model, rows, device: torch.device, seed: int, batch_size: int):
	values = []
	for start in range(0, int(rows["obs"].shape[0]), int(batch_size)):
		stop = min(start + int(batch_size), int(rows["obs"].shape[0]))
		batch = {key: value[start:stop] for key, value in rows.items() if key != "episode"}
		with torch.no_grad():
			values.append(float(_policy_loss_tensor(model, batch, device, int(seed) + start).item()))
	return float(sum(values) / len(values))


def _flatten_gradients(grads, parameters) -> torch.Tensor:
	values = []
	for grad, parameter in zip(grads, parameters):
		values.append(torch.zeros_like(parameter).reshape(-1) if grad is None else grad.detach().reshape(-1))
	return torch.cat(values)


def _set_flat_gradients(parameters, gradient: torch.Tensor):
	offset = 0
	for parameter in parameters:
		count = parameter.numel()
		parameter.grad = gradient[offset:offset + count].reshape_as(parameter).clone()
		offset += count
	if offset != gradient.numel():
		raise RuntimeError("Flattened policy gradient length mismatch.")


def _pcgrad_from_losses(losses: dict[str, torch.Tensor], parameters, weights: dict[str, float]):
	gradients = {}
	for index, task_id in enumerate(TASKS):
		grads = torch.autograd.grad(losses[task_id], parameters, retain_graph=index < len(TASKS) - 1, allow_unused=True)
		gradients[task_id] = _flatten_gradients(grads, parameters)
	projected = {}
	for task_id in TASKS:
		value = gradients[task_id].clone()
		for other in TASKS:
			if other == task_id:
				continue
			other_gradient = gradients[other]
			dot = torch.dot(value, other_gradient)
			if dot < 0:
				value -= dot / other_gradient.pow(2).sum().clamp_min(1.0e-12) * other_gradient
		projected[task_id] = value
	combined = torch.zeros_like(next(iter(projected.values())))
	for task_id, weight in weights.items():
		combined.add_(projected[task_id], alpha=float(weight))
	return combined


def _method_weights(name: str) -> OrderedDict[str, float]:
	if name == "current_weighted_loss":
		return OrderedDict(CURRENT_WEIGHTS)
	if name == "00186_only":
		return OrderedDict([("00186", 1.0)])
	if name == "pcgrad_policy_only":
		return OrderedDict(CURRENT_WEIGHTS)
	if name == "exclude_01125_policy_loss":
		return OrderedDict([("00186", 6.0 / 7.0), ("00256", 1.0 / 7.0)])
	raise ValueError(name)


def _policy_update(model, optimizer, batches, method: str, device: torch.device, seed: int):
	parameters = _policy_parameters(model)
	optimizer.zero_grad(set_to_none=True)
	if method == "pcgrad_policy_only":
		losses = OrderedDict((
			task_id,
			_policy_loss_tensor(model, batches[task_id], device, int(seed) + task_index),
		) for task_index, task_id in enumerate(TASKS))
		gradient = _pcgrad_from_losses(losses, parameters, _method_weights(method))
		_set_flat_gradients(parameters, gradient)
		optimizer.step()
		return {task_id: float(loss.detach().item()) for task_id, loss in losses.items()}

	weights = _method_weights(method)
	losses = OrderedDict((
		task_id,
		_policy_loss_tensor(model, batches[task_id], device, int(seed) + task_index),
	) for task_index, task_id in enumerate(weights))
	combined = sum(float(weights[task_id]) * loss for task_id, loss in losses.items())
	combined.backward()
	optimizer.step()
	return {task_id: float(loss.detach().item()) for task_id, loss in losses.items()}



def _evaluate_stage(model, base_model, base_policy, direct_model, direct_cfg, train_metric_rows, val_rows, state_bank, device: torch.device, args, stage: int):
	train_policy_loss = OrderedDict((
		task_id,
		_evaluate_loss(model, rows, device, int(args.seed) + 5000 + task_index, int(args.eval_batch_size)),
	) for task_index, (task_id, rows) in enumerate(train_metric_rows.items()))
	policy_loss = OrderedDict((
		task_id,
		_evaluate_loss(model, rows, device, int(args.seed) + 10000 + task_index, int(args.eval_batch_size)),
	) for task_index, (task_id, rows) in enumerate(val_rows.items()))
	action_l2 = mitigation._action_l2_to_direct(direct_model, model, state_bank, device)
	proposal_regret = mitigation._proposal_regret(direct_model, direct_cfg, model, state_bank, device, args)
	drift = mitigation._old_task_drifts(base_model, model, val_rows, device)
	parameter_delta = torch.linalg.vector_norm(_flatten_policy(model) - base_policy).item()
	return {
		"updates": int(stage),
		"train_policy_loss": train_policy_loss,
		"val_policy_loss": policy_loss,
		"contact_jam_action_l2": action_l2,
		"proposal_regret": proposal_regret,
		"old_task_action_drift": drift,
		"policy_parameter_delta_l2": float(parameter_delta),
	}


def _stage_reductions(stage: dict[str, Any], baseline: dict[str, Any]):
	return {
		"contact_jam_action_l2_reduction_fraction": 1.0 - float(stage["contact_jam_action_l2"]["mean"]) / max(float(baseline["contact_jam_action_l2"]["mean"]), 1.0e-8),
		"proposal_regret_reduction_fraction": 1.0 - float(stage["proposal_regret"]["mean"]) / max(float(baseline["proposal_regret"]["mean"]), 1.0e-8),
	}


def _classify(method_reports: dict[str, Any], args) -> dict[str, Any]:
	best = {}
	for name, report in method_reports.items():
		stages = report["stages"]
		best[name] = max(
			stages.values(),
			key=lambda item: max(item["reductions"]["contact_jam_action_l2_reduction_fraction"], item["reductions"]["proposal_regret_reduction_fraction"]),
		)
	def effective(name: str):
		stage = best[name]
		return (
			max(stage["reductions"].values()) >= float(args.min_improvement)
			and max(float(value["mean"]) for value in stage["old_task_action_drift"].values()) <= float(args.max_old_action_drift)
		)
	if effective("pcgrad_policy_only"):
		return {
			"classification": "PCGRAD_MULTI_STEP_EFFECTIVE",
			"method": "pcgrad_policy_only",
			"stage": best["pcgrad_policy_only"]["updates"],
			"reason": "PCGrad reached the 00186 proposal/action gate with controllable old-task action drift.",
		}
	for name in ("current_weighted_loss", "exclude_01125_policy_loss", "00186_only"):
		if effective(name):
			return {
				"classification": "OTHER_POLICY_WEIGHTING_EFFECTIVE",
				"method": name,
				"stage": best[name]["updates"],
				"reason": "A non-PCGrad policy-only weighting reached the 00186 gate with controllable old-task action drift.",
			}

	baseline_00186_loss = float(method_reports["00186_only"]["baseline"]["val_policy_loss"]["00186"])
	best_00186 = best["00186_only"]
	one_task_loss_down = any(
		float(stage["val_policy_loss"]["00186"]) < baseline_00186_loss
		for stage in method_reports["00186_only"]["stages"].values()
	)
	one_task_proposal_not_better = max(
		max(stage["reductions"].values())
		for stage in method_reports["00186_only"]["stages"].values()
	) < 0.05
	all_loss_down_without_proposal = all(
		min(float(stage["val_policy_loss"]["00186"]) for stage in report["stages"].values())
		< float(report["baseline"]["val_policy_loss"]["00186"])
		and max(max(stage["reductions"].values()) for stage in report["stages"].values()) < 0.05
		for report in method_reports.values()
	)
	if one_task_loss_down and one_task_proposal_not_better and all_loss_down_without_proposal:
		return {
			"classification": "POLICY_OBJECTIVE_MISALIGNMENT",
			"method": None,
			"stage": best_00186["updates"],
			"reason": "Even 00186-only imitation reduces validation policy loss without improving direct-reference proposal quality, so gradient conflict is not the dominant remaining mechanism.",
		}
	if not one_task_loss_down and one_task_proposal_not_better:
		return {
			"classification": "POLICY_CAPACITY_OR_TARGET_LIMIT",
			"method": None,
			"stage": best_00186["updates"],
			"reason": "00186-only policy-prior training neither fits its held-out imitation target nor improves proposal quality.",
		}
	return {
		"classification": "UNRESOLVED",
		"method": None,
		"stage": None,
		"reason": "Multi-step policy-only variants did not establish a single trainable mechanism under the fixed objective and frozen world model.",
	}


def _markdown(report: dict[str, Any]) -> str:
	classification = report["classification"]
	lines = [
		"# SRSA Phase 3.7 Multi-Step Policy-Only Trainability Audit",
		"",
		"本报告冻结 world model，仅在内存克隆中训练 `WorldModel._pi`。未写 checkpoint，未改变 reward、Q、dynamics、task_context、MPPI、sampler，也未进入 consolidation。",
		"",
		f"Status: `{report['status']}`",
		f"Final classification: `{classification['classification']}`",
		"",
		"## Conclusion",
		"",
		classification["reason"],
		"",
		"## Multi-Step Validation",
		"",
		"正值 reduction 表示相对 update=0 的改善；proposal regret 使用 Phase 3.3 冻结 direct scorer，较低更好。",
		"",
		"| Method | Updates | 00186 train loss | 00186 val loss | Contact/jam L2 reduction | Proposal-regret reduction | 01125 drift | 00256 drift | Param delta L2 |",
		"| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
	]
	for name, report_item in report["methods"].items():
		for stage in report_item["stages"].values():
			lines.append(
				f"| `{name}` | {stage['updates']} | {stage['train_policy_loss']['00186']:.6f} | {stage['val_policy_loss']['00186']:.6f} | "
				f"{stage['reductions']['contact_jam_action_l2_reduction_fraction']:+.3f} | "
				f"{stage['reductions']['proposal_regret_reduction_fraction']:+.3f} | "
				f"{stage['old_task_action_drift']['01125']['mean']:.4f} | "
				f"{stage['old_task_action_drift']['00256']['mean']:.4f} | {stage['policy_parameter_delta_l2']:.4f} |"
			)
	lines.extend([
		"",
		"## Protocol",
		"",
		f"- Episode split: `{report['split']['train_episodes']}` train / `{report['split']['val_episodes']}` validation across all tasks.",
		f"- Policy-only optimizer: Adam lr=`{report['policy_lr']}`; batch/task=`{report['batch_per_task']}`; updates=`{report['update_checkpoints']}`; fixed train metric rows/task=`{report['train_metric_rows_per_task']}`.",
		"- All methods use the same per-task replay batch seeds. PCGrad projects only policy-prior gradients; the rest of the world model has `requires_grad=False`.",
		"- `00186_only` is diagnostic only, not a retention-safe deployment recommendation.",
	])
	return "\n".join(lines) + "\n"


def build_report(args: argparse.Namespace):
	diagnosis = _load_json(args.phase32_diagnosis)
	if diagnosis.get("status") != "PASS":
		raise RuntimeError(f"Unexpected Phase 3.2 diagnosis status: {diagnosis.get('status')}")
	checkpoints = diagnosis.get("checkpoints") or {}
	multitask_checkpoint = resolve(checkpoints.get("multitask_rescue_best", ""))
	direct_checkpoint = resolve(checkpoints.get("direct_finetune", ""))
	replay_paths = OrderedDict(prior_audit.DEFAULT_REPLAYS)
	replay_paths["00186"] = str(diagnosis.get("replay", ""))
	replay_paths = OrderedDict((task_id, resolve(path)) for task_id, path in replay_paths.items())
	for path in (multitask_checkpoint, direct_checkpoint, *replay_paths.values(), resolve(args.rollout_root)):
		if not path.exists():
			raise FileNotFoundError(path)

	device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() and not args.cpu else "cpu")
	if device.type == "cuda":
		torch.cuda.set_device(device)
	base_model, cfg, compat = attribution._load_model(multitask_checkpoint, args, device)
	direct_model, direct_cfg, direct_compat = attribution._load_model(direct_checkpoint, args, device)
	base_policy = _flatten_policy(base_model).clone()
	policy_lr = float(args.policy_lr) if args.policy_lr is not None else float(cfg.lr)

	train_rows = OrderedDict()
	val_rows = OrderedDict()
	train_metric_rows = OrderedDict()
	task_vecs = OrderedDict()
	split_counts = {"train_episodes": 0, "val_episodes": 0, "tasks": OrderedDict()}
	for task_index, (task_id, path) in enumerate(replay_paths.items()):
		td, _ = prior_audit._load_snapshot(path)
		train_mask, val_mask = _episode_split(td, int(args.seed) + 97 * task_index)
		train_indices = torch.nonzero(train_mask, as_tuple=False).reshape(-1)
		val_indices = torch.nonzero(val_mask, as_tuple=False).reshape(-1)
		train_rows[task_id] = _rows_from_indices(td, train_indices)
		val_rows[task_id] = _rows_from_indices(td, val_indices)
		train_metric_rows[task_id] = _fixed_metric_rows(train_rows[task_id], int(args.train_metric_rows), int(args.seed) + 1000 + task_index)
		task_vecs[task_id] = prior_audit._unique_task_vec(td)
		train_episodes = int(torch.unique(train_rows[task_id]["episode"]).numel())
		val_episodes = int(torch.unique(val_rows[task_id]["episode"]).numel())
		split_counts["tasks"][task_id] = {
			"train_transitions": int(train_indices.numel()), "val_transitions": int(val_indices.numel()),
			"train_episodes": train_episodes, "val_episodes": val_episodes,
		}
		split_counts["train_episodes"] += train_episodes
		split_counts["val_episodes"] += val_episodes

	banks, _ = attribution._load_rollout_banks(args)
	state_bank = mitigation._contact_jam_states(banks, task_vecs["00186"])
	update_checkpoints = tuple(sorted(set((100, 500, 2000))))
	max_updates = max(update_checkpoints)
	methods = OrderedDict()
	method_names = ("current_weighted_loss", "00186_only", "pcgrad_policy_only", "exclude_01125_policy_loss")
	for method_index, method in enumerate(method_names):
		model = copy.deepcopy(base_model)
		_freeze_world_model_except_policy(model)
		optimizer = torch.optim.Adam(_policy_parameters(model), lr=policy_lr)
		generators = {task_id: torch.Generator().manual_seed(int(args.seed) + 10000 * method_index + 101 * task_index)
			for task_index, task_id in enumerate(TASKS)}
		baseline = _evaluate_stage(model, base_model, base_policy, direct_model, direct_cfg, train_metric_rows, val_rows, state_bank, device, args, 0)
		stages = OrderedDict()
		for update in range(1, max_updates + 1):
			batches = {task_id: _sample_batch(train_rows[task_id], int(args.batch_per_task), generators[task_id]) for task_id in TASKS}
			_policy_update(model, optimizer, batches, method, device, int(args.seed) + 100000 * method_index + update * 11)
			if update in update_checkpoints:
				stage = _evaluate_stage(model, base_model, base_policy, direct_model, direct_cfg, train_metric_rows, val_rows, state_bank, device, args, update)
				stage["reductions"] = _stage_reductions(stage, baseline)
				stages[str(update)] = stage
		methods[method] = {"weights": _method_weights(method), "baseline": baseline, "stages": stages}
		del model, optimizer
		if device.type == "cuda":
			torch.cuda.empty_cache()

	if not torch.equal(base_policy, _flatten_policy(base_model)):
		raise RuntimeError("The base checkpoint policy prior changed during Phase 3.7 audit.")
	classification = _classify(methods, args)
	status = "PASS" if classification["classification"] in {"PCGRAD_MULTI_STEP_EFFECTIVE", "OTHER_POLICY_WEIGHTING_EFFECTIVE"} else "WARNING"
	return {
		"status": status,
		"classification": classification,
		"base_policy_unchanged": True,
		"inputs": {
			"multitask_checkpoint": str(multitask_checkpoint), "direct_checkpoint": str(direct_checkpoint),
			"replays": {task_id: str(path) for task_id, path in replay_paths.items()},
			"rollout_root": str(resolve(args.rollout_root)),
		},
		"device": str(device),
		"checkpoint_compatibility": {"multitask": compat, "direct": direct_compat},
		"policy_lr": policy_lr,
		"batch_per_task": int(args.batch_per_task),
		"train_metric_rows_per_task": int(args.train_metric_rows),
		"update_checkpoints": list(update_checkpoints),
		"split": split_counts,
		"state_counts": {"contact_jam": int(state_bank["obs"].shape[0])},
		"proposal": {"num_candidates": int(args.num_candidates), "horizon": int(args.horizon), "scorer": "frozen_direct_model"},
		"gates": {"min_improvement": float(args.min_improvement), "max_old_action_drift": float(args.max_old_action_drift)},
		"methods": methods,
		"limitations": [
			"This is policy-prior-only optimization against replay actions; it is not a TD-MPC2 joint update and does not establish closed-loop success.",
			"Proposal regret is evaluated with the frozen Phase 3.3 direct scorer, not environment return ground truth.",
			"No original checkpoint, replay, sampler, or non-policy module is written or modified.",
		],
	}


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--phase32-diagnosis", default=DEFAULT_PHASE32_DIAGNOSIS)
	parser.add_argument("--rollout-root", default=DEFAULT_ROLLOUT_ROOT)
	parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
	parser.add_argument("--output-md", default=DEFAULT_OUTPUT_MD)
	parser.add_argument("--config", default="configs/train/srsa_01125_imitation_relaxed.yaml")
	parser.add_argument("--batch-size", type=int, default=256, help="Checkpoint config-loader batch size; audit sampling uses --batch-per-task.")
	parser.add_argument("--batch-per-task", type=int, default=128)
	parser.add_argument("--train-metric-rows", type=int, default=4096)
	parser.add_argument("--eval-batch-size", type=int, default=1024)
	parser.add_argument("--policy-lr", type=float, default=None)
	parser.add_argument("--proposal-batch-size", type=int, default=32)
	parser.add_argument("--num-candidates", type=int, default=64)
	parser.add_argument("--horizon", type=int, default=3)
	parser.add_argument("--min-improvement", type=float, default=0.20)
	parser.add_argument("--max-old-action-drift", type=float, default=0.05)
	parser.add_argument("--jam-lateral-threshold", type=float, default=0.008)
	parser.add_argument("--jam-keypoint-threshold", type=float, default=0.012)
	parser.add_argument("--jam-force-excursion-threshold", type=float, default=2.0)
	parser.add_argument("--seed", type=int, default=1)
	parser.add_argument("--gpu-id", type=int, default=0)
	parser.add_argument("--cpu", action="store_true")
	parser.add_argument("--dry-run", action="store_true")
	args = parser.parse_args()
	if args.dry_run:
		print(f"PASS dry-run: would run policy-only in-memory updates and write {resolve(args.output_json)} and {resolve(args.output_md)}.")
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
