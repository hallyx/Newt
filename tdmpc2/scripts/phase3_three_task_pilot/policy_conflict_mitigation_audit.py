#!/usr/bin/env python3
"""Read-only Phase 3.6 virtual policy-prior conflict-mitigation audit."""

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
import policy_prior_supervision_audit as prior_audit  # noqa: E402
from _common import (  # noqa: E402
	condition_batch,
	resolve,
	summarize_tensor,
	tensor_to_list,
	write_json,
	write_text,
)


DEFAULT_PHASE32_DIAGNOSIS = (
	"reports/phase3_three_task_pilot/phase3_2_diagnosis/"
	"standalone_vs_multitask_diagnosis.json"
)
DEFAULT_ROLLOUT_ROOT = "reports/phase3_three_task_pilot/phase3_3_rollouts"
DEFAULT_OUTPUT_JSON = "reports/phase3_three_task_pilot/phase3_6_policy_conflict_mitigation_audit.json"
DEFAULT_OUTPUT_MD = "reports/phase3_three_task_pilot/phase3_6_policy_conflict_mitigation_audit.md"
TASKS = prior_audit.TASKS
CURRENT_WEIGHTS = OrderedDict([
	("00186", 0.75),
	("01125", 0.125),
	("00256", 0.125),
])


def _load_json(path_value: str | Path) -> dict[str, Any]:
	path = resolve(path_value)
	if not path.exists():
		raise FileNotFoundError(path)
	return json.loads(path.read_text(encoding="utf-8"))


def _flatten_policy(model) -> torch.Tensor:
	return torch.cat([parameter.detach().reshape(-1).cpu() for parameter in model._pi.parameters()])


def _policy_parameters(model):
	return tuple(parameter for parameter in model._pi.parameters() if parameter.requires_grad)


def _policy_loss(model, rows, device: torch.device, seed: int) -> float:
	"""Evaluate the update_pi policy-prior term with a fixed reparameterization seed."""
	obs = rows["obs"].to(device)
	action = rows["action"].to(device)
	task = rows["task"].to(device)
	devices = [device.index] if device.type == "cuda" else []
	with torch.no_grad(), torch.random.fork_rng(devices=devices):
		torch.manual_seed(int(seed))
		if device.type == "cuda":
			torch.cuda.manual_seed_all(int(seed))
		z = model.encode(obs, task)
		pi_action, _ = model.pi(z, task)
		loss = prior_audit.td_math.masked_bc_per_timestep(
			pi_action.unsqueeze(0), action.unsqueeze(0), task.unsqueeze(0), model._action_masks
		).mean()
	return float(loss.item())


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
	return float((torch.dot(left, right) / (left.norm() * right.norm()).clamp_min(1.0e-12)).item())


def _weighted_gradient(gradients: dict[str, torch.Tensor], weights: dict[str, float]) -> torch.Tensor:
	result = torch.zeros_like(next(iter(gradients.values())))
	for task_id, weight in weights.items():
		result.add_(gradients[task_id], alpha=float(weight))
	return result


def _pcgrad(gradients: dict[str, torch.Tensor], weights: dict[str, float]) -> tuple[torch.Tensor, dict[str, Any]]:
	"""Deterministic PCGrad over policy-prior gradients only, then current-heavy averaging."""
	projected: dict[str, torch.Tensor] = {}
	projection_events = []
	for task_id in TASKS:
		value = gradients[task_id].clone()
		for other in TASKS:
			if other == task_id:
				continue
			other_gradient = gradients[other]
			dot = torch.dot(value, other_gradient)
			if dot < 0:
				coefficient = dot / other_gradient.pow(2).sum().clamp_min(1.0e-12)
				value = value - coefficient * other_gradient
				projection_events.append({
					"task": task_id,
					"against": other,
					"dot_before": float(dot.item()),
					"coefficient": float(coefficient.item()),
				})
		projected[task_id] = value
	return _weighted_gradient(projected, weights), {
		"projected_gradient_cosines": {
			f"{left}_vs_{right}": _cosine(projected[left], projected[right])
			for index, left in enumerate(TASKS) for right in TASKS[index + 1:]
		},
		"projection_events": projection_events,
	}


def _apply_virtual_policy_step(base_model, direction: torch.Tensor, *, update_l2: float, device: torch.device):
	"""Clone then mutate only the in-memory _pi parameters; the checkpoint/base model remain unchanged."""
	model = copy.deepcopy(base_model)
	parameters = _policy_parameters(model)
	direction = direction.to(device=device, dtype=parameters[0].dtype)
	scale = -float(update_l2) / float(direction.norm().item())
	offset = 0
	with torch.no_grad():
		for parameter in parameters:
			count = parameter.numel()
			parameter.add_(direction[offset:offset + count].reshape_as(parameter), alpha=scale)
			offset += count
	if offset != direction.numel():
		raise RuntimeError("Virtual policy parameter vector length mismatch.")
	return model


def _policy_mean(model, obs: torch.Tensor, task: torch.Tensor):
	return prior_audit._policy_mean(model, obs, task)[0]


def _contact_jam_states(banks, task_vec: torch.Tensor):
	columns = []
	for region in attribution.REGIONS:
		bank = banks[f"multitask_{region}"]
		mask = (bank["phase"] == 1) & (bank["outcome"] == 1)
		if mask.any():
			columns.append(bank["td"]["obs"][mask].detach().float())
	if not columns:
		raise RuntimeError("Phase 3.3 rollouts contain no 00186 contact/jam states.")
	obs = torch.cat(columns, dim=0)
	return {"obs": obs, "task_vec": task_vec.detach().float()}


@torch.no_grad()
def _proposal_regret(direct_model, direct_cfg, candidate_model, state_bank, device: torch.device, args):
	"""Use the fixed direct scorer from Phase 3.3 as the proposal-quality reference."""
	obs_all = state_bank["obs"]
	task_vec = state_bank["task_vec"]
	generator = torch.Generator(device=device).manual_seed(int(args.seed) + 41)
	eps_all = torch.randn(
		obs_all.shape[0], int(args.horizon), int(args.num_candidates), 3,
		device=device, generator=generator,
	)
	regrets = []
	for start in range(0, int(obs_all.shape[0]), int(args.proposal_batch_size)):
		stop = min(start + int(args.proposal_batch_size), int(obs_all.shape[0]))
		obs = obs_all[start:stop].to(device)
		eps = eps_all[start:stop]
		direct = attribution._policy_proposals(
			direct_model, obs, task_vec, eps, horizon=int(args.horizon), num_candidates=int(args.num_candidates)
		)
		candidate = attribution._policy_proposals(
			candidate_model, obs, task_vec, eps, horizon=int(args.horizon), num_candidates=int(args.num_candidates)
		)
		direct_score = attribution._score_candidates(direct_model, direct_cfg, obs, task_vec, direct["actions"])["total"]
		candidate_score = attribution._score_candidates(direct_model, direct_cfg, obs, task_vec, candidate["actions"])["total"]
		regrets.append((direct_score.max(dim=-1).values - candidate_score.max(dim=-1).values).detach().cpu())
	return summarize_tensor(torch.cat(regrets))


def _old_task_drifts(base_model, virtual_model, replay_rows, device: torch.device):
	result = OrderedDict()
	for task_id in ("01125", "00256"):
		rows = replay_rows[task_id]
		with torch.no_grad():
			base = _policy_mean(base_model, rows["obs"].to(device), rows["task"].to(device))
			virtual = _policy_mean(virtual_model, rows["obs"].to(device), rows["task"].to(device))
		result[task_id] = summarize_tensor(torch.linalg.vector_norm(virtual - base, dim=-1))
	return result


def _action_l2_to_direct(direct_model, candidate_model, state_bank, device: torch.device):
	obs = state_bank["obs"].to(device)
	task = condition_batch(state_bank["task_vec"], (obs.shape[0],), device)
	with torch.no_grad():
		direct = _policy_mean(direct_model, obs, task)
		candidate = _policy_mean(candidate_model, obs, task)
	return summarize_tensor(torch.linalg.vector_norm(candidate - direct, dim=-1))


def _direction_report(direction: torch.Tensor, gradients: dict[str, torch.Tensor]):
	return {
		"norm": float(direction.norm().item()),
		"cosine_to_task_gradient": {task_id: _cosine(direction, gradient) for task_id, gradient in gradients.items()},
	}


def _retention_risk(loss_delta: dict[str, float], drifts: dict[str, Any]) -> str:
	max_drift = max(float(item["mean"]) for item in drifts.values())
	old_loss_increase = max(float(loss_delta["01125"]), float(loss_delta["00256"]))
	if old_loss_increase > 0.05 or max_drift > 0.05:
		return "HIGH"
	if old_loss_increase > 0.01 or max_drift > 0.015:
		return "MEDIUM"
	return "LOW"


def _passes_gate(item: dict[str, Any], args) -> bool:
	return (
		float(item["contact_jam_action_l2_reduction_fraction"]) >= float(args.min_improvement)
		or float(item["proposal_regret_reduction_fraction"]) >= float(args.min_improvement)
	) and (
		max(float(value["mean"]) for value in item["old_task_action_drift"].values()) <= float(args.max_old_action_drift)
	) and float(item["policy_loss_delta"]["00256"]) <= float(args.max_00256_loss_increase)


def _recommend(methods: dict[str, Any], args) -> dict[str, Any]:
	passing = {name: item for name, item in methods.items() if _passes_gate(item, args)}
	if "current_heavy_pcgrad" in passing:
		return {
			"selection": "PCGRAD_POLICY_ONLY",
			"method": "current_heavy_pcgrad",
			"risk": passing["current_heavy_pcgrad"]["predicted_retention_risk"],
			"reason": "PCGrad is the least invasive passing policy-only update because it retains all task losses while removing observed negative projections.",
		}
	if "equal_task_policy_loss" in passing:
		return {
			"selection": "DECOUPLED_POLICY_TASK_SAMPLING",
			"method": "equal_task_policy_loss",
			"risk": passing["equal_task_policy_loss"]["predicted_retention_risk"],
			"reason": "Equal policy-only task weighting passes while avoiding a global replay-sampler change.",
		}
	if "exclude_01125_policy_loss" in passing:
		return {
			"selection": "TEMPORARILY_EXCLUDE_01125_POLICY_LOSS",
			"method": "exclude_01125_policy_loss",
			"risk": passing["exclude_01125_policy_loss"]["predicted_retention_risk"],
			"reason": "The audited policy-only gate passes only after excluding the conflicting 01125 policy loss; retention risk remains explicit.",
		}
	if "00186_only_policy_loss" in passing:
		return {
			"selection": "TEMPORARILY_EXCLUDE_01125_POLICY_LOSS",
			"method": "00186_only_policy_loss",
			"risk": "HIGH",
			"reason": "Only the diagnostic 00186-only direction passes; it has high old-task retention risk and cannot be used as a deployment recipe.",
		}
	return {
		"selection": "UNRESOLVED",
		"method": None,
		"risk": "LOW_LOCAL_RETENTION_RISK_BUT_NO_EFFICACY_SIGNAL",
		"reason": "At a training-scale local step, no policy-only virtual direction cleared the 00186 improvement, 00256-loss, and old-task-drift gates together.",
	}


def _markdown(report: dict[str, Any]) -> str:
	recommendation = report["recommendation"]
	lines = [
		"# SRSA Phase 3.6 Policy-Only Conflict Mitigation Audit",
		"",
		"本报告仅在内存克隆的 `WorldModel._pi` 上执行一次虚拟 policy-prior 参数更新。未训练、未写 checkpoint，且 reward、Q、dynamics、task_context、MPPI 和 replay sampler 均未修改。",
		"",
		f"Status: `{report['status']}`",
		f"Recommended policy-only scheme: `{recommendation['selection']}`",
		f"Predicted retention risk: `{recommendation['risk']}`",
		"",
		"## Recommendation",
		"",
		f"{recommendation['reason']}",
		"",
		"## Virtual Update Results",
		"",
		"所有方法使用相同三任务 batch，并将合成方向归一化到相同的 policy-parameter L2 位移预算。00186 contact/jam action L2 以 direct policy prior 在同一 multitask state 上的输出为参考；proposal regret 使用冻结的 direct scorer，较低更好。",
		"",
		"| Method | 00186 contact/jam L2 change | Proposal-regret change | 01125 drift | 00256 drift | 00256 loss delta | Retention risk | Gate |",
		"| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
	]
	for name, item in report["methods"].items():
		lines.append(
			f"| `{name}` | {item['contact_jam_action_l2_reduction_fraction']:+.3e} | "
			f"{item['proposal_regret_reduction_fraction']:+.3e} | "
			f"{item['old_task_action_drift']['01125']['mean']:.5f} | {item['old_task_action_drift']['00256']['mean']:.5f} | "
			f"{item['policy_loss_delta']['00256']:.5f} | `{item['predicted_retention_risk']}` | "
			f"{'PASS' if item['passes_gate'] else 'FAIL'} |"
		)
	lines.extend([
		"",
		"正值 reduction 表示改善；gate 需要 00186 action-L2 或 proposal regret 至少改善 20%、旧任务 action drift 不明显、且 00256 policy loss 不上升。",
		"",
		"## Fixed Audit Setup",
		"",
		f"- Current-heavy policy weights: `{report['method_weights']['current_replay_ratio']}`",
		f"- Virtual policy parameter delta L2: `{report['virtual_update']['parameter_delta_l2']:.6f}` "
		f"(`{report['virtual_update']['relative_parameter_delta']:.4%}` of base policy parameter norm).",
		f"- 00186 contact/jam states: `{report['state_counts']['contact_jam']}`; proposal candidates/state: `{report['proposal']['num_candidates']}`; horizon: `{report['proposal']['horizon']}`.",
		"",
		"## Risk Boundary",
		"",
		"这是单步、冻结 world-model 的局部方向审计，不是训练结果，也不能证明 closed-loop retention。若推荐为 `UNRESOLVED`，不得把任一方案直接接入训练；若推荐为具体方案，仍需单独实现 policy-only ablation 并重跑 family retention。",
	])
	return "\n".join(lines) + "\n"


def build_report(args: argparse.Namespace):
	diagnosis = _load_json(args.phase32_diagnosis)
	if diagnosis.get("status") != "PASS":
		raise RuntimeError(f"Unexpected Phase 3.2 diagnosis status: {diagnosis.get('status')}")
	checkpoint = resolve((diagnosis.get("checkpoints") or {}).get("multitask_rescue_best", ""))
	direct_checkpoint = resolve((diagnosis.get("checkpoints") or {}).get("direct_finetune", ""))
	replay_paths = OrderedDict(prior_audit.DEFAULT_REPLAYS)
	replay_paths["00186"] = str(diagnosis.get("replay", ""))
	replay_paths = OrderedDict((task_id, resolve(path)) for task_id, path in replay_paths.items())
	for path in (checkpoint, direct_checkpoint, *replay_paths.values(), resolve(args.rollout_root)):
		if not path.exists():
			raise FileNotFoundError(path)

	device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() and not args.cpu else "cpu")
	if device.type == "cuda":
		torch.cuda.set_device(device)
	base_model, cfg, compat = attribution._load_model(checkpoint, args, device)
	direct_model, direct_cfg, direct_compat = attribution._load_model(direct_checkpoint, args, device)
	base_before = _flatten_policy(base_model).clone()
	if int(compat["obs_dim"]) != 17 or int(compat["action_dim"]) != 3:
		raise RuntimeError(f"Unexpected multitask checkpoint contract: {compat}")

	replay_rows = OrderedDict()
	replay_task_vecs = OrderedDict()
	for index, (task_id, path) in enumerate(replay_paths.items()):
		td, _ = prior_audit._load_snapshot(path)
		replay_rows[task_id] = prior_audit._sample_rows(td, int(args.batch_size), int(args.seed) + 97 * index)
		replay_task_vecs[task_id] = prior_audit._unique_task_vec(td)

	gradients = OrderedDict()
	base_losses = OrderedDict()
	for task_index, task_id in enumerate(TASKS):
		rows = replay_rows[task_id]
		item = prior_audit._policy_prior_loss_and_gradient(
			base_model,
			rows["obs"].to(device), rows["action"].to(device), rows["task"].to(device),
			seed=int(args.seed) + task_index,
		)
		gradients[task_id] = item["gradient"]
		base_losses[task_id] = _policy_loss(base_model, rows, device, int(args.seed) + task_index)

	banks, _ = attribution._load_rollout_banks(args)
	state_bank = _contact_jam_states(banks, replay_task_vecs["00186"])
	base_action_l2 = _action_l2_to_direct(direct_model, base_model, state_bank, device)
	base_proposal_regret = _proposal_regret(direct_model, direct_cfg, base_model, state_bank, device, args)

	equal_weights = OrderedDict((task_id, 1.0 / len(TASKS)) for task_id in TASKS)
	exclude_01125 = OrderedDict([
		("00186", 6.0 / 7.0),
		("00256", 1.0 / 7.0),
	])
	directions = OrderedDict()
	direction_meta = OrderedDict()
	directions["current_replay_ratio"] = _weighted_gradient(gradients, CURRENT_WEIGHTS)
	direction_meta["current_replay_ratio"] = {"weights": CURRENT_WEIGHTS}
	directions["equal_task_policy_loss"] = _weighted_gradient(gradients, equal_weights)
	direction_meta["equal_task_policy_loss"] = {"weights": equal_weights}
	pc_direction, pc_meta = _pcgrad(gradients, CURRENT_WEIGHTS)
	directions["current_heavy_pcgrad"] = pc_direction
	direction_meta["current_heavy_pcgrad"] = {"weights": CURRENT_WEIGHTS, **pc_meta}
	directions["exclude_01125_policy_loss"] = _weighted_gradient(gradients, exclude_01125)
	direction_meta["exclude_01125_policy_loss"] = {"weights": exclude_01125}
	directions["00186_only_policy_loss"] = gradients["00186"].clone()
	direction_meta["00186_only_policy_loss"] = {"weights": OrderedDict([("00186", 1.0)])}

	policy_norm = float(base_before.norm().item())
	update_l2 = policy_norm * float(args.virtual_param_delta_ratio)
	methods = OrderedDict()
	for name, direction in directions.items():
		virtual_model = _apply_virtual_policy_step(base_model, direction, update_l2=update_l2, device=device)
		losses = OrderedDict((
			task_id,
			_policy_loss(virtual_model, rows, device, int(args.seed) + task_index),
		) for task_index, (task_id, rows) in enumerate(replay_rows.items()))
		loss_delta = OrderedDict((task_id, losses[task_id] - base_losses[task_id]) for task_id in TASKS)
		action_l2 = _action_l2_to_direct(direct_model, virtual_model, state_bank, device)
		proposal_regret = _proposal_regret(direct_model, direct_cfg, virtual_model, state_bank, device, args)
		drifts = _old_task_drifts(base_model, virtual_model, replay_rows, device)
		methods[name] = {
			"direction": _direction_report(direction, gradients),
			"method_detail": direction_meta[name],
			"policy_loss": losses,
			"policy_loss_delta": loss_delta,
			"contact_jam_action_l2": action_l2,
			"contact_jam_action_l2_reduction_fraction": 1.0 - float(action_l2["mean"]) / max(float(base_action_l2["mean"]), 1.0e-8),
			"proposal_regret": proposal_regret,
			"proposal_regret_reduction_fraction": 1.0 - float(proposal_regret["mean"]) / max(float(base_proposal_regret["mean"]), 1.0e-8),
			"old_task_action_drift": drifts,
			"predicted_retention_risk": _retention_risk(loss_delta, drifts),
		}
		methods[name]["passes_gate"] = _passes_gate(methods[name], args)
		del virtual_model
		if device.type == "cuda":
			torch.cuda.empty_cache()

	base_after = _flatten_policy(base_model)
	if not torch.equal(base_before, base_after):
		raise RuntimeError("Base policy prior changed during a virtual update audit.")
	recommendation = _recommend(methods, args)
	return {
		"status": "PASS" if recommendation["selection"] != "UNRESOLVED" else "WARNING",
		"recommendation": recommendation,
		"inputs": {
			"multitask_checkpoint": str(checkpoint),
			"direct_checkpoint": str(direct_checkpoint),
			"replays": {task_id: str(path) for task_id, path in replay_paths.items()},
			"rollout_root": str(resolve(args.rollout_root)),
		},
		"device": str(device),
		"checkpoint_compatibility": {"multitask": compat, "direct": direct_compat},
		"base_policy_unchanged": True,
		"method_weights": {
			"current_replay_ratio": CURRENT_WEIGHTS,
			"equal_task_policy_loss": equal_weights,
			"exclude_01125_policy_loss": exclude_01125,
		},
		"base_metrics": {
			"policy_loss": base_losses,
			"contact_jam_action_l2": base_action_l2,
			"proposal_regret": base_proposal_regret,
			"gradient_pair_cosine": {
				f"{left}_vs_{right}": _cosine(gradients[left], gradients[right])
				for index, left in enumerate(TASKS) for right in TASKS[index + 1:]
			},
		},
		"virtual_update": {
			"definition": "one equal-L2, normalized negative gradient step applied only to an in-memory clone of WorldModel._pi",
			"base_policy_parameter_norm": policy_norm,
			"parameter_delta_l2": update_l2,
			"relative_parameter_delta": float(args.virtual_param_delta_ratio),
		},
		"state_counts": {"contact_jam": int(state_bank["obs"].shape[0])},
		"proposal": {"num_candidates": int(args.num_candidates), "horizon": int(args.horizon), "scorer": "frozen_direct_model"},
		"gates": {
			"min_improvement": float(args.min_improvement),
			"max_old_action_drift": float(args.max_old_action_drift),
			"max_00256_loss_increase": float(args.max_00256_loss_increase),
		},
		"methods": methods,
		"limitations": [
			"Virtual updates expose only local supervised-policy direction effects; they are not optimizer trajectories or closed-loop rollouts.",
			"Proposal regret uses the Phase 3.3 direct checkpoint scorer as a fixed reference, not environment return ground truth.",
			"No virtual method modifies policy heads, reward/Q/dynamics/task context, MPPI, or the replay sampler.",
		],
	}


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--phase32-diagnosis", default=DEFAULT_PHASE32_DIAGNOSIS)
	parser.add_argument("--rollout-root", default=DEFAULT_ROLLOUT_ROOT)
	parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
	parser.add_argument("--output-md", default=DEFAULT_OUTPUT_MD)
	parser.add_argument("--config", default="configs/train/srsa_01125_imitation_relaxed.yaml")
	parser.add_argument("--batch-size", type=int, default=2048)
	parser.add_argument("--proposal-batch-size", type=int, default=32)
	parser.add_argument("--num-candidates", type=int, default=64)
	parser.add_argument("--horizon", type=int, default=3)
	parser.add_argument(
		"--virtual-param-delta-ratio", type=float, default=5.0e-7,
		help="Relative policy-parameter L2 for one local virtual step; default matches the 3e-4 training-LR scale.",
	)
	parser.add_argument("--min-improvement", type=float, default=0.20)
	parser.add_argument("--max-old-action-drift", type=float, default=0.05)
	parser.add_argument("--max-00256-loss-increase", type=float, default=0.0)
	parser.add_argument("--jam-lateral-threshold", type=float, default=0.008)
	parser.add_argument("--jam-keypoint-threshold", type=float, default=0.012)
	parser.add_argument("--jam-force-excursion-threshold", type=float, default=2.0)
	parser.add_argument("--seed", type=int, default=1)
	parser.add_argument("--gpu-id", type=int, default=0)
	parser.add_argument("--cpu", action="store_true")
	parser.add_argument("--dry-run", action="store_true")
	args = parser.parse_args()
	if args.dry_run:
		print(f"PASS dry-run: would write {resolve(args.output_json)} and {resolve(args.output_md)} without writing a checkpoint.")
		return 0
	report = build_report(args)
	write_json(report, args.output_json)
	write_text(_markdown(report), args.output_md)
	print(report["status"])
	print(f"Recommended policy-only scheme: {report['recommendation']['selection']}")
	print(f"Predicted retention risk: {report['recommendation']['risk']}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
