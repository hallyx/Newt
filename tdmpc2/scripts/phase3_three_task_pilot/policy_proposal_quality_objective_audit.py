#!/usr/bin/env python3
"""Read-only Phase 3.8 audit of policy-prior objectives against proposal quality.

The audit freezes the world model and mutates only deep-copied ``WorldModel._pi``
instances.  It deliberately never uses the direct checkpoint as a supervision
source: the direct model is an evaluation-only reference for the held-out
action-L2 and proposal-regret metrics.
"""

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
from _common import condition_batch, resolve, summarize_tensor, td_math, two_hot_scalar, write_json, write_text  # noqa: E402


DEFAULT_PHASE32_DIAGNOSIS = (
	"reports/phase3_three_task_pilot/phase3_2_diagnosis/"
	"standalone_vs_multitask_diagnosis.json"
)
DEFAULT_ROLLOUT_ROOT = "reports/phase3_three_task_pilot/phase3_3_rollouts"
DEFAULT_OUTPUT_JSON = "reports/phase3_three_task_pilot/phase3_8_policy_objective_audit.json"
DEFAULT_OUTPUT_MD = "reports/phase3_three_task_pilot/phase3_8_policy_objective_audit.md"
TASKS = prior_audit.TASKS
METHODS = (
	"replay_action_bc",
	"advantage_weighted_bc",
	"mppi_elite_distillation",
	"contact_jam_quality_weighted_bc",
	"elite_plus_contact_jam_weighting",
)


def _load_json(path_value: str | Path) -> dict[str, Any]:
	path = resolve(path_value)
	if not path.exists():
		raise FileNotFoundError(path)
	return json.loads(path.read_text(encoding="utf-8"))


def _rows_select(rows: dict[str, torch.Tensor], indices: torch.Tensor) -> dict[str, torch.Tensor]:
	return {key: value[indices] for key, value in rows.items()}


def _rows_fixed(rows: dict[str, torch.Tensor], size: int, seed: int) -> dict[str, torch.Tensor]:
	count = min(int(size), int(rows["obs"].shape[0]))
	generator = torch.Generator().manual_seed(int(seed))
	indices = torch.randperm(int(rows["obs"].shape[0]), generator=generator)[:count]
	return _rows_select(rows, indices)


def _episode_split_rows(rows: dict[str, torch.Tensor], seed: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
	episodes = rows["episode"].detach().long().reshape(-1)
	train = torch.zeros(episodes.shape[0], dtype=torch.bool)
	for episode in torch.unique(episodes, sorted=True):
		if (37 * int(episode.item()) + int(seed)) % 10 < 7:
			train |= episodes == episode
	val = ~train
	if not train.any() or not val.any():
		raise RuntimeError("Episode-wise split is degenerate.")
	return (
		_rows_select(rows, torch.nonzero(train, as_tuple=False).reshape(-1)),
		_rows_select(rows, torch.nonzero(val, as_tuple=False).reshape(-1)),
	)


def _replay_rows(path: Path, seed: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
	td, _ = prior_audit._load_snapshot(path)
	valid = prior_audit._finite_transition_mask(td)
	indices = torch.nonzero(valid, as_tuple=False).reshape(-1)
	if indices.numel() == 0:
		raise RuntimeError(f"Replay has no finite policy transitions: {path}")
	phase = prior_audit._phase_labels_from_replay(td)
	rows = {
		"obs": td["obs"][indices].detach().float(),
		"action": td["action"][indices].detach().float(),
		"task": td["task"][indices].detach().float(),
		"episode": td["episode"][indices].detach().long(),
		"phase": phase[indices].detach().long(),
	}
	train, val = _episode_split_rows(rows, seed)
	return train, val, prior_audit._unique_task_vec(td)


def _quality_rollout_rows(args: argparse.Namespace) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
	"""Use multitask rollout labels only; direct trajectories never enter supervision."""
	banks, _ = attribution._load_rollout_banks(args)
	columns: dict[str, list[torch.Tensor]] = {key: [] for key in ("obs", "action", "task", "episode", "phase", "outcome")}
	for region_index, region in enumerate(attribution.REGIONS):
		bank = banks[f"multitask_{region}"]
		td = bank["td"]
		valid = (
			torch.isfinite(td["obs"].detach().float()).all(dim=-1)
			& torch.isfinite(td["action"].detach().float()).all(dim=-1)
			& torch.isfinite(td["task"].detach().float()).all(dim=-1)
		)
		indices = torch.nonzero(valid, as_tuple=False).reshape(-1)
		if indices.numel() == 0:
			continue
		columns["obs"].append(td["obs"][indices].detach().float())
		columns["action"].append(td["action"][indices].detach().float())
		columns["task"].append(td["task"][indices].detach().float())
		# Region offsets preserve episode-disjoint splits across separate rollout files.
		columns["episode"].append(td["episode"][indices].detach().long() + region_index * 1_000_000)
		columns["phase"].append(bank["phase"][indices].detach().long())
		columns["outcome"].append(bank["outcome"][indices].detach().long())
	if not columns["obs"]:
		raise RuntimeError("Phase 3.3 multitask rollouts contain no finite 00186 transitions.")
	rows = {key: torch.cat(value, dim=0) for key, value in columns.items()}
	return _episode_split_rows(rows, int(args.seed) + 701)


@torch.no_grad()
def _mean_trajectory(model, obs: torch.Tensor, task: torch.Tensor, horizon: int, first_action: torch.Tensor | None = None):
	z = model.encode(obs, task)
	actions = []
	for step in range(int(horizon)):
		if step == 0 and first_action is not None:
			action = first_action
		else:
			_, info = model.pi(z, task)
			action = info["mean"]
		actions.append(action)
		z = model.next(z, action, task)
	return torch.stack(actions, dim=1)


@torch.no_grad()
def _frozen_q_advantage_weights(base_model, cfg, rows: dict[str, torch.Tensor], device: torch.device, args, seed: int):
	values = []
	for start in range(0, int(rows["obs"].shape[0]), int(args.elite_batch_size)):
		stop = min(start + int(args.elite_batch_size), int(rows["obs"].shape[0]))
		obs = rows["obs"][start:stop].to(device)
		action = rows["action"][start:stop].to(device)
		task = rows["task"][start:stop].to(device)
		with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
			torch.manual_seed(int(seed) + start)
			if device.type == "cuda":
				torch.cuda.manual_seed_all(int(seed) + start)
			z = base_model.encode(obs, task)
			_, info = base_model.pi(z, task)
			q_action = two_hot_scalar(base_model.Q(z, action, task, return_type="all"), cfg).mean(dim=0).reshape(-1)
			q_policy = two_hot_scalar(base_model.Q(z, info["mean"], task, return_type="all"), cfg).mean(dim=0).reshape(-1)
			values.append((q_action - q_policy).detach().cpu())
	advantage = torch.cat(values)
	normalized = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1.0e-6)
	weights = torch.exp((normalized / float(args.advantage_temperature)).clamp(-float(args.weight_clip), float(args.weight_clip)))
	weights = weights / weights.mean().clamp_min(1.0e-6)
	return weights, summarize_tensor(advantage)


def _quality_weights(rows: dict[str, torch.Tensor], args) -> tuple[torch.Tensor, dict[str, int]]:
	phase = rows["phase"].long()
	outcome = rows["outcome"].long()
	contact_or_insertion = phase >= 1
	success = outcome == 0
	jam = outcome == 1
	failure = outcome == 2
	weights = torch.ones_like(phase, dtype=torch.float32)
	weights[contact_or_insertion & success] = float(args.quality_success_weight)
	weights[contact_or_insertion & jam] = float(args.quality_jam_weight)
	weights[contact_or_insertion & failure] = float(args.quality_failure_weight)
	weights = weights / weights.mean().clamp_min(1.0e-6)
	return weights, {
		"contact_or_insertion_success": int((contact_or_insertion & success).sum().item()),
		"contact_or_insertion_jam": int((contact_or_insertion & jam).sum().item()),
		"contact_or_insertion_failure": int((contact_or_insertion & failure).sum().item()),
		"pre_contact": int((~contact_or_insertion).sum().item()),
	}


@torch.no_grad()
def _build_elite_cache(base_model, cfg, rows: dict[str, torch.Tensor], device: torch.device, args, seed: int):
	"""Create frozen multitask-MPPI elite targets without any direct-model input."""
	horizon = int(args.horizon)
	count = int(args.num_candidates)
	policy_count = int(args.num_policy_candidates)
	if policy_count <= 0 or policy_count >= count:
		raise ValueError("num_policy_candidates must be in [1, num_candidates - 1].")
	outputs: dict[str, list[torch.Tensor]] = {key: [] for key in ("elite_action", "elite_floor", "elite_score_mean", "prefix_score", "score_gain")}
	for start in range(0, int(rows["obs"].shape[0]), int(args.elite_batch_size)):
		stop = min(start + int(args.elite_batch_size), int(rows["obs"].shape[0]))
		obs = rows["obs"][start:stop].to(device)
		action = rows["action"][start:stop].to(device)
		task = rows["task"][start:stop].to(device)
		task_vec = task[0]
		if not torch.allclose(task, task_vec.expand_as(task)):
			raise RuntimeError("Elite distillation cache expects one 00186 task vector per source pool.")
		generator = torch.Generator(device=device).manual_seed(int(seed) + start)
		with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
			torch.manual_seed(int(seed) + start)
			if device.type == "cuda":
				torch.cuda.manual_seed_all(int(seed) + start)
			policy_eps = torch.randn(obs.shape[0], horizon, policy_count, action.shape[-1], generator=generator, device=device)
			policy = attribution._policy_proposals(
				base_model, obs, task_vec, policy_eps, horizon=horizon, num_candidates=policy_count,
			)["actions"]
			gaussian = (float(cfg.max_std) * torch.randn(
				obs.shape[0], horizon, count - policy_count, action.shape[-1], generator=generator, device=device,
			)).clamp(-1.0, 1.0)
			candidates = torch.cat((policy, gaussian), dim=2)
			scores = attribution._score_candidates(base_model, cfg, obs, task_vec, candidates)["total"]
			top_score, top_index = torch.topk(scores, k=int(args.num_elites), dim=1)
			top_actions = candidates[:, 0].gather(1, top_index.unsqueeze(-1).expand(-1, -1, action.shape[-1]))
			weights = torch.softmax(float(cfg.temperature) * (top_score - top_score.max(dim=1, keepdim=True).values), dim=1)
			elite_action = (weights.unsqueeze(-1) * top_actions).sum(dim=1)
			prefix = _mean_trajectory(base_model, obs, task, horizon, first_action=action)
			prefix_score = attribution._score_candidates(base_model, cfg, obs, task_vec, prefix.unsqueeze(2))["total"].reshape(-1)
			outputs["elite_action"].append(elite_action.detach().cpu())
			outputs["elite_floor"].append(top_score[:, -1].detach().cpu())
			outputs["elite_score_mean"].append(top_score.mean(dim=1).detach().cpu())
			outputs["prefix_score"].append(prefix_score.detach().cpu())
			outputs["score_gain"].append((top_score.mean(dim=1) - prefix_score).detach().cpu())
	return {key: torch.cat(value) for key, value in outputs.items()}


def _attach_cache(rows: dict[str, torch.Tensor], cache: dict[str, torch.Tensor], **extra: torch.Tensor) -> dict[str, torch.Tensor]:
	if any(int(value.shape[0]) != int(rows["obs"].shape[0]) for value in cache.values()):
		raise RuntimeError("Elite cache rows do not match supervision rows.")
	return {**rows, **cache, **extra}


def _sample_batch(rows: dict[str, torch.Tensor], size: int, generator: torch.Generator):
	indices = torch.randint(int(rows["obs"].shape[0]), (int(size),), generator=generator)
	return _rows_select(rows, indices)


def _per_row_bc(model, batch, target: torch.Tensor, device: torch.device, seed: int):
	obs = batch["obs"].to(device)
	task = batch["task"].to(device)
	target = target.to(device)
	with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
		torch.manual_seed(int(seed))
		if device.type == "cuda":
			torch.cuda.manual_seed_all(int(seed))
		z = model.encode(obs, task)
		pi_action, _ = model.pi(z, task)
		return td_math.masked_bc_per_timestep(
			pi_action.unsqueeze(0), target.unsqueeze(0), task.unsqueeze(0), model._action_masks,
		).squeeze(0)


def _objective_loss(model, batch, method: str, device: torch.device, seed: int) -> torch.Tensor:
	if method in ("mppi_elite_distillation", "elite_plus_contact_jam_weighting"):
		target = batch["elite_action"]
	else:
		target = batch["action"]
	per_row = _per_row_bc(model, batch, target, device, seed)
	if method == "advantage_weighted_bc":
		weights = batch["advantage_weight"].to(device)
	elif method in ("contact_jam_quality_weighted_bc", "elite_plus_contact_jam_weighting"):
		weights = batch["quality_weight"].to(device)
	else:
		weights = torch.ones_like(per_row)
	return (weights * per_row).sum() / weights.sum().clamp_min(1.0e-6)


def _evaluate_objective(model, rows: dict[str, torch.Tensor], method: str, device: torch.device, args, seed: int) -> float:
	values = []
	for start in range(0, int(rows["obs"].shape[0]), int(args.eval_batch_size)):
		stop = min(start + int(args.eval_batch_size), int(rows["obs"].shape[0]))
		batch = {key: value[start:stop] for key, value in rows.items()}
		with torch.no_grad():
			values.append(float(_objective_loss(model, batch, method, device, int(seed) + start).item()))
	return float(sum(values) / len(values))


def _flat_policy_device(model) -> torch.Tensor:
	return torch.cat([parameter.detach().reshape(-1) for parameter in model._pi.parameters()])


@torch.no_grad()
def _project_policy_delta(model, base_policy_device: torch.Tensor, budget: float) -> None:
	current = _flat_policy_device(model)
	delta = current - base_policy_device
	norm = delta.norm()
	if norm <= float(budget):
		return
	projected = base_policy_device + delta * (float(budget) / float(norm.item()))
	offset = 0
	for parameter in model._pi.parameters():
		count = parameter.numel()
		parameter.copy_(projected[offset:offset + count].reshape_as(parameter))
		offset += count


@torch.no_grad()
def _elite_score_coverage(model, base_model, cfg, eval_rows: dict[str, torch.Tensor], elite_cache: dict[str, torch.Tensor], device: torch.device, args):
	values = []
	for start in range(0, int(eval_rows["obs"].shape[0]), int(args.elite_batch_size)):
		stop = min(start + int(args.elite_batch_size), int(eval_rows["obs"].shape[0]))
		obs = eval_rows["obs"][start:stop].to(device)
		task = eval_rows["task"][start:stop].to(device)
		trajectory = _mean_trajectory(model, obs, task, int(args.horizon))
		score = attribution._score_candidates(base_model, cfg, obs, task[0], trajectory.unsqueeze(2))["total"].reshape(-1)
		floor = elite_cache["elite_floor"][start:stop].to(device)
		mean_elite = elite_cache["elite_score_mean"][start:stop].to(device)
		values.append(torch.stack((score >= floor, mean_elite - score), dim=1).detach().cpu())
	packed = torch.cat(values, dim=0)
	return {
		"coverage_fraction": float(packed[:, 0].mean().item()),
		"mean_elite_score_gap": float(packed[:, 1].mean().item()),
	}


def _old_task_rows(replay_paths: OrderedDict[str, Path], args) -> OrderedDict[str, dict[str, torch.Tensor]]:
	result = OrderedDict()
	for index, task_id in enumerate(("01125", "00256")):
		_, val, _ = _replay_rows(replay_paths[task_id], int(args.seed) + 97 * index)
		result[task_id] = _rows_fixed(val, int(args.old_task_val_rows), int(args.seed) + 800 + index)
	return result


def _stage_metrics(model, base_model, multitask_cfg, direct_model, direct_cfg, objective_rows, method, eval_rows, eval_elite, old_rows, base_policy, device, args, updates: int):
	train_loss = _evaluate_objective(model, objective_rows["train"], method, device, args, int(args.seed) + 100)
	val_loss = _evaluate_objective(model, objective_rows["val"], method, device, args, int(args.seed) + 200)
	state_bank = {"obs": eval_rows["obs"], "task_vec": eval_rows["task"][0]}
	action_l2 = mitigation._action_l2_to_direct(direct_model, model, state_bank, device)
	proposal_regret = mitigation._proposal_regret(direct_model, direct_cfg, model, state_bank, device, args)
	coverage = _elite_score_coverage(model, base_model, multitask_cfg, eval_rows, eval_elite, device, args)
	drift = mitigation._old_task_drifts(base_model, model, old_rows, device)
	return {
		"updates": int(updates),
		"train_objective_loss": train_loss,
		"val_objective_loss": val_loss,
		"heldout_contact_jam_action_l2": action_l2,
		"heldout_proposal_regret": proposal_regret,
		"elite_score_coverage": coverage,
		"old_task_action_drift": drift,
		"policy_parameter_delta_l2": float((_flat_policy_device(model) - base_policy).norm().item()),
	}


def _reductions(stage: dict[str, Any], baseline: dict[str, Any]):
	return {
		"contact_jam_action_l2_reduction_fraction": 1.0 - float(stage["heldout_contact_jam_action_l2"]["mean"]) / max(float(baseline["heldout_contact_jam_action_l2"]["mean"]), 1.0e-8),
		"proposal_regret_reduction_fraction": 1.0 - float(stage["heldout_proposal_regret"]["mean"]) / max(float(baseline["heldout_proposal_regret"]["mean"]), 1.0e-8),
	}


def _passes(stage: dict[str, Any], args) -> bool:
	return (
		float(stage["reductions"]["proposal_regret_reduction_fraction"]) >= float(args.min_proposal_regret_improvement)
		and float(stage["reductions"]["contact_jam_action_l2_reduction_fraction"]) > 0.0
		and max(float(item["mean"]) for item in stage["old_task_action_drift"].values()) <= float(args.max_old_action_drift)
	)


def _classify(methods: dict[str, Any], args) -> dict[str, Any]:
	labels = {
		"advantage_weighted_bc": "ADVANTAGE_WEIGHTED_BC_EFFECTIVE",
		"mppi_elite_distillation": "ELITE_DISTILLATION_EFFECTIVE",
		"contact_jam_quality_weighted_bc": "QUALITY_WEIGHTED_OBJECTIVE_EFFECTIVE",
		"elite_plus_contact_jam_weighting": "COMBINED_OBJECTIVE_EFFECTIVE",
	}
	for method in ("elite_plus_contact_jam_weighting", "mppi_elite_distillation", "advantage_weighted_bc", "contact_jam_quality_weighted_bc"):
		best = max(methods[method]["stages"].values(), key=lambda item: item["reductions"]["proposal_regret_reduction_fraction"])
		if _passes(best, args):
			return {
				"classification": labels[method],
				"method": method,
				"updates": int(best["updates"]),
				"reason": "The objective cleared held-out proposal-regret, action-L2, and old-task drift gates using frozen multitask targets only.",
			}
	strongest = max(
		(stage for report in methods.values() for stage in report["stages"].values()),
		key=lambda item: item["reductions"]["proposal_regret_reduction_fraction"],
	)
	strongest_name = next(
		name for name, report in methods.items()
		if any(stage is strongest for stage in report["stages"].values())
	)
	return {
		"classification": "NO_POLICY_OBJECTIVE_RESCUE",
		"method": None,
		"updates": None,
		"reason": (
			"No frozen-world-model policy objective improved held-out proposal regret by 20 percent while also improving "
			"contact/jam action L2 and preserving old-task policy actions. The strongest unsafe signal is "
			f"{strongest_name} at {strongest['updates']} updates: proposal regret reduction="
			f"{strongest['reductions']['proposal_regret_reduction_fraction']:.3f}, but max old-task drift="
			f"{max(float(item['mean']) for item in strongest['old_task_action_drift'].values()):.3f}."
		),
		"strongest_unsafe_signal": {
			"method": strongest_name,
			"updates": int(strongest["updates"]),
			"proposal_regret_reduction_fraction": float(strongest["reductions"]["proposal_regret_reduction_fraction"]),
			"contact_jam_action_l2_reduction_fraction": float(strongest["reductions"]["contact_jam_action_l2_reduction_fraction"]),
			"max_old_task_action_drift": max(float(item["mean"]) for item in strongest["old_task_action_drift"].values()),
		},
	}


def _markdown(report: dict[str, Any]) -> str:
	classification = report["classification"]
	lines = [
		"# SRSA Phase 3.8 Proposal-Quality Objective Audit",
		"",
		"本报告冻结 encoder、dynamics、reward、Q、task_context 和 MPPI，仅在内存 policy-prior clone 上做目标函数比较。direct checkpoint 只用于评估参考，绝不参与 target 构造。",
		"",
		f"Status: `{report['status']}`",
		f"Final classification: `{classification['classification']}`",
		"",
		"## Conclusion",
		"",
		classification["reason"],
		"",
		"## Held-Out Objective Results",
		"",
		"正值 reduction 表示比 update=0 更好；proposal regret 和 action-L2 均以 direct checkpoint 作为只读评估参考。",
		"",
		"| Objective | Updates | Train loss | Val loss | Contact/jam L2 reduction | Proposal-regret reduction | Elite coverage | Elite gap | 01125 drift | 00256 drift | Param delta |",
		"| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
	]
	for name, item in report["methods"].items():
		for stage in item["stages"].values():
			lines.append(
				f"| `{name}` | {stage['updates']} | {stage['train_objective_loss']:.6f} | {stage['val_objective_loss']:.6f} | "
				f"{stage['reductions']['contact_jam_action_l2_reduction_fraction']:+.3f} | "
				f"{stage['reductions']['proposal_regret_reduction_fraction']:+.3f} | "
				f"{stage['elite_score_coverage']['coverage_fraction']:.3f} | {stage['elite_score_coverage']['mean_elite_score_gap']:.3f} | "
				f"{stage['old_task_action_drift']['01125']['mean']:.4f} | {stage['old_task_action_drift']['00256']['mean']:.4f} | "
				f"{stage['policy_parameter_delta_l2']:.4f} |"
			)
	lines.extend([
		"",
		"## Frozen Elite Target Coverage",
		"",
		"| Source | Rows | Positive elite-score gain vs replay/rollout prefix | Mean elite-score gain |",
		"| --- | ---: | ---: | ---: |",
	])
	for name, item in report["target_sources"].items():
		lines.append(
			f"| `{name}` | {item['rows']} | {item['positive_gain_fraction']:.3f} | {item['mean_score_gain']:.3f} |"
		)
	lines.extend([
		"",
		"## Protocol",
		"",
		f"- 00186 replay and multitask rollout supervision are each split by episode. Replay split: `{report['split']['replay_train_rows']}` train / `{report['split']['replay_val_rows']}` validation rows; rollout quality split: `{report['split']['quality_train_rows']}` / `{report['split']['quality_val_rows']}` rows.",
		f"- Update checkpoints: `{report['update_checkpoints']}`. All objectives use Adam lr=`{report['policy_lr']}`, batch=`{report['batch_size']}`, and the same policy delta budget `L2 <= {report['policy_delta_budget']}`.",
		f"- Frozen MPPI target: `{report['elite_target']['num_policy_candidates']}` policy + `{report['elite_target']['num_gaussian_candidates']}` Gaussian candidates, top-`{report['elite_target']['num_elites']}` scorer elites, horizon=`{report['elite_target']['horizon']}`.",
		"- Quality weights use only Phase 3.3 multitask rollout labels: contact/insertion success is upweighted; jam and failure are downweighted. No direct rollout/action is used for supervision.",
		"- Original checkpoint policy parameters are compared bytewise after the audit; no checkpoint, replay, sampler, or non-policy module is written.",
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
	base_policy_device = _flat_policy_device(base_model).clone()
	policy_lr = float(args.policy_lr) if args.policy_lr is not None else float(cfg.lr)

	replay_train_all, replay_val_all, task_vec = _replay_rows(replay_paths["00186"], int(args.seed) + 1)
	quality_train_all, quality_val_all = _quality_rollout_rows(args)
	replay_train = _rows_fixed(replay_train_all, int(args.replay_train_rows), int(args.seed) + 10)
	replay_val = _rows_fixed(replay_val_all, int(args.replay_val_rows), int(args.seed) + 11)
	quality_train = _rows_fixed(quality_train_all, int(args.quality_train_rows), int(args.seed) + 12)
	quality_val = _rows_fixed(quality_val_all, int(args.quality_val_rows), int(args.seed) + 13)
	quality_train_weight, quality_train_counts = _quality_weights(quality_train, args)
	quality_val_weight, quality_val_counts = _quality_weights(quality_val, args)
	replay_train_advantage, replay_train_advantage_stats = _frozen_q_advantage_weights(base_model, cfg, replay_train, device, args, int(args.seed) + 300)
	replay_val_advantage, replay_val_advantage_stats = _frozen_q_advantage_weights(base_model, cfg, replay_val, device, args, int(args.seed) + 400)

	replay_train_elite = _build_elite_cache(base_model, cfg, replay_train, device, args, int(args.seed) + 500)
	replay_val_elite = _build_elite_cache(base_model, cfg, replay_val, device, args, int(args.seed) + 600)
	quality_train_elite = _build_elite_cache(base_model, cfg, quality_train, device, args, int(args.seed) + 700)
	quality_val_elite = _build_elite_cache(base_model, cfg, quality_val, device, args, int(args.seed) + 800)
	replay_train = _attach_cache(replay_train, replay_train_elite, advantage_weight=replay_train_advantage)
	replay_val = _attach_cache(replay_val, replay_val_elite, advantage_weight=replay_val_advantage)
	quality_train = _attach_cache(quality_train, quality_train_elite, quality_weight=quality_train_weight)
	quality_val = _attach_cache(quality_val, quality_val_elite, quality_weight=quality_val_weight)

	contact_jam = (quality_val["phase"] == 1) & (quality_val["outcome"] == 1)
	contact_jam_indices = torch.nonzero(contact_jam, as_tuple=False).reshape(-1)
	if contact_jam_indices.numel() == 0:
		raise RuntimeError("Held-out multitask rollout split has no contact/jam states.")
	eval_rows = _rows_fixed(_rows_select(quality_val, contact_jam_indices), int(args.eval_contact_jam_rows), int(args.seed) + 900)
	eval_elite = _build_elite_cache(base_model, cfg, eval_rows, device, args, int(args.seed) + 1000)
	old_rows = _old_task_rows(replay_paths, args)

	objective_rows = {
		"replay_action_bc": {"train": replay_train, "val": replay_val},
		"advantage_weighted_bc": {"train": replay_train, "val": replay_val},
		"mppi_elite_distillation": {"train": replay_train, "val": replay_val},
		"contact_jam_quality_weighted_bc": {"train": quality_train, "val": quality_val},
		"elite_plus_contact_jam_weighting": {"train": quality_train, "val": quality_val},
	}
	methods = OrderedDict()
	checkpoints_updates = tuple(sorted(set((100, 500, 2000))))
	for method_index, method in enumerate(METHODS):
		model = copy.deepcopy(base_model)
		phase37._freeze_world_model_except_policy(model)
		optimizer = torch.optim.Adam(phase37._policy_parameters(model), lr=policy_lr)
		generator = torch.Generator().manual_seed(int(args.seed) + 10_000 * method_index)
		base_policy = _flat_policy_device(model).clone()
		baseline = _stage_metrics(
			model, base_model, cfg, direct_model, direct_cfg, objective_rows[method], method, eval_rows, eval_elite,
			old_rows, base_policy, device, args, 0,
		)
		stages = OrderedDict()
		for update in range(1, max(checkpoints_updates) + 1):
			batch = _sample_batch(objective_rows[method]["train"], int(args.batch_size), generator)
			optimizer.zero_grad(set_to_none=True)
			loss = _objective_loss(model, batch, method, device, int(args.seed) + method_index * 100_000 + update)
			loss.backward()
			optimizer.step()
			_project_policy_delta(model, base_policy, float(args.policy_delta_budget))
			if update in checkpoints_updates:
				stage = _stage_metrics(
					model, base_model, cfg, direct_model, direct_cfg, objective_rows[method], method, eval_rows, eval_elite,
					old_rows, base_policy, device, args, update,
				)
				stage["reductions"] = _reductions(stage, baseline)
				stages[str(update)] = stage
		methods[method] = {"baseline": baseline, "stages": stages}
		del model, optimizer
		if device.type == "cuda":
			torch.cuda.empty_cache()

	if not torch.equal(base_policy_cpu, phase37._flatten_policy(base_model)):
		raise RuntimeError("Base checkpoint policy prior changed during Phase 3.8 audit.")
	classification = _classify(methods, args)
	status = "PASS" if classification["classification"] != "NO_POLICY_OBJECTIVE_RESCUE" else "WARNING"
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
		"batch_size": int(args.batch_size),
		"policy_delta_budget": float(args.policy_delta_budget),
		"update_checkpoints": list(checkpoints_updates),
		"elite_target": {
			"num_candidates": int(args.num_candidates),
			"num_policy_candidates": int(args.num_policy_candidates),
			"num_gaussian_candidates": int(args.num_candidates) - int(args.num_policy_candidates),
			"num_elites": int(args.num_elites),
			"horizon": int(args.horizon),
			"scorer": "frozen_multitask_world_model",
		},
		"split": {
			"replay_train_rows": int(replay_train["obs"].shape[0]),
			"replay_val_rows": int(replay_val["obs"].shape[0]),
			"quality_train_rows": int(quality_train["obs"].shape[0]),
			"quality_val_rows": int(quality_val["obs"].shape[0]),
			"heldout_contact_jam_rows": int(eval_rows["obs"].shape[0]),
			"quality_train_label_counts": quality_train_counts,
			"quality_val_label_counts": quality_val_counts,
		},
		"advantage": {"train": replay_train_advantage_stats, "val": replay_val_advantage_stats},
		"target_sources": {
			"replay": {
				"rows": int(replay_val["obs"].shape[0]),
				"positive_gain_fraction": float((replay_val_elite["score_gain"] > 0).float().mean().item()),
				"mean_score_gain": float(replay_val_elite["score_gain"].mean().item()),
			},
			"multitask_rollout_quality": {
				"rows": int(quality_val["obs"].shape[0]),
				"positive_gain_fraction": float((quality_val_elite["score_gain"] > 0).float().mean().item()),
				"mean_score_gain": float(quality_val_elite["score_gain"].mean().item()),
			},
			"heldout_contact_jam": {
				"rows": int(eval_rows["obs"].shape[0]),
				"positive_gain_fraction": float((eval_elite["score_gain"] > 0).float().mean().item()),
				"mean_score_gain": float(eval_elite["score_gain"].mean().item()),
			},
		},
		"gates": {
			"min_proposal_regret_improvement": float(args.min_proposal_regret_improvement),
			"max_old_action_drift": float(args.max_old_action_drift),
		},
		"methods": methods,
		"limitations": [
			"The direct checkpoint is evaluation-only; it supplies neither action targets nor loss weights.",
			"Quality labels come from Phase 3.3 multitask rollouts. Replay snapshots have no exact jam/success labels and are not silently treated as if they did.",
			"Frozen scorer elite targets test objective alignment, not environment-return ground truth or closed-loop success.",
			"No original checkpoint, replay, sampler, MPPI implementation, or non-policy module is written or modified.",
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
	parser.add_argument("--batch-size", type=int, default=128)
	parser.add_argument("--eval-batch-size", type=int, default=1024)
	parser.add_argument("--elite-batch-size", type=int, default=64)
	parser.add_argument("--proposal-batch-size", type=int, default=32)
	parser.add_argument("--replay-train-rows", type=int, default=4096)
	parser.add_argument("--replay-val-rows", type=int, default=2048)
	parser.add_argument("--quality-train-rows", type=int, default=4096)
	parser.add_argument("--quality-val-rows", type=int, default=2048)
	parser.add_argument("--old-task-val-rows", type=int, default=2048)
	parser.add_argument("--eval-contact-jam-rows", type=int, default=256)
	parser.add_argument("--policy-lr", type=float, default=None)
	parser.add_argument("--policy-delta-budget", type=float, default=3.5)
	parser.add_argument("--advantage-temperature", type=float, default=1.0)
	parser.add_argument("--weight-clip", type=float, default=2.0)
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
