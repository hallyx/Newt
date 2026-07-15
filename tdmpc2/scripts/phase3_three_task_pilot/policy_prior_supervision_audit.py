#!/usr/bin/env python3
"""Read-only Phase 3.5 audit of 00186 policy-prior supervision and gradients."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_AUDIT_DIR = SCRIPT_DIR.parent / "model_task_sensitivity"
if str(MODEL_AUDIT_DIR) not in sys.path:
	sys.path.insert(0, str(MODEL_AUDIT_DIR))

import planner_action_attribution_diagnosis as attribution  # noqa: E402
from _common import (  # noqa: E402
	condition_batch,
	resolve,
	summarize_tensor,
	tensor_to_list,
	td_math,
	tvsr,
	write_json,
	write_text,
)


DEFAULT_PHASE32_DIAGNOSIS = (
	"reports/phase3_three_task_pilot/phase3_2_diagnosis/"
	"standalone_vs_multitask_diagnosis.json"
)
DEFAULT_ROLLOUT_ROOT = "reports/phase3_three_task_pilot/phase3_3_rollouts"
DEFAULT_OUTPUT_JSON = "reports/phase3_three_task_pilot/phase3_5_policy_prior_audit.json"
DEFAULT_OUTPUT_MD = "reports/phase3_three_task_pilot/phase3_5_policy_prior_audit.md"
DEFAULT_REPLAYS = OrderedDict([
	(
		"01125",
		"logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_replay_from_01125/"
		"20260615_202326_launcher/replay/01125.pt",
	),
	(
		"00256",
		"logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_taskctx_repair_01125_00256/"
		"20260708_taskctx_repair_phase2_launcher/replay/00256.pt",
	),
])
TASKS = ("01125", "00256", "00186")
REGIONS = attribution.REGIONS
PHASES = attribution.PHASES
OUTCOMES = attribution.OUTCOMES


def _load_json(path_value: str | Path) -> dict[str, Any]:
	path = resolve(path_value)
	if not path.exists():
		raise FileNotFoundError(path)
	return json.loads(path.read_text(encoding="utf-8"))


def _load_tensordict(path: Path):
	obj = torch.load(path, map_location="cpu", weights_only=False)
	return obj.get("data", obj) if isinstance(obj, dict) else obj


def _load_snapshot(path: Path):
	obj = torch.load(path, map_location="cpu", weights_only=False)
	if not isinstance(obj, dict) or "data" not in obj:
		raise RuntimeError(f"Replay snapshot {path} is not a payload with a data field.")
	return obj["data"], obj.get("metadata", {})


def _unique_task_vec(td) -> torch.Tensor:
	if "task" not in td.keys():
		raise KeyError("Replay snapshot has no task tensor.")
	task = td["task"].detach().float().reshape(-1, td["task"].shape[-1])
	if task.shape[-1] != 6:
		raise RuntimeError(f"Expected a 6D task vector, got {tuple(task.shape)}.")
	unique = torch.unique(task, dim=0)
	if unique.shape[0] != 1:
		raise RuntimeError(
			"Phase 3.5 expects a per-task replay snapshot with exactly one task vector; "
			f"found {int(unique.shape[0])}."
		)
	return unique[0]


def _phase_labels_from_replay(td) -> torch.Tensor:
	"""Match the Phase 3.3 TCP-z phase heuristic without relying on stored step ids."""
	episodes = td["episode"].detach().long().reshape(-1)
	obs = td["obs"].detach().float()
	phase = torch.empty_like(episodes)
	for episode_id in torch.unique(episodes, sorted=True):
		idx = torch.nonzero(episodes == episode_id, as_tuple=False).reshape(-1)
		n = int(idx.numel())
		z = obs[idx, 2]
		downward = (z[0] - z).clamp_min(0.0)
		total = float(downward.max().item())
		if total >= 0.006:
			contact_hits = torch.nonzero(downward >= 0.25 * total, as_tuple=False).reshape(-1)
			insert_hits = torch.nonzero(downward >= 0.65 * total, as_tuple=False).reshape(-1)
			contact_at = int(contact_hits[0].item()) if contact_hits.numel() else max(1, int(0.27 * n))
			insert_at = int(insert_hits[0].item()) if insert_hits.numel() else max(contact_at + 1, int(0.60 * n))
		else:
			contact_at = max(1, int(0.27 * n))
			insert_at = max(contact_at + 1, int(0.60 * n))
		contact_at = min(max(contact_at, 1), max(1, n - 2))
		insert_at = min(max(insert_at, contact_at + 1), max(contact_at + 1, n - 1))
		phase[idx[:contact_at]] = 0
		phase[idx[contact_at:insert_at]] = 1
		phase[idx[insert_at:]] = 2
	return phase


def _finite_transition_mask(td) -> torch.Tensor:
	"""Replay snapshots retain terminal placeholders; exclude only non-finite supervision rows."""
	return (
		torch.isfinite(td["obs"].detach().float()).all(dim=-1)
		& torch.isfinite(td["action"].detach().float()).all(dim=-1)
		& torch.isfinite(td["task"].detach().float()).all(dim=-1)
	)


def _sample_rows(td, count: int, seed: int):
	valid_indices = torch.nonzero(_finite_transition_mask(td), as_tuple=False).reshape(-1)
	if valid_indices.numel() == 0:
		raise RuntimeError("Replay contains no finite obs/action/task transition for policy-prior supervision.")
	count = min(int(count), int(valid_indices.numel()))
	generator = torch.Generator().manual_seed(int(seed))
	indices = valid_indices[torch.randperm(int(valid_indices.numel()), generator=generator)[:count]]
	return {
		"obs": td["obs"][indices].detach().float(),
		"action": td["action"][indices].detach().float(),
		"task": td["task"][indices].detach().float(),
		"indices": indices,
		"finite_transition_count": int(valid_indices.numel()),
		"dropped_nonfinite_transition_count": int(td.batch_size[0] - valid_indices.numel()),
	}


@torch.no_grad()
def _policy_mean(model, obs: torch.Tensor, task: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
	z = model.encode(obs, task)
	_, info = model.pi(z, task)
	return info["mean"], info["log_std"].exp()


def _flatten_gradients(grads: tuple[torch.Tensor | None, ...], parameters) -> torch.Tensor:
	pieces = []
	for grad, parameter in zip(grads, parameters):
		pieces.append(torch.zeros_like(parameter).reshape(-1) if grad is None else grad.detach().reshape(-1))
	return torch.cat(pieces)


def _policy_prior_loss_and_gradient(model, obs, action, task, *, seed: int):
	"""Compute the exact masked policy-prior MSE term without an optimizer step."""
	parameters = tuple(parameter for parameter in model._pi.parameters() if parameter.requires_grad)
	devices = [obs.device] if obs.is_cuda else []
	with torch.random.fork_rng(devices=devices):
		torch.manual_seed(int(seed))
		if obs.is_cuda:
			torch.cuda.manual_seed_all(int(seed))
		z = model.encode(obs, task)
		pi_action, info = model.pi(z, task)
		loss_per_row = td_math.masked_bc_per_timestep(
			pi_action.unsqueeze(0), action.unsqueeze(0), task.unsqueeze(0), model._action_masks
		).squeeze(0)
		loss = loss_per_row.mean()
		grads = torch.autograd.grad(loss, parameters, allow_unused=True)
	gradient = _flatten_gradients(grads, parameters).cpu()
	return {
		"loss": float(loss.detach().item()),
		"loss_per_row": loss_per_row.detach().cpu(),
		"gradient": gradient,
		"mean_action": info["mean"].detach().cpu(),
	}


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
	denom = left.norm() * right.norm()
	return float((torch.dot(left, right) / denom.clamp_min(1.0e-12)).item())


def _gradient_report(model, replay_rows, device: torch.device, args) -> dict[str, Any]:
	per_task = OrderedDict()
	chunk_gradients: dict[str, list[torch.Tensor]] = defaultdict(list)
	chunk_losses: dict[str, list[float]] = defaultdict(list)
	for task_id in TASKS:
		rows = replay_rows[task_id]
		count = min(int(args.gradient_samples), int(rows["obs"].shape[0]))
		chunk_size = max(1, int(math.ceil(count / int(args.gradient_chunks))))
		for chunk_index, start in enumerate(range(0, count, chunk_size)):
			stop = min(start + chunk_size, count)
			item = _policy_prior_loss_and_gradient(
				model,
				rows["obs"][start:stop].to(device),
				rows["action"][start:stop].to(device),
				rows["task"][start:stop].to(device),
				# Match policy reparameterization noise across tasks within each chunk so
				# the cosine measures task gradients rather than independent sample noise.
				seed=int(args.seed) + chunk_index,
			)
			chunk_gradients[task_id].append(item["gradient"])
			chunk_losses[task_id].append(item["loss"])
		mean_gradient = torch.stack(chunk_gradients[task_id]).mean(dim=0)
		per_task[task_id] = {
			"policy_prior_loss": float(sum(chunk_losses[task_id]) / len(chunk_losses[task_id])),
			"policy_prior_gradient_norm": float(mean_gradient.norm().item()),
			"chunk_gradient_norm": summarize_tensor(torch.stack(chunk_gradients[task_id]).norm(dim=1)),
			"num_samples": count,
			"num_chunks": len(chunk_gradients[task_id]),
		}

	pairs = OrderedDict()
	for other in ("01125", "00256"):
		mean_left = torch.stack(chunk_gradients["00186"]).mean(dim=0)
		mean_right = torch.stack(chunk_gradients[other]).mean(dim=0)
		count = min(len(chunk_gradients["00186"]), len(chunk_gradients[other]))
		chunk_cosines = torch.tensor([
			_cosine(chunk_gradients["00186"][index], chunk_gradients[other][index])
			for index in range(count)
		])
		pairs[f"00186_vs_{other}"] = {
			"mean_gradient_cosine": _cosine(mean_left, mean_right),
			"chunk_gradient_cosine": summarize_tensor(chunk_cosines),
			"gradient_conflict_rate": float((chunk_cosines < 0.0).float().mean().item()),
			"strong_conflict_rate_cosine_lt_minus_0p05": float((chunk_cosines < -0.05).float().mean().item()),
		}
	return {
		"definition": (
			"masked_bc_per_timestep policy-prior MSE from TDMPC2.update_pi, differentiated only "
			"with respect to WorldModel._pi parameters; no optimizer or parameter mutation occurs."
		),
		"per_task": per_task,
		"pairs": pairs,
	}


def _rollout_success_centroids(banks) -> dict[tuple[str, str], torch.Tensor]:
	values: dict[tuple[str, str], list[torch.Tensor]] = defaultdict(list)
	values[("all", "all")] = []
	for name, bank in banks.items():
		if bank["model"] != "direct":
			continue
		td = bank["td"]
		for phase_id, phase_name in enumerate(PHASES):
			mask = (bank["phase"] == phase_id) & (bank["outcome"] == 0)
			if mask.any():
				actions = td["action"][mask].detach().float()
				values[(bank["region"], phase_name)].append(actions)
				values[("all", phase_name)].append(actions)
				values[("all", "all")].append(actions)
	return {
		key: torch.cat(item, dim=0).mean(dim=0)
		for key, item in values.items() if item
	}


def _rollout_action_rows(models, task_vec, banks, device: torch.device, args):
	"""Use multitask rollout states as the common state batch for action comparisons."""
	centroids = _rollout_success_centroids(banks)
	columns: dict[str, list[Any]] = defaultdict(list)
	alignment = OrderedDict()
	for region in REGIONS:
		multi = banks[f"multitask_{region}"]
		direct = banks[f"direct_{region}"]
		td = multi["td"]
		obs = td["obs"].detach().float()
		task = condition_batch(task_vec, (obs.shape[0],), device)
		direct_mean, _ = _policy_mean(models["direct"], obs.to(device), task)
		multi_mean, multi_std = _policy_mean(models["multitask"], obs.to(device), task)
		direct_obs = direct["td"]["obs"].detach().float()
		if not torch.equal(td["episode"], direct["td"]["episode"]) or not torch.equal(td["step_id"], direct["td"]["step_id"]):
			raise RuntimeError(f"Direct/multitask rollout indexing changed for region={region}; cannot align initial states.")
		state_l2 = torch.linalg.vector_norm(obs - direct_obs, dim=-1)
		exact = state_l2 < float(args.same_state_tolerance)
		alignment[region] = {
			"rows": int(obs.shape[0]),
			"same_state_rows": int(exact.sum().item()),
			"state_l2": summarize_tensor(state_l2),
			"reason_nonexact_rows_excluded_from_direct_mppi": (
				"The two policies diverge after the common initial state, so their later rollout states are physically different."
			),
		}
		for row in range(int(obs.shape[0])):
			phase_name = PHASES[int(multi["phase"][row].item())]
			outcome_name = OUTCOMES[int(multi["outcome"][row].item())]
			centroid = centroids.get((region, phase_name), centroids.get(("all", phase_name), centroids[("all", "all")]))
			columns["region"].append(region)
			columns["phase"].append(phase_name)
			columns["outcome"].append(outcome_name)
			columns["episode"].append(int(td["episode"][row].item()))
			columns["direct_policy"].append(direct_mean[row].detach().cpu())
			columns["multitask_policy"].append(multi_mean[row].detach().cpu())
			columns["multitask_policy_std"].append(multi_std[row].detach().cpu())
			columns["mppi_selected"].append(td["action"][row].detach().float().cpu())
			columns["success_action_reference"].append(centroid.detach().cpu())
			columns["direct_mppi_selected"].append(
				direct["td"]["action"][row].detach().float().cpu() if bool(exact[row].item()) else torch.full((3,), math.nan)
			)
			columns["same_state"].append(bool(exact[row].item()))
	return {
		"region": columns["region"],
		"phase": columns["phase"],
		"outcome": columns["outcome"],
		"episode": torch.tensor(columns["episode"], dtype=torch.long),
		"direct_policy": torch.stack(columns["direct_policy"]),
		"multitask_policy": torch.stack(columns["multitask_policy"]),
		"multitask_policy_std": torch.stack(columns["multitask_policy_std"]),
		"mppi_selected": torch.stack(columns["mppi_selected"]),
		"success_action_reference": torch.stack(columns["success_action_reference"]),
		"direct_mppi_selected": torch.stack(columns["direct_mppi_selected"]),
		"same_state": torch.tensor(columns["same_state"], dtype=torch.bool),
		"direct_success_action_centroids": {f"{key[0]}/{key[1]}": tensor_to_list(value) for key, value in centroids.items()},
		"alignment": alignment,
	}


def _label_mask(rows, *, region: str | None = None, phase: str | None = None, outcome: str | None = None):
	mask = torch.ones(len(rows["region"]), dtype=torch.bool)
	for key, expected in (("region", region), ("phase", phase), ("outcome", outcome)):
		if expected is not None:
			mask &= torch.tensor([value == expected for value in rows[key]], dtype=torch.bool)
	return mask


def _l2(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
	return torch.linalg.vector_norm(left - right, dim=-1)


def _action_metrics(rows, correction: torch.Tensor | None = None):
	multitask_policy = rows["multitask_policy"] if correction is None else correction
	values = {
		"direct_policy_vs_multitask_policy_l2": _l2(rows["direct_policy"], multitask_policy),
		"multitask_policy_vs_mppi_selected_l2": _l2(multitask_policy, rows["mppi_selected"]),
		"direct_policy_vs_mppi_selected_l2": _l2(rows["direct_policy"], rows["mppi_selected"]),
		"multitask_policy_vs_success_action_reference_l2": _l2(multitask_policy, rows["success_action_reference"]),
		"direct_policy_vs_success_action_reference_l2": _l2(rows["direct_policy"], rows["success_action_reference"]),
	}
	groups = OrderedDict()
	group_defs = [("all", None, None, None)]
	group_defs += [(f"phase/{name}", None, name, None) for name in PHASES]
	group_defs += [(f"outcome/{name}", None, None, name) for name in OUTCOMES]
	group_defs += [(f"region/{name}", name, None, None) for name in REGIONS]
	group_defs += [(f"{phase}/{outcome}", None, phase, outcome) for phase in PHASES for outcome in OUTCOMES]
	for name, region, phase, outcome in group_defs:
		mask = _label_mask(rows, region=region, phase=phase, outcome=outcome)
		if not mask.any():
			continue
		item = {"count": int(mask.sum().item())}
		for metric, value in values.items():
			item[metric] = summarize_tensor(value[mask])
		if bool(rows["same_state"][mask].any().item()):
			same = mask & rows["same_state"]
			item["direct_mppi_vs_multitask_mppi_l2_same_state_only"] = summarize_tensor(
				_l2(rows["direct_mppi_selected"][same], rows["mppi_selected"][same])
			)
		groups[name] = item
	return groups


def _episode_train_mask(rows, seed: int) -> torch.Tensor:
	values = []
	for region, episode in zip(rows["region"], rows["episode"].tolist()):
		stable_hash = sum(ord(char) for char in region) + 37 * int(episode) + int(seed)
		values.append((stable_hash % 10) < 7)
	mask = torch.tensor(values, dtype=torch.bool)
	if not mask.any() or bool(mask.all().item()):
		raise RuntimeError("Episode-wise correction split was degenerate.")
	return mask


def _fit_affine_correction(source: torch.Tensor, target: torch.Tensor, train_mask: torch.Tensor, ridge: float):
	"""Fit target ~= source @ W + b. This is an offline diagnostic, never checkpoint state."""
	x = torch.cat([source, torch.ones(source.shape[0], 1)], dim=1)
	x_train = x[train_mask]
	y_train = target[train_mask]
	regularizer = torch.eye(x_train.shape[1], dtype=x_train.dtype)
	regularizer[-1, -1] = 0.0
	weights = torch.linalg.solve(x_train.T @ x_train + float(ridge) * regularizer, x_train.T @ y_train)
	prediction = x @ weights
	return prediction, weights


def _correction_report(rows, args):
	train_mask = _episode_train_mask(rows, int(args.seed))
	test_mask = ~train_mask
	prediction, weights = _fit_affine_correction(
		rows["multitask_policy"], rows["direct_policy"], train_mask, float(args.correction_ridge)
	)
	baseline = _l2(rows["multitask_policy"], rows["direct_policy"])
	corrected = _l2(prediction, rows["direct_policy"])
	groups = OrderedDict()
	for name, phase, outcome in [("all", None, None)] + [
		(f"{phase}/{outcome}", phase, outcome) for phase in PHASES for outcome in OUTCOMES
	]:
		mask = _label_mask(rows, phase=phase, outcome=outcome) & test_mask
		if not mask.any():
			continue
		base_mean = float(baseline[mask].mean().item())
		corrected_mean = float(corrected[mask].mean().item())
		groups[name] = {
			"count": int(mask.sum().item()),
			"baseline_multitask_to_direct_policy_l2": summarize_tensor(baseline[mask]),
			"corrected_to_direct_policy_l2": summarize_tensor(corrected[mask]),
			"l2_reduction_fraction": 1.0 - corrected_mean / max(base_mean, 1.0e-8),
		}
	contact_jam = groups.get("contact/jam")
	return {
		"definition": "00186-only affine action residual fitted on episode-disjoint rows; it is not written into any checkpoint.",
		"target": "direct policy-prior mean evaluated on the same multitask rollout state",
		"train_rows": int(train_mask.sum().item()),
		"test_rows": int(test_mask.sum().item()),
		"ridge": float(args.correction_ridge),
		"weights": tensor_to_list(weights),
		"groups": groups,
		"contact_jam_l2_reduction_fraction": (
			float(contact_jam["l2_reduction_fraction"]) if contact_jam is not None else None
		),
		"prediction": prediction,
	}


def _replay_quality_report(replays, rollout_rows) -> dict[str, Any]:
	result = OrderedDict()
	for task_id, item in replays.items():
		td = item["td"]
		episodes = td["episode"].detach().long().reshape(-1)
		reward = td["reward"].detach().float().reshape(-1)
		phase = _phase_labels_from_replay(td)
		finite_mask = _finite_transition_mask(td)
		episode_returns = []
		for episode_id in torch.unique(episodes, sorted=True):
			idx = torch.nonzero(episodes == episode_id, as_tuple=False).reshape(-1)
			episode_returns.append((float(reward[idx].sum().item()), int(episode_id.item())))
		top_count = max(1, int(math.ceil(0.30 * len(episode_returns))))
		top_ids = {episode for _, episode in sorted(episode_returns, reverse=True)[:top_count]}
		top_mask = torch.tensor([int(value) in top_ids for value in episodes.tolist()], dtype=torch.bool)
		phase_report = OrderedDict()
		for phase_id, phase_name in enumerate(PHASES):
			mask = (phase == phase_id) & finite_mask
			if not mask.any():
				continue
			phase_report[phase_name] = {
				"transition_count": int(mask.sum().item()),
				"top_return_proxy_transition_fraction": float(top_mask[mask].float().mean().item()),
				"policy_target_action_variance": tensor_to_list(td["action"][mask].detach().float().var(dim=0, unbiased=False)),
			}
		result[task_id] = {
			"snapshot": str(item["path"]),
			"transition_count": int(td.batch_size[0]),
			"finite_policy_supervision_transition_count": int(finite_mask.sum().item()),
			"dropped_nonfinite_terminal_or_placeholder_count": int((~finite_mask).sum().item()),
			"episode_count": len(episode_returns),
			"successful_transition_fraction": "UNKNOWN_WITH_REASON: replay snapshots store no relaxed/strict/process success field.",
			"top_return_episode_proportion": float(top_count / len(episode_returns)),
			"top_return_definition": "highest 30 percent of replay episodes by undiscounted reward; proxy only, not success truth.",
			"contact_jam_success_action_coverage": (
				"UNKNOWN_WITH_REASON: replay has no episode quality labels; Phase 3.3 rollout coverage is reported separately."
			),
			"policy_target_action_variance": tensor_to_list(
				td["action"][finite_mask].detach().float().var(dim=0, unbiased=False)
			),
			"phase_action_coverage": phase_report,
		}

	coverage = OrderedDict()
	for phase in PHASES:
		for outcome in OUTCOMES:
			mask = _label_mask(rollout_rows, phase=phase, outcome=outcome)
			if not mask.any():
				continue
			coverage[f"{phase}/{outcome}"] = {
				"multitask_rollout_transition_count": int(mask.sum().item()),
				"mppi_selected_action_variance": tensor_to_list(
					rollout_rows["mppi_selected"][mask].var(dim=0, unbiased=False)
				),
				"direct_success_action_reference_available": bool(torch.isfinite(
					rollout_rows["success_action_reference"][mask]
				).all().item()),
			}
	return {
		"replay": result,
		"phase3_rollout_quality_coverage": coverage,
		"rollout_quality_source": "Phase 3.3 00186 direct/multitask eval rollouts with relaxed success and jam-proxy labels.",
	}


def _classify(gradients, replay_quality, correction):
	pairs = gradients["pairs"]
	conflict_rate = max(float(item["gradient_conflict_rate"]) for item in pairs.values())
	min_cosine = min(float(item["mean_gradient_cosine"]) for item in pairs.values())
	contact_jam_reduction = correction["contact_jam_l2_reduction_fraction"]
	all_reduction = float(correction["groups"]["all"]["l2_reduction_fraction"])
	correctable = max(
		all_reduction,
		float(contact_jam_reduction) if contact_jam_reduction is not None else -math.inf,
	) >= 0.40
	conflict = min_cosine < -0.05 or conflict_rate >= 0.50

	coverage = replay_quality["phase3_rollout_quality_coverage"]
	contact_success = int(coverage.get("contact/success", {}).get("multitask_rollout_transition_count", 0))
	insertion_success = int(coverage.get("insertion/success", {}).get("multitask_rollout_transition_count", 0))
	insufficient_success_targets = (contact_success + insertion_success) < 20
	if conflict:
		classification = "POLICY_GRADIENT_CONFLICT"
		if correctable:
			next_step = "Task-specific policy residual/distillation, updating only the policy prior, is justified after a separate implementation review."
		else:
			next_step = (
				"Audit a policy-prior-only task-balanced or conflict-mitigation objective first; the current "
				"contact/jam affine correction is too weak to justify residual distillation."
			)
	elif insufficient_success_targets:
		classification = "SUCCESS_TARGET_IMBALANCE"
		next_step = "Reweight existing successful/contact 00186 transitions before changing policy-prior structure."
	elif correctable:
		classification = "POLICY_MEAN_BIAS_CORRECTABLE"
		next_step = "Validate a task-specific policy residual/distillation ablation; do not alter reward/Q/dynamics."
	elif not conflict and all_reduction < 0.20 and not insufficient_success_targets:
		classification = "POLICY_CAPACITY_LIMIT"
		next_step = "Run a policy-prior capacity-only ablation after preserving the current task conditioning path."
	else:
		classification = "MIXED_OR_UNRESOLVED"
		next_step = "Keep the diagnosis read-only until a single mechanism dominates; do not resume acquisition or consolidation."
	return {
		"classification": classification,
		"conflict_detected": conflict,
		"correctable_mean_bias": correctable,
		"success_target_imbalance": insufficient_success_targets,
		"min_00186_gradient_cosine": min_cosine,
		"max_00186_gradient_conflict_rate": conflict_rate,
		"contact_jam_correction_reduction": contact_jam_reduction,
		"all_state_correction_reduction": all_reduction,
		"next_step": next_step,
	}


def _markdown(report: dict[str, Any]) -> str:
	classification = report["classification"]
	lines = [
		"# SRSA Phase 3.5 Policy Prior Supervision Audit",
		"",
		"本报告只做 checkpoint、replay 和既有 rollout 的离线检查；未训练主模型，未写入 checkpoint，未修改 MPPI、reward、Q、dynamics、task_context 或 sampler。",
		"",
		f"Status: `{report['status']}`",
		f"Final classification: `{classification['classification']}`",
		"",
		"## 结论",
		"",
		f"- 主分类：`{classification['classification']}`。",
		f"- 00186 与旧任务的最小 policy-prior gradient cosine：`{classification['min_00186_gradient_cosine']:.4f}`；最大 chunk conflict rate：`{classification['max_00186_gradient_conflict_rate']:.3f}`。",
		f"- 00186 affine correction 对 all-state / contact-jam 的 action-L2 降幅：`{classification['all_state_correction_reduction']:.3f}` / `{classification['contact_jam_correction_reduction']}`。",
		f"- 成功 target 是否不足：`{classification['success_target_imbalance']}`。",
		f"- 下一步：{classification['next_step']}",
		"",
		"## Policy Prior Loss And Gradients",
		"",
		"policy loss 使用 `TDMPC2.update_pi()` 中的 `masked_bc_per_timestep` policy-prior MSE；梯度仅对 `WorldModel._pi` 读取，使用 `autograd.grad`，不执行 optimizer step。",
		"",
		"| Task | Policy-prior loss | Gradient norm | Samples | Chunks |",
		"| --- | ---: | ---: | ---: | ---: |",
	]
	for task_id, item in report["gradients"]["per_task"].items():
		lines.append(
			f"| `{task_id}` | {item['policy_prior_loss']:.6f} | {item['policy_prior_gradient_norm']:.4f} | "
			f"{item['num_samples']} | {item['num_chunks']} |"
		)
	lines.extend([
		"",
		"| Pair | Mean gradient cosine | Chunk cosine mean | Conflict rate | Cosine < -0.05 |",
		"| --- | ---: | ---: | ---: | ---: |",
	])
	for name, item in report["gradients"]["pairs"].items():
		lines.append(
			f"| `{name}` | {item['mean_gradient_cosine']:.4f} | {item['chunk_gradient_cosine']['mean']:.4f} | "
			f"{item['gradient_conflict_rate']:.3f} | {item['strong_conflict_rate_cosine_lt_minus_0p05']:.3f} |"
		)
	lines.extend([
		"",
		"## Same-State Action Attribution",
		"",
		"逐状态比较固定在 multitask rollout 实际访问的 observation 上：direct/multitask policy prior 都在该同一 state 重新推理；`mppi_selected` 是该 multitask state 的真实 rollout action。direct 与 multitask 只在 episode 初始 state 相同，后续状态因控制分叉而不同，因此 direct MPPI action 只在严格同态 rows 上单列统计。`success_action_reference` 是同 region/phase 的 direct 成功轨迹 action centroid，不是伪造的后续逐状态配对。",
		"",
		"| Group | Count | Direct-vs-multitask pi L2 | Multitask pi-vs-MPPI L2 | Direct pi-vs-MPPI L2 | Multitask pi-vs-success reference L2 |",
		"| --- | ---: | ---: | ---: | ---: | ---: |",
	])
	for name in ["all"] + [f"phase/{phase}" for phase in PHASES] + [f"outcome/{outcome}" for outcome in OUTCOMES] + ["contact/jam", "insertion/jam"]:
		item = report["action_metrics"].get(name)
		if item is None:
			continue
		lines.append(
			f"| `{name}` | {item['count']} | {item['direct_policy_vs_multitask_policy_l2']['mean']:.4f} | "
			f"{item['multitask_policy_vs_mppi_selected_l2']['mean']:.4f} | "
			f"{item['direct_policy_vs_mppi_selected_l2']['mean']:.4f} | "
			f"{item['multitask_policy_vs_success_action_reference_l2']['mean']:.4f} |"
		)
	lines.extend([
		"",
		"## Offline 00186 Affine Correction",
		"",
		"只拟合评估 `multitask_policy_action -> direct_policy_action` 的 00186 affine residual；split 按 rollout episode，未写回模型。",
		"",
		"| Group | Test rows | Baseline L2 | Corrected L2 | Reduction |",
		"| --- | ---: | ---: | ---: | ---: |",
	])
	for name in ["all", "contact/jam", "insertion/jam", "contact/success", "insertion/success"]:
		item = report["correction"]["groups"].get(name)
		if item is None:
			continue
		lines.append(
			f"| `{name}` | {item['count']} | {item['baseline_multitask_to_direct_policy_l2']['mean']:.4f} | "
			f"{item['corrected_to_direct_policy_l2']['mean']:.4f} | {item['l2_reduction_fraction']:.3f} |"
		)
	lines.extend([
		"",
		"## Replay Target Coverage",
		"",
		"| Task | Transitions | Episodes | Exact successful-transition fraction | Top-return episode fraction |",
		"| --- | ---: | ---: | --- | ---: |",
	])
	for task_id, item in report["replay_quality"]["replay"].items():
		lines.append(
			f"| `{task_id}` | {item['transition_count']} | {item['episode_count']} | "
			f"{item['successful_transition_fraction']} | {item['top_return_episode_proportion']:.3f} |"
		)
	lines.extend([
		"",
		"Replay snapshots do not contain per-episode relaxed/strict/process success or jam labels. Consequently, exact successful-transition proportion and replay-only contact/jam success coverage are `UNKNOWN_WITH_REASON`; the report uses Phase 3.3 labeled rollout data for the quality-conditioned coverage rows above rather than silently treating top-return as success.",
		"",
		"## Inputs",
		"",
		f"- direct checkpoint: `{report['inputs']['direct_checkpoint']}`",
		f"- multitask checkpoint: `{report['inputs']['multitask_checkpoint']}`",
		f"- 00186 replay: `{report['inputs']['replays']['00186']}`",
		f"- Phase 3.3 rollout root: `{report['inputs']['rollout_root']}`",
	])
	return "\n".join(lines) + "\n"


def build_report(args: argparse.Namespace):
	diagnosis = _load_json(args.phase32_diagnosis)
	if diagnosis.get("status") != "PASS":
		raise RuntimeError(f"Unexpected Phase 3.2 diagnosis status: {diagnosis.get('status')}")
	checkpoints = diagnosis.get("checkpoints", {})
	direct_checkpoint = resolve(checkpoints.get("direct_finetune", ""))
	multitask_checkpoint = resolve(checkpoints.get("multitask_rescue_best", ""))
	replay_paths = OrderedDict(DEFAULT_REPLAYS)
	replay_paths["00186"] = str(diagnosis.get("replay", ""))
	replay_paths = OrderedDict((task_id, resolve(path)) for task_id, path in replay_paths.items())
	for path in (direct_checkpoint, multitask_checkpoint, *replay_paths.values(), resolve(args.rollout_root)):
		if not path.exists():
			raise FileNotFoundError(path)

	device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() and not args.cpu else "cpu")
	if device.type == "cuda":
		torch.cuda.set_device(device)
	models = OrderedDict()
	cfgs = OrderedDict()
	compat = OrderedDict()
	for label, checkpoint in (("direct", direct_checkpoint), ("multitask", multitask_checkpoint)):
		models[label], cfgs[label], compat[label] = attribution._load_model(checkpoint, args, device)
		if int(compat[label]["obs_dim"]) != 17 or int(compat[label]["action_dim"]) != 3:
			raise RuntimeError(f"{label} checkpoint is not compatible with 00186 17D/3D replay: {compat[label]}")

	replays = OrderedDict()
	for task_id, path in replay_paths.items():
		td, metadata = _load_snapshot(path)
		if tuple(td["obs"].shape[1:]) != (17,) or tuple(td["action"].shape[1:]) != (3,):
			raise RuntimeError(f"Unexpected replay contract task={task_id}: obs={td['obs'].shape}, action={td['action'].shape}")
		replays[task_id] = {"td": td, "metadata": metadata, "path": path, "task_vec": _unique_task_vec(td)}

	rows_per_task = max(int(args.gradient_samples), int(args.policy_compare_samples))
	replay_rows = OrderedDict((
		task_id,
		_sample_rows(item["td"], rows_per_task, int(args.seed) + 97 * index),
	) for index, (task_id, item) in enumerate(replays.items()))
	gradients = _gradient_report(models["multitask"], replay_rows, device, args)

	rollout_args = argparse.Namespace(**vars(args))
	rollout_args.rollout_root = args.rollout_root
	banks, rollout_metadata = attribution._load_rollout_banks(rollout_args)
	rollout_rows = _rollout_action_rows(models, replays["00186"]["task_vec"], banks, device, args)
	action_metrics = _action_metrics(rollout_rows)
	correction = _correction_report(rollout_rows, args)
	correction_metrics = _action_metrics(rollout_rows, correction=correction["prediction"])
	correction.pop("prediction")
	replay_quality = _replay_quality_report(replays, rollout_rows)
	classification = _classify(gradients, replay_quality, correction)

	messages = [
		{
			"level": "PASS",
			"message": "Policy-prior gradients, same-state action attribution, replay coverage, and offline correction completed without mutating a checkpoint.",
		},
		{
			"level": "WARNING",
			"message": "Replay snapshots lack exact relaxed/strict/process and jam labels; those quantities are explicitly UNKNOWN in replay-only coverage and Phase 3.3 rollout labels are used where available.",
		},
	]
	nonfinite_counts = {
		task_id: int(rows["dropped_nonfinite_transition_count"])
		for task_id, rows in replay_rows.items()
	}
	if any(nonfinite_counts.values()):
		messages.append({
			"level": "WARNING",
			"message": (
				"Excluded non-finite replay terminal/place-holder action rows from policy supervision: "
				+ ", ".join(f"{task_id}={count}" for task_id, count in nonfinite_counts.items())
			),
		})
	report = {
		"status": "PASS_WITH_CAVEAT",
		"messages": messages,
		"classification": classification,
		"inputs": {
			"phase32_diagnosis": str(resolve(args.phase32_diagnosis)),
			"direct_checkpoint": str(direct_checkpoint),
			"multitask_checkpoint": str(multitask_checkpoint),
			"replays": {task_id: str(path) for task_id, path in replay_paths.items()},
			"rollout_root": str(resolve(args.rollout_root)),
		},
		"device": str(device),
		"task_vectors": {task_id: tensor_to_list(item["task_vec"]) for task_id, item in replays.items()},
		"policy_supervision_row_filter": {
			"definition": "finite obs/action/task rows only; terminal/place-holder rows with non-finite action are excluded from gradient and target statistics.",
			"dropped_nonfinite_transition_count": nonfinite_counts,
		},
		"checkpoint_compatibility": compat,
		"gradients": gradients,
		"action_alignment": rollout_rows["alignment"],
		"direct_success_action_centroids": rollout_rows["direct_success_action_centroids"],
		"action_metrics": action_metrics,
		"corrected_action_metrics": correction_metrics,
		"correction": correction,
		"replay_quality": replay_quality,
		"rollout_metadata": rollout_metadata,
		"limitations": [
			"Direct and multitask rollout trajectories share indexed initial states but diverge after policy actions differ; direct MPPI selected actions are therefore reported only for exact same-state rows.",
			"Replay snapshots contain no exact relaxed/strict/process/jam field, so successful replay target fractions cannot be inferred without inventing a proxy.",
			"Affine correction is a diagnostic fit only. It neither changes a checkpoint nor establishes that a deployable residual will improve closed-loop control.",
		],
	}
	return report


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--phase32-diagnosis", default=DEFAULT_PHASE32_DIAGNOSIS)
	parser.add_argument("--rollout-root", default=DEFAULT_ROLLOUT_ROOT)
	parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
	parser.add_argument("--output-md", default=DEFAULT_OUTPUT_MD)
	parser.add_argument("--config", default="configs/train/srsa_01125_imitation_relaxed.yaml")
	parser.add_argument("--batch-size", type=int, default=256)
	parser.add_argument("--policy-compare-samples", type=int, default=4096)
	parser.add_argument("--gradient-samples", type=int, default=2048)
	parser.add_argument("--gradient-chunks", type=int, default=8)
	parser.add_argument("--correction-ridge", type=float, default=1.0e-4)
	parser.add_argument("--same-state-tolerance", type=float, default=1.0e-6)
	parser.add_argument("--jam-lateral-threshold", type=float, default=0.008)
	parser.add_argument("--jam-keypoint-threshold", type=float, default=0.012)
	parser.add_argument("--jam-force-excursion-threshold", type=float, default=2.0)
	parser.add_argument("--seed", type=int, default=1)
	parser.add_argument("--gpu-id", type=int, default=0)
	parser.add_argument("--cpu", action="store_true")
	parser.add_argument("--dry-run", action="store_true")
	args = parser.parse_args()
	if args.dry_run:
		print(f"PASS dry-run: would read Phase 3.2 diagnosis and Phase 3.3 rollouts, then write {resolve(args.output_json)} and {resolve(args.output_md)}")
		return 0
	report = build_report(args)
	write_json(report, args.output_json)
	write_text(_markdown(report), args.output_md)
	print(report["status"])
	print(f"Final classification: {report['classification']['classification']}")
	for message in report["messages"]:
		print(f"[{message['level']}] {message['message']}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
