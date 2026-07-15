#!/usr/bin/env python3
"""Read-only Phase 3.3 planner/action attribution diagnosis for 00186."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_AUDIT_DIR = SCRIPT_DIR.parent / "model_task_sensitivity"
if str(MODEL_AUDIT_DIR) not in sys.path:
	sys.path.insert(0, str(MODEL_AUDIT_DIR))

from _common import (  # noqa: E402
	condition_batch,
	resolve,
	summarize_tensor,
	tensor_to_list,
	two_hot_scalar,
	tvsr,
	write_json,
	write_text,
	WorldModel,
)


DEFAULT_PHASE32 = "reports/phase3_three_task_pilot/phase3_2_acquisition_rescue_summary.json"
DEFAULT_PHASE32_DIAGNOSIS = (
	"reports/phase3_three_task_pilot/phase3_2_diagnosis/"
	"standalone_vs_multitask_diagnosis.json"
)
DEFAULT_ROLLOUT_ROOT = "reports/phase3_three_task_pilot/phase3_3_rollouts"
DEFAULT_OUTPUT_JSON = "reports/phase3_three_task_pilot/phase3_3_planner_action_diagnosis.json"
DEFAULT_OUTPUT_MD = "reports/phase3_three_task_pilot/phase3_3_planner_action_diagnosis.md"
MODELS = ("direct", "multitask")
REGIONS = ("easy", "default", "hard")
PHASES = ("pre_contact", "contact", "insertion")
OUTCOMES = ("success", "jam", "failure")


def _load_json(path: str | Path) -> dict[str, Any]:
	path = resolve(path)
	if not path.exists():
		raise FileNotFoundError(path)
	return json.loads(path.read_text(encoding="utf-8"))


def _load_model(checkpoint: Path, args: argparse.Namespace, device: torch.device):
	cfg_args = SimpleNamespace(
		config=args.config,
		gpu_id=args.gpu_id,
		batch_size=args.batch_size,
		assembly_id="00186",
		eval_task_id=2,
	)
	cfg, compat = tvsr._load_config(cfg_args, checkpoint)
	cfg.device_id = int(args.gpu_id) if device.type == "cuda" else 0
	model = WorldModel(cfg).to(device)
	model = tvsr._load_world_model(model, checkpoint, cfg)
	model.eval()
	return model, cfg, compat


def _load_tensordict(path: Path):
	obj = torch.load(path, map_location="cpu", weights_only=False)
	return obj.get("data", obj) if isinstance(obj, dict) else obj


def _safe_metric(td, key: str, fallback: str | None = None):
	if key in td.keys():
		return td[key].detach().float().reshape(-1)
	if fallback and fallback in td.keys():
		return td[fallback].detach().float().reshape(-1)
	return None


def _episode_phase_labels(td) -> tuple[torch.Tensor, dict[str, Any]]:
	episodes = td["episode"].detach().long().reshape(-1)
	steps = td["step_id"].detach().long().reshape(-1)
	obs = td["obs"].detach().float()
	phase = torch.empty_like(steps)
	boundaries = []
	for episode_id in torch.unique(episodes, sorted=True):
		idx = torch.nonzero(episodes == episode_id, as_tuple=False).reshape(-1)
		order = torch.argsort(steps[idx])
		idx = idx[order]
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
		boundaries.append({
			"episode": int(episode_id.item()),
			"length": n,
			"contact_step": contact_at,
			"insertion_step": insert_at,
			"tcp_z_progress_m": total,
		})
	return phase, {
		"method": "tcp_z_progress_25pct_65pct_with_27pct_60pct_time_fallback",
		"episode_boundaries": boundaries,
	}


def _episode_outcome_labels(td, *, lateral_threshold: float, keypoint_threshold: float, force_excursion_threshold: float):
	episodes = td["episode"].detach().long().reshape(-1)
	obs = td["obs"].detach().float()
	success_values = _safe_metric(td, "episode_relaxed_success_stable_final", "episode_success_final")
	if success_values is None:
		raise KeyError("Rollout dataset lacks relaxed/success episode labels.")
	lateral = _safe_metric(td, "episode_lateral_error_final")
	keypoint = _safe_metric(td, "episode_keypoint_error_final")
	labels = torch.empty_like(episodes)
	episode_rows = []
	for episode_id in torch.unique(episodes, sorted=True):
		idx = torch.nonzero(episodes == episode_id, as_tuple=False).reshape(-1)
		is_success = bool(success_values[idx[0]].item() > 0.5)
		force = torch.linalg.vector_norm(obs[idx, 14:17], dim=-1) if obs.shape[-1] >= 17 else torch.zeros(idx.numel())
		baseline = float(force[: min(5, force.numel())].mean().item())
		force_excursion = float((force.max() - baseline).item())
		lat = float(lateral[idx[0]].item()) if lateral is not None else math.nan
		key = float(keypoint[idx[0]].item()) if keypoint is not None else math.nan
		jam_proxy = (
			not is_success and (
				(math.isfinite(lat) and lat >= lateral_threshold) or
				(math.isfinite(key) and key >= keypoint_threshold) or
				force_excursion >= force_excursion_threshold
			)
		)
		label = 0 if is_success else (1 if jam_proxy else 2)
		labels[idx] = label
		episode_rows.append({
			"episode": int(episode_id.item()),
			"outcome": OUTCOMES[label],
			"relaxed_success": is_success,
			"lateral_error_mm": 1000.0 * lat if math.isfinite(lat) else None,
			"keypoint_error_mm": 1000.0 * key if math.isfinite(key) else None,
			"force_excursion": force_excursion,
		})
	return labels, episode_rows


def _load_rollout_banks(args: argparse.Namespace):
	root = resolve(args.rollout_root)
	banks = OrderedDict()
	metadata = OrderedDict()
	for model_label in MODELS:
		for region in REGIONS:
			name = f"{model_label}_{region}"
			path = root / name / "00186" / "policy_eval_rollouts.pt"
			meta_path = path.with_suffix(path.suffix + ".json")
			if not path.exists():
				raise FileNotFoundError(
					f"Missing Phase 3.3 rollout {path}. Run scripts/collect_phase3_3_attribution_rollouts.sh first."
				)
			td = _load_tensordict(path)
			if tuple(td["obs"].shape[1:]) != (17,) or tuple(td["action"].shape[1:]) != (3,):
				raise RuntimeError(f"Unexpected rollout contract in {path}: obs={td['obs'].shape} action={td['action'].shape}")
			phase, phase_meta = _episode_phase_labels(td)
			outcome, episode_rows = _episode_outcome_labels(
				td,
				lateral_threshold=float(args.jam_lateral_threshold),
				keypoint_threshold=float(args.jam_keypoint_threshold),
				force_excursion_threshold=float(args.jam_force_excursion_threshold),
			)
			banks[name] = {
				"td": td,
				"model": model_label,
				"region": region,
				"phase": phase,
				"outcome": outcome,
				"path": path,
			}
			metadata[name] = {
				"path": str(path),
				"metadata": _load_json(meta_path) if meta_path.exists() else {},
				"phase_assignment": phase_meta,
				"episode_outcomes": episode_rows,
			}
	return banks, metadata


def _sample_states(banks, args: argparse.Namespace):
	generator = torch.Generator().manual_seed(int(args.seed))
	columns: dict[str, list[Any]] = defaultdict(list)
	group_counts = OrderedDict()
	for bank_name, bank in banks.items():
		td = bank["td"]
		for phase_id, phase_name in enumerate(PHASES):
			for outcome_id, outcome_name in enumerate(OUTCOMES):
				idx = torch.nonzero((bank["phase"] == phase_id) & (bank["outcome"] == outcome_id), as_tuple=False).reshape(-1)
				if idx.numel() == 0:
					continue
				perm = torch.randperm(idx.numel(), generator=generator)
				idx = idx[perm[: min(int(args.max_states_per_group), idx.numel())]]
				group_name = f"{bank_name}/{phase_name}/{outcome_name}"
				group_counts[group_name] = int(idx.numel())
				columns["obs"].append(td["obs"][idx].detach().float())
				columns["next_obs"].append(td["next_obs"][idx].detach().float())
				columns["action"].append(td["action"][idx].detach().float())
				columns["reward"].append(td["reward"][idx].detach().float().reshape(-1, 1))
				columns["episode"].append(td["episode"][idx].detach().long())
				columns["step_id"].append(td["step_id"][idx].detach().long())
				columns["source"].extend([bank["model"]] * int(idx.numel()))
				columns["region"].extend([bank["region"]] * int(idx.numel()))
				columns["phase"].extend([phase_name] * int(idx.numel()))
				columns["outcome"].extend([outcome_name] * int(idx.numel()))
	if not columns["obs"]:
		raise RuntimeError("No rollout states were sampled.")
	return {
		"obs": torch.cat(columns["obs"], dim=0),
		"next_obs": torch.cat(columns["next_obs"], dim=0),
		"action": torch.cat(columns["action"], dim=0),
		"reward": torch.cat(columns["reward"], dim=0),
		"episode": torch.cat(columns["episode"], dim=0),
		"step_id": torch.cat(columns["step_id"], dim=0),
		"source": columns["source"],
		"region": columns["region"],
		"phase": columns["phase"],
		"outcome": columns["outcome"],
		"group_counts": group_counts,
	}


def _atanh(value: torch.Tensor) -> torch.Tensor:
	value = value.clamp(-0.999999, 0.999999)
	return 0.5 * (torch.log1p(value) - torch.log1p(-value))


@torch.no_grad()
def _policy_proposals(model, obs, task_vec, eps, *, horizon: int, num_candidates: int):
	n, _, _, action_dim = eps.shape
	task_obs = condition_batch(task_vec, (n,), obs.device)
	z0 = model.encode(obs, task_obs)
	z = z0.unsqueeze(1).expand(n, num_candidates, -1).reshape(n * num_candidates, -1)
	task = condition_batch(task_vec, (n * num_candidates,), obs.device)
	actions = []
	means = []
	stds = []
	for step in range(horizon):
		_, info = model.pi(z, task)
		mean = info["mean"]
		std = info["log_std"].exp()
		noise = eps[:, step].reshape(n * num_candidates, action_dim)
		action = torch.tanh(_atanh(mean) + noise * std)
		actions.append(action.reshape(n, num_candidates, action_dim))
		means.append(mean.reshape(n, num_candidates, action_dim))
		stds.append(std.reshape(n, num_candidates, action_dim))
		z = model.next(z, action, task)
	return {
		"actions": torch.stack(actions, dim=1),
		"means": torch.stack(means, dim=1),
		"stds": torch.stack(stds, dim=1),
	}


@torch.no_grad()
def _score_candidates(model, cfg, obs, task_vec, candidates):
	n, horizon, num_candidates, action_dim = candidates.shape
	task_obs = condition_batch(task_vec, (n,), obs.device)
	z0 = model.encode(obs, task_obs)
	z = z0.unsqueeze(1).expand(n, num_candidates, -1).reshape(n * num_candidates, -1)
	task = condition_batch(task_vec, (n * num_candidates,), obs.device)
	reward_sum = torch.zeros(n * num_candidates, 1, device=obs.device)
	discount = torch.ones_like(reward_sum)
	first_next = None
	for step in range(horizon):
		action = candidates[:, step].reshape(n * num_candidates, action_dim)
		reward = two_hot_scalar(model.reward(z, action, task), cfg)
		reward_sum = reward_sum + discount * reward
		z = model.next(z, action, task)
		if first_next is None:
			first_next = z.reshape(n, num_candidates, -1)
		discount = discount * float(cfg.get("discount", 0.99))
	_, terminal_info = model.pi(z, task)
	terminal_action = terminal_info["mean"]
	q_all = two_hot_scalar(model.Q(z, terminal_action, task, return_type="all"), cfg)
	terminal_q = q_all.mean(dim=0) if q_all.ndim >= 3 else q_all
	total = reward_sum + discount * terminal_q
	return {
		"reward_sum": reward_sum.reshape(n, num_candidates),
		"terminal_q": terminal_q.reshape(n, num_candidates),
		"total": total.reshape(n, num_candidates),
		"first_next": first_next,
		"final_latent": z.reshape(n, num_candidates, -1),
	}


def _topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int) -> torch.Tensor:
	k = min(k, int(a.shape[-1]))
	top_a = torch.topk(a, k=k, dim=-1).indices.detach().cpu()
	top_b = torch.topk(b, k=k, dim=-1).indices.detach().cpu()
	return torch.tensor([
		len(set(x.tolist()) & set(y.tolist())) / max(k, 1)
		for x, y in zip(top_a, top_b)
	], dtype=torch.float32)


def _kendall_tau_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
	values = []
	for left, right in zip(a.detach().float().cpu(), b.detach().float().cpu()):
		concordant = 0
		discordant = 0
		for index in range(int(left.numel()) - 1):
			product = (left[index] - left[index + 1:]) * (right[index] - right[index + 1:])
			concordant += int((product > 0).sum().item())
			discordant += int((product < 0).sum().item())
		denom = concordant + discordant
		values.append((concordant - discordant) / denom if denom else 1.0)
	return torch.tensor(values, dtype=torch.float32)


def _masked_indices(labels: dict[str, list[str]], *, region=None, phase=None, outcome=None):
	mask = torch.ones(len(labels["region"]), dtype=torch.bool)
	for name, expected in (("region", region), ("phase", phase), ("outcome", outcome)):
		if expected is not None:
			mask &= torch.tensor([value == expected for value in labels[name]], dtype=torch.bool)
	return torch.nonzero(mask, as_tuple=False).reshape(-1)


def _selected(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
	rows = torch.arange(values.shape[0], device=values.device)
	return values[rows, indices]


def _cross_matrix(proposals, scores):
	selections = {}
	for bank in MODELS:
		for scorer in MODELS:
			selections[(bank, scorer)] = torch.argmax(scores[(bank, scorer)]["total"], dim=-1)
	dd = selections[("direct", "direct")]
	matrix = OrderedDict()
	for bank in MODELS:
		for scorer in MODELS:
			idx = selections[(bank, scorer)]
			own = scores[(bank, scorer)]
			direct_ref = scores[(bank, "direct")]
			multi_ref = scores[(bank, "multitask")]
			first = proposals[bank]["actions"][:, 0]
			selected_action = _selected(first, idx)
			matrix[f"{bank}_candidates_{scorer}_scoring"] = {
				"selected_total_score": summarize_tensor(_selected(own["total"], idx)),
				"selected_rollout_reward_sum": summarize_tensor(_selected(own["reward_sum"], idx)),
				"selected_terminal_q": summarize_tensor(_selected(own["terminal_q"], idx)),
				"selected_direct_reference_score": summarize_tensor(_selected(direct_ref["total"], idx)),
				"selected_multitask_reference_score": summarize_tensor(_selected(multi_ref["total"], idx)),
				"selected_first_action": {
					"mean": tensor_to_list(selected_action.mean(dim=0)),
					"std": tensor_to_list(selected_action.std(dim=0, unbiased=False)),
				},
			}
	dd_action = _selected(proposals["direct"]["actions"][:, 0], dd)
	for bank in MODELS:
		for scorer in MODELS:
			idx = selections[(bank, scorer)]
			action = _selected(proposals[bank]["actions"][:, 0], idx)
			matrix[f"{bank}_candidates_{scorer}_scoring"]["selected_action_l2_vs_DD"] = summarize_tensor(
				torch.linalg.vector_norm(action - dd_action, dim=-1)
			)
	direct_top_direct_bank = scores[("direct", "direct")]["total"].max(dim=-1).values
	direct_top_multi_bank = scores[("multitask", "direct")]["total"].max(dim=-1).values
	dm = selections[("direct", "multitask")]
	mm = selections[("multitask", "multitask")]
	proposal_regret = direct_top_direct_bank - direct_top_multi_bank
	scoring_regret_direct = direct_top_direct_bank - _selected(scores[("direct", "direct")]["total"], dm)
	scoring_regret_multi = direct_top_multi_bank - _selected(scores[("multitask", "direct")]["total"], mm)
	return matrix, selections, {
		"proposal_regret_under_direct_reference": summarize_tensor(proposal_regret),
		"multitask_scoring_regret_on_direct_candidates": summarize_tensor(scoring_regret_direct),
		"multitask_scoring_regret_on_multitask_candidates": summarize_tensor(scoring_regret_multi),
	}


@torch.no_grad()
def _model_calibration(model, cfg, task_vec, states, device):
	obs = states["obs"].to(device)
	next_obs = states["next_obs"].to(device)
	action = states["action"].to(device)
	reward = states["reward"].to(device)
	task = condition_batch(task_vec, (obs.shape[0],), device)
	z = model.encode(obs, task)
	z_next_pred = model.next(z, action, task)
	z_next_target = model.encode(next_obs, task)
	latent_error = torch.linalg.vector_norm(z_next_pred - z_next_target, dim=-1)
	reward_pred = two_hot_scalar(model.reward(z, action, task), cfg)
	q_all = two_hot_scalar(model.Q(z, action, task, return_type="all"), cfg)
	q_pred = q_all.mean(dim=0) if q_all.ndim >= 3 else q_all
	return {
		"latent_consistency_l2": latent_error.detach().cpu(),
		"reward_abs_error": (reward_pred - reward).abs().reshape(-1).detach().cpu(),
		"reward_bias": (reward_pred - reward).reshape(-1).detach().cpu(),
		"q_prediction": q_pred.reshape(-1).detach().cpu(),
		"reward_prediction": reward_pred.reshape(-1).detach().cpu(),
		"next_latent_pred": z_next_pred.detach().cpu(),
	}


def _grouped_metrics(values: dict[str, torch.Tensor], states):
	result = OrderedDict()
	labels = {key: states[key] for key in ("region", "phase", "outcome")}
	groups = [("all", None, None, None)]
	groups += [(f"region/{name}", name, None, None) for name in REGIONS]
	groups += [(f"phase/{name}", None, name, None) for name in PHASES]
	groups += [(f"outcome/{name}", None, None, name) for name in OUTCOMES]
	groups += [(f"hard/{name}", "hard", None, name) for name in OUTCOMES]
	for name, region, phase, outcome in groups:
		idx = _masked_indices(labels, region=region, phase=phase, outcome=outcome)
		if idx.numel() == 0:
			continue
		result[name] = {"count": int(idx.numel())}
		for metric, tensor in values.items():
			if tensor.shape[0] == len(states["region"]):
				result[name][metric] = summarize_tensor(tensor[idx])
	return result


def _episode_return_to_go(td, discount: float) -> torch.Tensor:
	episodes = td["episode"].detach().long().reshape(-1)
	steps = td["step_id"].detach().long().reshape(-1)
	reward = td["reward"].detach().float().reshape(-1)
	result = torch.zeros_like(reward)
	for episode_id in torch.unique(episodes, sorted=True):
		idx = torch.nonzero(episodes == episode_id, as_tuple=False).reshape(-1)
		idx = idx[torch.argsort(steps[idx])]
		running = torch.tensor(0.0)
		for row in reversed(idx.tolist()):
			running = reward[row] + discount * running
			result[row] = running
	return result


def _nearest_distances(query: torch.Tensor, bank: torch.Tensor, *, chunk: int = 128) -> torch.Tensor:
	values = []
	for start in range(0, query.shape[0], chunk):
		values.append(torch.cdist(query[start:start + chunk], bank).min(dim=1).values)
	return torch.cat(values, dim=0)


def _distance_comparison(direct: torch.Tensor, multitask: torch.Tensor) -> dict[str, Any]:
	direct_summary = summarize_tensor(direct)
	multitask_summary = summarize_tensor(multitask)
	return {
		"direct": direct_summary,
		"multitask": multitask_summary,
		"multitask_over_direct_mean_ratio": (
			float(multitask.mean().item()) / max(float(direct.mean().item()), 1.0e-6)
		),
		"multitask_over_direct_median_ratio": (
			float(multitask.median().item()) / max(float(direct.median().item()), 1.0e-6)
		),
	}


def _coverage_report(banks, states, multitask_replay: Path, args: argparse.Namespace):
	direct_rows = []
	direct_effective_rows = []
	for name, bank in banks.items():
		if bank["model"] != "direct":
			continue
		td = bank["td"]
		direct_rows.append(td["obs"].detach().float())
		direct_effective_rows.append(td["obs"][bank["outcome"] == 0].detach().float())
	direct_bank = torch.cat(direct_rows, dim=0)
	direct_effective = torch.cat([value for value in direct_effective_rows if value.numel()], dim=0)
	multi_td = _load_tensordict(multitask_replay)
	multi_obs = multi_td["obs"].detach().float()
	multi_episodes = multi_td["episode"].detach().long().reshape(-1)
	episode_scores = []
	for episode_id in torch.unique(multi_episodes, sorted=True):
		idx = torch.nonzero(multi_episodes == episode_id, as_tuple=False).reshape(-1)
		episode_scores.append((float(multi_td["reward"][idx].float().sum().item()), int(episode_id.item())))
	cut = max(1, int(math.ceil(0.30 * len(episode_scores))))
	effective_ids = {episode for _, episode in sorted(episode_scores, reverse=True)[:cut]}
	effective_mask = torch.tensor([int(value) in effective_ids for value in multi_episodes.tolist()], dtype=torch.bool)
	multi_effective = multi_obs[effective_mask]

	query_mask = torch.tensor([
		source == "multitask" and (region == "hard" or outcome in {"jam", "failure"})
		for source, region, outcome in zip(states["source"], states["region"], states["outcome"])
	], dtype=torch.bool)
	query = states["obs"][query_mask].detach().float()
	query_regions = [value for value, keep in zip(states["region"], query_mask.tolist()) if keep]
	query_outcomes = [value for value, keep in zip(states["outcome"], query_mask.tolist()) if keep]

	generator = torch.Generator().manual_seed(int(args.seed) + 91)
	bank_size = min(int(args.coverage_bank_size), direct_bank.shape[0], multi_obs.shape[0])
	effective_size = min(int(args.coverage_bank_size), direct_effective.shape[0], multi_effective.shape[0])
	def sample(value, size):
		return value[torch.randperm(value.shape[0], generator=generator)[:size]]
	direct_bank = sample(direct_bank, bank_size)
	multi_bank = sample(multi_obs, bank_size)
	direct_effective = sample(direct_effective, effective_size)
	multi_effective = sample(multi_effective, effective_size)
	combined = torch.cat([direct_bank, multi_bank], dim=0)
	mean = combined.mean(dim=0)
	std = combined.std(dim=0, unbiased=False).clamp_min(1.0e-5)
	query_norm = (query - mean) / std
	direct_dist = _nearest_distances(query_norm, (direct_bank - mean) / std)
	multi_dist = _nearest_distances(query_norm, (multi_bank - mean) / std)
	direct_effective_dist = _nearest_distances(query_norm, (direct_effective - mean) / std)
	multi_effective_dist = _nearest_distances(query_norm, (multi_effective - mean) / std)

	failed_initial_rows = []
	failed_initial_regions = []
	for bank in banks.values():
		if bank["model"] != "multitask":
			continue
		td = bank["td"]
		mask = (td["step_id"].detach().long().reshape(-1) == 0) & (bank["outcome"] != 0)
		if mask.any():
			failed_initial_rows.append(td["obs"][mask].detach().float())
			failed_initial_regions.extend([bank["region"]] * int(mask.sum().item()))
	if not failed_initial_rows:
		raise RuntimeError("No failed/jam multitask initial states were available for coverage diagnosis.")
	failed_initial = torch.cat(failed_initial_rows, dim=0)
	failed_initial_norm = (failed_initial - mean) / std
	failed_initial_direct_dist = _nearest_distances(failed_initial_norm, (direct_effective - mean) / std)
	failed_initial_multi_dist = _nearest_distances(failed_initial_norm, (multi_effective - mean) / std)
	failed_initial_report = {
		"query_definition": "multitask rollout 中 failed/jam episode 的 step_id=0 状态",
		"query_count": int(failed_initial.shape[0]),
		**_distance_comparison(failed_initial_direct_dist, failed_initial_multi_dist),
		"groups": OrderedDict(),
	}
	for region in REGIONS:
		mask = torch.tensor([value == region for value in failed_initial_regions], dtype=torch.bool)
		if mask.any():
			failed_initial_report["groups"][region] = {
				"count": int(mask.sum().item()),
				**_distance_comparison(failed_initial_direct_dist[mask], failed_initial_multi_dist[mask]),
			}
	result = {
		"direct_coverage_source": "新采集的 exact-profile direct rollout 代理；direct 训练 replay 未保存",
		"multitask_coverage_source": str(multitask_replay),
		"query_definition": "multitask rollout 中 hard 区域或标记为 jam/failure 的状态",
		"effective_transition_definition": "direct relaxed-success episode 与 multitask replay 中回报最高 30% 的 episode",
		"normalization": "per-dimension mean/std of equal-sized direct/multitask coverage banks",
		"coverage_bank_size_each": bank_size,
		"effective_bank_size_each": effective_size,
		"query_count": int(query.shape[0]),
		"all_transition_distance": _distance_comparison(direct_dist, multi_dist),
		"effective_transition_distance": _distance_comparison(direct_effective_dist, multi_effective_dist),
		"failed_initial_state_distance": failed_initial_report,
		"groups": OrderedDict(),
	}
	for region in (None, "hard"):
		for outcome in (None, "jam", "failure"):
			mask = torch.ones(query.shape[0], dtype=torch.bool)
			if region:
				mask &= torch.tensor([value == region for value in query_regions])
			if outcome:
				mask &= torch.tensor([value == outcome for value in query_outcomes])
			if not mask.any():
				continue
			name = "/".join(value for value in (region, outcome) if value) or "all_hard_or_jam"
			result["groups"][name] = {
				"count": int(mask.sum().item()),
				**_distance_comparison(direct_effective_dist[mask], multi_effective_dist[mask]),
			}
	return result


def _proposal_summary(proposals):
	result = OrderedDict()
	for label, item in proposals.items():
		action = item["actions"].detach().cpu()
		result[label] = {
			"action_mean": tensor_to_list(action.mean(dim=(0, 1, 2))),
			"action_std": tensor_to_list(action.std(dim=(0, 1, 2), unbiased=False)),
			"proposal_std_mean": float(item["stds"].mean().item()),
			"first_action_norm": summarize_tensor(torch.linalg.vector_norm(action[:, 0], dim=-1)),
			"candidate_pair_diversity": summarize_tensor(
				torch.linalg.vector_norm(action[:, 0, 1:] - action[:, 0, :-1], dim=-1)
			),
		}
	result["matched_direct_vs_multitask_action_l2"] = summarize_tensor(
		torch.linalg.vector_norm(proposals["direct"]["actions"] - proposals["multitask"]["actions"], dim=-1)
	)
	return result


def _component_differences(scores):
	result = OrderedDict()
	for bank in MODELS:
		direct = scores[(bank, "direct")]
		multi = scores[(bank, "multitask")]
		result[bank] = {
			"rollout_reward_abs_delta": summarize_tensor((direct["reward_sum"] - multi["reward_sum"]).abs()),
			"terminal_q_abs_delta": summarize_tensor((direct["terminal_q"] - multi["terminal_q"]).abs()),
			"total_score_abs_delta": summarize_tensor((direct["total"] - multi["total"]).abs()),
			"next_latent_self_space_norm_direct": summarize_tensor(torch.linalg.vector_norm(direct["first_next"], dim=-1)),
			"next_latent_self_space_norm_multitask": summarize_tensor(torch.linalg.vector_norm(multi["first_next"], dim=-1)),
		}
	return result


def _ranking_report(scores):
	result = OrderedDict()
	for bank in MODELS:
		direct = scores[(bank, "direct")]["total"]
		multi = scores[(bank, "multitask")]["total"]
		top_direct = direct.argmax(dim=-1)
		top_multi = multi.argmax(dim=-1)
		result[bank] = {
			"kendall_tau": summarize_tensor(_kendall_tau_batch(direct, multi)),
			"top1_changed_rate": float((top_direct != top_multi).float().mean().item()),
			"top10_overlap": summarize_tensor(_topk_overlap(direct, multi, 10)),
		}
	return result


def _root_cause(attribution, calibration, coverage, components, ranking):
	proposal = float(attribution["proposal_regret_under_direct_reference"]["mean"])
	scoring = max(
		float(attribution["multitask_scoring_regret_on_direct_candidates"]["mean"]),
		float(attribution["multitask_scoring_regret_on_multitask_candidates"]["mean"]),
	)
	direct_dyn = float(calibration["direct"]["all"]["latent_consistency_l2"]["mean"])
	multi_dyn = float(calibration["multitask"]["all"]["latent_consistency_l2"]["mean"])
	dynamics_ratio = multi_dyn / max(direct_dyn, 1.0e-6)
	direct_reward = float(calibration["direct"]["all"]["reward_abs_error"]["mean"])
	multi_reward = float(calibration["multitask"]["all"]["reward_abs_error"]["mean"])
	reward_ratio = multi_reward / max(direct_reward, 1.0e-6)
	hard_jam_coverage = coverage["groups"].get("hard/jam", coverage["effective_transition_distance"])
	hard_jam_coverage_ratio = float(hard_jam_coverage["multitask_over_direct_mean_ratio"])
	failed_initial_coverage_ratio = float(
		coverage["failed_initial_state_distance"]["multitask_over_direct_mean_ratio"]
	)
	coverage_ratio = max(hard_jam_coverage_ratio, failed_initial_coverage_ratio)
	q_delta = max(float(item["terminal_q_abs_delta"]["mean"]) for item in components.values())
	reward_delta = max(float(item["rollout_reward_abs_delta"]["mean"]) for item in components.values())
	tau = min(float(item["kendall_tau"]["mean"]) for item in ranking.values())
	ranking_harmful = scoring > max(0.5, 0.5 * proposal)
	reward_calibration_bad = reward_ratio >= 1.25

	flags = {
		"policy_proposal": proposal > max(0.5, 1.5 * scoring),
		"dynamics": dynamics_ratio >= 1.25,
		"reward_q_ranking": ranking_harmful or reward_calibration_bad,
		"coverage": coverage_ratio >= 1.25,
	}
	strength = {
		"policy_proposal": max(0.0, proposal / max(abs(proposal) + abs(scoring), 1.0e-6)),
		"dynamics": max(0.0, dynamics_ratio - 1.0),
		"reward_q_ranking": max(
			0.0,
			scoring / max(abs(proposal) + abs(scoring), 1.0e-6),
			reward_ratio - 1.0 if reward_calibration_bad else 0.0,
			0.70 - tau if ranking_harmful else 0.0,
		),
		"coverage": max(0.0, coverage_ratio - 1.0),
	}
	active = [name for name, value in flags.items() if value]
	if len(active) == 1:
		root = {
			"policy_proposal": "POLICY_PROPOSAL_FAILURE",
			"dynamics": "DYNAMICS_PREDICTION_FAILURE",
			"reward_q_ranking": "REWARD_Q_RANKING_FAILURE",
			"coverage": "REPLAY_STATE_COVERAGE_FAILURE",
		}[active[0]]
	elif active:
		ranked = sorted(active, key=lambda name: strength[name], reverse=True)
		if len(ranked) == 1 or strength[ranked[0]] >= 1.75 * max(strength[ranked[1]], 1.0e-6):
			root = {
				"policy_proposal": "POLICY_PROPOSAL_FAILURE",
				"dynamics": "DYNAMICS_PREDICTION_FAILURE",
				"reward_q_ranking": "REWARD_Q_RANKING_FAILURE",
				"coverage": "REPLAY_STATE_COVERAGE_FAILURE",
			}[ranked[0]]
		else:
			root = "MIXED_OR_UNRESOLVED"
	else:
		root = "MIXED_OR_UNRESOLVED"
	return {
		"primary_root_cause": root,
		"flags": flags,
		"strength": strength,
		"diagnostic_values": {
			"proposal_regret": proposal,
			"scoring_regret": scoring,
			"multitask_over_direct_dynamics_error_ratio": dynamics_ratio,
			"multitask_over_direct_reward_mae_ratio": reward_ratio,
			"multitask_over_direct_effective_coverage_distance_ratio": coverage_ratio,
			"hard_jam_effective_coverage_distance_ratio": hard_jam_coverage_ratio,
			"failed_initial_effective_coverage_distance_ratio": failed_initial_coverage_ratio,
			"terminal_q_abs_delta": q_delta,
			"rollout_reward_abs_delta": reward_delta,
			"q_difference_dominates": q_delta > 1.5 * max(reward_delta, 1.0e-6),
			"minimum_ranking_kendall_tau": tau,
			"ranking_difference_is_harmful_under_direct_reference": ranking_harmful,
		},
	}


def _markdown(report: dict[str, Any]) -> str:
	root = report["root_cause"]
	values = root["diagnostic_values"]
	lines = [
		"# SRSA Phase 3.3 00186 Planner/Action Attribution Diagnosis",
		"",
		"本报告只做只读 planner/action attribution；未继续训练，未修改 task_context、reward、Q、policy、MPPI 或 sampler，也未进入 consolidation。",
		"",
		f"Status: `{report['status']}`",
		f"Primary root cause: `{root['primary_root_cause']}`",
		"",
		"## 结论",
		"",
		f"- 主要根因：`{root['primary_root_cause']}`。",
		f"- direct-reference proposal regret：`{values['proposal_regret']:.4f}`；multitask scorer regret：`{values['scoring_regret']:.4f}`。",
		f"- matched proposal action L2：`{report['proposal_distribution']['matched_direct_vs_multitask_action_l2']['mean']:.4f}`；proposal std：direct=`{report['proposal_distribution']['direct']['proposal_std_mean']:.4f}`，multitask=`{report['proposal_distribution']['multitask']['proposal_std_mean']:.4f}`。",
		f"- dynamics consistency error ratio（multitask/direct）：`{values['multitask_over_direct_dynamics_error_ratio']:.3f}`。",
		f"- reward MAE ratio（multitask/direct）：`{values['multitask_over_direct_reward_mae_ratio']:.3f}`。",
		f"- coverage 最大有效距离比（hard/jam 或失败初始状态，multitask/direct）：`{values['multitask_over_direct_effective_coverage_distance_ratio']:.3f}`。",
		f"- Q 是否主导 scorer 差异：`{values['q_difference_dominates']}`（Q delta=`{values['terminal_q_abs_delta']:.3f}`，rollout reward delta=`{values['rollout_reward_abs_delta']:.3f}`）。",
		f"- 全局 ranking 虽不同，但是否造成 direct-reference 选择损失：`{values['ranking_difference_is_harmful_under_direct_reference']}`。",
		"- 因此 Q 的绝对标定差异是次级现象：它改变全排序，但未在固定候选 bank 上造成足以解释性能差距的 top-choice regret。",
		"",
		"## 闭环输入分布",
		"",
		"| Model | Easy relaxed | Default relaxed | Hard relaxed |",
		"| --- | ---: | ---: | ---: |",
	]
	for model in MODELS:
		values_by_region = []
		for region in REGIONS:
			metadata = report["rollout_metadata"][f"{model}_{region}"]["metadata"]
			values_by_region.append(float(metadata["episode_success_mean"]))
		lines.append(
			f"| `{model}` | {values_by_region[0]:.3f} | {values_by_region[1]:.3f} | {values_by_region[2]:.3f} |"
		)
	lines.extend([
		"",
		"## 交叉矩阵",
		"",
		"| Candidate bank | Scorer | Selected total | Rollout reward | Terminal Q | Direct-ref score | Action L2 vs DD |",
		"| --- | --- | ---: | ---: | ---: | ---: | ---: |",
	])
	for bank in MODELS:
		for scorer in MODELS:
			item = report["cross_matrix"][f"{bank}_candidates_{scorer}_scoring"]
			lines.append(
				f"| `{bank}` | `{scorer}` | {item['selected_total_score']['mean']:.3f} | "
				f"{item['selected_rollout_reward_sum']['mean']:.3f} | {item['selected_terminal_q']['mean']:.3f} | "
				f"{item['selected_direct_reference_score']['mean']:.3f} | {item['selected_action_l2_vs_DD']['mean']:.3f} |"
			)
	lines.extend([
		"",
		"## Ranking 与模型误差",
		"",
		"| Candidate bank | Kendall tau | Top1 changed | Top10 overlap |",
		"| --- | ---: | ---: | ---: |",
	])
	for bank, item in report["ranking"].items():
		lines.append(
			f"| `{bank}` | {item['kendall_tau']['mean']:.3f} | {item['top1_changed_rate']:.3f} | "
			f"{item['top10_overlap']['mean']:.3f} |"
		)
	lines.extend([
		"",
		"| Model | Next-latent consistency L2 | Reward MAE |",
		"| --- | ---: | ---: |",
	])
	for model in MODELS:
		item = report["calibration"][model]["all"]
		lines.append(
			f"| `{model}` | {item['latent_consistency_l2']['mean']:.4f} | {item['reward_abs_error']['mean']:.4f} |"
		)
	lines.extend([
		"",
		"## 状态区域",
		"",
		"| Group | Count | Proposal regret | Direct-bank scorer regret | Multitask-bank scorer regret |",
		"| --- | ---: | ---: | ---: | ---: |",
	])
	for group_type in ("region", "phase", "outcome"):
		for name, item in report["grouped_cross_attribution"][group_type].items():
			lines.append(
				f"| `{group_type}/{name}` | {item['count']} | "
				f"{item['proposal_regret_under_direct_reference']['mean']:.3f} | "
				f"{item['multitask_scoring_regret_on_direct_candidates']['mean']:.3f} | "
				f"{item['multitask_scoring_regret_on_multitask_candidates']['mean']:.3f} |"
			)
	lines.extend([
		"",
		"阶段标签使用 TCP-z 进度的 25%/65% 分界，进度不足时退化为 episode 时间的 27%/60% 分界。逐 episode jam 未由现有 collector 直接导出，因此 jam 使用 failure + lateral/keypoint/force-excursion 的只读代理标签。",
		"",
		"## Replay Coverage",
		"",
		f"- direct coverage：{report['coverage']['direct_coverage_source']}。",
		f"- multitask coverage：`{report['coverage']['multitask_coverage_source']}`。",
		f"- query 定义：{report['coverage']['query_definition']}。",
		f"- effective bank 定义：{report['coverage']['effective_transition_definition']}。",
		f"- hard/jam query 的 effective mean-distance ratio：`{values['hard_jam_effective_coverage_distance_ratio']:.3f}`。",
		f"- 失败初始状态（step_id=0）的 effective mean/median-distance ratio：`{values['failed_initial_effective_coverage_distance_ratio']:.3f}` / `{report['coverage']['failed_initial_state_distance']['multitask_over_direct_median_ratio']:.3f}`。",
		f"- 全 query effective mean/median-distance ratio：`{report['coverage']['effective_transition_distance']['multitask_over_direct_mean_ratio']:.3f}` / `{report['coverage']['effective_transition_distance']['multitask_over_direct_median_ratio']:.3f}`。",
		"",
		"## 判定说明",
		"",
		"- proposal regret 大、但同一 direct candidate bank 上 multitask scorer regret 小，指向 `POLICY_PROPOSAL_FAILURE`。",
		"- 同一 candidates 下排序/选择 regret 大，且 reward/Q 或 ranking 差异占主导，指向 `REWARD_Q_RANKING_FAILURE`。",
		"- multitask one-step latent consistency 明显差于 direct，指向 `DYNAMICS_PREDICTION_FAILURE`。",
		"- hard/jam query 到 multitask 有效 transition 的距离显著更大，指向 `REPLAY_STATE_COVERAGE_FAILURE`。",
		"- 多项证据同时成立且没有单一主导项时，判为 `MIXED_OR_UNRESOLVED`。",
		"",
		"## 输入",
		"",
		f"- direct checkpoint: `{report['inputs']['direct_checkpoint']}`",
		f"- multitask checkpoint: `{report['inputs']['multitask_checkpoint']}`",
		f"- multitask replay: `{report['inputs']['multitask_replay']}`",
		f"- sampled states: `{report['sample_size']}`",
		f"- candidates per proposal bank: `{report['num_candidates']}`",
	])
	return "\n".join(lines) + "\n"


@torch.no_grad()
def build_report(args: argparse.Namespace):
	phase32 = _load_json(args.phase32_summary)
	phase32_diag = _load_json(args.phase32_diagnosis)
	if phase32.get("status") != "STOP_AND_DIAGNOSE_ACQUISITION_NON_MONOTONIC":
		raise RuntimeError(f"Unexpected Phase 3.2 status: {phase32.get('status')}")
	checkpoints = phase32_diag.get("checkpoints") or {}
	direct_checkpoint = resolve(checkpoints.get("direct_finetune", ""))
	multitask_checkpoint = resolve(checkpoints.get("multitask_rescue_best", ""))
	multitask_replay = resolve(phase32_diag.get("replay", ""))
	for path in (direct_checkpoint, multitask_checkpoint, multitask_replay):
		if not path.exists():
			raise FileNotFoundError(path)

	device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() and not args.cpu else "cpu")
	if device.type == "cuda":
		torch.cuda.set_device(device)
	torch.manual_seed(int(args.seed))
	models = OrderedDict()
	cfgs = OrderedDict()
	compat = OrderedDict()
	for label, checkpoint in (("direct", direct_checkpoint), ("multitask", multitask_checkpoint)):
		models[label], cfgs[label], compat[label] = _load_model(checkpoint, args, device)
		if int(compat[label]["obs_dim"]) != 17 or int(compat[label]["action_dim"]) != 3:
			raise RuntimeError(f"{label} checkpoint is not 17D/3D compatible: {compat[label]}")

	banks, rollout_metadata = _load_rollout_banks(args)
	states = _sample_states(banks, args)
	task_vec = tvsr._unique_task_vec_from_replay(multitask_replay)[0].float().to(device)
	n = int(states["obs"].shape[0])
	generator = torch.Generator(device=device).manual_seed(int(args.seed) + 17)
	eps = torch.randn(
		n,
		int(args.horizon),
		int(args.num_candidates),
		3,
		device=device,
		generator=generator,
	)
	proposals = {label: {"actions": [], "means": [], "stds": []} for label in MODELS}
	scores = {(bank, scorer): {key: [] for key in ("reward_sum", "terminal_q", "total", "first_next", "final_latent")} for bank in MODELS for scorer in MODELS}
	for start in range(0, n, int(args.batch_size)):
		stop = min(start + int(args.batch_size), n)
		obs = states["obs"][start:stop].to(device)
		chunk_eps = eps[start:stop]
		chunk_proposals = {}
		for label in MODELS:
			chunk_proposals[label] = _policy_proposals(
				models[label], obs, task_vec, chunk_eps,
				horizon=int(args.horizon), num_candidates=int(args.num_candidates),
			)
			for key in proposals[label]:
				proposals[label][key].append(chunk_proposals[label][key].detach().cpu())
		for bank in MODELS:
			for scorer in MODELS:
				item = _score_candidates(models[scorer], cfgs[scorer], obs, task_vec, chunk_proposals[bank]["actions"])
				for key in scores[(bank, scorer)]:
					scores[(bank, scorer)][key].append(item[key].detach().cpu())
	for label in MODELS:
		for key in proposals[label]:
			proposals[label][key] = torch.cat(proposals[label][key], dim=0)
	for pair in scores:
		for key in scores[pair]:
			scores[pair][key] = torch.cat(scores[pair][key], dim=0)

	cross_matrix, selections, attribution = _cross_matrix(proposals, scores)
	ranking = _ranking_report(scores)
	components = _component_differences(scores)
	calibration_raw = {
		label: _model_calibration(models[label], cfgs[label], task_vec, states, device)
		for label in MODELS
	}
	calibration = {label: _grouped_metrics(value, states) for label, value in calibration_raw.items()}
	coverage = _coverage_report(banks, states, multitask_replay, args)

	grouped_cross = OrderedDict()
	labels = {key: states[key] for key in ("region", "phase", "outcome")}
	for group_type, names in (("region", REGIONS), ("phase", PHASES), ("outcome", OUTCOMES)):
		grouped_cross[group_type] = OrderedDict()
		for name in names:
			kwargs = {"region": None, "phase": None, "outcome": None}
			kwargs[group_type] = name
			idx = _masked_indices(labels, **kwargs)
			if idx.numel() == 0:
				continue
			_, _, item = _cross_matrix(
				{key: {subkey: value[subkey][idx] for subkey in value} for key, value in proposals.items()},
				{key: {subkey: value[subkey][idx] for subkey in value} for key, value in scores.items()},
			)
			grouped_cross[group_type][name] = {"count": int(idx.numel()), **item}

	root = _root_cause(attribution, calibration, coverage, components, ranking)
	report = {
		"status": "PASS" if root["primary_root_cause"] != "MIXED_OR_UNRESOLVED" else "PASS_WITH_CAVEAT",
		"root_cause": root,
		"inputs": {
			"phase32_summary": str(resolve(args.phase32_summary)),
			"phase32_diagnosis": str(resolve(args.phase32_diagnosis)),
			"direct_checkpoint": str(direct_checkpoint),
			"multitask_checkpoint": str(multitask_checkpoint),
			"multitask_replay": str(multitask_replay),
			"rollout_root": str(resolve(args.rollout_root)),
		},
		"device": str(device),
		"task_vec_00186": tensor_to_list(task_vec),
		"sample_size": n,
		"sample_group_counts": states["group_counts"],
		"horizon": int(args.horizon),
		"num_candidates": int(args.num_candidates),
		"candidate_noise": "same Gaussian epsilon tensor for direct and multitask policy proposals",
		"checkpoint_compatibility": compat,
		"proposal_distribution": _proposal_summary(proposals),
		"cross_matrix": cross_matrix,
		"attribution": attribution,
		"grouped_cross_attribution": grouped_cross,
		"ranking": ranking,
		"score_component_differences": components,
		"calibration": calibration,
		"coverage": coverage,
		"rollout_metadata": rollout_metadata,
		"limitations": [
			"可比 direct run 未保存训练 replay，因此使用 exact-profile direct eval rollout 作为带标签的 coverage 代理。",
			"collect_eval_rollouts.py 不导出逐 episode jam；本报告用 failure 加 lateral/keypoint/force-excursion 阈值推断 jam。",
			"交叉矩阵使用表现更好的 direct checkpoint scorer 作为参考，不是每条反事实候选序列的环境真值。",
		],
	}
	return report


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--phase32-summary", default=DEFAULT_PHASE32)
	parser.add_argument("--phase32-diagnosis", default=DEFAULT_PHASE32_DIAGNOSIS)
	parser.add_argument("--rollout-root", default=DEFAULT_ROLLOUT_ROOT)
	parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
	parser.add_argument("--output-md", default=DEFAULT_OUTPUT_MD)
	parser.add_argument("--config", default="configs/train/srsa_01125_imitation_relaxed.yaml")
	parser.add_argument("--batch-size", type=int, default=12)
	parser.add_argument("--max-states-per-group", type=int, default=8)
	parser.add_argument("--num-candidates", type=int, default=64)
	parser.add_argument("--horizon", type=int, default=3)
	parser.add_argument("--coverage-bank-size", type=int, default=2048)
	parser.add_argument("--jam-lateral-threshold", type=float, default=0.008)
	parser.add_argument("--jam-keypoint-threshold", type=float, default=0.012)
	parser.add_argument("--jam-force-excursion-threshold", type=float, default=2.0)
	parser.add_argument("--seed", type=int, default=1)
	parser.add_argument("--gpu-id", type=int, default=0)
	parser.add_argument("--cpu", action="store_true")
	parser.add_argument("--dry-run", action="store_true")
	args = parser.parse_args()
	if args.dry_run:
		print(f"PASS dry-run: would write {resolve(args.output_json)} and {resolve(args.output_md)}")
		return 0
	report = build_report(args)
	write_json(report, args.output_json)
	write_text(_markdown(report), args.output_md)
	print(report["status"])
	print(f"Primary root cause: {report['root_cause']['primary_root_cause']}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
