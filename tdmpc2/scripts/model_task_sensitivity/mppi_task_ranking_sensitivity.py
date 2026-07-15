#!/usr/bin/env python3
"""Audit whether task_vec_6 changes fixed-candidate MPPI return ranking."""

from __future__ import annotations

import argparse
from collections import OrderedDict

import torch

from _common import (
	add_common_args,
	add_message,
	condition_batch,
	l2,
	load_model_bundle,
	load_task_conditions,
	load_trimmed_replay_batch,
	output_dir,
	print_status,
	status_from_messages,
	summarize_tensor,
	tensor_to_list,
	two_hot_scalar,
	write_json,
)


def _kendall_tau(x: torch.Tensor, y: torch.Tensor) -> float:
	x = x.detach().float().cpu()
	y = y.detach().float().cpu()
	n = int(x.numel())
	if n < 2:
		return 1.0
	concordant = 0
	discordant = 0
	for i in range(n - 1):
		dx = x[i] - x[i + 1:]
		dy = y[i] - y[i + 1:]
		prod = dx * dy
		concordant += int((prod > 0).sum().item())
		discordant += int((prod < 0).sum().item())
	denom = concordant + discordant
	return float((concordant - discordant) / denom) if denom > 0 else 1.0


@torch.no_grad()
def _trajectory_returns(model, cfg, obs, actions, vec):
	batch_size, horizon, num_candidates, action_dim = actions.shape
	task_obs = condition_batch(vec, (batch_size,), obs.device)
	z0 = model.encode(obs, task_obs)
	z = z0.unsqueeze(1).expand(batch_size, num_candidates, z0.shape[-1]).reshape(batch_size * num_candidates, -1)
	task = condition_batch(vec, (batch_size * num_candidates,), obs.device)
	G = torch.zeros(batch_size * num_candidates, 1, device=obs.device)
	discount = torch.ones_like(G)
	for t in range(horizon):
		action = actions[:, t].reshape(batch_size * num_candidates, action_dim)
		reward = two_hot_scalar(model.reward(z, action, task), cfg)
		G = G + discount * reward
		z = model.next(z, action, task)
		discount = discount * float(cfg.get("discount", 0.99))
	terminal_action, _ = model.pi(z, task)
	value = model.Q(z, terminal_action, task, return_type="avg")
	return (G + discount * value).reshape(batch_size, num_candidates)


def _topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int) -> torch.Tensor:
	k = min(int(k), int(a.shape[-1]))
	top_a = torch.topk(a, k=k, dim=-1).indices
	top_b = torch.topk(b, k=k, dim=-1).indices
	overlaps = []
	for row_a, row_b in zip(top_a, top_b):
		set_a = set(int(x) for x in row_a.detach().cpu().tolist())
		set_b = set(int(x) for x in row_b.detach().cpu().tolist())
		overlaps.append(len(set_a & set_b) / max(k, 1))
	return torch.tensor(overlaps, dtype=torch.float32)


def _ranking_metrics(base_returns, other_returns, actions, topk):
	top_base = torch.argmax(base_returns, dim=-1)
	top_other = torch.argmax(other_returns, dim=-1)
	batch = int(base_returns.shape[0])
	first_actions = actions[:, 0]
	selected_base = first_actions[torch.arange(batch, device=actions.device), top_base]
	selected_other = first_actions[torch.arange(batch, device=actions.device), top_other]
	margin = base_returns[torch.arange(batch, device=actions.device), top_base] - base_returns[torch.arange(batch, device=actions.device), top_other]
	taus = torch.tensor([_kendall_tau(base_returns[i], other_returns[i]) for i in range(batch)], dtype=torch.float32)
	return {
		"kendall_tau": summarize_tensor(taus),
		"top1_changed_rate": float((top_base != top_other).float().mean().item()),
		"topk_overlap": summarize_tensor(_topk_overlap(base_returns, other_returns, int(topk))),
		"selected_action_L2": summarize_tensor(l2(selected_base, selected_other)),
		"return_margin_correct_vs_wrong": summarize_tensor(margin),
	}


@torch.no_grad()
def build_report(args):
	bundle = load_model_bundle(args)
	model = bundle["model"]
	cfg = bundle["cfg"]
	device = bundle["device"]
	conditions = load_task_conditions(args)
	batch = load_trimmed_replay_batch(args, bundle["compat"], device)
	obs = batch["obs"]
	action_dim = int(bundle["compat"]["action_dim"] or cfg.action_dim)
	base_label = str(args.base_label)
	swap_label = str(args.swap_label)
	if base_label not in conditions:
		raise KeyError(f"base label {base_label!r} not found in conditions={list(conditions.keys())}")
	if swap_label not in conditions:
		raise KeyError(f"swap label {swap_label!r} not found in conditions={list(conditions.keys())}")
	generator = torch.Generator(device=device).manual_seed(int(args.seed))
	actions = torch.empty(
		obs.shape[0],
		int(args.horizon),
		int(args.num_candidates),
		action_dim,
		device=device,
	).uniform_(-1.0, 1.0, generator=generator)
	returns: OrderedDict[str, torch.Tensor] = OrderedDict()
	for label, vec in conditions.items():
		returns[label] = _trajectory_returns(model, cfg, obs, actions, vec)
	base_returns = returns[base_label]
	metrics: OrderedDict[str, dict] = OrderedDict()
	for label, item in returns.items():
		if label == base_label:
			continue
		metrics[label] = _ranking_metrics(base_returns, item, actions, int(args.topk))
	swap_metrics = metrics[swap_label]
	zero_metrics = metrics.get("zero", {})
	random_metrics = metrics.get("random", {})
	messages: list[dict] = []
	zero_tau = float((zero_metrics.get("kendall_tau") or {}).get("mean", 1.0))
	random_tau = float((random_metrics.get("kendall_tau") or {}).get("mean", 1.0))
	zero_changed = float(zero_metrics.get("top1_changed_rate", 0.0))
	random_changed = float(random_metrics.get("top1_changed_rate", 0.0))
	if min(zero_tau, random_tau) >= float(args.ranking_same_tau) and max(zero_changed, random_changed) <= float(args.top1_same_eps):
		add_message(messages, "WARNING", "correct vs zero/random ranking is almost identical; task_vec likely does not enter planning objective strongly.")
	else:
		add_message(messages, "PASS", "Fixed-candidate return ranking changes for at least one non-realistic task condition.")
	report = {
		"status": status_from_messages(messages),
		"checkpoint": str(bundle["paths"]["checkpoint"]),
		"replay": str(bundle["paths"]["task_b_replay"]),
		"device": str(device),
		"sample_size": int(obs.shape[0]),
		"num_candidates": int(args.num_candidates),
		"horizon": int(args.horizon),
		"topk": int(args.topk),
		"candidate_source": "uniform[-1,1] fixed by seed",
		"conditions": {label: tensor_to_list(vec) for label, vec in conditions.items()},
		"base_label": base_label,
		"swap_label": swap_label,
		"metrics_vs_base": metrics,
		"kendall_tau_correct_swap": swap_metrics["kendall_tau"],
		"kendall_tau_correct_zero": (zero_metrics.get("kendall_tau") or {}),
		"kendall_tau_correct_random": (random_metrics.get("kendall_tau") or {}),
		"top1_changed_rate": {label: item["top1_changed_rate"] for label, item in metrics.items()},
		"topk_overlap": {label: item["topk_overlap"] for label, item in metrics.items()},
		"selected_action_L2": {label: item["selected_action_L2"] for label, item in metrics.items()},
		"return_margin_correct_vs_wrong": {label: item["return_margin_correct_vs_wrong"] for label, item in metrics.items()},
		"thresholds": {
			"ranking_same_tau": float(args.ranking_same_tau),
			"top1_same_eps": float(args.top1_same_eps),
		},
		"messages": messages,
	}
	return report


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	add_common_args(parser)
	parser.add_argument("--base-label", default="00256")
	parser.add_argument("--swap-label", default="01125")
	parser.add_argument("--horizon", type=int, default=5)
	parser.add_argument("--num-candidates", type=int, default=128)
	parser.add_argument("--topk", type=int, default=10)
	parser.add_argument("--ranking-same-tau", type=float, default=0.999)
	parser.add_argument("--top1-same-eps", type=float, default=0.0)
	args = parser.parse_args()
	if args.dry_run:
		out = output_dir(args) / "mppi_task_ranking_sensitivity.json"
		print_status("WARNING", [{"level": "WARNING", "message": "Dry-run requested; model not loaded."}])
		write_json({"status": "DRY_RUN", "output": str(out)}, out, dry_run=True)
		return 0
	report = build_report(args)
	print_status(report["status"], report.get("messages", []))
	write_json(report, output_dir(args) / "mppi_task_ranking_sensitivity.json", dry_run=False)
	return 1 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
	raise SystemExit(main())

