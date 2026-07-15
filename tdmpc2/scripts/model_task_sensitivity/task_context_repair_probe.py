#!/usr/bin/env python3
"""Probe task_context repair wiring without changing training state."""

from __future__ import annotations

import argparse
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

from _common import (
	add_common_args,
	add_message,
	abs_delta,
	condition_batch,
	l2,
	load_task_conditions,
	load_trimmed_replay_batch,
	output_dir,
	print_status,
	require_existing_inputs,
	resolve,
	select_device,
	status_from_messages,
	summarize_tensor,
	tensor_to_list,
	two_hot_scalar,
	tvsr,
	write_json,
	WorldModel,
)


def _load_variant(args, checkpoint: Path, device: torch.device, *, repair_enabled: bool, raw_residual_scale: float):
	cfg, compat = tvsr._load_config(args, checkpoint)
	cfg.device_id = int(args.gpu_id) if device.type == "cuda" else 0
	cfg.task_context_repair_enabled = bool(repair_enabled)
	cfg.task_raw_residual_scale = float(raw_residual_scale)
	cfg.task_recon_coef = 0.0
	cfg.task_spread_coef = 0.0
	model = WorldModel(cfg).to(device)
	model = tvsr._load_world_model(model, checkpoint, cfg)
	model.eval()
	return model, cfg, compat


@torch.no_grad()
def _contexts(model, conditions: OrderedDict[str, torch.Tensor], device: torch.device):
	labels = list(conditions.keys())
	task = torch.stack([vec.to(device=device, dtype=torch.float32) for vec in conditions.values()], dim=0)
	dummy = task.new_zeros(task.shape[0], 1)
	info = model.task_context_repair_info(dummy, task, reconstruct=True)
	if info is None:
		raise RuntimeError("WorldModel.task_context_repair_info() returned None; axial task conditioning is required.")
	context = info["task_context"]
	vec_norm = info["task_vec_norm"]
	recon = info["task_recon"]
	pairwise: OrderedDict[str, dict[str, float]] = OrderedDict()
	for i, left in enumerate(labels):
		pairwise[left] = OrderedDict()
		for j, right in enumerate(labels):
			dist = torch.linalg.vector_norm(context[i] - context[j]).item()
			pairwise[left][right] = float(dist)
	idx = torch.triu_indices(context.shape[0], context.shape[0], offset=1, device=device)
	ctx_dist = torch.cdist(context, context, p=2)[idx[0], idx[1]]
	vec_dist = torch.cdist(vec_norm, vec_norm, p=2)[idx[0], idx[1]]
	if ctx_dist.numel() >= 2 and torch.std(ctx_dist) > 1.0e-8 and torch.std(vec_dist) > 1.0e-8:
		corr = torch.corrcoef(torch.stack([ctx_dist, vec_dist], dim=0))[0, 1]
		corr_value = float(corr.detach().cpu().item())
	else:
		corr_value = math.nan
	target = vec_norm
	sse = (recon - target).pow(2).sum()
	sst = (target - target.mean(dim=0, keepdim=True)).pow(2).sum()
	r2 = 1.0 - sse / sst.clamp_min(1.0e-8)
	return {
		"labels": labels,
		"task_context_pairwise_matrix": pairwise,
		"task_context_L2": {
			f"{left}_vs_{right}": float(torch.linalg.vector_norm(context[i] - context[j]).item())
			for i, left in enumerate(labels)
			for j, right in enumerate(labels)
			if i < j
		},
		"task_vec_norm_pairwise_L2": {
			f"{left}_vs_{right}": float(torch.linalg.vector_norm(vec_norm[i] - vec_norm[j]).item())
			for i, left in enumerate(labels)
			for j, right in enumerate(labels)
			if i < j
		},
		"task_vec_reconstruction_r2": float(r2.detach().cpu().item()),
		"ctx_task_distance_corr": corr_value,
		"context_vectors": {label: tensor_to_list(context[i]) for i, label in enumerate(labels)},
		"task_vec_norm": {label: tensor_to_list(vec_norm[i]) for i, label in enumerate(labels)},
		"task_recon": {label: tensor_to_list(recon[i]) for i, label in enumerate(labels)},
		"context_l2_summary": summarize_tensor(ctx_dist),
		"vec_l2_summary": summarize_tensor(vec_dist),
	}


@torch.no_grad()
def _forward_outputs(model, cfg, obs, action, vec):
	task = condition_batch(vec, (obs.shape[0],), obs.device)
	z = model.encode(obs, task)
	next_z = model.next(z, action, task)
	reward = two_hot_scalar(model.reward(z, action, task), cfg)
	q = two_hot_scalar(model.Q(z, action, task, return_type="all"), cfg).mean(dim=0)
	return {"latent": z, "next_latent": next_z, "reward": reward, "q": q}


def _output_delta(a, b):
	return {
		"latent_l2": summarize_tensor(l2(a["latent"], b["latent"])),
		"next_latent_l2": summarize_tensor(l2(a["next_latent"], b["next_latent"])),
		"reward_abs": summarize_tensor(abs_delta(a["reward"], b["reward"])),
		"q_abs": summarize_tensor(abs_delta(a["q"], b["q"])),
	}


@torch.no_grad()
def _sensitivity(model, cfg, conditions, obs, action, base_label):
	outputs = OrderedDict()
	for label, vec in conditions.items():
		outputs[label] = _forward_outputs(model, cfg, obs, action, vec)
	base = outputs[base_label]
	comparisons = OrderedDict()
	max_mean = 0.0
	for label, item in outputs.items():
		if label == base_label:
			continue
		delta = _output_delta(base, item)
		comparisons[label] = delta
		for key in ("latent_l2", "next_latent_l2", "reward_abs", "q_abs"):
			max_mean = max(max_mean, float(delta[key]["mean"]))
	return {"comparisons_vs_base": comparisons, "max_mean_delta": max_mean}


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


def _top1_changed(base_returns, other_returns):
	return float((torch.argmax(base_returns, dim=-1) != torch.argmax(other_returns, dim=-1)).float().mean().item())


@torch.no_grad()
def _ranking_sensitivity(model, cfg, conditions, obs, base_label, *, seed: int, horizon: int, num_candidates: int, action_dim: int):
	generator = torch.Generator(device=obs.device).manual_seed(int(seed))
	actions = torch.empty(obs.shape[0], horizon, num_candidates, action_dim, device=obs.device).uniform_(-1.0, 1.0, generator=generator)
	returns = OrderedDict()
	for label, vec in conditions.items():
		returns[label] = _trajectory_returns(model, cfg, obs, actions, vec)
	base_returns = returns[base_label]
	metrics = OrderedDict()
	for label, item in returns.items():
		if label == base_label:
			continue
		metrics[label] = {
			"return_abs": summarize_tensor(abs_delta(base_returns, item)),
			"top1_changed_rate": _top1_changed(base_returns, item),
		}
	return metrics


def _equivalence_report(disabled_model, zero_model, cfg, conditions, obs, action, base_label):
	disabled = _forward_outputs(disabled_model, cfg, obs, action, conditions[base_label])
	zero = _forward_outputs(zero_model, cfg, obs, action, conditions[base_label])
	return _output_delta(disabled, zero)


def _max_equivalence_delta(report: dict[str, Any]) -> float:
	out = 0.0
	for value in report.values():
		if isinstance(value, dict) and "max" in value:
			out = max(out, float(value["max"]))
	return out


def build_report(args):
	paths = require_existing_inputs(args, need_checkpoint=True, need_replay=True)
	device = select_device(args)
	torch.manual_seed(int(args.seed))
	repair_checkpoint = resolve(args.repair_checkpoint) if args.repair_checkpoint else paths["checkpoint"]
	if not repair_checkpoint.exists():
		raise FileNotFoundError(f"Repair checkpoint not found: {repair_checkpoint}")
	disabled_model, disabled_cfg, compat = _load_variant(
		args,
		paths["checkpoint"],
		device,
		repair_enabled=False,
		raw_residual_scale=0.0,
	)
	zero_model, _, _ = _load_variant(
		args,
		paths["checkpoint"],
		device,
		repair_enabled=True,
		raw_residual_scale=0.0,
	)
	repair_model, repair_cfg, _ = _load_variant(
		args,
		repair_checkpoint,
		device,
		repair_enabled=True,
		raw_residual_scale=float(args.raw_residual_scale),
	)
	conditions = load_task_conditions(args)
	batch = load_trimmed_replay_batch(args, compat, device)
	obs = batch["obs"]
	action = batch["action"]
	base_label = str(args.base_label)
	if base_label not in conditions:
		raise KeyError(f"base label {base_label!r} not found in conditions={list(conditions.keys())}")
	disabled_context = _contexts(disabled_model, conditions, device)
	repair_context = _contexts(repair_model, conditions, device)
	equivalence = _equivalence_report(disabled_model, zero_model, disabled_cfg, conditions, obs, action, base_label)
	disabled_sensitivity = _sensitivity(disabled_model, disabled_cfg, conditions, obs, action, base_label)
	repair_sensitivity = _sensitivity(repair_model, repair_cfg, conditions, obs, action, base_label)
	action_dim = int(compat["action_dim"] or repair_cfg.action_dim)
	ranking = {
		"disabled": _ranking_sensitivity(
			disabled_model,
			disabled_cfg,
			conditions,
			obs[: int(args.ranking_batch_size)],
			base_label,
			seed=int(args.seed),
			horizon=int(args.ranking_horizon),
			num_candidates=int(args.num_candidates),
			action_dim=action_dim,
		),
		"repair": _ranking_sensitivity(
			repair_model,
			repair_cfg,
			conditions,
			obs[: int(args.ranking_batch_size)],
			base_label,
			seed=int(args.seed),
			horizon=int(args.ranking_horizon),
			num_candidates=int(args.num_candidates),
			action_dim=action_dim,
		),
	}
	messages: list[dict[str, Any]] = []
	equivalence_max = _max_equivalence_delta(equivalence)
	if equivalence_max > float(args.equivalence_eps):
		add_message(messages, "FAIL", "disabled and repair-enabled alpha=0 outputs differ.", max_delta=equivalence_max)
	else:
		add_message(messages, "PASS", "disabled and repair-enabled alpha=0 outputs are equivalent.", max_delta=equivalence_max)
	real_pair_key = f"{args.task_a_label}_vs_{args.task_b_label}"
	repair_real_l2 = float(repair_context["task_context_L2"].get(real_pair_key, 0.0))
	if repair_real_l2 <= float(args.context_collapse_eps):
		add_message(messages, "WARNING", "repair task_context is still collapsed for the real task pair.", task_context_l2=repair_real_l2)
	if repair_context["task_vec_reconstruction_r2"] < float(args.recon_r2_warning):
		add_message(messages, "WARNING", "task_vec reconstruction R2 is still low; repair likely not trained yet.", r2=repair_context["task_vec_reconstruction_r2"])
	if repair_sensitivity["max_mean_delta"] + float(args.sensitivity_drop_eps) < disabled_sensitivity["max_mean_delta"]:
		add_message(messages, "WARNING", "repair sensitivity is lower than disabled baseline.", disabled=disabled_sensitivity["max_mean_delta"], repair=repair_sensitivity["max_mean_delta"])
	else:
		add_message(messages, "PASS", "latent/reward/Q sensitivity is preserved relative to disabled baseline.", disabled=disabled_sensitivity["max_mean_delta"], repair=repair_sensitivity["max_mean_delta"])
	report = {
		"status": status_from_messages(messages),
		"checkpoint": str(paths["checkpoint"]),
		"repair_checkpoint": str(repair_checkpoint),
		"device": str(device),
		"sample_size": int(obs.shape[0]),
		"conditions": {label: tensor_to_list(vec) for label, vec in conditions.items()},
		"base_label": base_label,
		"raw_residual_scale": float(args.raw_residual_scale),
		"disabled_alpha0_equivalence": equivalence,
		"disabled_alpha0_equivalence_max_delta": equivalence_max,
		"disabled_context": disabled_context,
		"repair_context": repair_context,
		"disabled_latent_reward_q_sensitivity": disabled_sensitivity,
		"repair_latent_reward_q_sensitivity": repair_sensitivity,
		"mppi_ranking_sensitivity": ranking,
		"thresholds": {
			"equivalence_eps": float(args.equivalence_eps),
			"context_collapse_eps": float(args.context_collapse_eps),
			"recon_r2_warning": float(args.recon_r2_warning),
			"sensitivity_drop_eps": float(args.sensitivity_drop_eps),
		},
		"messages": messages,
	}
	return report


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	add_common_args(parser)
	parser.add_argument("--base-label", default="00256")
	parser.add_argument("--repair-checkpoint", default=None)
	parser.add_argument("--raw-residual-scale", type=float, default=0.1)
	parser.add_argument("--equivalence-eps", type=float, default=1.0e-6)
	parser.add_argument("--context-collapse-eps", type=float, default=1.0e-5)
	parser.add_argument("--recon-r2-warning", type=float, default=0.1)
	parser.add_argument("--sensitivity-drop-eps", type=float, default=1.0e-8)
	parser.add_argument("--ranking-batch-size", type=int, default=32)
	parser.add_argument("--ranking-horizon", type=int, default=3)
	parser.add_argument("--num-candidates", type=int, default=64)
	args = parser.parse_args()
	out = output_dir(args) / "task_context_repair_probe.json"
	if args.dry_run:
		print_status("WARNING", [{"level": "WARNING", "message": "Dry-run requested; model not loaded."}])
		write_json({"status": "DRY_RUN", "output": str(out)}, out, dry_run=True)
		return 0
	report = build_report(args)
	print_status(report["status"], report.get("messages", []))
	write_json(report, out, dry_run=False)
	return 1 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
	raise SystemExit(main())
