#!/usr/bin/env python3
"""Audit latent, dynamics, reward, and Q sensitivity to task_vec_6."""

from __future__ import annotations

import argparse
from collections import OrderedDict

import torch

from _common import (
	add_common_args,
	add_message,
	abs_delta,
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


@torch.no_grad()
def _forward(model, cfg, obs, action, vec, *, fixed_z=None):
	task = condition_batch(vec, (obs.shape[0],), obs.device)
	z = model.encode(obs, task)
	z_for_heads = z if fixed_z is None else fixed_z
	next_z = model.next(z_for_heads, action, task)
	reward = two_hot_scalar(model.reward(z_for_heads, action, task), cfg)
	q_logits = model.Q(z_for_heads, action, task, return_type="all")
	q = two_hot_scalar(q_logits, cfg).mean(dim=0)
	return {"task": task, "latent": z, "next_latent": next_z, "reward": reward, "q": q}


@torch.no_grad()
def _rollout(model, cfg, z0, action, vec, steps: int):
	task = condition_batch(vec, (z0.shape[0],), z0.device)
	z = z0
	out = []
	for _ in range(int(steps)):
		reward = two_hot_scalar(model.reward(z, action, task), cfg)
		z = model.next(z, action, task)
		out.append({"z": z, "reward": reward})
	return out


def _compare(base, other):
	return {
		"latent_l2": summarize_tensor(l2(base["latent"], other["latent"])),
		"next_latent_l2": summarize_tensor(l2(base["next_latent"], other["next_latent"])),
		"reward_abs": summarize_tensor(abs_delta(base["reward"], other["reward"])),
		"q_abs": summarize_tensor(abs_delta(base["q"], other["q"])),
	}


def _rollout_compare(base_rollout, other_rollout):
	items = []
	for step, (base, other) in enumerate(zip(base_rollout, other_rollout), start=1):
		items.append({
			"step": step,
			"latent_l2": summarize_tensor(l2(base["z"], other["z"])),
			"reward_abs": summarize_tensor(abs_delta(base["reward"], other["reward"])),
		})
	return items


@torch.no_grad()
def build_report(args):
	bundle = load_model_bundle(args)
	model = bundle["model"]
	cfg = bundle["cfg"]
	device = bundle["device"]
	conditions = load_task_conditions(args)
	batch = load_trimmed_replay_batch(args, bundle["compat"], device)
	obs = batch["obs"]
	action = batch["action"]
	base_label = str(args.base_label)
	swap_label = str(args.swap_label)
	if base_label not in conditions:
		raise KeyError(f"base label {base_label!r} not found in conditions={list(conditions.keys())}")
	if swap_label not in conditions:
		raise KeyError(f"swap label {swap_label!r} not found in conditions={list(conditions.keys())}")
	outputs: OrderedDict[str, dict] = OrderedDict()
	for label, vec in conditions.items():
		outputs[label] = _forward(model, cfg, obs, action, vec)
	fixed_z = outputs[base_label]["latent"].detach()
	fixed_z_outputs: OrderedDict[str, dict] = OrderedDict()
	for label, vec in conditions.items():
		fixed_z_outputs[label] = _forward(model, cfg, obs, action, vec, fixed_z=fixed_z)
	base = outputs[base_label]
	comparisons = OrderedDict()
	fixed_z_comparisons = OrderedDict()
	for label in conditions.keys():
		if label == base_label:
			continue
		comparisons[label] = _compare(base, outputs[label])
		fixed_z_comparisons[label] = _compare(
			{
				"latent": base["latent"],
				"next_latent": fixed_z_outputs[base_label]["next_latent"],
				"reward": fixed_z_outputs[base_label]["reward"],
				"q": fixed_z_outputs[base_label]["q"],
			},
			{
				"latent": outputs[label]["latent"],
				"next_latent": fixed_z_outputs[label]["next_latent"],
				"reward": fixed_z_outputs[label]["reward"],
				"q": fixed_z_outputs[label]["q"],
			},
		)
	rollout = None
	if int(args.rollout_steps) > 0:
		base_rollout = _rollout(model, cfg, fixed_z, action, conditions[base_label], int(args.rollout_steps))
		rollout = OrderedDict()
		for label, vec in conditions.items():
			if label == base_label:
				continue
			rollout[label] = _rollout_compare(base_rollout, _rollout(model, cfg, fixed_z, action, vec, int(args.rollout_steps)))
	swap = comparisons[swap_label]
	zero = comparisons.get("zero", {})
	random = comparisons.get("random", {})
	noise_eps = float(args.noise_eps)
	max_key_delta = max(
		float(swap["next_latent_l2"]["mean"]),
		float(swap["reward_abs"]["mean"]),
		float(swap["q_abs"]["mean"]),
		float((zero.get("next_latent_l2") or {}).get("mean", 0.0)),
		float((zero.get("reward_abs") or {}).get("mean", 0.0)),
		float((zero.get("q_abs") or {}).get("mean", 0.0)),
		float((random.get("next_latent_l2") or {}).get("mean", 0.0)),
		float((random.get("reward_abs") or {}).get("mean", 0.0)),
		float((random.get("q_abs") or {}).get("mean", 0.0)),
	)
	messages: list[dict] = []
	if max_key_delta <= noise_eps:
		add_message(messages, "WARNING", "model ignores task_vec despite runtime task effectiveness.", max_key_delta=max_key_delta, noise_eps=noise_eps)
	else:
		add_message(messages, "PASS", "At least one latent/dynamics/reward/Q metric changes above noise threshold.", max_key_delta=max_key_delta)
	report = {
		"status": status_from_messages(messages),
		"checkpoint": str(bundle["paths"]["checkpoint"]),
		"replay": str(bundle["paths"]["task_b_replay"]),
		"device": str(device),
		"sample_size": int(obs.shape[0]),
		"sample_indices_first10": [int(x) for x in batch["indices"][:10].detach().cpu().tolist()],
		"conditions": {label: tensor_to_list(vec) for label, vec in conditions.items()},
		"base_label": base_label,
		"swap_label": swap_label,
		"comparisons_vs_base": comparisons,
		"fixed_z_comparisons_vs_base": fixed_z_comparisons,
		"multi_step_rollout_divergence": rollout,
		"latent_delta_correct_swap": swap["latent_l2"],
		"latent_delta_correct_zero": (zero.get("latent_l2") or {}),
		"next_latent_delta_correct_swap": swap["next_latent_l2"],
		"next_latent_delta_correct_zero": (zero.get("next_latent_l2") or {}),
		"reward_delta_correct_swap": swap["reward_abs"],
		"reward_delta_correct_zero": (zero.get("reward_abs") or {}),
		"Q_delta_correct_swap": swap["q_abs"],
		"Q_delta_correct_zero": (zero.get("q_abs") or {}),
		"thresholds": {"noise_eps": noise_eps},
		"messages": messages,
	}
	return report


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	add_common_args(parser)
	parser.add_argument("--base-label", default="00256")
	parser.add_argument("--swap-label", default="01125")
	parser.add_argument("--noise-eps", type=float, default=1.0e-6)
	parser.add_argument("--rollout-steps", type=int, default=5)
	args = parser.parse_args()
	if args.dry_run:
		out = output_dir(args) / "latent_dynamics_reward_sensitivity.json"
		print_status("WARNING", [{"level": "WARNING", "message": "Dry-run requested; model not loaded."}])
		write_json({"status": "DRY_RUN", "output": str(out)}, out, dry_run=True)
		return 0
	report = build_report(args)
	print_status(report["status"], report.get("messages", []))
	write_json(report, output_dir(args) / "latent_dynamics_reward_sensitivity.json", dry_run=False)
	return 1 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
	raise SystemExit(main())

