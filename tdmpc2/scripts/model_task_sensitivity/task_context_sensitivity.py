#!/usr/bin/env python3
"""Audit AxialTaskEncoder task_context sensitivity for SRSA task_vec_6."""

from __future__ import annotations

import argparse
from collections import OrderedDict

import torch
import torch.nn.functional as F

from _common import (
	DEFAULT_OUTPUT_DIR,
	add_common_args,
	add_message,
	condition_batch,
	load_model_bundle,
	load_task_conditions,
	output_dir,
	print_status,
	status_from_messages,
	summarize_tensor,
	tensor_to_list,
	write_json,
)


def _pairwise(contexts: OrderedDict[str, torch.Tensor]) -> dict[str, dict[str, dict[str, float]]]:
	labels = list(contexts.keys())
	matrix: dict[str, dict[str, dict[str, float]]] = {}
	for left in labels:
		matrix[left] = {}
		for right in labels:
			a = contexts[left].reshape(-1)
			b = contexts[right].reshape(-1)
			matrix[left][right] = {
				"l2": float(torch.linalg.vector_norm(a - b).item()),
				"cosine_similarity": float(F.cosine_similarity(a.view(1, -1), b.view(1, -1), dim=-1).item()),
				"max_abs": float((a - b).abs().max().item()),
			}
	return matrix


@torch.no_grad()
def build_report(args):
	bundle = load_model_bundle(args)
	model = bundle["model"]
	device = bundle["device"]
	conditions = load_task_conditions(args)
	dummy = torch.zeros(1, int(bundle["compat"]["obs_dim"]), device=device)
	contexts: OrderedDict[str, torch.Tensor] = OrderedDict()
	for label, vec in conditions.items():
		task = condition_batch(vec, (1,), device)
		context = model.task_context(dummy, task)
		if context is None:
			raise RuntimeError("WorldModel.task_context() returned None; checkpoint is not task-conditioned.")
		contexts[label] = context.squeeze(0).detach().float().cpu()
	stack = torch.stack([item.reshape(-1) for item in contexts.values()], dim=0)
	pairwise = _pairwise(contexts)
	task_a = str(args.task_a_label)
	task_b = str(args.task_b_label)
	real_l2 = float(pairwise[task_a][task_b]["l2"])
	zero_l2 = max(
		float(pairwise[task_a].get("zero", {}).get("l2", 0.0)),
		float(pairwise[task_b].get("zero", {}).get("l2", 0.0)),
	)
	random_l2 = max(
		float(pairwise[task_a].get("random", {}).get("l2", 0.0)),
		float(pairwise[task_b].get("random", {}).get("l2", 0.0)),
	)
	messages: list[dict] = []
	if real_l2 <= float(args.collapse_l2_eps):
		add_message(messages, "WARNING", "01125 vs 00256 task_context_L2 is near zero; task_context collapse likely.", l2=real_l2)
	if ("zero" in conditions or "random" in conditions) and max(zero_l2, random_l2) <= float(args.fail_l2_eps):
		add_message(messages, "FAIL", "zero/random task_context is also nearly identical to real task_context.", zero_l2=zero_l2, random_l2=random_l2)
	if not messages:
		add_message(messages, "PASS", "Task contexts are separated above collapse thresholds.", l2_01125_00256=real_l2)
	report = {
		"status": status_from_messages(messages),
		"checkpoint": str(bundle["paths"]["checkpoint"]),
		"config": str(bundle["paths"]["config"]),
		"device": str(device),
		"conditions": {label: tensor_to_list(vec) for label, vec in conditions.items()},
		"contexts": {label: tensor_to_list(ctx) for label, ctx in contexts.items()},
		"task_context_dim": int(stack.shape[-1]),
		"task_context_pairwise_matrix": pairwise,
		"task_context_L2_01125_00256": real_l2,
		"task_context_per_dim_std": tensor_to_list(stack.std(dim=0, unbiased=False)),
		"task_context_per_dim_std_summary": summarize_tensor(stack.std(dim=0, unbiased=False)),
		"thresholds": {
			"collapse_l2_eps": float(args.collapse_l2_eps),
			"fail_l2_eps": float(args.fail_l2_eps),
		},
		"messages": messages,
	}
	return report


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	add_common_args(parser)
	parser.add_argument("--collapse-l2-eps", type=float, default=1.0e-5)
	parser.add_argument("--fail-l2-eps", type=float, default=1.0e-6)
	args = parser.parse_args()
	if args.dry_run:
		out = output_dir(args) / "task_context_sensitivity.json"
		print_status("WARNING", [{"level": "WARNING", "message": "Dry-run requested; model not loaded."}])
		write_json({"status": "DRY_RUN", "output": str(out)}, out, dry_run=True)
		return 0
	report = build_report(args)
	print_status(report["status"], report.get("messages", []))
	write_json(report, output_dir(args) / "task_context_sensitivity.json", dry_run=False)
	return 1 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
	raise SystemExit(main())

