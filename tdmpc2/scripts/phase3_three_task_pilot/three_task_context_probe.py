#!/usr/bin/env python3
"""Read-only three-task context/reconstruction structure probe for Phase 3.1."""

from __future__ import annotations

import argparse
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

from _common import (  # noqa: E402
	add_common_args,
	load_model_bundle,
	print_status,
	resolve,
	summarize_tensor,
	tensor_to_list,
	tvsr,
	write_json,
)


DEFAULT_OUTPUT = "reports/phase3_three_task_pilot/three_task_context_probe.json"
DEFAULT_TASK_C_REPLAY = (
	"logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_taskctx_repair_01125_00256_00186/"
	"20260712_phase3_1_00186_launcher/replay/00186.pt"
)


def _task_vec(replay_fp: str | Path) -> torch.Tensor:
	return tvsr._unique_task_vec_from_replay(resolve(replay_fp))[0].float()


@torch.no_grad()
def build_report(args: argparse.Namespace) -> dict[str, Any]:
	bundle = load_model_bundle(args)
	model = bundle["model"]
	device = bundle["device"]
	conditions: OrderedDict[str, torch.Tensor] = OrderedDict([
		(str(args.task_a_label), _task_vec(args.task_a_replay)),
		(str(args.task_b_label), _task_vec(args.task_b_replay)),
		(str(args.task_c_label), _task_vec(args.task_c_replay)),
	])
	task = torch.stack([vec.to(device=device, dtype=torch.float32) for vec in conditions.values()], dim=0)
	dummy = task.new_zeros(task.shape[0], 1)
	info = model.task_context_repair_info(dummy, task, reconstruct=True)
	if info is None:
		raise RuntimeError("WorldModel.task_context_repair_info() returned None")
	context = info["task_context"]
	vec_norm = info["task_vec_norm"]
	recon = info["task_recon"]
	labels = list(conditions.keys())

	pairwise: OrderedDict[str, dict[str, float]] = OrderedDict()
	ctx_values = []
	vec_values = []
	for i, left in enumerate(labels):
		pairwise[left] = OrderedDict()
		for j, right in enumerate(labels):
			distance = float(torch.linalg.vector_norm(context[i] - context[j]).item())
			pairwise[left][right] = distance
			if i < j:
				ctx_values.append(distance)
				vec_values.append(float(torch.linalg.vector_norm(vec_norm[i] - vec_norm[j]).item()))

	target = vec_norm
	sse = (recon - target).pow(2).sum()
	sst = (target - target.mean(dim=0, keepdim=True)).pow(2).sum().clamp_min(1.0e-8)
	r2 = float((1.0 - sse / sst).item())
	recon_loss = float(torch.mean((recon - target).pow(2)).item())
	ctx_tensor = torch.tensor(ctx_values, dtype=torch.float32)
	vec_tensor = torch.tensor(vec_values, dtype=torch.float32)
	if ctx_tensor.numel() >= 2 and ctx_tensor.std() > 1.0e-8 and vec_tensor.std() > 1.0e-8:
		corr = float(torch.corrcoef(torch.stack([ctx_tensor, vec_tensor]))[0, 1].item())
	else:
		corr = math.nan

	d_near = pairwise[str(args.task_a_label)][str(args.task_b_label)]
	d_c_a = pairwise[str(args.task_c_label)][str(args.task_a_label)]
	d_c_b = pairwise[str(args.task_c_label)][str(args.task_b_label)]
	collapse = bool(min(ctx_values) <= float(args.collapse_eps))
	structure_gate = bool(d_near < min(d_c_a, d_c_b))
	messages = []
	if collapse:
		messages.append({"level": "FAIL", "message": "At least one three-task context pair is collapsed."})
	else:
		messages.append({"level": "PASS", "message": "All three real task contexts are separated."})
	if not structure_gate:
		messages.append({"level": "WARNING", "message": "01125/00256 context distance is not the smallest pair."})
	else:
		messages.append({"level": "PASS", "message": "Context geometry preserves the expected near-pair ordering."})
	if r2 < float(args.min_recon_r2):
		messages.append({"level": "FAIL", "message": f"Three-task reconstruction R2={r2:.6f} is below threshold."})
	else:
		messages.append({"level": "PASS", "message": f"Three-task reconstruction R2={r2:.6f}."})
	levels = {item["level"] for item in messages}
	status = "FAIL" if "FAIL" in levels else ("WARNING" if "WARNING" in levels else "PASS")
	return {
		"status": status,
		"checkpoint": str(bundle["paths"]["checkpoint"]),
		"device": str(device),
		"conditions": {label: tensor_to_list(vec) for label, vec in conditions.items()},
		"task_context_pairwise_matrix": pairwise,
		"task_context_l2": {
			f"{labels[i]}_vs_{labels[j]}": pairwise[labels[i]][labels[j]]
			for i in range(len(labels)) for j in range(i + 1, len(labels))
		},
		"task_vec_norm_pairwise_l2": {
			f"{labels[i]}_vs_{labels[j]}": vec_values.pop(0)
			for i in range(len(labels)) for j in range(i + 1, len(labels))
		},
		"task_reconstruction_r2": r2,
		"task_recon_loss": recon_loss,
		"ctx_task_distance_corr": corr,
		"context_collapse": collapse,
		"near_pair_structure_gate": structure_gate,
		"context_l2_summary": summarize_tensor(ctx_tensor),
		"context_vectors": {label: tensor_to_list(context[i]) for i, label in enumerate(labels)},
		"task_recon": {label: tensor_to_list(recon[i]) for i, label in enumerate(labels)},
		"thresholds": {"collapse_eps": float(args.collapse_eps), "min_recon_r2": float(args.min_recon_r2)},
		"messages": messages,
	}


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	add_common_args(parser)
	parser.set_defaults(include_zero=False, include_random=False)
	parser.add_argument("--task-c-label", default="00186")
	parser.add_argument("--task-c-replay", default=DEFAULT_TASK_C_REPLAY)
	parser.add_argument("--output", default=DEFAULT_OUTPUT)
	parser.add_argument("--collapse-eps", type=float, default=1.0e-4)
	parser.add_argument("--min-recon-r2", type=float, default=0.8)
	args = parser.parse_args()
	if args.dry_run:
		print(f"PASS dry-run: would write {resolve(args.output)}")
		return 0
	report = build_report(args)
	print_status(report["status"], report["messages"])
	write_json(report, args.output)
	return 1 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
	raise SystemExit(main())
