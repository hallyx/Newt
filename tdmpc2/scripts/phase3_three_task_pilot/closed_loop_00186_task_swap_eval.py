#!/usr/bin/env python3
"""Read-only 00186 correct/swap/zero/random closed-loop evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_AUDIT_DIR = SCRIPT_DIR.parent / "model_task_sensitivity"
if str(MODEL_AUDIT_DIR) not in sys.path:
	sys.path.insert(0, str(MODEL_AUDIT_DIR))

import closed_loop_task_swap_eval as phase2  # noqa: E402
from _common import resolve, tensor_to_list, tvsr, write_json, write_text  # noqa: E402


DEFAULT_CHECKPOINT = (
	"logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_taskctx_repair_01125_00256_00186/"
	"20260712_phase3_1_00186_stage-3_asm-00186/models/latest.pt"
)
DEFAULT_REPLAY_01125 = (
	"logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_replay_from_01125/"
	"20260615_202326_launcher/replay/01125.pt"
)
DEFAULT_REPLAY_00256 = (
	"logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_taskctx_repair_01125_00256/"
	"20260708_taskctx_repair_phase2_launcher/replay/00256.pt"
)
DEFAULT_REPLAY_00186 = (
	"logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_taskctx_repair_01125_00256_00186/"
	"20260712_phase3_1_00186_launcher/replay/00186.pt"
)
DEFAULT_OUTPUT_DIR = Path("reports/phase3_three_task_pilot")


def _vec(path: str | Path) -> list[float]:
	return tensor_to_list(tvsr._unique_task_vec_from_replay(resolve(path))[0].float())


def _markdown(report: dict[str, Any]) -> str:
	lines = [
		"# Phase 3.1 00186 Closed-Loop Task-Vector Swap Eval",
		"",
		"本报告只做只读闭环评估；未修改模型、训练、sampler、reward、Q、policy 或 MPPI。",
		"",
		f"Status: `{report['status']}`",
		"",
		"| Condition | Relaxed | Strict | Process | Reward | Lateral mm | Keypoint mm | Jam | Failure modes |",
		"| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
	]
	for condition, item in report["results"].items():
		metrics = item.get("metrics") or {}
		modes = ", ".join(metrics.get("failure_modes") or []) or "none"
		lines.append(
			f"| `{condition}` | {float(metrics.get('relaxed_success') or 0):.3f} | "
			f"{float(metrics.get('strict_success') or 0):.3f} | {float(metrics.get('process_success') or 0):.3f} | "
			f"{float(metrics.get('reward') or 0):.2f} | {float(metrics.get('lateral_error_mm') or 0):.3f} | "
			f"{float(metrics.get('keypoint_error_mm') or 0):.3f} | {float(metrics.get('jamming_rate') or 0):.3f} | {modes} |"
		)
	lines.extend([
		"",
		f"- zero/random measurable effect: `{report['zero_random_measurable_effect']}`。",
		f"- correct relaxed acquisition gate: `{report['correct_relaxed_gate']}`。",
	])
	return "\n".join(lines) + "\n"


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
	parser.add_argument("--replay-01125", default=DEFAULT_REPLAY_01125)
	parser.add_argument("--replay-00256", default=DEFAULT_REPLAY_00256)
	parser.add_argument("--replay-00186", default=DEFAULT_REPLAY_00186)
	parser.add_argument("--python", default=phase2.DEFAULT_PYTHON)
	parser.add_argument("--config-dir", default=phase2.DEFAULT_CONFIG_DIR)
	parser.add_argument("--config-name", default=phase2.DEFAULT_CONFIG_NAME)
	parser.add_argument("--isaaclab-dir", default="/home/gpuserver/IsaacLab")
	parser.add_argument("--srsa-dir", default="/home/gpuserver/hx/github/srsa")
	parser.add_argument("--num-envs", type=int, default=256)
	parser.add_argument("--gpu-id", type=int, default=0)
	parser.add_argument("--episodes", type=int, default=20)
	parser.add_argument("--seed", type=int, default=1)
	parser.add_argument("--subprocess-timeout", type=float, default=900.0)
	parser.add_argument("--run-root", default=str(DEFAULT_OUTPUT_DIR / "closed_loop_00186_runs"))
	parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_DIR / "closed_loop_00186_task_swap.json"))
	parser.add_argument("--output-md", default=str(DEFAULT_OUTPUT_DIR / "closed_loop_00186_task_swap.md"))
	parser.add_argument("--reuse-existing", action="store_true")
	parser.add_argument("--dry-run", action="store_true")
	args = parser.parse_args()
	args.checkpoint = str(resolve(args.checkpoint))
	args.run_root = str(resolve(args.run_root))

	vecs = {
		"correct": _vec(args.replay_00186),
		"swap_01125": _vec(args.replay_01125),
		"swap_00256": _vec(args.replay_00256),
		"zero": [0.0] * 6,
	}
	generator = torch.Generator().manual_seed(int(args.seed))
	vecs["random"] = tensor_to_list(torch.empty(6).uniform_(-1.0, 1.0, generator=generator))
	results: dict[str, dict[str, Any]] = {}
	for condition in ("correct", "swap_01125", "swap_00256", "zero", "random"):
		force = None if condition == "correct" else vecs[condition]
		results[condition] = phase2._run_condition(
			args,
			assembly_id="00186",
			condition=condition,
			force_vec=force,
			run_root=resolve(args.run_root),
		)
	if args.dry_run:
		print("PASS dry-run")
		return 0
	correct = results["correct"]
	correct_metrics = correct.get("metrics") or {}
	effects = []
	for condition, item in results.items():
		if item.get("status") != "DONE":
			continue
		metrics = item.get("metrics") or {}
		metrics["failure_modes"] = phase2._failure_modes(metrics, correct_metrics if condition != "correct" else None)
		if condition != "correct":
			item["delta_vs_correct"] = phase2._delta_record(item, correct)
			if condition in {"zero", "random"}:
				delta = item["delta_vs_correct"]
				effects.append(
					abs(delta["reward"]) >= 5.0
					or abs(delta["strict_success"]) >= 0.05
					or abs(delta["process_success"]) >= 0.05
					or abs(delta["lateral_error_mm"]) >= 0.10
					or abs(delta["keypoint_error_mm"]) >= 0.25
				)
	failed = [name for name, item in results.items() if item.get("status") != "DONE"]
	report = {
		"status": "FAIL" if failed else "PASS",
		"checkpoint": args.checkpoint,
		"task_vectors": vecs,
		"results": results,
		"failed_conditions": failed,
		"correct_relaxed_gate": float(correct_metrics.get("relaxed_success") or 0.0) >= 0.75,
		"zero_random_measurable_effect": any(effects),
	}
	write_json(report, args.output_json)
	write_text(_markdown(report), args.output_md)
	print(report["status"])
	return 1 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
	raise SystemExit(main())
