#!/usr/bin/env python3
"""Audit size sensitivity of SRSA reward/success-relevant diagnostic quantities."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from _common import (
	DEFAULT_OUTPUT_DIR,
	DEFAULT_TASK_HASH_CSV,
	add_message,
	decode_task_vec,
	load_task_id_vectors,
	print_status,
	status_from_messages,
	write_json_report,
)


def _clamped_tolerance(base: float, scale: float, min_value: float | None, max_value: float | None) -> float:
	tol = base * float(scale)
	if min_value is not None:
		tol = max(tol, float(min_value))
	if max_value is not None:
		tol = min(tol, float(max_value))
	return float(tol)


def _metrics_for_task(task: dict[str, Any], args) -> dict[str, Any]:
	decoded = decode_task_vec(task["task_vec_6"])
	radial = max(float(decoded["radial_clearance_m_if_default_reference"]), 1.0e-8)
	target_depth = max(float(decoded["target_depth_m_if_default_reference"]), 1.0e-8)
	lateral_abs = float(args.lateral_error_abs_m)
	keypoint_abs = float(args.keypoint_error_abs_m)
	current_depth = float(args.current_depth_m)
	yaw_error = float(args.yaw_error_rad)
	force = float(args.force_n)
	strict_lateral_tol = _clamped_tolerance(radial, args.strict_lateral_tol_scale, args.strict_lateral_tol_min, args.strict_lateral_tol_max)
	relaxed_lateral_tol = _clamped_tolerance(radial, args.relaxed_lateral_tol_scale, args.relaxed_lateral_tol_min, args.relaxed_lateral_tol_max)
	strict_keypoint_tol = _clamped_tolerance(radial, args.strict_keypoint_tol_scale, args.strict_keypoint_tol_min, args.strict_keypoint_tol_max)
	relaxed_keypoint_tol = _clamped_tolerance(radial, args.relaxed_keypoint_tol_scale, args.relaxed_keypoint_tol_min, args.relaxed_keypoint_tol_max)
	depth_progress = current_depth / target_depth
	return {
		"task_id": task["task_id"],
		"assembly_id": task["assembly_id"],
		"task_hash": task["task_hash"],
		"task_vec_6": task["task_vec_6"],
		"decoded": decoded,
		"inputs": {
			"lateral_error_abs_m": lateral_abs,
			"keypoint_error_abs_m": keypoint_abs,
			"current_depth_m": current_depth,
			"yaw_error_rad": yaw_error,
			"force_n": force,
		},
		"metrics": {
			"lateral_error_abs": lateral_abs,
			"lateral_error_over_clearance": lateral_abs / radial,
			"keypoint_error_abs": keypoint_abs,
			"keypoint_error_over_clearance": keypoint_abs / radial,
			"depth_progress_current_over_target": depth_progress,
			"depth_remaining_m": max(0.0, target_depth - current_depth),
			"yaw_error": yaw_error,
			"yaw_required": bool(float(task["task_vec_6"][5]) > 0.5),
			"force_over_limit": force / max(float(args.force_limit_n), 1.0e-8),
			"strict_lateral_tol": strict_lateral_tol,
			"relaxed_lateral_tol": relaxed_lateral_tol,
			"strict_keypoint_tol": strict_keypoint_tol,
			"relaxed_keypoint_tol": relaxed_keypoint_tol,
			"strict_lateral_ok": lateral_abs <= strict_lateral_tol,
			"relaxed_lateral_ok": lateral_abs <= relaxed_lateral_tol,
			"strict_keypoint_ok": keypoint_abs <= strict_keypoint_tol,
			"relaxed_keypoint_ok": keypoint_abs <= relaxed_keypoint_tol,
			"strict_depth_ok": depth_progress >= float(args.strict_depth_fraction),
			"relaxed_depth_ok": depth_progress >= float(args.relaxed_depth_fraction),
		},
	}


def _metric_deltas(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
	deltas = {}
	metrics_a = a["metrics"]
	metrics_b = b["metrics"]
	for key, av in metrics_a.items():
		bv = metrics_b.get(key)
		if isinstance(av, bool) or isinstance(bv, bool):
			deltas[key] = float(int(bool(bv)) - int(bool(av)))
		elif isinstance(av, (int, float)) and isinstance(bv, (int, float)):
			deltas[key] = float(bv - av)
	return deltas


def build_report(args):
	messages: list[dict[str, Any]] = []
	tasks = load_task_id_vectors(args.task_hash_csv)
	task_a = tasks.get(str(args.task_id_a).zfill(5))
	task_b = tasks.get(str(args.task_id_b).zfill(5))
	if task_a is None or task_b is None:
		add_message(messages, "FAIL", "Missing task vectors in task_id_to_hash.csv.")
		return {"status": "FAIL", "messages": messages}
	rec_a = _metrics_for_task(task_a, args)
	rec_b = _metrics_for_task(task_b, args)
	deltas = _metric_deltas(rec_a, rec_b)
	nonzero = {key: value for key, value in deltas.items() if abs(float(value)) > float(args.eps)}
	if nonzero:
		add_message(messages, "PASS", "Size-dependent surrogate reward/success metrics change between task vectors.", changed_metrics=sorted(nonzero.keys()))
	else:
		add_message(messages, "FAIL", "All size-dependent surrogate reward/success metric deltas are near zero.")
	add_message(messages, "WARNING", "This script uses analytic diagnostic quantities, not the full Isaac reward function.")
	return {
		"status": status_from_messages(messages),
		"runtime_reward_fn_check": "SKIPPED",
		"method": "analytic_surrogate_from_task_vec_6_and_eval_success_formulas",
		"task_a": rec_a,
		"task_b": rec_b,
		"deltas_b_minus_a": deltas,
		"nonzero_deltas": nonzero,
		"code_evidence": [
			"tdmpc2/envs/isaaclab.py:_compute_srsa_eval_success computes depth_fraction=current_depth/target_depth.",
			"tdmpc2/envs/isaaclab.py:_compute_srsa_eval_success computes lateral/keypoint tolerances from radial_clearance.",
			"tdmpc2/envs/isaaclab.py:_compute_srsa_eval_success uses radial_clearance in jam thresholds.",
		],
		"messages": messages,
	}


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--task-hash-csv", default=str(DEFAULT_TASK_HASH_CSV))
	parser.add_argument("--task-id-a", default="01125")
	parser.add_argument("--task-id-b", default="00256")
	parser.add_argument("--lateral-error-abs-m", type=float, default=0.0010)
	parser.add_argument("--keypoint-error-abs-m", type=float, default=0.0020)
	parser.add_argument("--current-depth-m", type=float, default=0.0100)
	parser.add_argument("--yaw-error-rad", type=float, default=0.05)
	parser.add_argument("--force-n", type=float, default=5.0)
	parser.add_argument("--force-limit-n", type=float, default=50.0)
	parser.add_argument("--strict-lateral-tol-scale", type=float, default=2.0)
	parser.add_argument("--strict-lateral-tol-min", type=float, default=0.0005)
	parser.add_argument("--strict-lateral-tol-max", type=float, default=0.0020)
	parser.add_argument("--relaxed-lateral-tol-scale", type=float, default=2.0)
	parser.add_argument("--relaxed-lateral-tol-min", type=float, default=0.0010)
	parser.add_argument("--relaxed-lateral-tol-max", type=float, default=0.0030)
	parser.add_argument("--strict-keypoint-tol-scale", type=float, default=2.0)
	parser.add_argument("--strict-keypoint-tol-min", type=float, default=0.0010)
	parser.add_argument("--strict-keypoint-tol-max", type=float, default=0.0030)
	parser.add_argument("--relaxed-keypoint-tol-scale", type=float, default=2.0)
	parser.add_argument("--relaxed-keypoint-tol-min", type=float, default=0.0010)
	parser.add_argument("--relaxed-keypoint-tol-max", type=float, default=0.0030)
	parser.add_argument("--strict-depth-fraction", type=float, default=0.90)
	parser.add_argument("--relaxed-depth-fraction", type=float, default=0.85)
	parser.add_argument("--eps", type=float, default=1.0e-9)
	parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
	parser.add_argument("--dry-run", action="store_true")
	args = parser.parse_args()

	report = build_report(args)
	output_dir = Path(args.output_dir).expanduser()
	if not output_dir.is_absolute():
		output_dir = DEFAULT_OUTPUT_DIR.parent.parent / output_dir
	print_status(report["status"], report.get("messages", []))
	write_json_report(report, output_dir / "reward_size_sensitivity.json", dry_run=args.dry_run)
	return 1 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
	raise SystemExit(main())
