#!/usr/bin/env python3
"""Probe which SRSA size overrides are representable without changing training code."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from _common import (
	AXIAL_TASK_VEC_FIELDS,
	DEFAULT_OUTPUT_DIR,
	DEFAULT_TASK_HASH_CSV,
	add_message,
	close_env,
	collect_runtime_geometry,
	compare_geometry_records,
	decode_task_vec,
	launch_probe_env,
	load_task_id_vectors,
	print_status,
	status_from_messages,
	task_vec_hash,
	write_json_report,
)


def _case(name: str, base_vec: list[float], new_vec: list[float], env_overrides: dict[str, Any]) -> dict[str, Any]:
	changed = []
	for index, field in enumerate(AXIAL_TASK_VEC_FIELDS):
		if abs(float(base_vec[index]) - float(new_vec[index])) > 1.0e-12:
			changed.append(field)
	return {
		"name": name,
		"base_task_vec_6": base_vec,
		"override_task_vec_6": new_vec,
		"base_hash": task_vec_hash(base_vec),
		"override_hash": task_vec_hash(new_vec),
		"changed_fields": changed,
		"env_override_keys": sorted(env_overrides.keys()),
		"env_overrides": env_overrides,
		"base_decode": decode_task_vec(base_vec),
		"override_decode": decode_task_vec(new_vec),
	}


def build_override_cases(base_vec: list[float], args) -> list[dict[str, Any]]:
	base_vec = [float(x) for x in base_vec[:6]]
	cases = []

	scale = math.exp(base_vec[1]) * float(args.scale_multiplier)
	scale_vec = list(base_vec)
	scale_vec[1] = math.log(max(scale, 1.0e-8))
	cases.append(_case(
		"scale_multiplier",
		base_vec,
		scale_vec,
		{
			"srsa_axial_scale_range": f"{scale:.12g},{scale:.12g}",
			"srsa_axial_fixed_plug_scale": False,
		},
	))

	clearance_vec = list(base_vec)
	clearance_vec[2] *= float(args.clearance_multiplier)
	clearance_vec[3] *= float(args.clearance_multiplier)
	decoded_clearance = decode_task_vec(clearance_vec)
	diametral = decoded_clearance["diametral_clearance_m_if_default_reference"]
	cases.append(_case(
		"clearance_multiplier",
		base_vec,
		clearance_vec,
		{
			"srsa_axial_clearance_range": f"{diametral:.12g},{diametral:.12g}",
			"srsa_axial_clearance_base": diametral,
			"srsa_axial_clearance_jitter_ratio": 0.0,
		},
	))

	depth_vec = list(base_vec)
	depth_vec[4] *= float(args.depth_multiplier)
	decoded_depth = decode_task_vec(depth_vec)
	target_depth = decoded_depth["target_depth_m_if_default_reference"]
	cases.append(_case(
		"target_depth_multiplier",
		base_vec,
		depth_vec,
		{
			"srsa_axial_depth_range": f"{target_depth:.12g},{target_depth:.12g}",
			"srsa_axial_target_depth_range": f"{target_depth:.12g},{target_depth:.12g}",
			"srsa_axial_depth_base": target_depth,
			"srsa_axial_depth_jitter_ratio": 0.0,
		},
	))

	yaw_vec = list(base_vec)
	yaw_vec[5] = float(args.yaw_requirement)
	cases.append(_case(
		"yaw_requirement",
		base_vec,
		yaw_vec,
		{"srsa_axial_yaw_requirement": bool(float(args.yaw_requirement) > 0.5)},
	))
	return cases


def _launch_override_case(assembly_id: str, case: dict[str, Any]) -> dict[str, Any]:
	base_env = None
	override_env = None
	try:
		_, base_env = launch_probe_env(assembly_id)
		_, override_env = launch_probe_env(assembly_id, extra_overrides=case["env_overrides"])
		base_record = collect_runtime_geometry(base_env)
		override_record = collect_runtime_geometry(override_env)
		return {
			"runtime_check": "DONE",
			"base_record": base_record,
			"override_record": override_record,
			"comparison": compare_geometry_records(base_record, override_record),
		}
	except Exception as exc:
		return {"runtime_check": "FAILED", "error": repr(exc)}
	finally:
		if base_env is not None:
			close_env(base_env)
		if override_env is not None:
			close_env(override_env)


def build_report(args):
	messages: list[dict[str, Any]] = []
	tasks = load_task_id_vectors(args.task_hash_csv)
	task = tasks.get(str(args.assembly_id).zfill(5))
	if task is None:
		add_message(messages, "FAIL", "Missing assembly/task vector in task_id_to_hash.csv.", assembly_id=str(args.assembly_id).zfill(5))
		return {"status": "FAIL", "messages": messages}
	cases = build_override_cases(task["task_vec_6"], args)
	if args.launch_env:
		for case in cases:
			case["runtime_probe"] = _launch_override_case(task["assembly_id"], case)
			if case["runtime_probe"].get("runtime_check") == "FAILED":
				add_message(messages, "FAIL", "Runtime override probe failed.", case=case["name"], error=case["runtime_probe"].get("error"))
	else:
		for case in cases:
			case["runtime_probe"] = {
				"runtime_check": "SKIPPED",
				"reason": "Use --launch-env to inspect real asset/scale/AABB/target changes.",
			}
		add_message(messages, "WARNING", "Runtime override probe skipped; static support only.")

	for case in cases:
		if not case["changed_fields"]:
			add_message(messages, "FAIL", "Override case does not change any task_vec field.", case=case["name"])
		else:
			add_message(messages, "PASS", "Override case changes task_vec fields.", case=case["name"], changed_fields=case["changed_fields"])

	report = {
		"status": status_from_messages(messages),
		"assembly_id": task["assembly_id"],
		"task_id": task["task_id"],
		"base_task_hash": task["task_hash"],
		"base_task_vec_6": task["task_vec_6"],
		"launch_env": bool(args.launch_env),
		"override_cases": cases,
		"messages": messages,
	}
	return report


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--task-hash-csv", default=str(DEFAULT_TASK_HASH_CSV))
	parser.add_argument("--assembly-id", default="00256")
	parser.add_argument("--scale-multiplier", type=float, default=1.10)
	parser.add_argument("--clearance-multiplier", type=float, default=1.50)
	parser.add_argument("--depth-multiplier", type=float, default=1.20)
	parser.add_argument("--yaw-requirement", type=float, default=1.0)
	parser.add_argument("--launch-env", action="store_true")
	parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
	parser.add_argument("--dry-run", action="store_true")
	args = parser.parse_args()

	report = build_report(args)
	output_dir = Path(args.output_dir).expanduser()
	if not output_dir.is_absolute():
		output_dir = DEFAULT_OUTPUT_DIR.parent.parent / output_dir
	print_status(report["status"], report.get("messages", []))
	write_json_report(report, output_dir / "size_override_probe.json", dry_run=args.dry_run)
	return 1 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
	raise SystemExit(main())
