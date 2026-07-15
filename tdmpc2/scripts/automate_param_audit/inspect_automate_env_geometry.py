#!/usr/bin/env python3
"""Inspect AutoMate/SRSA runtime geometry signals for two assemblies."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

from _common import (
	DEFAULT_OUTPUT_DIR,
	DEFAULT_TASK_HASH_CSV,
	add_message,
	append_runtime_error,
	close_env,
	collect_runtime_geometry,
	compare_geometry_records,
	compare_task_vectors,
	launch_probe_env,
	load_json,
	load_task_id_vectors,
	print_status,
	status_from_messages,
	write_json_report,
)


def _static_record(task: dict[str, Any], *, launch_env: bool) -> dict[str, Any]:
	return {
		"status": "SKIPPED" if not launch_env else "PENDING",
		"runtime_geometry_check": "SKIPPED" if not launch_env else "PENDING",
		"reason": None if launch_env else "Use --launch-env to inspect real USD/mesh/collision/prim runtime data.",
		"task_id": task.get("task_id"),
		"assembly_id": task.get("assembly_id"),
		"task_hash": task.get("task_hash"),
		"task_vec_6": task.get("task_vec_6"),
		"replay_fp": task.get("replay_fp"),
	}


def _launch_record(task: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
	env = None
	try:
		cfg, env = launch_probe_env(str(task["assembly_id"]))
		record = collect_runtime_geometry(env, cfg=cfg)
		record.update({
			"status": "PASS",
			"runtime_geometry_check": "DONE",
			"task_id": task.get("task_id"),
			"assembly_id": task.get("assembly_id"),
			"task_hash": task.get("task_hash"),
			"expected_task_vec_6_from_replay": task.get("task_vec_6"),
		})
		return record
	except Exception as exc:
		log_path = append_runtime_error(exc, context=f"inspect geometry assembly_id={task.get('assembly_id')}", output_dir=output_dir)
		return {
			"status": "FAIL",
			"runtime_geometry_check": "FAILED",
			"task_id": task.get("task_id"),
			"assembly_id": task.get("assembly_id"),
			"task_hash": task.get("task_hash"),
			"expected_task_vec_6_from_replay": task.get("task_vec_6"),
			"error": repr(exc),
			"runtime_launch_error_log": str(log_path),
		}
	finally:
		if env is not None:
			close_env(env)


def _output_dir(args) -> Path:
	output_dir = Path(args.output_dir).expanduser()
	if not output_dir.is_absolute():
		output_dir = DEFAULT_OUTPUT_DIR.parent.parent / output_dir
	return output_dir.resolve()


def _runtime_record_path(output_dir: Path, task_id: str) -> Path:
	return output_dir / f"geometry_{str(task_id).zfill(5)}_runtime.json"


def _run_single_runtime_probe(task: dict[str, Any], args, *, output_dir: Path) -> dict[str, Any]:
	task_id = str(task["task_id"]).zfill(5)
	cmd = [
		sys.executable,
		str(Path(__file__).resolve()),
		"--launch-env",
		"--single-task-id",
		task_id,
		"--task-hash-csv",
		str(args.task_hash_csv),
		"--output-dir",
		str(output_dir),
	]
	try:
		result = subprocess.run(
			cmd,
			cwd=str(Path(__file__).resolve().parents[3]),
			text=True,
			stdout=subprocess.PIPE,
			stderr=subprocess.STDOUT,
			check=False,
			timeout=float(args.subprocess_timeout),
		)
	except subprocess.TimeoutExpired as exc:
		output = exc.output or ""
		if isinstance(output, bytes):
			output = output.decode("utf-8", errors="replace")
		error = TimeoutError(f"runtime geometry subprocess timed out for task_id={task_id} after {args.subprocess_timeout}s")
		log_path = append_runtime_error(error, context=f"inspect geometry subprocess timeout task_id={task_id}", output_dir=output_dir)
		return {
			"status": "FAIL",
			"runtime_geometry_check": "FAILED",
			"task_id": task_id,
			"assembly_id": task.get("assembly_id"),
			"task_hash": task.get("task_hash"),
			"expected_task_vec_6_from_replay": task.get("task_vec_6"),
			"error": repr(error),
			"runtime_launch_error_log": str(log_path),
			"subprocess_timeout_s": float(args.subprocess_timeout),
			"subprocess_output_tail": output[-6000:],
		}
	record_path = _runtime_record_path(output_dir, task_id)
	record = load_json(record_path, default=None)
	if isinstance(record, dict):
		record["subprocess_returncode"] = result.returncode
		if result.returncode != 0 or record.get("status") == "FAIL":
			record["subprocess_output_tail"] = result.stdout[-6000:]
		return record
	error = RuntimeError(f"runtime geometry subprocess failed for task_id={task_id} rc={result.returncode}\n{result.stdout[-6000:]}")
	log_path = append_runtime_error(error, context=f"inspect geometry subprocess task_id={task_id}", output_dir=output_dir)
	return {
		"status": "FAIL",
		"runtime_geometry_check": "FAILED",
		"task_id": task_id,
		"assembly_id": task.get("assembly_id"),
		"task_hash": task.get("task_hash"),
		"expected_task_vec_6_from_replay": task.get("task_vec_6"),
		"error": repr(error),
		"runtime_launch_error_log": str(log_path),
		"subprocess_returncode": result.returncode,
		"subprocess_output_tail": result.stdout[-6000:],
	}


def _run_single_mode(args) -> int:
	messages: list[dict[str, Any]] = []
	output_dir = _output_dir(args)
	tasks = load_task_id_vectors(args.task_hash_csv)
	task_id = str(args.single_task_id).zfill(5)
	task = tasks.get(task_id)
	if task is None:
		add_message(messages, "FAIL", f"Missing task vector for task_id={task_id}.")
		record = {"status": "FAIL", "runtime_geometry_check": "FAILED", "task_id": task_id, "messages": messages}
	else:
		record = _launch_record(task, output_dir=output_dir)
		if record.get("status") == "FAIL":
			add_message(messages, "FAIL", f"Runtime geometry probe failed for task_id={task_id}.", error=record.get("error"))
		else:
			add_message(messages, "PASS", f"Runtime geometry probe completed for task_id={task_id}.")
		record["messages"] = messages
		record["status"] = status_from_messages(messages)
	print_status(record["status"], messages)
	write_json_report(record, _runtime_record_path(output_dir, task_id), dry_run=args.dry_run)
	return 1 if record["status"] == "FAIL" else 0


def build_report(args):
	messages: list[dict[str, Any]] = []
	output_dir = _output_dir(args)
	tasks = load_task_id_vectors(args.task_hash_csv)
	task_a = tasks.get(str(args.task_id_a).zfill(5))
	task_b = tasks.get(str(args.task_id_b).zfill(5))
	if task_a is None or task_b is None:
		add_message(messages, "FAIL", "Missing task vectors in task_id_to_hash.csv.")
		return None, None, {"status": "FAIL", "messages": messages}

	if args.launch_env and args.dry_run:
		rec_a = _static_record(task_a, launch_env=False)
		rec_b = _static_record(task_b, launch_env=False)
		add_message(messages, "WARNING", "Dry-run requested; runtime geometry env launch skipped.")
	elif args.launch_env:
		rec_a = _run_single_runtime_probe(task_a, args, output_dir=output_dir)
		rec_b = _run_single_runtime_probe(task_b, args, output_dir=output_dir)
	else:
		rec_a = _static_record(task_a, launch_env=False)
		rec_b = _static_record(task_b, launch_env=False)
		add_message(messages, "WARNING", "Runtime geometry check skipped; rerun with --launch-env for real asset/AABB/scale data.")

	compare = {
		"task_vector_compare": compare_task_vectors(task_a, task_b),
		"runtime_geometry_check": "DONE" if args.launch_env and not args.dry_run else "SKIPPED",
		"geometry_comparison": compare_geometry_records(rec_a, rec_b) if args.launch_env and not args.dry_run else None,
		"asset_path_differs": None,
		"visual_aabb_differs": None,
		"collision_aabb_differs": None,
		"prim_scale_differs": None,
		"target_depth_differs": None,
		"success_threshold_differs": None,
		"geometry_effective": None,
		"objective_effective": None,
	}

	if args.launch_env and not args.dry_run:
		if rec_a.get("status") == "FAIL" or rec_b.get("status") == "FAIL":
			add_message(messages, "FAIL", "One or both runtime geometry probes failed.", records=[rec_a.get("error"), rec_b.get("error")])
		else:
			geom_cmp = compare["geometry_comparison"] or {}
			for key in (
				"asset_path_differs",
				"visual_aabb_differs",
				"collision_aabb_differs",
				"prim_scale_differs",
				"target_depth_differs",
				"success_threshold_differs",
				"geometry_effective",
				"objective_effective",
			):
				compare[key] = geom_cmp.get(key)
			if geom_cmp.get("num_comparable_fields", 0) <= 0:
				add_message(messages, "WARNING", "Runtime env launched, but no comparable geometry/objective fields were found.")
			elif (
				compare["task_vector_compare"]["task_vecs_differ"]
				and not bool(compare["geometry_effective"])
				and not bool(compare["objective_effective"])
			):
				add_message(
					messages,
					"FAIL",
					"task_vec differs but asset path/AABB/scale/target depth/threshold are identical; size parameters may be label-only.",
					task_ids=[task_a["task_id"], task_b["task_id"]],
				)
			else:
				if bool(compare["geometry_effective"]):
					add_message(messages, "PASS", "Runtime asset/geometry records differ for the compared assemblies.")
				if bool(compare["objective_effective"]):
					add_message(messages, "PASS", "Runtime target/threshold records differ for the compared assemblies.")
				if not bool(compare["geometry_effective"]) and not bool(compare["objective_effective"]):
					add_message(messages, "WARNING", "Runtime records differ only outside the requested geometry/objective fields.")

	if not messages:
		add_message(messages, "PASS", "Geometry probe completed.")
	compare["status"] = status_from_messages(messages)
	compare["messages"] = messages
	return rec_a, rec_b, compare


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--task-hash-csv", default=str(DEFAULT_TASK_HASH_CSV))
	parser.add_argument("--task-id-a", default="01125")
	parser.add_argument("--task-id-b", default="00256")
	parser.add_argument("--launch-env", action="store_true", help="Actually launch IsaacLab/SRSA envs for geometry inspection.")
	parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
	parser.add_argument("--dry-run", action="store_true")
	parser.add_argument("--subprocess-timeout", type=float, default=180.0)
	parser.add_argument("--single-task-id", default=None, help=argparse.SUPPRESS)
	args = parser.parse_args()

	if args.single_task_id is not None:
		return _run_single_mode(args)

	rec_a, rec_b, compare = build_report(args)
	output_dir = _output_dir(args)
	print_status(compare["status"], compare.get("messages", []))
	suffix = "_runtime" if args.launch_env else ""
	if rec_a is not None:
		write_json_report(rec_a, output_dir / f"geometry_{str(args.task_id_a).zfill(5)}{suffix}.json", dry_run=args.dry_run)
	if rec_b is not None:
		write_json_report(rec_b, output_dir / f"geometry_{str(args.task_id_b).zfill(5)}{suffix}.json", dry_run=args.dry_run)
	write_json_report(compare, output_dir / f"geometry_compare_{str(args.task_id_a).zfill(5)}_{str(args.task_id_b).zfill(5)}{suffix}.json", dry_run=args.dry_run)
	return 1 if compare["status"] == "FAIL" else 0


if __name__ == "__main__":
	raise SystemExit(main())
