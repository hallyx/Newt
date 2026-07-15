#!/usr/bin/env python3
"""Read-only runtime smoke probe for the Phase 3.1 00186 assembly."""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
AUDIT_DIR = REPO_ROOT / "tdmpc2" / "scripts" / "automate_param_audit"
if str(AUDIT_DIR) not in sys.path:
	sys.path.insert(0, str(AUDIT_DIR))

from _common import close_env, collect_runtime_geometry, launch_probe_env, tensor_like_to_python  # noqa: E402
from automate_contact_size_sensitivity import (  # noqa: E402
	_contact_value,
	_force_value,
	_metric_value,
	_step_env,
	_zero_action,
)


DEFAULT_CANDIDATES = REPO_ROOT / "reports" / "phase3_easy_third_task" / "third_task_candidates.json"
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "phase3_three_task_pilot" / "00186_runtime_smoke.json"


def _resolve(value: str | Path) -> Path:
	path = Path(value).expanduser()
	if not path.is_absolute():
		path = REPO_ROOT / path
	return path.resolve()


def _expected_task_vec(path: Path, task_id: str) -> list[float]:
	payload = json.loads(path.read_text(encoding="utf-8"))
	for item in payload.get("ranked_candidates", []):
		if str(item.get("candidate_task_id")).zfill(5) == str(task_id).zfill(5):
			vec = [float(value) for value in item.get("task_vec_6", [])]
			if len(vec) != 6:
				raise ValueError(f"candidate task_vec_6 must have length 6, got {vec}")
			return vec
	raise KeyError(f"task_id={task_id} is missing from {path}")


def _runtime_vec(record: dict[str, Any]) -> list[float] | None:
	value = record.get("current_task_vec")
	if isinstance(value, dict):
		value = value.get("values")
	if isinstance(value, list) and value and isinstance(value[0], list):
		value = value[0]
	if not isinstance(value, list) or len(value) < 6:
		return None
	return [float(item) for item in value[:6]]


def _find_target_depth(record: dict[str, Any]) -> float | None:
	for key in ("target_insertion_depth", "insertion_depth", "current_insertion_depth"):
		value = record.get(key)
		if isinstance(value, dict):
			value = value.get("values")
		if isinstance(value, list):
			value = value[0] if value else None
		if isinstance(value, (int, float)) and math.isfinite(float(value)):
			return float(value)
	params = record.get("current_task_params")
	if isinstance(params, dict):
		for key in ("target_insertion_depth", "insertion_depth"):
			value = params.get(key)
			if isinstance(value, (int, float)) and math.isfinite(float(value)):
				return float(value)
	return None


def _reward_scalar(step_result: Any) -> float | None:
	if not isinstance(step_result, tuple) or len(step_result) < 2:
		return None
	value = step_result[1]
	try:
		import torch
		if torch.is_tensor(value):
			flat = value.detach().float().cpu().reshape(-1)
			return float(flat[0].item()) if flat.numel() else None
	except Exception:
		pass
	if isinstance(value, (list, tuple)) and value:
		value = value[0]
	return float(value) if isinstance(value, (int, float)) else None


def build_report(args: argparse.Namespace) -> dict[str, Any]:
	expected = _expected_task_vec(_resolve(args.candidates_report), args.task_id)
	env = None
	try:
		cfg, env = launch_probe_env(
			args.task_id,
			extra_overrides={
				"srsa_task_template_fp": str(REPO_ROOT / "data" / "srsa_axial_task_templates.json"),
				"srsa_mesh_geometry_fp": str(REPO_ROOT / "data" / "srsa_mesh_geometry_params.csv"),
				"srsa_param_template_id": 2,
				"eval_task_template_exact": True,
				"srsa_axial_reference_anchor_assembly_id": "01125",
				"srsa_axial_reference_anchor_task_type_id": 0,
				"srsa_axial_recompute_manifest_task_vecs": True,
				"srsa_axial_clearance_depth_templates": "1.0:1.0",
			},
		)
		unwrapped = getattr(env, "unwrapped", env)
		reset = getattr(env, "reset", None)
		if callable(reset):
			reset()
		geometry = collect_runtime_geometry(env, cfg=cfg)
		action = _zero_action(env, cfg)
		step_rows = []
		for step_idx in range(int(args.steps)):
			step_result = _step_env(env, action)
			step_rows.append({
				"step": step_idx,
				"reward": _reward_scalar(step_result),
				"force": float(_force_value(unwrapped)),
				"contact": float(_contact_value(unwrapped)),
				"depth": float(_metric_value(unwrapped, "current_depth", default=0.0)),
				"lateral_error": float(_metric_value(unwrapped, "lateral_error", default=0.0)),
				"jam": float(_metric_value(unwrapped, "jam", default=0.0)),
			})
		runtime_vec = _runtime_vec(geometry)
		vec_delta = None if runtime_vec is None else [runtime_vec[i] - expected[i] for i in range(6)]
		max_abs_delta = None if vec_delta is None else max(abs(value) for value in vec_delta)
		target_depth = _find_target_depth(geometry)
		reward_available = any(row["reward"] is not None and math.isfinite(row["reward"]) for row in step_rows)
		contact_metrics_available = all(
			all(math.isfinite(float(row[key])) for key in ("force", "contact", "depth", "lateral_error", "jam"))
			for row in step_rows
		)
		checks = {
			"env_asset_loaded": bool(geometry.get("runtime_check") == "DONE"),
			"runtime_task_vec_available": runtime_vec is not None,
			"runtime_task_vec_matches_expected": bool(max_abs_delta is not None and max_abs_delta <= float(args.task_vec_tol)),
			"target_depth_available": target_depth is not None,
			"reward_available": reward_available,
			"contact_metrics_available": contact_metrics_available,
		}
		status = "PASS" if all(checks.values()) else "FAIL"
		return {
			"status": status,
			"task_id": str(args.task_id).zfill(5),
			"expected_task_vec_6": expected,
			"runtime_task_vec_6": runtime_vec,
			"task_vec_delta": vec_delta,
			"task_vec_max_abs_delta": max_abs_delta,
			"task_vec_tolerance": float(args.task_vec_tol),
			"target_depth": target_depth,
			"checks": checks,
			"runtime_geometry": geometry,
			"runtime_steps": step_rows,
			"cfg_summary": {
				"assembly_id": str(getattr(cfg, "assembly_id", args.task_id)),
				"num_envs": int(getattr(cfg, "num_envs", 1)),
				"headless": bool(getattr(cfg, "isaaclab_headless", True)),
			},
		}
	finally:
		if env is not None:
			close_env(env)


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--task-id", default="00186")
	parser.add_argument("--candidates-report", default=str(DEFAULT_CANDIDATES))
	parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
	parser.add_argument("--steps", type=int, default=8)
	parser.add_argument("--task-vec-tol", type=float, default=1.0e-5)
	parser.add_argument("--dry-run", action="store_true")
	args = parser.parse_args()
	output = _resolve(args.output)
	if args.dry_run:
		print(f"PASS dry-run: would launch task_id={args.task_id} and write {output}")
		return 0
	try:
		report = build_report(args)
	except Exception as exc:
		output.parent.mkdir(parents=True, exist_ok=True)
		error_fp = output.parent / "runtime_smoke_error.log"
		error_fp.write_text(traceback.format_exc(), encoding="utf-8")
		report = {
			"status": "FAIL",
			"task_id": str(args.task_id).zfill(5),
			"error": repr(exc),
			"error_log": str(error_fp),
		}
	output.parent.mkdir(parents=True, exist_ok=True)
	output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
	print(f"{report['status']} wrote {output}")
	return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
	raise SystemExit(main())
