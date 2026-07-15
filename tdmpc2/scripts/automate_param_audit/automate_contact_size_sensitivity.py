#!/usr/bin/env python3
"""Best-effort read-only contact-dynamics size sensitivity probe for SRSA."""

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
	compare_task_vectors,
	launch_probe_env,
	load_json,
	load_task_id_vectors,
	print_status,
	status_from_messages,
	tensor_like_to_python,
	write_json_report,
)


def _first_float(value: Any, default: float = 0.0) -> float:
	if isinstance(value, dict) and "values" in value:
		values = value.get("values") or []
		return float(values[0]) if values else default
	try:
		import torch
		if torch.is_tensor(value):
			flat = value.detach().cpu().reshape(-1)
			return float(flat[0].item()) if flat.numel() else default
	except Exception:
		pass
	if isinstance(value, (list, tuple)) and value:
		return _first_float(value[0], default=default)
	if isinstance(value, (int, float, bool)):
		return float(value)
	return default


def _zero_action(env: Any, cfg: Any):
	try:
		import torch
	except Exception as exc:
		raise RuntimeError("torch is required for --launch-env contact probe") from exc
	action_space = getattr(env, "action_space", None)
	if action_space is not None and hasattr(action_space, "shape"):
		shape = tuple(int(x) for x in action_space.shape)
		if len(shape) == 1:
			shape = (int(getattr(env.unwrapped if hasattr(env, "unwrapped") else env, "num_envs", 1)), *shape)
		return torch.zeros(shape, dtype=torch.float32, device=getattr(env.unwrapped if hasattr(env, "unwrapped") else env, "device", "cpu"))
	num_envs = int(getattr(env.unwrapped if hasattr(env, "unwrapped") else env, "num_envs", 1))
	action_dim = int(getattr(cfg, "srsa_env_action_dim", getattr(cfg, "isaaclab_action_dim", 6)))
	return torch.zeros((num_envs, action_dim), dtype=torch.float32, device=getattr(env.unwrapped if hasattr(env, "unwrapped") else env, "device", "cpu"))


def _reset_env(env: Any):
	reset = getattr(env, "reset", None)
	if callable(reset):
		return reset()
	return None


def _step_env(env: Any, action: Any):
	step = getattr(env, "step", None)
	if callable(step):
		return step(action)
	raise RuntimeError("env has no step()")


def _metric_value(unwrapped: Any, key: str, default: float = 0.0) -> float:
	for fn_name in ("_compute_depth_contact_jam", "_compute_srsa_success_metrics"):
		fn = getattr(unwrapped, fn_name, None)
		if callable(fn):
			try:
				metrics = fn(update_state=False) if fn_name == "_compute_srsa_success_metrics" else fn()
				if isinstance(metrics, dict) and key in metrics:
					return _first_float(metrics[key], default=default)
			except Exception:
				pass
	return default


def _force_value(unwrapped: Any) -> float:
	for attr in ("flange_force_norm", "held_asset_contact_force_world", "flange_force_world", "force", "forces", "flange_force"):
		if hasattr(unwrapped, attr):
			value = getattr(unwrapped, attr)
			try:
				import torch
				if torch.is_tensor(value):
					return float(value.detach().abs().max().item())
			except Exception:
				pass
			return abs(_first_float(value, default=0.0))
	return 0.0


def _contact_value(unwrapped: Any) -> float:
	if hasattr(unwrapped, "flange_force_flag"):
		return _first_float(getattr(unwrapped, "flange_force_flag"), default=0.0)
	return 1.0 if _force_value(unwrapped) > 1.0e-8 else 0.0


def _runtime_curve(task: dict[str, Any], args, *, output_dir: Path) -> dict[str, Any]:
	env = None
	try:
		cfg, env = launch_probe_env(task["assembly_id"])
		unwrapped = getattr(env, "unwrapped", env)
		_reset_env(env)
		action = _zero_action(env, cfg)
		contact_curve = []
		force_curve = []
		depth_curve = []
		lateral_curve = []
		jam_curve = []
		success_curve = []
		for step_idx in range(int(args.steps)):
			try:
				_step_env(env, action)
			except Exception as exc:
				return {"runtime_contact_check": "FAILED", "error": f"step {step_idx}: {exc!r}"}
			contact = _contact_value(unwrapped)
			force = _force_value(unwrapped)
			depth = _metric_value(unwrapped, "current_depth", default=_first_float(getattr(unwrapped, "current_depth", getattr(unwrapped, "current_insertion_depth", None)), default=0.0))
			jam = _metric_value(unwrapped, "jam", default=0.0)
			held_pos = getattr(unwrapped, "held_pos", None)
			fixed_pos = getattr(unwrapped, "fixed_pos", None)
			lateral = None
			try:
				import torch
				if torch.is_tensor(held_pos) and torch.is_tensor(fixed_pos):
					lateral = torch.linalg.norm((held_pos - fixed_pos)[..., :2], dim=-1)
			except Exception:
				lateral = None
			lateral_value = _metric_value(unwrapped, "lateral_error", default=_first_float(lateral, default=0.0))
			success = _metric_value(unwrapped, "success", default=_first_float(getattr(unwrapped, "success", getattr(unwrapped, "succeeded", None)), default=0.0))
			contact_curve.append(float(contact))
			force_curve.append(float(abs(force)))
			depth_curve.append(float(depth))
			lateral_curve.append(_first_float(lateral, default=0.0))
			if lateral_curve[-1] == 0.0 and lateral_value != 0.0:
				lateral_curve[-1] = float(lateral_value)
			jam_curve.append(float(jam))
			success_curve.append(float(success))
		first_contact = next((idx for idx, val in enumerate(contact_curve) if val > 0.5), None)
		return {
			"runtime_contact_check": "DONE",
			"task_id": task["task_id"],
			"assembly_id": task["assembly_id"],
			"task_hash": task["task_hash"],
			"action_sequence": "zero_action",
			"episode_length": len(contact_curve),
			"first_contact_time": first_contact,
			"max_force": max(force_curve) if force_curve else 0.0,
			"mean_force": sum(force_curve) / max(len(force_curve), 1),
			"contact_count": int(sum(1 for val in contact_curve if val > 0.5)),
			"insertion_depth_curve": depth_curve,
			"lateral_error_curve": lateral_curve,
			"contact_curve": contact_curve,
			"force_curve": force_curve,
			"jamming_wedging_flag": bool(any(val > 0.5 for val in jam_curve)),
			"jam_curve": jam_curve,
			"final_success": bool(success_curve[-1] > 0.5) if success_curve else None,
			"limitations": [
				"Same zero action sequence is used for both assemblies.",
				"Script sets num_envs=1/headless and uses the same config seed, but exact identical physical initial state across different assemblies depends on the SRSA/Isaac reset API.",
			],
		}
	except Exception as exc:
		log_path = append_runtime_error(exc, context=f"contact probe assembly_id={task.get('assembly_id')}", output_dir=output_dir)
		return {
			"runtime_contact_check": "FAILED",
			"task_id": task.get("task_id"),
			"assembly_id": task.get("assembly_id"),
			"task_hash": task.get("task_hash"),
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


def _runtime_curve_path(output_dir: Path, task_id: str) -> Path:
	return output_dir / f"contact_{str(task_id).zfill(5)}_runtime.json"


def _run_single_contact_subprocess(task: dict[str, Any], args, *, output_dir: Path) -> dict[str, Any]:
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
		"--steps",
		str(args.steps),
		"--diff-eps",
		str(args.diff_eps),
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
		error = TimeoutError(f"runtime contact subprocess timed out for task_id={task_id} after {args.subprocess_timeout}s")
		log_path = append_runtime_error(error, context=f"contact subprocess timeout task_id={task_id}", output_dir=output_dir)
		return {
			"runtime_contact_check": "FAILED",
			"task_id": task_id,
			"assembly_id": task.get("assembly_id"),
			"task_hash": task.get("task_hash"),
			"error": repr(error),
			"runtime_launch_error_log": str(log_path),
			"subprocess_timeout_s": float(args.subprocess_timeout),
			"subprocess_output_tail": output[-6000:],
		}
	record = load_json(_runtime_curve_path(output_dir, task_id), default=None)
	if isinstance(record, dict):
		record["subprocess_returncode"] = result.returncode
		if result.returncode != 0 or record.get("runtime_contact_check") == "FAILED":
			record["subprocess_output_tail"] = result.stdout[-6000:]
		return record
	error = RuntimeError(f"runtime contact subprocess failed for task_id={task_id} rc={result.returncode}\n{result.stdout[-6000:]}")
	log_path = append_runtime_error(error, context=f"contact subprocess task_id={task_id}", output_dir=output_dir)
	return {
		"runtime_contact_check": "FAILED",
		"task_id": task_id,
		"assembly_id": task.get("assembly_id"),
		"task_hash": task.get("task_hash"),
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
		record = {"runtime_contact_check": "FAILED", "task_id": task_id, "messages": messages}
	else:
		record = _runtime_curve(task, args, output_dir=output_dir)
		if record.get("runtime_contact_check") == "FAILED":
			add_message(messages, "FAIL", f"Runtime contact probe failed for task_id={task_id}.", error=record.get("error"))
		else:
			add_message(messages, "PASS", f"Runtime contact probe completed for task_id={task_id}.")
		record["messages"] = messages
	record["status"] = status_from_messages(messages)
	print_status(record["status"], messages)
	write_json_report(record, _runtime_curve_path(output_dir, task_id), dry_run=args.dry_run)
	return 1 if record["status"] == "FAIL" else 0


def _curve_diff(a_curve, b_curve) -> dict[str, Any]:
	if not isinstance(a_curve, list) or not isinstance(b_curve, list) or not a_curve or not b_curve:
		return {"comparable": False, "reason": "one or both curves are missing"}
	n = min(len(a_curve), len(b_curve))
	diffs = [abs(float(b_curve[i]) - float(a_curve[i])) for i in range(n)]
	return {
		"comparable": True,
		"length_a": len(a_curve),
		"length_b": len(b_curve),
		"max_abs_diff": max(diffs) if diffs else 0.0,
		"mean_abs_diff": sum(diffs) / max(len(diffs), 1),
		"same_with_eps_1e-8": all(diff <= 1.0e-8 for diff in diffs) and len(a_curve) == len(b_curve),
	}


def _diff_runtime(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
	keys = ("first_contact_time", "max_force", "mean_force", "contact_count", "final_success", "episode_length")
	result = {
		key: {
			"a": a.get(key),
			"b": b.get(key),
			"same": a.get(key) == b.get(key),
		}
		for key in keys
	}
	for key in ("force_curve", "insertion_depth_curve", "lateral_error_curve", "contact_curve", "jam_curve"):
		result[key] = _curve_diff(a.get(key), b.get(key))
	return result


def _contact_effective(diff: dict[str, Any], eps: float) -> bool:
	for key in ("first_contact_time", "max_force", "mean_force", "contact_count", "final_success"):
		item = diff.get(key) or {}
		if not item.get("same", True):
			return True
	for key in ("force_curve", "insertion_depth_curve", "lateral_error_curve", "contact_curve", "jam_curve"):
		item = diff.get(key) or {}
		if item.get("comparable") and float(item.get("max_abs_diff", 0.0)) > eps:
			return True
	return False


def build_report(args):
	messages: list[dict[str, Any]] = []
	output_dir = _output_dir(args)
	tasks = load_task_id_vectors(args.task_hash_csv)
	task_a = tasks.get(str(args.task_id_a).zfill(5))
	task_b = tasks.get(str(args.task_id_b).zfill(5))
	if task_a is None or task_b is None:
		add_message(messages, "FAIL", "Missing task vectors in task_id_to_hash.csv.")
		return {"status": "FAIL", "messages": messages}

	if not args.launch_env or args.dry_run:
		if args.launch_env and args.dry_run:
			add_message(messages, "WARNING", "Dry-run requested; runtime contact env launch skipped.")
		else:
			add_message(messages, "WARNING", "Runtime contact probe skipped; use --launch-env to run a same-action rollout probe.")
		return {
			"status": status_from_messages(messages),
			"runtime_contact_check": "SKIPPED",
			"reason": "Dry-run skipped Isaac env launch." if args.launch_env and args.dry_run else "No Isaac env launched.",
			"task_ids": [task_a["task_id"], task_b["task_id"]],
			"task_vector_compare": compare_task_vectors(task_a, task_b),
			"contact_effective": None,
			"messages": messages,
		}

	rec_a = _run_single_contact_subprocess(task_a, args, output_dir=output_dir)
	rec_b = _run_single_contact_subprocess(task_b, args, output_dir=output_dir)
	if rec_a.get("runtime_contact_check") == "FAILED" or rec_b.get("runtime_contact_check") == "FAILED":
		add_message(messages, "FAIL", "Runtime contact probe failed.", errors=[rec_a.get("error"), rec_b.get("error")])
		diff = _diff_runtime(rec_a, rec_b)
		contact_effective = None
	else:
		diff = _diff_runtime(rec_a, rec_b)
		contact_effective = _contact_effective(diff, float(args.diff_eps))
		if contact_effective:
			add_message(messages, "PASS", "Contact rollout metrics differ between assemblies.")
		else:
			add_message(messages, "WARNING", "Contact rollout metrics are nearly identical; contact dynamics may not be size-sensitive for this pair.")
	report = {
		"status": status_from_messages(messages),
		"runtime_contact_check": "DONE" if rec_a.get("runtime_contact_check") == "DONE" and rec_b.get("runtime_contact_check") == "DONE" else "FAILED",
		"task_vector_compare": compare_task_vectors(task_a, task_b),
		"contact_effective": contact_effective,
		"same_initial_state_limit": "Same seed and zero action sequence are used, but exact same physical initial state across different assembly geometry is not guaranteed by the exposed SRSA reset API.",
		"task_a": rec_a,
		"task_b": rec_b,
		"metric_compare": diff,
		"messages": messages,
	}
	return report


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--task-hash-csv", default=str(DEFAULT_TASK_HASH_CSV))
	parser.add_argument("--task-id-a", default="01125")
	parser.add_argument("--task-id-b", default="00256")
	parser.add_argument("--steps", type=int, default=16)
	parser.add_argument("--diff-eps", type=float, default=1.0e-8)
	parser.add_argument("--launch-env", action="store_true")
	parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
	parser.add_argument("--dry-run", action="store_true")
	parser.add_argument("--subprocess-timeout", type=float, default=180.0)
	parser.add_argument("--single-task-id", default=None, help=argparse.SUPPRESS)
	args = parser.parse_args()

	if args.single_task_id is not None:
		return _run_single_mode(args)

	report = build_report(args)
	output_dir = _output_dir(args)
	print_status(report["status"], report.get("messages", []))
	suffix = "_runtime" if args.launch_env else ""
	write_json_report(report, output_dir / f"contact_size_sensitivity{suffix}.json", dry_run=args.dry_run)
	return 1 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
	raise SystemExit(main())
