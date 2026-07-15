#!/usr/bin/env python3
"""Probe whether eval task input uses runtime current_task_vec or fallback task id."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from _common import (
	DEFAULT_OUTPUT_DIR,
	TDMPC2_ROOT,
	add_message,
	print_status,
	status_from_messages,
	write_json_report,
)

if str(TDMPC2_ROOT) not in sys.path:
	sys.path.insert(0, str(TDMPC2_ROOT))

from collect_eval_rollouts import _model_task_input  # noqa: E402


class ConfigDict(dict):
	def __getattr__(self, key):
		try:
			return self[key]
		except KeyError as exc:
			raise AttributeError(key) from exc

	def __setattr__(self, key, value):
		self[key] = value


class MockEnv:
	def __init__(self, current_task_vec):
		self.unwrapped = SimpleNamespace(current_task_vec=current_task_vec)


def parse_vec6(raw):
	if raw is None:
		return None
	text = str(raw).strip().strip("'\"")
	if not text or text.lower() in {"none", "null"}:
		return None
	if text.startswith("[") and text.endswith("]"):
		values = json.loads(text)
	else:
		values = [item for item in text.replace(";", ",").replace(" ", ",").split(",") if item.strip()]
	values = [float(value) for value in values]
	if len(values) != 6:
		raise ValueError(f"Expected 6 values for task_vec_6, got {len(values)}: {values}")
	return values


def _tensor_preview(tensor):
	if tensor is None or not torch.is_tensor(tensor):
		return None
	return {
		"shape": list(tensor.shape),
		"dtype": str(tensor.dtype),
		"values_first": tensor.reshape(-1, tensor.shape[-1])[0].detach().cpu().tolist() if tensor.ndim >= 2 else tensor.detach().cpu().tolist(),
	}


def _classify_task_input(model_tasks, fallback_tasks):
	if torch.is_tensor(model_tasks) and model_tasks.is_floating_point() and model_tasks.ndim == 2 and model_tasks.shape[-1] == 6:
		return "runtime_current_task_vec"
	if torch.is_tensor(model_tasks) and torch.equal(model_tasks.detach().cpu(), fallback_tasks.detach().cpu()):
		return "fallback_task_id"
	return "other"


def _fallback_tasks(num_envs, fallback_task_id, device):
	return torch.full((int(num_envs),), int(fallback_task_id), dtype=torch.long, device=device)


def build_mock_probe(args):
	device = torch.device(args.device)
	cfg = ConfigDict({
		"num_envs": int(args.num_envs),
		"srsa_use_runtime_task_vec": bool(args.srsa_use_runtime_task_vec),
		"task_conditioning": str(args.task_conditioning),
	})
	vec = parse_vec6(args.runtime_task_vec)
	current_task_vec = None
	if vec is not None:
		current_task_vec = torch.as_tensor(vec, dtype=torch.float32, device=device).view(1, 6).expand(int(args.num_envs), 6).contiguous()
	env = MockEnv(current_task_vec)
	fallback = _fallback_tasks(args.num_envs, args.fallback_task_id, device)
	model_tasks = _model_task_input(cfg, env, fallback)
	return cfg, env, fallback, model_tasks, "mock"


def build_launch_env_probe(args):
	if args.dry_run:
		cfg = ConfigDict({
			"num_envs": int(args.num_envs),
			"srsa_use_runtime_task_vec": bool(args.srsa_use_runtime_task_vec),
			"task_conditioning": str(args.task_conditioning),
		})
		fallback = _fallback_tasks(args.num_envs, args.fallback_task_id, torch.device(args.device))
		return cfg, MockEnv(None), fallback, fallback, "launch_env_skipped_by_dry_run"

	from omegaconf import OmegaConf  # noqa: WPS433
	from config import Config, parse_cfg  # noqa: WPS433
	from envs import make_env  # noqa: WPS433

	cfg = OmegaConf.structured(Config)
	if args.config_yaml:
		cfg = OmegaConf.merge(cfg, OmegaConf.load(args.config_yaml))
	if args.override:
		cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.override))
	cfg = parse_cfg(cfg)
	device = torch.device(f"cuda:{cfg.device_id}" if torch.cuda.is_available() and str(args.device).startswith("cuda") else args.device)
	env = make_env(cfg)
	fallback = _fallback_tasks(cfg.num_envs, args.fallback_task_id, device)
	model_tasks = _model_task_input(cfg, env, fallback)
	return cfg, env, fallback, model_tasks, "launch_env"


def build_report(args):
	messages = []
	env = None
	try:
		if args.launch_env:
			cfg, env, fallback, model_tasks, probe_mode = build_launch_env_probe(args)
		else:
			cfg, env, fallback, model_tasks, probe_mode = build_mock_probe(args)
	finally:
		if env is not None and args.launch_env and not args.dry_run and hasattr(env, "close"):
			env.close()

	input_mode = _classify_task_input(model_tasks, fallback)
	task_conditioning = str(cfg.get("task_conditioning", "") if hasattr(cfg, "get") else getattr(cfg, "task_conditioning", ""))
	num_envs = int(cfg.get("num_envs", args.num_envs) if hasattr(cfg, "get") else getattr(cfg, "num_envs", args.num_envs))
	srsa_use_runtime = bool(cfg.get("srsa_use_runtime_task_vec", args.srsa_use_runtime_task_vec) if hasattr(cfg, "get") else getattr(cfg, "srsa_use_runtime_task_vec", args.srsa_use_runtime_task_vec))

	if probe_mode == "launch_env_skipped_by_dry_run":
		add_message(messages, "WARNING", "--dry-run with --launch-env skipped environment creation.")
	if input_mode == "runtime_current_task_vec":
		add_message(messages, "PASS", "_model_task_input() returned runtime current_task_vec.", mode=input_mode)
	elif input_mode == "fallback_task_id":
		add_message(messages, "WARNING", "_model_task_input() fell back to task id tensor.", mode=input_mode)
		if task_conditioning.lower() == "axial_params" and int(args.fallback_task_id) == 0:
			add_message(messages, "WARNING", "Axial mode fallback task id is 0; this can silently select _task_vecs[0].")
		if args.fail_on_fallback:
			add_message(messages, "FAIL", "Fallback task id was used and --fail-on-fallback is set.")
	else:
		add_message(messages, "WARNING", "_model_task_input() returned an unexpected task input type.", mode=input_mode)

	current_task_vec = getattr(getattr(env, "unwrapped", env), "current_task_vec", None) if env is not None else None
	if srsa_use_runtime and task_conditioning.lower() == "axial_params":
		if not torch.is_tensor(current_task_vec):
			add_message(messages, "WARNING", "Runtime current_task_vec is missing.")
		elif current_task_vec.ndim != 2 or current_task_vec.shape[0] != num_envs or current_task_vec.shape[-1] != 6:
			add_message(
				messages,
				"WARNING",
				"Runtime current_task_vec has invalid shape for _model_task_input().",
				current_task_vec_shape=list(current_task_vec.shape),
				num_envs=num_envs,
			)

	status = status_from_messages(messages)
	return {
		"status": status,
		"probe_mode": probe_mode,
		"input_mode": input_mode,
		"task_conditioning": task_conditioning,
		"srsa_use_runtime_task_vec": srsa_use_runtime,
		"num_envs": num_envs,
		"fallback_task_id": int(args.fallback_task_id),
		"runtime_current_task_vec": _tensor_preview(current_task_vec),
		"model_task_input": _tensor_preview(model_tasks),
		"fallback_tasks": _tensor_preview(fallback),
		"messages": messages,
	}


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--runtime-task-vec", default=None, help="Optional comma/JSON task_vec_6 for mock probe mode.")
	parser.add_argument("--num-envs", type=int, default=64)
	parser.add_argument("--fallback-task-id", type=int, default=0)
	parser.add_argument("--task-conditioning", default="axial_params")
	parser.add_argument("--srsa-use-runtime-task-vec", action=argparse.BooleanOptionalAction, default=True)
	parser.add_argument("--device", default="cpu")
	parser.add_argument("--launch-env", action="store_true", help="Create the configured eval env and probe its real current_task_vec.")
	parser.add_argument("--config-yaml", default=None, help="Optional OmegaConf YAML merged into Config when --launch-env is set.")
	parser.add_argument("--override", action="append", default=[], help="OmegaConf dotlist override for --launch-env. Can be repeated.")
	parser.add_argument("--fail-on-fallback", action="store_true", help="Treat fallback task id as FAIL instead of WARNING.")
	parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for reports.")
	parser.add_argument("--dry-run", action="store_true", help="Run probe without writing report; with --launch-env, do not create env.")
	args = parser.parse_args()

	report = build_report(args)
	output_dir = Path(args.output_dir).expanduser()
	if not output_dir.is_absolute():
		output_dir = DEFAULT_OUTPUT_DIR.parent.parent / output_dir
	output_dir = output_dir.resolve()
	print_status(report["status"], report["messages"])
	write_json_report(report, output_dir / "eval_task_input_probe.json", dry_run=args.dry_run)
	return 1 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
	raise SystemExit(main())
