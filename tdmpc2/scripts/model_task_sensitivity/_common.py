#!/usr/bin/env python3
"""Shared read-only helpers for model task-sensitivity audit scripts."""

from __future__ import annotations

import json
import math
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
TDMPC2_ROOT = SCRIPT_DIR.parents[1]
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "model_task_sensitivity"

DEFAULT_CHECKPOINT = (
	"logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_rawtask_adapter_01125_00256/"
	"20260625_0010_rawtask_alpha001_sitelimited_stage-2_asm-00256/models/"
	"best_step-99840_s-0p9531.pt"
)
DEFAULT_CONFIG = "configs/train/srsa_01125_imitation_relaxed.yaml"
DEFAULT_TASK_A_REPLAY = (
	"logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_replay_from_01125/"
	"20260615_202326_launcher/replay/01125.pt"
)
DEFAULT_TASK_B_REPLAY = (
	"logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_rawtask_adapter_01125_00256/"
	"20260625_0010_rawtask_alpha001_sitelimited_launcher/replay/00256.pt"
)


def ensure_import_paths() -> None:
	for path in (TDMPC2_ROOT, SCRIPTS_ROOT):
		path_str = str(path)
		if path_str not in sys.path:
			sys.path.insert(0, path_str)


ensure_import_paths()

import task_vec_sensitivity_report as tvsr  # noqa: E402
from common import math as td_math  # noqa: E402
from common.world_model import WorldModel  # noqa: E402


def resolve(path_value: str | Path) -> Path:
	path = Path(path_value).expanduser()
	if not path.is_absolute():
		path = REPO_ROOT / path
	return path.resolve()


def add_common_args(parser) -> None:
	parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
	parser.add_argument("--config", default=DEFAULT_CONFIG)
	parser.add_argument("--task-a-label", default="01125")
	parser.add_argument("--task-b-label", default="00256")
	parser.add_argument("--task-a-replay", default=DEFAULT_TASK_A_REPLAY)
	parser.add_argument("--task-b-replay", default=DEFAULT_TASK_B_REPLAY)
	parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
	parser.add_argument("--batch-size", type=int, default=256)
	parser.add_argument("--seed", type=int, default=1)
	parser.add_argument("--gpu-id", type=int, default=0)
	parser.add_argument("--cpu", action="store_true")
	parser.add_argument("--assembly-id", default="00256")
	parser.add_argument("--eval-task-id", type=int, default=2)
	parser.add_argument("--include-zero", action="store_true", default=True)
	parser.add_argument("--include-random", action="store_true", default=True)
	parser.add_argument("--dry-run", action="store_true")


def output_dir(args) -> Path:
	path = Path(args.output_dir).expanduser()
	if not path.is_absolute():
		path = REPO_ROOT / path
	return path.resolve()


def write_json(report: dict[str, Any], path: str | Path, *, dry_run: bool = False) -> None:
	path = resolve(path)
	if dry_run:
		print(f"[dry-run] would write JSON report: {path}")
		return
	path.parent.mkdir(parents=True, exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		json.dump(report, f, indent=2, ensure_ascii=False)
		f.write("\n")
	print(f"Wrote JSON report: {path}")


def write_text(text: str, path: str | Path, *, dry_run: bool = False) -> None:
	path = resolve(path)
	if dry_run:
		print(f"[dry-run] would write text report: {path}")
		return
	path.parent.mkdir(parents=True, exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		f.write(text)
		if not text.endswith("\n"):
			f.write("\n")
	print(f"Wrote text report: {path}")


def add_message(messages: list[dict[str, Any]], level: str, message: str, **extra: Any) -> None:
	item = {"level": str(level).upper(), "message": str(message)}
	item.update(extra)
	messages.append(item)


def status_from_messages(messages: list[dict[str, Any]]) -> str:
	levels = {str(item.get("level", "")).upper() for item in messages}
	if "FAIL" in levels:
		return "FAIL"
	if "WARNING" in levels:
		return "WARNING"
	return "PASS"


def print_status(status: str, messages: list[dict[str, Any]]) -> None:
	print(status)
	for item in messages:
		print(f"[{item.get('level', 'INFO')}] {item.get('message', '')}")


def require_existing_inputs(args, *, need_checkpoint: bool = True, need_replay: bool = True) -> dict[str, Path]:
	paths = {
		"checkpoint": resolve(args.checkpoint),
		"config": resolve(args.config),
		"task_a_replay": resolve(args.task_a_replay),
		"task_b_replay": resolve(args.task_b_replay),
	}
	required = ["config"]
	if need_checkpoint:
		required.append("checkpoint")
	if need_replay:
		required.extend(["task_a_replay", "task_b_replay"])
	missing = [f"{name}={paths[name]}" for name in required if not paths[name].exists()]
	if missing:
		raise FileNotFoundError("Missing required Phase 1.0 input(s): " + "; ".join(missing))
	return paths


def select_device(args) -> torch.device:
	if torch.cuda.is_available() and not bool(args.cpu):
		device = torch.device(f"cuda:{int(args.gpu_id)}")
		torch.cuda.set_device(device)
		return device
	return torch.device("cpu")


def load_model_bundle(args):
	paths = require_existing_inputs(args, need_checkpoint=True, need_replay=True)
	device = select_device(args)
	torch.manual_seed(int(args.seed))
	cfg, compat = tvsr._load_config(args, paths["checkpoint"])
	cfg.device_id = int(args.gpu_id) if device.type == "cuda" else 0
	model = WorldModel(cfg).to(device)
	model = tvsr._load_world_model(model, paths["checkpoint"], cfg)
	model.eval()
	return {
		"paths": paths,
		"device": device,
		"cfg": cfg,
		"compat": compat,
		"model": model,
	}


def load_task_conditions(args) -> OrderedDict[str, torch.Tensor]:
	paths = require_existing_inputs(args, need_checkpoint=False, need_replay=True)
	conditions: OrderedDict[str, torch.Tensor] = OrderedDict()
	conditions[str(args.task_a_label)] = tvsr._unique_task_vec_from_replay(paths["task_a_replay"])[0].float()
	conditions[str(args.task_b_label)] = tvsr._unique_task_vec_from_replay(paths["task_b_replay"])[0].float()
	if bool(args.include_zero):
		conditions["zero"] = torch.zeros(6, dtype=torch.float32)
	if bool(args.include_random):
		generator = torch.Generator().manual_seed(int(args.seed))
		conditions["random"] = torch.empty(6, dtype=torch.float32).uniform_(-1.0, 1.0, generator=generator)
	return conditions


def load_trimmed_replay_batch(args, compat: dict[str, Any], device: torch.device):
	paths = require_existing_inputs(args, need_checkpoint=False, need_replay=True)
	td, metadata = tvsr._load_replay_tensordict(paths["task_b_replay"])
	obs, action, reward, indices = tvsr._sample_replay_rows(td, int(args.batch_size), int(args.seed))
	expected_obs_dim = int(compat["obs_dim"])
	if int(obs.shape[-1]) > expected_obs_dim:
		obs = obs[..., :expected_obs_dim].contiguous()
	elif int(obs.shape[-1]) < expected_obs_dim:
		raise RuntimeError(f"Replay obs dim {int(obs.shape[-1])} < checkpoint obs dim {expected_obs_dim}.")
	expected_action_dim = int(compat["action_dim"] or action.shape[-1])
	if int(action.shape[-1]) > expected_action_dim:
		action = action[..., :expected_action_dim].contiguous()
	elif int(action.shape[-1]) < expected_action_dim:
		raise RuntimeError(f"Replay action dim {int(action.shape[-1])} < checkpoint action dim {expected_action_dim}.")
	return {
		"obs": obs.to(device),
		"action": action.to(device),
		"reward": reward.to(device),
		"indices": indices,
		"metadata": metadata,
	}


def condition_batch(vec: torch.Tensor, batch_shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
	return vec.to(device=device, dtype=torch.float32).view(*([1] * len(batch_shape)), -1).expand(*batch_shape, -1).contiguous()


def summarize_tensor(value: torch.Tensor) -> dict[str, float]:
	value = value.detach().float().reshape(-1).cpu()
	if value.numel() == 0:
		return {"mean": math.nan, "median": math.nan, "p95": math.nan, "max": math.nan}
	return {
		"mean": float(value.mean().item()),
		"median": float(value.median().item()),
		"p95": float(torch.quantile(value, 0.95).item()),
		"max": float(value.max().item()),
	}


def l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
	return torch.linalg.vector_norm(a - b, dim=-1)


def abs_delta(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
	return (a - b).abs()


def two_hot_scalar(logits: torch.Tensor, cfg) -> torch.Tensor:
	return td_math.two_hot_inv(logits, cfg)


def tensor_to_list(value: torch.Tensor) -> list[float]:
	return [float(x) for x in value.detach().cpu().reshape(-1).tolist()]


def safe_mean(report: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
	if not isinstance(report, dict):
		return default
	value = report.get(key)
	if isinstance(value, dict) and "mean" in value:
		return float(value["mean"])
	try:
		return float(value)
	except Exception:
		return default

