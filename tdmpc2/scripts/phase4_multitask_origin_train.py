#!/usr/bin/env python3
"""Phase 4.0 matched-budget multi-task origin ablation training.

The sampler is deliberately independent of the online-family replay sampler:
every update contains all 3 tasks x all 3 insertion phases, and the per-cell
remainder rotates across updates.  B/C/D consume the same sampled index stream.
Physical CUDA1 must be exposed as logical cuda:0.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
TDMPC2_ROOT = SCRIPT_DIR.parents[0]
REPO_ROOT = SCRIPT_DIR.parents[1]
MODEL_AUDIT_DIR = SCRIPT_DIR.parent / "model_task_sensitivity"
PHASE3_DIR = SCRIPT_DIR / "phase3_three_task_pilot"
for path in (TDMPC2_ROOT, SCRIPT_DIR.parent, MODEL_AUDIT_DIR, PHASE3_DIR):
	if str(path) not in sys.path:
		sys.path.insert(0, str(path))

import task_vec_sensitivity_report as tvsr  # noqa: E402
import policy_prior_supervision_audit as prior_audit  # noqa: E402
from common.world_model import WorldModel  # noqa: E402
from tdmpc2 import TDMPC2  # noqa: E402


TASKS = ("01125", "00256", "00186")
PHASES = ("pre_contact", "contact", "insertion")
VARIANTS = ("B", "C", "D")
DEFAULT_SOURCE_CHECKPOINT = (
	"logs/isaaclab-srsa-assembly/1/"
	"srsa_axial_online_family_taskctx_repair_01125_00256_00186_rescue/"
	"20260713_phase3_2_rescue_00186_stage-3_asm-00186/models/"
	"best_step-50176_s-0p2461.pt"
)
DEFAULT_REPLAYS = OrderedDict([
	("01125", "logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_replay_from_01125/20260615_202326_launcher/replay/01125.pt"),
	("00256", "logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_taskctx_repair_01125_00256/20260708_taskctx_repair_phase2_launcher/replay/00256.pt"),
	("00186", "logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_taskctx_repair_01125_00256_00186_rescue/20260713_phase3_2_rescue_00186_launcher/replay/00186.pt"),
])


def resolve(value: str | Path) -> Path:
	path = Path(value).expanduser()
	return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def _sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as handle:
		for block in iter(lambda: handle.read(1024 * 1024), b""):
			digest.update(block)
	return digest.hexdigest()


def _state_hash(state: dict[str, torch.Tensor], keys: list[str] | None = None) -> str:
	digest = hashlib.sha256()
	for key in sorted(keys if keys is not None else state):
		digest.update(key.encode("utf-8"))
		digest.update(state[key].detach().cpu().contiguous().numpy().tobytes())
	return digest.hexdigest()


def _write_json(value: Any, path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _require_cuda1(gpu_id: int) -> torch.device:
	visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
	if visible != "1" or int(gpu_id) != 0:
		raise RuntimeError(
			"Phase 4.0 is pinned to physical CUDA1: use CUDA_VISIBLE_DEVICES=1 and --gpu-id 0; "
			f"got CUDA_VISIBLE_DEVICES={visible!r}, gpu_id={gpu_id}."
		)
	if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
		raise RuntimeError("Expected exactly one visible CUDA device (physical CUDA1).")
	torch.cuda.set_device(0)
	return torch.device("cuda:0")


def _load_snapshot(path: Path):
	payload = torch.load(path, map_location="cpu", weights_only=False)
	if not isinstance(payload, dict) or "data" not in payload:
		raise RuntimeError(f"Replay is not a Newt snapshot payload: {path}")
	return payload["data"], dict(payload.get("metadata", {}))


def _episode_split(td, train_fraction: float, seed: int):
	episodes = torch.unique(td["episode"].detach().long().reshape(-1), sorted=True)
	generator = torch.Generator().manual_seed(int(seed))
	episodes = episodes[torch.randperm(int(episodes.numel()), generator=generator)]
	cut = min(max(1, int(round(float(train_fraction) * episodes.numel()))), int(episodes.numel()) - 1)
	return episodes[:cut], episodes[cut:]


def _valid_starts(td, allowed_episodes: torch.Tensor, horizon: int):
	episode = td["episode"].detach().long().reshape(-1)
	phase = prior_audit._phase_labels_from_replay(td)
	by_cell: dict[int, torch.Tensor] = {}
	for phase_id in range(len(PHASES)):
		rows = []
		for episode_id in allowed_episodes.tolist():
			idx = torch.nonzero(episode == int(episode_id), as_tuple=False).reshape(-1)
			if idx.numel() < horizon + 1:
				continue
			for local in range(0, int(idx.numel()) - horizon):
				seq = idx[local:local + horizon + 1]
				# Buffer semantics: obs[0:H+1], then action/reward/task[1:H+1].
				if not torch.all(episode[seq] == episode[seq[0]]):
					continue
				# The cell label is the first supervised transition. Later rollout
				# steps may cross a phase boundary, matching the native H=3 loss.
				if int(phase[seq[1]].item()) == phase_id:
					rows.append(int(seq[0].item()))
		if not rows:
			raise RuntimeError(f"No horizon-{horizon} sequences for phase={PHASES[phase_id]}")
		by_cell[phase_id] = torch.tensor(rows, dtype=torch.long)
	return phase, by_cell


class TaskPhaseBalancedBuffer:
	"""Exact task x phase balanced subsequence sampler with rotating remainder."""

	def __init__(self, replay_items, batch_size: int, horizon: int, seed: int):
		self.replay_items = replay_items
		self.batch_size = int(batch_size)
		self.horizon = int(horizon)
		self.generator = torch.Generator().manual_seed(int(seed))
		self.sample_index = 0
		self.cells = [(task, phase_id) for task in TASKS for phase_id in range(len(PHASES))]
		self.last_batch_task_counts = None
		self.last_batch_phase_counts = None
		self.last_batch_cell_counts = None

	def state_dict(self):
		return {"generator_state": self.generator.get_state(), "sample_index": self.sample_index}

	def load_state_dict(self, state):
		self.generator.set_state(state["generator_state"])
		self.sample_index = int(state["sample_index"])

	def _counts(self):
		base, remainder = divmod(self.batch_size, len(self.cells))
		counts = [base] * len(self.cells)
		for offset in range(remainder):
			counts[(self.sample_index + offset) % len(self.cells)] += 1
		return counts

	def sample(self, device: torch.device):
		obs_columns, action_columns, reward_columns, task_columns = [], [], [], []
		cell_counts = OrderedDict()
		for (task, phase_id), count in zip(self.cells, self._counts()):
			starts = self.replay_items[task]["train_starts"][phase_id]
			choice = torch.randint(int(starts.numel()), (count,), generator=self.generator)
			selected = starts[choice]
			seq = selected[:, None] + torch.arange(self.horizon + 1)[None, :]
			td = self.replay_items[task]["td"]
			obs_columns.append(td["obs"][seq].detach().float().permute(1, 0, 2))
			action_columns.append(td["action"][seq[:, 1:]].detach().float().permute(1, 0, 2))
			reward_columns.append(
				td["reward"][seq[:, 1:]].detach().float().reshape(count, self.horizon, 1).permute(1, 0, 2)
			)
			task_columns.append(td["task"][seq[:, 1:]].detach().float().permute(1, 0, 2))
			cell_counts[f"{task}/{PHASES[phase_id]}"] = count
		order = torch.randperm(self.batch_size, generator=self.generator)
		obs = torch.cat(obs_columns, dim=1)[:, order]
		action = torch.cat(action_columns, dim=1)[:, order]
		reward = torch.cat(reward_columns, dim=1)[:, order]
		task_vec = torch.cat(task_columns, dim=1)[:, order]
		self.sample_index += 1
		self.last_batch_cell_counts = cell_counts
		self.last_batch_task_counts = OrderedDict((task, sum(v for k, v in cell_counts.items() if k.startswith(task + "/"))) for task in TASKS)
		self.last_batch_phase_counts = OrderedDict((phase, sum(v for k, v in cell_counts.items() if k.endswith("/" + phase))) for phase in PHASES)
		return (
			obs.to(device, non_blocking=True).contiguous(),
			action.to(device, non_blocking=True).contiguous(),
			reward.to(device, non_blocking=True).contiguous(),
			task_vec.to(device, non_blocking=True).contiguous(),
		)


def _load_replays(args):
	paths = OrderedDict([
		("01125", resolve(args.replay_01125)),
		("00256", resolve(args.replay_00256)),
		("00186", resolve(args.replay_00186)),
	])
	items = OrderedDict()
	for task_index, (task, path) in enumerate(paths.items()):
		if not path.exists():
			raise FileNotFoundError(path)
		td, metadata = _load_snapshot(path)
		for field, shape in (("obs", 17), ("action", 3), ("task", 6)):
			if field not in td.keys() or int(td[field].shape[-1]) != shape:
				raise RuntimeError(f"Unexpected {task} replay {field} contract: {getattr(td.get(field, None), 'shape', None)}")
		unique_task = torch.unique(td["task"].detach().float().reshape(-1, 6), dim=0)
		if unique_task.shape[0] != 1:
			raise RuntimeError(f"Expected one task vector in {path}, found {unique_task.shape[0]}")
		train_eps, val_eps = _episode_split(td, args.train_fraction, args.split_seed + task_index)
		phase, train_starts = _valid_starts(td, train_eps, args.horizon)
		_, val_starts = _valid_starts(td, val_eps, args.horizon)
		items[task] = {
			"path": path,
			"td": td,
			"metadata": metadata,
			"task_vec": unique_task[0],
			"phase": phase,
			"train_episodes": train_eps,
			"val_episodes": val_eps,
			"train_starts": train_starts,
			"val_starts": val_starts,
		}
	return items


def _load_cfg(args, source_checkpoint: Path):
	cfg_args = SimpleNamespace(
		config=str(resolve(args.config)), gpu_id=int(args.gpu_id), batch_size=int(args.batch_size),
		assembly_id="00186", eval_task_id=2,
	)
	cfg, compat = tvsr._load_config(cfg_args, source_checkpoint)
	cfg.device_id = int(args.gpu_id)
	cfg.gpu_id = int(args.gpu_id)
	cfg.batch_size = int(args.batch_size)
	cfg.num_envs = int(args.batch_size)
	cfg.horizon = int(args.horizon)
	cfg.compile = False
	cfg.lr_schedule = None
	cfg.rank = 0
	cfg.world_size = 1
	cfg.finetune = True  # Preserve the source checkpoint's action-mask architecture.
	cfg.multitask_continuation_enabled = False
	cfg.multitask_prox_reg_enabled = False
	cfg.latent_residual_enabled = False
	return cfg, compat


def _preserved_key(key: str, variant: str) -> bool:
	encoder_prefixes = (
		"_task_encoder.", "_encoder.", "_contact_encoder.",
		"_task_context_adapters.encoder.",
	)
	dynamics_prefixes = ("_dynamics.", "_task_context_adapters.dynamics.")
	if variant in {"C", "D"} and key.startswith(encoder_prefixes):
		return True
	if variant == "D" and key.startswith(dynamics_prefixes):
		return True
	return False


def _model_for_variant(cfg, source_state, fresh_state, variant: str, device: torch.device):
	model = WorldModel(copy.deepcopy(cfg)).to(device)
	state = {key: value.clone() for key, value in fresh_state.items()}
	# These buffers are architecture/task-contract constants, not learned weights.
	buffer_keys = [key for key in source_state if key in state and key not in dict(model.named_parameters())]
	for key in buffer_keys:
		state[key] = source_state[key].clone()
	preserved = [key for key in source_state if key in state and _preserved_key(key, variant)]
	for key in preserved:
		state[key] = source_state[key].clone()
	model.load_state_dict(state, strict=True)
	return model, preserved, buffer_keys


def _rng_state():
	return {
		"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
		"cuda": torch.cuda.get_rng_state_all(),
	}


def _set_rng_state(state):
	random.setstate(state["python"])
	np.random.set_state(state["numpy"])
	torch.set_rng_state(state["torch"])
	torch.cuda.set_rng_state_all(state["cuda"])


def _seed_all(seed: int):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def _checkpoint_payload(agent, *, variant: str, updates: int, sampler, protocol, progress: bool):
	metadata = agent._checkpoint_metadata()
	metadata.update({
		"phase": "4.0", "phase4_variant": variant, "phase4_updates": int(updates),
		"phase4_progress_checkpoint": bool(progress), "phase4_protocol": protocol,
	})
	return {
		"model": agent.model.state_dict(), "optim": agent.optim.state_dict(),
		"pi_optim": agent.pi_optim.state_dict(), "scale": agent.scale.state_dict(),
		"metadata": metadata,
		"phase4_resume": {
			"updates": int(updates), "sampler": sampler.state_dict(), "rng": _rng_state(),
		},
	}


def _metric_float(value):
	if torch.is_tensor(value):
		return float(value.detach().float().mean().cpu().item())
	return float(value)


def train_variant(args, variant, cfg, source_state, fresh_state, replay_items, protocol, device):
	output_dir = resolve(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	final_path = output_dir / f"variant_{variant}.pt"
	progress_path = output_dir / f"variant_{variant}.progress.pt"
	log_path = output_dir / f"variant_{variant}_train.json"
	if final_path.exists() and not args.overwrite:
		print(f"[phase4.0] {variant}: final checkpoint exists, skipping: {final_path}", flush=True)
		return

	_seed_all(int(args.train_seed))
	model, preserved, buffer_keys = _model_for_variant(cfg, source_state, fresh_state, variant, device)
	agent = TDMPC2(model, copy.deepcopy(cfg)).to(device)
	sampler = TaskPhaseBalancedBuffer(replay_items, args.batch_size, args.horizon, args.sample_seed)
	start = 0
	if progress_path.exists() and not args.no_resume:
		payload = torch.load(progress_path, map_location=device, weights_only=False)
		agent.model.load_state_dict(payload["model"], strict=True)
		agent.optim.load_state_dict(payload["optim"])
		agent.pi_optim.load_state_dict(payload["pi_optim"])
		agent.scale.load_state_dict(payload["scale"])
		resume = payload["phase4_resume"]
		sampler.load_state_dict(resume["sampler"])
		_set_rng_state(resume["rng"])
		start = int(resume["updates"])
		print(f"[phase4.0] {variant}: resumed at update {start}", flush=True)

	initial = agent.model.state_dict()
	init_record = {
		"variant": variant,
		"preserved_parameter_keys": preserved,
		"preserved_parameter_count": int(sum(initial[key].numel() for key in preserved)),
		"architecture_buffer_keys_copied": buffer_keys,
		"initial_model_sha256": _state_hash(initial),
		"preserved_source_sha256": _state_hash(source_state, preserved) if preserved else None,
		"preserved_loaded_sha256": _state_hash(initial, preserved) if preserved else None,
	}
	if preserved and init_record["preserved_source_sha256"] != init_record["preserved_loaded_sha256"]:
		raise RuntimeError(f"{variant} preserved-module hash mismatch")

	history = []
	started = time.time()
	for update in range(start, int(args.updates)):
		metrics = agent.update(sampler)
		completed = update + 1
		if completed == 1 or completed % int(args.log_every) == 0 or completed == int(args.updates):
			row = {
				"update": completed,
				"elapsed_seconds": time.time() - started,
				"metrics": {key: _metric_float(value) for key, value in metrics.items()},
				"task_counts": sampler.last_batch_task_counts,
				"phase_counts": sampler.last_batch_phase_counts,
				"cell_counts": sampler.last_batch_cell_counts,
			}
			history.append(row)
			print(
				f"[phase4.0] {variant} update={completed}/{args.updates} "
				f"loss={row['metrics'].get('total_loss', float('nan')):.4f} "
				f"elapsed={row['elapsed_seconds']:.1f}s", flush=True,
			)
		if completed % int(args.save_every) == 0 and completed < int(args.updates):
			torch.save(_checkpoint_payload(
				agent, variant=variant, updates=completed, sampler=sampler,
				protocol=protocol, progress=True,
			), progress_path)

	torch.save(_checkpoint_payload(
		agent, variant=variant, updates=args.updates, sampler=sampler,
		protocol=protocol, progress=False,
	), final_path)
	record = {
		"status": "PASS", "variant": variant, "checkpoint": str(final_path),
		"checkpoint_sha256": _sha256(final_path), "updates": int(args.updates),
		"initialization": init_record, "protocol": protocol, "history": history,
		"final_sample_index": sampler.sample_index,
	}
	_write_json(record, log_path)
	print(f"[phase4.0] {variant}: wrote {final_path}", flush=True)
	if progress_path.exists():
		progress_path.unlink()


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--variant", choices=("all", *VARIANTS), default="all")
	parser.add_argument("--source-checkpoint", default=DEFAULT_SOURCE_CHECKPOINT)
	parser.add_argument("--replay-01125", default=DEFAULT_REPLAYS["01125"])
	parser.add_argument("--replay-00256", default=DEFAULT_REPLAYS["00256"])
	parser.add_argument("--replay-00186", default=DEFAULT_REPLAYS["00186"])
	parser.add_argument("--config", default="configs/train/srsa_01125_imitation_relaxed.yaml")
	parser.add_argument("--output-dir", default="reports/phase4_0_multitask_origin/checkpoints")
	parser.add_argument("--updates", type=int, default=8447)
	parser.add_argument("--batch-size", type=int, default=1024)
	parser.add_argument("--horizon", type=int, default=3)
	parser.add_argument("--train-fraction", type=float, default=0.80)
	parser.add_argument("--split-seed", type=int, default=4040)
	parser.add_argument("--sample-seed", type=int, default=4041)
	parser.add_argument("--init-seed", type=int, default=4042)
	parser.add_argument("--train-seed", type=int, default=4043)
	parser.add_argument("--gpu-id", type=int, default=0)
	parser.add_argument("--log-every", type=int, default=100)
	parser.add_argument("--save-every", type=int, default=500)
	parser.add_argument("--no-resume", action="store_true")
	parser.add_argument("--overwrite", action="store_true")
	parser.add_argument("--dry-run", action="store_true")
	args = parser.parse_args()

	if args.updates <= 0 or args.batch_size < 9 or args.horizon != 3:
		raise ValueError("Require positive updates, batch_size >= 9, and the unchanged horizon=3 architecture.")
	if not 0.0 < args.train_fraction < 1.0:
		raise ValueError("train-fraction must be in (0,1).")
	source_checkpoint = resolve(args.source_checkpoint)
	if not source_checkpoint.exists():
		raise FileNotFoundError(source_checkpoint)
	for value in (args.replay_01125, args.replay_00256, args.replay_00186, args.config):
		if not resolve(value).exists():
			raise FileNotFoundError(resolve(value))
	if args.dry_run:
		print("PASS dry-run")
		print("device: physical cuda1 via CUDA_VISIBLE_DEVICES=1, logical cuda:0")
		print(f"variants: {VARIANTS if args.variant == 'all' else (args.variant,)}")
		print(f"matched optimizer updates per train arm: {args.updates}")
		print("batch contract: every update contains all 9 task x phase cells")
		return 0

	device = _require_cuda1(args.gpu_id)
	replay_items = _load_replays(args)
	cfg, compat = _load_cfg(args, source_checkpoint)
	payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
	source_state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
	_seed_all(args.init_seed)
	fresh_model = WorldModel(copy.deepcopy(cfg)).cpu()
	fresh_state = {key: value.detach().cpu().clone() for key, value in fresh_model.state_dict().items()}
	del fresh_model

	replay_manifest = OrderedDict()
	for task, item in replay_items.items():
		replay_manifest[task] = {
			"path": str(item["path"]), "sha256": _sha256(item["path"]),
			"transitions": int(item["td"].shape[0]),
			"train_episodes": int(item["train_episodes"].numel()),
			"val_episodes": int(item["val_episodes"].numel()),
			"train_valid_sequences_by_phase": {
				PHASES[index]: int(values.numel()) for index, values in item["train_starts"].items()
			},
			"val_valid_sequences_by_phase": {
				PHASES[index]: int(values.numel()) for index, values in item["val_starts"].items()
			},
			"task_vec": [float(value) for value in item["task_vec"].tolist()],
		}
	protocol = {
		"source_checkpoint_A": str(source_checkpoint),
		"source_checkpoint_A_sha256": _sha256(source_checkpoint),
		"matched_update_definition": (
			"8447 is the observed three-task optimizer exposure in Phase 3.1 (6086) plus Phase 3.2 (2361); "
			"pre-00186 single-task ancestry is excluded from the matched causal budget."
		),
		"updates": int(args.updates), "batch_size": int(args.batch_size), "horizon": int(args.horizon),
		"batch_cells": [f"{task}/{phase}" for task in TASKS for phase in PHASES],
		"batch_balance": "all 9 cells every update; counts differ by at most one and remainder rotates",
		"episode_split": {"train_fraction": args.train_fraction, "split_seed": args.split_seed},
		"sample_seed": args.sample_seed, "init_seed": args.init_seed, "train_seed": args.train_seed,
		"architecture_compatibility": compat, "replays": replay_manifest,
		"initialization": {
			"B": "full fresh initialization",
			"C": "fresh model plus source task/obs/contact encoders and encoder-side task-context adapter",
			"D": "C plus source dynamics and dynamics-side task-context adapter",
			"training": "all parameters are trainable in B/C/D; retained modules specify initialization only",
		},
		"prohibitions": {
			"new_tasks": False, "elite_distillation": False, "counterfactual_reward_or_residual": False,
			"mppi_modified": False,
		},
		"device": {"physical": "cuda1", "visible": os.environ.get("CUDA_VISIBLE_DEVICES"), "logical": "cuda:0", "name": torch.cuda.get_device_name(0)},
	}
	_write_json(protocol, resolve(args.output_dir) / "protocol.json")

	variants = VARIANTS if args.variant == "all" else (args.variant,)
	for variant in variants:
		train_variant(args, variant, cfg, source_state, fresh_state, replay_items, protocol, device)
		torch.cuda.empty_cache()
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
