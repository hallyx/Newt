#!/usr/bin/env python3
"""Phase 4.0 fixed held-out offline evaluation for variants A/B/C/D."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
TDMPC2_ROOT = SCRIPT_DIR.parents[0]
REPO_ROOT = SCRIPT_DIR.parents[1]
PHASE3_DIR = SCRIPT_DIR / "phase3_three_task_pilot"
MODEL_AUDIT_DIR = SCRIPT_DIR / "model_task_sensitivity"
for path in (TDMPC2_ROOT, SCRIPT_DIR, PHASE3_DIR, MODEL_AUDIT_DIR):
	if str(path) not in sys.path:
		sys.path.insert(0, str(path))

import phase4_multitask_origin_train as phase4_train  # noqa: E402
import planner_action_attribution_diagnosis as attribution  # noqa: E402
import policy_conflict_mitigation_audit as mitigation  # noqa: E402
from common import math as td_math  # noqa: E402
from common.world_model import WorldModel  # noqa: E402
import task_vec_sensitivity_report as tvsr  # noqa: E402


TASKS = phase4_train.TASKS
PHASES = phase4_train.PHASES
VARIANTS = ("A", "B", "C", "D")


def resolve(value: str | Path) -> Path:
	path = Path(value).expanduser()
	return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def _sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as handle:
		for block in iter(lambda: handle.read(1024 * 1024), b""):
			digest.update(block)
	return digest.hexdigest()


def _write_json(value: Any, path: Path):
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _require_cuda1(gpu_id: int):
	visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
	if visible != "1" or int(gpu_id) != 0:
		raise RuntimeError("Use physical CUDA1 via CUDA_VISIBLE_DEVICES=1 and logical --gpu-id 0.")
	if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
		raise RuntimeError("Expected exactly one visible CUDA device.")
	torch.cuda.set_device(0)
	return torch.device("cuda:0")


def _corr(left: torch.Tensor, right: torch.Tensor) -> float:
	left = left.detach().float().reshape(-1).cpu()
	right = right.detach().float().reshape(-1).cpu()
	finite = torch.isfinite(left) & torch.isfinite(right)
	left, right = left[finite], right[finite]
	if left.numel() < 2 or left.std(unbiased=False) <= 1e-8 or right.std(unbiased=False) <= 1e-8:
		return math.nan
	return float(torch.corrcoef(torch.stack((left, right)))[0, 1].item())


def _summary(value: torch.Tensor) -> dict[str, float | int]:
	value = value.detach().float().reshape(-1).cpu()
	value = value[torch.isfinite(value)]
	if value.numel() == 0:
		return {"count": 0, "mean": math.nan, "p50": math.nan, "p95": math.nan, "max": math.nan}
	return {
		"count": int(value.numel()), "mean": float(value.mean().item()),
		"p50": float(torch.quantile(value, .50).item()), "p95": float(torch.quantile(value, .95).item()),
		"max": float(value.max().item()),
	}


def _calibration(prediction: torch.Tensor, target: torch.Tensor):
	delta = prediction.detach().float().reshape(-1).cpu() - target.detach().float().reshape(-1).cpu()
	return {
		"prediction": _summary(prediction), "target": _summary(target),
		"mae": float(delta.abs().mean().item()),
		"rmse": float(torch.sqrt(delta.square().mean()).item()),
		"bias": float(delta.mean().item()), "pearson": _corr(prediction, target),
	}


def _discounted_return_to_go(td, discount: float):
	episodes = td["episode"].detach().long().reshape(-1)
	reward = td["reward"].detach().float().reshape(-1)
	result = torch.zeros_like(reward)
	for episode_id in torch.unique(episodes, sorted=True).tolist():
		idx = torch.nonzero(episodes == int(episode_id), as_tuple=False).reshape(-1)
		running = reward.new_zeros(())
		for row in reversed(idx.tolist()):
			running = reward[row] + float(discount) * running
			result[row] = running
	return result


def _fixed_eval_bank(replay_items, per_cell: int, seed: int, discount: float):
	generator = torch.Generator().manual_seed(int(seed))
	bank = OrderedDict()
	for task in TASKS:
		item = replay_items[task]
		item["return_to_go"] = _discounted_return_to_go(item["td"], discount)
		for phase_id, phase in enumerate(PHASES):
			starts = item["val_starts"][phase_id]
			count = min(int(per_cell), int(starts.numel()))
			order = torch.randperm(int(starts.numel()), generator=generator)[:count]
			bank[f"{task}/{phase}"] = starts[order]
	return bank


def _load_model(checkpoint: Path, args, device):
	cfg_args = SimpleNamespace(
		config=str(resolve(args.config)), gpu_id=args.gpu_id, batch_size=args.eval_batch_size,
		assembly_id="00186", eval_task_id=2,
	)
	cfg, compat = tvsr._load_config(cfg_args, checkpoint)
	cfg.device_id = args.gpu_id
	cfg.compile = False
	model = WorldModel(cfg).to(device)
	model = tvsr._load_world_model(model, checkpoint, cfg)
	model.eval()
	return model, cfg, compat


@torch.no_grad()
def _task_context_report(model, replay_items, device):
	labels = list(TASKS)
	task = torch.stack([replay_items[label]["task_vec"] for label in labels]).to(device)
	dummy = task.new_zeros(task.shape[0], 1)
	info = model.task_context_repair_info(dummy, task, reconstruct=True)
	if info is None:
		return {"available": False}
	context, target, recon = info["task_context"], info["task_vec_norm"], info["task_recon"]
	distances = OrderedDict()
	ctx_values, vec_values = [], []
	for left in range(len(labels)):
		for right in range(left + 1, len(labels)):
			name = f"{labels[left]}_vs_{labels[right]}"
			distances[name] = float(torch.linalg.vector_norm(context[left] - context[right]).item())
			ctx_values.append(distances[name])
			vec_values.append(float(torch.linalg.vector_norm(target[left] - target[right]).item()))
	sse = (recon - target).square().sum()
	sst = (target - target.mean(0, keepdim=True)).square().sum().clamp_min(1e-8)
	return {
		"available": True,
		"pairwise_l2": distances,
		"context_distance_vs_task_distance_pearson": _corr(torch.tensor(ctx_values), torch.tensor(vec_values)),
		"task_reconstruction_r2": float((1.0 - sse / sst).item()),
		"task_reconstruction_mse": float((recon - target).square().mean().item()),
		"context_vectors": {label: [float(value) for value in context[index].cpu().tolist()] for index, label in enumerate(labels)},
	}


@torch.no_grad()
def _cell_report(model, cfg, replay_items, bank, device, batch_size: int):
	result = OrderedDict()
	all_dynamics = {horizon: [] for horizon in range(1, 4)}
	all_reward_pred, all_reward_target, all_q_pred, all_q_target = [], [], [], []
	for cell, starts in bank.items():
		task, phase = cell.split("/")
		item = replay_items[task]
		td = item["td"]
		dynamics = {horizon: [] for horizon in range(1, 4)}
		reward_pred, reward_target, q_pred, q_target = [], [], [], []
		for offset in range(0, int(starts.numel()), int(batch_size)):
			selected = starts[offset:offset + int(batch_size)]
			seq = selected[:, None] + torch.arange(4)[None, :]
			obs = td["obs"][seq].detach().float().permute(1, 0, 2).to(device)
			action = td["action"][seq[:, 1:]].detach().float().permute(1, 0, 2).to(device)
			reward = td["reward"][seq[:, 1:]].detach().float().reshape(-1, 3, 1).permute(1, 0, 2).to(device)
			task_seq = td["task"][seq[:, 1:]].detach().float().permute(1, 0, 2).to(device)
			encoded = model.encode(obs[1:], task_seq)
			z = model.encode(obs[0], task_seq[0])
			for horizon in range(1, 4):
				z = model.next(z, action[horizon - 1], task_seq[horizon - 1])
				error = torch.linalg.vector_norm(z - encoded[horizon - 1], dim=-1)
				dynamics[horizon].append(error.cpu())
				all_dynamics[horizon].append(error.cpu())

			# Calibration uses actual encoded states, not recursively predicted z.
			actual_z = model.encode(obs[:-1], task_seq)
			rew = td_math.two_hot_inv(model.reward(actual_z, action, task_seq), cfg)
			q = td_math.two_hot_inv(model.Q(actual_z, action, task_seq, return_type="all"), cfg).mean(0)
			rows = seq[:, 1:].reshape(-1)
			q_truth = item["return_to_go"][rows].reshape(selected.numel(), 3, 1).permute(1, 0, 2)
			reward_pred.append(rew.cpu()); reward_target.append(reward.cpu())
			q_pred.append(q.cpu()); q_target.append(q_truth.cpu())
			all_reward_pred.append(rew.cpu()); all_reward_target.append(reward.cpu())
			all_q_pred.append(q.cpu()); all_q_target.append(q_truth.cpu())
		result[cell] = {
			"task": task, "phase": phase, "sequences": int(starts.numel()),
			"multi_step_dynamics_latent_l2": {
				str(horizon): _summary(torch.cat(dynamics[horizon])) for horizon in range(1, 4)
			},
			"reward_calibration": _calibration(torch.cat(reward_pred, dim=1), torch.cat(reward_target, dim=1)),
			"q_calibration_to_replay_return": _calibration(torch.cat(q_pred, dim=1), torch.cat(q_target, dim=1)),
		}
	return {
		"by_task_phase": result,
		"aggregate": {
			"multi_step_dynamics_latent_l2": {
				str(horizon): _summary(torch.cat(all_dynamics[horizon])) for horizon in range(1, 4)
			},
			"reward_calibration": _calibration(torch.cat(all_reward_pred, dim=1), torch.cat(all_reward_target, dim=1)),
			"q_calibration_to_replay_return": _calibration(torch.cat(all_q_pred, dim=1), torch.cat(all_q_target, dim=1)),
		},
	}


def _proposal_regret(model, args, replay_items, device):
	rollout_args = SimpleNamespace(
		rollout_root=args.rollout_root, jam_lateral_threshold=.008, jam_keypoint_threshold=.012,
		jam_force_excursion_threshold=2.0,
	)
	banks, _ = attribution._load_rollout_banks(rollout_args)
	state_bank = mitigation._contact_jam_states(banks, replay_items["00186"]["task_vec"])
	if state_bank["obs"].shape[0] > args.proposal_states:
		generator = torch.Generator().manual_seed(args.eval_seed + 91)
		idx = torch.randperm(state_bank["obs"].shape[0], generator=generator)[:args.proposal_states]
		state_bank = {"obs": state_bank["obs"][idx], "task_vec": state_bank["task_vec"]}
	direct_checkpoint = resolve(args.direct_checkpoint)
	direct_args = SimpleNamespace(
		config=args.config, gpu_id=args.gpu_id, batch_size=args.eval_batch_size,
	)
	direct_model, direct_cfg, _ = attribution._load_model(direct_checkpoint, direct_args, device)
	metric_args = SimpleNamespace(
		seed=args.eval_seed, horizon=3, num_candidates=args.proposal_candidates,
		proposal_batch_size=args.proposal_batch_size,
	)
	return {
		"definition": "fixed Phase 3.3 direct-finetune scorer on held-out 00186 contact/jam states",
		"direct_scorer_checkpoint": str(direct_checkpoint),
		"states": int(state_bank["obs"].shape[0]),
		"summary": mitigation._proposal_regret(direct_model, direct_cfg, model, state_bank, device, metric_args),
	}


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--variant", default="all")
	parser.add_argument("--checkpoint", default=None, help="Explicit checkpoint for a single named variant.")
	parser.add_argument("--source-checkpoint", default=phase4_train.DEFAULT_SOURCE_CHECKPOINT)
	parser.add_argument("--checkpoint-dir", default="reports/phase4_0_multitask_origin/checkpoints")
	parser.add_argument("--output-dir", default="reports/phase4_0_multitask_origin/offline")
	parser.add_argument("--replay-01125", default=phase4_train.DEFAULT_REPLAYS["01125"])
	parser.add_argument("--replay-00256", default=phase4_train.DEFAULT_REPLAYS["00256"])
	parser.add_argument("--replay-00186", default=phase4_train.DEFAULT_REPLAYS["00186"])
	parser.add_argument("--direct-checkpoint", default="logs/isaaclab-srsa-assembly/1/srsa_axial_direct_finetune_from_01125/20260525_112528_asm-00186/models/best_step-600000_s-0p4133.pt")
	parser.add_argument("--rollout-root", default="reports/phase3_three_task_pilot/phase3_3_rollouts")
	parser.add_argument("--config", default="configs/train/srsa_01125_imitation_relaxed.yaml")
	parser.add_argument("--eval-per-cell", type=int, default=256)
	parser.add_argument("--eval-batch-size", type=int, default=256)
	parser.add_argument("--eval-seed", type=int, default=4050)
	parser.add_argument("--split-seed", type=int, default=4040)
	parser.add_argument("--train-fraction", type=float, default=.80)
	parser.add_argument("--proposal-states", type=int, default=128)
	parser.add_argument("--proposal-candidates", type=int, default=128)
	parser.add_argument("--proposal-batch-size", type=int, default=32)
	parser.add_argument("--gpu-id", type=int, default=0)
	parser.add_argument("--dry-run", action="store_true")
	args = parser.parse_args()

	checkpoints = OrderedDict([("A", resolve(args.source_checkpoint))])
	for variant in ("B", "C", "D"):
		checkpoints[variant] = resolve(args.checkpoint_dir) / f"variant_{variant}.pt"
	variants = VARIANTS if args.variant == "all" else (args.variant,)
	if args.checkpoint is not None:
		if args.variant == "all":
			raise ValueError("--checkpoint requires a single --variant label.")
		checkpoints[args.variant] = resolve(args.checkpoint)
	for variant in variants:
		if variant not in checkpoints:
			raise ValueError(f"Unknown variant={variant!r}; provide --checkpoint for a custom label.")
		if not checkpoints[variant].exists():
			raise FileNotFoundError(checkpoints[variant])
	if args.dry_run:
		print("PASS dry-run")
		for variant in variants:
			print(f"{variant}: {checkpoints[variant]}")
		print("device: physical cuda1 via CUDA_VISIBLE_DEVICES=1, logical cuda:0")
		return 0

	device = _require_cuda1(args.gpu_id)
	replay_args = SimpleNamespace(
		replay_01125=args.replay_01125, replay_00256=args.replay_00256, replay_00186=args.replay_00186,
		train_fraction=args.train_fraction, split_seed=args.split_seed, horizon=3,
	)
	replay_items = phase4_train._load_replays(replay_args)
	discount = .95
	bank = _fixed_eval_bank(replay_items, args.eval_per_cell, args.eval_seed, discount)
	for variant in variants:
		checkpoint = checkpoints[variant]
		hash_before = _sha256(checkpoint)
		model, cfg, compat = _load_model(checkpoint, args, device)
		report = {
			"status": "PASS", "variant": variant, "checkpoint": str(checkpoint),
			"checkpoint_sha256_before": hash_before,
			"device": {"physical": "cuda1", "visible": os.environ.get("CUDA_VISIBLE_DEVICES"), "logical": "cuda:0"},
			"heldout_contract": {
				"episode_disjoint_train_fraction": args.train_fraction, "split_seed": args.split_seed,
				"eval_seed": args.eval_seed, "requested_sequences_per_task_phase": args.eval_per_cell,
				"phase_label": "phase of first supervised transition in each native horizon-3 sequence",
				"q_target": f"replay discounted return-to-go, gamma={discount}",
			},
			"compatibility": compat,
			"task_context_structure": _task_context_report(model, replay_items, device),
			"world_model": _cell_report(model, cfg, replay_items, bank, device, args.eval_batch_size),
			"proposal_regret": _proposal_regret(model, args, replay_items, device),
		}
		del model
		torch.cuda.empty_cache()
		report["checkpoint_sha256_after"] = _sha256(checkpoint)
		report["checkpoint_unchanged"] = report["checkpoint_sha256_before"] == report["checkpoint_sha256_after"]
		if not report["checkpoint_unchanged"]:
			raise RuntimeError(f"Checkpoint mutated during offline eval: {checkpoint}")
		output = resolve(args.output_dir) / f"variant_{variant}.json"
		_write_json(report, output)
		print(f"[phase4.0] offline {variant}: wrote {output}", flush=True)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
