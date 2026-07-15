#!/usr/bin/env python3
"""Read-only 00186 standalone/direct/multitask acquisition diagnosis."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_AUDIT_DIR = SCRIPT_DIR.parent / "model_task_sensitivity"
if str(MODEL_AUDIT_DIR) not in sys.path:
	sys.path.insert(0, str(MODEL_AUDIT_DIR))

from _common import (  # noqa: E402
	condition_batch,
	resolve,
	summarize_tensor,
	tensor_to_list,
	two_hot_scalar,
	tvsr,
	write_json,
	WorldModel,
)


DEFAULT_STANDALONE = (
	"logs/isaaclab-srsa-assembly/1/srsa_axial_online/20260517_231404_asm-00186/"
	"models/best_step-4000000_s-1p0000.pt"
)
DEFAULT_DIRECT = (
	"logs/isaaclab-srsa-assembly/1/srsa_axial_direct_finetune_from_01125/"
	"20260525_112528_asm-00186/models/best_step-600000_s-0p4133.pt"
)
DEFAULT_MULTITASK = (
	"logs/isaaclab-srsa-assembly/1/"
	"srsa_axial_online_family_taskctx_repair_01125_00256_00186_rescue/"
	"20260713_phase3_2_rescue_00186_stage-3_asm-00186/models/"
	"best_step-50176_s-0p2461.pt"
)
DEFAULT_REPLAY = (
	"logs/isaaclab-srsa-assembly/1/"
	"srsa_axial_online_family_taskctx_repair_01125_00256_00186_rescue/"
	"20260713_phase3_2_rescue_00186_launcher/replay/00186.pt"
)
DEFAULT_OUTPUT = (
	"reports/phase3_three_task_pilot/phase3_2_diagnosis/"
	"standalone_vs_multitask_diagnosis.json"
)


def _load_model(checkpoint: Path, args: argparse.Namespace, device: torch.device):
	cfg_args = SimpleNamespace(
		config=args.config,
		gpu_id=args.gpu_id,
		batch_size=args.batch_size,
		assembly_id="00186",
		eval_task_id=2,
	)
	cfg, compat = tvsr._load_config(cfg_args, checkpoint)
	cfg.device_id = int(args.gpu_id) if device.type == "cuda" else 0
	model = WorldModel(cfg).to(device)
	model = tvsr._load_world_model(model, checkpoint, cfg)
	model.eval()
	return model, cfg, compat


def _per_dim(value: torch.Tensor) -> dict[str, list[float]]:
	value = value.detach().float().cpu()
	return {
		"mean": tensor_to_list(value.mean(dim=0)),
		"std": tensor_to_list(value.std(dim=0, unbiased=False)),
		"p95_abs": tensor_to_list(torch.quantile(value.abs(), 0.95, dim=0)),
	}


def _correlation(x: torch.Tensor, y: torch.Tensor) -> float:
	x = x.detach().float().reshape(-1).cpu()
	y = y.detach().float().reshape(-1).cpu()
	if x.numel() < 2 or x.std() <= 1.0e-8 or y.std() <= 1.0e-8:
		return math.nan
	return float(torch.corrcoef(torch.stack([x, y]))[0, 1].item())


def _match_action_dim(action: torch.Tensor, action_dim: int) -> torch.Tensor:
	if int(action.shape[-1]) == int(action_dim):
		return action
	if int(action.shape[-1]) > int(action_dim):
		return action[..., :action_dim].contiguous()
	padding = action.new_zeros(*action.shape[:-1], int(action_dim) - int(action.shape[-1]))
	return torch.cat([action, padding], dim=-1).contiguous()


@torch.no_grad()
def _trajectory_returns(model, cfg, obs, candidates, task_vec):
	batch_size, horizon, num_candidates, action_dim = candidates.shape
	task_obs = condition_batch(task_vec, (batch_size,), obs.device)
	z0 = model.encode(obs, task_obs)
	z = z0.unsqueeze(1).expand(batch_size, num_candidates, z0.shape[-1]).reshape(batch_size * num_candidates, -1)
	task = condition_batch(task_vec, (batch_size * num_candidates,), obs.device)
	value = torch.zeros(batch_size * num_candidates, 1, device=obs.device)
	discount = torch.ones_like(value)
	for step in range(horizon):
		action = candidates[:, step].reshape(batch_size * num_candidates, action_dim)
		value = value + discount * two_hot_scalar(model.reward(z, action, task), cfg)
		z = model.next(z, action, task)
		discount = discount * float(cfg.get("discount", 0.99))
	terminal_action, _ = model.pi(z, task)
	value = value + discount * model.Q(z, terminal_action, task, return_type="avg")
	return value.reshape(batch_size, num_candidates)


@torch.no_grad()
def _model_report(model, cfg, obs, replay_action, replay_reward, task_vec, candidates, action_dim):
	replay_action = _match_action_dim(replay_action, action_dim)
	model_candidates = _match_action_dim(candidates, action_dim)
	task = condition_batch(task_vec, (obs.shape[0],), obs.device)
	z = model.encode(obs, task)
	_, pi_info = model.pi(z, task)
	pi_mean = pi_info["mean"]
	pi_std = pi_info["log_std"].exp()
	reward_pred = two_hot_scalar(model.reward(z, replay_action, task), cfg)
	q_pred = two_hot_scalar(model.Q(z, replay_action, task, return_type="all"), cfg).mean(dim=0)
	returns = _trajectory_returns(model, cfg, obs[: model_candidates.shape[0]], model_candidates, task_vec)
	top = torch.argmax(returns, dim=-1)
	first_actions = model_candidates[:, 0]
	selected_full = first_actions[torch.arange(first_actions.shape[0], device=obs.device), top]
	selected = selected_full[..., : int(candidates.shape[-1])]
	pi_mean_env = pi_mean[..., : int(candidates.shape[-1])]
	return {
		"pi_mean": pi_mean_env,
		"pi_std": pi_std,
		"reward_pred": reward_pred,
		"q_pred": q_pred,
		"candidate_returns": returns,
		"selected_action": selected,
		"summary": {
			"policy_action_distribution": {
				"model_action_dim": int(action_dim),
				"common_env_action_dims": int(pi_mean_env.shape[-1]),
				"per_dim": _per_dim(pi_mean),
				"common_env_per_dim": _per_dim(pi_mean_env),
				"norm": summarize_tensor(torch.linalg.vector_norm(pi_mean, dim=-1)),
				"std_mean": float(pi_std.mean().item()),
			},
			"reward_calibration": {
				"prediction": summarize_tensor(reward_pred),
				"target": summarize_tensor(replay_reward),
				"mae": float((reward_pred - replay_reward).abs().mean().item()),
				"bias": float((reward_pred - replay_reward).mean().item()),
				"correlation": _correlation(reward_pred, replay_reward),
			},
			"q_calibration": {
				"prediction": summarize_tensor(q_pred),
				"q_reward_correlation": _correlation(q_pred, replay_reward),
			},
			"fixed_candidate_mppi": {
				"selected_action_per_dim": _per_dim(selected),
				"selected_action_norm": summarize_tensor(torch.linalg.vector_norm(selected, dim=-1)),
				"selected_return": summarize_tensor(returns.max(dim=-1).values),
				"return_margin_top1_top2": summarize_tensor(
					torch.topk(returns, k=2, dim=-1).values[:, 0] - torch.topk(returns, k=2, dim=-1).values[:, 1]
				),
			},
		},
	}


def _pairwise(outputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
	result = OrderedDict()
	labels = list(outputs)
	for i, left in enumerate(labels):
		for right in labels[i + 1:]:
			a = outputs[left]
			b = outputs[right]
			result[f"{left}_vs_{right}"] = {
				"policy_action_mean_l2": summarize_tensor(torch.linalg.vector_norm(a["pi_mean"] - b["pi_mean"], dim=-1)),
				"mppi_selected_action_l2": summarize_tensor(torch.linalg.vector_norm(a["selected_action"] - b["selected_action"], dim=-1)),
				"reward_prediction_abs_delta": summarize_tensor((a["reward_pred"] - b["reward_pred"]).abs()),
				"q_prediction_abs_delta": summarize_tensor((a["q_pred"] - b["q_pred"]).abs()),
			}
	return result


@torch.no_grad()
def build_report(args: argparse.Namespace) -> dict[str, Any]:
	device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() and not args.cpu else "cpu")
	if device.type == "cuda":
		torch.cuda.set_device(device)
	torch.manual_seed(int(args.seed))
	replay_path = resolve(args.replay)
	td, replay_metadata = tvsr._load_replay_tensordict(replay_path)
	obs, action, reward, indices = tvsr._sample_replay_rows(td, int(args.batch_size), int(args.seed))
	obs = obs.to(device)
	action = action.to(device)
	reward = reward.to(device)
	task_vec = tvsr._unique_task_vec_from_replay(replay_path)[0].float().to(device)
	checkpoints = OrderedDict([
		("standalone", resolve(args.standalone_checkpoint)),
		("direct_finetune", resolve(args.direct_checkpoint)),
		("multitask_rescue_best", resolve(args.multitask_checkpoint)),
	])
	for path in checkpoints.values():
		if not path.exists():
			raise FileNotFoundError(path)
	candidate_batch = min(int(args.candidate_batch_size), int(obs.shape[0]))
	generator = torch.Generator(device=device).manual_seed(int(args.seed))
	candidates = torch.empty(
		candidate_batch,
		int(args.horizon),
		int(args.num_candidates),
		int(action.shape[-1]),
		device=device,
	).uniform_(-1.0, 1.0, generator=generator)
	outputs = OrderedDict()
	compatibility = OrderedDict()
	for label, checkpoint in checkpoints.items():
		model, cfg, compat = _load_model(checkpoint, args, device)
		expected_obs_dim = int(compat["obs_dim"])
		if int(obs.shape[-1]) < expected_obs_dim:
			raise RuntimeError(f"Replay obs dim {obs.shape[-1]} < {label} obs dim {expected_obs_dim}")
		model_obs = obs[..., :expected_obs_dim]
		action_dim = int(compat.get("action_dim") or action.shape[-1])
		outputs[label] = _model_report(model, cfg, model_obs, action, reward, task_vec, candidates, action_dim)
		compatibility[label] = compat
	report = {
		"status": "PASS",
		"device": str(device),
		"replay": str(replay_path),
		"replay_metadata": replay_metadata,
		"task_vec_00186": tensor_to_list(task_vec),
		"sample_size": int(obs.shape[0]),
		"sample_indices_first10": [int(x) for x in indices[:10].tolist()],
		"candidate_batch_size": candidate_batch,
		"num_candidates": int(args.num_candidates),
		"horizon": int(args.horizon),
		"candidate_source": "same uniform[-1,1] candidates for all models",
		"checkpoints": {key: str(value) for key, value in checkpoints.items()},
		"checkpoint_compatibility": compatibility,
		"models": {key: value["summary"] for key, value in outputs.items()},
		"pairwise": _pairwise(outputs),
	}
	return report


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--standalone-checkpoint", default=DEFAULT_STANDALONE)
	parser.add_argument("--direct-checkpoint", default=DEFAULT_DIRECT)
	parser.add_argument("--multitask-checkpoint", default=DEFAULT_MULTITASK)
	parser.add_argument("--replay", default=DEFAULT_REPLAY)
	parser.add_argument("--config", default="configs/train/srsa_01125_imitation_relaxed.yaml")
	parser.add_argument("--output", default=DEFAULT_OUTPUT)
	parser.add_argument("--batch-size", type=int, default=512)
	parser.add_argument("--candidate-batch-size", type=int, default=64)
	parser.add_argument("--num-candidates", type=int, default=128)
	parser.add_argument("--horizon", type=int, default=3)
	parser.add_argument("--seed", type=int, default=1)
	parser.add_argument("--gpu-id", type=int, default=0)
	parser.add_argument("--cpu", action="store_true")
	parser.add_argument("--dry-run", action="store_true")
	args = parser.parse_args()
	if args.dry_run:
		print(f"PASS dry-run: would write {resolve(args.output)}")
		return 0
	report = build_report(args)
	write_json(report, args.output)
	print("PASS")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
