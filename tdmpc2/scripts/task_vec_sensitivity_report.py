#!/usr/bin/env python3
"""Offline paired task-vector sensitivity report for SRSA TD-MPC2 checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import torch
from omegaconf import OmegaConf

TDMPC2_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TDMPC2_ROOT.parent
if str(TDMPC2_ROOT) not in sys.path:
	sys.path.insert(0, str(TDMPC2_ROOT))

from common import MODEL_SIZE, math as td_math  # noqa: E402
from common.layers import api_model_conversion, legacy_api_model_conversion  # noqa: E402
from common.world_model import WorldModel  # noqa: E402
from config import Config, parse_cfg  # noqa: E402


DEFAULT_CHECKPOINT = (
	"logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_acquire_from_01125/"
	"20260618_001734_stage-2_asm-00256/models/latest.pt"
)
DEFAULT_REPLAY = (
	"logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_acquire_from_01125/"
	"20260618_001734_launcher/replay/00256.pt"
)
DEFAULT_ANCHOR_REPLAY = (
	"logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_replay_from_01125/"
	"20260615_202326_launcher/replay/01125.pt"
)


def _resolve(path_value: str | Path) -> Path:
	path = Path(path_value).expanduser()
	if not path.is_absolute():
		path = REPO_ROOT / path
	return path.resolve()


def _checkpoint_state_dict(checkpoint_fp: Path):
	obj = torch.load(checkpoint_fp, map_location="cpu", weights_only=False)
	return obj["model"] if isinstance(obj, dict) and "model" in obj else obj


def _checkpoint_metadata(checkpoint_fp: Path):
	obj = torch.load(checkpoint_fp, map_location="cpu", weights_only=False)
	if isinstance(obj, dict) and isinstance(obj.get("metadata"), dict):
		return dict(obj["metadata"])
	return {}


def _state_tensor(state_dict, key: str):
	for candidate in (key, f"module.{key}"):
		if candidate in state_dict:
			return state_dict[candidate]
	return None


def _infer_model_size(enc_dim: int):
	for name, values in MODEL_SIZE.items():
		if int(values.get("enc_dim", -1)) == int(enc_dim):
			return name
	return None


def _infer_checkpoint_compat(checkpoint_fp: Path):
	state_dict = _checkpoint_state_dict(checkpoint_fp)
	metadata = _checkpoint_metadata(checkpoint_fp)
	enc_weight = _state_tensor(state_dict, "_encoder.state.0.weight")
	if enc_weight is None:
		raise KeyError(f"Could not infer encoder input from checkpoint={checkpoint_fp}")
	task_emb = _state_tensor(state_dict, "_task_emb.weight")
	task_vecs = _state_tensor(state_dict, "_task_vecs")
	task_encoder = _state_tensor(state_dict, "_task_encoder.type_encoder.weight")
	action_masks = _state_tensor(state_dict, "_action_masks")
	if task_emb is not None:
		task_conditioning = "id_embedding"
		task_dim = int(task_emb.shape[-1])
	elif task_vecs is not None or task_encoder is not None:
		task_conditioning = "axial_params"
		task_dim = 64
	else:
		task_conditioning = "none"
		task_dim = 0
	adapter_sites = {}
	adapter_hidden_dim = None
	adapter_source = None
	adapter_alpha = metadata.get("task_context_adapter_alpha", None)
	for site in ("encoder", "dynamics", "pi", "reward", "q"):
		weight = _state_tensor(state_dict, f"_task_context_adapters.{site}.net.0.weight")
		if weight is not None:
			adapter_sites[site] = True
			adapter_hidden_dim = int(weight.shape[0])
			input_dim = int(weight.shape[1])
			if input_dim == int(task_dim):
				adapter_source = "task_context"
			elif input_dim == 6:
				adapter_source = "raw_task_vec"
			elif input_dim == int(task_dim) + 6:
				adapter_source = "both"
			else:
				adapter_source = "task_context"
	if adapter_source is None and metadata.get("task_context_adapter_source", None) is not None:
		adapter_source = str(metadata["task_context_adapter_source"])
	obs_dim = int(enc_weight.shape[1]) - task_dim
	if obs_dim <= 0:
		raise ValueError(
			f"Could not infer positive obs_dim from checkpoint={checkpoint_fp}: "
			f"encoder_in={int(enc_weight.shape[1])}, task_dim={task_dim}."
		)
	return {
		"model_size": _infer_model_size(int(enc_weight.shape[0])),
		"obs_dim": obs_dim,
		"action_dim": int(action_masks.shape[-1]) if action_masks is not None else None,
		"task_conditioning": task_conditioning,
		"task_dim": task_dim,
		"task_context_adapter_enabled": bool(adapter_sites),
		"task_context_adapter_hidden_dim": adapter_hidden_dim,
		"task_context_adapter_alpha": float(adapter_alpha) if adapter_alpha is not None else None,
		"task_context_adapter_lr_scale": float(metadata["task_context_adapter_lr_scale"]) if metadata.get("task_context_adapter_lr_scale", None) is not None else None,
		"task_context_adapter_source": adapter_source,
		"task_context_adapter_apply_encoder": bool(adapter_sites.get("encoder", False)),
		"task_context_adapter_apply_dynamics": bool(adapter_sites.get("dynamics", False)),
		"task_context_adapter_apply_policy": bool(adapter_sites.get("pi", False)),
		"task_context_adapter_apply_reward": bool(adapter_sites.get("reward", False)),
		"task_context_adapter_apply_q": bool(adapter_sites.get("q", False)),
		"task_vec_normalization_enabled": bool(metadata.get("task_vec_normalization_enabled", False)),
		"task_vec_normalization_mean": metadata.get("task_vec_normalization_mean", None),
		"task_vec_normalization_std": metadata.get("task_vec_normalization_std", None),
		"task_vec_normalization_eps": metadata.get("task_vec_normalization_eps", None),
		"task_context_repair_enabled": bool(metadata.get("task_context_repair_enabled", False)),
		"task_recon_coef": float(metadata["task_recon_coef"]) if metadata.get("task_recon_coef", None) is not None else None,
		"task_spread_coef": float(metadata["task_spread_coef"]) if metadata.get("task_spread_coef", None) is not None else None,
		"task_raw_residual_scale": float(metadata["task_raw_residual_scale"]) if metadata.get("task_raw_residual_scale", None) is not None else None,
		"task_spread_near_threshold": float(metadata["task_spread_near_threshold"]) if metadata.get("task_spread_near_threshold", None) is not None else None,
		"task_spread_far_threshold": float(metadata["task_spread_far_threshold"]) if metadata.get("task_spread_far_threshold", None) is not None else None,
		"task_spread_margin": float(metadata["task_spread_margin"]) if metadata.get("task_spread_margin", None) is not None else None,
	}


def _load_config(args, checkpoint_fp: Path):
	config_fp = _resolve(args.config)
	cfg = OmegaConf.structured(Config)
	file_cfg = OmegaConf.load(config_fp)
	if "defaults" in file_cfg:
		file_cfg.pop("defaults")
	cfg = OmegaConf.merge(cfg, file_cfg)
	compat = _infer_checkpoint_compat(checkpoint_fp)
	if compat["model_size"] is not None:
		cfg.model_size = compat["model_size"]
	cfg.task_conditioning = compat["task_conditioning"]
	cfg.task_context_adapter_enabled = bool(compat.get("task_context_adapter_enabled", False))
	if compat.get("task_context_adapter_hidden_dim", None) is not None:
		cfg.task_context_adapter_hidden_dim = int(compat["task_context_adapter_hidden_dim"])
	if compat.get("task_context_adapter_alpha", None) is not None:
		cfg.task_context_adapter_alpha = float(compat["task_context_adapter_alpha"])
	if compat.get("task_context_adapter_lr_scale", None) is not None:
		cfg.task_context_adapter_lr_scale = float(compat["task_context_adapter_lr_scale"])
	if compat.get("task_context_adapter_source", None) is not None:
		cfg.task_context_adapter_source = str(compat["task_context_adapter_source"])
	cfg.task_context_adapter_apply_encoder = bool(compat.get("task_context_adapter_apply_encoder", False))
	cfg.task_context_adapter_apply_dynamics = bool(compat.get("task_context_adapter_apply_dynamics", False))
	cfg.task_context_adapter_apply_policy = bool(compat.get("task_context_adapter_apply_policy", False))
	cfg.task_context_adapter_apply_reward = bool(compat.get("task_context_adapter_apply_reward", False))
	cfg.task_context_adapter_apply_q = bool(compat.get("task_context_adapter_apply_q", False))
	cfg.task_vec_normalization_enabled = bool(compat.get("task_vec_normalization_enabled", False))
	if compat.get("task_vec_normalization_mean", None) is not None:
		cfg.task_vec_normalization_mean = compat["task_vec_normalization_mean"]
	if compat.get("task_vec_normalization_std", None) is not None:
		cfg.task_vec_normalization_std = compat["task_vec_normalization_std"]
	if compat.get("task_vec_normalization_eps", None) is not None:
		cfg.task_vec_normalization_eps = float(compat["task_vec_normalization_eps"])
	cfg.task_context_repair_enabled = bool(compat.get("task_context_repair_enabled", False))
	for key in (
		"task_recon_coef",
		"task_spread_coef",
		"task_raw_residual_scale",
		"task_spread_near_threshold",
		"task_spread_far_threshold",
		"task_spread_margin",
	):
		if compat.get(key, None) is not None:
			setattr(cfg, key, float(compat[key]))
	if compat["action_dim"] is not None:
		cfg.srsa_policy_action_dim = int(compat["action_dim"])
		cfg.isaaclab_action_dim = int(compat["action_dim"])
		cfg.action_dim = int(compat["action_dim"])
	cfg.obs_shape = {"state": [int(compat["obs_dim"])]}
	cfg.checkpoint = str(checkpoint_fp)
	cfg.gpu_id = int(args.gpu_id)
	cfg.num_gpus = 1
	cfg.rank = 0
	cfg.world_size = 1
	cfg.multiproc = False
	cfg.num_envs = max(1, int(args.batch_size))
	cfg.compile = False
	cfg.enable_wandb = False
	cfg.save_agent = False
	cfg.finetune = True
	cfg.assembly_id = str(args.assembly_id)
	cfg.eval_task_id = int(args.eval_task_id)
	cfg = parse_cfg(cfg)
	cfg.device_id = int(args.gpu_id)
	cfg.action_dim = int(compat["action_dim"] or cfg.get("action_dim", 0) or cfg.get("srsa_policy_action_dim", 3))
	cfg.obs_shape = {"state": [int(compat["obs_dim"])]}
	cfg.rank = 0
	cfg.world_size = 1
	cfg.num_envs = max(1, int(args.batch_size))
	cfg.checkpoint = str(checkpoint_fp)
	return cfg, compat


def _load_replay_tensordict(path: Path):
	payload = torch.load(path, map_location="cpu", weights_only=False)
	if isinstance(payload, dict) and payload.get("format") == "newt_buffer_snapshot_v1":
		return payload["data"], dict(payload.get("metadata", {}))
	return payload, {}


def _load_world_model(model: WorldModel, checkpoint_fp: Path, cfg):
	state_dict = _checkpoint_state_dict(checkpoint_fp)
	target_state = model.state_dict()
	prefix = "module." if any(key.startswith("module.") for key in state_dict.keys()) else ""
	if bool(getattr(cfg, "finetune", False)):
		if getattr(model, "_task_emb", None) is not None:
			state_dict[prefix + "_task_emb.weight"] = model._task_emb.weight
		if getattr(model, "_task_encoder", None) is not None:
			state_dict[prefix + "_task_vecs"] = model._task_vecs
		state_dict[prefix + "_action_masks"] = model._action_masks
	state_dict = api_model_conversion(target_state, state_dict)
	for key in ("_task_vecs", "_action_masks"):
		if key not in target_state:
			continue
		for source_key in (key, f"module.{key}"):
			if source_key not in state_dict:
				continue
			if tuple(state_dict[source_key].shape) != tuple(target_state[key].shape):
				state_dict[source_key] = target_state[key]
	repair_prefixes = (
		"_task_encoder.raw_residual.",
		"_task_encoder.decoder.",
		"module._task_encoder.raw_residual.",
		"module._task_encoder.decoder.",
	)
	for key, value in target_state.items():
		if key in state_dict:
			continue
		if any(key.startswith(prefix) for prefix in repair_prefixes):
			state_dict[key] = value
	try:
		model.load_state_dict(state_dict)
	except Exception as load_error:
		try:
			model.load_state_dict(legacy_api_model_conversion(target_state, state_dict))
		except Exception:
			raise RuntimeError(f"Failed to load checkpoint into WorldModel: {checkpoint_fp}") from load_error
	return model


def _unique_task_vec_from_replay(path: Path):
	td, metadata = _load_replay_tensordict(path)
	if "task" not in td.keys():
		raise KeyError(f"Replay does not contain `task`: {path}")
	task = td.get("task")
	flat = task.reshape(-1, task.shape[-1]).float()
	unique = torch.unique(flat, dim=0)
	if unique.shape[0] != 1:
		raise ValueError(f"Expected one unique task vector in {path}, got {int(unique.shape[0])}.")
	return unique[0], metadata


def _parse_vec(text: str) -> torch.Tensor:
	parts = [item for item in str(text).strip().strip("[]").replace(";", ",").split(",") if item.strip()]
	vec = torch.tensor([float(item) for item in parts], dtype=torch.float32)
	if vec.numel() != 6:
		raise ValueError(f"Expected 6 task-vector values, got {vec.numel()}: {text}")
	return vec


def _parse_labeled_vec(text: str):
	if "=" not in text:
		raise ValueError(f"Expected LABEL=v0,v1,v2,v3,v4,v5, got: {text}")
	label, vec_text = text.split("=", 1)
	label = label.strip()
	if not label:
		raise ValueError(f"Empty task-vector label in: {text}")
	return label, _parse_vec(vec_text)


def _parse_labeled_replay(text: str):
	if "=" not in text:
		raise ValueError(f"Expected LABEL=/path/to/replay.pt, got: {text}")
	label, path_text = text.split("=", 1)
	label = label.strip()
	if not label:
		raise ValueError(f"Empty replay label in: {text}")
	return label, _resolve(path_text.strip())


def _condition_vectors(args, replay_fp: Path):
	conditions = OrderedDict()
	correct_vec, correct_meta = _unique_task_vec_from_replay(replay_fp)
	conditions[str(args.base_label)] = correct_vec
	metadata = {"base_replay": correct_meta}

	anchor_fp = _resolve(args.anchor_replay) if args.anchor_replay else None
	if anchor_fp is not None and anchor_fp.exists():
		conditions["anchor_from_replay"] = _unique_task_vec_from_replay(anchor_fp)[0]
	for item in args.condition_from_replay:
		label, path = _parse_labeled_replay(item)
		conditions[label] = _unique_task_vec_from_replay(path)[0]
	for item in args.task_vec:
		label, vec = _parse_labeled_vec(item)
		conditions[label] = vec
	if args.include_zero:
		conditions.setdefault("zero", torch.zeros(6, dtype=torch.float32))
	if args.include_random:
		generator = torch.Generator().manual_seed(int(args.seed))
		conditions.setdefault("random", torch.empty(6, dtype=torch.float32).uniform_(-1.0, 1.0, generator=generator))
	if args.include_extreme:
		conditions.setdefault("extreme", torch.tensor([0.0, -0.5, 0.5, 0.5, 2.0, 0.0], dtype=torch.float32))
	if str(args.base_label) not in conditions:
		raise ValueError(f"Base label {args.base_label!r} is not present in conditions.")
	return conditions, metadata


def _sample_replay_rows(td, batch_size: int, seed: int):
	keys = list(td.keys())
	for key in ("obs", "action", "reward"):
		if key not in keys:
			raise KeyError(f"Replay is missing `{key}`.")
	obs = td.get("obs").float()
	action = td.get("action").float()
	reward = td.get("reward").float()
	mask = torch.isfinite(obs).all(dim=-1) & torch.isfinite(action).all(dim=-1) & torch.isfinite(reward)
	valid = torch.nonzero(mask, as_tuple=False).squeeze(-1)
	if valid.numel() == 0:
		raise RuntimeError("No finite replay rows found after filtering obs/action/reward.")
	generator = torch.Generator().manual_seed(int(seed))
	order = valid[torch.randperm(valid.numel(), generator=generator)]
	indices = order[: min(int(batch_size), int(order.numel()))]
	return obs[indices], action[indices], reward[indices], indices


def _l2(a, b):
	return torch.linalg.vector_norm(a - b, dim=-1)


def _abs(a, b):
	return (a - b).abs()


def _summary_tensor(value):
	value = value.detach().float().reshape(-1).cpu()
	return {
		"mean": float(value.mean().item()),
		"median": float(value.median().item()),
		"p95": float(torch.quantile(value, 0.95).item()),
		"max": float(value.max().item()),
	}


@torch.no_grad()
def _forward_condition(model, cfg, obs, action, task_vec):
	task = task_vec.to(device=obs.device, dtype=torch.float32).view(1, -1).repeat(obs.shape[0], 1)
	z = model.encode(obs, task)
	_, pi_info = model.pi(z, task)
	action_mean = pi_info["mean"]
	q_logits = model.Q(z, action, task, return_type="all")
	q_scalar = td_math.two_hot_inv(q_logits, cfg).mean(dim=0)
	reward_logits = model.reward(z, action, task)
	reward_scalar = td_math.two_hot_inv(reward_logits, cfg)
	next_z = model.next(z, action, task)
	return {
		"latent": z,
		"action_mean": action_mean,
		"q_scalar": q_scalar,
		"reward_scalar": reward_scalar,
		"next_latent": next_z,
	}


def _compare_outputs(base, other):
	return {
		"latent_l2": _summary_tensor(_l2(base["latent"], other["latent"])),
		"action_mean_l2": _summary_tensor(_l2(base["action_mean"], other["action_mean"])),
		"action_mean_abs": _summary_tensor(_abs(base["action_mean"], other["action_mean"])),
		"q_abs": _summary_tensor(_abs(base["q_scalar"], other["q_scalar"])),
		"reward_abs": _summary_tensor(_abs(base["reward_scalar"], other["reward_scalar"])),
		"next_latent_l2": _summary_tensor(_l2(base["next_latent"], other["next_latent"])),
	}


def _condition_stats(outputs):
	return {
		"latent_norm": _summary_tensor(torch.linalg.vector_norm(outputs["latent"], dim=-1)),
		"action_mean_norm": _summary_tensor(torch.linalg.vector_norm(outputs["action_mean"], dim=-1)),
		"q_scalar": _summary_tensor(outputs["q_scalar"]),
		"reward_scalar": _summary_tensor(outputs["reward_scalar"]),
		"next_latent_norm": _summary_tensor(torch.linalg.vector_norm(outputs["next_latent"], dim=-1)),
	}


def run(args):
	checkpoint_fp = _resolve(args.checkpoint)
	replay_fp = _resolve(args.replay)
	output_fp = _resolve(args.output)
	if not checkpoint_fp.exists():
		raise FileNotFoundError(f"Checkpoint not found: {checkpoint_fp}")
	if not replay_fp.exists():
		raise FileNotFoundError(f"Replay not found: {replay_fp}")
	device = torch.device(f"cuda:{int(args.gpu_id)}" if torch.cuda.is_available() and not args.cpu else "cpu")
	if device.type == "cuda":
		torch.cuda.set_device(device)
	torch.manual_seed(int(args.seed))

	cfg, compat = _load_config(args, checkpoint_fp)
	cfg.device_id = int(args.gpu_id) if device.type == "cuda" else 0
	model = WorldModel(cfg).to(device)
	model = _load_world_model(model, checkpoint_fp, cfg)
	model.eval()

	td, replay_metadata = _load_replay_tensordict(replay_fp)
	obs, action, reward, indices = _sample_replay_rows(td, args.batch_size, args.seed)
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
	obs = obs.to(device)
	action = action.to(device)

	conditions, condition_metadata = _condition_vectors(args, replay_fp)
	outputs = OrderedDict()
	for label, vec in conditions.items():
		outputs[label] = _forward_condition(model, cfg, obs, action, vec)
	base = outputs[str(args.base_label)]
	comparisons = OrderedDict()
	for label, item in outputs.items():
		if label == str(args.base_label):
			continue
		comparisons[label] = _compare_outputs(base, item)

	report = {
		"checkpoint": str(checkpoint_fp),
		"replay": str(replay_fp),
		"output": str(output_fp),
		"device": str(device),
		"seed": int(args.seed),
		"sample_size": int(obs.shape[0]),
		"sample_indices_first10": [int(x) for x in indices[:10].cpu().tolist()],
		"checkpoint_compat": compat,
		"replay_metadata": replay_metadata,
		"condition_metadata": condition_metadata,
		"conditions": {
			label: [float(x) for x in vec.tolist()]
			for label, vec in conditions.items()
		},
		"condition_stats": {
			label: _condition_stats(item)
			for label, item in outputs.items()
		},
		"comparisons_vs_base": comparisons,
	}
	output_fp.parent.mkdir(parents=True, exist_ok=True)
	with open(output_fp, "w", encoding="utf-8") as f:
		json.dump(report, f, indent=2, ensure_ascii=True)
		f.write("\n")

	print(f"Wrote sensitivity report: {output_fp}")
	print(f"Base label: {args.base_label} sample_size={int(obs.shape[0])} device={device}")
	for label, metrics in comparisons.items():
		print(
			f"{label:>18s} "
			f"action_l2={metrics['action_mean_l2']['mean']:.6g} "
			f"q_abs={metrics['q_abs']['mean']:.6g} "
			f"reward_abs={metrics['reward_abs']['mean']:.6g} "
			f"next_z_l2={metrics['next_latent_l2']['mean']:.6g}"
		)
	return report


def build_parser():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="TD-MPC2 checkpoint to inspect.")
	parser.add_argument("--replay", default=DEFAULT_REPLAY, help="Replay snapshot providing paired obs/action samples.")
	parser.add_argument("--anchor-replay", default=DEFAULT_ANCHOR_REPLAY, help="Optional replay whose unique task vector is used as anchor.")
	parser.add_argument("--config", default="configs/train/srsa_01125_imitation_relaxed.yaml", help="Training config used to build the model.")
	parser.add_argument("--output", default="logs/task_vec_sensitivity/00256_v2_offline_report.json", help="Output JSON report path.")
	parser.add_argument("--batch-size", type=int, default=512, help="Number of finite replay transitions to sample.")
	parser.add_argument("--seed", type=int, default=1)
	parser.add_argument("--gpu-id", type=int, default=0, help="Visible CUDA device index. Use with CUDA_VISIBLE_DEVICES when needed.")
	parser.add_argument("--cpu", action="store_true", help="Run on CPU even if CUDA is available.")
	parser.add_argument("--assembly-id", default="00256")
	parser.add_argument("--eval-task-id", type=int, default=2)
	parser.add_argument("--base-label", default="correct_from_replay")
	parser.add_argument(
		"--task-vec",
		action="append",
		default=[],
		help="Add a task vector condition as LABEL=v0,v1,v2,v3,v4,v5.",
	)
	parser.add_argument(
		"--condition-from-replay",
		action="append",
		default=[],
		help="Add a task vector condition from a replay snapshot as LABEL=/path/to/replay.pt.",
	)
	parser.add_argument("--include-zero", action=argparse.BooleanOptionalAction, default=True)
	parser.add_argument("--include-random", action=argparse.BooleanOptionalAction, default=True)
	parser.add_argument("--include-extreme", action=argparse.BooleanOptionalAction, default=True)
	return parser


def main():
	args = build_parser().parse_args()
	run(args)


if __name__ == "__main__":
	main()
