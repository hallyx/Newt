from __future__ import annotations

import json
import os
from pathlib import Path
from time import monotonic
from typing import Optional

os.environ["MUJOCO_GL"] = os.getenv("MUJOCO_GL", "egl")
os.environ["LAZY_LEGACY_OP"] = "0"
os.environ["TORCH_DISTRIBUTED_TIMEOUT"] = "1800"
os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

import hydra
import torch
from hydra.core.config_store import ConfigStore
from tensordict import TensorDict
from termcolor import colored

from common import set_seed
from common.logger import Logger
from common.world_model import WorldModel
from config import Config, parse_cfg
from offline_io import summarize_compact_dataset
from tdmpc2 import TDMPC2
from zmq_action_publisher import make_eval_zmq_observation_receiver, make_eval_zmq_publisher


torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

cs = ConfigStore.instance()
cs.store(name="config", node=Config)


def _checkpoint_state_dict(checkpoint_fp):
	obj = torch.load(checkpoint_fp, map_location="cpu", weights_only=False)
	return obj["model"] if isinstance(obj, dict) and "model" in obj else obj


def _get_state_tensor(state_dict, key: str):
	for candidate in (key, f"module.{key}"):
		if candidate in state_dict:
			return state_dict[candidate]
	return None


def _infer_checkpoint_io(checkpoint_fp):
	state_dict = _checkpoint_state_dict(checkpoint_fp)
	enc_weight = _get_state_tensor(state_dict, "_encoder.state.0.weight")
	if enc_weight is None:
		raise KeyError(f"Could not find `_encoder.state.0.weight` in checkpoint={checkpoint_fp}.")
	task_emb = _get_state_tensor(state_dict, "_task_emb.weight")
	task_vecs = _get_state_tensor(state_dict, "_task_vecs")
	task_encoder = _get_state_tensor(state_dict, "_task_encoder.type_encoder.weight")
	if task_emb is not None:
		task_dim = int(task_emb.shape[-1])
		task_conditioning = "id_embedding"
	elif task_vecs is not None or task_encoder is not None:
		task_dim = 64
		task_conditioning = "axial_params"
	else:
		task_dim = 0
		task_conditioning = "none"
	obs_dim = int(enc_weight.shape[1]) - task_dim
	if obs_dim <= 0:
		raise ValueError(
			f"Could not infer positive obs_dim from checkpoint={checkpoint_fp}: "
			f"encoder_in={int(enc_weight.shape[1])}, task_dim={task_dim}."
		)
	action_masks = _get_state_tensor(state_dict, "_action_masks")
	action_dim = int(action_masks.shape[-1]) if action_masks is not None else 6
	return {
		"obs_dim": obs_dim,
		"action_dim": action_dim,
		"task_dim": task_dim,
		"task_conditioning": task_conditioning,
	}


def _configure_real_hil_cfg(cfg):
	if cfg.world_size != 1:
		raise ValueError("Real HIL collection only supports a single process/GPU.")
	if cfg.num_envs != 1:
		print(colored("Forcing num_envs=1 for real HIL collection.", "yellow", attrs=["bold"]))
		cfg.num_envs = 1
	compat = _infer_checkpoint_io(cfg.checkpoint)
	if str(cfg.task_conditioning).lower() != compat["task_conditioning"]:
		raise ValueError(
			"Real HIL config does not match checkpoint task conditioning: "
			f"cfg={cfg.task_conditioning}, checkpoint={compat['task_conditioning']}."
		)
	cfg.eval_mode = "real"
	cfg.eval_real_mode = "closed_loop"
	cfg.eval_zmq_enabled = True
	cfg.obs = "state"
	cfg.obs_shape = {"state": (int(compat["obs_dim"]),)}
	cfg.action_dim = int(compat["action_dim"])
	cfg.action_dims = [cfg.action_dim] if not cfg.action_dims else [
		cfg.action_dim if int(dim) <= 0 else min(int(dim), cfg.action_dim)
		for dim in cfg.action_dims
	]
	max_steps = cfg.get("hil_collect_max_steps", None) or cfg.get("eval_real_steps", None)
	cfg.episode_length = int(max_steps or cfg.get("episode_length", None) or cfg.get("isaaclab_max_episode_steps", 75))
	cfg.episode_lengths = [cfg.episode_length for _ in cfg.episode_lengths]
	return compat


def _real_task_id(cfg) -> int:
	if cfg.get("eval_task_id", None) is not None:
		return int(cfg.eval_task_id)
	if cfg.get("srsa_task_template_id", None) is not None:
		return int(cfg.srsa_task_template_id)
	return 0


def _real_task_input(cfg, obs_receiver, message, device):
	if (
		bool(cfg.get("eval_real_use_msg_task_vec", True))
		and str(cfg.get("task_conditioning", "")).lower() == "axial_params"
	):
		task_vec = obs_receiver.task_vec_tensor(message, device=device)
		if task_vec is not None:
			return task_vec
	task_id = _real_task_id(cfg)
	if str(cfg.get("task_conditioning", "")).lower() == "axial_params":
		task_vectors = cfg.get("task_vectors", None) or []
		if len(task_vectors) == 1:
			return torch.tensor(task_vectors, dtype=torch.float32, device=device)
		if 0 <= task_id < len(task_vectors):
			return torch.tensor([task_vectors[task_id]], dtype=torch.float32, device=device)
	return torch.tensor([task_id], dtype=torch.long, device=device)


def _as_float_tensor(value, *, shape: tuple[int, ...], name: str) -> torch.Tensor:
	tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
	expected = 1
	for dim in shape:
		expected *= int(dim)
	if tensor.numel() != expected:
		raise ValueError(f"Expected `{name}` with {expected} values, got {tensor.numel()}.")
	return tensor.reshape(shape).contiguous()


def _message_scalar(message: dict, key: str, default: float = 0.0) -> float:
	value = message.get(key, None)
	if value is None:
		return float(default)
	if torch.is_tensor(value):
		value = value.detach().cpu().reshape(-1)[0].item()
	elif isinstance(value, (list, tuple)):
		value = value[0] if value else default
	return float(value)


def _message_success(message: dict, key: str) -> float:
	for candidate in (key, "success", "succeed", "is_success", "episode_success"):
		if candidate in message:
			return 1.0 if bool(message[candidate]) else 0.0
	return 0.0


def _action_key_candidates(cfg) -> list[str]:
	raw = str(cfg.get("hil_collect_action_keys", "") or "")
	return [item.strip() for item in raw.split(",") if item.strip()]


def _executed_action_from_message(message: dict, cfg, fallback: torch.Tensor) -> tuple[torch.Tensor, bool, Optional[str]]:
	for key in _action_key_candidates(cfg):
		if key not in message:
			continue
		action = _as_float_tensor(message[key], shape=tuple(fallback.shape), name=key)
		return action, True, key
	if bool(cfg.get("hil_collect_require_actual_action", False)):
		raise KeyError(
			"Robot observation message did not include an executed action. "
			f"Expected one of {tuple(_action_key_candidates(cfg))}."
		)
	return fallback.detach().cpu().to(torch.float32).reshape(tuple(fallback.shape)).contiguous(), False, None


def _intervened_from_message(message: dict, cfg, action_key: Optional[str]) -> bool:
	key = cfg.get("hil_collect_intervened_key", "intervened")
	if key in message:
		return bool(message[key])
	return action_key == "intervene_action"


def _empty_columns():
	return {
		"obs": [],
		"next_obs": [],
		"action": [],
		"policy_action": [],
		"intervened": [],
		"reward": [],
		"done": [],
		"terminated": [],
		"truncated": [],
		"episode": [],
		"step_id": [],
		"task": [],
		"episode_return_running": [],
		"episode_return_final": [],
		"episode_success_final": [],
		"episode_failure_final": [],
		"terminal_success": [],
		"terminal_failure": [],
		"success_episode": [],
	}


def _append_episode(columns, rows, *, episode_id: int, task_id: int, final_return: float, final_success: float):
	final_failure = 1.0 - final_success
	last_index = len(rows) - 1
	for index, row in enumerate(rows):
		is_terminal = index == last_index
		columns["obs"].append(row["obs"])
		columns["next_obs"].append(row["next_obs"])
		columns["action"].append(row["action"])
		columns["policy_action"].append(row["policy_action"])
		columns["intervened"].append(row["intervened"])
		columns["reward"].append(row["reward"])
		columns["done"].append(row["done"])
		columns["terminated"].append(row["terminated"])
		columns["truncated"].append(row["truncated"])
		columns["episode"].append(torch.tensor(episode_id, dtype=torch.int64))
		columns["step_id"].append(row["step_id"])
		columns["task"].append(torch.tensor(task_id, dtype=torch.int64))
		columns["episode_return_running"].append(row["episode_return_running"])
		columns["episode_return_final"].append(torch.tensor(final_return, dtype=torch.float32))
		columns["episode_success_final"].append(torch.tensor(final_success, dtype=torch.float32))
		columns["episode_failure_final"].append(torch.tensor(final_failure, dtype=torch.float32))
		columns["terminal_success"].append(torch.tensor(bool(is_terminal and final_success > 0.5), dtype=torch.bool))
		columns["terminal_failure"].append(torch.tensor(bool(is_terminal and final_success <= 0.5), dtype=torch.bool))
		columns["success_episode"].append(torch.tensor(final_success, dtype=torch.float32))


def _stack_columns(columns):
	if len(columns["obs"]) == 0:
		raise RuntimeError("No completed real HIL episodes were collected.")
	data = {key: torch.stack(values, dim=0).contiguous() for key, values in columns.items()}
	return TensorDict(data, batch_size=(data["obs"].shape[0],))


def _write_json(path: Path, obj):
	path.parent.mkdir(parents=True, exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		json.dump(obj, f, indent=2, ensure_ascii=True)


def _resolve_output_fp(cfg) -> Path:
	if cfg.get("hil_collect_output_fp", None):
		return Path(cfg.hil_collect_output_fp).expanduser().resolve()
	return Path(cfg.work_dir) / "data" / "real_hil_rollouts.pt"


def _task_vec_6(cfg):
	if str(cfg.get("task_conditioning", "")).lower() != "axial_params":
		return None
	task_id = _real_task_id(cfg)
	task_vectors = cfg.get("task_vectors", None) or []
	if len(task_vectors) == 1:
		return [float(x) for x in task_vectors[0]]
	if 0 <= task_id < len(task_vectors):
		return [float(x) for x in task_vectors[task_id]]
	return None


def _make_agent(cfg):
	model = WorldModel(cfg).to(f"cuda:{cfg.device_id}")
	agent = TDMPC2(model, cfg)
	agent.load(cfg.checkpoint)
	agent.eval()
	agent.model.eval()
	return agent


@torch.no_grad()
def collect_real_hil(agent: TDMPC2, cfg):
	device = torch.device(f"cuda:{cfg.device_id}")
	obs_dim = int(cfg.obs_shape["state"][0])
	max_steps = int(cfg.episode_length)
	target_episodes = int(cfg.get("hil_collect_episodes", 10))
	use_mpc = cfg.mpc if cfg.get("hil_collect_mpc", None) is None else bool(cfg.hil_collect_mpc)
	policy_task_id = _real_task_id(cfg)
	dataset_task_id = 0
	columns = _empty_columns()
	episode_returns = []
	episode_successes = []
	intervention_steps = 0
	intervention_segments = 0
	total_steps = 0
	start_time = monotonic()

	print(colored(
		"Starting real HIL collection: "
		f"episodes={target_episodes}, obs_dim={obs_dim}, action_dim={cfg.action_dim}, "
		f"max_steps={max_steps}, mpc={use_mpc}, "
		f"obs_endpoint={cfg.eval_real_obs_server}, action_endpoint={cfg.eval_zmq_server}.",
		"cyan",
		attrs=["bold"],
	))
	print(colored(
		"Robot side should execute either Newt policy action or human override, then include "
		"`executed_action`/`actual_action`/`intervene_action`, `reward`, `done`, and optional `success` "
		"in the next observation message.",
		"cyan",
		attrs=["bold"],
	))

	with make_eval_zmq_publisher(cfg) as action_publisher, make_eval_zmq_observation_receiver(cfg) as obs_receiver:
		message = obs_receiver.recv()
		for episode_id in range(target_episodes):
			while obs_receiver.is_done(message):
				message = obs_receiver.recv()
			rows = []
			running_return = 0.0
			already_intervened = False
			final_success = 0.0
			for step_id in range(max_steps):
				obs = obs_receiver.obs_tensor(message, obs_dim=obs_dim, device=device)
				model_tasks = _real_task_input(cfg, obs_receiver, message, device)
				t0 = torch.tensor([step_id == 0], dtype=torch.bool, device=device)
				torch.compiler.cudagraph_mark_step_begin()
				policy_action, _ = agent(
					obs,
					t0=t0,
					step=1 if use_mpc else 0,
					eval_mode=True,
					task=model_tasks,
					mpc=use_mpc,
				)
				action_publisher.send_action(
					policy_action,
					step=total_steps,
					episode_step=step_id,
					task_id=policy_task_id,
				)
				next_message = obs_receiver.recv()
				next_obs = obs_receiver.obs_tensor(next_message, obs_dim=obs_dim, device=device)

				policy_action_cpu = policy_action.detach().cpu().reshape(-1).to(torch.float32)
				executed_action, action_key_found, action_key = _executed_action_from_message(
					next_message,
					cfg,
					policy_action_cpu,
				)
				intervened = _intervened_from_message(next_message, cfg, action_key)
				if intervened:
					intervention_steps += 1
					if not already_intervened:
						intervention_segments += 1
				already_intervened = intervened

				reward = _message_scalar(next_message, cfg.get("hil_collect_reward_key", "reward"), default=0.0)
				running_return += reward
				done_by_robot = obs_receiver.is_done(next_message)
				done_by_limit = step_id + 1 >= max_steps
				done = bool(done_by_robot or done_by_limit)
				terminated = bool(next_message.get("terminated", False))
				truncated = bool(next_message.get("truncated", done_by_limit or (done and not terminated)))
				if done:
					final_success = _message_success(next_message, cfg.get("hil_collect_success_key", "success"))

				rows.append({
					"obs": obs.detach().cpu().reshape(-1).to(torch.float32),
					"next_obs": next_obs.detach().cpu().reshape(-1).to(torch.float32),
					"action": executed_action,
					"policy_action": policy_action_cpu,
					"intervened": torch.tensor(intervened, dtype=torch.bool),
					"reward": torch.tensor(reward, dtype=torch.float32),
					"done": torch.tensor(done, dtype=torch.bool),
					"terminated": torch.tensor(terminated, dtype=torch.bool),
					"truncated": torch.tensor(truncated, dtype=torch.bool),
					"step_id": torch.tensor(step_id, dtype=torch.int32),
					"episode_return_running": torch.tensor(running_return, dtype=torch.float32),
				})
				total_steps += 1
				message = next_message
				if done:
					action_publisher.send_done(step=total_steps, episode_step=step_id + 1, task_id=policy_task_id)
					break
				if not action_key_found and bool(cfg.get("hil_collect_require_actual_action", False)):
					raise RuntimeError("Unreachable: missing actual action should have raised earlier.")

			_append_episode(
				columns,
				rows,
				episode_id=episode_id,
				task_id=dataset_task_id,
				final_return=running_return,
				final_success=final_success,
			)
			episode_returns.append(running_return)
			episode_successes.append(final_success)
			elapsed = monotonic() - start_time
			print(colored(
				f"collected episode={episode_id + 1}/{target_episodes} "
				f"length={len(rows)} return={running_return:.4g} success={final_success:.0f} "
				f"intervention_steps={intervention_steps} elapsed={elapsed:.1f}s",
				"cyan",
				attrs=["bold"],
			), flush=True)
			if episode_id + 1 < target_episodes:
				message = obs_receiver.recv()

	dataset = _stack_columns(columns)
	summary = summarize_compact_dataset(dataset)
	summary.update({
		"intervention_steps": int(intervention_steps),
		"intervention_segments": int(intervention_segments),
		"intervention_step_fraction": float(intervention_steps / max(1, total_steps)),
	})
	metadata = {
		"checkpoint": str(Path(cfg.checkpoint).expanduser().resolve()),
		"output": str(_resolve_output_fp(cfg)),
		"task_id": int(dataset_task_id),
		"policy_eval_task_id": int(policy_task_id),
		"task_vec_6": _task_vec_6(cfg),
		"num_episodes": target_episodes,
		"num_transitions": int(summary["num_transitions"]),
		"episode_return_mean": float(sum(episode_returns) / max(1, len(episode_returns))),
		"episode_success_mean": float(sum(episode_successes) / max(1, len(episode_successes))),
		"mpc": bool(use_mpc),
		"obs_endpoint": cfg.eval_real_obs_server,
		"action_endpoint": cfg.eval_zmq_server,
		"summary": summary,
	}
	return dataset, metadata


@hydra.main(version_base=None, config_name="config")
def launch(cfg: Config):
	if not torch.cuda.is_available():
		raise RuntimeError("Real HIL collection requires CUDA because Newt inference runs on GPU.")
	cfg = parse_cfg(cfg)
	cfg.rank = 0
	cfg.world_size = 1
	cfg.num_envs = 1
	cfg.device_id = cfg.gpu_id
	set_seed(cfg.seed)
	torch.cuda.set_device(cfg.device_id)
	if not cfg.checkpoint:
		raise ValueError("`checkpoint` must be provided for real HIL collection.")
	if not os.path.exists(cfg.checkpoint):
		raise FileNotFoundError(f"Checkpoint file not found: {cfg.checkpoint}")

	output_fp = _resolve_output_fp(cfg)
	if output_fp.exists() and not cfg.get("hil_collect_overwrite", False):
		raise FileExistsError(f"Output already exists: {output_fp}. Use hil_collect_overwrite=true to replace it.")
	output_fp.parent.mkdir(parents=True, exist_ok=True)

	compat = _configure_real_hil_cfg(cfg)
	print(colored("Work dir:", "yellow", attrs=["bold"]), cfg.work_dir)
	print(colored(
		f"Checkpoint I/O: obs_dim={compat['obs_dim']} action_dim={compat['action_dim']} "
		f"task_dim={compat['task_dim']} task_conditioning={compat['task_conditioning']}",
		"blue",
		attrs=["bold"],
	))
	logger = Logger(cfg)
	agent = _make_agent(cfg)
	dataset, metadata = collect_real_hil(agent, cfg)
	torch.save(dataset, output_fp)
	metadata["output"] = str(output_fp)
	metadata_fp = output_fp.with_suffix(output_fp.suffix + ".json")
	_write_json(metadata_fp, metadata)

	manifest_fp = cfg.get("hil_collect_manifest_fp", None)
	if manifest_fp:
		manifest_fp = Path(manifest_fp).expanduser().resolve()
		entry = {
			"task_id": int(metadata["task_id"]),
			"task_name": f"{cfg.task}-real-hil",
			"assembly_id": cfg.get("assembly_id", None),
			"source_fp": str(output_fp),
			"action_dim": int(cfg.action_dim),
			"max_episode_steps": int(cfg.episode_length),
			"task_vec_6": metadata.get("task_vec_6"),
			"num_episodes": int(metadata["num_episodes"]),
			"num_transitions": int(metadata["num_transitions"]),
			"success_count": int(round(metadata["episode_success_mean"] * metadata["num_episodes"])),
			"failure_count": int(metadata["num_episodes"] - round(metadata["episode_success_mean"] * metadata["num_episodes"])),
			"success_rate": float(metadata["episode_success_mean"]),
		}
		_write_json(manifest_fp, {"tasks": [entry]})
		print(colored(f"Saved offline manifest: {manifest_fp}", "green", attrs=["bold"]))

	logger.finish()
	print(colored(
		f"Saved real HIL dataset: {output_fp} "
		f"transitions={metadata['num_transitions']:,} "
		f"intervention_fraction={metadata['summary']['intervention_step_fraction']:.4f}",
		"green",
		attrs=["bold"],
	))
	print(colored(f"Saved metadata: {metadata_fp}", "green", attrs=["bold"]))


if __name__ == "__main__":
	launch()
