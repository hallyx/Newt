from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

os.environ["MUJOCO_GL"] = os.getenv("MUJOCO_GL", "egl")
os.environ["LAZY_LEGACY_OP"] = "0"

import hydra
import torch
from hydra.core.config_store import ConfigStore
from termcolor import colored

from config import Config, parse_cfg
from eval import _real_action_for_command_frame
from zmq_action_publisher import make_eval_zmq_observation_receiver, make_eval_zmq_publisher


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


def _parse_vector3(value, *, name: str) -> torch.Tensor:
	if value is None:
		raise ValueError(f"`{name}` must be provided.")
	if isinstance(value, str):
		value = value.strip().strip("[]()")
		value = [item.strip() for item in value.split(",") if item.strip()]
	tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
	if tensor.numel() != 3:
		raise ValueError(f"`{name}` must contain 3 values, got {tensor.numel()}.")
	return tensor.contiguous()


def _message_tcp_pos_base(message: dict) -> Optional[torch.Tensor]:
	ee = message.get("end_effector", {}) if isinstance(message.get("end_effector", {}), dict) else {}
	pos = message.get("tcp_pos", message.get("position", ee.get("position", None)))
	if pos is not None:
		return _parse_vector3(pos, name="tcp_pos")
	matrix = message.get("O_T_EE", ee.get("O_T_EE", None))
	if matrix is None:
		return None
	matrix = torch.as_tensor(matrix, dtype=torch.float32).reshape(-1)
	if matrix.numel() != 16:
		return None
	return torch.stack([matrix[12], matrix[13], matrix[14]]).contiguous()


def _target_pos_base(cfg) -> torch.Tensor:
	value = cfg.get("real_interp_target_pos", None)
	if value is None:
		value = cfg.get("eval_real_socket_pos", None)
	return _parse_vector3(value, name="real_interp_target_pos/eval_real_socket_pos")


def interpolation_delta_to_target(
	current_pos_base: torch.Tensor,
	target_pos_base: torch.Tensor,
	step_size: float,
) -> tuple[torch.Tensor, float]:
	"""Return one base-frame delta step from current TCP position toward target."""
	error = target_pos_base - current_pos_base
	distance = float(torch.linalg.norm(error).item())
	if distance <= 1.0e-12:
		return torch.zeros(3, dtype=torch.float32), distance
	step = min(float(step_size), distance)
	return (error / distance * step).to(torch.float32).contiguous(), distance


def _to_command_delta(
	base_delta: torch.Tensor,
	obs: torch.Tensor,
	cfg,
	obs_receiver,
) -> torch.Tensor:
	action_order = cfg.get("eval_zmq_action_order", "dx,dy,dz,droll,dpitch,dyaw")
	if isinstance(action_order, str):
		action_order = [item.strip() for item in action_order.split(",") if item.strip()]
	action_dim = len(action_order)
	action = torch.zeros((1, action_dim), dtype=obs.dtype, device=obs.device)
	action[0, :3] = base_delta.to(device=obs.device, dtype=obs.dtype)
	convert_cfg = {
		"eval_zmq_action_frame": "base",
		"eval_zmq_command_frame": cfg.get("eval_zmq_command_frame", "base"),
	}
	return _real_action_for_command_frame(action, obs, convert_cfg, obs_receiver).detach().cpu().reshape(-1)


def _open_log(cfg):
	fp = cfg.get("real_interp_log_fp", None)
	if not fp:
		return None
	path = Path(hydra.utils.to_absolute_path(str(fp))).expanduser()
	path.parent.mkdir(parents=True, exist_ok=True)
	return open(path, "w", encoding="utf-8")


def _write_log(log_f, row: dict):
	if log_f is None:
		return
	log_f.write(json.dumps(row, ensure_ascii=True) + "\n")
	log_f.flush()


@hydra.main(version_base=None, config_name="config")
def launch(cfg: Config):
	cfg = parse_cfg(cfg)
	cfg.rank = 0
	cfg.world_size = 1
	cfg.num_envs = 1
	cfg.eval_mode = "real"
	cfg.eval_real_mode = "closed_loop"
	cfg.eval_zmq_enabled = True
	cfg.eval_zmq_action_frame = "base"
	if cfg.get("eval_zmq_command_frame", None) is None:
		cfg.eval_zmq_command_frame = "base"

	target_pos = _target_pos_base(cfg)
	step_size = float(cfg.get("real_interp_step_size", 0.0001))
	stop_distance = float(cfg.get("real_interp_stop_distance", 0.001))
	steps = int(cfg.get("real_interp_steps", 50))
	dry_run = bool(cfg.get("real_interp_dry_run", True))
	obs_dim = int(cfg.get("real_interp_obs_dim", 17))
	if step_size <= 0.0:
		raise ValueError("real_interp_step_size must be positive.")
	if steps <= 0:
		raise ValueError("real_interp_steps must be positive.")

	print(colored(
		"Real interpolation probe: "
		f"target_pos_base={target_pos.tolist()} step_size={step_size:g} "
		f"stop_distance={stop_distance:g} command_frame={cfg.eval_zmq_command_frame} "
		f"dry_run={dry_run}",
		"cyan",
		attrs=["bold"],
	))
	if dry_run:
		print(colored("Dry run only: no ZMQ command will be sent. Set real_interp_dry_run=false to move.", "yellow"))

	log_f = _open_log(cfg)
	try:
		with make_eval_zmq_observation_receiver(cfg) as obs_receiver, make_eval_zmq_publisher(cfg) as action_publisher:
			message = obs_receiver.recv()
			for step in range(steps):
				current_pos = _message_tcp_pos_base(message)
				if current_pos is None:
					raise KeyError("Observation has no TCP base position.")
				base_delta, distance = interpolation_delta_to_target(current_pos, target_pos, step_size)
				obs = obs_receiver.obs_tensor(message, obs_dim=obs_dim, device=torch.device("cpu"))
				command_delta = _to_command_delta(base_delta, obs, cfg, obs_receiver)
				row = {
					"step": step,
					"state_seq": message.get("seq", None),
					"current_pos_base": current_pos.tolist(),
					"target_pos_base": target_pos.tolist(),
					"distance_before": distance,
					"base_delta": base_delta.tolist(),
					"command_delta": command_delta.tolist(),
					"command_frame": cfg.eval_zmq_command_frame,
					"dry_run": dry_run,
				}
				print(
					f"step={step:03d} dist={distance:.6f} "
					f"base_delta={[round(x, 7) for x in base_delta.tolist()]} "
					f"cmd_delta={[round(x, 7) for x in command_delta[:3].tolist()]} frame={cfg.eval_zmq_command_frame}"
				)
				if distance <= stop_distance:
					row["stopped"] = "target_reached"
					_write_log(log_f, row)
					break
				if not dry_run:
					action_publisher.send_action(
						command_delta,
						step=step,
						episode_step=step,
						task_id=None,
						state_seq=message.get("seq", None),
						state_timestamp=message.get("timestamp", None),
						preprocessed=True,
						raw_action=base_delta.tolist(),
					)
				next_message = obs_receiver.recv()
				next_pos = _message_tcp_pos_base(next_message)
				if next_pos is not None:
					next_dist = float(torch.linalg.norm(next_pos - target_pos).item())
					row["distance_after"] = next_dist
					row["distance_delta"] = next_dist - distance
					print(f"        after_dist={next_dist:.6f} delta={next_dist - distance:+.6f}")
				_write_log(log_f, row)
				message = next_message
			if not dry_run:
				action_publisher.send_action(
					[0.0] * len(command_delta),
					step=steps,
					episode_step=steps,
					preprocessed=True,
					raw_action=[0.0, 0.0, 0.0],
				)
	finally:
		if log_f is not None:
			log_f.close()


if __name__ == "__main__":
	launch()
