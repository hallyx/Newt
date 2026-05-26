import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ["TORCH_DISTRIBUTED_TIMEOUT"] = "1800"
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
os.environ['TORCH_LOGS'] = "+recompiles"
import warnings
warnings.filterwarnings('ignore')

import json
import csv
import math
from collections import defaultdict
from pathlib import Path
from time import monotonic

import torch
import hydra
from hydra.core.config_store import ConfigStore
from termcolor import colored

from common import barrier, set_seed
from common.logger import Logger
from common.world_model import WorldModel
from config import Config, apply_eval_task_template, parse_cfg
from envs import make_env
from tdmpc2 import TDMPC2
from trainer import Trainer
from zmq_action_publisher import make_eval_zmq_observation_receiver, make_eval_zmq_publisher

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

cs = ConfigStore.instance()
cs.store(name="config", node=Config)


def setup(rank, world_size, port):
	os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
	os.environ["MASTER_PORT"] = port
	torch.distributed.init_process_group(
		backend="nccl",
		rank=rank,
		world_size=world_size
	)
	return port


SUCCESS_DIAGNOSTIC_KEYS = (
	"official_success_latched",
	"official_success_terminal",
	"process_success_terminal",
	"relaxed_process_success_terminal",
	"relaxed_success_stable",
	"relaxed_success_episode",
	"strict_success_stable",
	"strict_success_episode",
	"official_success",
	"current_official_success",
	"process_success",
	"relaxed_process_success",
	"episode_relaxed_process_success",
	"relaxed_terminal_process_success",
	"episode_process_success",
	"terminal_process_success",
	"dual_success",
	"depth_fraction",
	"lateral_error",
	"angle_error",
	"orientation_error",
	"yaw_error",
	"keypoint_error",
	"jam",
)


def empty_metrics():
	metrics = {'reward': [], 'length': [], 'success': [], 'score': []}
	for key in SUCCESS_DIAGNOSTIC_KEYS:
		metrics[key] = []
	return metrics


class EvalTraceRecorder:
	"""
	Write a small JSONL trace of inference inputs/outputs for sim2real debugging.
	"""

	def __init__(self, cfg, *, phase: str):
		self.cfg = cfg
		self.phase = phase
		self.enabled = bool(cfg.get('eval_trace_enabled', False)) and int(cfg.get('rank', 0)) == 0
		self.max_steps = max(0, int(cfg.get('eval_trace_steps', 16) or 0))
		self.count = 0
		self.env_index = cfg.get('eval_trace_env_index', None)
		if self.env_index is None:
			self.env_index = cfg.get('eval_zmq_env_index', 0)
		self.env_index = int(self.env_index)
		self.include_next_obs = bool(cfg.get('eval_trace_include_next_obs', True))
		self.include_action_info = bool(cfg.get('eval_trace_include_action_info', True))
		self.include_raw_msg = bool(cfg.get('eval_trace_include_raw_msg', False))
		self._fh = None
		self.fp = None
		if not self.enabled or self.max_steps <= 0:
			self.enabled = False
			return
		raw_fp = cfg.get('eval_trace_fp', None)
		if raw_fp:
			self.fp = Path(raw_fp).expanduser()
			if not self.fp.is_absolute():
				self.fp = Path(cfg.work_dir) / self.fp
		else:
			self.fp = Path(cfg.work_dir) / "data" / f"{phase}_trace.jsonl"
		self.fp.parent.mkdir(parents=True, exist_ok=True)
		self._fh = open(self.fp, "w", encoding="utf-8")
		self._write({
			"type": "metadata",
			"phase": self.phase,
			"env_index": self.env_index,
			"max_steps": self.max_steps,
			"checkpoint": cfg.get('checkpoint', None),
			"eval_mode": cfg.get('eval_mode', None),
			"eval_real_mode": cfg.get('eval_real_mode', None),
			"obs_shape": cfg.get('obs_shape', None),
			"action_dim": cfg.get('action_dim', None),
			"task_conditioning": cfg.get('task_conditioning', None),
			"eval_zmq_action_scale": cfg.get('eval_zmq_action_scale', None),
			"eval_zmq_action_frame": cfg.get('eval_zmq_action_frame', None),
			"eval_zmq_command_frame": cfg.get('eval_zmq_command_frame', None) or cfg.get('eval_zmq_action_frame', None),
			"eval_zmq_action_order": cfg.get('eval_zmq_action_order', None),
		})
		print(colored(f"Recording eval trace to {self.fp}", "cyan", attrs=["bold"]))

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc, tb):
		self.close()

	def close(self):
		if self._fh is not None:
			self._fh.close()
			self._fh = None

	def _write(self, record):
		if self._fh is None:
			return
		self._fh.write(json.dumps(self._json_safe(record), ensure_ascii=True) + "\n")
		self._fh.flush()

	def _select_tensor(self, value):
		value = value.detach().cpu()
		if value.ndim >= 1 and value.shape[0] > self.env_index:
			value = value[self.env_index]
		if value.numel() == 1:
			return value.item()
		return value.tolist()

	def _json_safe(self, value):
		if torch.is_tensor(value):
			return self._select_tensor(value)
		if hasattr(value, "keys") and hasattr(value, "get"):
			return {str(key): self._json_safe(value.get(key)) for key in value.keys()}
		if isinstance(value, dict):
			return {str(key): self._json_safe(item) for key, item in value.items()}
		if isinstance(value, (list, tuple)):
			return [self._json_safe(item) for item in value]
		if isinstance(value, Path):
			return str(value)
		return value

	def record(
		self,
		*,
		step,
		episode_step,
		obs,
		action,
		task=None,
		action_info=None,
		sent_action=None,
		next_obs=None,
		reward=None,
		done=None,
		info=None,
		source_message=None,
	):
		if not self.enabled or self.count >= self.max_steps:
			return
		record = {
			"type": "inference_step",
			"trace_index": self.count,
			"phase": self.phase,
			"step": int(step),
			"episode_step": None if episode_step is None else int(episode_step),
			"env_index": self.env_index,
			"obs": obs,
			"action": action,
		}
		if task is not None:
			record["task"] = task
		if self.include_action_info and action_info is not None:
			record["action_info"] = action_info
		if sent_action is not None:
			record["sent_action"] = sent_action
		if self.include_next_obs and next_obs is not None:
			record["next_obs"] = next_obs
		if reward is not None:
			record["reward"] = reward
		if done is not None:
			record["done"] = done
		if isinstance(info, dict):
			for key in ("success", *SUCCESS_DIAGNOSTIC_KEYS):
				if key in info:
					record[key] = info[key]
		if source_message is not None:
			record["source_seq"] = source_message.get("seq", None)
			record["source_timestamp"] = source_message.get("timestamp", None)
			if self.include_raw_msg:
				record["source_message"] = source_message
		self._write(record)
		self.count += 1


def _real_eval_mode(cfg) -> str:
	return str(cfg.get('eval_real_mode', 'stream')).strip().lower().replace('-', '_')


def _is_real_closed_loop(cfg) -> bool:
	return cfg.get('eval_mode', 'sim') == 'real' and _real_eval_mode(cfg) in {
		'closed_loop',
		'robot_closed_loop',
		'obs_closed_loop',
	}


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


def _configure_real_closed_loop_cfg(cfg):
	if cfg.world_size != 1:
		raise ValueError("`eval_real_mode=closed_loop` only supports a single process/GPU.")
	if cfg.num_envs != 1:
		print(colored("Forcing num_envs=1 for real closed-loop eval.", "yellow", attrs=["bold"]))
		cfg.num_envs = 1
	compat = _infer_checkpoint_io(cfg.checkpoint)
	if str(cfg.task_conditioning).lower() != compat["task_conditioning"]:
		raise ValueError(
			"Real closed-loop config does not match checkpoint task conditioning: "
			f"cfg={cfg.task_conditioning}, checkpoint={compat['task_conditioning']}."
		)
	cfg.obs = 'state'
	cfg.obs_shape = {'state': (int(compat["obs_dim"]),)}
	cfg.action_dim = int(compat["action_dim"])
	if not cfg.action_dims:
		cfg.action_dims = [cfg.action_dim]
	else:
		cfg.action_dims = [
			cfg.action_dim if int(dim) <= 0 else min(int(dim), cfg.action_dim)
			for dim in cfg.action_dims
		]
	cfg.episode_length = int(
		cfg.get('eval_real_steps', None) or
		cfg.get('episode_length', None) or
		cfg.get('isaaclab_max_episode_steps', 75)
	)
	cfg.episode_lengths = [cfg.episode_length for _ in cfg.episode_lengths]
	return compat


def _real_task_id(cfg) -> int:
	if cfg.get('eval_task_id', None) is not None:
		return int(cfg.eval_task_id)
	if cfg.get('srsa_task_template_id', None) is not None:
		return int(cfg.srsa_task_template_id)
	return 0


def _real_task_input(cfg, obs_receiver, message, device):
	if (
		bool(cfg.get('eval_real_use_msg_task_vec', True)) and
		str(cfg.get('task_conditioning', '')).lower() == 'axial_params'
	):
		task_vec = obs_receiver.task_vec_tensor(message, device=device)
		if task_vec is not None:
			return task_vec
	task_id = _real_task_id(cfg)
	if str(cfg.get('task_conditioning', '')).lower() == 'axial_params':
		task_vectors = cfg.get('task_vectors', None) or []
		if len(task_vectors) == 1:
			return torch.tensor(task_vectors, dtype=torch.float32, device=device)
		if 0 <= task_id < len(task_vectors):
			return torch.tensor([task_vectors[task_id]], dtype=torch.float32, device=device)
	return torch.tensor([task_id], dtype=torch.long, device=device)


def _to_jsonable(value):
	if value is None:
		return None
	if torch.is_tensor(value):
		value = value.detach().cpu()
		if value.ndim == 0:
			return value.item()
		return value.tolist()
	if isinstance(value, dict):
		return {str(k): _to_jsonable(v) for k, v in value.items()}
	if isinstance(value, (list, tuple)):
		return [_to_jsonable(v) for v in value]
	return value


def _message_tcp_pos_base(message: dict):
	ee = message.get("end_effector", {}) if isinstance(message.get("end_effector", {}), dict) else {}
	pos = message.get("tcp_pos", message.get("position", ee.get("position", None)))
	if pos is not None:
		return _to_jsonable(pos)
	matrix = message.get("O_T_EE", ee.get("O_T_EE", None))
	if matrix is None:
		return None
	matrix = list(matrix)
	if len(matrix) != 16:
		return None
	return [float(matrix[12]), float(matrix[13]), float(matrix[14])]


def _make_real_debug_log(cfg):
	if not bool(cfg.get("eval_real_debug_log", True)):
		return None, None
	fp = cfg.get("eval_real_debug_log_fp", None)
	if fp is None:
		fp = Path(cfg.work_dir) / "real_closed_loop_debug.jsonl"
	else:
		fp = Path(hydra.utils.to_absolute_path(str(fp))).expanduser()
	fp.parent.mkdir(parents=True, exist_ok=True)
	return fp, open(fp, "a", encoding="utf-8")


def _quat_wxyz_to_matrix(quat: torch.Tensor) -> torch.Tensor:
	quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
	w, x, y, z = quat.unbind(dim=-1)
	row0 = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], dim=-1)
	row1 = torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], dim=-1)
	row2 = torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], dim=-1)
	return torch.stack([row0, row1, row2], dim=-2)


def _canonical_frame_name(frame: str) -> str:
	frame = str(frame).strip().lower()
	if frame in {"ee", "tcp", "end_effector"}:
		return "tcp"
	if frame in {"base", "world", "robot", "robot_base", "global"}:
		return "base"
	if frame in {"socket", "target", "hole"}:
		return "socket"
	return frame


def _socket_rot_base_to_socket(obs: torch.Tensor, obs_receiver=None) -> torch.Tensor:
	quat = getattr(obs_receiver, "_socket_quat", None)
	if quat is None:
		return torch.eye(3, dtype=obs.dtype, device=obs.device)
	quat = torch.as_tensor(quat, dtype=obs.dtype, device=obs.device)
	rot_socket_to_base = _quat_wxyz_to_matrix(quat.reshape(1, 4))[0]
	return rot_socket_to_base.transpose(-1, -2)


def _transform_action_vector(action: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
	if action.ndim > 1 and matrix.ndim == 2:
		matrix = matrix.expand(action.shape[:-1] + matrix.shape)
	return torch.einsum("...ij,...j->...i", matrix, action)


def _real_action_for_command_frame(action: torch.Tensor, obs: torch.Tensor, cfg, obs_receiver=None) -> torch.Tensor:
	policy_frame = _canonical_frame_name(cfg.get("eval_zmq_action_frame", "socket"))
	command_frame = _canonical_frame_name(cfg.get("eval_zmq_command_frame", policy_frame))
	if policy_frame == command_frame:
		return action
	if policy_frame not in {"socket", "tcp", "base"} or command_frame not in {"socket", "tcp", "base"}:
		raise ValueError(
			f"Unsupported real action frame conversion: {policy_frame!r} -> {command_frame!r}. "
			"Supported frames are socket, tcp/ee/end_effector, and base/world."
		)
	if obs.shape[-1] < 7:
		raise ValueError("Action frame conversion needs obs_tcp_quat_socket_wxyz in obs[3:7].")
	rot_tcp_to_socket = _quat_wxyz_to_matrix(obs[..., 3:7])
	rot_socket_to_tcp = rot_tcp_to_socket.transpose(-1, -2)
	rot_base_to_socket = _socket_rot_base_to_socket(obs, obs_receiver)
	rot_socket_to_base = rot_base_to_socket.transpose(-1, -2)

	command = action.clone()
	for start, stop in ((0, 3), (3, 6)):
		if action.shape[-1] <= start:
			continue
		part = action[..., start:stop]
		if policy_frame == "socket":
			part_socket = part
		elif policy_frame == "tcp":
			part_socket = _transform_action_vector(part, rot_tcp_to_socket)
		else:
			part_socket = _transform_action_vector(part, rot_base_to_socket)

		if command_frame == "socket":
			command[..., start:stop] = part_socket
		elif command_frame == "tcp":
			command[..., start:stop] = _transform_action_vector(part_socket, rot_socket_to_tcp)
		else:
			command[..., start:stop] = _transform_action_vector(part_socket, rot_socket_to_base)
	return command


def _shape_real_policy_action(action: torch.Tensor, step_idx: int, cfg) -> torch.Tensor:
	scale = float(cfg.get("eval_zmq_action_scale", 1.0))
	shaped = action * scale
	warmup_steps = max(int(cfg.get("eval_zmq_warmup_steps", 0)), 0)
	if warmup_steps > 0:
		warmup_scale = min(max(int(step_idx) + 1, 0), warmup_steps) / float(warmup_steps)
		shaped = shaped * warmup_scale
	max_trans = cfg.get("eval_zmq_max_trans_delta", None)
	if max_trans is not None:
		shaped[..., :3] = shaped[..., :3].clamp(-abs(float(max_trans)), abs(float(max_trans)))
	max_rot = cfg.get("eval_zmq_max_rot_delta", None)
	if max_rot is not None:
		shaped[..., 3:6] = shaped[..., 3:6].clamp(-abs(float(max_rot)), abs(float(max_rot)))
	return shaped


def _command_delta_to_socket(delta, obs_cpu: torch.Tensor, command_frame: str, obs_receiver=None):
	if delta is None or len(delta) < 3:
		return None
	delta_tensor = torch.as_tensor(delta, dtype=obs_cpu.dtype).reshape(-1)
	frame = _canonical_frame_name(command_frame)
	if frame == "socket":
		return delta_tensor[:3].tolist()
	if frame == "tcp" and obs_cpu.numel() >= 7:
		rot_tcp_to_socket = _quat_wxyz_to_matrix(obs_cpu[3:7].reshape(1, 4))[0]
		return (rot_tcp_to_socket @ delta_tensor[:3]).tolist()
	if frame == "base":
		rot_base_to_socket = _socket_rot_base_to_socket(obs_cpu.reshape(1, -1), obs_receiver)
		return (rot_base_to_socket @ delta_tensor[:3]).tolist()
	return None


def _write_real_debug_row(
	handle,
	*,
	step_idx: int,
	elapsed_s: float,
	task_id: int,
	message: dict,
	obs: torch.Tensor,
	action: torch.Tensor,
	send_message: dict,
	obs_receiver,
	info,
):
	if handle is None:
		return
	obs_cpu = obs.detach().cpu().reshape(-1)
	action_cpu = action.detach().cpu().reshape(-1)
	sent_delta = send_message.get("delta") if isinstance(send_message, dict) else None
	command_frame = send_message.get("action_frame") if isinstance(send_message, dict) else None
	sent_delta_socket = _command_delta_to_socket(sent_delta, obs_cpu, command_frame, obs_receiver)
	next_pos_socket = None
	if sent_delta_socket is not None and obs_cpu.numel() >= 3:
		next_pos_socket = (obs_cpu[:3] + torch.as_tensor(sent_delta_socket, dtype=obs_cpu.dtype)).tolist()
	pi_std = None
	if info is not None and info.get("pi_std", None) is not None:
		pi_std = float(torch.as_tensor(info["pi_std"]).detach().cpu().reshape(-1)[0].item())
	row = {
		"step": int(step_idx),
		"elapsed_s": float(elapsed_s),
		"task_id": int(task_id),
		"obs_seq": message.get("seq", None),
		"obs_timestamp": message.get("timestamp", None),
		"robot_time": message.get("robot_time", None),
		"obs_period": message.get("period", None),
		"robot_tcp_pos_base": _message_tcp_pos_base(message),
		"tcp_offset_ee": _to_jsonable(getattr(obs_receiver, "_tcp_offset_ee", None)),
		"socket_pos_base": _to_jsonable(getattr(obs_receiver, "_socket_pos", None)),
		"socket_quat_wxyz": _to_jsonable(getattr(obs_receiver, "_socket_quat", None)),
		"obs_tcp_pos_socket": _to_jsonable(obs_cpu[:3]),
		"obs_tcp_quat_socket_wxyz": _to_jsonable(obs_cpu[3:7]) if obs_cpu.numel() >= 7 else None,
		"obs_tcp_linvel_socket": _to_jsonable(obs_cpu[7:10]) if obs_cpu.numel() >= 10 else None,
		"obs_tcp_angvel_socket": _to_jsonable(obs_cpu[10:13]) if obs_cpu.numel() >= 13 else None,
		"obs_gripper_width": float(obs_cpu[13].item()) if obs_cpu.numel() >= 14 else None,
		"obs_force_or_wrench": _to_jsonable(obs_cpu[14:]) if obs_cpu.numel() > 14 else [],
		"policy_action": _to_jsonable(action_cpu),
		"policy_action_frame": send_message.get("policy_action_frame") if isinstance(send_message, dict) else None,
		"command_raw_delta": send_message.get("raw_action") if isinstance(send_message, dict) else None,
		"command_delta_sent": sent_delta,
		"command_frame": command_frame,
		"command_delta_sent_socket_est": sent_delta_socket,
		"command_send_ok": send_message.get("send_ok") if isinstance(send_message, dict) else None,
		"estimated_next_tcp_pos_socket": next_pos_socket,
		"pi_std": pi_std,
		"message_has_direct_obs": "obs" in message,
		"obs_meta": message.get("obs_meta", None),
	}
	handle.write(json.dumps(row, ensure_ascii=False) + "\n")
	handle.flush()


@torch.no_grad()
def eval_real_closed_loop(agent: TDMPC2, cfg, logger: Logger):
	"""
	Closed-loop real-robot inference.

	Robot side publishes the latest canonical observation over ZMQ. Newt consumes
	that observation, runs the current policy/planner, and sends one 6D delta
	action back to the robot action receiver.
	"""
	device = torch.device(f"cuda:{cfg.device_id}")
	obs_dim = int(cfg.obs_shape['state'][0])
	max_steps = int(cfg.get('eval_real_steps', None) or cfg.episode_length)
	use_mpc = bool(cfg.get('mpc', True))
	task_id = _real_task_id(cfg)
	step_count = 0
	last_log = None
	start_time = monotonic()

	print(colored(
		"Starting real closed-loop inference: "
		f"obs_dim={obs_dim}, action_dim={cfg.action_dim}, max_steps={max_steps}, "
		f"obs_endpoint={cfg.eval_real_obs_server} ({cfg.eval_real_obs_socket_type}), "
		f"action_endpoint={cfg.eval_zmq_server}.",
		"cyan",
		attrs=["bold"],
	))
	print(colored(
		"Robot observation can be either direct `obs` or libfranka robot_state. "
		"Direct obs layout: [tcp_pos_socket(3), tcp_quat_wxyz(4), tcp_linvel_socket(3), "
		"tcp_angvel_socket(3), gripper_width(1), optional force/wrench]. "
		"libfranka mode needs --full-state plus socket pose calibration and force fields for force checkpoints.",
		"cyan",
		attrs=["bold"],
	))

	debug_log_fp, debug_log = _make_real_debug_log(cfg)
	if debug_log_fp is not None:
		print(colored(f"Real debug log: {debug_log_fp}", "cyan", attrs=["bold"]))
	try:
		with EvalTraceRecorder(cfg, phase="real_closed_loop") as trace, \
			make_eval_zmq_publisher(cfg) as action_publisher, \
			make_eval_zmq_observation_receiver(cfg) as obs_receiver:
			if trace.enabled and trace.env_index != 0:
				raise ValueError("Real closed-loop trace only supports eval_trace_env_index=0 because num_envs is forced to 1.")
			message = obs_receiver.recv()
			for step_idx in range(max_steps):
				if obs_receiver.is_done(message):
					action_publisher.send_done(step=step_idx, episode_step=step_idx, task_id=task_id)
					break
				obs = obs_receiver.obs_tensor(message, obs_dim=obs_dim, device=device)
				model_tasks = _real_task_input(cfg, obs_receiver, message, device)
				t0 = torch.tensor([step_idx == 0], dtype=torch.bool, device=device)
				torch.compiler.cudagraph_mark_step_begin()
				action, info = agent(
					obs,
					t0=t0,
					step=1 if use_mpc else 0,
					eval_mode=True,
					task=model_tasks,
					mpc=use_mpc,
				)
				policy_delta = _shape_real_policy_action(action, step_idx, cfg)
				command_action = _real_action_for_command_frame(policy_delta, obs, cfg, obs_receiver)
				episode_step = int(message.get("episode_step", step_idx))
				send_message = action_publisher.send_action(
					command_action,
					step=step_idx,
					episode_step=episode_step,
					task_id=task_id,
					state_seq=message.get("seq", None),
					state_timestamp=message.get("timestamp", None),
					preprocessed=True,
					raw_action=policy_delta.detach().reshape(-1).to("cpu", dtype=torch.float32).tolist(),
				) or {}
				step_count += 1
				elapsed_s = monotonic() - start_time
				_write_real_debug_row(
					debug_log,
					step_idx=step_idx,
					elapsed_s=elapsed_s,
					task_id=task_id,
					message=message,
					obs=obs,
					action=action,
					send_message=send_message,
					obs_receiver=obs_receiver,
					info=info,
				)
				if (
					last_log is None or
					elapsed_s - last_log >= float(cfg.get('progress_log_interval_sec', 30.0)) or
					step_idx == max_steps - 1
				):
					last_log = elapsed_s
					action_max = float(action.detach().abs().max().item())
					pi_std = info.get("pi_std", None) if info is not None else None
					pi_std_text = "n/a" if pi_std is None else f"{float(torch.as_tensor(pi_std).detach().cpu().item()):.4g}"
					print(colored(
						f"real progress step={step_count}/{max_steps} "
						f"elapsed={elapsed_s:.1f}s action_abs_max={action_max:.4g} pi_std={pi_std_text}",
						"cyan",
						attrs=["bold"],
					), flush=True)
				if step_idx + 1 >= max_steps:
					break
				message = obs_receiver.recv()
			action_publisher.send_done(step=step_count, episode_step=step_count, task_id=task_id)
	finally:
		if debug_log is not None:
			debug_log.close()

	elapsed = monotonic() - start_time
	return {
		"step": step_count,
		"episode": 1,
		"episode_reward": 0.0,
		"episode_score": 0.0,
		"episode_length": step_count,
		"episode_success": 0.0,
		"eval_real_steps": step_count,
		"elapsed_time": elapsed,
		"steps_per_second": step_count / max(elapsed, 1.0e-6),
	}


def eval_by_trials(trainer: Trainer, total_trials: int):
	"""
	Evaluate for an exact total number of completed episodes across all envs.
	For SRSA, final_info.success is the configured wrapper success metric
	(e.g. terminal_process by default), with official/process diagnostics
	logged separately.
	"""
	local_target = total_trials
	if trainer.cfg.world_size > 1:
		local_target = total_trials // trainer.cfg.world_size
		if trainer.cfg.rank < (total_trials % trainer.cfg.world_size):
			local_target += 1

	task_results = defaultdict(empty_metrics)
	obs, info = trainer.env.reset()
	episode_reward = torch.zeros(trainer.cfg.num_envs, device=trainer._rollout_device)
	episode_len = torch.zeros(trainer.cfg.num_envs, device=trainer._rollout_device)
	completed = 0
	trace_step = 0

	if trainer.cfg.save_video:
		trainer.logger.video.init(trainer.env, enabled=trainer.cfg.rank == 0)

	with EvalTraceRecorder(trainer.cfg, phase="sim") as trace, make_eval_zmq_publisher(trainer.cfg) as action_publisher:
		if trace.enabled and not (0 <= trace.env_index < trainer.cfg.num_envs):
			raise ValueError(
				f"eval_trace_env_index={trace.env_index} is out of range for num_envs={trainer.cfg.num_envs}."
			)
		while completed < local_target:
			use_mpc = trainer._step > 0 or trainer.cfg.finetune
			torch.compiler.cudagraph_mark_step_begin()
			model_tasks = trainer._model_tasks()
			prev_obs = obs
			trace_env_index = trace.env_index
			trace_episode_step = int(episode_len[trace_env_index].item()) if trace.enabled else None
			action, action_info = trainer.agent(
				obs,
				t0=episode_len == 0,
				step=trainer._step,
				eval_mode=True,
				task=model_tasks,
				mpc=use_mpc,
			)
			sent_message = None
			if trainer.cfg.rank == 0:
				env_index = int(trainer.cfg.get('eval_zmq_env_index', 0))
				sent_message = action_publisher.send_action(
					action,
					step=trainer._step,
					episode_step=int(episode_len[env_index].item()),
					task_id=int(trainer._tasks[env_index].item()),
				)
			obs, reward, terminated, truncated, info = trainer.env.step(action)

			done = terminated | truncated
			episode_reward += reward
			episode_len += 1
			trace.record(
				step=trace_step,
				episode_step=trace_episode_step,
				obs=prev_obs,
				action=action,
				task=model_tasks,
				action_info=action_info,
				sent_action=None if sent_message is None else sent_message.get("action", None),
				next_obs=obs,
				reward=reward,
				done=done,
				info=info,
			)
			trace_step += 1

			if trainer.cfg.rank == 0:
				env_index = int(trainer.cfg.get('eval_zmq_env_index', 0))
				if bool(done[env_index].item()):
					action_publisher.send_done(
						step=trainer._step,
						episode_step=int(episode_len[env_index].item()),
						task_id=int(trainer._tasks[env_index].item()),
					)

			if 'final_info' in info:
				for i in range(trainer.cfg.num_envs):
					if not done[i]:
						continue
					if completed >= local_target:
						break
					task_id = trainer._tasks[i].item()
					task_name = trainer.cfg.global_tasks[task_id]
					task_results[task_name]['reward'].append(episode_reward[i].item())
					task_results[task_name]['length'].append(episode_len[i].item())
					task_results[task_name]['success'].append(info['final_info']['success'][i].item())
					task_results[task_name]['score'].append(info['final_info']['score'][i].item())
					for metric_key in SUCCESS_DIAGNOSTIC_KEYS:
						if metric_key in info['final_info']:
							value = torch.nan_to_num(info['final_info'][metric_key][i], nan=0.0)
							task_results[task_name][metric_key].append(value.item())
					episode_reward[i] = 0.0
					episode_len[i] = 0.0
					completed += 1

			if trainer.cfg.save_video and completed == 0:
				trainer.logger.video.record(trainer.env)

	if trainer.cfg.save_video:
		trainer.logger.video.save(trainer._step)

	barrier()

	if trainer.cfg.world_size > 1:
		gathered_results = [None for _ in range(trainer.cfg.world_size)] if trainer.cfg.rank == 0 else None
		torch.distributed.gather_object(task_results, gathered_results, dst=0)
		if trainer.cfg.rank == 0:
			merged_results = defaultdict(empty_metrics)
			for rank_results in gathered_results:
				for task_name, metrics in rank_results.items():
					for metric_name, values in metrics.items():
						merged_results[task_name][metric_name].extend(values)
			task_results = merged_results
		else:
			return None

	if trainer.cfg.rank != 0:
		return None

	metrics = {}
	total_count = 0
	total_success = 0.0
	for task_name, values in task_results.items():
		if len(values['reward']) == 0:
			continue
		task_count = len(values['reward'])
		total_count += task_count
		total_success += sum(values['success'])
		prefix = f'eval/{task_name}'
		metrics[f'{prefix}/episode_reward'] = sum(values['reward']) / task_count
		metrics[f'{prefix}/episode_length'] = sum(values['length']) / task_count
		metrics[f'{prefix}/episode_success'] = sum(values['success']) / task_count
		metrics[f'{prefix}/episode_score'] = sum(values['score']) / task_count
		for metric_key in SUCCESS_DIAGNOSTIC_KEYS:
			if len(values[metric_key]) > 0:
				metrics[f'{prefix}/episode_{metric_key}'] = sum(values[metric_key]) / len(values[metric_key])

	if total_count == 0:
		raise RuntimeError('No completed evaluation episodes were collected.')

	metrics['episode_reward'] = sum(sum(v['reward']) for v in task_results.values()) / total_count
	metrics['episode_length'] = sum(sum(v['length']) for v in task_results.values()) / total_count
	metrics['episode_success'] = total_success / total_count
	metrics['episode_score'] = sum(sum(v['score']) for v in task_results.values()) / total_count
	for metric_key in SUCCESS_DIAGNOSTIC_KEYS:
		metric_values = [
			value
			for task_values in task_results.values()
			for value in task_values[metric_key]
		]
		if len(metric_values) > 0:
			metrics[f'episode_{metric_key}'] = sum(metric_values) / len(metric_values)
	metrics['eval_trials'] = total_count
	return metrics


def _metric_value(metrics, *names, default=0.0):
	for name in names:
		for key in (name, f"episode_{name}"):
			if key in metrics:
				return float(metrics[key])
	return default


def _write_eval_summary(cfg, metrics):
	if int(cfg.get('rank', 0)) != 0:
		return
	output_dir = Path(cfg.work_dir) / "eval_summary"
	output_dir.mkdir(parents=True, exist_ok=True)
	official_latched = _metric_value(metrics, "official_success_latched", "official_success")
	official_terminal = _metric_value(metrics, "official_success_terminal", "current_official_success")
	relaxed_success = _metric_value(metrics, "relaxed_success_stable", "relaxed_terminal_process_success")
	relaxed_process_success = _metric_value(metrics, "relaxed_process_success_terminal", "relaxed_process_success")
	strict_success = _metric_value(metrics, "strict_success_stable", "terminal_process_success", "success")
	process_success = _metric_value(metrics, "process_success_terminal", "process_success")
	depth_fraction = _metric_value(metrics, "depth_fraction")
	lateral_error = _metric_value(metrics, "lateral_error")
	angle_error = _metric_value(metrics, "angle_error", "orientation_error", "yaw_error")
	keypoint_error = _metric_value(metrics, "keypoint_error")
	row = {
		"assembly_id": str(cfg.get('assembly_id', "")),
		"official_success_latched": official_latched,
		"official_success_terminal": official_terminal,
		"relaxed_success": relaxed_success,
		"relaxed_process_success": relaxed_process_success,
		"strict_success": strict_success,
		"process_success": process_success,
		"mean_depth_fraction": depth_fraction,
		"mean_lateral_error_mm": lateral_error * 1000.0,
		"mean_angle_error_deg": math.degrees(angle_error),
		"mean_keypoint_error_mm": keypoint_error * 1000.0,
		"episode_len_mean": _metric_value(metrics, "episode_length", "length"),
		"official_relaxed_gap": official_latched - relaxed_success,
		"relaxed_strict_gap": relaxed_success - strict_success,
		"official_strict_gap": official_latched - strict_success,
	}
	json_fp = output_dir / "eval_summary.json"
	csv_fp = output_dir / "eval_summary.csv"
	with open(json_fp, "w", encoding="utf-8") as f:
		json.dump(row, f, ensure_ascii=True, indent=2)
	with open(csv_fp, "w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=list(row.keys()))
		writer.writeheader()
		writer.writerow(row)
	print(colored(f"Saved eval summary JSON: {json_fp}", "green", attrs=["bold"]))
	print(colored(f"Saved eval summary CSV: {csv_fp}", "green", attrs=["bold"]))


def evaluate(rank: int, cfg: dict):
	"""
	Script for checkpoint evaluation.
	Loads a trained model, runs Trainer.eval(), logs metrics, and exits.
	"""
	if cfg.world_size > 1:
		setup(rank, cfg.world_size, cfg.port)
		print(colored('Rank:', 'yellow', attrs=['bold']), rank)
	set_seed(cfg.seed + rank)
	cfg.rank = rank
	cfg.device_id = cfg.gpu_id + rank
	torch.cuda.set_device(cfg.device_id)

	if not cfg.checkpoint:
		raise ValueError('`checkpoint` must be provided for evaluation.')
	if not os.path.exists(cfg.checkpoint):
		raise FileNotFoundError(f'Checkpoint file not found: {cfg.checkpoint}')
	if cfg.num_global_tasks > 1 and cfg.task != 'soup' and cfg.eval_task_id is None:
		raise ValueError(
			'`eval_task_id` must be provided when evaluating a multitask checkpoint outside of soup mode.'
		)
	real_closed_loop = _is_real_closed_loop(cfg)
	real_compat = None
	if real_closed_loop:
		real_compat = _configure_real_closed_loop_cfg(cfg)

	def make_agent(cfg):
		model = WorldModel(cfg).to(f"cuda:{cfg.device_id}")
		agent = TDMPC2(model, cfg)
		agent.load(cfg.checkpoint)
		agent.eval()
		agent.model.eval()
		return agent

	cfg.save_agent = False
	if real_closed_loop:
		logger = Logger(cfg)
		agent = make_agent(cfg)
		try:
			if cfg.rank == 0:
				print(colored(f'Evaluating checkpoint: {cfg.checkpoint}', 'blue', attrs=['bold']))
				print(colored('Evaluation mode: real closed_loop', 'blue', attrs=['bold']))
				print(colored(
					f"Checkpoint I/O: obs_dim={real_compat['obs_dim']} "
					f"action_dim={real_compat['action_dim']} task_dim={real_compat['task_dim']}",
					'blue',
					attrs=['bold'],
				))
			eval_metrics = eval_real_closed_loop(agent, cfg, logger)
			logger.log(eval_metrics, 'eval')
			logger.finish()
			if cfg.rank == 0:
				print(colored('Real closed-loop inference completed successfully.', 'green', attrs=['bold']))
			return
		except Exception as e:
			print(colored(f'[Rank {cfg.rank}] Real closed-loop eval crashed with exception: {repr(e)}', 'red', attrs=['bold']))
			raise
		finally:
			if torch.distributed.is_initialized():
				torch.distributed.destroy_process_group()

	env = make_env(cfg)
	logger = Logger(cfg)
	agent = make_agent(cfg)
	trainer = Trainer(
		cfg=cfg,
		env=env,
		agent=agent,
		buffer=None,
		logger=logger,
	)
	barrier()
	try:
		if cfg.rank == 0:
			print(colored(f'Evaluating checkpoint: {cfg.checkpoint}', 'blue', attrs=['bold']))
			print(colored(f'Evaluation mode: {cfg.eval_mode}', 'blue', attrs=['bold']))
			if cfg.eval_task_id is not None:
				print(colored(f'Evaluation task_id: {cfg.eval_task_id}', 'blue', attrs=['bold']))
			if cfg.eval_zmq_enabled:
				print(colored(f'Sending eval actions over ZMQ to {cfg.eval_zmq_server}', 'blue', attrs=['bold']))
		if cfg.mpc:
			trainer._step = 1
		if cfg.eval_trials is not None:
			eval_metrics = eval_by_trials(trainer, cfg.eval_trials)
		else:
			if cfg.rank == 0 and cfg.eval_trace_enabled:
				print(colored(
					"`eval_trace_enabled=true` currently records sim eval only when `eval_trials` is set; "
					"running Trainer.eval() without a trace file.",
					'yellow',
					attrs=['bold'],
				))
			eval_metrics = trainer.eval()
		eval_metrics.update(trainer.common_metrics())
		if cfg.task == 'soup':
			trainer.logger.pprint_multitask(eval_metrics, cfg)
		trainer.logger.log(eval_metrics, 'eval')
		if cfg.rank == 0 and cfg.eval_trials is not None:
			success_metric = str(cfg.get('srsa_eval_success_metric', 'success'))
			print(colored(
				f"Eval success ({success_metric}) over {int(eval_metrics['eval_trials'])} trials: "
				f"{float(eval_metrics['episode_success']):.4f}",
				'green',
				attrs=['bold'],
			))
			_write_eval_summary(cfg, eval_metrics)
		trainer.logger.finish()
		if cfg.rank == 0:
			print(colored('Evaluation completed successfully.', 'green', attrs=['bold']))
	except Exception as e:
		print(colored(f'[Rank {cfg.rank}] Evaluation crashed with exception: {repr(e)}', 'red', attrs=['bold']))
		raise
	finally:
		if torch.distributed.is_initialized():
			torch.distributed.destroy_process_group()


@hydra.main(version_base=None, config_name="config")
def launch(cfg: Config):
	assert torch.cuda.is_available()
	if cfg.checkpoint:
		cfg.checkpoint = str(
			Path(hydra.utils.to_absolute_path(str(cfg.checkpoint))).expanduser().resolve()
		)
	cfg = parse_cfg(cfg)
	cfg = apply_eval_task_template(cfg)
	cfg.enable_wandb = cfg.enable_wandb
	print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)

	available_gpus = torch.cuda.device_count() - cfg.gpu_id
	assert available_gpus > 0, \
		f'gpu_id={cfg.gpu_id} leaves no visible CUDA devices (total={torch.cuda.device_count()}).'
	if cfg.multiproc:
		requested_gpus = cfg.num_gpus if cfg.num_gpus is not None else available_gpus
		assert requested_gpus > 0, f'num_gpus must be positive, got {requested_gpus}.'
		assert requested_gpus <= available_gpus, \
			f'Requested num_gpus={requested_gpus}, but only {available_gpus} GPUs are available from gpu_id={cfg.gpu_id}.'
		cfg.world_size = requested_gpus
	else:
		cfg.world_size = 1
	if cfg.world_size > 1:
		gpu_range = f'{cfg.gpu_id}-{cfg.gpu_id + cfg.world_size - 1}'
		print(colored(f'Using {cfg.world_size} GPUs for evaluation (cuda:{gpu_range})', 'green', attrs=['bold']))

	if cfg.world_size > 1:
		cfg.port = os.getenv("MASTER_PORT", str(12355 + int(os.getpid()) % 1000))
		torch.multiprocessing.spawn(
			evaluate,
			args=(cfg,),
			nprocs=cfg.world_size,
			join=True,
		)
	else:
		evaluate(0, cfg)


if __name__ == '__main__':
	launch()
