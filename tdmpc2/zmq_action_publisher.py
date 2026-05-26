from __future__ import annotations

import math
import time
from typing import Optional

import torch


class ZMQActionPublisher:
	"""
	Send eval actions to a Franka-side ZMQ receiver.

	The message intentionally keeps the SpaceMouse client's `delta` field so the
	same robot-side receiver can consume learned-policy increments.
	"""

	def __init__(
		self,
		server: str,
		*,
		env_index: int = 0,
		rate: float = 0.0,
		action_scale: float = 1.0,
		max_trans_delta: Optional[float] = None,
		max_rot_delta: Optional[float] = None,
		warmup_steps: int = 0,
		send_timeout_ms: int = 0,
		send_done: bool = True,
		enabled: bool = True,
		action_frame: str = "socket",
		command_frame: Optional[str] = None,
		action_order: Optional[list[str]] = None,
	):
		self.enabled = bool(enabled)
		self.server = server
		self.env_index = int(env_index)
		self.period = 1.0 / float(rate) if rate and rate > 0.0 else 0.0
		self.action_scale = float(action_scale)
		self.max_trans_delta = None if max_trans_delta is None else abs(float(max_trans_delta))
		self.max_rot_delta = None if max_rot_delta is None else abs(float(max_rot_delta))
		self.warmup_steps = max(int(warmup_steps), 0)
		self.send_timeout_ms = max(int(send_timeout_ms), 0)
		self._send_done = bool(send_done)
		self.action_frame = str(action_frame)
		self.command_frame = str(command_frame) if command_frame is not None else self.action_frame
		self.action_order = list(action_order or ["dx", "dy", "dz", "droll", "dpitch", "dyaw"])
		self._context = None
		self._socket = None
		self._seq = 0
		self._last_send_time: Optional[float] = None

		if not self.enabled:
			return
		try:
			import zmq
		except ModuleNotFoundError as exc:
			raise ModuleNotFoundError(
				"`eval_zmq_enabled=true` requires pyzmq. Install `pyzmq` in the runtime environment."
			) from exc

		self._zmq = zmq
		self._context = zmq.Context()
		self._socket = self._context.socket(zmq.PUSH)
		self._socket.setsockopt(zmq.SNDHWM, 1)
		self._socket.setsockopt(zmq.LINGER, 0)
		self._socket.setsockopt(zmq.SNDTIMEO, self.send_timeout_ms)
		self._socket.connect(server)

	def close(self):
		if self._socket is not None:
			self._socket.close()
			self._socket = None
		if self._context is not None:
			self._context.term()
			self._context = None

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc, tb):
		self.close()

	def _select_action(self, action) -> list[float]:
		if torch.is_tensor(action):
			value = action.detach()
			if value.ndim >= 2:
				value = value[self.env_index]
			value = value.reshape(-1).to("cpu", dtype=torch.float32)
			return (value * self.action_scale).tolist()
		value = action
		if hasattr(action, "__len__") and len(action) > 0:
			first = action[0]
			if hasattr(first, "__len__") or torch.is_tensor(first):
				value = action[self.env_index]
		if torch.is_tensor(value):
			value = value.detach().reshape(-1).to("cpu", dtype=torch.float32)
			return (value * self.action_scale).tolist()
		return [float(x) * self.action_scale for x in value]

	def _shape_delta(self, delta: list[float], step: int) -> list[float]:
		tensor = torch.as_tensor(delta, dtype=torch.float32).reshape(-1)
		if self.warmup_steps > 0:
			scale = min(max(int(step) + 1, 0), self.warmup_steps) / float(self.warmup_steps)
			tensor = tensor * scale
		if self.max_trans_delta is not None:
			tensor[:3] = tensor[:3].clamp(-self.max_trans_delta, self.max_trans_delta)
		if self.max_rot_delta is not None:
			tensor[3:6] = tensor[3:6].clamp(-self.max_rot_delta, self.max_rot_delta)
		return tensor.tolist()

	def _maybe_sleep_for_rate(self):
		if self.period <= 0.0:
			return
		now = time.monotonic()
		if self._last_send_time is not None:
			sleep_time = self.period - (now - self._last_send_time)
			if sleep_time > 0.0:
				time.sleep(sleep_time)
		self._last_send_time = time.monotonic()

	def send_action(
		self,
		action,
		*,
		step: int,
		episode_step: Optional[int] = None,
		task_id: Optional[int] = None,
		state_seq: Optional[int] = None,
		state_timestamp: Optional[float] = None,
		done: bool = False,
		preprocessed: bool = False,
		raw_action: Optional[list[float]] = None,
	):
		if not self.enabled or self._socket is None:
			return None
		if preprocessed:
			delta = action.detach().reshape(-1).to("cpu", dtype=torch.float32).tolist() if torch.is_tensor(action) else [float(x) for x in action]
			raw_delta = list(raw_action) if raw_action is not None else list(delta)
		else:
			delta = self._select_action(action)
			raw_delta = list(delta)
			delta = self._shape_delta(delta, step)
		expected_dim = len(self.action_order)
		if len(delta) == 3 and expected_dim == 6:
			delta = [*delta, 0.0, 0.0, 0.0]
		if len(delta) != expected_dim:
			raise ValueError(
				f"Expected a {expected_dim}D action for ZMQ robot control "
				f"with action_order={self.action_order}, got {len(delta)}D."
			)
		self._maybe_sleep_for_rate()
		message = {
			"seq": self._seq,
			"timestamp": time.time(),
			"command": {
				"type": "cartesian_delta",
				"delta": delta,
				"frame": self.command_frame,
				"policy_frame": self.action_frame,
				"state_seq": None if state_seq is None else int(state_seq),
				"state_timestamp": None if state_timestamp is None else float(state_timestamp),
			},
			"delta": delta,
			"buttons": [],
			"gripper_toggle": False,
			"slow_mode": False,
			"action": delta,
			"raw_action": raw_delta,
			"step": int(step),
			"episode_step": None if episode_step is None else int(episode_step),
			"task_id": None if task_id is None else int(task_id),
			"done": bool(done),
			"source": "newt_eval",
			"action_frame": self.command_frame,
			"command_frame": self.command_frame,
			"policy_action_frame": self.action_frame,
			"action_order": self.action_order,
		}
		try:
			flags = self._zmq.NOBLOCK if self.send_timeout_ms <= 0 else 0
			self._socket.send_json(message, flags=flags)
			message["send_ok"] = True
		except self._zmq.Again:
			message["send_ok"] = False
		self._seq += 1
		return message

	def send_done(
		self,
		*,
		step: int,
		episode_step: Optional[int] = None,
		task_id: Optional[int] = None,
		state_seq: Optional[int] = None,
		state_timestamp: Optional[float] = None,
	):
		if not self._send_done:
			return
		return self.send_action(
			[0.0] * len(self.action_order),
			step=step,
			episode_step=episode_step,
			task_id=task_id,
			state_seq=state_seq,
			state_timestamp=state_timestamp,
			done=True,
		)


def make_eval_zmq_publisher(cfg) -> ZMQActionPublisher:
	enabled = bool(cfg.get("eval_zmq_enabled", False)) and int(cfg.get("rank", 0)) == 0
	action_order = cfg.get("eval_zmq_action_order", "dx,dy,dz,droll,dpitch,dyaw")
	if isinstance(action_order, str):
		action_order = [item.strip() for item in action_order.split(",") if item.strip()]
	return ZMQActionPublisher(
		cfg.get("eval_zmq_server", "tcp://localhost:5555"),
		env_index=cfg.get("eval_zmq_env_index", 0),
		rate=cfg.get("eval_zmq_rate", 0.0),
		action_scale=cfg.get("eval_zmq_action_scale", 1.0),
		max_trans_delta=cfg.get("eval_zmq_max_trans_delta", None),
		max_rot_delta=cfg.get("eval_zmq_max_rot_delta", None),
		warmup_steps=cfg.get("eval_zmq_warmup_steps", 0),
		send_timeout_ms=cfg.get("eval_zmq_send_timeout_ms", 0),
		send_done=cfg.get("eval_zmq_send_done", True),
		enabled=enabled,
		action_frame=cfg.get("eval_zmq_action_frame", "socket"),
		command_frame=cfg.get("eval_zmq_command_frame", None),
		action_order=action_order,
	)


class ZMQObservationReceiver:
	"""
	Receive real-robot observations for closed-loop Newt inference.

	Expected JSON payload:
	{
		"obs": [17 floats],
		"task_vec_6": [6 floats],  # optional
		"done": false,             # optional
		"episode_step": 0          # optional
	}
	"""

	def __init__(
		self,
		server: str,
		*,
		socket_type: str = "sub",
		connect: bool = True,
		timeout_ms: int = 1000,
		obs_key: str = "obs",
		task_vec_key: str = "task_vec_6",
		done_key: str = "done",
		state_format: str = "auto",
		socket_pos=None,
		socket_quat_wxyz=None,
		socket_quat_xyzw=None,
		socket_euler_xyz=None,
		socket_euler_degrees: bool = False,
		tcp_offset_ee=None,
		use_initial_pose_as_socket: bool = False,
		gripper_width_default: float = 0.0,
		force_scale: float = 50.0,
		zero_missing_force: bool = True,
		enabled: bool = True,
	):
		self.enabled = bool(enabled)
		self.server = server
		self.socket_type = str(socket_type).strip().lower()
		self.connect = bool(connect)
		self.timeout_ms = int(timeout_ms)
		self.obs_key = str(obs_key)
		self.task_vec_key = str(task_vec_key)
		self.done_key = str(done_key)
		self.state_format = str(state_format).strip().lower()
		self.use_initial_pose_as_socket = bool(use_initial_pose_as_socket)
		self.gripper_width_default = float(gripper_width_default)
		self.force_scale = float(force_scale)
		self.zero_missing_force = bool(zero_missing_force)
		self._socket_pos = self._coerce_vector(socket_pos, 3, "eval_real_socket_pos")
		self._socket_quat = self._coerce_quat_or_euler(
			socket_quat_wxyz,
			socket_quat_xyzw,
			socket_euler_xyz,
			socket_euler_degrees,
		)
		self._tcp_offset_ee = self._coerce_vector(tcp_offset_ee, 3, "eval_real_tcp_offset_ee")
		self._prev_pos = None
		self._prev_quat = None
		self._prev_time = None
		self._context = None
		self._socket = None

		if not self.enabled:
			return
		try:
			import zmq
		except ModuleNotFoundError as exc:
			raise ModuleNotFoundError(
				"`eval_real_mode=closed_loop` requires pyzmq. Install `pyzmq` in the runtime environment."
			) from exc

		self._zmq = zmq
		self._context = zmq.Context()
		if self.socket_type in {"sub", "subscribe", "pubsub", "pub_sub"}:
			self._socket = self._context.socket(zmq.SUB)
			self._socket.setsockopt(zmq.RCVHWM, 1)
			self._socket.setsockopt(zmq.SUBSCRIBE, b"")
		elif self.socket_type in {"pull", "pushpull", "push_pull"}:
			self._socket = self._context.socket(zmq.PULL)
			self._socket.setsockopt(zmq.RCVHWM, 1)
		else:
			raise ValueError(f"Unknown eval_real_obs_socket_type={self.socket_type!r}; use 'sub' or 'pull'.")
		self._socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
		self._socket.setsockopt(zmq.LINGER, 0)
		if self.connect:
			self._socket.connect(server)
		else:
			self._socket.bind(server)

	def close(self):
		if self._socket is not None:
			self._socket.close()
			self._socket = None
		if self._context is not None:
			self._context.term()
			self._context = None

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc, tb):
		self.close()

	def recv(self) -> dict:
		if not self.enabled or self._socket is None:
			raise RuntimeError("ZMQObservationReceiver is disabled.")
		try:
			message = self._socket.recv_json()
		except self._zmq.Again as exc:
			raise TimeoutError(
				f"Timed out waiting {self.timeout_ms} ms for real-robot observation on {self.server}."
			) from exc
		if not isinstance(message, dict):
			raise TypeError(f"Expected observation JSON object, got {type(message)}.")
		while True:
			try:
				latest = self._socket.recv_json(flags=self._zmq.NOBLOCK)
				if isinstance(latest, dict):
					message = latest
			except self._zmq.Again:
				break
		return message

	def obs_tensor(self, message: dict, *, obs_dim: int, device) -> torch.Tensor:
		if self.state_format in {"libfranka", "robot_state"}:
			obs = self._build_obs_from_robot_state(message, obs_dim)
		else:
			obs = message.get(self.obs_key, None)
			if obs is None:
				obs = message.get("observation", message.get("state", None))
			if obs is None:
				obs = self._build_obs_from_robot_state(message, obs_dim)
		obs = torch.as_tensor(obs, dtype=torch.float32, device=device).reshape(1, -1)
		if self.zero_missing_force and obs.shape[-1] == 14 and int(obs_dim) > 14:
			pad = torch.zeros((1, int(obs_dim) - 14), dtype=obs.dtype, device=obs.device)
			obs = torch.cat([obs, pad], dim=-1)
		if obs.shape[-1] != int(obs_dim):
			raise ValueError(f"Expected real observation dim {obs_dim}, got {obs.shape[-1]}.")
		return obs.contiguous()

	def task_vec_tensor(self, message: dict, *, device) -> Optional[torch.Tensor]:
		task_vec = message.get(self.task_vec_key, None)
		if task_vec is None:
			task_vec = message.get("task_vec", message.get("axial_task_vec_6", None))
		if task_vec is None:
			return None
		task_vec = torch.as_tensor(task_vec, dtype=torch.float32, device=device).reshape(1, -1)
		if task_vec.shape[-1] != 6:
			raise ValueError(f"Expected task_vec_6 dim 6, got {task_vec.shape[-1]}.")
		return task_vec.contiguous()

	def is_done(self, message: dict) -> bool:
		return bool(message.get(self.done_key, message.get("done", False)))

	@staticmethod
	def _coerce_vector(value, dim: int, name: str):
		if value is None:
			return None
		if isinstance(value, str):
			value = value.strip().strip("[]()")
			value = [item.strip() for item in value.split(",") if item.strip()]
		tensor = torch.as_tensor(value, dtype=torch.float64).reshape(-1)
		if tensor.numel() != dim:
			raise ValueError(f"{name} must contain {dim} values, got {tensor.numel()}.")
		return tensor

	@classmethod
	def _coerce_quat(cls, quat_wxyz, quat_xyzw):
		if quat_wxyz is not None and quat_xyzw is not None:
			raise ValueError("Provide only one of eval_real_socket_quat_wxyz or eval_real_socket_quat_xyzw.")
		if quat_xyzw is not None:
			quat_xyzw = cls._coerce_vector(quat_xyzw, 4, "eval_real_socket_quat_xyzw")
			quat = torch.stack([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
		elif quat_wxyz is not None:
			quat = cls._coerce_vector(quat_wxyz, 4, "eval_real_socket_quat_wxyz")
		else:
			return None
		return cls._normalize_quat(quat)

	@classmethod
	def _euler_xyz_to_quat(cls, euler_xyz, *, degrees: bool = False, name: str = "euler_xyz"):
		roll, pitch, yaw = cls._coerce_vector(euler_xyz, 3, name).unbind()
		if degrees:
			scale = math.pi / 180.0
			roll = roll * scale
			pitch = pitch * scale
			yaw = yaw * scale
		cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
		cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
		cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
		return cls._normalize_quat(torch.stack([
			cr * cp * cy + sr * sp * sy,
			sr * cp * cy - cr * sp * sy,
			cr * sp * cy + sr * cp * sy,
			cr * cp * sy - sr * sp * cy,
		]))

	@classmethod
	def _coerce_quat_or_euler(cls, quat_wxyz, quat_xyzw, euler_xyz, euler_degrees: bool):
		num_rotation_inputs = sum(value is not None for value in (quat_wxyz, quat_xyzw, euler_xyz))
		if num_rotation_inputs > 1:
			raise ValueError(
				"Provide only one of eval_real_socket_quat_wxyz, "
				"eval_real_socket_quat_xyzw, or eval_real_socket_euler_xyz."
			)
		if euler_xyz is not None:
			return cls._euler_xyz_to_quat(
				euler_xyz,
				degrees=euler_degrees,
				name="eval_real_socket_euler_xyz",
			)
		return cls._coerce_quat(quat_wxyz, quat_xyzw)

	@staticmethod
	def _normalize_quat(quat):
		quat = torch.as_tensor(quat, dtype=torch.float64).reshape(4)
		quat = quat / torch.linalg.norm(quat).clamp_min(1.0e-12)
		if quat[0] < 0:
			quat = -quat
		return quat

	@classmethod
	def _quat_conj(cls, quat):
		quat = cls._normalize_quat(quat)
		return torch.stack([quat[0], -quat[1], -quat[2], -quat[3]])

	@classmethod
	def _quat_mul(cls, a, b):
		a = cls._normalize_quat(a)
		b = cls._normalize_quat(b)
		aw, ax, ay, az = a.unbind()
		bw, bx, by, bz = b.unbind()
		return cls._normalize_quat(torch.stack([
			aw * bw - ax * bx - ay * by - az * bz,
			aw * bx + ax * bw + ay * bz - az * by,
			aw * by - ax * bz + ay * bw + az * bx,
			aw * bz + ax * by - ay * bx + az * bw,
		]))

	@classmethod
	def _quat_to_matrix(cls, quat):
		w, x, y, z = cls._normalize_quat(quat).unbind()
		return torch.tensor([
			[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
			[2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
			[2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
		], dtype=torch.float64)

	@classmethod
	def _matrix_to_quat(cls, matrix):
		m = torch.as_tensor(matrix, dtype=torch.float64).reshape(3, 3)
		trace = float(torch.trace(m).item())
		if trace > 0.0:
			s = math.sqrt(trace + 1.0) * 2.0
			w = 0.25 * s
			x = float((m[2, 1] - m[1, 2]).item()) / s
			y = float((m[0, 2] - m[2, 0]).item()) / s
			z = float((m[1, 0] - m[0, 1]).item()) / s
		else:
			diag = [float(m[i, i].item()) for i in range(3)]
			axis = int(max(range(3), key=lambda i: diag[i]))
			if axis == 0:
				s = math.sqrt(max(1.0 + diag[0] - diag[1] - diag[2], 1.0e-12)) * 2.0
				w = float((m[2, 1] - m[1, 2]).item()) / s
				x = 0.25 * s
				y = float((m[0, 1] + m[1, 0]).item()) / s
				z = float((m[0, 2] + m[2, 0]).item()) / s
			elif axis == 1:
				s = math.sqrt(max(1.0 + diag[1] - diag[0] - diag[2], 1.0e-12)) * 2.0
				w = float((m[0, 2] - m[2, 0]).item()) / s
				x = float((m[0, 1] + m[1, 0]).item()) / s
				y = 0.25 * s
				z = float((m[1, 2] + m[2, 1]).item()) / s
			else:
				s = math.sqrt(max(1.0 + diag[2] - diag[0] - diag[1], 1.0e-12)) * 2.0
				w = float((m[1, 0] - m[0, 1]).item()) / s
				x = float((m[0, 2] + m[2, 0]).item()) / s
				y = float((m[1, 2] + m[2, 1]).item()) / s
				z = 0.25 * s
		return cls._normalize_quat(torch.tensor([w, x, y, z], dtype=torch.float64))

	@staticmethod
	def _nested(message: dict, *path):
		value = message
		for key in path:
			if not isinstance(value, dict) or key not in value:
				return None
			value = value[key]
		return value

	def _extract_pose(self, message: dict):
		ee = message.get("end_effector", {}) if isinstance(message.get("end_effector", {}), dict) else {}
		matrix = message.get("O_T_EE", ee.get("O_T_EE", None))
		if matrix is not None:
			matrix = torch.as_tensor(matrix, dtype=torch.float64).reshape(4, 4).T
			pos = matrix[:3, 3]
			quat = self._matrix_to_quat(matrix[:3, :3])
			if self._tcp_offset_ee is not None:
				pos = pos + matrix[:3, :3] @ self._tcp_offset_ee
			return pos, quat

		pos = message.get("tcp_pos", message.get("position", ee.get("position", None)))
		if pos is None:
			raise KeyError(
				"libfranka robot_state must include end_effector.position or O_T_EE. "
				"Run the lower-machine server with --full-state, or send direct `obs`."
			)
		pos = self._coerce_vector(pos, 3, "end_effector.position")
		quat_xyzw = message.get("tcp_quat_xyzw", ee.get("orientation_quat_xyzw", None))
		quat_wxyz = message.get("tcp_quat_wxyz", ee.get("orientation_quat_wxyz", None))
		euler_xyz = message.get("tcp_euler_xyz", ee.get("orientation_euler_xyz", None))
		if quat_wxyz is None and quat_xyzw is None and euler_xyz is None:
			raise KeyError(
				"Real closed-loop canonical obs needs TCP orientation. "
				"Run libfranka zmq_cartesian_teleop_server.py with --full-state, "
				"or include end_effector.orientation_quat_xyzw / tcp_quat_wxyz / orientation_euler_xyz."
			)
		quat = self._coerce_quat_or_euler(quat_wxyz, quat_xyzw, euler_xyz, False)
		if self._tcp_offset_ee is not None:
			pos = pos + self._quat_to_matrix(quat) @ self._tcp_offset_ee
		return pos, quat

	def _extract_socket_pose(self, message: dict, tcp_pos, tcp_quat):
		socket = message.get("socket", message.get("target_socket", message.get("target", None)))
		if isinstance(socket, dict):
			pos = socket.get("position", socket.get("pos", None))
			quat_wxyz = socket.get("quat_wxyz", socket.get("orientation_quat_wxyz", None))
			quat_xyzw = socket.get("quat_xyzw", socket.get("orientation_quat_xyzw", None))
			euler_xyz = socket.get("euler_xyz", socket.get("orientation_euler_xyz", None))
			if pos is not None and (quat_wxyz is not None or quat_xyzw is not None or euler_xyz is not None):
				return self._coerce_vector(pos, 3, "socket.position"), self._coerce_quat_or_euler(
					quat_wxyz,
					quat_xyzw,
					euler_xyz,
					False,
				)
		if self._socket_pos is not None and self._socket_quat is not None:
			return self._socket_pos, self._socket_quat
		if self.use_initial_pose_as_socket:
			if self._socket_pos is None:
				self._socket_pos = tcp_pos.detach().clone()
				self._socket_quat = tcp_quat.detach().clone()
			return self._socket_pos, self._socket_quat
		raise KeyError(
			"Cannot build socket-frame canonical obs from libfranka state without socket pose. "
			"Provide eval_real_socket_pos and eval_real_socket_quat_wxyz/xyzw, include a `socket` object "
			"in each state message, or send prebuilt `obs` directly."
		)

	def _extract_gripper_width(self, message: dict):
		value = message.get("gripper_width", None)
		gripper = message.get("gripper", None)
		if value is None and isinstance(gripper, dict):
			value = gripper.get("width", None)
			if value is None and gripper.get("is_closed", None) is True:
				value = 0.0
		if value is None:
			value = self.gripper_width_default
		return float(value)

	def _extract_force_obs(self, message: dict, force_dim: int, socket_rot):
		for key in ("force_obs", "flange_force_obs", "wrench_obs", "flange_wrench_obs"):
			value = message.get(key, None)
			if value is not None:
				value = torch.as_tensor(value, dtype=torch.float64).reshape(-1)
				if value.numel() >= force_dim:
					return value[:force_dim]

		wrench = None
		source_frame = "socket"
		for key in ("wrench_socket", "force_socket", "K_F_ext_hat_K"):
			value = message.get(key, None)
			if value is not None:
				wrench = torch.as_tensor(value, dtype=torch.float64).reshape(-1)
				source_frame = "socket"
				break
		if wrench is None:
			value = message.get("O_F_ext_hat_K", None)
			if value is None:
				value = self._nested(message, "external_wrench", "O_F_ext_hat_K")
			if value is None:
				value = self._nested(message, "end_effector", "external_wrench_base")
			if value is not None:
				wrench = torch.as_tensor(value, dtype=torch.float64).reshape(-1)
				source_frame = "base"
		if wrench is None:
			if self.zero_missing_force:
				return torch.zeros(force_dim, dtype=torch.float64)
			raise KeyError(
				f"Checkpoint expects {14 + force_dim}D obs, but robot state lacks force/wrench fields. "
				"Publish `force_obs`/`flange_force_obs` if already normalized, or raw `O_F_ext_hat_K`/`K_F_ext_hat_K`."
			)
		if wrench.numel() == 3:
			wrench = torch.cat([wrench, wrench.new_zeros(3)])
		if wrench.numel() < force_dim:
			raise ValueError(f"Expected at least {force_dim} force values, got {wrench.numel()}.")
		if source_frame == "base":
			rot_inv = socket_rot.T
			force = rot_inv @ wrench[:3]
			torque = rot_inv @ wrench[3:6]
			wrench = torch.cat([force, torque])
		return wrench[:force_dim] / max(self.force_scale, 1.0e-12)

	def _build_obs_from_robot_state(self, message: dict, obs_dim: int):
		if self.state_format not in {"auto", "libfranka", "robot_state"}:
			raise KeyError(
				f"Observation message must contain `{self.obs_key}` or use eval_real_state_format=libfranka."
			)
		tcp_pos, tcp_quat = self._extract_pose(message)
		socket_pos, socket_quat = self._extract_socket_pose(message, tcp_pos, tcp_quat)
		socket_rot = self._quat_to_matrix(socket_quat)
		tcp_pos_socket = socket_rot.T @ (tcp_pos - socket_pos)
		tcp_quat_socket = self._quat_mul(self._quat_conj(socket_quat), tcp_quat)

		now = float(message.get("robot_time", message.get("timestamp", time.time())))
		linvel_base = message.get("tcp_linvel", self._nested(message, "end_effector", "linear_velocity"))
		angvel_base = message.get("tcp_angvel", self._nested(message, "end_effector", "angular_velocity"))
		if linvel_base is not None:
			linvel_base = self._coerce_vector(linvel_base, 3, "tcp_linvel")
		if angvel_base is not None:
			angvel_base = self._coerce_vector(angvel_base, 3, "tcp_angvel")
		if (linvel_base is None or angvel_base is None) and self._prev_pos is not None and self._prev_time is not None:
			dt = max(now - self._prev_time, 1.0e-6)
			if linvel_base is None:
				linvel_base = (tcp_pos - self._prev_pos) / dt
			if angvel_base is None:
				dq = self._quat_mul(tcp_quat, self._quat_conj(self._prev_quat))
				vec = dq[1:]
				vec_norm = torch.linalg.norm(vec)
				if vec_norm > 1.0e-12:
					angle = 2.0 * math.atan2(float(vec_norm.item()), float(dq[0].item()))
					angvel_base = vec / vec_norm * (angle / dt)
				else:
					angvel_base = torch.zeros(3, dtype=torch.float64)
		if linvel_base is None:
			linvel_base = torch.zeros(3, dtype=torch.float64)
		if angvel_base is None:
			angvel_base = torch.zeros(3, dtype=torch.float64)
		self._prev_pos = tcp_pos.detach().clone()
		self._prev_quat = tcp_quat.detach().clone()
		self._prev_time = now

		rot_inv = socket_rot.T
		parts = [
			tcp_pos_socket,
			tcp_quat_socket,
			rot_inv @ linvel_base,
			rot_inv @ angvel_base,
			torch.tensor([self._extract_gripper_width(message)], dtype=torch.float64),
		]
		if obs_dim > 14:
			parts.append(self._extract_force_obs(message, obs_dim - 14, socket_rot))
		obs = torch.cat(parts).to(dtype=torch.float32)
		return obs.tolist()


def make_eval_zmq_observation_receiver(cfg) -> ZMQObservationReceiver:
	real_mode = str(cfg.get("eval_real_mode", "stream")).lower().replace("-", "_")
	enabled = (
		str(cfg.get("eval_mode", "sim")).lower() == "real" and
		real_mode in {
			"closed_loop",
			"robot_closed_loop",
			"obs_closed_loop",
		}
	)
	return ZMQObservationReceiver(
		cfg.get("eval_real_obs_server", "tcp://localhost:5556"),
		socket_type=cfg.get("eval_real_obs_socket_type", "sub"),
		connect=cfg.get("eval_real_obs_connect", True),
		timeout_ms=cfg.get("eval_real_obs_timeout_ms", 1000),
		obs_key=cfg.get("eval_real_obs_key", "obs"),
		task_vec_key=cfg.get("eval_real_task_vec_key", "task_vec_6"),
		done_key=cfg.get("eval_real_done_key", "done"),
		state_format=cfg.get("eval_real_state_format", "auto"),
		socket_pos=cfg.get("eval_real_socket_pos", None),
		socket_quat_wxyz=cfg.get("eval_real_socket_quat_wxyz", None),
		socket_quat_xyzw=cfg.get("eval_real_socket_quat_xyzw", None),
		socket_euler_xyz=cfg.get("eval_real_socket_euler_xyz", None),
		socket_euler_degrees=cfg.get("eval_real_socket_euler_degrees", False),
		tcp_offset_ee=cfg.get("eval_real_tcp_offset_ee", None),
		use_initial_pose_as_socket=cfg.get("eval_real_use_initial_pose_as_socket", False),
		gripper_width_default=cfg.get("eval_real_gripper_width_default", 0.0),
		force_scale=cfg.get("eval_real_force_scale", 50.0),
		zero_missing_force=cfg.get("eval_real_zero_missing_force", True),
		enabled=enabled,
	)
