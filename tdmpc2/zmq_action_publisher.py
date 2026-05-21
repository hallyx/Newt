from __future__ import annotations

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
		send_done: bool = True,
		enabled: bool = True,
		action_frame: str = "socket",
		action_order: Optional[list[str]] = None,
	):
		self.enabled = bool(enabled)
		self.server = server
		self.env_index = int(env_index)
		self.period = 1.0 / float(rate) if rate and rate > 0.0 else 0.0
		self.action_scale = float(action_scale)
		self._send_done = bool(send_done)
		self.action_frame = str(action_frame)
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
		done: bool = False,
	):
		if not self.enabled or self._socket is None:
			return
		delta = self._select_action(action)
		if len(delta) != 6:
			raise ValueError(f"Expected a 6D action for ZMQ robot control, got {len(delta)}D.")
		self._maybe_sleep_for_rate()
		message = {
			"seq": self._seq,
			"timestamp": time.time(),
			"delta": delta,
			"buttons": [],
			"gripper_toggle": False,
			"slow_mode": False,
			"action": delta,
			"step": int(step),
			"episode_step": None if episode_step is None else int(episode_step),
			"task_id": None if task_id is None else int(task_id),
			"done": bool(done),
			"source": "newt_eval",
			"action_frame": self.action_frame,
			"action_order": self.action_order,
		}
		try:
			self._socket.send_json(message, flags=self._zmq.NOBLOCK)
		except self._zmq.Again:
			pass
		self._seq += 1

	def send_done(self, *, step: int, episode_step: Optional[int] = None, task_id: Optional[int] = None):
		if not self._send_done:
			return
		self.send_action(
			[0.0] * 6,
			step=step,
			episode_step=episode_step,
			task_id=task_id,
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
		send_done=cfg.get("eval_zmq_send_done", True),
		enabled=enabled,
		action_frame=cfg.get("eval_zmq_action_frame", "socket"),
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
		connect: bool = True,
		timeout_ms: int = 1000,
		obs_key: str = "obs",
		task_vec_key: str = "task_vec_6",
		done_key: str = "done",
		enabled: bool = True,
	):
		self.enabled = bool(enabled)
		self.server = server
		self.connect = bool(connect)
		self.timeout_ms = int(timeout_ms)
		self.obs_key = str(obs_key)
		self.task_vec_key = str(task_vec_key)
		self.done_key = str(done_key)
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
		self._socket = self._context.socket(zmq.PULL)
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
		return message

	def obs_tensor(self, message: dict, *, obs_dim: int, device) -> torch.Tensor:
		obs = message.get(self.obs_key, None)
		if obs is None:
			obs = message.get("observation", message.get("state", None))
		if obs is None:
			raise KeyError(
				f"Observation message must contain `{self.obs_key}` "
				"or one of `observation`/`state`."
			)
		obs = torch.as_tensor(obs, dtype=torch.float32, device=device).reshape(1, -1)
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
		connect=cfg.get("eval_real_obs_connect", True),
		timeout_ms=cfg.get("eval_real_obs_timeout_ms", 1000),
		obs_key=cfg.get("eval_real_obs_key", "obs"),
		task_vec_key=cfg.get("eval_real_task_vec_key", "task_vec_6"),
		done_key=cfg.get("eval_real_done_key", "done"),
		enabled=enabled,
	)
