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
	):
		self.enabled = bool(enabled)
		self.server = server
		self.env_index = int(env_index)
		self.period = 1.0 / float(rate) if rate and rate > 0.0 else 0.0
		self.action_scale = float(action_scale)
		self.send_done = bool(send_done)
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
		}
		try:
			self._socket.send_json(message, flags=self._zmq.NOBLOCK)
		except self._zmq.Again:
			pass
		self._seq += 1

	def send_done(self, *, step: int, episode_step: Optional[int] = None, task_id: Optional[int] = None):
		if not self.send_done:
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
	return ZMQActionPublisher(
		cfg.get("eval_zmq_server", "tcp://localhost:5555"),
		env_index=cfg.get("eval_zmq_env_index", 0),
		rate=cfg.get("eval_zmq_rate", 0.0),
		action_scale=cfg.get("eval_zmq_action_scale", 1.0),
		send_done=cfg.get("eval_zmq_send_done", True),
		enabled=enabled,
	)
