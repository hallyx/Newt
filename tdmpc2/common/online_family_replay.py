from __future__ import annotations

import json
import hashlib
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Optional

import torch

try:
	from tensordict import TensorDict
except ModuleNotFoundError:
	TensorDict = None

try:
	from termcolor import colored
except ModuleNotFoundError:
	colored = None

from common.buffer import Buffer


def _print(message: str, color: str = "cyan"):
	if colored is None:
		print(message)
	else:
		print(colored(message, color, attrs=["bold"]))


def _cfg_get(cfg, key, default=None):
	return cfg.get(key, default) if hasattr(cfg, "get") else getattr(cfg, key, default)


def _resolve_manifest_path(path_value: Optional[str]):
	if path_value is None or str(path_value).strip() == "":
		return None
	path = Path(str(path_value)).expanduser()
	return path if path.is_absolute() else (Path.cwd() / path).resolve()


def _resolve_entry_path(path_value: str, manifest_dir: Path):
	path = Path(str(path_value)).expanduser()
	if path.is_absolute():
		return path
	candidate = (manifest_dir / path).resolve()
	if candidate.exists():
		return candidate
	return (Path.cwd() / path).resolve()


def _manifest_entries(payload: Any):
	if payload is None:
		return []
	if isinstance(payload, list):
		return payload
	if isinstance(payload, dict):
		if isinstance(payload.get("tasks"), list):
			return payload["tasks"]
		if isinstance(payload.get("replays"), list):
			return payload["replays"]
	return []


def _cat_sequence_dim(values):
	first = values[0]
	if TensorDict is not None and isinstance(first, TensorDict):
		return torch.cat(values, dim=1)
	if isinstance(first, dict):
		return {
			key: _cat_sequence_dim([value[key] for value in values])
			for key in first.keys()
		}
	return torch.cat(values, dim=1).contiguous()


def _index_sequence_dim(value, order):
	if TensorDict is not None and isinstance(value, TensorDict):
		return value[:, order].contiguous()
	if isinstance(value, dict):
		return {
			key: _index_sequence_dim(item, order)
			for key, item in value.items()
		}
	return value[:, order].contiguous()


def _cat_optional_tensors(values):
	if all(value is None for value in values):
		return None
	if any(value is None for value in values):
		raise ValueError("Cannot mix replay batches with and without task tensors.")
	return torch.cat(values, dim=1).contiguous()


def _safe_count_key(value: Any):
	value = "none" if value is None else str(value)
	return value.strip() or "none"


def _task_vec_hash(vec):
	rounded = [round(float(item), 8) for item in vec]
	text = ",".join(f"{item:.8g}" for item in rounded)
	return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _task_hash_counts(task):
	if task is None:
		return {}
	if not torch.is_tensor(task) or not task.is_floating_point() or task.ndim < 2:
		return {}
	try:
		first = task[0] if task.ndim >= 3 else task
		flat = first.reshape(-1, first.shape[-1]).detach().cpu().float()
	except Exception:
		return {}
	counts = Counter()
	for vec in flat:
		counts[_task_vec_hash(vec.tolist())] += 1
	return dict(counts)


class OnlineFamilyReplayBuffer:
	"""
	Replay wrapper for staged SRSA family training.

	New rollouts are always written to `current_buffer`. Sampling mixes current,
	anchor, and history replay while keeping the original Buffer tuple contract.
	"""

	def __init__(self, current_buffer, replay_buffers, cfg, task_order=None):
		self.current_buffer = current_buffer
		self.replay_buffers = OrderedDict((str(k), v) for k, v in replay_buffers.items())
		self.cfg = cfg
		self.current_task_id = str(_cfg_get(cfg, "online_family_current_task_id", None) or _cfg_get(cfg, "assembly_id", "current"))
		self.anchor_task_id = str(_cfg_get(cfg, "online_family_anchor_task_id", "01125"))
		self.current_template_id = _cfg_get(cfg, "online_family_current_template_id", None)
		if self.current_template_id is None:
			self.current_template_id = _cfg_get(cfg, "srsa_param_template_id", _cfg_get(cfg, "srsa_task_template_id", None))
		self.current_template_id = _safe_count_key(
			self.current_template_id if self.current_template_id is not None else _cfg_get(cfg, "srsa_axial_clearance_depth_templates", "default")
		)
		self.current_condition_id = _safe_count_key(
			_cfg_get(cfg, "online_family_current_condition_id", None) or f"{self.current_task_id}|{self.current_template_id}"
		)
		self.current_ratio = max(0.0, float(_cfg_get(cfg, "online_family_current_ratio", 0.50)))
		self.anchor_ratio = max(0.0, float(_cfg_get(cfg, "online_family_anchor_ratio", 0.20)))
		self.history_ratio = max(0.0, float(_cfg_get(cfg, "online_family_history_ratio", 0.30)))
		self.min_current_episodes = max(0, int(_cfg_get(cfg, "online_family_min_current_episodes", 5)))
		self.sample_balance = str(_cfg_get(cfg, "online_family_sample_balance", "ratio")).strip().lower()
		self.bootstrap_min_episodes_per_condition = max(
			0,
			int(_cfg_get(cfg, "multi_task_bootstrap_min_episodes_per_condition", self.min_current_episodes)),
		)
		self.bootstrap_current_only = bool(_cfg_get(cfg, "multi_task_bootstrap_current_only", True))
		self.batch_size = int(_cfg_get(cfg, "batch_size", 1024))
		self.task_order = [str(task_id) for task_id in (task_order or self.replay_buffers.keys())]
		self.last_batch_source_counts = {}
		self.last_batch_source_distribution = {}
		self.last_batch_task_counts = {}
		self.last_batch_assembly_counts = {}
		self.last_batch_template_counts = {}
		self.last_batch_condition_counts = {}
		self.last_batch_task_hash_counts = {}

	@property
	def capacity(self):
		return self.current_buffer.capacity

	@property
	def num_eps(self):
		return self.current_buffer.num_eps

	@property
	def num_transitions(self):
		return getattr(self.current_buffer, "num_transitions", len(self.current_buffer))

	def __len__(self):
		return len(self.current_buffer)

	@classmethod
	def from_manifest(cls, current_buffer, cfg):
		manifest_fp = _resolve_manifest_path(_cfg_get(cfg, "online_family_replay_manifest_fp", None))
		if manifest_fp is None or not manifest_fp.exists():
			if manifest_fp is not None:
				_print(f"[online-family-replay] manifest not found yet, starting with current replay only: {manifest_fp}")
			return cls(current_buffer=current_buffer, replay_buffers={}, cfg=cfg, task_order=[])

		with open(manifest_fp, "r", encoding="utf-8") as f:
			payload = json.load(f)
		manifest_dir = manifest_fp.parent
		entries = _manifest_entries(payload)
		current_task_id = str(_cfg_get(cfg, "online_family_current_task_id", None) or _cfg_get(cfg, "assembly_id", "current"))
		current_template_id = _cfg_get(cfg, "online_family_current_template_id", None)
		if current_template_id is None:
			current_template_id = _cfg_get(cfg, "srsa_param_template_id", _cfg_get(cfg, "srsa_task_template_id", None))
		current_template_id = _safe_count_key(
			current_template_id if current_template_id is not None else _cfg_get(cfg, "srsa_axial_clearance_depth_templates", "default")
		)
		current_condition_id = _safe_count_key(
			_cfg_get(cfg, "online_family_current_condition_id", None) or f"{current_task_id}|{current_template_id}"
		)
		max_episodes = _cfg_get(cfg, "online_family_replay_max_episodes_per_task", None)
		replay_buffers = OrderedDict()
		task_order = []
		for entry in entries:
			if not isinstance(entry, dict):
				continue
			task_id = str(entry.get("task_id") or entry.get("assembly_id") or "").strip()
			replay_fp = entry.get("replay_fp") or entry.get("buffer_fp") or entry.get("path")
			if not task_id or not replay_fp:
				continue
			condition_id = _safe_count_key(entry.get("condition_id", None) or task_id)
			if condition_id == current_condition_id:
				continue
			replay_path = _resolve_entry_path(replay_fp, manifest_dir)
			if not replay_path.exists():
				_print(f"[online-family-replay] skipping missing replay for task_id={task_id}: {replay_path}", color="yellow")
				continue
			buffer = Buffer.load(
				replay_path,
				cfg=cfg,
				storage_device=_cfg_get(cfg, "online_family_replay_storage_device", "cpu"),
				max_episodes=max_episodes,
			)
			metadata = dict(getattr(buffer, "metadata", {}) or {})
			metadata.update({
				"task_id": task_id,
				"assembly_id": str(entry.get("assembly_id", task_id)),
				"template_id": _safe_count_key(entry.get("template_id", metadata.get("template_id", "default"))),
				"condition_id": condition_id,
				"role": str(entry.get("role", metadata.get("role", "history"))),
				"replay_fp": str(replay_path),
			})
			buffer.metadata = metadata
			replay_key = condition_id
			if replay_key in replay_buffers:
				replay_key = f"{condition_id}#{len(replay_buffers)}"
			replay_buffers[replay_key] = buffer
			task_order.append(replay_key)
		_print(
			f"[online-family-replay] loaded {len(replay_buffers)} previous replay buffers "
			f"for current_task_id={current_task_id}."
		)
		return cls(current_buffer=current_buffer, replay_buffers=replay_buffers, cfg=cfg, task_order=task_order)

	def set_storage_device(self, device):
		if hasattr(self.current_buffer, "set_storage_device"):
			self.current_buffer.set_storage_device(device)

	def add(self, td, world_size=1, rank=0):
		return self.current_buffer.add(td, world_size, rank)

	def save_current(self, path, metadata=None):
		metadata = dict(metadata or {})
		metadata.setdefault("task_id", self.current_task_id)
		metadata.setdefault("assembly_id", self.current_task_id)
		metadata.setdefault("template_id", self.current_template_id)
		metadata.setdefault("condition_id", self.current_condition_id)
		metadata.setdefault("role", "current")
		metadata.setdefault("anchor_task_id", self.anchor_task_id)
		return self.current_buffer.save_current(path, metadata=metadata)

	def _history_task_ids(self):
		return [
			replay_key for replay_key in self.task_order
			if replay_key in self.replay_buffers and not self._is_anchor_replay(replay_key)
		]

	def _replay_metadata(self, replay_key):
		buffer = self.replay_buffers[replay_key]
		return dict(getattr(buffer, "metadata", {}) or {})

	def _is_anchor_replay(self, replay_key):
		metadata = self._replay_metadata(replay_key)
		return (
			str(metadata.get("task_id", "")) == self.anchor_task_id or
			str(metadata.get("assembly_id", "")) == self.anchor_task_id or
			str(metadata.get("role", "")) == "anchor"
		)

	def _anchor_replay_key(self):
		for replay_key in self.task_order:
			if replay_key in self.replay_buffers and self._is_anchor_replay(replay_key):
				return replay_key
		return None

	def _current_label(self):
		return {
			"role": "current",
			"task_id": self.current_task_id,
			"assembly_id": self.current_task_id,
			"template_id": self.current_template_id,
			"condition_id": self.current_condition_id,
		}

	def _replay_label(self, role, replay_key):
		metadata = self._replay_metadata(replay_key)
		return {
			"role": role,
			"task_id": _safe_count_key(metadata.get("task_id", replay_key)),
			"assembly_id": _safe_count_key(metadata.get("assembly_id", metadata.get("task_id", replay_key))),
			"template_id": _safe_count_key(metadata.get("template_id", "default")),
			"condition_id": _safe_count_key(metadata.get("condition_id", replay_key)),
		}

	def _component_weights(self):
		weights = OrderedDict()
		weights[("current", self.current_condition_id)] = self.current_ratio
		anchor_key = self._anchor_replay_key()
		if anchor_key is not None:
			weights[("anchor", anchor_key)] = self.anchor_ratio
		history_ids = self._history_task_ids()
		if history_ids:
			per_history = self.history_ratio / len(history_ids)
			for replay_key in history_ids:
				weights[("history", replay_key)] = per_history
		weights = OrderedDict((key, value) for key, value in weights.items() if value > 0)
		total = sum(weights.values())
		if total <= 0:
			return OrderedDict({("current", self.current_condition_id): 1.0})
		return OrderedDict((key, value / total) for key, value in weights.items())

	def _condition_uniform_weights(self):
		weights = OrderedDict()
		weights[("current", self.current_condition_id)] = 1.0
		for replay_key in self.task_order:
			if replay_key not in self.replay_buffers:
				continue
			condition_id = _safe_count_key(self._replay_metadata(replay_key).get("condition_id", replay_key))
			if condition_id == self.current_condition_id:
				continue
			role = "anchor" if self._is_anchor_replay(replay_key) else "history"
			weights[(role, replay_key)] = 1.0
		total = sum(weights.values())
		return OrderedDict((key, value / total) for key, value in weights.items())

	def _counts_from_weights(self, weights, batch_size):
		raw = OrderedDict((key, value * batch_size) for key, value in weights.items())
		counts = OrderedDict((key, int(value)) for key, value in raw.items())
		remainder = batch_size - sum(counts.values())
		order = sorted(raw, key=lambda key: raw[key] - counts[key], reverse=True)
		for key in order[:remainder]:
			counts[key] += 1
		if batch_size >= len(counts):
			for key in list(counts.keys()):
				if counts[key] > 0:
					continue
				donor = max(counts, key=lambda item: counts[item])
				if counts[donor] <= 1:
					break
				counts[donor] -= 1
				counts[key] = 1
		return OrderedDict((key, value) for key, value in counts.items() if value > 0)

	def _sample_component(self, role, replay_key, count, device):
		buffer = self.current_buffer if role == "current" else self.replay_buffers[replay_key]
		return buffer.sample(device=device, batch_size=count)

	def _shuffle(self, obs, action, reward, task, labels):
		num_cols = int(action.shape[1])
		order = torch.randperm(num_cols, device=action.device)
		obs = _index_sequence_dim(obs, order)
		action = action[:, order].contiguous()
		reward = reward[:, order].contiguous()
		if task is not None:
			task = task[:, order].contiguous()
		order_cpu = order.detach().cpu().tolist()
		labels = [labels[index] for index in order_cpu]
		return obs, action, reward, task, labels

	def sample(self, device, batch_size=None):
		batch_size = int(batch_size if batch_size is not None else self.batch_size)
		min_current_episodes = self.min_current_episodes
		if self.sample_balance in {"condition_uniform", "uniform_condition"} and self.bootstrap_current_only:
			min_current_episodes = max(min_current_episodes, self.bootstrap_min_episodes_per_condition)
		if not self.replay_buffers or self.current_buffer.num_eps < min_current_episodes:
			batch = self.current_buffer.sample(device=device, batch_size=batch_size)
			label = self._current_label()
			self.last_batch_source_counts = {"current": batch_size}
			self.last_batch_source_distribution = {"current": 1.0}
			self.last_batch_task_counts = {self.current_task_id: batch_size}
			self.last_batch_assembly_counts = {label["assembly_id"]: batch_size}
			self.last_batch_template_counts = {label["template_id"]: batch_size}
			self.last_batch_condition_counts = {label["condition_id"]: batch_size}
			self.last_batch_task_hash_counts = _task_hash_counts(batch[3])
			return batch

		if self.sample_balance in {"condition_uniform", "uniform_condition"}:
			weights = self._condition_uniform_weights()
		else:
			weights = self._component_weights()
		counts = self._counts_from_weights(weights, batch_size)
		batches = []
		labels = []
		for (role, replay_key), count in counts.items():
			batches.append(self._sample_component(role, replay_key, count, device))
			label = self._current_label() if role == "current" else self._replay_label(role, replay_key)
			labels.extend([label] * count)
		obs = _cat_sequence_dim([batch[0] for batch in batches])
		action = torch.cat([batch[1] for batch in batches], dim=1).contiguous()
		reward = torch.cat([batch[2] for batch in batches], dim=1).contiguous()
		task = _cat_optional_tensors([batch[3] for batch in batches])
		obs, action, reward, task, labels = self._shuffle(obs, action, reward, task, labels)
		source_counts = Counter(label["role"] for label in labels)
		task_counts = Counter(label["task_id"] for label in labels)
		assembly_counts = Counter(label["assembly_id"] for label in labels)
		template_counts = Counter(label["template_id"] for label in labels)
		condition_counts = Counter(label["condition_id"] for label in labels)
		total = max(1, len(labels))
		self.last_batch_source_counts = dict(source_counts)
		self.last_batch_source_distribution = {
			key: float(value) / total
			for key, value in sorted(source_counts.items())
		}
		self.last_batch_task_counts = dict(task_counts)
		self.last_batch_assembly_counts = dict(assembly_counts)
		self.last_batch_template_counts = dict(template_counts)
		self.last_batch_condition_counts = dict(condition_counts)
		self.last_batch_task_hash_counts = _task_hash_counts(task)
		return obs, action, reward, task
