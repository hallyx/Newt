from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
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


def _print_warning(message: str):
	if colored is not None:
		print(colored(message, "yellow", attrs=["bold"]))
	else:
		print(message)


def get_active_tasks(global_step: int, task_ids: list[str], stage_steps: int, mode: str):
	"""Return the active task ids for the current continual multitask stage."""
	if not task_ids:
		raise ValueError("multitask task_ids must be non-empty.")
	mode = str(mode).strip().lower()
	if mode == "all_at_once":
		return list(task_ids)
	if mode != "progressive":
		raise ValueError("multitask_curriculum_mode must be one of: all_at_once, progressive.")
	stage_steps = max(1, int(stage_steps))
	stage_idx = int(global_step) // stage_steps
	num_active = min(len(task_ids), stage_idx + 1)
	return list(task_ids[:num_active])


@dataclass
class MultiTaskBatch:
	"""Tuple-compatible batch with task ids retained for logging and audits."""

	obs: Any
	action: torch.Tensor
	reward: torch.Tensor
	task: torch.Tensor
	task_id: list[str]

	def __iter__(self):
		yield self.obs
		yield self.action
		yield self.reward
		yield self.task


class OfflineTaskReplayBuffer:
	"""Adapter that exposes one task from an OfflineSequenceDataset as a replay buffer."""

	def __init__(self, dataset, dataset_task_id: int, task_vec_6, task_label: Optional[str] = None):
		self.dataset = dataset
		self.dataset_task_id = int(dataset_task_id)
		self.task_label = str(task_label if task_label is not None else dataset_task_id)
		task_vec = torch.as_tensor(task_vec_6, dtype=torch.float32).reshape(-1)
		if int(task_vec.numel()) != 6:
			raise ValueError(f"Expected task_vec_6 dim 6 for task {self.task_label}, got {int(task_vec.numel())}.")
		self.task_vec_6 = task_vec.contiguous()

	def __len__(self):
		if hasattr(self.dataset, "valid_start_count"):
			return int(self.dataset.valid_start_count(self.dataset_task_id))
		return 0

	def sample(self, batch_size: int, device: Optional[str | torch.device] = None):
		obs, action, reward, _task_ids = self.dataset.sample_task(
			self.dataset_task_id,
			batch_size=int(batch_size),
			device=device,
		)
		horizon, batch = action.shape[:2]
		task_vec = self.task_vec_6.to(action.device, non_blocking=True).view(1, 1, 6)
		task = task_vec.expand(horizon, batch, 6).contiguous()
		return MultiTaskBatch(
			obs=obs,
			action=action,
			reward=reward,
			task=task,
			task_id=[self.task_label] * batch,
		)


class MultiTaskReplayPool:
	"""
	Replay pool that samples fixed-size batches from all active tasks.

	`task_id` is retained for sampling and logging only. The model input is the
	6D axial task vector stored in `batch.task`.
	"""

	def __init__(
		self,
		task_ids,
		anchor_task_id,
		sampling_mode="balanced",
		task_sampling_weights=None,
		anchor_min_ratio=0.2,
		new_task_min_ratio=0.2,
		hard_case_ratio=0.0,
		batch_size: Optional[int] = None,
	):
		self.task_ids = [str(task_id) for task_id in task_ids]
		self.anchor_task_id = str(anchor_task_id)
		self.sampling_mode = str(sampling_mode or "balanced").strip().lower()
		if self.sampling_mode not in {"balanced", "weighted", "proportional"}:
			raise ValueError("multitask_sampling_mode must be one of: balanced, weighted, proportional.")
		self.task_sampling_weights = {
			str(key): float(value)
			for key, value in (task_sampling_weights or {}).items()
		}
		self.anchor_min_ratio = max(0.0, float(anchor_min_ratio))
		self.new_task_min_ratio = max(0.0, float(new_task_min_ratio))
		self.hard_case_ratio = max(0.0, min(1.0, float(hard_case_ratio)))
		self.batch_size = int(batch_size) if batch_size is not None else None
		self.buffers = {}
		self.hard_case_buffers = {}
		self.active_tasks = []
		self.current_new_task_id = None
		self.last_batch_task_counts = {}
		self.last_batch_task_distribution = {}
		self.last_task_vec_stats = {}
		self.last_sample_task_ids = []
		self._warned = set()

	def _warn_once(self, key, message):
		if key in self._warned:
			return
		self._warned.add(key)
		_print_warning(message)

	def add_task_buffer(self, task_id: str, replay_buffer, hard_case_buffer=None):
		task_id = str(task_id)
		if len(replay_buffer) <= 0:
			self._warn_once(
				("empty-buffer", task_id),
				f"[multitask-replay-warning] task_id={task_id} has no valid samples at registration time.",
			)
		self.buffers[task_id] = replay_buffer
		if hard_case_buffer is not None:
			if len(hard_case_buffer) <= 0:
				self._warn_once(
					("empty-hard-buffer", task_id),
					f"[multitask-replay-warning] task_id={task_id} hard-case buffer is empty; normal replay will be used.",
				)
			else:
				self.hard_case_buffers[task_id] = hard_case_buffer

	def set_active_tasks(self, task_ids: list[str], current_new_task_id: str | None = None):
		active = [str(task_id) for task_id in task_ids]
		missing = [task_id for task_id in active if task_id not in self.buffers or len(self.buffers[task_id]) <= 0]
		for task_id in missing:
			self._warn_once(
				("missing-active", task_id),
				f"[multitask-replay-warning] active task_id={task_id} has no data; falling back to available tasks.",
			)
		active = [task_id for task_id in active if task_id in self.buffers and len(self.buffers[task_id]) > 0]
		if not active:
			raise ValueError("No active multitask replay buffers have valid data.")
		self.active_tasks = active
		self.current_new_task_id = str(current_new_task_id) if current_new_task_id is not None else None

	def _base_probabilities(self, active_tasks, *, hard: bool = False):
		if self.sampling_mode == "balanced":
			return {task_id: 1.0 / len(active_tasks) for task_id in active_tasks}
		if self.sampling_mode == "weighted":
			values = {task_id: max(0.0, self.task_sampling_weights.get(task_id, 0.0)) for task_id in active_tasks}
			total = sum(values.values())
			if total <= 0:
				self._warn_once(
					"bad-weighted-probs",
					"[multitask-replay-warning] weighted sampling requested but active weights are empty; using balanced sampling.",
				)
				return {task_id: 1.0 / len(active_tasks) for task_id in active_tasks}
			return {task_id: value / total for task_id, value in values.items()}
		buffer_map = self.hard_case_buffers if hard else self.buffers
		values = {task_id: max(0, len(buffer_map.get(task_id, []))) for task_id in active_tasks}
		total = sum(values.values())
		if total <= 0:
			return {task_id: 1.0 / len(active_tasks) for task_id in active_tasks}
		return {task_id: float(value) / total for task_id, value in values.items()}

	def _apply_min_ratios(self, probs):
		if not probs:
			return probs
		minimums = {}
		if self.anchor_task_id in probs and self.anchor_min_ratio > 0:
			minimums[self.anchor_task_id] = max(minimums.get(self.anchor_task_id, 0.0), self.anchor_min_ratio)
		if (
			self.current_new_task_id in probs and
			self.current_new_task_id is not None and
			self.new_task_min_ratio > 0
		):
			minimums[self.current_new_task_id] = max(
				minimums.get(self.current_new_task_id, 0.0),
				self.new_task_min_ratio,
			)
		min_total = sum(minimums.values())
		if min_total >= 1.0:
			return {
				task_id: minimums.get(task_id, 0.0) / min_total
				for task_id in probs
			}
		remaining_tasks = [task_id for task_id in probs if task_id not in minimums]
		remaining_mass = 1.0 - min_total
		remaining_base = sum(probs[task_id] for task_id in remaining_tasks)
		out = {task_id: minimums.get(task_id, 0.0) for task_id in probs}
		if remaining_tasks and remaining_base > 0:
			for task_id in remaining_tasks:
				out[task_id] = remaining_mass * probs[task_id] / remaining_base
		elif remaining_tasks:
			for task_id in remaining_tasks:
				out[task_id] = remaining_mass / len(remaining_tasks)
		else:
			normalizer = sum(out.values())
			out = {task_id: value / normalizer for task_id, value in out.items()}
		return out

	def _counts_from_probs(self, probs, batch_size):
		batch_size = int(batch_size)
		raw = {task_id: probs[task_id] * batch_size for task_id in probs}
		counts = {task_id: int(raw[task_id]) for task_id in raw}
		remainder = batch_size - sum(counts.values())
		fractional = sorted(raw, key=lambda task_id: raw[task_id] - counts[task_id], reverse=True)
		for task_id in fractional[:remainder]:
			counts[task_id] += 1
		if batch_size >= len(counts):
			zero_tasks = [task_id for task_id, count in counts.items() if count == 0 and probs[task_id] > 0]
			for task_id in zero_tasks:
				donor = max(counts, key=lambda key: counts[key])
				if counts[donor] <= 1:
					break
				counts[donor] -= 1
				counts[task_id] = 1
		return {task_id: count for task_id, count in counts.items() if count > 0}

	def _sample_counts(self, batch_size, *, hard: bool = False):
		active = [task_id for task_id in self.active_tasks]
		if hard:
			active = [task_id for task_id in active if task_id in self.hard_case_buffers and len(self.hard_case_buffers[task_id]) > 0]
		if not active:
			return {}
		probs = self._base_probabilities(active, hard=hard)
		if not hard:
			probs = self._apply_min_ratios(probs)
		return self._counts_from_probs(probs, batch_size)

	def _concat_batches(self, batches):
		if not batches:
			raise ValueError("Cannot concatenate an empty list of multitask batches.")
		obs = _cat_sequence_dim([batch.obs for batch in batches])
		action = torch.cat([batch.action for batch in batches], dim=1)
		reward = torch.cat([batch.reward for batch in batches], dim=1)
		task = torch.cat([batch.task for batch in batches], dim=1)
		task_ids = [task_id for batch in batches for task_id in batch.task_id]
		return MultiTaskBatch(obs=obs, action=action, reward=reward, task=task, task_id=task_ids)

	def _sample_from(self, counts, *, hard: bool, device):
		buffer_map = self.hard_case_buffers if hard else self.buffers
		batches = []
		for task_id, count in counts.items():
			buffer = buffer_map[task_id]
			if count > len(buffer):
				self._warn_once(
					("undersized", hard, task_id),
					f"[multitask-replay-warning] task_id={task_id} requested {count} samples "
					f"from {len(buffer)} valid starts; sampling with replacement.",
				)
			batches.append(buffer.sample(count, device=device))
		return batches

	def _ensure_min_count(self, counts, task_id, min_count, budget):
		if task_id is None or task_id not in self.active_tasks or task_id not in self.buffers:
			return counts
		min_count = int(min_count)
		if min_count <= 0:
			return counts
		if min_count > int(budget):
			self._warn_once(
				("min-ratio-budget", task_id, budget),
				f"[multitask-replay-warning] requested minimum count {min_count} for task_id={task_id} "
				f"but normal replay budget is only {budget}; hard-case ratio may reduce the final ratio.",
			)
			min_count = int(budget)
		current = int(counts.get(task_id, 0))
		if current >= min_count:
			return counts
		counts[task_id] = current
		needed = min_count - current
		for donor in sorted(counts, key=lambda key: counts[key], reverse=True):
			if donor == task_id:
				continue
			while needed > 0 and counts.get(donor, 0) > 1:
				counts[donor] -= 1
				counts[task_id] += 1
				needed -= 1
			if needed <= 0:
				break
		if needed > 0:
			counts[task_id] += needed
		total = sum(counts.values())
		while total > int(budget):
			donor = max((key for key in counts if key != task_id), key=lambda key: counts[key], default=None)
			if donor is None or counts[donor] <= 0:
				break
			counts[donor] -= 1
			total -= 1
		return {key: value for key, value in counts.items() if value > 0}

	def _shuffle_batch(self, batch: MultiTaskBatch):
		num_cols = int(batch.action.shape[1])
		order = torch.randperm(num_cols, device=batch.action.device)
		obs = _index_sequence_dim(batch.obs, order)
		action = batch.action[:, order].contiguous()
		reward = batch.reward[:, order].contiguous()
		task = batch.task[:, order].contiguous()
		order_cpu = order.detach().cpu().tolist()
		task_ids = [batch.task_id[index] for index in order_cpu]
		return MultiTaskBatch(obs=obs, action=action, reward=reward, task=task, task_id=task_ids)

	def _update_last_stats(self, batch: MultiTaskBatch):
		counts = Counter(batch.task_id)
		total = max(1, len(batch.task_id))
		self.last_sample_task_ids = list(batch.task_id)
		self.last_batch_task_counts = dict(counts)
		self.last_batch_task_distribution = {
			task_id: float(count) / total
			for task_id, count in sorted(counts.items())
		}
		if self.anchor_task_id in self.active_tasks and counts.get(self.anchor_task_id, 0) <= 0:
			self._warn_once(
				("missing-anchor-batch", total),
				f"[multitask-replay-warning] sampled batch has no anchor task {self.anchor_task_id}.",
			)
		stats = {}
		task_tensor = batch.task.detach()
		for task_id in sorted(counts):
			cols = [index for index, label in enumerate(batch.task_id) if label == task_id]
			values = task_tensor[:, cols].reshape(-1, task_tensor.shape[-1]).float()
			stats[task_id] = {
				"mean": values.mean(dim=0).cpu().tolist(),
				"std": values.std(dim=0, unbiased=False).cpu().tolist(),
			}
		self.last_task_vec_stats = stats

	def sample(self, batch_size: Optional[int] = None, device: Optional[str | torch.device] = None):
		if not self.active_tasks:
			self.set_active_tasks(self.task_ids, current_new_task_id=self.task_ids[-1] if self.task_ids else None)
		batch_size = int(batch_size if batch_size is not None else self.batch_size)
		if batch_size <= 0:
			raise ValueError(f"Expected positive batch_size, got {batch_size}.")
		hard_count = int(round(batch_size * self.hard_case_ratio))
		normal_count = batch_size - hard_count
		batches = []
		if normal_count > 0:
			normal_counts = self._sample_counts(normal_count, hard=False)
			anchor_min_count = int(torch.ceil(torch.tensor(batch_size * self.anchor_min_ratio)).item())
			new_min_count = int(torch.ceil(torch.tensor(batch_size * self.new_task_min_ratio)).item())
			normal_counts = self._ensure_min_count(normal_counts, self.anchor_task_id, anchor_min_count, normal_count)
			normal_counts = self._ensure_min_count(normal_counts, self.current_new_task_id, new_min_count, normal_count)
			batches.extend(self._sample_from(normal_counts, hard=False, device=device))
		if hard_count > 0:
			hard_counts = self._sample_counts(hard_count, hard=True)
			if hard_counts:
				batches.extend(self._sample_from(hard_counts, hard=True, device=device))
			else:
				self._warn_once(
					"missing-hard-buffers",
					"[multitask-replay-warning] hard_case_ratio > 0 but no hard-case samples are available; using normal replay.",
				)
				batches.extend(self._sample_from(self._sample_counts(hard_count, hard=False), hard=False, device=device))
		batch = self._shuffle_batch(self._concat_batches(batches))
		if int(batch.task.shape[-1]) != 6:
			raise ValueError(f"Expected task_vec_6 dim 6, got {int(batch.task.shape[-1])}.")
		self._update_last_stats(batch)
		return batch


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
