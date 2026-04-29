from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


def _to_cpu_contiguous(tensor: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
	value = tensor.detach().cpu().contiguous()
	if dtype is not None:
		value = value.to(dtype)
	return value


@dataclass
class OfflineDatasetStats:
	num_transitions: int
	num_episodes: int
	num_selected_episodes: int
	num_selected_transitions: int
	num_valid_starts: int
	horizon: int
	filter_mode: str
	obs_shape: tuple[int, ...]
	action_shape: tuple[int, ...]
	has_terminal_success: bool
	has_terminal_failure: bool
	has_task_ids: bool
	num_tasks: int


class OfflineSequenceDataset:
	"""
	Compact offline sequence sampler for Phase 1 canonical Newt training.

	Loads a compact transition dataset and samples fixed-length subsequences in a
	format compatible with `agent.update(buffer)`, i.e. returns:

	- obs:    (H+1, B, obs_dim)
	- action: (H,   B, act_dim)
	- reward: (H,   B, 1)
	- task:   (H, B)
	"""

	def __init__(
		self,
		path: str | Path,
		batch_size: int,
		horizon: int,
		filter_mode: str = "all",
		success_key: str = "episode_success_final",
		failure_key: str = "episode_failure_final",
		device: str = "cpu",
	):
		self._path = Path(path).expanduser().resolve()
		if not self._path.exists():
			raise FileNotFoundError(f"Offline dataset not found: {self._path}")
		if horizon < 1:
			raise ValueError(f"Expected horizon >= 1, got {horizon}")
		if batch_size < 1:
			raise ValueError(f"Expected batch_size >= 1, got {batch_size}")
		self._batch_size = int(batch_size)
		self._horizon = int(horizon)
		self._filter_mode = filter_mode
		self._success_key = success_key
		self._failure_key = failure_key
		self._device = torch.device(device)

		obj = torch.load(self._path, map_location="cpu", weights_only=False)
		if not hasattr(obj, "keys"):
			raise TypeError(f"Expected a TensorDict-like offline dataset, got {type(obj)}")

		required = ("obs", "next_obs", "action", "reward", "done", "episode")
		missing = [key for key in required if key not in obj.keys()]
		if missing:
			raise KeyError(f"Missing required offline dataset keys: {missing}")

		self.obs = _to_cpu_contiguous(obj["obs"], torch.float32)
		self.next_obs = _to_cpu_contiguous(obj["next_obs"], torch.float32)
		self.action = _to_cpu_contiguous(obj["action"], torch.float32)
		self.reward = _to_cpu_contiguous(obj["reward"], torch.float32)
		self.done = _to_cpu_contiguous(obj["done"], torch.bool)
		self.episode = _to_cpu_contiguous(obj["episode"], torch.int64).view(-1)
		self.terminated = _to_cpu_contiguous(obj["terminated"], torch.bool) if "terminated" in obj.keys() else None
		self.truncated = _to_cpu_contiguous(obj["truncated"], torch.bool) if "truncated" in obj.keys() else None
		self.step_id = _to_cpu_contiguous(obj["step_id"], torch.int64).view(-1) if "step_id" in obj.keys() else None
		self.task = _to_cpu_contiguous(obj["task"], torch.int64).view(-1) if "task" in obj.keys() else None

		self.success_final = self._resolve_episode_label(obj, success_key, fallback_key="success_episode")
		self.failure_final = self._resolve_episode_label(obj, failure_key, fallback_key=None)
		self.terminal_success = _to_cpu_contiguous(obj["terminal_success"], torch.bool).view(-1) if "terminal_success" in obj.keys() else None
		self.terminal_failure = _to_cpu_contiguous(obj["terminal_failure"], torch.bool).view(-1) if "terminal_failure" in obj.keys() else None
		self.episode_return_final = _to_cpu_contiguous(obj["episode_return_final"], torch.float32).view(-1) if "episode_return_final" in obj.keys() else None
		self.episode_return_running = _to_cpu_contiguous(obj["episode_return_running"], torch.float32).view(-1) if "episode_return_running" in obj.keys() else None

		self._episode_indices = self._build_episode_indices()
		self._episode_task_ids = self._build_episode_task_ids()
		self._selected_episode_ids = self._select_episodes(filter_mode)
		self._valid_starts = self._build_valid_starts()
		if len(self._valid_starts) == 0:
			raise ValueError(
				f"No valid subsequences found for horizon={self._horizon} with filter_mode='{self._filter_mode}'."
			)
		self._stats = OfflineDatasetStats(
			num_transitions=int(self.obs.shape[0]),
			num_episodes=len(self._episode_indices),
			num_selected_episodes=len(self._selected_episode_ids),
			num_selected_transitions=int(sum(len(self._episode_indices[ep]) for ep in self._selected_episode_ids)),
			num_valid_starts=int(len(self._valid_starts)),
			horizon=self._horizon,
			filter_mode=self._filter_mode,
			obs_shape=tuple(self.obs.shape[1:]),
			action_shape=tuple(self.action.shape[1:]),
			has_terminal_success=self.terminal_success is not None,
			has_terminal_failure=self.terminal_failure is not None,
			has_task_ids=self.task is not None,
			num_tasks=len(set(self._episode_task_ids.values())),
		)

	def _resolve_episode_label(self, obj, preferred_key: str, fallback_key: Optional[str]) -> Optional[torch.Tensor]:
		if preferred_key in obj.keys():
			return _to_cpu_contiguous(obj[preferred_key], torch.float32).view(-1)
		if fallback_key is not None and fallback_key in obj.keys():
			return _to_cpu_contiguous(obj[fallback_key], torch.float32).view(-1)
		return None

	def _build_episode_indices(self) -> dict[int, torch.Tensor]:
		episode_ids = self.episode.tolist()
		index_map: dict[int, list[int]] = {}
		for idx, episode_id in enumerate(episode_ids):
			index_map.setdefault(int(episode_id), []).append(idx)
		episode_indices: dict[int, torch.Tensor] = {}
		for episode_id, indices in index_map.items():
			idx_tensor = torch.tensor(indices, dtype=torch.int64)
			if self.step_id is not None:
				order = torch.argsort(self.step_id[idx_tensor], stable=True)
				idx_tensor = idx_tensor[order]
			episode_indices[episode_id] = idx_tensor
		return episode_indices

	def _build_episode_task_ids(self) -> dict[int, int]:
		if self.task is None:
			return {episode_id: 0 for episode_id in self._episode_indices.keys()}
		episode_task_ids = {}
		for episode_id, indices in self._episode_indices.items():
			task_ids = torch.unique(self.task[indices], sorted=True)
			if task_ids.numel() != 1:
				raise ValueError(
					f"Episode {episode_id} has multiple task ids: {task_ids.tolist()}. "
					"Expected exactly one task id per episode."
				)
			episode_task_ids[episode_id] = int(task_ids.item())
		return episode_task_ids

	def _select_episodes(self, filter_mode: str) -> list[int]:
		if filter_mode not in ("all", "success_only", "failure_only"):
			raise ValueError(f"Invalid filter_mode '{filter_mode}'. Expected one of: all, success_only, failure_only.")
		episode_ids = sorted(self._episode_indices.keys())
		if filter_mode == "all":
			return episode_ids
		if filter_mode == "success_only":
			if self.success_final is None:
				raise KeyError("success_only requested but no success label exists in the offline dataset.")
			return [ep for ep in episode_ids if self._episode_label(self.success_final, ep) > 0.5]
		if self.failure_final is not None:
			return [ep for ep in episode_ids if self._episode_label(self.failure_final, ep) > 0.5]
		if self.success_final is not None:
			return [ep for ep in episode_ids if self._episode_label(self.success_final, ep) <= 0.5]
		raise KeyError("failure_only requested but no failure or success episode label exists in the offline dataset.")

	def _episode_label(self, tensor: torch.Tensor, episode_id: int) -> float:
		indices = self._episode_indices[episode_id]
		return float(tensor[indices[-1]].item())

	def _build_valid_starts(self) -> torch.Tensor:
		starts = []
		for episode_id in self._selected_episode_ids:
			length = len(self._episode_indices[episode_id])
			if length < self._horizon:
				continue
			max_start = length - self._horizon
			starts.extend((episode_id, offset) for offset in range(max_start + 1))
		return starts

	@property
	def stats(self) -> OfflineDatasetStats:
		return self._stats

	def __len__(self) -> int:
		return len(self._valid_starts)

	def sample(self, device: Optional[str | torch.device] = None):
		device = torch.device(device) if device is not None else self._device
		indices = torch.randint(0, len(self._valid_starts), (self._batch_size,))

		obs_batch = []
		action_batch = []
		reward_batch = []
		task_batch = []
		for idx in indices.tolist():
			episode_id, start_offset = self._valid_starts[idx]
			seq_indices = self._episode_indices[episode_id][start_offset : start_offset + self._horizon]
			last_index = int(seq_indices[-1].item())
			obs_seq = torch.cat(
				[
					self.obs[seq_indices],
					self.next_obs[last_index : last_index + 1],
				],
				dim=0,
			)
			action_seq = self.action[seq_indices]
			reward_seq = self.reward[seq_indices]
			if reward_seq.ndim == 1:
				reward_seq = reward_seq.unsqueeze(-1)
			obs_batch.append(obs_seq)
			action_batch.append(action_seq)
			reward_batch.append(reward_seq)
			task_batch.append(
				torch.full((self._horizon,), self._episode_task_ids[episode_id], dtype=torch.int64)
			)

		obs = torch.stack(obs_batch, dim=1).to(device, non_blocking=True)
		action = torch.stack(action_batch, dim=1).to(device, non_blocking=True)
		reward = torch.stack(reward_batch, dim=1).to(device, non_blocking=True)
		task = torch.stack(task_batch, dim=1).to(device, non_blocking=True)
		return obs.contiguous(), action.contiguous(), reward.contiguous(), task
