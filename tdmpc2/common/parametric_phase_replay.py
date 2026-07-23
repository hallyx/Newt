"""Phase-balanced online replay for single-family parametric SRSA training.

Each episode is assigned to the nearest configured clearance/depth anchor from
its *runtime* task vector.  Horizon starts are indexed when the episode enters
the buffer, then every update is balanced over anchor x insertion phase cells.
The buffer does not synthesize or rewrite task vectors.
"""

from __future__ import annotations

from collections import OrderedDict

import torch

from common.buffer import Buffer


PHASES = ("pre_contact", "contact", "insertion")


def _phase_labels(obs: torch.Tensor) -> torch.Tensor:
	"""Match the Phase 3/4 TCP-z phase heuristic for one complete episode."""
	obs = obs.detach().float()
	n = int(obs.shape[0])
	phase = torch.empty((n,), dtype=torch.long, device=obs.device)
	z = obs[:, 2]
	downward = (z[0] - z).clamp_min(0.0)
	total = float(downward.max().item())
	if total >= 0.006:
		contact_hits = torch.nonzero(downward >= 0.25 * total, as_tuple=False).reshape(-1)
		insert_hits = torch.nonzero(downward >= 0.65 * total, as_tuple=False).reshape(-1)
		contact_at = int(contact_hits[0].item()) if contact_hits.numel() else max(1, int(0.27 * n))
		insert_at = int(insert_hits[0].item()) if insert_hits.numel() else max(contact_at + 1, int(0.60 * n))
	else:
		contact_at = max(1, int(0.27 * n))
		insert_at = max(contact_at + 1, int(0.60 * n))
	contact_at = min(max(contact_at, 1), max(1, n - 2))
	insert_at = min(max(insert_at, contact_at + 1), max(contact_at + 1, n - 1))
	phase[:contact_at] = 0
	phase[contact_at:insert_at] = 1
	phase[insert_at:] = 2
	return phase


class ParametricPhaseReplayBuffer(Buffer):
	"""Online Buffer with exact anchor x phase balance and rotating remainder."""

	def __init__(self, *args, anchor_centers, anchor_names=None, seed=4201, **kwargs):
		super().__init__(*args, **kwargs)
		centers = torch.as_tensor(anchor_centers, dtype=torch.float32).reshape(-1, 2)
		if centers.shape[0] <= 0 or not torch.isfinite(centers).all():
			raise ValueError("anchor_centers must contain finite [clearance_abs_norm, depth_abs_norm] rows.")
		self.anchor_centers = centers.contiguous()
		self.anchor_names = tuple(anchor_names or [f"anchor_{index}" for index in range(centers.shape[0])])
		if len(self.anchor_names) != int(centers.shape[0]):
			raise ValueError("anchor_names and anchor_centers must have equal length.")
		self._cells = [(anchor_id, phase_id) for anchor_id in range(len(self.anchor_names)) for phase_id in range(3)]
		self._cell_starts = {cell: [] for cell in self._cells}
		self._generator = torch.Generator().manual_seed(int(seed))
		self._sample_index = 0
		self.last_batch_anchor_counts = None
		self.last_batch_phase_counts = None
		self.last_batch_cell_counts = None
		self.last_batch_task_vec_std = None
		self.episode_contract = {
			"episodes_indexed": 0,
			"episodes_task_vec_changed_within_episode": 0,
			"max_within_episode_task_vec_linf": 0.0,
		}

	def _anchor_id(self, task_vec: torch.Tensor) -> int:
		point = task_vec.detach().float()[[2, 4]].cpu()
		# Normalize each axis so clearance does not dominate depth (or vice versa).
		scale = self.anchor_centers.abs().mean(dim=0).clamp_min(1.0e-6)
		distance = torch.linalg.vector_norm((self.anchor_centers - point) / scale, dim=-1)
		return int(torch.argmin(distance).item())

	def add(self, td, world_size=1, rank=0):
		if int(td.shape[0]) != 1:
			raise ValueError("ParametricPhaseReplayBuffer.add expects one complete episode.")
		episode = td[0]
		length = int(episode.shape[0])
		if len(self._buffer) + length > int(self.capacity):
			raise RuntimeError(
				"ParametricPhaseReplayBuffer forbids circular overwrite because indexed horizon starts must remain valid; "
				f"capacity={self.capacity}, current={len(self._buffer)}, incoming={length}."
			)
		task = episode["task"].detach().float().reshape(length, -1)
		if task.shape[-1] != 6 or not torch.isfinite(task).all():
			raise RuntimeError(f"Expected finite task_vec_6 for the whole episode, got {tuple(task.shape)}.")
		within_linf = float((task - task[0]).abs().max().item())
		self.episode_contract["episodes_indexed"] += 1
		self.episode_contract["max_within_episode_task_vec_linf"] = max(
			float(self.episode_contract["max_within_episode_task_vec_linf"]), within_linf,
		)
		if within_linf > 1.0e-7:
			self.episode_contract["episodes_task_vec_changed_within_episode"] += 1
			raise RuntimeError(
				"Runtime task parameters changed inside an episode; Phase 4.2 requires one fixed parameter per episode. "
				f"max task_vec L-inf drift={within_linf:.6g}."
			)

		anchor_id = self._anchor_id(task[0])
		phase = _phase_labels(episode["obs"])
		base = int(len(self._buffer))
		for local_start in range(max(0, length - self._horizon)):
			first_supervised = local_start + 1
			phase_id = int(phase[first_supervised].item())
			self._cell_starts[(anchor_id, phase_id)].append(base + local_start)
		return super().add(td, world_size=world_size, rank=rank)

	def _counts(self, batch_size: int):
		base, remainder = divmod(int(batch_size), len(self._cells))
		counts = [base] * len(self._cells)
		for offset in range(remainder):
			counts[(self._sample_index + offset) % len(self._cells)] += 1
		return counts

	def sample(self, device, batch_size=None):
		batch_size = int(batch_size if batch_size is not None else self._batch_size)
		if batch_size < len(self._cells):
			raise ValueError(
				f"batch_size={batch_size} cannot cover all {len(self._cells)} anchor x phase cells."
			)
		starts = []
		cell_counts = OrderedDict()
		for (anchor_id, phase_id), count in zip(self._cells, self._counts(batch_size)):
			pool = self._cell_starts[(anchor_id, phase_id)]
			if not pool:
				indexed = {
					f"{self.anchor_names[a]}/{PHASES[p]}": len(self._cell_starts[(a, p)])
					for a, p in self._cells
				}
				raise RuntimeError(
					"Parametric balanced replay cell is empty after seeding: "
					f"{self.anchor_names[anchor_id]}/{PHASES[phase_id]}; indexed_starts={indexed}."
				)
			choice = torch.randint(len(pool), (count,), generator=self._generator)
			starts.extend(pool[int(index)] for index in choice.tolist())
			cell_counts[f"{self.anchor_names[anchor_id]}/{PHASES[phase_id]}"] = int(count)

		order = torch.randperm(batch_size, generator=self._generator)
		start_tensor = torch.as_tensor(starts, dtype=torch.long)[order]
		indices = start_tensor[:, None] + torch.arange(self._horizon + 1, dtype=torch.long)[None, :]
		storage_indices = indices.to(self._storage_device) if self._storage_device.type == "cuda" else indices
		td = self._storage[storage_indices].permute(1, 0)
		batch = self._prepare_batch(td, device)
		self._sample_index += 1
		self.last_batch_cell_counts = cell_counts
		self.last_batch_anchor_counts = OrderedDict(
			(name, sum(value for key, value in cell_counts.items() if key.startswith(name + "/")))
			for name in self.anchor_names
		)
		self.last_batch_phase_counts = OrderedDict(
			(phase_name, sum(value for key, value in cell_counts.items() if key.endswith("/" + phase_name)))
			for phase_name in PHASES
		)
		task = batch[3]
		self.last_batch_task_vec_std = [float(value) for value in task.detach().float().reshape(-1, 6).std(dim=0, unbiased=False).cpu().tolist()]
		return batch

	def _snapshot_metadata(self, extra_metadata=None):
		metadata = super()._snapshot_metadata(extra_metadata)
		metadata.update({
			"parametric_phase_replay": True,
			"parametric_anchor_names": list(self.anchor_names),
			"parametric_anchor_centers": self.anchor_centers.tolist(),
			"parametric_episode_contract": dict(self.episode_contract),
			"parametric_indexed_starts_by_cell": {
				f"{self.anchor_names[anchor_id]}/{PHASES[phase_id]}": len(self._cell_starts[(anchor_id, phase_id)])
				for anchor_id, phase_id in self._cells
			},
		})
		return metadata
