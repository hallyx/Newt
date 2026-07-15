from copy import deepcopy
import hashlib
from pathlib import Path

import torch
from termcolor import colored
from tensordict.tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage, TensorStorage
from torchrl.data.replay_buffers.samplers import SliceSampler


class Buffer():
	"""
	Replay buffer for Newt training. Based on torchrl.
	Uses CUDA memory if available, and CPU memory otherwise.
	"""

	def __init__(
			self,
			capacity: int = 1_000_000,
			batch_size: int = 1024,
			horizon: int = 3,
			multiproc: bool = False,
			cache_values: bool = False,
			storage_device: str = 'cuda:0',
			prefetch='auto',
	):
		self.set_storage_device(storage_device)
		self._multiproc = multiproc
		self._sampler_use_gpu = self._storage_device.type != 'cuda'
		self._prefetch = None if (self._multiproc or self._storage_device.type == 'cuda') else 8
		if prefetch != 'auto':
			self._prefetch = prefetch
		self._capacity = capacity
		self._batch_size = batch_size
		self._sample_size = batch_size * (horizon + 1)
		self._horizon = horizon
		self._sampler = SliceSampler(
			num_slices=batch_size,
			end_key=None,
			traj_key='episode',
			truncated_key=None,
			strict_length=True,
			cache_values=cache_values,
			use_gpu=self._sampler_use_gpu,
			compile=False,
		)
		self._storage = LazyTensorStorage(
			self._capacity,
			device=self._storage_device,
			shared_init=self._multiproc,
		)
		self._buffer = ReplayBuffer(
			storage=self._storage,
			sampler=self._sampler,
			pin_memory=False,
			prefetch=self._prefetch,
			batch_size=self._sample_size,
			shared=self._multiproc,
		)
		self._num_eps = 0
		self._num_demos = 0
		self.metadata = {}

	def __len__(self):
		"""Return the number of stored transitions."""
		return len(self._buffer)

	@property
	def capacity(self):
		"""Return the capacity of the buffer."""
		return self._capacity

	@property
	def num_eps(self):
		"""Return the number of episodes in the buffer."""
		return self._num_eps

	@property
	def num_transitions(self):
		"""Return the number of stored transitions."""
		return len(self)
	
	def set_storage_device(self, device):
		"""
		Set the storage device for the buffer.
		"""
		if isinstance(device, str):
			device = torch.device(device)
		if hasattr(self, '_storage_device') and self._storage_device == device:
			return
		elif hasattr(self, '_storage_device'):
			print(f'[{self.__class__.__name__}] Changing storage device from {self._storage_device} to {device}.')
		else:
			print(f'[{self.__class__.__name__}] Setting storage device to {device}.')
		self._storage_device = device

	def print_requirements(self, tds):
		"""Use the first episode to estimate storage requirements."""
		print(f'[{self.__class__.__name__}] Buffer capacity: {self._capacity:,}')
		bytes_per_step = sum([
				(v.numel()*v.element_size() if not isinstance(v, TensorDict) \
				else sum([x.numel()*x.element_size() for x in v.values()])) \
			for v in tds.values()
		]) / len(tds)
		total_bytes = bytes_per_step*self._capacity
		print(f'[{self.__class__.__name__}] Storage required: {total_bytes/1e9:.2f} GB')
		print(f'[{self.__class__.__name__}] Using {self._storage_device} memory for storage.')
		print(f'[{self.__class__.__name__}] SliceSampler use_gpu={self._sampler_use_gpu}, prefetch={self._prefetch}.')

	def _valid_storage(self):
		if len(self._buffer) <= 0:
			raise RuntimeError("Buffer is empty, nothing to save.")
		return self._buffer[:len(self._buffer)].detach().cpu()

	def _snapshot_metadata(self, extra_metadata=None):
		metadata = dict(extra_metadata or {})
		metadata.setdefault("format", "newt_buffer_snapshot_v1")
		metadata.setdefault("num_episodes", int(self._num_eps + self._num_demos))
		metadata.setdefault("num_transitions", int(len(self._buffer)))
		metadata.setdefault("capacity", int(self._capacity))
		metadata.setdefault("batch_size", int(self._batch_size))
		metadata.setdefault("horizon", int(self._horizon))
		try:
			td = self._valid_storage()
			if "obs" in td.keys():
				obs = td.get("obs")
				metadata.setdefault("obs_shape", list(obs.shape[1:]))
			if "action" in td.keys():
				metadata.setdefault("action_shape", list(td.get("action").shape[1:]))
			if "task" in td.keys():
				task = td.get("task")
				metadata.setdefault("task_shape", list(task.shape[1:]))
				if task.is_floating_point() and task.ndim >= 2 and int(task.shape[-1]) <= 64:
					task_flat = task.reshape(-1, task.shape[-1]).detach().cpu().float()
					task_unique = torch.unique(task_flat, dim=0)
					metadata.setdefault("task_vec_unique", int(task_unique.shape[0]))
					hashes = []
					values = []
					for vec in task_unique[:32]:
						rounded = [round(float(item), 8) for item in vec.tolist()]
						digest = hashlib.sha1(",".join(f"{item:.8g}" for item in rounded).encode("utf-8")).hexdigest()[:12]
						hashes.append(digest)
						if len(values) < 8:
							values.append(rounded)
					metadata.setdefault("task_vec_hashes", hashes)
					if values:
						metadata.setdefault("task_vec_unique_values", values)
		except Exception:
			pass
		return metadata

	def save(self, path, metadata=None):
		"""
		Save the buffer to disk.
		"""
		path = Path(path).expanduser()
		path.parent.mkdir(parents=True, exist_ok=True)
		payload = {
			"format": "newt_buffer_snapshot_v1",
			"metadata": self._snapshot_metadata(metadata),
			"data": self._valid_storage(),
		}
		torch.save(payload, path)
		print(colored(f"Saved replay buffer snapshot: {path}", "green", attrs=["bold"]))
		return path

	def save_current(self, path, metadata=None):
		"""Save the current writable replay buffer."""
		return self.save(path, metadata=metadata)

	def load_snapshot_data(self, tds, metadata=None, max_episodes=None):
		"""Load stored transitions into an empty buffer."""
		assert len(self._buffer) == 0, "Expected an empty buffer when loading a snapshot."
		if max_episodes is not None:
			max_episodes = int(max_episodes)
			if max_episodes > 0 and "episode" in tds.keys():
				episode_ids = torch.unique(tds["episode"].detach().cpu(), sorted=True)
				episode_ids = episode_ids[-max_episodes:]
				mask = torch.isin(tds["episode"].detach().cpu(), episode_ids)
				tds = tds[mask]
		tds = tds.to(self._storage_device)
		self._buffer.extend(tds)
		if metadata and metadata.get("num_episodes", None) is not None and max_episodes is None:
			self._num_eps = int(metadata["num_episodes"])
		elif "episode" in tds.keys() and len(tds) > 0:
			self._num_eps = int(torch.unique(tds["episode"].detach().cpu()).numel())
		else:
			self._num_eps = 0
		print(colored(
			f"Loaded replay buffer snapshot with {self._num_eps} episodes and {len(self._buffer)} transitions.",
			"green",
			attrs=["bold"],
		))
		return self._num_eps

	@classmethod
	def load(
			cls,
			path,
			cfg=None,
			capacity=None,
			batch_size=None,
			horizon=None,
			multiproc=False,
			cache_values=False,
			storage_device=None,
			max_episodes=None,
			prefetch=None,
	):
		"""Load a replay buffer snapshot saved with `save()`."""
		path = Path(path).expanduser()
		payload = torch.load(path, map_location="cpu", weights_only=False)
		if isinstance(payload, dict) and payload.get("format") == "newt_buffer_snapshot_v1":
			tds = payload["data"]
			metadata = dict(payload.get("metadata", {}))
		else:
			tds = payload
			metadata = {}
		if cfg is not None:
			batch_size = batch_size if batch_size is not None else int(cfg.get("batch_size", 1024))
			horizon = horizon if horizon is not None else int(cfg.get("horizon", 3))
			storage_device = storage_device if storage_device is not None else cfg.get("online_family_replay_storage_device", "cpu")
			capacity = capacity if capacity is not None else int(len(tds))
		batch_size = int(batch_size if batch_size is not None else metadata.get("batch_size", 1024))
		horizon = int(horizon if horizon is not None else metadata.get("horizon", 3))
		storage_device = storage_device if storage_device is not None else "cpu"
		capacity = int(capacity if capacity is not None else max(int(len(tds)), 1))
		capacity = max(capacity, int(len(tds)))
		buffer = cls(
			capacity=capacity,
			batch_size=batch_size,
			horizon=horizon,
			multiproc=multiproc,
			cache_values=cache_values,
			storage_device=storage_device,
			prefetch=prefetch,
		)
		buffer.load_snapshot_data(tds, metadata=metadata, max_episodes=max_episodes)
		buffer.metadata = metadata
		return buffer

	def load_demos(self, tds):
		"""
		Load a demonstration dataset into the buffer.
		"""
		assert self._num_eps == 0, \
			'Expected an empty buffer when loading demos!'
		self._num_demos = tds['episode'].max().item() + 1
		self.print_requirements(tds[tds['episode'] == 0])
		self._buffer.extend(tds)
		print(colored(f'Added {self._num_demos} demonstrations to {self.__class__.__name__}. Capacity: {len(self._buffer)}/{self.capacity}.', 'green', attrs=['bold']))
		return self._num_demos

	def next_episode_id(self, world_size=1, rank=0):
		"""
		Return the next episode ID to be used.
		This is useful for ensuring unique episode IDs across processes.
		"""
		return self._num_demos + self._num_eps * world_size + rank

	def add(self, td, world_size=1, rank=0):
		"""Add an episode to the buffer."""
		num_new_eps = td.shape[0]
		assert num_new_eps == 1, \
			'Expected a single episode to be added at a time. Use `load` for multiple episodes.'
		if self._num_eps == 0 and rank == 0:
			self.print_requirements(td[0])
		td['episode'] = torch.full_like(td['reward'], self.next_episode_id(world_size, rank), dtype=torch.int64)
		for i in range(num_new_eps):
			self._buffer.extend(td[i])
		self._num_eps += num_new_eps
		return self._num_eps

	def _prepare_batch(self, td, device):
		"""
		Prepare a sampled batch for training (post-processing).
		Expects `td` to be a TensorDict with batch size TxB.
		"""
		td = td.select("obs", "action", "reward", "task", strict=False).to(device, non_blocking=True)
		obs = td.get('obs').contiguous()
		action = td.get('action')[1:].contiguous()
		reward = td.get('reward')[1:].unsqueeze(-1).contiguous()
		task = td.get('task', None)
		if task is not None:
			task = task[1:].contiguous()
		
		return obs, action, reward, task

	def sample(self, device, batch_size=None):
		"""Sample a batch of subsequences from the buffer."""
		batch_size = int(batch_size if batch_size is not None else self._batch_size)
		if batch_size <= 0:
			raise ValueError(f"Expected positive batch_size, got {batch_size}.")
		sample_size = batch_size * (self._horizon + 1)
		old_num_slices = self._sampler.num_slices
		old_batch_size = self._buffer._batch_size
		self._sampler.num_slices = batch_size
		self._buffer._batch_size = sample_size
		try:
			td = self._buffer.sample().view(-1, self._horizon+1).permute(1, 0)
		finally:
			self._sampler.num_slices = old_num_slices
			self._buffer._batch_size = old_batch_size
		return self._prepare_batch(td, device)


class EnsembleBuffer(Buffer):
	"""
	Replay buffer for co-training on offline and online data.
	"""

	def __init__(
		self,
		offline_buffer: Buffer,
		*args,
		**kwargs
	):
		kwargs['batch_size'] = kwargs['batch_size'] // 2  # Use half the batch size for each buffer
		self._offline = offline_buffer
		super().__init__(*args, **kwargs)

	def set_storage_device(self, device):
		self._offline.set_storage_device(device)
		super().set_storage_device(device)

	def save_current(self, path, metadata=None):
		"""Save only the online/current half of an ensemble buffer."""
		return super().save(path, metadata=metadata)

	def sample(self, device, batch_size=None):
		"""Sample a batch of subsequences from the two buffers."""
		if batch_size is None:
			offline_batch_size = None
			online_batch_size = None
		else:
			batch_size = int(batch_size)
			offline_batch_size = batch_size // 2
			online_batch_size = batch_size - offline_batch_size
		obs0, action0, reward0, task0 = self._offline.sample(device, batch_size=offline_batch_size)
		try:
			obs1, action1, reward1, task1 = super().sample(device, batch_size=online_batch_size)
		except Exception as e:
			print('Failed to sample from online buffer!', e)
			raise
		
		# Combine the samples
		obs = torch.cat([obs0, obs1], dim=1)
		action = torch.cat([action0, action1], dim=1)
		reward = torch.cat([reward0, reward1], dim=1)
		task = None
		if task0 is not None and task1 is not None:
			task = torch.cat([task0, task1], dim=1)

		return obs, action, reward, task
