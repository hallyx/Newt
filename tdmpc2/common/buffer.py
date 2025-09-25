from copy import deepcopy

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
	):
		self.set_storage_device('cuda:0')
		self._capacity = capacity
		self._batch_size = batch_size
		self._sample_size = batch_size * (horizon + 1)
		self._horizon = horizon
		self._multiproc = multiproc
		self._sampler = SliceSampler(
			num_slices=batch_size,
			end_key=None,
			traj_key='episode',
			truncated_key=None,
			strict_length=True,
			cache_values=cache_values,
			use_gpu=True,
			compile=False,
		)
		self._storage = LazyTensorStorage(
			self._capacity,
			device=self._storage_device,
		)
		self._buffer = ReplayBuffer(
			storage=self._storage,
			sampler=self._sampler,
			pin_memory=False,
			prefetch=None if self._multiproc else 8,
			batch_size=self._sample_size,
			shared=self._multiproc,
		)
		self._num_eps = 0
		self._num_demos = 0

	@property
	def capacity(self):
		"""Return the capacity of the buffer."""
		return self._capacity

	@property
	def num_eps(self):
		"""Return the number of episodes in the buffer."""
		return self._num_eps
	
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

	def save(self, path):
		"""
		Save the buffer to disk.
		"""
		assert self._num_eps > 0, "Buffer is empty, nothing to save."
		torch.save(self._buffer.storage._storage, path)

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

	def sample(self, device):
		"""Sample a batch of subsequences from the buffer."""
		td = self._buffer.sample().view(-1, self._horizon+1).permute(1, 0)
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

	def sample(self, device):
		"""Sample a batch of subsequences from the two buffers."""
		obs0, action0, reward0, task0 = self._offline.sample(device)
		try:
			obs1, action1, reward1, task1 = super().sample(device)
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
