import torch
import torch.nn as nn

from common import layers, math, init
from models.axial_task_encoder import AxialTaskEncoder
from tensordict import TensorDict


class WorldModel(nn.Module):
	"""
	TD-MPC2 implicit world model architecture.
	Can be used for both single-task and multi-task experiments.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self._multitask = cfg.num_global_tasks is not None and cfg.num_global_tasks > 1
		self._task_emb = None
		self._task_encoder = None
		self._task_conditioning = str(cfg.get('task_conditioning', 'axial_params')).lower()
		if self._task_conditioning in {'axial', 'axial_params', 'param', 'param_only'}:
			self._task_conditioning = 'axial_params'
			task_vectors = cfg.get('task_vectors', None)
			if task_vectors is None or len(task_vectors) == 0:
				task_vectors = [[0.0] * int(cfg.get('axial_task_vec_dim', 6))]
			task_vectors = torch.tensor(task_vectors, dtype=torch.float32)
			if task_vectors.ndim != 2 or task_vectors.shape[-1] != int(cfg.get('axial_task_vec_dim', 6)):
				raise ValueError(f'Expected task_vectors shape (N, 6), got {tuple(task_vectors.shape)}.')
			self.register_buffer('_task_vecs', task_vectors)
			self._task_encoder = AxialTaskEncoder(task_dim=cfg.task_dim)
			if cfg.rank == 0:
				print(f'Using AxialTaskEncoder param-only task conditioning: {tuple(task_vectors.shape)} -> {cfg.task_dim}D.')
		elif self._task_conditioning in {'none', 'disabled'} or cfg.task_dim <= 0:
			self._task_conditioning = 'none'
			if cfg.rank == 0:
				print('Task conditioning disabled.')
		elif cfg.finetune:
			self._task_emb = nn.Embedding(200, cfg.task_dim)
			self._task_emb._parameters['weight'] = torch.tensor(self.cfg.task_embeddings[:1], dtype=torch.float32).repeat(200, 1)
			print(f'Using task-id embedding ablation for task {self.cfg.task}.')
		else:
			self._task_emb = nn.Embedding(len(cfg.task_embeddings), cfg.task_dim) if cfg.task_dim > 0 else None
			if self._task_emb is not None:
				if cfg.disable_task_emb:
					self._task_emb._parameters['weight'] = torch.zeros_like(self._task_emb._parameters['weight'])
					if cfg.rank == 0:
						print('Warning: Task embeddings are DISABLED by setting them to all zeros.')
				elif not cfg.learn_task_emb:
					self._task_emb._parameters['weight'] = torch.tensor(self.cfg.task_embeddings, dtype=torch.float32)
					if cfg.rank == 0:
						print('Using pre-computed task-id embeddings as an ablation.')
				elif cfg.rank == 0:
					print('Using learnable task-id embeddings as an ablation.')
		if self._task_emb is not None:
			self._task_emb.weight.requires_grad = bool(cfg.learn_task_emb) and not cfg.disable_task_emb
		if cfg.finetune:
			self.register_buffer("_action_masks", torch.zeros(200, cfg.action_dim))
			self._action_masks[:, :cfg.action_dims[0]] = 1.
		else:
			self.register_buffer("_action_masks", torch.zeros(len(cfg.action_dims), cfg.action_dim))
			for i in range(len(cfg.action_dims)):
				self._action_masks[i, :cfg.action_dims[i]] = 1.
		self._encoder = layers.enc(cfg)
		self._dynamics = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg))
		self._reward = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1))
		self._pi = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim)
		self._Qs = layers.QOnlineTargetEnsemble(cfg)
		self.apply(init.weight_init)
		init.zero_(self._reward[-1].weight)
		for i in range(cfg.num_q):
			init.zero_(self._Qs.online._Qs[i][-1].weight)
			init.zero_(self._Qs.target._Qs[i][-1].weight)
		self._Qs.hard_update_target()
		self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
		self.register_buffer("log_std_dif", torch.tensor(cfg.log_std_max) - self.log_std_min)

	def __repr__(self):
		repr = 'Newt World Model\n'
		modules = []
		if self._task_encoder is not None:
			modules.append(('Axial task encoder', self._task_encoder))
		modules.extend([
			('Encoder', self._encoder),
			('Dynamics', self._dynamics),
			('Reward', self._reward),
			('Policy prior', self._pi),
			('Q-functions', self._Qs.online),
		])
		for name, module in modules:
			params = "{:,}".format(sum(p.numel() for p in module.parameters() if p.requires_grad))
			repr += f"{name} ({params}): {module}\n"
		repr += "Learnable parameters: {:,}".format(self.total_params)
		return repr

	@property
	def total_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)

	def to(self, *args, **kwargs):
		super().to(*args, **kwargs)
		return self

	def train(self, mode=True):
		"""
		Overriding `train` method to keep target Q-networks in eval mode.
		"""
		super().train(mode)
		self._Qs.target.train(False)
		return self

	def soft_update_target_Q(self):
		"""
		Soft-update target Q-networks using Polyak averaging.
		"""
		self._Qs.soft_update_target()

	def _broadcast_task_ids(self, task, x):
		x_batch_shape = x.shape[:-1]
		if task is None:
			num_tasks = (
				self._task_vecs.shape[0] if hasattr(self, '_task_vecs')
				else self._task_emb.num_embeddings
			)
			if num_tasks == 1:
				return torch.zeros(x_batch_shape, device=x.device, dtype=torch.long)
			raise ValueError("Task ids are required when using multi-task conditioning.")
		if isinstance(task, int):
			task = torch.tensor([task], device=x.device)
		task = task.to(x.device, non_blocking=True)

		if task.ndim == 1 and len(x_batch_shape) > 1:
			if task.shape[0] == x_batch_shape[0]:
				task = task.view(task.shape[0], *([1] * (len(x_batch_shape) - 1)))
			elif task.shape[0] == x_batch_shape[-1]:
				task = task.view(*([1] * (len(x_batch_shape) - 1)), task.shape[0])
		while task.ndim < len(x_batch_shape):
			task = task.unsqueeze(-1)
		try:
			return task.expand(*x_batch_shape).long()
		except RuntimeError:
			if task.numel() == int(torch.tensor(x_batch_shape).prod().item()):
				return task.reshape(*x_batch_shape).long()
			raise ValueError(
				f"Incompatible task shape: got {tuple(task.shape)}, expected broadcastable to {x_batch_shape} "
				f"(x.shape = {tuple(x.shape)})"
			)

	def _expand_task_context(self, context, x):
		x_batch_shape = x.shape[:-1]
		context_dim = context.shape[-1]
		while context.ndim < x.ndim:
			context = context.unsqueeze(-2)
		try:
			return context.expand(*x_batch_shape, context_dim)
		except RuntimeError:
			if context.numel() == int(torch.tensor((*x_batch_shape, context_dim)).prod().item()):
				return context.reshape(*x_batch_shape, context_dim)
			raise ValueError(
				f"Incompatible task context shape: got {tuple(context.shape)}, "
				f"expected broadcastable to {(*x_batch_shape, context_dim)}."
			)

	def task_context(self, x, task):
		if self._task_encoder is not None:
			if (
				task is not None and
				torch.is_tensor(task) and
				task.is_floating_point() and
				task.ndim > 0 and
				task.shape[-1] == self.cfg.axial_task_vec_dim
			):
				task_vec = task.to(device=x.device, dtype=torch.float32, non_blocking=True)
			else:
				task_ids = self._broadcast_task_ids(task, x)
				task_vec = self._task_vecs[task_ids]
			return self._expand_task_context(self._task_encoder(task_vec), x)

		if not hasattr(self, '_task_emb') or self._task_emb is None:
			return None
		task_ids = self._broadcast_task_ids(task, x)
		return self._expand_task_context(self._task_emb(task_ids), x)

	def task_emb(self, x, task):
		"""
		Appends the task context to input x.

		Main path: task id -> task_vec_6 -> AxialTaskEncoder -> c_task.
		A task-id embedding path is retained only as an ablation.
		"""
		context = self.task_context(x, task)
		if context is None:
			return x
		return torch.cat([x, context], dim=-1)

	def encode(self, obs, task):
		"""
		Encodes an observation into its latent representation.
		This implementation assumes a single state-based observation.
		"""
		if self.cfg.obs == 'state':
			return self._encoder[self.cfg.obs](self.task_emb(obs, task))
		assert isinstance(obs, TensorDict), "Expected observation to be a TensorDict"
		z = torch.cat([self.task_emb(obs['state'], task), obs['rgb']], dim=-1)
		return self._encoder['state'](z)

		# z_rgb = self._encoder['rgb'](obs['rgb'])
		# return torch.stack((z_state, z_rgb), dim=0).mean(0)
		
		# z_state = self._encoder['state'](self.task_emb(obs['state'], task))
		# z_cat = torch.cat([z_state, self.task_emb(obs['rgb'], task)], dim=-1)
		# out = self._encoder['rgb'](z_cat)
		
		return out

	def next(self, z, a, task):
		"""
		Predicts the next latent state given the current latent state and action.
		"""
		z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		return self._dynamics(z)

	def reward(self, z, a, task):
		"""
		Predicts instantaneous (single-step) reward.
		"""
		z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		return self._reward(z)
	
	def pi(self, z, task):
		"""
		Samples an action from the policy prior.
		The policy prior is a Gaussian distribution with
		mean and (log) std predicted by a neural network.
		"""
		z = self.task_emb(z, task)

		# Gaussian policy prior
		mean, log_std = self._pi(z).chunk(2, dim=-1)
		log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
		eps = torch.randn_like(mean)

		action_mask = self._action_masks[task]  # shape: (*batch_dims, action_dim)
		while action_mask.ndim < mean.ndim:
			action_mask = action_mask.unsqueeze(-2)  # Add sequence dim (or other mid-batch dim)
		action_mask = action_mask.expand_as(mean)  # Ensure shape matches mean

		mean = mean * action_mask
		log_std = log_std * action_mask
		eps = eps * action_mask

		action_dims = action_mask.sum(-1, keepdim=True)
		log_prob = math.gaussian_logprob(eps, log_std)

		# Scale log probability by action dimensions
		size = eps.shape[-1] if action_dims is None else action_dims
		scaled_log_prob = log_prob * size

		# Reparameterization trick
		action = mean + eps * log_std.exp()
		mean, action, log_prob = math.squash(mean, action, log_prob)

		entropy_scale = scaled_log_prob / (log_prob + 1e-8)
		info = TensorDict({
			"mean": mean,
			"log_std": log_std,
			"entropy": -log_prob,
			"scaled_entropy": -log_prob * entropy_scale,
		})
		return action, info

	def Q(self, z, a, task, return_type='min', target=False, detach=False):
		"""
		Predict state-action value.
		`return_type` can be one of [`min`, `avg`, `all`]:
			- `min`: return the minimum of two randomly subsampled Q-values.
			- `avg`: return the average of two randomly subsampled Q-values.
			- `all`: return all Q-values.
		`target` specifies whether to use the target Q-networks or not.
		"""
		assert return_type in {'min', 'avg', 'all'}
		z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)

		out = self._Qs(z, target=target)
		if detach:
			out = out.detach()

		if return_type == 'all':
			return out

		qidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
		Q = math.two_hot_inv(out[qidx], self.cfg)
		if return_type == "min":
			return Q.min(0).values
		return Q.sum(0) / 2
