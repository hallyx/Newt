from copy import deepcopy

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from common import math
from common.scale import RunningScale
from common.layers import api_model_conversion, legacy_api_model_conversion


class TDMPC2(torch.nn.Module):
	"""
	Newt-based TD-MPC2 agent. Implements training + inference.
	Can be used for both single-task and multi-task experiments,
	and supports both state and pixel observations.
	"""

	def __init__(self, model, cfg):
		super().__init__()
		self.cfg = deepcopy(cfg)
		self.cfg.action_dim = cfg.action_dim
		self.device = torch.device(f'cuda:{self.cfg.device_id}')
		self.model = model
		self._latent_residual_enabled = bool(getattr(self.model, '_latent_residual_enabled', False))
		self._latent_residual_freeze_base = (
			self._latent_residual_enabled and
			bool(self.cfg.get('latent_residual_freeze_base_wm', True))
		)
		self._last_latent_residual_info = None
		self._residual_force_history = None
		self._optim_step = 0
		self._proximal_reference = None
		self._proximal_reference_filter = ("_encoder", "_dynamics")
		self._warned_missing_proximal_reference = False
		if self._latent_residual_freeze_base:
			self.model.freeze_base_world_model()
		if self._latent_residual_enabled and self.cfg.compile:
			self.cfg.compile = False
			if self.cfg.rank == 0:
				print('Disabled torch.compile for latent residual adapter mode.')
		if self._latent_residual_enabled and self.cfg.rank == 0:
			print(
				'Latent residual adapter enabled: '
				f'freeze_base_wm={self._latent_residual_freeze_base}, '
				f'alpha={self.cfg.get("latent_residual_alpha", 0.1)}, '
				f'gate={self.cfg.get("latent_residual_gate_mode", "always")}.'
			)
			if not bool(self.cfg.get('mpc', True)):
				print(
					'Actor-only mode is active: latent residual changes model prediction, '
					'but actions will not improve unless MPC or an action residual/distillation path is used.'
				)
			if str(self.cfg.get('latent_residual_gate_mode', 'always')).lower() == 'contact':
				print(
					'Latent residual contact gate will use force/contact features when available; '
					'calls without them fall back to always-on residual gating.'
				)
		if self._latent_residual_freeze_base:
			residual_params = list(self.model.latent_residual_parameters())
			if len(residual_params) == 0:
				raise ValueError("latent_residual_freeze_base_wm=true but no residual adapter parameters exist.")
			optim_groups = [{'params': residual_params}]
		else:
			optim_groups = [
				{'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
				{'params': self.model._dynamics.parameters()},
				{'params': self.model._reward.parameters()},
				{'params': self.model._Qs.online.parameters()},
				{'params': self.model._pi.parameters()},
			]
			if getattr(self.model, '_task_encoder', None) is not None:
				optim_groups.append({'params': self.model._task_encoder.parameters()})
			if getattr(self.model, '_task_emb', None) is not None and self.model._task_emb.weight.requires_grad:
				optim_groups.append({'params': self.model._task_emb.parameters()})
			if getattr(self.model, '_contact_encoder', None) is not None:
				optim_groups.append({'params': self.model._contact_encoder.parameters()})
			if self._latent_residual_enabled:
				optim_groups.append({'params': self.model.latent_residual_parameters()})
		self.optim = torch.optim.Adam(optim_groups, lr=self.cfg.lr, capturable=True)
		self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5, capturable=True)
		if self.cfg.lr_schedule:
			self.scheduler = math.MultiWarmupConstantLR(
				[self.optim, self.pi_optim],
				warmup_steps=self.cfg.warmup_steps,
			)
			if self.cfg.rank == 0:
				print(f'Using {self.cfg.lr_schedule} learning rate schedule with {self.cfg.warmup_steps} warmup steps.')
		elif self.cfg.rank == 0:
			print('No learning rate schedule specified, using constant LR.')
		self.model.eval()
		self.maxq_pi = True
		self.scale = RunningScale(self.cfg)
		self.discount = torch.tensor(self.cfg.discounts, device=self.device, dtype=torch.float32)
		if self.cfg.rank == 0:
			print('Episode length:', self.cfg.episode_length)
			print('Discount factor:', self.discount)
		self._prev_mean = torch.zeros(self.cfg.num_envs, self.cfg.horizon, self.cfg.action_dim, device=self.device)
		self.rho = torch.pow(self.cfg.rho, torch.arange(self.cfg.horizon+1, device=self.device))
		self.rho = self.rho / self.rho.sum()
		if self.cfg.compile:
			if self.cfg.rank == 0:
				print('Compiling methods...')
			self.pi = torch.compile(self._pi, mode="reduce-overhead")
			self.sample_pi_trajs = torch.compile(self._sample_pi_trajs, mode="reduce-overhead")
			self.mppi = torch.compile(self._mppi, mode="reduce-overhead")
			self.loss_fn = torch.compile(self._loss_fn, mode="reduce-overhead")
		else:
			self.pi = self._pi
			self.sample_pi_trajs = self._sample_pi_trajs
			self.mppi = self._mppi
			self.loss_fn = self._loss_fn

	def set_proximal_reference(self, module_filter=("_encoder", "_dynamics")):
		"""
		Store a frozen copy of selected trunk parameters for optional continual-learning
		proximal regularization.
		"""
		self._proximal_reference_filter = tuple(str(item) for item in module_filter)
		self._proximal_reference = {}
		for name, param in self.model.named_parameters():
			if any(token in name for token in self._proximal_reference_filter):
				self._proximal_reference[name] = param.detach().clone()
		if self.cfg.rank == 0:
			print(
				"Stored proximal reference for "
				f"{len(self._proximal_reference)} parameter tensors "
				f"(filter={self._proximal_reference_filter})."
			)

	def _proximal_loss(self):
		if not bool(self.cfg.get('multitask_prox_reg_enabled', False)):
			return None
		if not self._proximal_reference:
			if self.cfg.rank == 0 and not self._warned_missing_proximal_reference:
				print(
					"Warning: multitask_prox_reg_enabled=true but no proximal reference is set; "
					"skipping proximal loss."
				)
				self._warned_missing_proximal_reference = True
			return None
		loss = torch.zeros((), device=self.device)
		for name, param in self.model.named_parameters():
			ref = self._proximal_reference.get(name, None)
			if ref is None:
				continue
			loss = loss + (param - ref.to(param.device, non_blocking=True)).pow(2).mean()
		return loss

	def _checkpoint_metadata(self):
		metadata = {
			"task_conditioning": self.cfg.get("task_conditioning", None),
			"num_global_tasks": self.cfg.get("num_global_tasks", None),
		}
		if bool(self.cfg.get("multitask_continuation_enabled", False)):
			metadata.update({
				"multitask_continuation_enabled": True,
				"task_ids": list(self.cfg.get("multitask_task_ids", []) or []),
				"anchor_task_id": self.cfg.get("multitask_anchor_task_id", None),
				"active_tasks": list(self.cfg.get("multitask_current_active_tasks", []) or []),
				"task_vec_dim": int(self.cfg.get("axial_task_vec_dim", 6)),
				"reference_checkpoint_path": self.cfg.get(
					"multitask_reference_checkpoint_path",
					self.cfg.get("checkpoint", ""),
				),
			})
		return metadata

	def _is_task_vec(self, task):
		return (
			task is not None and
			torch.is_tensor(task) and
			task.is_floating_point() and
			task.ndim > 0 and
			task.shape[-1] == int(self.cfg.get('axial_task_vec_dim', 6))
		)

	def _task_ids_for_shape(self, task, batch_shape):
		batch_shape = tuple(int(dim) for dim in batch_shape)
		if self._is_task_vec(task):
			task_ids = torch.zeros(task.shape[:-1], dtype=torch.long, device=self.device)
		elif task is None:
			if len(self.discount) != 1:
				raise ValueError("Task ids are required when using multi-task discrete metadata.")
			return torch.zeros(batch_shape, dtype=torch.long, device=self.device)
		else:
			if not torch.is_tensor(task):
				task = torch.tensor([task], device=self.device)
			task_ids = task.to(self.device, non_blocking=True).long()
		if task_ids.ndim == 0:
			task_ids = task_ids.view(1)
		if task_ids.ndim == 1 and len(batch_shape) > 1:
			if task_ids.shape[0] == batch_shape[0]:
				task_ids = task_ids.view(task_ids.shape[0], *([1] * (len(batch_shape) - 1)))
			elif task_ids.shape[0] == batch_shape[-1]:
				task_ids = task_ids.view(*([1] * (len(batch_shape) - 1)), task_ids.shape[0])
		while task_ids.ndim < len(batch_shape):
			task_ids = task_ids.unsqueeze(-1)
		try:
			return task_ids.expand(*batch_shape).long()
		except RuntimeError:
			if task_ids.numel() == int(torch.tensor(batch_shape).prod().item()):
				return task_ids.reshape(*batch_shape).long()
			raise

	def _discount_for(self, task, batch_shape):
		task_ids = self._task_ids_for_shape(task, batch_shape)
		return self.discount[task_ids].unsqueeze(-1)

	def _action_mask_for(self, task, batch_shape):
		task_ids = self._task_ids_for_shape(task, batch_shape)
		return self.model._action_masks[task_ids]

	def _repeat_task_for_trajs(self, task, repeats):
		if task is None:
			return None
		if not torch.is_tensor(task):
			task = torch.tensor([task], device=self.device)
		task = task.to(self.device, non_blocking=True)
		if self._is_task_vec(task):
			return task.unsqueeze(1).repeat(1, repeats, 1).reshape(-1, task.shape[-1])
		return task.unsqueeze(1).repeat(1, repeats).reshape(-1)

	def _set_latent_residual_alpha_scale(self, step=None):
		if not self._latent_residual_enabled:
			return
		warmup = int(self.cfg.get('latent_residual_alpha_warmup_steps', 0) or 0)
		if warmup <= 0:
			scale = 1.0
		else:
			step_value = self._optim_step if step is None else int(step)
			scale = min(1.0, max(0.0, float(step_value + 1) / float(warmup)))
		self.model.set_latent_residual_alpha_scale(scale)

	def _residual_force_dim(self):
		return int(self.cfg.get('latent_residual_force_dim', self.cfg.get('contact_force_dim', 6)) or 0)

	def _residual_force_history_len(self):
		return int(self.cfg.get('latent_residual_force_history_len', self.cfg.get('contact_history_len', 4)) or 0)

	def _extract_residual_force_obs(self, obs):
		if not (self._latent_residual_enabled and bool(self.cfg.get('latent_residual_use_force', True))):
			return None
		if isinstance(obs, TensorDict):
			if 'state' not in obs.keys():
				return None
			obs = obs['state']
		if obs is None or not torch.is_tensor(obs) or obs.shape[-1] <= 14:
			return None
		if not bool(self.cfg.get('isaaclab_canonical_append_force', False)):
			return None
		force_dim = self._residual_force_dim()
		if force_dim <= 0:
			return None
		available_dim = min(
			int(obs.shape[-1]) - 14,
			max(1, int(self.cfg.get('isaaclab_canonical_force_dim', min(force_dim, 3)) or min(force_dim, 3))),
		)
		if available_dim <= 0:
			return None
		force = obs[..., 14:14 + available_dim].to(device=self.device, dtype=torch.float32, non_blocking=True)
		if available_dim < force_dim:
			pad = force.new_zeros(*force.shape[:-1], force_dim - available_dim)
			force = torch.cat([force, pad], dim=-1)
		elif available_dim > force_dim:
			force = force[..., :force_dim]
		return force

	def _update_residual_force_history(self, obs, t0=None):
		force = self._extract_residual_force_obs(obs)
		if force is None:
			self._residual_force_history = None
			return None
		history_len = self._residual_force_history_len()
		if history_len <= 0:
			return None
		batch_shape = force.shape[:-1]
		expected_shape = (*batch_shape, history_len, force.shape[-1])
		if self._residual_force_history is None or tuple(self._residual_force_history.shape) != tuple(expected_shape):
			self._residual_force_history = force.new_zeros(*expected_shape)
		if t0 is not None:
			t0 = t0.to(device=force.device, dtype=torch.bool, non_blocking=True)
			while t0.ndim < len(batch_shape):
				t0 = t0.unsqueeze(-1)
			reset = t0.reshape(*batch_shape, 1, 1)
			self._residual_force_history = torch.where(
				reset,
				torch.zeros_like(self._residual_force_history),
				self._residual_force_history,
			)
		self._residual_force_history = torch.cat(
			[self._residual_force_history[..., 1:, :], force.unsqueeze(-2)],
			dim=-2,
		)
		return self._residual_force_history

	def _loss_residual_force_histories(self, obs):
		force = self._extract_residual_force_obs(obs)
		if force is None:
			return [None] * self.cfg.horizon
		history_len = self._residual_force_history_len()
		if history_len <= 0:
			return [None] * self.cfg.horizon
		histories = []
		for t in range(self.cfg.horizon):
			start = max(0, t - history_len + 1)
			window = force[start:t+1]
			if window.shape[0] < history_len:
				pad = window.new_zeros(history_len - window.shape[0], *window.shape[1:])
				window = torch.cat([pad, window], dim=0)
			histories.append(window.permute(1, 0, 2).contiguous())
		return histories

	def latent_residual_metrics(self):
		if not self._latent_residual_enabled:
			return {}
		out = {
			"latent_residual_enabled": torch.tensor(1.0, device=self.device),
			"latent_residual_alpha": torch.tensor(float(self.cfg.get('latent_residual_alpha', 0.1)), device=self.device),
		}
		if self._last_latent_residual_info:
			out.update({
				k: v.detach() if torch.is_tensor(v) else torch.tensor(float(v), device=self.device)
				for k, v in self._last_latent_residual_info.items()
				if not str(k).startswith('_')
			})
		return out

	def save(self, fp):
		"""
		Save state dict of the agent to filepath.

		Args:
			fp (str): Filepath to save state dict to.
		"""
		torch.save({
			"model": self.model.state_dict(),
			"optim": self.optim.state_dict(),
			"pi_optim": self.pi_optim.state_dict(),
			"scale": self.scale.state_dict(),
			"metadata": self._checkpoint_metadata(),
		}, fp)

	def load(self, fp):
		"""
		Load a saved state dict from filepath (or dictionary) into current agent.

		Args:
			fp (str or dict): Filepath or state dict to load.
		"""
		if isinstance(fp, dict):
			state_dict = fp
		else:
			state_dict = torch.load(fp, map_location=torch.get_default_device(), weights_only=False)
		state_dict = state_dict["model"] if "model" in state_dict else state_dict
		
		# Retain task-specific buffers/embeddings when finetuning, but keep learned
		# shared encoders (e.g. AxialTaskEncoder) from the checkpoint.
		if self.cfg.finetune:
			prefix = "module." if any(key.startswith("module.") for key in state_dict.keys()) else ""
			if getattr(self.model, '_task_emb', None) is not None:
				state_dict[prefix+"_task_emb.weight"] = self.model._task_emb.weight
			if getattr(self.model, '_task_encoder', None) is not None:
				state_dict[prefix+"_task_vecs"] = self.model._task_vecs
			state_dict[prefix+"_action_masks"] = self.model._action_masks

		state_dict = api_model_conversion(self.model.state_dict(), state_dict)
		target_state = self.model.state_dict()
		for key in ("_task_vecs", "_action_masks"):
			if key not in target_state:
				continue
			for source_key in (key, f"module.{key}"):
				if source_key not in state_dict:
					continue
				if tuple(state_dict[source_key].shape) == tuple(target_state[key].shape):
					continue
				if self.cfg.rank == 0:
					print(
						f"Using current {key} from config instead of checkpoint metadata: "
						f"checkpoint_shape={tuple(state_dict[source_key].shape)} "
						f"current_shape={tuple(target_state[key].shape)}."
					)
				state_dict[source_key] = target_state[key]
		try:
			self.model.load_state_dict(state_dict)
		except Exception as load_error:
			try:
				legacy_state_dict = legacy_api_model_conversion(self.model.state_dict(), state_dict)
				out = self.model.load_state_dict(legacy_state_dict)
			except Exception:
				raise RuntimeError(
					"Failed to load checkpoint into the current model. "
					"The checkpoint does not appear to be compatible with the current config, "
					"and legacy API conversion also failed."
				) from load_error
			print(out)
			print('Successfully loaded checkpoint after converting from legacy API.')
		if self._latent_residual_freeze_base:
			self.model.freeze_base_world_model()
		return
	
	@torch.no_grad()
	def _pi(self, obs, task=None):
		"""
		Select an action using the policy network.
		"""
		z = self.model.encode(obs, task)
		action, info = self.model.pi(z, task)
		return action, info

	@torch.no_grad()
	def forward(self, obs, t0, step, eval_mode=False, task=None, mpc=None):
		"""
		Select an action by planning in the latent space of the world model.

		Args:
			obs (torch.Tensor): Observation from the environment.
			t0 (torch.Tensor): Whether this is the first observation in the episode.
			step (int): Current environment step.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (torch.Tensor): Task index.
			mpc (bool): Whether to use model predictive control.

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		if isinstance(obs, dict):
			obs = TensorDict(obs)
		obs = obs.to(self.device, non_blocking=True)
		if task is not None and not isinstance(task, torch.Tensor):
			task = torch.tensor([task], device=self.device)
		if task is not None and task.device != self.device:
			task = task.to(self.device, non_blocking=True)
		self._set_latent_residual_alpha_scale(step)
		residual_force_history = self._update_residual_force_history(obs, t0=t0)
		mpc = mpc if mpc is not None else self.cfg.mpc
		if mpc:
			if t0.device != self.device:
				t0 = t0.to(self.device, non_blocking=True)
			action, info = self.plan(
				obs,
				t0=t0,
				step=step,
				eval_mode=eval_mode,
				task=task,
				residual_force_history=residual_force_history,
			)
		else:
			action, action_info = self.pi(obs, task)
			if eval_mode:
				action = action_info["mean"]
			if self._latent_residual_enabled:
				z = self.model.encode(obs, task)
				_, residual_info = self.model.next(
					z,
					action,
					task,
					residual_force_history=residual_force_history,
					return_info=True,
				)
				residual_info.pop("_z_next_base", None)
				self._last_latent_residual_info = residual_info
			info = TensorDict({
				"pi_mean": action_info["mean"].mean(),
				"pi_std": action_info["log_std"].exp().mean(),
			})
			if self._latent_residual_enabled and self._last_latent_residual_info:
				info.update(TensorDict({
					k: v.detach()
					for k, v in self._last_latent_residual_info.items()
					if torch.is_tensor(v) and not str(k).startswith('_')
				}))
		if (
			self.cfg.task.startswith('isaaclab-') or
			self.cfg.isaaclab_env_id.startswith('Isaac-') or
			self.cfg.get('isaaclab_backend', 'auto') == 'srsa' or
			self.cfg.get('isaaclab_task_package', None) == 'SRSA.tasks'
		):
			return action, info
		return action.cpu(), info
	
	@torch.no_grad()
	def _estimate_value(self, z, actions, task, residual_force_history=None):
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""
		G = torch.zeros(self.cfg.num_envs, self.cfg.num_samples, 1, dtype=torch.float32, device=z.device)
		discount = torch.ones(self.cfg.num_envs, self.cfg.num_samples, 1, dtype=torch.float32, device=z.device)
		for t in range(self.cfg.horizon):
			reward = math.two_hot_inv(self.model.reward(z, actions[:, t], task), self.cfg)
			z = self.model.next(
				z,
				actions[:, t],
				task,
				residual_force_history=residual_force_history,
			)
			G = G + discount * reward
			discount_update = self._discount_for(task, z.shape[:-1])
			discount = discount * discount_update
		action, _ = self.model.pi(z, task)
		value = self.model.Q(z, action, task, return_type='avg')
		return G + discount * value
	
	@torch.no_grad()
	def _sample_pi_trajs(self, z, task=None, residual_force_history=None):
		pi_actions = torch.empty(self.cfg.num_envs, self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
		_z = z.unsqueeze(1).repeat(1, self.cfg.num_pi_trajs, 1).view(self.cfg.num_envs * self.cfg.num_pi_trajs, -1)
		_task = self._repeat_task_for_trajs(task, self.cfg.num_pi_trajs)
		_force_history = None
		if residual_force_history is not None:
			_force_history = residual_force_history.unsqueeze(1).repeat(
				1, self.cfg.num_pi_trajs, 1, 1
			).view(self.cfg.num_envs * self.cfg.num_pi_trajs, *residual_force_history.shape[-2:])
		for t in range(self.cfg.horizon - 1):
			a, _ = self.model.pi(_z, _task)
			pi_actions[:, t] = a.view(self.cfg.num_envs, self.cfg.num_pi_trajs, self.cfg.action_dim)
			_z = self.model.next(_z, a, _task, residual_force_history=_force_history)
		a, _ = self.model.pi(_z, _task)
		pi_actions[:, -1] = a.view(self.cfg.num_envs, self.cfg.num_pi_trajs, self.cfg.action_dim)
		return pi_actions
	
	@torch.no_grad()
	def _mppi(self, z, pi_actions, task, mean, std, residual_force_history=None):
		"""
		MPPI loop.
		"""
		actions = torch.empty(self.cfg.num_envs, self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
		if self.cfg.num_pi_trajs > 0:
			actions[:, :, :self.cfg.num_pi_trajs] = pi_actions
		action_mask = self._action_mask_for(task, (self.cfg.num_envs,)).unsqueeze(1).unsqueeze(1)

		# Iterate MPPI
		for _ in range(self.cfg.iterations):

			# Sample new actions
			r = torch.randn(self.cfg.num_envs, self.cfg.horizon, self.cfg.num_samples - self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)
			actions_sample = mean.unsqueeze(2) + std.unsqueeze(2) * r
			actions[:, :, self.cfg.num_pi_trajs:] = actions_sample.clamp(-1, 1)
			actions = actions * action_mask

			# Compute elite actions
			value = self._estimate_value(z, actions, task, residual_force_history=residual_force_history).nan_to_num(0)
			elite_idxs = torch.topk(value.squeeze(2), self.cfg.num_elites, dim=1).indices
			elite_value = torch.gather(value, 1, elite_idxs.unsqueeze(2))
			elite_actions = actions.gather(
				dim=2,
				index=elite_idxs[:, None, :, None].expand(-1, self.cfg.horizon, self.cfg.num_elites, self.cfg.action_dim)
			)

			# Update parameters
			score = torch.exp(self.cfg.temperature * (elite_value - elite_value.max(1, keepdim=True).values))
			score = score / (score.sum(dim=1, keepdim=True) + 1e-9)
			score_exp = score.unsqueeze(1)
			mean = (score_exp * elite_actions).sum(dim=2) / (score_exp.sum(dim=2) + 1e-9)
			std = ((score_exp * (elite_actions - mean.unsqueeze(2)) ** 2).sum(dim=2) /
				(score_exp.sum(dim=2) + 1e-9)).sqrt().clamp(self.cfg.min_std, self.cfg.max_std)
			mean = mean * action_mask.squeeze(2)
			std = std * action_mask.squeeze(2)

		# Select action
		rand_idx = math.gumbel_softmax_sample(score.squeeze(2), temperature=self.cfg.temperature, dim=1)
		selected_actions = elite_actions.gather(
			dim=2,
			index=rand_idx[:, None, None, None].expand(-1, self.cfg.horizon, 1, self.cfg.action_dim)
		).squeeze(2)
		action, std_out = selected_actions[:, 0], std[:, 0]

		return action.clamp(-1, 1), mean, std_out

	@torch.no_grad()
	def plan(self, obs, t0, step, eval_mode=False, task=None, residual_force_history=None):
		"""
		Plan a sequence of actions using the learned world model.

		Args:
			obs (torch.Tensor): Observation from the environment.
			t0 (torch.Tensor): Whether this is the first observation in the episode.
			step (int): Current environment step.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (torch.Tensor): Task index.

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		# Sample policy trajectories
		z0 = self.model.encode(obs, task)
		if self.cfg.num_pi_trajs > 0:
			pi_actions = self.sample_pi_trajs(z0, task, residual_force_history=residual_force_history)
		else:
			pi_actions = None

		# Initialize state and parameters
		z = z0.unsqueeze(1).repeat(1, self.cfg.num_samples, 1)
		mppi_force_history = None
		if residual_force_history is not None:
			mppi_force_history = residual_force_history.unsqueeze(1)
		shifted = torch.cat([self._prev_mean[:, 1:], torch.zeros_like(self._prev_mean[:, :1])], dim=1)
		base_mean = torch.where(t0.view(self._prev_mean.shape[0], 1, 1), torch.zeros_like(shifted), shifted)
		base_std = torch.full((self.cfg.num_envs, self.cfg.horizon, self.cfg.action_dim), self.cfg.max_std, device=self.device)

		if self.cfg.constrained_planning:
			# Init planning with policy statistics
			pi_mean = pi_actions.mean(2)  # (Ne, H, Np, A) -> (Ne, H, A)
			pi_std = pi_actions.std(2).clamp(self.cfg.min_std, self.cfg.max_std)

			if step < self.cfg.constraint_start_step:
				w = 1.0
			else:
				num = max(0, step - self.cfg.constraint_start_step)
				den = max(1, self.cfg.constraint_final_step)
				w = max(self.cfg.constraint_min_weight, 1.0 - (num / den))
			w = torch.as_tensor(w, device=self.device, dtype=base_mean.dtype).view(1, 1, 1)

			# Linearly annealed mix of policy and base prior
			mean = w * pi_mean + (1.0 - w) * base_mean
			std = w * pi_std + (1.0 - w) * base_std
		else:
			# Use base prior
			mean, std, w = base_mean, base_std, 0.

		# Optimize with MPPI
		action, out_mean, out_std = self.mppi(
			z,
			pi_actions,
			task,
			mean,
			std,
			residual_force_history=mppi_force_history,
		)
		self._prev_mean = out_mean.clone()
		if not eval_mode:
			action = (action + out_std * torch.randn_like(action)).clamp(-1, 1)

		info = TensorDict({
			"pi_mean": pi_actions.mean() if self.cfg.num_pi_trajs > 0 else None,
			"pi_std": pi_actions.std() if self.cfg.num_pi_trajs > 0 else None,
		})
		if self._latent_residual_enabled:
			_, residual_info = self.model.next(
				z0,
				action,
				task,
				residual_force_history=residual_force_history,
				return_info=True,
			)
			residual_info.pop("_z_next_base", None)
			self._last_latent_residual_info = residual_info
			info.update(TensorDict({
				k: v.detach()
				for k, v in residual_info.items()
				if torch.is_tensor(v) and not str(k).startswith('_')
			}))

		return action, info
		
	def update_pi(self, zs, action, task):
		"""
		Update policy using a sequence of latent states.

		Args:
			zs (torch.Tensor): Sequence of latent states.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			float: Loss of the policy update.
		"""
		self.model._Qs.track_grad(False)

		pi_action, info = self.model.pi(zs, task)

		# Policy prior loss
		pi_prior_loss = (math.masked_bc_per_timestep(pi_action[:-1], action, task, self.model._action_masks) \
				   * self.rho[:-1, None]).sum(0)

		# Normalized Q-loss
		qs = self.model.Q(zs, pi_action, task, return_type='avg')
		with torch.no_grad():
			self.scale.update(qs[0])  # local update
			if torch.distributed.is_initialized():
				torch.distributed.all_reduce(self.scale.value, op=torch.distributed.ReduceOp.SUM)
				self.scale.value /= self.cfg.world_size
		qs = self.scale(qs)
		maxq_loss = ((-self.cfg.entropy_coef*info["scaled_entropy"] - qs) * self.rho[:, None, None]).sum(dim=(0,2))
		
		# Compute total policy loss
		pi_loss = (pi_prior_loss + maxq_loss).mean()
		
		pi_loss.backward()
		pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
		self.pi_optim.step()
		self.pi_optim.zero_grad(set_to_none=True)
		if getattr(self.model, '_task_encoder', None) is not None:
			self.model._task_encoder.zero_grad(set_to_none=True)
		if getattr(self.model, '_task_emb', None) is not None:
			self.model._task_emb.zero_grad(set_to_none=True)
		self.model._Qs.track_grad(True)

		info = TensorDict({
			"pi_prior_loss": pi_prior_loss.mean(),
			"pi_loss": pi_loss,
			"pi_grad_norm": pi_grad_norm,
			"pi_entropy": info["entropy"],
			"pi_scaled_entropy": info["scaled_entropy"],
			"pi_std": info["log_std"].exp().mean(),
			"pi_max_std": info["log_std"].exp().max(),
			"pi_scale": self.scale.value,
		})
		return info

	@torch.no_grad()
	def _td_target(self, next_z, reward, task):
		"""
		Compute the TD-target from a reward and the observation at the following time step.

		Args:
			next_z (torch.Tensor): Latent state at the following time step.
			reward (torch.Tensor): Reward at the current time step.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: TD-target.
		"""
		action, _ = self.model.pi(next_z, task)
		discount = self._discount_for(task, next_z.shape[:-1])
		return reward + discount * self.model.Q(next_z, action, task, return_type='min', target=True)

	def _loss_fn(self, obs, action, reward, task=None):
		"""
		Compute the model loss for a batch of data.
		"""
		# Compute targets
		with torch.no_grad():
			next_z = self.model.encode(obs[1:], task)
			td_targets = self._td_target(next_z, reward, task)

		# Latent rollout
		zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
		z = self.model.encode(obs[0], task[0])
		zs[0] = z
		consistency_loss = 0
		consistency_loss_base = 0
		latent_residual_reg_loss = 0
		residual_infos = []
		residual_force_histories = self._loss_residual_force_histories(obs)
		for t, (_action, _next_z, _task) in enumerate(zip(action.unbind(0), next_z.unbind(0), task.unbind(0))):
			if self._latent_residual_enabled:
				z, residual_info = self.model.next(
					z,
					_action,
					_task,
					residual_force_history=residual_force_histories[t],
					return_info=True,
				)
				z_next_base = residual_info.pop("_z_next_base")
				consistency_loss_base = consistency_loss_base + F.mse_loss(z_next_base.detach(), _next_z) * self.rho[t]
				if bool(self.cfg.get('latent_residual_train_only_contact_phase', False)):
					gate = residual_info.get("_gate", None)
					if gate is not None:
						weight = gate.detach()
						denom = weight.mean().clamp_min(1.0e-6)
						per_sample_err = (z - _next_z).pow(2).mean(dim=-1, keepdim=True)
						consistency_loss = consistency_loss + ((per_sample_err * weight).mean() / denom) * self.rho[t]
						delta_sq = residual_info.get("_delta_z_sq", None)
						if delta_sq is not None:
							latent_residual_reg_loss = latent_residual_reg_loss + ((delta_sq * weight).mean() / denom) * self.rho[t]
						else:
							latent_residual_reg_loss = latent_residual_reg_loss + residual_info["delta_z_l2"] * self.rho[t]
					else:
						consistency_loss = consistency_loss + F.mse_loss(z, _next_z) * self.rho[t]
						latent_residual_reg_loss = latent_residual_reg_loss + residual_info["delta_z_l2"] * self.rho[t]
				else:
					consistency_loss = consistency_loss + F.mse_loss(z, _next_z) * self.rho[t]
					latent_residual_reg_loss = latent_residual_reg_loss + residual_info["delta_z_l2"] * self.rho[t]
				residual_infos.append(residual_info)
			else:
				z = self.model.next(z, _action, _task)
				consistency_loss = consistency_loss + F.mse_loss(z, _next_z) * self.rho[t]
			zs[t+1] = z

		# Predictions
		_zs = zs[:-1]
		qs = self.model.Q(_zs, action, task, return_type='all')
		reward_preds = self.model.reward(_zs, action, task)

		# Compute losses
		reward_loss, value_loss = 0, 0
		for t, (rew_pred_unbind, rew_unbind, td_targets_unbind, qs_unbind) in enumerate(zip(reward_preds.unbind(0), reward.unbind(0), td_targets.unbind(0), qs.unbind(1))):
			reward_loss = reward_loss + math.soft_ce(rew_pred_unbind, rew_unbind, self.cfg).mean() * self.rho[t]
			for _, qs_unbind_unbind in enumerate(qs_unbind.unbind(0)):
				value_loss = value_loss + math.soft_ce(qs_unbind_unbind, td_targets_unbind, self.cfg).mean() * self.rho[t]
		value_loss = value_loss / self.cfg.num_q

		if not self.maxq_pi: # Behavior cloning
			pi_action, pi_info = self.model.pi(_zs, task)
			bc_loss = math.masked_bc_per_timestep(pi_action, action, task, self.model._action_masks)
			entropy_loss = -self.cfg.entropy_coef*pi_info["scaled_entropy"].squeeze(-1)
			pi_prior_loss = ((bc_loss + entropy_loss) * self.rho[:-1, None]).mean()
			pi_info = TensorDict({
				"bc_loss": bc_loss,
				"entropy_loss": entropy_loss,
				"pi_prior_loss": pi_prior_loss,
				"pi_entropy": pi_info["entropy"],
				"pi_scaled_entropy": pi_info["scaled_entropy"],
				"pi_std": pi_info["log_std"].exp().mean(),
				"pi_max_std": pi_info["log_std"].exp().max(),
			})
		else:
			pi_prior_loss = 0
			pi_info = TensorDict({})

		total_loss = (
			self.cfg.consistency_coef * consistency_loss +
			self.cfg.reward_coef * reward_loss +
			self.cfg.value_coef * value_loss +
			self.cfg.prior_coef * pi_prior_loss
		)
		if self._latent_residual_enabled:
			total_loss = total_loss + float(self.cfg.get('latent_residual_reg_coef', 1.0e-4)) * latent_residual_reg_loss

		info = TensorDict({
			"consistency_loss": consistency_loss,
			"reward_loss": reward_loss,
			"value_loss": value_loss,
			"total_loss": total_loss,
		})
		if self._latent_residual_enabled:
			info.update(TensorDict({
				"consistency_loss_base": consistency_loss_base,
				"consistency_loss_adapted": consistency_loss,
				"latent_residual_reg_loss": latent_residual_reg_loss,
				"latent_residual_reg_coef": torch.tensor(
					float(self.cfg.get('latent_residual_reg_coef', 1.0e-4)),
					device=self.device,
				),
			}))
			if len(residual_infos) > 0:
				for key in residual_infos[0].keys():
					if key.startswith('_'):
						continue
					values = [step_info[key] for step_info in residual_infos if key in step_info]
					if len(values) > 0 and torch.is_tensor(values[0]):
						info[key] = torch.stack(values).mean()
		info.update(pi_info)

		return total_loss, zs.detach(), info.detach()

	def _update(self, obs, action, reward, task=None):
		# Prepare for update
		self.model.train()

		# Step the learning rate scheduler
		if self.cfg.lr_schedule:
			self.scheduler.step()
		self._set_latent_residual_alpha_scale()

		# Compute loss
		torch.compiler.cudagraph_mark_step_begin()
		total_loss, zs, info = self.loss_fn(obs, action, reward, task)
		proximal_loss = self._proximal_loss()
		if proximal_loss is not None:
			proximal_coef = float(self.cfg.get('multitask_prox_reg_coef', 1.0e-4))
			total_loss = total_loss + proximal_coef * proximal_loss
			info.update(TensorDict({
				"multitask_proximal_loss": proximal_loss.detach(),
				"multitask_proximal_coef": torch.tensor(proximal_coef, device=self.device),
			}))

		# Update model
		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		self.optim.step()
		self.optim.zero_grad(set_to_none=True)

		# Update target Q-functions
		self.model.soft_update_target_Q()

		if self.maxq_pi and not self._latent_residual_freeze_base:
			# Max-Q policy update
			pi_info = self.update_pi(zs, action, task[:1])
			info.update(pi_info)
		self._optim_step += 1
		
		# Return training statistics
		self.model.eval()
		info.update({
			"grad_norm": grad_norm,
		})
		if self.cfg.lr_schedule:
			info.update({
				"lr_enc": self.scheduler.current_lr(0, 0),
				"lr": self.scheduler.current_lr(0, 1),
				"lr_pi": self.scheduler.current_lr(1, 0),
			})
		return info.detach().mean()

	def update(self, buffer):
		"""
		Main update function. Corresponds to one iteration of model learning.

		Args:
			buffer (common.buffer.Buffer): Replay buffer.

		Returns:
			dict: Dictionary of training statistics.
		"""
		obs, action, reward, task = buffer.sample(device=self.device)
		kwargs = {}
		if task is not None:
			kwargs["task"] = task
		return self._update(obs, action, reward, **kwargs)
