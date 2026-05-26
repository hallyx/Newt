import torch
import torch.nn as nn


class TaskConditionedLatentContactResidualAdapter(nn.Module):
	"""
	Small residual adapter for task-conditioned latent contact dynamics.

	The adapter predicts a clipped latent residual around the frozen/base
	transition:

		z_next = z_next_base + gate * alpha * delta_z

	Missing optional inputs are replaced with zeros so enabling the module does
	not require a new observation or replay-buffer schema.
	"""

	def __init__(
		self,
		latent_dim: int,
		action_dim: int,
		task_dim: int = 0,
		force_history_len: int = 0,
		force_dim: int = 0,
		contact_feature_dim: int = 0,
		hidden_dim: int = 256,
		num_layers: int = 2,
		alpha: float = 0.1,
		delta_z_clip: float = 0.1,
		gate_mode: str = "always",
		use_force: bool = True,
		use_task_vec: bool = True,
		use_z_next_base: bool = True,
		contact_force_threshold: float = 0.0,
	):
		super().__init__()
		self.latent_dim = int(latent_dim)
		self.action_dim = int(action_dim)
		self.task_dim = int(task_dim) if use_task_vec else 0
		self.force_history_len = int(force_history_len) if use_force else 0
		self.force_dim = int(force_dim) if use_force else 0
		self.contact_feature_dim = int(contact_feature_dim)
		self.hidden_dim = int(hidden_dim)
		self.num_layers = int(num_layers)
		self.alpha = float(alpha)
		self.delta_z_clip = float(delta_z_clip)
		self.gate_mode = str(gate_mode).lower()
		self.use_force = bool(use_force)
		self.use_task_vec = bool(use_task_vec)
		self.use_z_next_base = bool(use_z_next_base)
		self.contact_force_threshold = float(contact_force_threshold)

		if self.gate_mode not in {"always", "contact"}:
			raise ValueError(f"latent residual gate_mode must be 'always' or 'contact', got {gate_mode!r}.")
		if self.latent_dim < 1:
			raise ValueError(f"latent_dim must be positive, got {latent_dim}.")
		if self.action_dim < 1:
			raise ValueError(f"action_dim must be positive, got {action_dim}.")
		if self.num_layers < 1:
			raise ValueError(f"num_layers must be >= 1, got {num_layers}.")

		self.force_input_dim = self.force_history_len * self.force_dim
		input_dim = self.latent_dim + self.action_dim
		if self.use_z_next_base:
			input_dim += self.latent_dim
		if self.task_dim > 0:
			input_dim += self.task_dim
		input_dim += self.force_input_dim + self.contact_feature_dim
		self.input_dim = int(input_dim)

		layers = []
		in_dim = self.input_dim
		for _ in range(self.num_layers):
			layers.extend([
				nn.Linear(in_dim, self.hidden_dim),
				nn.LayerNorm(self.hidden_dim),
				nn.Mish(inplace=False),
			])
			in_dim = self.hidden_dim
		layers.append(nn.Linear(in_dim, self.latent_dim))
		self.net = nn.Sequential(*layers)

		self.depth_head = nn.Linear(self.latent_dim, 1)
		self.radial_head = nn.Linear(self.latent_dim, 1)
		self.jam_head = nn.Linear(self.latent_dim, 1)
		self.force_head = nn.Linear(self.latent_dim, max(1, self.force_dim))

		self.reset_residual_to_zero()

	def reset_residual_to_zero(self):
		"""Start from an exact no-op residual."""
		last = self.net[-1]
		if isinstance(last, nn.Linear):
			nn.init.zeros_(last.weight)
			nn.init.zeros_(last.bias)

	def _zeros(self, batch_shape, dim, ref):
		return ref.new_zeros(*batch_shape, int(dim))

	def _align_feature(self, value, dim, batch_shape, ref):
		if dim <= 0:
			return None
		if value is None:
			return self._zeros(batch_shape, dim, ref)
		value = value.to(device=ref.device, dtype=ref.dtype, non_blocking=True)
		if value.shape[-1] > dim:
			value = value[..., :dim]
		elif value.shape[-1] < dim:
			pad = value.new_zeros(*value.shape[:-1], dim - value.shape[-1])
			value = torch.cat([value, pad], dim=-1)
		while value.ndim < len(batch_shape) + 1:
			value = value.unsqueeze(-2)
		try:
			return value.expand(*batch_shape, dim)
		except RuntimeError:
			if value.numel() == int(torch.tensor((*batch_shape, dim), device=value.device).prod().item()):
				return value.reshape(*batch_shape, dim)
			return self._zeros(batch_shape, dim, ref)

	def _align_history(self, force_hist, batch_shape, ref):
		if self.force_input_dim <= 0:
			return None, None, False
		if force_hist is None or not torch.is_tensor(force_hist):
			zeros = self._zeros(batch_shape, self.force_input_dim, ref)
			return zeros, None, False

		force_hist = force_hist.to(device=ref.device, dtype=ref.dtype, non_blocking=True)
		if force_hist.ndim < 3:
			zeros = self._zeros(batch_shape, self.force_input_dim, ref)
			return zeros, None, False
		if force_hist.shape[-1] > self.force_dim:
			force_hist = force_hist[..., :self.force_dim]
		elif force_hist.shape[-1] < self.force_dim:
			pad = force_hist.new_zeros(*force_hist.shape[:-1], self.force_dim - force_hist.shape[-1])
			force_hist = torch.cat([force_hist, pad], dim=-1)
		if force_hist.shape[-2] > self.force_history_len:
			force_hist = force_hist[..., -self.force_history_len:, :]
		elif force_hist.shape[-2] < self.force_history_len:
			pad = force_hist.new_zeros(
				*force_hist.shape[:-2],
				self.force_history_len - force_hist.shape[-2],
				self.force_dim,
			)
			force_hist = torch.cat([pad, force_hist], dim=-2)
		while force_hist.ndim < len(batch_shape) + 2:
			force_hist = force_hist.unsqueeze(-3)
		try:
			force_hist = force_hist.expand(*batch_shape, self.force_history_len, self.force_dim)
		except RuntimeError:
			expected = (*batch_shape, self.force_history_len, self.force_dim)
			if force_hist.numel() == int(torch.tensor(expected, device=force_hist.device).prod().item()):
				force_hist = force_hist.reshape(*expected)
			else:
				zeros = self._zeros(batch_shape, self.force_input_dim, ref)
				return zeros, None, False
		return force_hist.reshape(*batch_shape, self.force_input_dim), force_hist, True

	def _gate(self, force_hist, contact_feature, batch_shape, ref):
		fallback = False
		if self.gate_mode == "always":
			return ref.new_ones(*batch_shape, 1), fallback
		if contact_feature is not None and self.contact_feature_dim > 0:
			active = contact_feature[..., :1] > 0.0
			return active.to(dtype=ref.dtype), fallback
		if force_hist is not None:
			force_norm = torch.linalg.norm(force_hist[..., -1, : min(3, self.force_dim)], dim=-1, keepdim=True)
			active = force_norm > self.contact_force_threshold
			return active.to(dtype=ref.dtype), fallback
		fallback = True
		return ref.new_ones(*batch_shape, 1), fallback

	def forward(
		self,
		z,
		action,
		z_next_base,
		task_vec=None,
		force_hist=None,
		contact_feature=None,
		alpha_scale=None,
	):
		batch_shape = z.shape[:-1]
		parts = [z, action]
		if self.use_z_next_base:
			parts.append(z_next_base)
		if self.task_dim > 0:
			parts.append(self._align_feature(task_vec, self.task_dim, batch_shape, z))
		flat_force, aligned_force, has_force = self._align_history(force_hist, batch_shape, z)
		if flat_force is not None:
			parts.append(flat_force)
		aligned_contact = None
		if self.contact_feature_dim > 0:
			aligned_contact = self._align_feature(contact_feature, self.contact_feature_dim, batch_shape, z)
			parts.append(aligned_contact)

		adapter_input = torch.cat(parts, dim=-1)
		delta_z = self.net(adapter_input)
		if self.delta_z_clip > 0:
			delta_z = torch.clamp(delta_z, -self.delta_z_clip, self.delta_z_clip)

		gate, gate_fallback = self._gate(aligned_force, aligned_contact, batch_shape, z)
		scale = self.alpha
		if alpha_scale is not None:
			scale = scale * alpha_scale.to(device=z.device, dtype=z.dtype)
		while torch.is_tensor(scale) and scale.ndim < gate.ndim:
			scale = scale.unsqueeze(-1)
		z_next_adapted = z_next_base + gate * scale * delta_z

		delta_norm = torch.linalg.norm(delta_z, dim=-1)
		info = {
			"_gate": gate,
			"_delta_z_sq": delta_z.pow(2).mean(dim=-1, keepdim=True),
			"latent_residual_enabled": z.new_tensor(1.0),
			"latent_residual_alpha": z.new_tensor(self.alpha) if not torch.is_tensor(scale) else scale.mean(),
			"delta_z_norm_mean": delta_norm.mean(),
			"delta_z_norm_max": delta_norm.max(),
			"delta_z_l2": delta_z.pow(2).mean(),
			"gate_mean": gate.mean(),
			"gate_active_ratio": (gate > 0.5).to(dtype=z.dtype).mean(),
			"gate_fallback": z.new_tensor(float(gate_fallback)),
			"base_z_next_norm": torch.linalg.norm(z_next_base, dim=-1).mean(),
			"adapted_z_next_norm": torch.linalg.norm(z_next_adapted, dim=-1).mean(),
			"force_hist_available": z.new_tensor(float(has_force)),
		}
		return z_next_adapted, info

	def auxiliary_predictions(self, z_next_adapted):
		return {
			"depth_progress_pred": self.depth_head(z_next_adapted),
			"radial_error_pred": self.radial_head(z_next_adapted),
			"jam_logit": self.jam_head(z_next_adapted),
			"force_pred": self.force_head(z_next_adapted),
		}
