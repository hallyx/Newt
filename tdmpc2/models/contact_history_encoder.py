import torch
import torch.nn as nn


class ContactHistoryEncoder(nn.Module):
	"""
	MLP encoder for recent contact and motion history.

	Inputs:
	- force_history:    (..., H, force_dim)
	- action_history:   (..., H, action_dim)
	- ee_delta_history: (..., H, ee_delta_dim), optional

	Output:
	- contact_context:  (..., context_dim)
	"""

	def __init__(
		self,
		history_len: int,
		context_dim: int,
		force_dim: int = 6,
		action_dim: int = 6,
		ee_delta_dim: int = 6,
		hidden_dim: int = 128,
		num_layers: int = 2,
		use_ee_delta: bool = True,
	):
		super().__init__()
		self.history_len = int(history_len)
		self.context_dim = int(context_dim)
		self.force_dim = int(force_dim)
		self.action_dim = int(action_dim)
		self.ee_delta_dim = int(ee_delta_dim)
		self.hidden_dim = int(hidden_dim)
		self.num_layers = int(num_layers)
		self.use_ee_delta = bool(use_ee_delta)
		if self.history_len < 1:
			raise ValueError(f"history_len must be >= 1, got {history_len}.")
		if self.context_dim < 1:
			raise ValueError(f"context_dim must be >= 1, got {context_dim}.")
		if self.num_layers < 1:
			raise ValueError(f"num_layers must be >= 1, got {num_layers}.")

		step_dim = self.force_dim + self.action_dim
		if self.use_ee_delta:
			step_dim += self.ee_delta_dim
		input_dim = self.history_len * step_dim
		layers = []
		in_dim = input_dim
		for _ in range(self.num_layers):
			layers.extend([
				nn.Linear(in_dim, self.hidden_dim),
				nn.LayerNorm(self.hidden_dim),
				nn.Mish(inplace=False),
			])
			in_dim = self.hidden_dim
		layers.append(nn.Linear(in_dim, self.context_dim))
		self.net = nn.Sequential(*layers)

	def _check_history(self, value: torch.Tensor, name: str, dim: int, ref_shape=None) -> torch.Tensor:
		if value is None:
			raise ValueError(f"{name} is required.")
		if not torch.is_tensor(value):
			value = torch.as_tensor(value)
		if value.ndim < 3:
			raise ValueError(f"{name} must have shape (..., H, {dim}), got {tuple(value.shape)}.")
		if int(value.shape[-2]) != self.history_len or int(value.shape[-1]) != dim:
			raise ValueError(
				f"{name} must have shape (..., {self.history_len}, {dim}), got {tuple(value.shape)}."
			)
		if ref_shape is not None and tuple(value.shape[:-2]) != tuple(ref_shape):
			raise ValueError(
				f"{name} batch shape {tuple(value.shape[:-2])} does not match {tuple(ref_shape)}."
			)
		return value.to(dtype=torch.float32)

	def forward(
		self,
		force_history: torch.Tensor,
		action_history: torch.Tensor,
		ee_delta_history: torch.Tensor | None = None,
	) -> torch.Tensor:
		force_history = self._check_history(force_history, "force_history", self.force_dim)
		batch_shape = force_history.shape[:-2]
		action_history = self._check_history(action_history, "action_history", self.action_dim, batch_shape)
		parts = [force_history, action_history]
		if self.use_ee_delta:
			if ee_delta_history is None:
				ee_delta_history = torch.zeros(
					*batch_shape,
					self.history_len,
					self.ee_delta_dim,
					device=force_history.device,
					dtype=force_history.dtype,
				)
			else:
				ee_delta_history = self._check_history(
					ee_delta_history,
					"ee_delta_history",
					self.ee_delta_dim,
					batch_shape,
				).to(device=force_history.device)
			parts.append(ee_delta_history)
		x = torch.cat(parts, dim=-1).reshape(*batch_shape, -1)
		return self.net(x)
