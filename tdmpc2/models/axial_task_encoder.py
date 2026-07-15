import torch
import torch.nn as nn


class AxialTaskEncoder(nn.Module):
	"""
	Factorized encoder for axial mating task parameters.

	Input task_vec_6:
	[
		task_type_id_float,
		log_scale,
		clearance_abs_norm,
		clearance_rel_norm,
		depth_abs_norm,
		yaw_requirement_float,
	]
	"""

	def __init__(
		self,
		task_dim: int = 64,
		task_vec_dim: int = 6,
		repair_enabled: bool = False,
		raw_residual_scale: float = 0.0,
		normalize_inputs: bool = False,
		normalization_eps: float = 1.0e-6,
	):
		super().__init__()
		if int(task_dim) != 64:
			raise ValueError(f"AxialTaskEncoder outputs 64 dims, got task_dim={task_dim}.")
		if int(task_vec_dim) != 6:
			raise ValueError(f"AxialTaskEncoder expects 6D task vectors, got task_vec_dim={task_vec_dim}.")
		self.task_dim = int(task_dim)
		self.task_vec_dim = int(task_vec_dim)
		self.repair_enabled = bool(repair_enabled)
		self.raw_residual_scale = float(raw_residual_scale)
		self.normalize_inputs = bool(normalize_inputs)
		self.normalization_eps = float(normalization_eps)
		self.register_buffer("_norm_mean", torch.zeros(self.task_vec_dim), persistent=False)
		self.register_buffer("_norm_std", torch.ones(self.task_vec_dim), persistent=False)
		self.type_encoder = nn.Embedding(2, 8)
		self.metric_encoder = nn.Sequential(
			nn.Linear(4, 32),
			nn.LayerNorm(32),
			nn.Mish(inplace=False),
			nn.Linear(32, 32),
			nn.LayerNorm(32),
			nn.Mish(inplace=False),
		)
		self.yaw_encoder = nn.Sequential(
			nn.Linear(1, 8),
			nn.LayerNorm(8),
			nn.Mish(inplace=False),
		)
		self.fusion = nn.Sequential(
			nn.Linear(48, 64),
			nn.LayerNorm(64),
			nn.Mish(inplace=False),
			nn.Linear(64, 64),
			nn.LayerNorm(64),
			nn.Mish(inplace=False),
		)
		self.raw_residual = nn.Sequential(
			nn.Linear(self.task_vec_dim, 64),
			nn.LayerNorm(64),
			nn.Mish(inplace=False),
			nn.Linear(64, self.task_dim),
		)
		self.decoder = nn.Sequential(
			nn.Linear(self.task_dim, 64),
			nn.LayerNorm(64),
			nn.Mish(inplace=False),
			nn.Linear(64, self.task_vec_dim),
		)
		self.reset_repair_to_identity()

	def set_normalization_stats(self, mean: torch.Tensor, std: torch.Tensor, eps: float | None = None) -> None:
		mean = mean.detach().to(device=self._norm_mean.device, dtype=self._norm_mean.dtype).reshape(-1)
		std = std.detach().to(device=self._norm_std.device, dtype=self._norm_std.dtype).reshape(-1)
		if mean.numel() != self.task_vec_dim or std.numel() != self.task_vec_dim:
			raise ValueError(
				f"Expected normalization stats with {self.task_vec_dim} values, "
				f"got mean={mean.numel()} std={std.numel()}."
			)
		self._norm_mean.copy_(mean)
		self._norm_std.copy_(std.clamp_min(float(self.normalization_eps if eps is None else eps)))
		if eps is not None:
			self.normalization_eps = float(eps)

	def normalize_task_vec(self, task_vec_6: torch.Tensor) -> torch.Tensor:
		if not self.normalize_inputs:
			return task_vec_6
		mean = self._norm_mean.to(device=task_vec_6.device, dtype=task_vec_6.dtype)
		std = self._norm_std.to(device=task_vec_6.device, dtype=task_vec_6.dtype)
		return (task_vec_6 - mean) / std.clamp_min(self.normalization_eps)

	def reset_repair_to_identity(self) -> None:
		last = self.raw_residual[-1]
		nn.init.zeros_(last.weight)
		if last.bias is not None:
			nn.init.zeros_(last.bias)

	def _encode_axial(self, task_vec_6: torch.Tensor) -> torch.Tensor:
		task_type = task_vec_6[..., 0].round().long().clamp(0, 1)
		type_emb = self.type_encoder(task_type)
		metric_emb = self.metric_encoder(task_vec_6[..., 1:5])
		yaw_emb = self.yaw_encoder(task_vec_6[..., 5:6])
		return self.fusion(torch.cat([type_emb, metric_emb, yaw_emb], dim=-1))

	def axial_context(self, task_vec_6: torch.Tensor) -> torch.Tensor:
		if task_vec_6.shape[-1] != self.task_vec_dim:
			raise ValueError(f"Expected task_vec_6 with last dim {self.task_vec_dim}, got shape={tuple(task_vec_6.shape)}.")
		task_vec_6 = self.normalize_task_vec(task_vec_6.to(dtype=torch.float32))
		return self._encode_axial(task_vec_6)

	def forward_parts(self, task_vec_6: torch.Tensor) -> dict[str, torch.Tensor]:
		if task_vec_6.shape[-1] != self.task_vec_dim:
			raise ValueError(f"Expected task_vec_6 with last dim {self.task_vec_dim}, got shape={tuple(task_vec_6.shape)}.")
		task_vec_6 = self.normalize_task_vec(task_vec_6.to(dtype=torch.float32))
		axial_context = self._encode_axial(task_vec_6)
		raw_residual = self.raw_residual(task_vec_6)
		if self.repair_enabled and self.raw_residual_scale != 0.0:
			task_context = axial_context + self.raw_residual_scale * raw_residual
		else:
			task_context = axial_context
		return {
			"task_vec_norm": task_vec_6,
			"axial_context": axial_context,
			"raw_residual": raw_residual,
			"task_context": task_context,
		}

	def forward(self, task_vec_6: torch.Tensor) -> torch.Tensor:
		return self.forward_parts(task_vec_6)["task_context"]

	def reconstruct(self, task_context: torch.Tensor) -> torch.Tensor:
		if task_context.shape[-1] != self.task_dim:
			raise ValueError(f"Expected task_context with last dim {self.task_dim}, got shape={tuple(task_context.shape)}.")
		return self.decoder(task_context.to(dtype=torch.float32))
