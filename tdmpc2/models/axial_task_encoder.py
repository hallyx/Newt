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

	def __init__(self, task_dim: int = 64):
		super().__init__()
		if int(task_dim) != 64:
			raise ValueError(f"AxialTaskEncoder outputs 64 dims, got task_dim={task_dim}.")
		self.task_dim = int(task_dim)
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

	def forward(self, task_vec_6: torch.Tensor) -> torch.Tensor:
		if task_vec_6.shape[-1] != 6:
			raise ValueError(f"Expected task_vec_6 with last dim 6, got shape={tuple(task_vec_6.shape)}.")
		task_vec_6 = task_vec_6.to(dtype=torch.float32)
		task_type = task_vec_6[..., 0].round().long().clamp(0, 1)
		type_emb = self.type_encoder(task_type)
		metric_emb = self.metric_encoder(task_vec_6[..., 1:5])
		yaw_emb = self.yaw_encoder(task_vec_6[..., 5:6])
		return self.fusion(torch.cat([type_emb, metric_emb, yaw_emb], dim=-1))
