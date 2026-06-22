import torch
import torch.nn as nn


class TaskContextFiLMAdapter(nn.Module):
	"""
	Zero-initialized task-conditioned FiLM residual for latent features.

	The adapter starts as an exact identity:
	    y = x + alpha * (gamma(task) * x + beta(task))
	where the final projection that produces gamma/beta is initialized to zero.
	This lets old checkpoints load into an initially equivalent model while
	creating a trainable path where task context can become indispensable.
	"""

	def __init__(
		self,
		feature_dim: int,
		task_dim: int,
		hidden_dim: int = 128,
		alpha: float = 1.0,
	):
		super().__init__()
		self.feature_dim = int(feature_dim)
		self.task_dim = int(task_dim)
		self.hidden_dim = int(hidden_dim)
		self.alpha = float(alpha)
		if self.feature_dim <= 0:
			raise ValueError(f"feature_dim must be positive, got {feature_dim}.")
		if self.task_dim <= 0:
			raise ValueError(f"task_dim must be positive, got {task_dim}.")
		if self.hidden_dim <= 0:
			raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
		self.net = nn.Sequential(
			nn.Linear(self.task_dim, self.hidden_dim),
			nn.LayerNorm(self.hidden_dim),
			nn.Mish(inplace=False),
			nn.Linear(self.hidden_dim, 2 * self.feature_dim),
		)
		self.reset_to_identity()

	def reset_to_identity(self):
		final = self.net[-1]
		nn.init.zeros_(final.weight)
		nn.init.zeros_(final.bias)

	def forward(self, x: torch.Tensor, task_context: torch.Tensor) -> torch.Tensor:
		if task_context is None:
			return x
		if task_context.shape[-1] != self.task_dim:
			raise ValueError(
				f"Expected task_context dim {self.task_dim}, got shape={tuple(task_context.shape)}."
			)
		params = self.net(task_context.to(device=x.device, dtype=x.dtype, non_blocking=True))
		gamma, beta = params.chunk(2, dim=-1)
		return x + self.alpha * (gamma * x + beta)

	def metrics(self):
		final = self.net[-1]
		return {
			"final_weight_norm": final.weight.detach().norm(),
			"final_bias_norm": final.bias.detach().norm(),
			"alpha": torch.tensor(float(self.alpha), device=final.weight.device),
		}
