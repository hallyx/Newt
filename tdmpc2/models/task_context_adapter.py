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
		self._last_delta_l2_mean = None
		self._last_delta_l2_p95 = None
		self._last_relative_delta_norm = None
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
		delta = self.alpha * (gamma * x + beta)
		compiler = getattr(torch, "compiler", None)
		is_compiling = False
		if compiler is not None and hasattr(compiler, "is_compiling"):
			is_compiling = bool(compiler.is_compiling())
		if not is_compiling:
			with torch.no_grad():
				delta_f = delta.detach().float()
				x_f = x.detach().float()
				if delta_f.ndim == 1:
					delta_norm = delta_f.norm().view(1)
					x_norm = x_f.norm().view(1)
				else:
					delta_norm = delta_f.reshape(-1, delta_f.shape[-1]).norm(dim=-1)
					x_norm = x_f.reshape(-1, x_f.shape[-1]).norm(dim=-1)
				self._last_delta_l2_mean = delta_norm.mean()
				self._last_delta_l2_p95 = torch.quantile(delta_norm, 0.95)
				self._last_relative_delta_norm = delta_norm.mean() / x_norm.mean().clamp_min(1.0e-6)
		return x + delta

	def metrics(self):
		final = self.net[-1]
		out = {
			"final_weight_norm": final.weight.detach().norm(),
			"final_bias_norm": final.bias.detach().norm(),
			"alpha": torch.tensor(float(self.alpha), device=final.weight.device),
		}
		if self._last_delta_l2_mean is not None:
			out.update({
				"delta_l2_mean": self._last_delta_l2_mean.detach(),
				"delta_l2_p95": self._last_delta_l2_p95.detach(),
				"relative_delta_norm": self._last_relative_delta_norm.detach(),
			})
		return out
