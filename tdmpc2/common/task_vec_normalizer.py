from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


class TaskVecNormalizer:
	"""Lightweight normalizer for axial task_vec_6 tensors."""

	def __init__(self, mean, std, eps: float = 1.0e-6):
		self.mean = torch.as_tensor(mean, dtype=torch.float32)
		self.std = torch.as_tensor(std, dtype=torch.float32)
		self.eps = float(eps)
		if self.mean.shape != self.std.shape:
			raise ValueError(f"mean/std shape mismatch: {tuple(self.mean.shape)} vs {tuple(self.std.shape)}")

	def to(self, device):
		self.mean = self.mean.to(device)
		self.std = self.std.to(device)
		return self

	def normalize(self, task_vec: torch.Tensor) -> torch.Tensor:
		mean = self.mean.to(device=task_vec.device, dtype=task_vec.dtype)
		std = self.std.to(device=task_vec.device, dtype=task_vec.dtype)
		return (task_vec - mean) / std.clamp_min(self.eps)

	def state_dict(self):
		return {
			"mean": [float(x) for x in self.mean.detach().cpu().tolist()],
			"std": [float(x) for x in self.std.detach().cpu().tolist()],
			"eps": float(self.eps),
		}

	@classmethod
	def from_state_dict(cls, payload: dict[str, Any]):
		return cls(payload["mean"], payload["std"], eps=float(payload.get("eps", 1.0e-6)))

	@classmethod
	def load(cls, path):
		path = Path(path).expanduser()
		with open(path, "r", encoding="utf-8") as f:
			payload = json.load(f)
		stats = payload.get("stats", payload)
		return cls.from_state_dict(stats)

	def save(self, path):
		path = Path(path).expanduser()
		path.parent.mkdir(parents=True, exist_ok=True)
		with open(path, "w", encoding="utf-8") as f:
			json.dump({"stats": self.state_dict()}, f, ensure_ascii=True, indent=2)
			f.write("\n")
		return path


def normalize_task_vec(task_vec: torch.Tensor, stats: dict[str, Any], eps: float = 1.0e-6) -> torch.Tensor:
	return TaskVecNormalizer(
		stats["mean"],
		stats["std"],
		eps=float(stats.get("eps", eps)),
	).normalize(task_vec)
