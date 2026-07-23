#!/usr/bin/env python3
"""Held-out offline diagnostics for Phase 4.2 single-family pretraining arms."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for path in (SCRIPT_DIR.parent, SCRIPT_DIR, SCRIPT_DIR / "phase3_three_task_pilot"):
	if str(path) not in sys.path:
		sys.path.insert(0, str(path))

import phase4_multitask_origin_train as phase4_train  # noqa: E402
import phase4_multitask_origin_offline_eval as phase4_eval  # noqa: E402


def resolve(value: str | Path) -> Path:
	path = Path(value).expanduser()
	return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def _parse_templates(value: str):
	return [[float(item) for item in pair.split(":")] for pair in value.split(";") if pair.strip()]


def _load_snapshot(path: Path):
	payload = torch.load(path, map_location="cpu", weights_only=False)
	if not isinstance(payload, dict) or "data" not in payload:
		raise RuntimeError(f"Expected Newt replay snapshot: {path}")
	return payload["data"], dict(payload.get("metadata", {}))


def _anchor_assignment(td, templates, clearance_base, depth_base, reference_radius, reference_depth):
	centers = torch.tensor([
		[0.5 * clearance_base * pair[0] / reference_radius, depth_base * pair[1] / reference_depth]
		for pair in templates
	], dtype=torch.float32)
	scale = centers.abs().mean(0).clamp_min(1.0e-6)
	episodes = td["episode"].detach().long().reshape(-1)
	assignment = {}
	for episode_id in torch.unique(episodes, sorted=True).tolist():
		idx = torch.nonzero(episodes == int(episode_id), as_tuple=False).reshape(-1)
		vec = td["task"][idx[0]].detach().float()[[2, 4]]
		distance = torch.linalg.vector_norm((centers - vec) / scale, dim=-1)
		assignment[int(episode_id)] = int(torch.argmin(distance).item())
	return assignment, centers


def _build_items(args, td):
	templates = _parse_templates(args.anchor_templates)
	assignment, centers = _anchor_assignment(
		td, templates, args.clearance_base, args.depth_base, args.reference_radius, args.reference_depth,
	)
	items = OrderedDict()
	for anchor_id, pair in enumerate(templates):
		label = f"c{pair[0]:g}_d{pair[1]:g}"
		episodes = torch.tensor([key for key, value in assignment.items() if value == anchor_id], dtype=torch.long)
		if episodes.numel() < 2:
			raise RuntimeError(f"Anchor {label} has only {episodes.numel()} episode(s); need train/held-out split.")
		generator = torch.Generator().manual_seed(args.split_seed + anchor_id)
		episodes = episodes[torch.randperm(episodes.numel(), generator=generator)]
		cut = min(max(1, int(round(args.train_fraction * episodes.numel()))), int(episodes.numel()) - 1)
		train_eps, val_eps = episodes[:cut], episodes[cut:]
		phase, train_starts = phase4_train._valid_starts(td, train_eps, args.horizon)
		_, val_starts = phase4_train._valid_starts(td, val_eps, args.horizon)
		mask = torch.isin(td["episode"].detach().long().reshape(-1), episodes)
		mean_vec = td["task"][mask].detach().float().reshape(-1, 6).mean(0)
		items[label] = {
			"td": td,
			"task_vec": mean_vec,
			"phase": phase,
			"train_episodes": train_eps,
			"val_episodes": val_eps,
			"train_starts": train_starts,
			"val_starts": val_starts,
			"center": centers[anchor_id],
		}
	return items


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--arm", required=True)
	parser.add_argument("--checkpoint", required=True)
	parser.add_argument("--replay", required=True)
	parser.add_argument("--output", required=True)
	parser.add_argument("--config", default="configs/train/srsa_01125_imitation_relaxed.yaml")
	parser.add_argument("--anchor-templates", required=True)
	parser.add_argument("--clearance-base", type=float, default=0.000114)
	parser.add_argument("--depth-base", type=float, default=0.009)
	parser.add_argument("--reference-radius", type=float, default=0.003993)
	parser.add_argument("--reference-depth", type=float, default=0.015)
	parser.add_argument("--horizon", type=int, default=3)
	parser.add_argument("--train-fraction", type=float, default=.80)
	parser.add_argument("--split-seed", type=int, default=4250)
	parser.add_argument("--eval-seed", type=int, default=4251)
	parser.add_argument("--eval-per-cell", type=int, default=256)
	parser.add_argument("--eval-batch-size", type=int, default=256)
	parser.add_argument("--gpu-id", type=int, default=0)
	parser.add_argument("--dry-run", action="store_true")
	args = parser.parse_args()
	checkpoint, replay, output = resolve(args.checkpoint), resolve(args.replay), resolve(args.output)
	for path in (checkpoint, replay, resolve(args.config)):
		if not path.exists():
			raise FileNotFoundError(path)
	if args.dry_run:
		print("PASS dry-run")
		print(f"arm={args.arm} checkpoint={checkpoint} replay={replay}")
		print(f"anchors={args.anchor_templates}; episode-disjoint 80/20 held-out")
		return 0

	device = phase4_eval._require_cuda1(args.gpu_id)
	td, metadata = _load_snapshot(replay)
	items = _build_items(args, td)
	labels = tuple(items.keys())
	# Reuse the already validated Phase 4.0 calibration implementation with a
	# dynamic set of single-family anchors.
	phase4_eval.TASKS = labels
	for item in items.values():
		item["return_to_go"] = phase4_eval._discounted_return_to_go(td, .95)
	bank = phase4_eval._fixed_eval_bank(items, args.eval_per_cell, args.eval_seed, .95)
	model_args = SimpleNamespace(
		config=args.config, gpu_id=args.gpu_id, eval_batch_size=args.eval_batch_size,
	)
	model, cfg, compat = phase4_eval._load_model(checkpoint, model_args, device)
	report = {
		"status": "PASS",
		"phase": "4.2",
		"arm": args.arm,
		"checkpoint": str(checkpoint),
		"checkpoint_sha256": phase4_eval._sha256(checkpoint),
		"replay": str(replay),
		"replay_metadata": metadata,
		"device": {"physical": "cuda1", "visible": os.environ.get("CUDA_VISIBLE_DEVICES"), "logical": "cuda:0"},
		"heldout_contract": {
			"episode_disjoint_train_fraction": args.train_fraction,
			"anchors": list(labels),
			"phase_label": "phase of first supervised transition in native horizon-3 sequence",
		},
		"anchor_counts": {
			label: {
				"train_episodes": int(item["train_episodes"].numel()),
				"heldout_episodes": int(item["val_episodes"].numel()),
			}
			for label, item in items.items()
		},
		"architecture_compatibility": compat,
		"task_context_structure": phase4_eval._task_context_report(model, items, device),
		"world_model": phase4_eval._cell_report(model, cfg, items, bank, device, args.eval_batch_size),
	}
	output.parent.mkdir(parents=True, exist_ok=True)
	output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
	print(f"[phase4.2] offline {args.arm}: wrote {output}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
