#!/usr/bin/env python3
"""Compute task_vec_6 normalization stats from replay snapshots or manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path_value, base_dir=None):
	path = Path(path_value).expanduser()
	if path.is_absolute():
		return path.resolve()
	if base_dir is not None:
		candidate = (base_dir / path).resolve()
		if candidate.exists():
			return candidate
	return (REPO_ROOT / path).resolve()


def _manifest_entries(payload):
	if isinstance(payload, list):
		return payload
	if isinstance(payload, dict):
		for key in ("tasks", "replays", "conditions"):
			if isinstance(payload.get(key), list):
				return payload[key]
	return []


def _load_replay_task_vectors(path: Path, unique_only: bool):
	payload = torch.load(path, map_location="cpu", weights_only=False)
	if isinstance(payload, dict) and payload.get("format") == "newt_buffer_snapshot_v1":
		td = payload["data"]
	else:
		td = payload
	if "task" not in td.keys():
		raise KeyError(f"Replay does not contain `task`: {path}")
	task = td.get("task").reshape(-1, td.get("task").shape[-1]).float()
	if unique_only:
		task = torch.unique(task, dim=0)
	return task


def _task_vec_from_entry(entry):
	for key in ("task_vec_6", "task_vec", "axial_task_vec_6", "axial_task_vec"):
		value = entry.get(key, None) if isinstance(entry, dict) else None
		if value is None:
			continue
		vec = torch.tensor([float(x) for x in value], dtype=torch.float32)
		if vec.numel() == 6:
			return vec.view(1, 6)
	return None


def _hash_vec(vec):
	rounded = [round(float(item), 8) for item in vec]
	text = ",".join(f"{item:.8g}" for item in rounded)
	return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _collect_vectors(args):
	vectors = []
	sources = []
	for replay in args.replay:
		path = _resolve(replay)
		task = _load_replay_task_vectors(path, unique_only=not args.all_transitions)
		vectors.append(task)
		sources.append({"type": "replay", "path": str(path), "vectors": int(task.shape[0])})

	for manifest in args.manifest:
		manifest_path = _resolve(manifest)
		with open(manifest_path, "r", encoding="utf-8") as f:
			payload = json.load(f)
		for entry in _manifest_entries(payload):
			if not isinstance(entry, dict):
				continue
			replay_fp = entry.get("replay_fp") or entry.get("buffer_fp") or entry.get("path")
			if replay_fp:
				replay_path = _resolve(replay_fp, base_dir=manifest_path.parent)
				task = _load_replay_task_vectors(replay_path, unique_only=not args.all_transitions)
				vectors.append(task)
				sources.append({
					"type": "manifest_replay",
					"manifest": str(manifest_path),
					"path": str(replay_path),
					"condition_id": entry.get("condition_id"),
					"vectors": int(task.shape[0]),
				})
				continue
			task = _task_vec_from_entry(entry)
			if task is not None:
				vectors.append(task)
				sources.append({
					"type": "manifest_task_vec",
					"manifest": str(manifest_path),
					"condition_id": entry.get("condition_id"),
					"vectors": 1,
				})
	if not vectors:
		raise RuntimeError("No task vectors found. Provide --replay or --manifest with task tensors/task_vec_6 fields.")
	return torch.cat(vectors, dim=0), sources


def run(args):
	task_vecs, sources = _collect_vectors(args)
	if task_vecs.shape[-1] != 6:
		raise ValueError(f"Expected task_vec_6, got shape={tuple(task_vecs.shape)}")
	unique = torch.unique(task_vecs.float(), dim=0)
	std = task_vecs.std(dim=0, unbiased=False).clamp_min(float(args.eps))
	report = {
		"num_vectors": int(task_vecs.shape[0]),
		"num_unique": int(unique.shape[0]),
		"stats": {
			"mean": [float(x) for x in task_vecs.mean(dim=0).tolist()],
			"std": [float(x) for x in std.tolist()],
			"min": [float(x) for x in task_vecs.min(dim=0).values.tolist()],
			"max": [float(x) for x in task_vecs.max(dim=0).values.tolist()],
			"eps": float(args.eps),
		},
		"task_vec_hashes": [_hash_vec(vec.tolist()) for vec in unique],
		"sources": sources,
	}
	output = _resolve(args.output) if args.output else None
	if output is not None:
		output.parent.mkdir(parents=True, exist_ok=True)
		with open(output, "w", encoding="utf-8") as f:
			json.dump(report, f, ensure_ascii=True, indent=2)
			f.write("\n")
		print(f"Wrote task vec stats: {output}")
	else:
		json.dump(report, sys.stdout, ensure_ascii=True, indent=2)
		print()
	return report


def build_parser():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--manifest", action="append", default=[], help="Replay/condition manifest JSON.")
	parser.add_argument("--replay", action="append", default=[], help="Replay snapshot .pt containing a task tensor.")
	parser.add_argument("--output", default="data/task_vec_stats.json", help="Output JSON stats path.")
	parser.add_argument("--eps", type=float, default=1.0e-6)
	parser.add_argument("--all-transitions", action="store_true", help="Use every transition instead of unique vectors per replay.")
	return parser


def main():
	run(build_parser().parse_args())


if __name__ == "__main__":
	main()
