#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path


def _load_manifest(path: Path):
	if not path.exists():
		return {
			"version": 1,
			"kind": "online_family_replay_manifest",
			"tasks": [],
		}
	with open(path, "r", encoding="utf-8") as f:
		payload = json.load(f)
	if "tasks" not in payload or not isinstance(payload["tasks"], list):
		payload["tasks"] = []
	return payload


def _display_path(path: Path, manifest_path: Path, absolute: bool):
	path = path.expanduser().resolve()
	if absolute:
		return str(path)
	try:
		return str(path.relative_to(manifest_path.parent.resolve()))
	except ValueError:
		return str(path)


def _read_snapshot_metadata(path: Path):
	try:
		import torch
		payload = torch.load(path, map_location="cpu", weights_only=False)
	except Exception:
		return {}
	if isinstance(payload, dict):
		return dict(payload.get("metadata", {}))
	return {}


def parse_args():
	parser = argparse.ArgumentParser(description="Update an online family replay manifest.")
	parser.add_argument("--manifest", required=True, help="Manifest JSON path to create/update.")
	parser.add_argument("--task-id", required=True, help="Logical task id, usually the assembly id.")
	parser.add_argument("--assembly-id", default=None, help="Assembly id. Defaults to --task-id.")
	parser.add_argument("--template-id", default=None, help="Template/parameter id for condition-level replay.")
	parser.add_argument("--condition-id", default=None, help="Explicit condition id. Defaults to assembly_id|template_id.")
	parser.add_argument("--replay-fp", required=True, help="Saved replay snapshot path.")
	parser.add_argument("--checkpoint", default=None, help="Stage checkpoint produced with this replay.")
	parser.add_argument("--stage-index", type=int, default=None, help="Stage index in the launcher curriculum.")
	parser.add_argument("--anchor-task-id", default="01125", help="Anchor task id.")
	parser.add_argument("--role", default=None, choices=["anchor", "history"], help="Manifest role. Defaults from task id.")
	parser.add_argument("--absolute-paths", action="store_true", help="Store absolute replay/checkpoint paths.")
	return parser.parse_args()


def main():
	args = parse_args()
	manifest_path = Path(args.manifest).expanduser().resolve()
	replay_path = Path(args.replay_fp).expanduser().resolve()
	if not replay_path.is_file():
		raise FileNotFoundError(f"Replay snapshot not found: {replay_path}")
	checkpoint_path = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else None
	if checkpoint_path is not None and not checkpoint_path.is_file():
		raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

	manifest_path.parent.mkdir(parents=True, exist_ok=True)
	payload = _load_manifest(manifest_path)
	now = _dt.datetime.now(_dt.timezone.utc).isoformat()
	metadata = _read_snapshot_metadata(replay_path)
	task_id = str(args.task_id)
	assembly_id = str(args.assembly_id or args.task_id)
	template_id = args.template_id
	if template_id is None:
		template_id = metadata.get("template_id", None)
	if template_id is not None:
		template_id = str(template_id)
	condition_id = args.condition_id or metadata.get("condition_id", None)
	if condition_id is None:
		condition_id = f"{assembly_id}|{template_id if template_id is not None else 'default'}"
	role = args.role or ("anchor" if task_id == str(args.anchor_task_id) else "history")
	entry = {
		"task_id": task_id,
		"assembly_id": assembly_id,
		"template_id": template_id,
		"condition_id": str(condition_id),
		"role": role,
		"replay_fp": _display_path(replay_path, manifest_path, args.absolute_paths),
		"updated_at": now,
	}
	if args.stage_index is not None:
		entry["stage_index"] = int(args.stage_index)
	if checkpoint_path is not None:
		entry["checkpoint"] = _display_path(checkpoint_path, manifest_path, args.absolute_paths)
	for key in [
		"num_episodes",
		"num_transitions",
		"horizon",
		"obs_shape",
		"action_shape",
		"task_shape",
		"task_vec_unique",
		"task_vec_hashes",
		"task_vec_unique_values",
	]:
		if key in metadata:
			entry[key] = metadata[key]

	tasks = [item for item in payload["tasks"] if str(item.get("condition_id", item.get("task_id"))) != str(condition_id)]
	tasks.append(entry)
	tasks.sort(key=lambda item: (0 if item.get("role") == "anchor" else 1, item.get("stage_index", 10**9), str(item.get("task_id"))))
	payload["tasks"] = tasks
	payload["updated_at"] = now
	payload["anchor_task_id"] = str(args.anchor_task_id)

	with open(manifest_path, "w", encoding="utf-8") as f:
		json.dump(payload, f, ensure_ascii=True, indent=2)
		f.write("\n")
	print(f"[manifest] updated {manifest_path} with task_id={task_id} replay={replay_path}")


if __name__ == "__main__":
	main()
