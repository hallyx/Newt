#!/usr/bin/env python3
"""Shared read-only helpers for SRSA task consistency checks."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
TDMPC2_ROOT = SCRIPT_DIR.parents[1]
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "task_consistency"


def resolve_path(path_value: str | Path, *, base_dir: Path | None = None) -> Path:
	path = Path(path_value).expanduser()
	if path.is_absolute():
		return path.resolve()
	if base_dir is not None:
		candidate = (base_dir / path).resolve()
		if candidate.exists():
			return candidate
	return (REPO_ROOT / path).resolve()


def safe_name(value: str | Path) -> str:
	text = Path(value).stem
	text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
	return text or "replay"


def task_vec_hash(vec: Any) -> str:
	rounded = [round(float(item), 8) for item in list(vec)]
	text = ",".join(f"{item:.8g}" for item in rounded)
	return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def load_replay_snapshot(path: str | Path) -> tuple[Path, dict[str, Any], Any]:
	replay_path = resolve_path(path)
	payload = torch.load(replay_path, map_location="cpu", weights_only=False)
	if isinstance(payload, dict) and payload.get("format") == "newt_buffer_snapshot_v1":
		return replay_path, dict(payload.get("metadata", {}) or {}), payload.get("data")
	return replay_path, {}, payload


def get_task_tensor(data: Any):
	if data is None:
		return None
	if hasattr(data, "keys") and "task" in data.keys():
		try:
			return data.get("task")
		except TypeError:
			return data["task"]
	if isinstance(data, dict):
		return data.get("task")
	return None


def metadata_task_fields(metadata: dict[str, Any]) -> dict[str, Any]:
	keys = ("task_id", "assembly_id", "template_id", "condition_id", "role")
	return {key: metadata.get(key) for key in keys}


def summarize_task_tensor(task: torch.Tensor | None) -> dict[str, Any]:
	if task is None:
		return {
			"exists": False,
			"shape": None,
			"per_transition_shape": None,
			"is_floating_point": False,
			"last_dim": None,
			"unique_count": 0,
			"unique_task_vecs": [],
		}
	shape = list(task.shape)
	per_transition_shape = list(task.shape[1:]) if task.ndim >= 2 else []
	last_dim = int(task.shape[-1]) if task.ndim > 0 else None
	summary = {
		"exists": True,
		"shape": shape,
		"per_transition_shape": per_transition_shape,
		"is_floating_point": bool(torch.is_tensor(task) and task.is_floating_point()),
		"last_dim": last_dim,
		"unique_count": 0,
		"unique_task_vecs": [],
	}
	if not torch.is_tensor(task) or task.numel() == 0:
		return summary
	if task.ndim == 0:
		flat = task.reshape(1, 1)
	elif task.ndim == 1:
		flat = task.reshape(-1, 1)
	else:
		flat = task.reshape(-1, task.shape[-1])
	flat = flat.detach().cpu()
	if flat.is_floating_point():
		flat_for_unique = flat.float()
	else:
		flat_for_unique = flat.to(torch.float32)
	unique, inverse, counts = torch.unique(
		flat_for_unique,
		dim=0,
		return_inverse=True,
		return_counts=True,
	)
	items = []
	for index, vec in enumerate(unique):
		values = [float(x) for x in vec.tolist()]
		items.append({
			"hash": task_vec_hash(values),
			"count": int(counts[index].item()),
			"values": values[:6],
			"full_values": values,
		})
	items.sort(key=lambda item: (-int(item["count"]), str(item["hash"])))
	summary["unique_count"] = int(unique.shape[0])
	summary["unique_task_vecs"] = items
	summary["hash_counts"] = {item["hash"]: int(item["count"]) for item in items}
	return summary


def compare_metadata_task_shape(metadata: dict[str, Any], task_summary: dict[str, Any]) -> dict[str, Any]:
	expected = metadata.get("task_shape", None)
	if expected is None:
		return {
			"present": False,
			"matches": None,
			"metadata_task_shape": None,
			"actual_per_transition_shape": task_summary.get("per_transition_shape"),
		}
	actual = task_summary.get("per_transition_shape")
	expected_list = list(expected) if isinstance(expected, (list, tuple)) else [expected]
	return {
		"present": True,
		"matches": list(expected_list) == list(actual or []),
		"metadata_task_shape": expected_list,
		"actual_per_transition_shape": actual,
	}


def manifest_entries(payload: Any) -> list[dict[str, Any]]:
	if isinstance(payload, list):
		return [entry for entry in payload if isinstance(entry, dict)]
	if not isinstance(payload, dict):
		return []
	for key in ("tasks", "replays", "conditions"):
		value = payload.get(key)
		if isinstance(value, list):
			return [entry for entry in value if isinstance(entry, dict)]
	return []


def load_manifest(path: str | Path) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
	manifest_path = resolve_path(path)
	with open(manifest_path, "r", encoding="utf-8") as f:
		payload = json.load(f)
	return manifest_path, payload, manifest_entries(payload)


def replay_path_from_entry(entry: dict[str, Any], manifest_path: Path) -> Path | None:
	raw = entry.get("replay_fp") or entry.get("buffer_fp") or entry.get("path")
	if not raw:
		return None
	return resolve_path(raw, base_dir=manifest_path.parent)


def status_from_messages(messages: list[dict[str, str]]) -> str:
	levels = {str(item.get("level", "")).upper() for item in messages}
	if "FAIL" in levels:
		return "FAIL"
	if "WARNING" in levels:
		return "WARNING"
	return "PASS"


def add_message(messages: list[dict[str, str]], level: str, message: str, **extra: Any) -> None:
	item = {"level": str(level).upper(), "message": str(message)}
	for key, value in extra.items():
		item[key] = value
	messages.append(item)


def print_status(status: str, messages: list[dict[str, str]]) -> None:
	print(status)
	for item in messages:
		level = item.get("level", "INFO")
		message = item.get("message", "")
		print(f"[{level}] {message}")


def write_json_report(report: dict[str, Any], output_path: Path, *, dry_run: bool) -> None:
	if dry_run:
		print(f"[dry-run] would write JSON report: {output_path}")
		return
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with open(output_path, "w", encoding="utf-8") as f:
		json.dump(report, f, ensure_ascii=False, indent=2)
		f.write("\n")
	print(f"Wrote JSON report: {output_path}")


def write_csv(rows: list[dict[str, Any]], output_path: Path, *, dry_run: bool) -> None:
	if dry_run:
		print(f"[dry-run] would write CSV report: {output_path}")
		return
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fieldnames = []
	for row in rows:
		for key in row.keys():
			if key not in fieldnames:
				fieldnames.append(key)
	with open(output_path, "w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		for row in rows:
			writer.writerow(row)
	print(f"Wrote CSV report: {output_path}")


def entropy_norm_from_counts(counts: dict[str, int]) -> float:
	total = float(sum(max(0, int(value)) for value in counts.values()))
	if total <= 0 or len(counts) <= 1:
		return 0.0
	probs = [max(0.0, float(value)) / total for value in counts.values() if int(value) > 0]
	entropy = -sum(p * math.log(max(p, 1.0e-12)) for p in probs)
	return float(entropy / math.log(float(len(probs)))) if len(probs) > 1 else 0.0


def task_vec_std(task: torch.Tensor | None) -> list[float] | None:
	if task is None or not torch.is_tensor(task) or task.ndim < 2 or task.shape[-1] != 6:
		return None
	flat = task.detach().cpu().float().reshape(-1, task.shape[-1])
	return [float(x) for x in flat.std(dim=0, unbiased=False).tolist()]


def counter_to_plain(counter: Counter | dict[Any, Any]) -> dict[str, int]:
	return {str(key): int(value) for key, value in dict(counter or {}).items()}
