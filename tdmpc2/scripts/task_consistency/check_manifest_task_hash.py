#!/usr/bin/env python3
"""Check task_id / assembly_id / template_id consistency against replay task hashes."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from _common import (
	DEFAULT_OUTPUT_DIR,
	add_message,
	compare_metadata_task_shape,
	get_task_tensor,
	load_manifest,
	load_replay_snapshot,
	metadata_task_fields,
	print_status,
	replay_path_from_entry,
	status_from_messages,
	summarize_task_tensor,
	task_vec_hash,
	write_csv,
	write_json_report,
)


def _entry_label(entry, key, default=""):
	value = entry.get(key, default)
	if value is None:
		return default
	return str(value)


def _primary_hash(task_summary):
	items = task_summary.get("unique_task_vecs", [])
	if len(items) != 1:
		return None
	return str(items[0]["hash"])


def _primary_values(task_summary):
	items = task_summary.get("unique_task_vecs", [])
	if len(items) != 1:
		return None
	return items[0].get("values")


def inspect_entry(entry, manifest_path: Path):
	replay_path = replay_path_from_entry(entry, manifest_path)
	if replay_path is None:
		return {
			"entry": dict(entry),
			"replay_fp": None,
			"error": "missing replay_fp/buffer_fp/path",
			"task_summary": None,
		}
	replay_path, replay_metadata, data = load_replay_snapshot(replay_path)
	task_summary = summarize_task_tensor(get_task_tensor(data))
	shape_check = compare_metadata_task_shape(replay_metadata, task_summary)
	merged_metadata = dict(replay_metadata)
	for key in ("task_id", "assembly_id", "template_id", "condition_id", "role"):
		if entry.get(key, None) is not None:
			merged_metadata[key] = entry.get(key)
	return {
		"entry": dict(entry),
		"replay_fp": str(replay_path),
		"replay_metadata_task_fields": metadata_task_fields(replay_metadata),
		"effective_task_fields": metadata_task_fields(merged_metadata),
		"metadata_task_shape_check": shape_check,
		"task_summary": task_summary,
		"primary_hash": _primary_hash(task_summary),
		"primary_values": _primary_values(task_summary),
	}


def build_report(args):
	manifest_path, payload, entries = load_manifest(args.manifest)
	messages = []
	if not entries:
		add_message(messages, "FAIL", "Manifest has no replay entries.", manifest_fp=str(manifest_path))

	entry_reports = []
	task_to_hashes = defaultdict(set)
	hash_to_tasks = defaultdict(set)
	condition_to_hashes = defaultdict(set)
	assembly_template_to_hashes = defaultdict(set)
	csv_rows = []

	for index, entry in enumerate(entries):
		item = inspect_entry(entry, manifest_path)
		entry_reports.append(item)
		if item.get("error"):
			add_message(
				messages,
				"FAIL",
				item["error"],
				entry_index=index,
				task_id=_entry_label(entry, "task_id"),
				assembly_id=_entry_label(entry, "assembly_id"),
			)
			continue

		fields = item["effective_task_fields"]
		task_id = _entry_label(fields, "task_id", f"entry_{index}")
		assembly_id = _entry_label(fields, "assembly_id", task_id)
		template_id = _entry_label(fields, "template_id", "default")
		condition_id = _entry_label(fields, "condition_id", f"{assembly_id}|{template_id}")
		role = _entry_label(fields, "role", "")
		task_summary = item["task_summary"]
		unique_items = task_summary.get("unique_task_vecs", [])

		if not task_summary.get("exists", False):
			add_message(messages, "FAIL", "Replay has no task tensor.", task_id=task_id, replay_fp=item["replay_fp"])
		elif task_summary.get("last_dim") != 6:
			add_message(
				messages,
				"FAIL",
				f"Replay task tensor last dim is {task_summary.get('last_dim')}, expected 6.",
				task_id=task_id,
				replay_fp=item["replay_fp"],
			)
		if item["metadata_task_shape_check"]["present"] and not item["metadata_task_shape_check"]["matches"]:
			add_message(
				messages,
				"FAIL",
				"Replay metadata task_shape mismatch.",
				task_id=task_id,
				replay_fp=item["replay_fp"],
			)
		if len(unique_items) != 1:
			add_message(
				messages,
				"FAIL",
				f"Replay entry maps to {len(unique_items)} task hashes; expected exactly 1.",
				task_id=task_id,
				assembly_id=assembly_id,
				replay_fp=item["replay_fp"],
			)

		for unique in unique_items:
			hash_value = str(unique["hash"])
			task_to_hashes[task_id].add(hash_value)
			hash_to_tasks[hash_value].add(task_id)
			condition_to_hashes[condition_id].add(hash_value)
			assembly_template_to_hashes[f"{assembly_id}|{template_id}"].add(hash_value)
			csv_rows.append({
				"task_id": task_id,
				"assembly_id": assembly_id,
				"template_id": template_id,
				"condition_id": condition_id,
				"role": role,
				"replay_fp": item["replay_fp"],
				"task_hash": hash_value,
				"count": int(unique.get("count", 0)),
				"values": json.dumps(unique.get("values", []), ensure_ascii=False),
				"unique_hash_count_for_replay": int(len(unique_items)),
			})

	for task_id, hashes in sorted(task_to_hashes.items()):
		if len(hashes) > 1:
			add_message(
				messages,
				"FAIL",
				"Same task_id maps to multiple task hashes.",
				task_id=task_id,
				hashes=sorted(hashes),
			)
	for hash_value, task_ids in sorted(hash_to_tasks.items()):
		if len(task_ids) > 1:
			add_message(
				messages,
				"FAIL",
				"Different task_id values map to the same task hash.",
				task_hash=hash_value,
				task_ids=sorted(task_ids),
			)
	for condition_id, hashes in sorted(condition_to_hashes.items()):
		if len(hashes) > 1:
			add_message(
				messages,
				"FAIL",
				"Same condition_id maps to multiple task hashes.",
				condition_id=condition_id,
				hashes=sorted(hashes),
			)
	for key, hashes in sorted(assembly_template_to_hashes.items()):
		if len(hashes) > 1:
			add_message(
				messages,
				"FAIL",
				"Same assembly_id/template_id maps to multiple task hashes.",
				assembly_template=key,
				hashes=sorted(hashes),
			)

	for pair in args.require_distinct_task_ids:
		parts = [part.strip() for part in pair.split(",") if part.strip()]
		if len(parts) != 2:
			add_message(messages, "WARNING", f"Ignoring invalid --require-distinct-task-ids item: {pair!r}")
			continue
		a, b = parts
		if a in task_to_hashes and b in task_to_hashes:
			if task_to_hashes[a] & task_to_hashes[b]:
				add_message(
					messages,
					"FAIL",
					"Required distinct task ids share at least one task hash.",
					task_ids=[a, b],
					shared_hashes=sorted(task_to_hashes[a] & task_to_hashes[b]),
				)
			else:
				add_message(messages, "PASS", "Required distinct task ids have different task hashes.", task_ids=[a, b])
		else:
			add_message(messages, "WARNING", "Required distinct task ids are not both present.", task_ids=[a, b])

	if not any(item.get("level") == "FAIL" for item in messages):
		add_message(messages, "PASS", "Manifest task hash consistency checks passed.", manifest_fp=str(manifest_path))

	status = status_from_messages(messages)
	return {
		"status": status,
		"manifest_fp": str(manifest_path),
		"num_entries": len(entries),
		"task_id_to_hashes": {task_id: sorted(hashes) for task_id, hashes in sorted(task_to_hashes.items())},
		"hash_to_task_ids": {hash_value: sorted(task_ids) for hash_value, task_ids in sorted(hash_to_tasks.items())},
		"condition_id_to_hashes": {key: sorted(value) for key, value in sorted(condition_to_hashes.items())},
		"assembly_template_to_hashes": {key: sorted(value) for key, value in sorted(assembly_template_to_hashes.items())},
		"entries": entry_reports,
		"messages": messages,
		"csv_rows": csv_rows,
	}


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("manifest", help="Path to an online-family replay manifest JSON.")
	parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for reports.")
	parser.add_argument(
		"--require-distinct-task-ids",
		action="append",
		default=["01125,00256"],
		help="Comma-separated pair of task ids that must not share a task hash. Can be repeated.",
	)
	parser.add_argument("--dry-run", action="store_true", help="Run checks without writing reports.")
	args = parser.parse_args()

	report = build_report(args)
	output_dir = Path(args.output_dir).expanduser()
	if not output_dir.is_absolute():
		output_dir = DEFAULT_OUTPUT_DIR.parent.parent / output_dir
	output_dir = output_dir.resolve()
	csv_rows = report.pop("csv_rows")
	print_status(report["status"], report["messages"])
	write_json_report(report, output_dir / "manifest_task_hash_report.json", dry_run=args.dry_run)
	write_csv(csv_rows, output_dir / "task_id_to_hash.csv", dry_run=args.dry_run)
	return 1 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
	raise SystemExit(main())
