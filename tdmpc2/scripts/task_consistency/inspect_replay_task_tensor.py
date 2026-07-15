#!/usr/bin/env python3
"""Inspect the task tensor stored in a Newt replay snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import (
	DEFAULT_OUTPUT_DIR,
	add_message,
	compare_metadata_task_shape,
	get_task_tensor,
	load_replay_snapshot,
	metadata_task_fields,
	print_status,
	safe_name,
	status_from_messages,
	summarize_task_tensor,
	write_json_report,
)


def build_report(args):
	replay_path, metadata, data = load_replay_snapshot(args.replay)
	task = get_task_tensor(data)
	task_summary = summarize_task_tensor(task)
	shape_check = compare_metadata_task_shape(metadata, task_summary)
	messages = []

	if not task_summary["exists"]:
		add_message(messages, "FAIL", "Replay snapshot has no data['task'] tensor.", replay_fp=str(replay_path))
	elif task_summary["last_dim"] != 6:
		add_message(
			messages,
			"FAIL",
			f"Expected task tensor last dim 6, got {task_summary['last_dim']}.",
			replay_fp=str(replay_path),
		)
	else:
		add_message(messages, "PASS", "Replay task tensor exists with last dim 6.", replay_fp=str(replay_path))

	if not task_summary.get("is_floating_point", False):
		add_message(messages, "WARNING", "Task tensor is not floating point; this is not the SRSA task_vec_6 main path.")

	if shape_check["present"] and not shape_check["matches"]:
		add_message(
			messages,
			"FAIL",
			"metadata['task_shape'] does not match actual per-transition task shape.",
			metadata_task_shape=shape_check["metadata_task_shape"],
			actual_per_transition_shape=shape_check["actual_per_transition_shape"],
		)
	elif shape_check["present"]:
		add_message(messages, "PASS", "metadata['task_shape'] matches actual per-transition task shape.")
	else:
		add_message(messages, "WARNING", "metadata['task_shape'] is missing; actual task tensor was still inspected.")

	if task_summary["unique_count"] <= 0:
		add_message(messages, "FAIL", "No unique task vectors could be computed.")
	elif task_summary["unique_count"] > 1:
		add_message(
			messages,
			"WARNING",
			f"Replay contains {task_summary['unique_count']} unique task vectors.",
			replay_fp=str(replay_path),
		)
	else:
		add_message(messages, "PASS", "Replay contains exactly one unique task vector.")

	status = status_from_messages(messages)
	return {
		"status": status,
		"replay_fp": str(replay_path),
		"metadata_task_fields": metadata_task_fields(metadata),
		"metadata_task_shape_check": shape_check,
		"task_tensor": task_summary,
		"messages": messages,
	}


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("replay", help="Path to a replay snapshot .pt file.")
	parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for JSON reports.")
	parser.add_argument("--output", default=None, help="Optional explicit output JSON path.")
	parser.add_argument("--dry-run", action="store_true", help="Run checks and print output path without writing files.")
	args = parser.parse_args()

	report = build_report(args)
	output_path = Path(args.output).expanduser() if args.output else Path(args.output_dir).expanduser() / f"{safe_name(report['replay_fp'])}_task_tensor_report.json"
	if not output_path.is_absolute():
		output_path = DEFAULT_OUTPUT_DIR.parent.parent / output_path
	print_status(report["status"], report["messages"])
	write_json_report(report, output_path.resolve(), dry_run=args.dry_run)
	return 1 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
	raise SystemExit(main())
