#!/usr/bin/env python3
"""Sample OnlineFamilyReplayBuffer batches and compare task labels with task hashes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from _common import (
	DEFAULT_OUTPUT_DIR,
	TDMPC2_ROOT,
	add_message,
	counter_to_plain,
	entropy_norm_from_counts,
	get_task_tensor,
	load_manifest,
	load_replay_snapshot,
	print_status,
	replay_path_from_entry,
	status_from_messages,
	summarize_task_tensor,
	task_vec_std,
	write_json_report,
)

if str(TDMPC2_ROOT) not in sys.path:
	sys.path.insert(0, str(TDMPC2_ROOT))

from common.buffer import Buffer  # noqa: E402
from common.online_family_replay import OnlineFamilyReplayBuffer  # noqa: E402


class ConfigDict(dict):
	def __getattr__(self, key):
		try:
			return self[key]
		except KeyError as exc:
			raise AttributeError(key) from exc

	def __setattr__(self, key, value):
		self[key] = value


def _single_hash_from_replay(path):
	replay_path, metadata, data = load_replay_snapshot(path)
	summary = summarize_task_tensor(get_task_tensor(data))
	hashes = [str(item["hash"]) for item in summary.get("unique_task_vecs", [])]
	return replay_path, metadata, summary, set(hashes)


def _label_from_metadata(metadata, fallback):
	return str(metadata.get("task_id") or metadata.get("assembly_id") or fallback)


def _build_label_hash_map(current_replay, manifest):
	label_to_hashes = {}
	replay_summaries = []
	current_path, current_metadata, current_summary, current_hashes = _single_hash_from_replay(current_replay)
	current_label = _label_from_metadata(current_metadata, current_path.stem)
	label_to_hashes.setdefault(current_label, set()).update(current_hashes)
	replay_summaries.append({
		"role": "current",
		"task_id": current_label,
		"replay_fp": str(current_path),
		"hashes": sorted(current_hashes),
		"unique_count": int(current_summary.get("unique_count", 0)),
	})

	manifest_path, _, entries = load_manifest(manifest)
	for entry in entries:
		replay_path = replay_path_from_entry(entry, manifest_path)
		if replay_path is None:
			continue
		try:
			path, metadata, summary, hashes = _single_hash_from_replay(replay_path)
		except Exception as exc:
			replay_summaries.append({
				"role": str(entry.get("role", "")),
				"task_id": str(entry.get("task_id", "")),
				"replay_fp": str(replay_path),
				"error": str(exc),
			})
			continue
		label = str(entry.get("task_id") or entry.get("assembly_id") or metadata.get("task_id") or metadata.get("assembly_id") or path.stem)
		label_to_hashes.setdefault(label, set()).update(hashes)
		replay_summaries.append({
			"role": str(entry.get("role", metadata.get("role", ""))),
			"task_id": label,
			"assembly_id": str(entry.get("assembly_id", metadata.get("assembly_id", label))),
			"condition_id": str(entry.get("condition_id", metadata.get("condition_id", ""))),
			"replay_fp": str(path),
			"hashes": sorted(hashes),
			"unique_count": int(summary.get("unique_count", 0)),
		})
	return {key: sorted(value) for key, value in label_to_hashes.items()}, replay_summaries, current_metadata


def _expected_hash_counts(label_counts, label_to_hashes):
	expected = {}
	missing = {}
	ambiguous = {}
	for label, count in label_counts.items():
		hashes = label_to_hashes.get(str(label), [])
		if not hashes:
			missing[str(label)] = int(count)
			continue
		if len(hashes) != 1:
			ambiguous[str(label)] = hashes
			continue
		expected[hashes[0]] = expected.get(hashes[0], 0) + int(count)
	return expected, missing, ambiguous


def build_cfg(args, current_metadata):
	current_task_id = args.current_task_id or current_metadata.get("task_id") or current_metadata.get("assembly_id") or Path(args.current_replay).stem
	current_template_id = args.current_template_id or current_metadata.get("template_id") or "default"
	current_condition_id = args.current_condition_id or current_metadata.get("condition_id") or f"{current_task_id}|{current_template_id}"
	return ConfigDict({
		"batch_size": int(args.batch_size),
		"horizon": int(args.horizon),
		"online_family_replay_manifest_fp": str(args.manifest),
		"online_family_replay_storage_device": str(args.storage_device),
		"online_family_current_task_id": str(current_task_id),
		"online_family_anchor_task_id": str(args.anchor_task_id),
		"online_family_current_template_id": str(current_template_id),
		"online_family_current_condition_id": str(current_condition_id),
		"online_family_current_ratio": float(args.current_ratio),
		"online_family_anchor_ratio": float(args.anchor_ratio),
		"online_family_history_ratio": float(args.history_ratio),
		"online_family_min_current_episodes": int(args.min_current_episodes),
		"online_family_replay_max_episodes_per_task": None,
		"online_family_sample_balance": str(args.sample_balance),
		"multi_task_bootstrap_min_episodes_per_condition": 0,
		"multi_task_bootstrap_current_only": False,
		"assembly_id": str(current_task_id),
	})


def build_report(args):
	messages = []
	label_to_hashes, replay_summaries, current_metadata = _build_label_hash_map(args.current_replay, args.manifest)
	cfg = build_cfg(args, current_metadata)
	current_buffer = Buffer.load(
		args.current_replay,
		cfg=cfg,
		storage_device=str(args.storage_device),
		batch_size=int(args.batch_size),
		horizon=int(args.horizon),
	)
	family = OnlineFamilyReplayBuffer.from_manifest(current_buffer, cfg)
	device = torch.device(args.device)
	batches = []

	for batch_index in range(int(args.num_batches)):
		obs, action, reward, task = family.sample(device=device, batch_size=int(args.batch_size))
		label_counts = counter_to_plain(getattr(family, "last_batch_task_counts", {}))
		hash_counts = counter_to_plain(getattr(family, "last_batch_task_hash_counts", {}))
		condition_counts = counter_to_plain(getattr(family, "last_batch_condition_counts", {}))
		expected_hash_counts, missing_labels, ambiguous_labels = _expected_hash_counts(label_counts, label_to_hashes)
		hash_match = expected_hash_counts == hash_counts if expected_hash_counts else False
		std = task_vec_std(task)
		max_std = max(std) if std else 0.0
		current_count = int(label_counts.get(str(cfg.online_family_current_task_id), 0))
		anchor_count = int(label_counts.get(str(cfg.online_family_anchor_task_id), 0))
		batch_report = {
			"batch_index": batch_index,
			"obs_shape": str(obs.shape) if hasattr(obs, "shape") else None,
			"action_shape": list(action.shape),
			"reward_shape": list(reward.shape),
			"task_shape": list(task.shape) if task is not None else None,
			"batch_task_label_counts": label_counts,
			"batch_task_hash_counts": hash_counts,
			"expected_hash_counts_from_labels": expected_hash_counts,
			"missing_label_hash_mapping": missing_labels,
			"ambiguous_label_hash_mapping": ambiguous_labels,
			"label_hash_counts_match": bool(hash_match),
			"task_vec_std": std,
			"task_vec_std_max": float(max_std),
			"condition_counts": condition_counts,
			"condition_entropy_norm": entropy_norm_from_counts(condition_counts),
			"has_current_task_label": current_count > 0,
			"has_anchor_task_label": anchor_count > 0,
			"num_task_hashes": len(hash_counts),
		}
		batches.append(batch_report)

		if not hash_match:
			add_message(
				messages,
				"FAIL",
				"Batch task label counts do not match actual task hash counts.",
				batch_index=batch_index,
				label_counts=label_counts,
				hash_counts=hash_counts,
				expected_hash_counts=expected_hash_counts,
			)
		if missing_labels:
			add_message(messages, "FAIL", "Missing label->hash mapping for batch labels.", batch_index=batch_index, labels=missing_labels)
		if ambiguous_labels:
			add_message(messages, "FAIL", "Ambiguous label->hash mapping for batch labels.", batch_index=batch_index, labels=ambiguous_labels)
		if max_std <= float(args.std_eps):
			add_message(messages, "FAIL", "task_vec_std is zero or too small for a mixed batch.", batch_index=batch_index, task_vec_std=std)
		if current_count <= 0:
			add_message(messages, "FAIL", "Mixed batch does not contain current task label.", batch_index=batch_index, current_task_id=str(cfg.online_family_current_task_id))
		if anchor_count <= 0:
			add_message(messages, "FAIL", "Mixed batch does not contain anchor task label.", batch_index=batch_index, anchor_task_id=str(cfg.online_family_anchor_task_id))
		if len(hash_counts) < 2:
			add_message(messages, "FAIL", "Mixed batch contains fewer than two task hashes.", batch_index=batch_index, hash_counts=hash_counts)

	if not any(item.get("level") == "FAIL" for item in messages):
		add_message(messages, "PASS", "Online-family sampled batches have consistent task labels and task hashes.")

	status = status_from_messages(messages)
	return {
		"status": status,
		"config": dict(cfg),
		"manifest_fp": str(args.manifest),
		"current_replay_fp": str(args.current_replay),
		"label_to_hashes": label_to_hashes,
		"replay_summaries": replay_summaries,
		"batches": batches,
		"messages": messages,
	}


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--manifest", required=True, help="Path to online-family replay manifest JSON.")
	parser.add_argument("--current-replay", required=True, help="Path to current replay snapshot.")
	parser.add_argument("--current-task-id", default=None, help="Current task id. Defaults to current replay metadata.")
	parser.add_argument("--current-template-id", default=None, help="Current template id. Defaults to current replay metadata.")
	parser.add_argument("--current-condition-id", default=None, help="Current condition id. Defaults to current replay metadata.")
	parser.add_argument("--anchor-task-id", default="01125", help="Anchor task id expected in mixed batches.")
	parser.add_argument("--current-ratio", type=float, default=0.50)
	parser.add_argument("--anchor-ratio", type=float, default=0.50)
	parser.add_argument("--history-ratio", type=float, default=0.0)
	parser.add_argument("--sample-balance", default="ratio")
	parser.add_argument("--min-current-episodes", type=int, default=0)
	parser.add_argument("--batch-size", type=int, default=1024)
	parser.add_argument("--horizon", type=int, default=3)
	parser.add_argument("--num-batches", type=int, default=4)
	parser.add_argument("--device", default="cpu", help="Torch device for sampling. Use cpu for read-only checks.")
	parser.add_argument("--storage-device", default="cpu", help="Replay storage device.")
	parser.add_argument("--std-eps", type=float, default=1.0e-8)
	parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for reports.")
	parser.add_argument("--dry-run", action="store_true", help="Run checks without writing report.")
	args = parser.parse_args()

	report = build_report(args)
	output_dir = Path(args.output_dir).expanduser()
	if not output_dir.is_absolute():
		output_dir = DEFAULT_OUTPUT_DIR.parent.parent / output_dir
	output_dir = output_dir.resolve()
	print_status(report["status"], report["messages"])
	write_json_report(report, output_dir / "online_family_sample_mix_report.json", dry_run=args.dry_run)
	return 1 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
	raise SystemExit(main())
