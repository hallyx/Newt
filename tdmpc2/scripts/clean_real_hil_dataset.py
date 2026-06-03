from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import torch
from tensordict import TensorDict


SUCCESS_KEYS = (
	"episode_success_final",
	"success_episode",
	"episode_strict_success_stable_final",
	"episode_strict_success_episode_final",
	"episode_relaxed_success_stable_final",
	"episode_relaxed_success_episode_final",
	"terminal_success",
)

SUCCESS_FINAL_KEYS = (
	"episode_success_final",
	"success_episode",
	"episode_strict_success_stable_final",
	"episode_strict_success_episode_final",
	"episode_relaxed_success_stable_final",
	"episode_relaxed_success_episode_final",
)


def _keys(td) -> list[str]:
	return list(td.keys()) if hasattr(td, "keys") else list(td.keys())


def _load_tensordict(path: Path):
	td = torch.load(path, map_location="cpu", weights_only=False)
	if not hasattr(td, "keys"):
		raise TypeError(f"Expected a TensorDict-like dataset: {path}")
	for key in ("obs", "next_obs", "action", "done", "episode", "step_id"):
		if key not in td.keys():
			raise KeyError(f"Missing required HIL dataset key: {key}")
	return td


def _episode_index_map(td) -> dict[int, torch.Tensor]:
	episode = td["episode"].reshape(-1).to(torch.int64)
	step_id = td["step_id"].reshape(-1).to(torch.int64)
	out: dict[int, list[int]] = {}
	for row, episode_id in enumerate(episode.tolist()):
		out.setdefault(int(episode_id), []).append(row)
	ordered = {}
	for episode_id, rows in out.items():
		idx = torch.tensor(rows, dtype=torch.int64)
		order = torch.argsort(step_id[idx], stable=True)
		ordered[episode_id] = idx[order]
	return ordered


def _bool_rows(value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
	return value[indices].reshape(-1).bool()


def _float_rows(value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
	return value[indices].reshape(-1).float()


def _first_done_offset(td, indices: torch.Tensor) -> int | None:
	done = _bool_rows(td["done"], indices)
	found = torch.nonzero(done, as_tuple=False).reshape(-1)
	if found.numel() == 0:
		return None
	return int(found[0].item())


def _final_success(td, indices: torch.Tensor) -> tuple[bool, bool, str | None]:
	for key in SUCCESS_KEYS:
		if key not in td.keys():
			continue
		values = td[key][indices]
		if key == "terminal_success":
			return True, bool(values.reshape(-1).bool().any().item()), key
		return True, float(values.reshape(-1).float()[-1].item()) > 0.5, key
	return False, False, None


def _intervention_mask(td, indices: torch.Tensor) -> torch.Tensor:
	if "intervened" not in td.keys():
		return torch.zeros((int(indices.numel()),), dtype=torch.bool)
	return _bool_rows(td["intervened"], indices)


def _contact_mask(td, indices: torch.Tensor, *, force_tail_dim: int, threshold: float) -> torch.Tensor:
	if threshold <= 0 or force_tail_dim <= 0:
		return torch.zeros((int(indices.numel()),), dtype=torch.bool)
	obs = td["obs"][indices].float()
	if obs.ndim < 2 or obs.shape[-1] < force_tail_dim:
		return torch.zeros((int(indices.numel()),), dtype=torch.bool)
	force = obs[..., -force_tail_dim:]
	return torch.linalg.norm(force, dim=-1).reshape(-1) >= float(threshold)


def _finite_actions(td, indices: torch.Tensor, *, expected_action_dim: int | None) -> tuple[bool, str | None]:
	action = td["action"][indices]
	if expected_action_dim is not None and int(action.shape[-1]) != int(expected_action_dim):
		return False, f"action_dim_{int(action.shape[-1])}_expected_{int(expected_action_dim)}"
	if not torch.isfinite(action.float()).all().item():
		return False, "non_finite_action"
	return True, None


def _valid_obs(td, *, expected_obs_dim: int | None) -> tuple[bool, str | None]:
	if expected_obs_dim is not None and int(td["obs"].shape[-1]) != int(expected_obs_dim):
		return False, f"obs_dim_{int(td['obs'].shape[-1])}_expected_{int(expected_obs_dim)}"
	if not torch.isfinite(td["obs"].float()).all().item():
		return False, "non_finite_obs"
	if not torch.isfinite(td["next_obs"].float()).all().item():
		return False, "non_finite_next_obs"
	return True, None


def _choose_window(
	length: int,
	*,
	success: bool,
	signal_positions: list[int],
	max_episode_steps: int,
	success_pre_window: int,
	failure_pre_window: int,
	failure_post_window: int,
	drop_failure_without_signal: bool,
) -> tuple[int | None, int | None, str | None]:
	if length <= 0:
		return None, None, "empty_episode"
	max_episode_steps = max(1, int(max_episode_steps))
	if success:
		end = length - 1
		default_start = max(0, end - max(1, int(success_pre_window)) + 1)
		start = min(signal_positions[0], default_start) if signal_positions else default_start
		if end - start + 1 > max_episode_steps:
			start = max(0, end - max_episode_steps + 1)
		return start, end, None
	if not signal_positions:
		if drop_failure_without_signal:
			return None, None, "failure_without_intervention_or_contact"
		end = length - 1
		return max(0, end - max_episode_steps + 1), end, None
	start = max(0, signal_positions[0] - max(0, int(failure_pre_window)))
	end = min(length - 1, signal_positions[-1] + max(0, int(failure_post_window)))
	if end - start + 1 > max_episode_steps:
		end = min(length - 1, signal_positions[-1] + max(0, int(failure_post_window)))
		start = max(0, end - max_episode_steps + 1)
	return start, end, None


def _set_last_bool(chunk: dict[str, torch.Tensor], key: str, value: bool):
	if key not in chunk:
		return
	out = torch.zeros_like(chunk[key]).bool()
	out.reshape(-1)[-1] = bool(value)
	chunk[key] = out


def _set_all_float(chunk: dict[str, torch.Tensor], key: str, value: float):
	if key not in chunk:
		return
	chunk[key] = torch.full_like(chunk[key].float(), float(value))


def _rewrite_segment_labels(
	chunk: dict[str, torch.Tensor],
	*,
	new_episode_id: int,
	success: bool,
	reaches_raw_terminal: bool,
):
	length = int(chunk["obs"].shape[0])
	chunk["episode"] = torch.full_like(chunk["episode"].to(torch.int64), int(new_episode_id))
	chunk["step_id"] = torch.arange(length, dtype=chunk["step_id"].dtype).reshape_as(chunk["step_id"])
	_set_last_bool(chunk, "done", True)
	_set_last_bool(chunk, "terminated", bool(reaches_raw_terminal and success))
	_set_last_bool(chunk, "truncated", not reaches_raw_terminal)
	_set_last_bool(chunk, "terminal_success", success)
	_set_last_bool(chunk, "terminal_failure", not success)
	for key in SUCCESS_FINAL_KEYS:
		_set_all_float(chunk, key, 1.0 if success else 0.0)
	_set_all_float(chunk, "episode_failure_final", 0.0 if success else 1.0)
	if "episode_return_running" in chunk and "episode_return_final" in chunk:
		final_return = float(chunk["episode_return_running"].reshape(-1)[-1].item())
		_set_all_float(chunk, "episode_return_final", final_return)


def _slice_chunk(td, keys: list[str], indices: torch.Tensor) -> dict[str, torch.Tensor]:
	chunk = {}
	for key in keys:
		value = td[key]
		if int(value.shape[0]) != int(td["obs"].shape[0]):
			continue
		chunk[key] = value[indices].detach().cpu().contiguous().clone()
	return chunk


def _length_stats(lengths: list[int]) -> dict:
	if not lengths:
		return {"count": 0}
	values = sorted(int(x) for x in lengths)
	def percentile(q: float) -> int:
		if len(values) == 1:
			return values[0]
		index = round((len(values) - 1) * q)
		return values[int(index)]
	return {
		"count": len(values),
		"min": values[0],
		"max": values[-1],
		"mean": float(mean(values)),
		"p50": percentile(0.50),
		"p90": percentile(0.90),
		"p95": percentile(0.95),
		"p99": percentile(0.99),
	}


def _dataset_counts(td) -> dict:
	done = td["done"].reshape(-1).bool()
	episode = td["episode"].reshape(-1).to(torch.int64)
	success_count = 0
	if done.any():
		for key in SUCCESS_KEYS:
			if key not in td.keys():
				continue
			if key == "terminal_success":
				success_values = td[key].reshape(-1).bool()[done].float()
			else:
				success_values = td[key].reshape(-1).float()[done]
			success_count = int((success_values > 0.5).sum().item())
			break
	num_episodes = int(torch.unique(episode).numel())
	intervention_fraction = None
	if "intervened" in td.keys():
		intervention_fraction = float(td["intervened"].reshape(-1).float().mean().item())
	return {
		"num_transitions": int(td["obs"].shape[0]),
		"num_episodes": num_episodes,
		"num_done": int(done.to(torch.int64).sum().item()),
		"success_count": success_count,
		"failure_count": max(0, num_episodes - success_count),
		"intervention_step_fraction": intervention_fraction,
	}


def clean_dataset(args):
	input_fp = Path(args.input).expanduser().resolve()
	output_fp = Path(args.output).expanduser().resolve()
	metadata_fp = (
		Path(args.metadata_out).expanduser().resolve()
		if args.metadata_out
		else Path(f"{output_fp}.json")
	)
	if output_fp.exists() and not args.overwrite:
		raise FileExistsError(f"Output exists: {output_fp}. Use --overwrite.")
	if metadata_fp.exists() and not args.overwrite:
		raise FileExistsError(f"Metadata output exists: {metadata_fp}. Use --overwrite.")

	td = _load_tensordict(input_fp)
	ok, reason = _valid_obs(td, expected_obs_dim=args.expected_obs_dim)
	if not ok:
		raise ValueError(reason)

	keys = _keys(td)
	episode_map = _episode_index_map(td)
	clean_chunks = []
	records = []
	raw_lengths = []
	clean_lengths = []
	raw_success_count = 0

	for episode_id, raw_indices in sorted(episode_map.items()):
		raw_length = int(raw_indices.numel())
		raw_lengths.append(raw_length)
		reasons = []
		done_offset = _first_done_offset(td, raw_indices)
		if done_offset is None:
			reasons.append("missing_done")
			if args.require_done:
				records.append({
					"episode": episode_id,
					"status": "dropped",
					"reasons": reasons,
					"raw_length": raw_length,
				})
				continue
			done_offset = raw_length - 1
		usable_indices = raw_indices[: done_offset + 1]
		has_success_label, success, success_key = _final_success(td, usable_indices)
		if success:
			raw_success_count += 1
		if args.require_success_label and not has_success_label:
			reasons.append("missing_success_label")
			records.append({
				"episode": episode_id,
				"status": "dropped",
				"reasons": reasons,
				"raw_length": raw_length,
				"usable_length": int(usable_indices.numel()),
			})
			continue
		action_ok, action_reason = _finite_actions(
			td,
			usable_indices,
			expected_action_dim=args.expected_action_dim,
		)
		if not action_ok:
			reasons.append(action_reason)
			records.append({
				"episode": episode_id,
				"status": "dropped",
				"reasons": reasons,
				"raw_length": raw_length,
				"usable_length": int(usable_indices.numel()),
				"success": success,
			})
			continue

		intervention = _intervention_mask(td, usable_indices)
		contact = _contact_mask(
			td,
			usable_indices,
			force_tail_dim=args.force_tail_dim,
			threshold=args.contact_force_threshold,
		)
		signal = intervention | contact
		signal_positions = torch.nonzero(signal, as_tuple=False).reshape(-1).tolist()
		if raw_length > args.hard_max_episode_steps and not signal_positions and not success:
			reasons.append("overlong_without_signal")
			records.append({
				"episode": episode_id,
				"status": "dropped",
				"reasons": reasons,
				"raw_length": raw_length,
				"usable_length": int(usable_indices.numel()),
				"success": success,
			})
			continue

		start, end, drop_reason = _choose_window(
			int(usable_indices.numel()),
			success=success,
			signal_positions=[int(x) for x in signal_positions],
			max_episode_steps=args.max_episode_steps,
			success_pre_window=args.success_pre_window,
			failure_pre_window=args.failure_pre_window,
			failure_post_window=args.failure_post_window,
			drop_failure_without_signal=args.drop_failure_without_signal,
		)
		if drop_reason is not None:
			reasons.append(drop_reason)
			records.append({
				"episode": episode_id,
				"status": "dropped",
				"reasons": reasons,
				"raw_length": raw_length,
				"usable_length": int(usable_indices.numel()),
				"success": success,
			})
			continue
		assert start is not None and end is not None
		kept_indices = usable_indices[start : end + 1]
		if int(kept_indices.numel()) < int(args.min_episode_steps):
			reasons.append("too_short_after_windowing")
			records.append({
				"episode": episode_id,
				"status": "dropped",
				"reasons": reasons,
				"raw_length": raw_length,
				"usable_length": int(usable_indices.numel()),
				"kept_length": int(kept_indices.numel()),
				"success": success,
			})
			continue

		new_episode_id = len(clean_chunks)
		chunk = _slice_chunk(td, keys, kept_indices)
		_rewrite_segment_labels(
			chunk,
			new_episode_id=new_episode_id,
			success=success,
			reaches_raw_terminal=(end == int(usable_indices.numel()) - 1),
		)
		clean_chunks.append(chunk)
		clean_lengths.append(int(kept_indices.numel()))
		if int(kept_indices.numel()) < int(usable_indices.numel()):
			reasons.append("windowed")
		if raw_length > args.max_episode_steps:
			reasons.append("raw_over_max_episode_steps")
		records.append({
			"episode": episode_id,
			"new_episode": new_episode_id,
			"status": "kept",
			"reasons": reasons,
			"raw_length": raw_length,
			"usable_length": int(usable_indices.numel()),
			"kept_start": int(start),
			"kept_end": int(end),
			"kept_length": int(kept_indices.numel()),
			"success": success,
			"success_key": success_key,
			"intervention_steps": int(intervention.to(torch.int64).sum().item()),
			"contact_steps": int(contact.to(torch.int64).sum().item()),
		})

	if not clean_chunks:
		raise RuntimeError("No HIL episodes survived cleaning.")

	all_keys = sorted({key for chunk in clean_chunks for key in chunk.keys()})
	merged = {}
	for key in all_keys:
		merged[key] = torch.cat([chunk[key] for chunk in clean_chunks], dim=0).contiguous()
	clean_td = TensorDict(merged, batch_size=(merged["obs"].shape[0],))

	output_fp.parent.mkdir(parents=True, exist_ok=True)
	metadata_fp.parent.mkdir(parents=True, exist_ok=True)
	torch.save(clean_td, output_fp)

	clean_counts = _dataset_counts(clean_td)
	report = {
		"input": str(input_fp),
		"output": str(output_fp),
		"params": {
			"max_episode_steps": int(args.max_episode_steps),
			"hard_max_episode_steps": int(args.hard_max_episode_steps),
			"success_pre_window": int(args.success_pre_window),
			"failure_pre_window": int(args.failure_pre_window),
			"failure_post_window": int(args.failure_post_window),
			"contact_force_threshold": float(args.contact_force_threshold),
			"force_tail_dim": int(args.force_tail_dim),
			"drop_failure_without_signal": bool(args.drop_failure_without_signal),
		},
		"raw": {
			**_dataset_counts(td),
			"lengths": _length_stats(raw_lengths),
			"success_count_from_episode_labels": int(raw_success_count),
		},
		"clean": {
			**clean_counts,
			"lengths": _length_stats(clean_lengths),
			"retained_transition_fraction": float(
				clean_counts["num_transitions"] / max(1, int(td["obs"].shape[0]))
			),
			"ready_for_training": bool(clean_counts["success_count"] >= int(args.min_success_episodes)),
			"min_success_episodes": int(args.min_success_episodes),
		},
		"episodes": records,
	}
	with open(metadata_fp, "w", encoding="utf-8") as f:
		json.dump(report, f, indent=2, ensure_ascii=True)

	if args.fail_under_min_success and clean_counts["success_count"] < int(args.min_success_episodes):
		raise RuntimeError(
			f"Only {clean_counts['success_count']} clean success episodes; "
			f"need at least {int(args.min_success_episodes)}."
		)
	return output_fp, metadata_fp, report


def _parse_args():
	parser = argparse.ArgumentParser(
		description="Audit and window real HIL rollout data before conservative offline fine-tuning."
	)
	parser.add_argument("--input", required=True, help="Raw real HIL TensorDict .pt file.")
	parser.add_argument("--output", required=True, help="Cleaned TensorDict .pt output.")
	parser.add_argument("--metadata-out", default=None, help="Sidecar JSON output. Default: <output>.json")
	parser.add_argument("--overwrite", action="store_true")
	parser.add_argument("--max-episode-steps", type=int, default=74)
	parser.add_argument("--hard-max-episode-steps", type=int, default=120)
	parser.add_argument("--success-pre-window", type=int, default=74)
	parser.add_argument("--failure-pre-window", type=int, default=24)
	parser.add_argument("--failure-post-window", type=int, default=50)
	parser.add_argument("--min-episode-steps", type=int, default=3)
	parser.add_argument("--expected-obs-dim", type=int, default=17)
	parser.add_argument("--expected-action-dim", type=int, default=3)
	parser.add_argument("--force-tail-dim", type=int, default=6)
	parser.add_argument("--contact-force-threshold", type=float, default=1.0)
	parser.add_argument("--min-success-episodes", type=int, default=5)
	parser.add_argument("--fail-under-min-success", action="store_true")
	parser.add_argument("--require-done", action=argparse.BooleanOptionalAction, default=True)
	parser.add_argument("--require-success-label", action=argparse.BooleanOptionalAction, default=True)
	parser.add_argument("--drop-failure-without-signal", action=argparse.BooleanOptionalAction, default=True)
	return parser.parse_args()


def main():
	output_fp, metadata_fp, report = clean_dataset(_parse_args())
	print(f"Saved clean HIL dataset: {output_fp}")
	print(f"Saved HIL audit metadata: {metadata_fp}")
	print(
		"Raw transitions={raw:,} clean transitions={clean:,} clean_success={success} "
		"ready_for_training={ready}".format(
			raw=report["raw"]["num_transitions"],
			clean=report["clean"]["num_transitions"],
			success=report["clean"]["success_count"],
			ready=report["clean"]["ready_for_training"],
		)
	)


if __name__ == "__main__":
	main()
