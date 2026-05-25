from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch

try:
	from tensordict import TensorDict
except ModuleNotFoundError:
	TensorDict = None


OPTIONAL_FIELDS = [
	"episode_return_running",
	"episode_return_final",
	"episode_success_final",
	"episode_failure_final",
	"terminal_success",
	"terminal_failure",
	"success_episode",
	"episode_official_success_latched_final",
	"episode_official_success_terminal_final",
	"episode_process_success_terminal_final",
	"episode_relaxed_process_success_terminal_final",
	"episode_relaxed_success_stable_final",
	"episode_relaxed_success_episode_final",
	"episode_strict_success_stable_final",
	"episode_strict_success_episode_final",
	"episode_depth_fraction_final",
	"episode_lateral_error_final",
	"episode_angle_error_final",
	"episode_keypoint_error_final",
]


OPTIONAL_FIELD_DTYPES = {
	"episode_return_running": torch.float32,
	"episode_return_final": torch.float32,
	"episode_success_final": torch.float32,
	"episode_failure_final": torch.float32,
	"terminal_success": torch.bool,
	"terminal_failure": torch.bool,
	"success_episode": torch.float32,
	"episode_official_success_latched_final": torch.float32,
	"episode_official_success_terminal_final": torch.float32,
	"episode_process_success_terminal_final": torch.float32,
	"episode_relaxed_process_success_terminal_final": torch.float32,
	"episode_relaxed_success_stable_final": torch.float32,
	"episode_relaxed_success_episode_final": torch.float32,
	"episode_strict_success_stable_final": torch.float32,
	"episode_strict_success_episode_final": torch.float32,
	"episode_depth_fraction_final": torch.float32,
	"episode_lateral_error_final": torch.float32,
	"episode_angle_error_final": torch.float32,
	"episode_keypoint_error_final": torch.float32,
}


def _assert_keys(obj, keys):
	missing = [key for key in keys if key not in obj.keys()]
	if missing:
		raise KeyError(f"Missing required keys in source dataset: {missing}")


def _as_tensor(value, key):
	if not isinstance(value, torch.Tensor):
		raise TypeError(f"Expected tensor for key '{key}', got {type(value)}")
	return value.detach().cpu().contiguous()


def load_offline_manifest(manifest_fp: str | Path) -> list[dict]:
	manifest_fp = Path(manifest_fp).expanduser().resolve()
	if not manifest_fp.exists():
		raise FileNotFoundError(f"Offline manifest not found: {manifest_fp}")
	with open(manifest_fp, "r", encoding="utf-8") as f:
		manifest = json.load(f)
	tasks = manifest.get("tasks")
	if not isinstance(tasks, list) or len(tasks) == 0:
		raise ValueError(f"Offline manifest at {manifest_fp} must contain a non-empty 'tasks' list.")
	entries = sorted(tasks, key=lambda item: int(item["task_id"]))
	task_ids = [int(item["task_id"]) for item in entries]
	expected = list(range(len(entries)))
	if task_ids != expected:
		raise ValueError(f"Offline manifest task_id values must be consecutive starting at 0. Got {task_ids}.")
	return entries


def _coerce_optional_value(value: torch.Tensor, key: str) -> torch.Tensor:
	dtype = OPTIONAL_FIELD_DTYPES[key]
	return value.to(dtype)


def _extract_source_tensors(obj, *, obs_key: str, next_obs_key: str, action_key: str):
	required = [
		obs_key,
		next_obs_key,
		action_key,
		"reward",
		"done",
		"terminated",
		"truncated",
		"episode",
		"step_id",
	]
	_assert_keys(obj, required)

	data = {
		"obs": _as_tensor(obj[obs_key], obs_key).to(torch.float32),
		"next_obs": _as_tensor(obj[next_obs_key], next_obs_key).to(torch.float32),
		"action": _as_tensor(obj[action_key], action_key).to(torch.float32),
		"reward": _as_tensor(obj["reward"], "reward").to(torch.float32),
		"done": _as_tensor(obj["done"], "done").to(torch.bool),
		"terminated": _as_tensor(obj["terminated"], "terminated").to(torch.bool),
		"truncated": _as_tensor(obj["truncated"], "truncated").to(torch.bool),
		"episode": _as_tensor(obj["episode"], "episode").to(torch.int64),
		"step_id": _as_tensor(obj["step_id"], "step_id").to(torch.int32),
	}
	for key in OPTIONAL_FIELDS:
		if key not in obj.keys():
			continue
		data[key] = _coerce_optional_value(_as_tensor(obj[key], key), key)
	return data


def _concatenate_compact_data(chunks: list[dict]):
	fields = sorted({key for chunk in chunks for key in chunk.keys()})
	data = {}
	for field in fields:
		values = [chunk[field] for chunk in chunks if field in chunk]
		if len(values) != len(chunks):
			shape_ref = next(chunk[field].shape[1:] for chunk in chunks if field in chunk)
			dtype = OPTIONAL_FIELD_DTYPES.get(field, values[0].dtype)
			default_fill = False if dtype == torch.bool else 0
			filled_values = []
			for chunk in chunks:
				if field in chunk:
					filled_values.append(chunk[field])
					continue
				num_rows = chunk["obs"].shape[0]
				fill = torch.full((num_rows, *shape_ref), default_fill, dtype=dtype)
				filled_values.append(fill)
			values = filled_values
		data[field] = torch.cat(values, dim=0).contiguous()
	return data


def summarize_compact_dataset(td):
	obs = td["obs"]
	action = td["action"]
	reward = td["reward"]
	episode = td["episode"]
	done = td["done"]
	summary = {
		"num_transitions": int(obs.shape[0]),
		"obs_shape": list(obs.shape[1:]),
		"action_shape": list(action.shape[1:]),
		"reward_shape": list(reward.shape[1:]),
		"num_episodes": int(episode.max().item()) + 1,
		"num_done": int(done.to(torch.int64).sum().item()),
		"obs_dtype": str(obs.dtype),
		"action_dtype": str(action.dtype),
		"reward_dtype": str(reward.dtype),
	}
	final_done = done.squeeze(-1) if done.ndim > 1 else done
	for key in ("episode_success_final", "success_episode"):
		if key in td and final_done.any():
			success = td[key]
			success = success.squeeze(-1) if success.ndim > 1 else success
			summary["final_success_mean"] = float(success[final_done].float().mean().item())
			break
	for key in ("episode_failure_final", "terminal_failure", "terminal_success", "episode_return_final"):
		if key in td:
			value = td[key]
			value = value.squeeze(-1) if value.ndim > 1 else value
			if value.dtype == torch.bool:
				summary[f"{key}_count"] = int(value.to(torch.int64).sum().item())
			else:
				summary[f"{key}_min"] = float(value.min().item())
				summary[f"{key}_max"] = float(value.max().item())
	return summary


def export_compact_dataset(
	input_fp: str | Path,
	output_fp: str | Path,
	*,
	obs_key: str = "obs",
	next_obs_key: str = "next_obs",
	action_key: str = "action",
	metadata_fp: Optional[str | Path] = None,
	overwrite: bool = False,
):
	input_fp = Path(input_fp).expanduser().resolve()
	output_fp = Path(output_fp).expanduser().resolve()
	metadata_fp = (
		Path(metadata_fp).expanduser().resolve()
		if metadata_fp is not None
		else output_fp.with_suffix(output_fp.suffix + ".json")
	)

	if not input_fp.exists():
		raise FileNotFoundError(f"Input dataset not found: {input_fp}")
	if output_fp.exists() and not overwrite:
		raise FileExistsError(f"Output already exists: {output_fp}. Use overwrite=True to replace it.")
	output_fp.parent.mkdir(parents=True, exist_ok=True)
	metadata_fp.parent.mkdir(parents=True, exist_ok=True)

	obj = torch.load(input_fp, map_location="cpu", weights_only=False)
	if not hasattr(obj, "keys"):
		raise TypeError(f"Expected a TensorDict-like object, got {type(obj)}")

	required = [
		obs_key,
		next_obs_key,
		action_key,
		"reward",
		"done",
		"terminated",
		"truncated",
		"episode",
		"step_id",
	]
	_assert_keys(obj, required)
	data = _extract_source_tensors(
		obj,
		obs_key=obs_key,
		next_obs_key=next_obs_key,
		action_key=action_key,
	)

	if TensorDict is not None:
		batch_size = (data["obs"].shape[0],)
		compact = TensorDict(data, batch_size=batch_size)
	else:
		compact = data

	torch.save(compact, output_fp)

	summary = {
		"source": str(input_fp),
		"output": str(output_fp),
		"obs_key": obs_key,
		"next_obs_key": next_obs_key,
		"action_key": action_key,
		"fields": list(data.keys()),
	}
	summary.update(summarize_compact_dataset(compact))
	with open(metadata_fp, "w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2, ensure_ascii=True)
	return output_fp, metadata_fp, summary


def export_multitask_compact_dataset(
	manifest_fp: str | Path,
	output_fp: str | Path,
	*,
	obs_key: str = "obs",
	next_obs_key: str = "next_obs",
	action_key: str = "action",
	metadata_fp: Optional[str | Path] = None,
	overwrite: bool = False,
):
	manifest_fp = Path(manifest_fp).expanduser().resolve()
	output_fp = Path(output_fp).expanduser().resolve()
	metadata_fp = (
		Path(metadata_fp).expanduser().resolve()
		if metadata_fp is not None
		else output_fp.with_suffix(output_fp.suffix + ".json")
	)

	if output_fp.exists() and not overwrite:
		raise FileExistsError(f"Output already exists: {output_fp}. Use overwrite=True to replace it.")
	output_fp.parent.mkdir(parents=True, exist_ok=True)
	metadata_fp.parent.mkdir(parents=True, exist_ok=True)

	entries = load_offline_manifest(manifest_fp)
	episode_offset = 0
	chunks = []
	task_map = []
	for entry in entries:
		task_id = int(entry["task_id"])
		task_name = entry.get("task_name", f"task_{task_id}")
		source_fp = Path(entry["source_fp"]).expanduser().resolve()
		if not source_fp.exists():
			raise FileNotFoundError(f"Offline source for task_id={task_id} not found: {source_fp}")
		obj = torch.load(source_fp, map_location="cpu", weights_only=False)
		if not hasattr(obj, "keys"):
			raise TypeError(f"Expected a TensorDict-like object for task_id={task_id}, got {type(obj)}")
		data = _extract_source_tensors(
			obj,
			obs_key=entry.get("obs_key", obs_key),
			next_obs_key=entry.get("next_obs_key", next_obs_key),
			action_key=entry.get("action_key", action_key),
		)
		local_episode = data["episode"].view(-1)
		unique_episode, inverse = torch.unique(local_episode, sorted=True, return_inverse=True)
		data["episode"] = (inverse.to(torch.int64) + episode_offset).view_as(data["episode"])
		episode_offset += int(unique_episode.numel())
		data["task"] = torch.full_like(data["episode"], task_id, dtype=torch.int64)
		chunks.append(data)
		task_meta = {
			"task_id": task_id,
			"task_name": task_name,
			"assembly_id": entry.get("assembly_id"),
			"source_fp": str(source_fp),
			"num_transitions": int(data["obs"].shape[0]),
			"num_episodes": int(unique_episode.numel()),
			"action_dim": int(data["action"].shape[-1]),
			"obs_shape": list(data["obs"].shape[1:]),
		}
		for optional_key in ("task_vec_6", "task_param_vec", "srsa_params", "srsa_sampler", "success_metrics"):
			if optional_key in entry:
				task_meta[optional_key] = entry[optional_key]
		task_map.append(task_meta)

	data = _concatenate_compact_data(chunks)
	if TensorDict is not None:
		compact = TensorDict(data, batch_size=(data["obs"].shape[0],))
	else:
		compact = data
	torch.save(compact, output_fp)

	summary = {
		"manifest": str(manifest_fp),
		"output": str(output_fp),
		"fields": list(data.keys()),
		"num_tasks": len(task_map),
		"task_map": task_map,
	}
	summary.update(summarize_compact_dataset(compact))
	with open(metadata_fp, "w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2, ensure_ascii=True)
	return output_fp, metadata_fp, summary
