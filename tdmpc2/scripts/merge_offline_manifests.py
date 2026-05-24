from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tensordict import TensorDict


def _load_manifest(path: Path) -> list[dict]:
	with open(path, "r", encoding="utf-8") as f:
		obj = json.load(f)
	tasks = obj.get("tasks", obj if isinstance(obj, list) else None)
	if not isinstance(tasks, list):
		raise ValueError(f"Manifest must contain a tasks list: {path}")
	return tasks


def _normalize_assembly_id(value) -> str:
	value = str(value).strip().strip("'\"")
	return value.zfill(5) if value.isdigit() and len(value) < 5 else value


def _tensor_keys(td) -> list[str]:
	return list(td.keys()) if hasattr(td, "keys") else list(td.keys())


def _default_like(num_rows: int, ref: torch.Tensor):
	fill = False if ref.dtype == torch.bool else 0
	return torch.full((num_rows, *ref.shape[1:]), fill, dtype=ref.dtype)


def _concat_tensordicts(paths: list[Path], *, task_id: int):
	chunks = []
	episode_offset = 0
	all_keys = set()
	loaded = []
	for path in paths:
		td = torch.load(path, map_location="cpu", weights_only=False)
		if not hasattr(td, "keys"):
			raise TypeError(f"Expected TensorDict-like rollout file: {path}")
		data = {key: td[key].detach().cpu().contiguous() for key in _tensor_keys(td)}
		episode = data["episode"].reshape(-1).to(torch.int64)
		unique_episode, inverse = torch.unique(episode, sorted=True, return_inverse=True)
		data["episode"] = (inverse + episode_offset).reshape_as(data["episode"])
		episode_offset += int(unique_episode.numel())
		data["task"] = torch.full_like(data["episode"], int(task_id), dtype=torch.int64)
		loaded.append(data)
		all_keys.update(data.keys())
	for key in sorted(all_keys):
		ref = next(data[key] for data in loaded if key in data)
		for data in loaded:
			if key not in data:
				data[key] = _default_like(int(data["obs"].shape[0]), ref)
	for data in loaded:
		chunks.append({key: data[key] for key in sorted(all_keys)})
	merged = {key: torch.cat([chunk[key] for chunk in chunks], dim=0).contiguous() for key in sorted(all_keys)}
	return TensorDict(merged, batch_size=(merged["obs"].shape[0],))


def _episode_success(td) -> tuple[int, int]:
	done = td["done"].reshape(-1).bool()
	success_key = "episode_strict_success_stable_final" if "episode_strict_success_stable_final" in td.keys() else "episode_success_final"
	success = td[success_key].reshape(-1).float()
	success_done = success[done] if done.any() else success
	success_count = int((success_done > 0.5).sum().item())
	return success_count, int(success_done.numel()) - success_count


def main():
	parser = argparse.ArgumentParser(description="Merge rollout manifests by assembly_id and rewrite task/episode ids.")
	parser.add_argument("manifest_fps", nargs="+", help="Input manifest JSON files.")
	parser.add_argument("--output-manifest-fp", required=True, help="Output merged manifest JSON.")
	parser.add_argument("--output-dir", default=None, help="Directory for merged rollout TensorDict files.")
	parser.add_argument("--overwrite", action="store_true")
	args = parser.parse_args()

	output_manifest = Path(args.output_manifest_fp).expanduser().resolve()
	output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else output_manifest.parent / f"{output_manifest.stem}_rollouts"
	if output_manifest.exists() and not args.overwrite:
		raise FileExistsError(f"Output manifest exists: {output_manifest}. Use --overwrite.")
	output_dir.mkdir(parents=True, exist_ok=True)

	grouped: dict[str, list[dict]] = {}
	for raw_path in args.manifest_fps:
		path = Path(raw_path).expanduser().resolve()
		for entry in _load_manifest(path):
			assembly_id = _normalize_assembly_id(entry["assembly_id"])
			item = dict(entry)
			item["assembly_id"] = assembly_id
			grouped.setdefault(assembly_id, []).append(item)

	entries = []
	for task_id, assembly_id in enumerate(sorted(grouped.keys())):
		group = grouped[assembly_id]
		source_paths = [Path(item["source_fp"]).expanduser().resolve() for item in group]
		for source_path in source_paths:
			if not source_path.exists():
				raise FileNotFoundError(f"Rollout source not found: {source_path}")
		merged_td = _concat_tensordicts(source_paths, task_id=task_id)
		task_dir = output_dir / assembly_id
		task_dir.mkdir(parents=True, exist_ok=True)
		output_fp = task_dir / "policy_eval_rollouts.pt"
		torch.save(merged_td, output_fp)
		success_count, failure_count = _episode_success(merged_td)
		template = group[-1]
		entry = {
			"task_id": task_id,
			"task_name": template.get("task_name", f"isaaclab-srsa-assembly-{assembly_id}"),
			"assembly_id": assembly_id,
			"source_fp": str(output_fp),
			"action_dim": int(merged_td["action"].shape[-1]),
			"obs_shape": list(merged_td["obs"].shape[1:]),
			"max_episode_steps": int(template.get("max_episode_steps", 74)),
			"task_vec_6": template.get("task_vec_6", template.get("task_param_vec")),
			"task_param_vec": template.get("task_param_vec", template.get("task_vec_6")),
			"num_episodes": int(torch.unique(merged_td["episode"].reshape(-1)).numel()),
			"num_transitions": int(merged_td["obs"].shape[0]),
			"success_count": success_count,
			"failure_count": failure_count,
			"success_rate": success_count / max(1, success_count + failure_count),
			"source_manifests": [str(Path(path).expanduser().resolve()) for path in args.manifest_fps],
		}
		for key in ("srsa_params", "srsa_sampler", "success_metrics", "weak_task_requires_online_boost"):
			if key in template:
				entry[key] = template[key]
		entries.append(entry)

	manifest = {"tasks": entries}
	output_manifest.parent.mkdir(parents=True, exist_ok=True)
	with open(output_manifest, "w", encoding="utf-8") as f:
		json.dump(manifest, f, indent=2, ensure_ascii=True)
	print(f"Saved merged manifest: {output_manifest}")
	print(f"Saved merged rollout files under: {output_dir}")


if __name__ == "__main__":
	main()
