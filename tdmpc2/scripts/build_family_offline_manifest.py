from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf


TDMPC2_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TDMPC2_ROOT.parent
if str(TDMPC2_ROOT) not in sys.path:
	sys.path.insert(0, str(TDMPC2_ROOT))

from config import Config, _load_srsa_task_template_tasks  # noqa: E402


REQUIRED_ROLLOUT_KEYS = (
	"obs",
	"next_obs",
	"action",
	"reward",
	"done",
	"terminated",
	"truncated",
	"episode",
	"step_id",
)


def _normalize_assembly_id(value) -> str:
	text = str(value).strip().strip("'\"")
	return text.zfill(5) if text.isdigit() and len(text) < 5 else text


def _resolve_path(path: str | Path, *, base_dir: Path = REPO_ROOT) -> Path:
	path = Path(path).expanduser()
	if not path.is_absolute():
		path = base_dir / path
	return path.resolve()


def _parse_assembly_ids(raw: list[str]) -> list[str]:
	ids: list[str] = []
	for item in raw:
		text = str(item).strip().strip("'\"")
		if text.startswith("[") and text.endswith("]"):
			text = text[1:-1]
		for part in text.replace(";", ",").replace(" ", ",").split(","):
			part = part.strip().strip("'\"")
			if part:
				ids.append(_normalize_assembly_id(part))
	return list(dict.fromkeys(ids))


def _load_task_template(
	assembly_id: str,
	*,
	template_fp: Path,
	mesh_geometry_fp: Path,
	param_template_id: int,
	reference_anchor_assembly_id: str,
	reference_anchor_task_type_id: int,
) -> dict:
	cfg = OmegaConf.structured(Config)
	cfg.assembly_id = assembly_id
	cfg.srsa_task_template_fp = str(template_fp)
	cfg.srsa_mesh_geometry_fp = str(mesh_geometry_fp)
	cfg.srsa_param_template_id = int(param_template_id)
	cfg.srsa_task_template_id = int(param_template_id)
	cfg.srsa_axial_reference_anchor_assembly_id = _normalize_assembly_id(reference_anchor_assembly_id)
	cfg.srsa_axial_reference_anchor_task_type_id = int(reference_anchor_task_type_id)
	tasks = _load_srsa_task_template_tasks(cfg, str(template_fp))
	for task in tasks:
		if int(task.get("task_id", -1)) == int(param_template_id):
			task_vec = task.get("task_vec_6", task.get("task_param_vec"))
			if task_vec is None or len(task_vec) != 6:
				raise ValueError(f"Template for assembly_id={assembly_id} did not produce task_vec_6 dim 6.")
			return task
	raise ValueError(f"srsa_param_template_id={param_template_id} not found in {template_fp}")


def _candidate_paths(assembly_id: str, source_templates: list[str], search_roots: list[Path]) -> list[Path]:
	candidates: list[Path] = []
	for template in source_templates:
		path = _resolve_path(template.format(assembly_id=assembly_id, task_id=assembly_id))
		if path.exists():
			candidates.append(path)
	for root in search_roots:
		if not root.exists():
			continue
		patterns = (
			f"**/{assembly_id}/policy_eval_rollouts.pt",
			f"**/{assembly_id}/*rollouts*.pt",
			f"**/*{assembly_id}*rollouts*.pt",
		)
		for pattern in patterns:
			for path in root.glob(pattern):
				if path.is_file():
					candidates.append(path.resolve())
	deduped = list(dict.fromkeys(candidates))
	return sorted(deduped, key=lambda path: path.stat().st_mtime, reverse=True)


def _resolve_source_fp(
	assembly_id: str,
	*,
	source_templates: list[str],
	search_roots: list[Path],
	prefer_newest: bool,
) -> Path:
	matches = _candidate_paths(assembly_id, source_templates, search_roots)
	if not matches:
		template_hint = ", ".join(source_templates) if source_templates else "<none>"
		root_hint = ", ".join(str(path) for path in search_roots)
		raise FileNotFoundError(
			f"No rollout source found for assembly_id={assembly_id}. "
			f"Checked source_templates=[{template_hint}] and search_roots=[{root_hint}]."
		)
	if len(matches) > 1 and not prefer_newest:
		joined = "\n  ".join(str(path) for path in matches)
		raise RuntimeError(
			f"Multiple rollout sources found for assembly_id={assembly_id}; pass --prefer-newest "
			f"or provide an exact --source-template.\n  {joined}"
		)
	return matches[0]


def _tensor_keys(obj) -> set[str]:
	return set(obj.keys()) if hasattr(obj, "keys") else set()


def _load_rollout_summary(source_fp: Path) -> dict:
	obj = torch.load(source_fp, map_location="cpu", weights_only=False)
	keys = _tensor_keys(obj)
	missing = [key for key in REQUIRED_ROLLOUT_KEYS if key not in keys]
	if missing:
		raise KeyError(f"Rollout source {source_fp} is missing required keys: {missing}")
	obs = obj["obs"]
	action = obj["action"]
	episode = obj["episode"].reshape(-1)
	done = obj["done"].reshape(-1).bool()
	if obs.ndim < 2:
		raise ValueError(f"Rollout source {source_fp} has invalid obs shape: {tuple(obs.shape)}")
	if action.ndim < 2:
		raise ValueError(f"Rollout source {source_fp} has invalid action shape: {tuple(action.shape)}")
	summary = {
		"obs_shape": list(obs.shape[1:]),
		"action_dim": int(action.shape[-1]),
		"num_transitions": int(obs.shape[0]),
		"num_episodes": int(torch.unique(episode.to(torch.int64)).numel()),
		"num_done": int(done.to(torch.int64).sum().item()),
		"fields": sorted(keys),
	}
	success_key = None
	for key in (
		"episode_strict_success_stable_final",
		"episode_strict_success_episode_final",
		"episode_success_final",
		"success_episode",
	):
		if key in keys:
			success_key = key
			break
	if success_key is not None and done.any():
		success = obj[success_key].reshape(-1).float()
		final_success = success[done]
		success_count = int((final_success > 0.5).sum().item())
		summary.update({
			"success_key": success_key,
			"success_count": success_count,
			"failure_count": int(final_success.numel()) - success_count,
			"success_rate": success_count / max(1, int(final_success.numel())),
		})
	return summary


def _check_expected_shapes(summary: dict, *, source_fp: Path, expected_obs_dim: int | None, expected_action_dim: int | None):
	if expected_obs_dim is not None:
		obs_shape = list(summary.get("obs_shape", []))
		actual_obs_dim = int(obs_shape[-1]) if obs_shape else None
		if actual_obs_dim != int(expected_obs_dim):
			raise ValueError(
				f"Rollout source {source_fp} has obs_dim={actual_obs_dim}, expected {expected_obs_dim}."
			)
	if expected_action_dim is not None and int(summary["action_dim"]) != int(expected_action_dim):
		raise ValueError(
			f"Rollout source {source_fp} has action_dim={summary['action_dim']}, expected {expected_action_dim}."
		)


def _manifest_entry(
	task_id: int,
	assembly_id: str,
	source_fp: Path,
	template: dict,
	summary: dict,
) -> dict:
	task_vec = [float(value) for value in template.get("task_vec_6", template.get("task_param_vec"))]
	entry = {
		"task_id": int(task_id),
		"task_name": template.get("task_name", f"isaaclab-srsa-assembly-{assembly_id}"),
		"assembly_id": assembly_id,
		"source_fp": str(source_fp),
		"action_dim": int(summary["action_dim"]),
		"obs_shape": summary["obs_shape"],
		"max_episode_steps": int(template.get("max_episode_steps", 74)),
		"task_vec_6": task_vec,
		"task_param_vec": task_vec,
		"num_episodes": int(summary["num_episodes"]),
		"num_transitions": int(summary["num_transitions"]),
	}
	for key in ("success_key", "success_count", "failure_count", "success_rate"):
		if key in summary:
			entry[key] = summary[key]
	for key in ("srsa_params", "srsa_sampler", "mesh_geometry"):
		if key in template:
			entry[key] = template[key]
	return entry


def main():
	parser = argparse.ArgumentParser(
		description=(
			"Build a Newt offline manifest for shared SRSA family multitask continuation. "
			"The script uses existing rollout .pt files and computes true task_vec_6 from the SRSA template/mesh CSV."
		)
	)
	parser.add_argument(
		"--assembly-ids",
		nargs="+",
		default=["01125", "00004", "00014", "00062", "00271"],
		help="Assembly ids to include. JSON-like strings are accepted.",
	)
	parser.add_argument(
		"--source-template",
		action="append",
		default=[],
		help="Rollout path template containing {assembly_id}, e.g. logs/.../{assembly_id}/policy_eval_rollouts.pt.",
	)
	parser.add_argument(
		"--search-root",
		action="append",
		default=["logs", "data"],
		help="Directory to scan for <assembly_id>/policy_eval_rollouts.pt when --source-template is not enough.",
	)
	parser.add_argument(
		"--output-manifest-fp",
		default="data/offline_manifest_01125_family_multitask.json",
		help="Output manifest JSON path.",
	)
	parser.add_argument("--srsa-task-template-fp", default="data/srsa_axial_task_templates.json")
	parser.add_argument("--srsa-mesh-geometry-fp", default="data/srsa_mesh_geometry_params.csv")
	parser.add_argument("--srsa-param-template-id", type=int, default=2)
	parser.add_argument("--reference-anchor-assembly-id", default="01125")
	parser.add_argument("--reference-anchor-task-type-id", type=int, default=0)
	parser.add_argument("--expected-obs-dim", type=int, default=None, help="Optional rollout obs dim guard.")
	parser.add_argument("--expected-action-dim", type=int, default=None, help="Optional rollout action dim guard.")
	parser.add_argument("--prefer-newest", action="store_true", help="Use the newest matching rollout when multiple exist.")
	parser.add_argument("--allow-missing", action="store_true", help="Write a partial manifest instead of failing on missing sources.")
	parser.add_argument("--overwrite", action="store_true")
	parser.add_argument("--dry-run", action="store_true")
	args = parser.parse_args()

	assembly_ids = _parse_assembly_ids(args.assembly_ids)
	if not assembly_ids:
		raise ValueError("No assembly ids were provided.")
	output_fp = _resolve_path(args.output_manifest_fp)
	if output_fp.exists() and not args.overwrite and not args.dry_run:
		raise FileExistsError(f"Output manifest exists: {output_fp}. Use --overwrite.")
	template_fp = _resolve_path(args.srsa_task_template_fp)
	mesh_geometry_fp = _resolve_path(args.srsa_mesh_geometry_fp)
	if not template_fp.exists():
		raise FileNotFoundError(f"SRSA task template JSON not found: {template_fp}")
	if not mesh_geometry_fp.exists():
		raise FileNotFoundError(f"SRSA mesh geometry CSV not found: {mesh_geometry_fp}")
	search_roots = [_resolve_path(path) for path in args.search_root]

	entries = []
	missing = []
	for assembly_id in assembly_ids:
		template = _load_task_template(
			assembly_id,
			template_fp=template_fp,
			mesh_geometry_fp=mesh_geometry_fp,
			param_template_id=args.srsa_param_template_id,
			reference_anchor_assembly_id=args.reference_anchor_assembly_id,
			reference_anchor_task_type_id=args.reference_anchor_task_type_id,
		)
		try:
			source_fp = _resolve_source_fp(
				assembly_id,
				source_templates=args.source_template,
				search_roots=search_roots,
				prefer_newest=args.prefer_newest,
			)
			summary = _load_rollout_summary(source_fp)
			_check_expected_shapes(
				summary,
				source_fp=source_fp,
				expected_obs_dim=args.expected_obs_dim,
				expected_action_dim=args.expected_action_dim,
			)
		except Exception as exc:
			if not args.allow_missing:
				raise
			missing.append({"assembly_id": assembly_id, "error": str(exc)})
			continue
		entries.append(_manifest_entry(len(entries), assembly_id, source_fp, template, summary))

	if not entries:
		raise RuntimeError("No manifest entries were built. Provide rollout sources or remove --allow-missing.")
	manifest = {
		"schema": "newt.offline_multitask_manifest.v1",
		"description": "Shared SRSA family multitask replay manifest for Scheme A continuation.",
		"source": {
			"builder": str(Path(__file__).resolve()),
			"srsa_task_template_fp": str(template_fp),
			"srsa_mesh_geometry_fp": str(mesh_geometry_fp),
			"srsa_param_template_id": int(args.srsa_param_template_id),
			"reference_anchor_assembly_id": _normalize_assembly_id(args.reference_anchor_assembly_id),
			"reference_anchor_task_type_id": int(args.reference_anchor_task_type_id),
			"expected_obs_dim": args.expected_obs_dim,
			"expected_action_dim": args.expected_action_dim,
		},
		"tasks": entries,
	}
	if missing:
		manifest["missing"] = missing

	print(json.dumps(manifest, indent=2, ensure_ascii=True))
	if args.dry_run:
		print(f"[builder] dry run complete; manifest was not written: {output_fp}")
		return
	output_fp.parent.mkdir(parents=True, exist_ok=True)
	with open(output_fp, "w", encoding="utf-8") as f:
		json.dump(manifest, f, indent=2, ensure_ascii=True)
	print(f"[builder] saved manifest: {output_fp}")
	if missing:
		print(f"[builder] warning: skipped {len(missing)} missing assembly ids.")


if __name__ == "__main__":
	main()
