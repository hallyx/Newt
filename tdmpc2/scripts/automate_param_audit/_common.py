#!/usr/bin/env python3
"""Shared read-only helpers for AutoMate/SRSA size-parameter audit scripts."""

from __future__ import annotations

import ast
import builtins
import csv
import hashlib
import json
import math
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
TDMPC2_ROOT = SCRIPT_DIR.parents[1]
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "automate_param_audit"
DEFAULT_TASK_HASH_CSV = REPO_ROOT / "reports" / "task_consistency" / "task_id_to_hash.csv"
RUNTIME_LAUNCH_ERROR_LOG = DEFAULT_OUTPUT_DIR / "runtime_launch_error.log"
LOCAL_ISAAC_ASSET_ROOT_CANDIDATES = (
	Path("/home/gpuserver/isaacsim_assets/Assets/Isaac/5.1"),
	Path("/home/gpuserver/isaacsim_assets/Assets/Isaac/4.5"),
)
REMOTE_ISAAC_ASSET_ROOT_PREFIXES = (
	"https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1",
	"https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5",
	"omniverse://localhost/NVIDIA/Assets/Isaac/5.1",
	"omniverse://localhost/NVIDIA/Assets/Isaac/4.5",
)

AXIAL_TASK_VEC_FIELDS = (
	"task_type_id_float",
	"log_scale",
	"clearance_abs_norm",
	"clearance_rel_norm",
	"depth_abs_norm",
	"yaw_requirement_float",
)

# These mirror the defaults in tdmpc2/config.py. They are intentionally local
# constants so static audit scripts can run without importing Hydra/Isaac stacks.
DEFAULT_REFERENCE_RADIUS_M = 0.003993
DEFAULT_REFERENCE_DEPTH_M = 0.015
DEFAULT_BASE_PLUG_DIAMETER_M = 0.007986


def ensure_tdmpc2_on_path() -> None:
	if str(TDMPC2_ROOT) not in sys.path:
		sys.path.insert(0, str(TDMPC2_ROOT))


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
	text = str(value)
	text = Path(text).stem if "/" in text or "\\" in text else text
	text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
	return text or "report"


def task_vec_hash(vec: list[float]) -> str:
	rounded = [round(float(item), 8) for item in list(vec)]
	text = ",".join(f"{item:.8g}" for item in rounded)
	return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def load_json(path: str | Path, *, default: Any = None) -> Any:
	path = resolve_path(path)
	if not path.exists():
		return default
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def write_json_report(report: dict[str, Any], output_path: str | Path, *, dry_run: bool) -> None:
	path = resolve_path(output_path)
	if dry_run:
		print(f"[dry-run] would write JSON report: {path}")
		return
	path.parent.mkdir(parents=True, exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		json.dump(report, f, ensure_ascii=False, indent=2)
		f.write("\n")
	print(f"Wrote JSON report: {path}")


def write_text_report(text: str, output_path: str | Path, *, dry_run: bool) -> None:
	path = resolve_path(output_path)
	if dry_run:
		print(f"[dry-run] would write text report: {path}")
		return
	path.parent.mkdir(parents=True, exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		f.write(text)
		if not text.endswith("\n"):
			f.write("\n")
	print(f"Wrote text report: {path}")


def write_csv(rows: list[dict[str, Any]], output_path: str | Path, *, dry_run: bool) -> None:
	path = resolve_path(output_path)
	if dry_run:
		print(f"[dry-run] would write CSV report: {path}")
		return
	path.parent.mkdir(parents=True, exist_ok=True)
	fieldnames: list[str] = []
	for row in rows:
		for key in row.keys():
			if key not in fieldnames:
				fieldnames.append(key)
	with open(path, "w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		for row in rows:
			writer.writerow(row)
	print(f"Wrote CSV report: {path}")


def add_message(messages: list[dict[str, Any]], level: str, message: str, **extra: Any) -> None:
	item: dict[str, Any] = {"level": str(level).upper(), "message": str(message)}
	item.update(extra)
	messages.append(item)


def status_from_messages(messages: list[dict[str, Any]]) -> str:
	levels = {str(item.get("level", "")).upper() for item in messages}
	if "FAIL" in levels:
		return "FAIL"
	if "WARNING" in levels:
		return "WARNING"
	return "PASS"


def print_status(status: str, messages: list[dict[str, Any]]) -> None:
	print(status)
	for item in messages:
		print(f"[{item.get('level', 'INFO')}] {item.get('message', '')}")


def unknown(reason: str, **extra: Any) -> dict[str, Any]:
	item = {"status": "UNKNOWN_WITH_REASON", "reason": str(reason)}
	item.update(extra)
	return item


def append_runtime_error(exc: BaseException, *, context: str, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> Path:
	output_dir = resolve_path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	path = output_dir / "runtime_launch_error.log"
	with open(path, "a", encoding="utf-8") as f:
		f.write("\n" + "=" * 80 + "\n")
		f.write(f"context: {context}\n")
		f.write(f"error: {repr(exc)}\n")
		f.write(traceback.format_exc())
		f.write("\n")
	return path


def local_isaac_asset_root() -> Path | None:
	for path in LOCAL_ISAAC_ASSET_ROOT_CANDIDATES:
		if (path / "Isaac" / "Environments" / "Grid" / "default_environment.usd").exists():
			return path
	return None


def _apply_local_isaac_asset_root() -> str | None:
	root = local_isaac_asset_root()
	if root is None:
		return None
	root_str = str(root)
	try:
		import carb
		settings = carb.settings.get_settings()
		settings.set("/persistent/isaac/asset_root/cloud", root_str)
		settings.set("/persistent/isaac/asset_root/default", root_str)
	except Exception:
		return None
	assets_mod = sys.modules.get("isaaclab.utils.assets")
	if assets_mod is not None:
		setattr(assets_mod, "NUCLEUS_ASSET_ROOT_DIR", root_str)
		setattr(assets_mod, "NVIDIA_NUCLEUS_DIR", f"{root_str}/NVIDIA")
		setattr(assets_mod, "ISAAC_NUCLEUS_DIR", f"{root_str}/Isaac")
		setattr(assets_mod, "ISAACLAB_NUCLEUS_DIR", f"{root_str}/Isaac/IsaacLab")
	_patch_ground_plane_cfg_path(root_str)
	_patch_usd_reference_paths(root_str)
	_patch_automate_metadata_open(root_str)
	_patch_trimesh_load_paths(root_str)
	return root_str


def _localize_isaac_usd_path(path_value: Any, root_str: str) -> Any:
	if not isinstance(path_value, str):
		return path_value
	for prefix in REMOTE_ISAAC_ASSET_ROOT_PREFIXES:
		if path_value.startswith(prefix):
			return root_str + path_value[len(prefix):]
	return path_value


def _patch_usd_reference_paths(root_str: str) -> None:
	try:
		import isaaclab.sim.utils.prims as prims_mod
	except Exception:
		return
	orig = getattr(prims_mod, "add_usd_reference", None)
	if orig is None or getattr(orig, "_newt_local_asset_root_patched", False):
		return

	def add_usd_reference_local(*args: Any, **kwargs: Any) -> Any:
		if len(args) >= 2:
			args = (args[0], _localize_isaac_usd_path(args[1], root_str), *args[2:])
		if "usd_path" in kwargs:
			kwargs["usd_path"] = _localize_isaac_usd_path(kwargs["usd_path"], root_str)
		return orig(*args, **kwargs)

	add_usd_reference_local._newt_local_asset_root_patched = True  # type: ignore[attr-defined]
	prims_mod.add_usd_reference = add_usd_reference_local


def _patch_automate_metadata_open(root_str: str) -> None:
	automate_root = Path(root_str) / "Isaac" / "IsaacLab" / "AutoMate"
	if not (automate_root / "plug_grasps.json").exists():
		return
	orig = builtins.open
	if getattr(orig, "_newt_automate_metadata_patched", False):
		return

	def open_local_automate_metadata(file: Any, *args: Any, **kwargs: Any) -> Any:
		try:
			path = Path(file)
			if not path.is_absolute():
				if path.name in {"plug_grasps.json", "disassembly_dist.json"}:
					local_path = automate_root / path.name
					if local_path.exists():
						file = str(local_path)
				elif path.name == "disassemble_traj.json":
					assembly_id = os.environ.get("NEWT_AUTOMATE_AUDIT_ASSEMBLY_ID", "")
					local_path = automate_root / assembly_id / path.name
					if local_path.exists():
						file = str(local_path)
		except Exception:
			pass
		return orig(file, *args, **kwargs)

	open_local_automate_metadata._newt_automate_metadata_patched = True  # type: ignore[attr-defined]
	builtins.open = open_local_automate_metadata


def _local_automate_basename_path(root_str: str, basename: str) -> Path | None:
	automate_root = Path(root_str) / "Isaac" / "IsaacLab" / "AutoMate"
	if basename in {"plug_grasps.json", "disassembly_dist.json"}:
		candidate = automate_root / basename
	elif basename in {"disassemble_traj.json", "plug.obj", "socket.obj", "plug.usd", "socket.usd"}:
		assembly_id = os.environ.get("NEWT_AUTOMATE_AUDIT_ASSEMBLY_ID", "")
		candidate = automate_root / assembly_id / basename
	else:
		return None
	return candidate if candidate.exists() else None


def _patch_trimesh_load_paths(root_str: str) -> None:
	try:
		import trimesh
		import trimesh.exchange.load as load_mod
	except Exception:
		return
	orig = getattr(load_mod, "load", None)
	orig_mesh = getattr(load_mod, "load_mesh", None)
	if orig is not None and not getattr(orig, "_newt_automate_mesh_patched", False):

		def load_local_automate_mesh(file_obj: Any, *args: Any, **kwargs: Any) -> Any:
			try:
				path = Path(file_obj)
				if not path.is_absolute():
					local_path = _local_automate_basename_path(root_str, path.name)
					if local_path is not None:
						file_obj = str(local_path)
			except Exception:
				pass
			return orig(file_obj, *args, **kwargs)

		load_local_automate_mesh._newt_automate_mesh_patched = True  # type: ignore[attr-defined]
		load_mod.load = load_local_automate_mesh
		try:
			trimesh.load = load_local_automate_mesh
		except Exception:
			pass
		industreal_mod = sys.modules.get("isaaclab_tasks.direct.automate.industreal_algo_utils")
		if industreal_mod is not None:
			setattr(industreal_mod, "load", load_local_automate_mesh)
	if orig_mesh is not None and not getattr(orig_mesh, "_newt_automate_mesh_patched", False):

		def load_mesh_local_automate(file_obj: Any, *args: Any, **kwargs: Any) -> Any:
			try:
				path = Path(file_obj)
				if not path.is_absolute():
					local_path = _local_automate_basename_path(root_str, path.name)
					if local_path is not None:
						file_obj = str(local_path)
			except Exception:
				pass
			return orig_mesh(file_obj, *args, **kwargs)

		load_mesh_local_automate._newt_automate_mesh_patched = True  # type: ignore[attr-defined]
		load_mod.load_mesh = load_mesh_local_automate
		try:
			trimesh.load_mesh = load_mesh_local_automate
		except Exception:
			pass


def _patch_dataclass_default(cls: Any, field_name: str, value: Any) -> None:
	fields = getattr(cls, "__dataclass_fields__", None)
	if isinstance(fields, dict) and field_name in fields:
		try:
			fields[field_name].default = value
		except Exception:
			pass


def _patch_ground_plane_cfg_path(root_str: str) -> None:
	local_ground_usd = f"{root_str}/Isaac/Environments/Grid/default_environment.usd"
	if not Path(local_ground_usd).exists():
		return
	module_names = (
		"isaaclab.sim.spawners.from_files.from_files_cfg",
		"isaaclab.sim.spawners.from_files",
		"isaaclab.sim.spawners",
		"isaaclab.sim",
	)
	for module_name in module_names:
		try:
			module = __import__(module_name, fromlist=["GroundPlaneCfg"])
			cls = getattr(module, "GroundPlaneCfg", None)
			if cls is None:
				continue
			setattr(cls, "usd_path", local_ground_usd)
			_patch_dataclass_default(cls, "usd_path", local_ground_usd)
		except Exception:
			continue


def patch_app_launcher_for_local_assets(cfg: Any) -> str | None:
	root = local_isaac_asset_root()
	if root is None:
		return None
	ensure_tdmpc2_on_path()
	from envs import isaaclab as newt_isaaclab

	newt_isaaclab._add_isaaclab_to_sys_path(cfg.isaaclab_dir)
	try:
		import isaaclab.app as app_mod
	except Exception:
		return None
	orig = app_mod.AppLauncher
	if getattr(orig, "_newt_local_asset_root_patched", False):
		return str(root)

	class LocalAssetRootAppLauncher(orig):
		_newt_local_asset_root_patched = True

		def __init__(self, *args: Any, **kwargs: Any) -> None:
			_apply_local_isaac_asset_root()
			super().__init__(*args, **kwargs)
			applied = _apply_local_isaac_asset_root()
			if applied:
				print(f"[automate_param_audit] Isaac asset root set to local mirror: {applied}")

	app_mod.AppLauncher = LocalAssetRootAppLauncher
	return str(root)


def _parse_vec_literal(value: Any) -> list[float] | None:
	if value is None:
		return None
	if isinstance(value, (list, tuple)):
		return [float(x) for x in value]
	text = str(value).strip()
	if not text:
		return None
	try:
		parsed = json.loads(text)
	except Exception:
		try:
			parsed = ast.literal_eval(text)
		except Exception:
			parsed = None
	if isinstance(parsed, (list, tuple)):
		return [float(x) for x in parsed]
	return None


def load_task_id_vectors(csv_path: str | Path = DEFAULT_TASK_HASH_CSV) -> dict[str, dict[str, Any]]:
	path = resolve_path(csv_path)
	results: dict[str, dict[str, Any]] = {}
	if not path.exists():
		return results
	with open(path, "r", encoding="utf-8", newline="") as f:
		reader = csv.DictReader(f)
		for row in reader:
			task_id = str(row.get("task_id") or row.get("assembly_id") or "").zfill(5)
			vec = _parse_vec_literal(row.get("values"))
			if not task_id or vec is None:
				continue
			results[task_id] = {
				"task_id": task_id,
				"assembly_id": str(row.get("assembly_id") or task_id).zfill(5),
				"template_id": str(row.get("template_id") or "default"),
				"condition_id": str(row.get("condition_id") or f"{task_id}|default"),
				"role": str(row.get("role") or ""),
				"replay_fp": str(row.get("replay_fp") or ""),
				"task_hash": str(row.get("task_hash") or task_vec_hash(vec)),
				"task_vec_6": vec[:6],
				"count": int(float(row.get("count") or 0)),
			}
	return results


def decode_task_vec(
	vec: list[float],
	*,
	reference_radius_m: float = DEFAULT_REFERENCE_RADIUS_M,
	reference_depth_m: float = DEFAULT_REFERENCE_DEPTH_M,
	base_plug_diameter_m: float = DEFAULT_BASE_PLUG_DIAMETER_M,
) -> dict[str, float]:
	values = [float(x) for x in vec[:6]]
	scale_ratio = math.exp(values[1])
	radial_clearance_m = values[2] * reference_radius_m
	plug_diameter_m = base_plug_diameter_m * scale_ratio
	male_radius_m = max(0.5 * plug_diameter_m, 1.0e-8)
	rel_clearance_from_decoded_male_radius = radial_clearance_m / male_radius_m
	target_depth_m = values[4] * reference_depth_m
	return {
		"task_type_id_float": values[0],
		"log_scale": values[1],
		"scale_ratio": scale_ratio,
		"plug_diameter_m_if_base_scale": plug_diameter_m,
		"clearance_abs_norm": values[2],
		"clearance_rel_norm": values[3],
		"radial_clearance_m_if_default_reference": radial_clearance_m,
		"diametral_clearance_m_if_default_reference": 2.0 * radial_clearance_m,
		"clearance_rel_norm_recomputed_from_default_reference": rel_clearance_from_decoded_male_radius,
		"depth_abs_norm": values[4],
		"target_depth_m_if_default_reference": target_depth_m,
		"yaw_requirement_float": values[5],
	}


def compare_task_vectors(task_a: dict[str, Any], task_b: dict[str, Any]) -> dict[str, Any]:
	vec_a = [float(x) for x in task_a.get("task_vec_6", [])[:6]]
	vec_b = [float(x) for x in task_b.get("task_vec_6", [])[:6]]
	deltas = {}
	for index, field in enumerate(AXIAL_TASK_VEC_FIELDS):
		a = vec_a[index] if index < len(vec_a) else None
		b = vec_b[index] if index < len(vec_b) else None
		deltas[field] = None if a is None or b is None else float(b - a)
	return {
		"task_a": task_a.get("task_id"),
		"task_b": task_b.get("task_id"),
		"hash_a": task_a.get("task_hash"),
		"hash_b": task_b.get("task_hash"),
		"task_vec_a": vec_a,
		"task_vec_b": vec_b,
		"task_vec_deltas_b_minus_a": deltas,
		"task_vecs_differ": bool(vec_a != vec_b),
	}


def tensor_like_to_python(value: Any, *, max_items: int = 64) -> Any:
	try:
		import torch
	except Exception:
		torch = None
	if torch is not None and torch.is_tensor(value):
		data = value.detach().cpu().reshape(-1)
		items = data[:max_items].tolist()
		return {
			"shape": list(value.shape),
			"dtype": str(value.dtype),
			"values": [float(x) if isinstance(x, (int, float)) else x for x in items],
			"truncated": int(data.numel()) > max_items,
		}
	if isinstance(value, dict):
		return {str(k): tensor_like_to_python(v, max_items=max_items) for k, v in value.items()}
	if isinstance(value, (list, tuple)):
		return [tensor_like_to_python(v, max_items=max_items) for v in list(value)[:max_items]]
	if isinstance(value, (str, int, float, bool)) or value is None:
		return value
	return str(value)


def collect_string_attrs(obj: Any, *, names: tuple[str, ...], max_items: int = 48) -> dict[str, Any]:
	found: dict[str, Any] = {}
	for name in dir(obj):
		lower = name.lower()
		if not any(token in lower for token in names):
			continue
		if name.startswith("__"):
			continue
		try:
			value = getattr(obj, name)
		except Exception:
			continue
		if isinstance(value, (str, int, float, bool)) or value is None:
			found[name] = value
		if len(found) >= max_items:
			break
	return found


def build_srsa_cfg_for_probe(assembly_id: str, extra_overrides: dict[str, Any] | None = None) -> Any:
	ensure_tdmpc2_on_path()
	from omegaconf import OmegaConf
	from config import Config, parse_cfg

	cfg = OmegaConf.structured(Config)
	cfg.isaaclab_backend = "srsa"
	cfg.task = "isaaclab-srsa-assembly"
	cfg.assembly_id = str(assembly_id).zfill(5)
	cfg.gpu_id = 0
	OmegaConf.update(cfg, "device_id", 0, force_add=True)
	cfg.num_envs = 1
	cfg.batch_size = 1
	cfg.isaaclab_max_episode_steps = 32
	cfg.isaaclab_episode_length_s = 2.0
	cfg.isaaclab_headless = True
	cfg.isaaclab_enable_cameras = False
	cfg.save_video = False
	cfg.srsa_use_runtime_task_vec = True
	cfg.task_conditioning = "axial_params"
	cfg.srsa_enable_flange_force_sensor = True
	cfg.srsa_flange_force_sensor_obs_frame = "socket"
	cfg.srsa_flange_force_sensor_force_threshold = 1.0
	if extra_overrides:
		for key, value in extra_overrides.items():
			OmegaConf.update(cfg, key, value, merge=True)
	return parse_cfg(cfg)


def launch_probe_env(assembly_id: str, extra_overrides: dict[str, Any] | None = None) -> tuple[Any, Any]:
	ensure_tdmpc2_on_path()
	from envs import make_env

	cfg = build_srsa_cfg_for_probe(assembly_id, extra_overrides=extra_overrides)
	patch_app_launcher_for_local_assets(cfg)
	prev_cwd = Path.cwd()
	automate_root = None
	root = local_isaac_asset_root()
	if root is not None:
		candidate = root / "Isaac" / "IsaacLab" / "AutoMate"
		if (candidate / "plug_grasps.json").exists() and (candidate / "disassembly_dist.json").exists():
			automate_root = candidate
	try:
		old_assembly_id_env = os.environ.get("NEWT_AUTOMATE_AUDIT_ASSEMBLY_ID")
		os.environ["NEWT_AUTOMATE_AUDIT_ASSEMBLY_ID"] = str(assembly_id).zfill(5)
		if automate_root is not None:
			os.chdir(automate_root)
		env = make_env(cfg)
	finally:
		if old_assembly_id_env is None:
			os.environ.pop("NEWT_AUTOMATE_AUDIT_ASSEMBLY_ID", None)
		else:
			os.environ["NEWT_AUTOMATE_AUDIT_ASSEMBLY_ID"] = old_assembly_id_env
		os.chdir(prev_cwd)
	return cfg, env


def close_env(env: Any) -> None:
	for name in ("close", "shutdown"):
		fn = getattr(env, name, None)
		if callable(fn):
			try:
				fn()
			except Exception:
				pass
			return


def collect_runtime_geometry(env: Any, cfg: Any | None = None) -> dict[str, Any]:
	unwrapped = getattr(env, "unwrapped", env)
	success_metrics = _collect_success_metrics(unwrapped)
	cfg_task = getattr(unwrapped, "cfg_task", None) or getattr(unwrapped, "cfg", None)
	usd_paths = _collect_cfg_asset_paths(cfg_task)
	prim_paths = _collect_prim_paths(unwrapped)
	usd_stage = _get_stage(unwrapped)
	prim_records = _collect_prim_records(usd_stage, prim_paths)
	info: dict[str, Any] = {
		"runtime_check": "DONE",
		"assembly_id": tensor_like_to_python(getattr(unwrapped, "assembly_id", getattr(cfg, "assembly_id", None))),
		"current_task_vec": tensor_like_to_python(getattr(unwrapped, "current_task_vec", None)),
		"current_task_params": tensor_like_to_python(getattr(unwrapped, "current_task_params", None)),
		"current_task_param_tensors": tensor_like_to_python(getattr(unwrapped, "current_task_param_tensors", None)),
		"path_like_attrs": collect_string_attrs(unwrapped, names=("asset", "mesh", "usd", "path", "assembly")),
		"scale_like_attrs": collect_string_attrs(unwrapped, names=("scale", "diameter", "clearance", "depth", "tol")),
		"loaded_asset_paths": usd_paths,
		"prim_paths": prim_paths,
		"prim_records": prim_records,
		"target_pose": _collect_target_pose(unwrapped),
		"success_thresholds": _collect_success_thresholds(unwrapped, cfg, success_metrics),
		"success_metrics_snapshot": tensor_like_to_python(success_metrics),
		"read_limits": [],
	}
	for key in ("held_asset", "fixed_asset"):
		if key not in prim_records:
			info["read_limits"].append(f"{key} prim path could not be resolved.")
	for attr in ("held_pos", "fixed_pos", "held_quat", "fixed_quat", "plug_diameter", "hole_diameter", "current_insertion_depth"):
		if hasattr(unwrapped, attr):
			info[attr] = tensor_like_to_python(getattr(unwrapped, attr))
		else:
			info[attr] = unknown(f"env has no attribute {attr!r}")
	params = getattr(unwrapped, "current_task_params", None)
	if isinstance(params, dict):
		for key in (
			"radial_clearance",
			"clearance",
			"insertion_depth",
			"target_insertion_depth",
			"success_pos_tol",
			"scale_ratio",
			"plug_scale_xy",
			"hole_scale_xy",
			"plug_diameter",
			"hole_diameter",
			"yaw_requirement",
			"yaw_requirement_float",
		):
			if key in params:
				info[key] = tensor_like_to_python(params[key])
			else:
				info[key] = unknown(f"current_task_params has no {key!r}")
	else:
		info["current_task_params_limit"] = unknown("env.current_task_params is not a dict")
	return info


def compare_geometry_records(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
	fields = (
		"path_like_attrs",
		"scale_like_attrs",
		"loaded_asset_paths",
		"prim_paths",
		"current_task_vec",
		"current_task_params",
		"current_task_param_tensors",
		"target_pose",
		"success_thresholds",
		"radial_clearance",
		"clearance",
		"insertion_depth",
		"target_insertion_depth",
		"success_pos_tol",
		"plug_scale_xy",
		"hole_scale_xy",
		"plug_diameter",
		"hole_diameter",
	)
	comparable = {}
	for field in fields:
		av = a.get(field)
		bv = b.get(field)
		comparable[field] = {
			"a": av,
			"b": bv,
			"same": av == bv,
			"comparable": av is not None and bv is not None,
		}
	comparable_values = [item for item in comparable.values() if item["comparable"]]
	prim_compare = _compare_prim_records(a.get("prim_records") or {}, b.get("prim_records") or {})
	asset_path_differs = _field_diff(comparable, "loaded_asset_paths")
	scale_differs = _prim_any_diff(prim_compare, ("scale",)) or _field_diff(comparable, "scale_like_attrs")
	visual_aabb_differs = _prim_any_diff(prim_compare, ("visual_aabb", "world_aabb"))
	collision_aabb_differs = _prim_any_diff(prim_compare, ("collision_aabb",))
	target_depth_differs = _field_diff(comparable, "target_pose") or _field_diff(comparable, "insertion_depth") or _field_diff(comparable, "target_insertion_depth")
	threshold_differs = _field_diff(comparable, "success_thresholds") or _field_diff(comparable, "success_pos_tol")
	return {
		"field_comparisons": comparable,
		"prim_comparisons": prim_compare,
		"asset_path_differs": asset_path_differs,
		"visual_aabb_differs": visual_aabb_differs,
		"collision_aabb_differs": collision_aabb_differs,
		"prim_scale_differs": scale_differs,
		"target_depth_differs": target_depth_differs,
		"success_threshold_differs": threshold_differs,
		"geometry_effective": bool(asset_path_differs or visual_aabb_differs or collision_aabb_differs or scale_differs),
		"objective_effective": bool(target_depth_differs or threshold_differs),
		"any_runtime_field_differs": any(not item["same"] for item in comparable_values),
		"num_comparable_fields": len(comparable_values),
	}


def _collect_cfg_asset_paths(cfg_task: Any) -> dict[str, Any]:
	if cfg_task is None:
		return {
			"held_usd_path": unknown("cfg_task is unavailable"),
			"fixed_usd_path": unknown("cfg_task is unavailable"),
		}
	result = {}
	for logical, attr in (("held", "held_asset"), ("fixed", "fixed_asset")):
		asset = getattr(cfg_task, attr, None)
		spawn = getattr(asset, "spawn", None)
		result[f"{logical}_usd_path"] = getattr(spawn, "usd_path", unknown(f"cfg_task.{attr}.spawn.usd_path unavailable"))
		result[f"{logical}_spawn_scale"] = tensor_like_to_python(getattr(spawn, "scale", unknown(f"cfg_task.{attr}.spawn.scale unavailable")))
	for logical, attr in (("held", "held_asset_cfg"), ("fixed", "fixed_asset_cfg")):
		asset_cfg = getattr(cfg_task, attr, None)
		result[f"{logical}_diameter_cfg"] = tensor_like_to_python(getattr(asset_cfg, "diameter", unknown(f"cfg_task.{attr}.diameter unavailable")))
	return result


def _get_stage(unwrapped: Any) -> Any:
	scene = getattr(unwrapped, "scene", None)
	stage = getattr(scene, "stage", None)
	if stage is not None:
		return stage
	try:
		from omni.usd import get_context
		return get_context().get_stage()
	except Exception:
		return None


def _collect_prim_paths(unwrapped: Any) -> dict[str, Any]:
	scene = getattr(unwrapped, "scene", None)
	env_paths = getattr(scene, "env_prim_paths", None)
	env_path = None
	try:
		env_path = str(env_paths[0]) if env_paths else None
	except Exception:
		env_path = None
	if env_path:
		return {
			"env_prim_path": env_path,
			"held_asset": f"{env_path}/HeldAsset",
			"fixed_asset": f"{env_path}/FixedAsset",
			"peg": f"{env_path}/HeldAsset",
			"socket": f"{env_path}/FixedAsset",
		}
	return {
		"env_prim_path": unknown("scene.env_prim_paths unavailable"),
		"held_asset": unknown("scene.env_prim_paths unavailable"),
		"fixed_asset": unknown("scene.env_prim_paths unavailable"),
		"peg": unknown("scene.env_prim_paths unavailable"),
		"socket": unknown("scene.env_prim_paths unavailable"),
	}


def _collect_prim_records(stage: Any, prim_paths: dict[str, Any]) -> dict[str, Any]:
	if stage is None:
		return {
			"held_asset": unknown("USD stage unavailable"),
			"fixed_asset": unknown("USD stage unavailable"),
		}
	records = {}
	for key in ("held_asset", "fixed_asset"):
		path = prim_paths.get(key)
		if not isinstance(path, str):
			records[key] = unknown(f"{key} prim path unavailable")
			continue
		records[key] = _inspect_usd_prim(stage, path)
	return records


def _inspect_usd_prim(stage: Any, prim_path: str) -> dict[str, Any]:
	try:
		from pxr import Gf, UsdGeom
	except Exception as exc:
		return unknown(f"pxr UsdGeom unavailable: {exc!r}", prim_path=prim_path)
	prim = stage.GetPrimAtPath(prim_path)
	if not prim or not prim.IsValid():
		return unknown("prim path is invalid", prim_path=prim_path)
	record: dict[str, Any] = {"prim_path": prim_path, "type_name": str(prim.GetTypeName())}
	try:
		xformable = UsdGeom.Xformable(prim)
		matrix = xformable.ComputeLocalToWorldTransform(0)
		translation = matrix.ExtractTranslation()
		record["world_transform_translation"] = [float(translation[0]), float(translation[1]), float(translation[2])]
		record["scale"] = _xform_scale(xformable)
	except Exception as exc:
		record["world_transform"] = unknown(f"failed to compute transform: {exc!r}")
	try:
		cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=True)
		box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
		record["visual_aabb"] = _range3d_to_dict(box)
		record["world_aabb"] = record["visual_aabb"]
	except Exception as exc:
		record["visual_aabb"] = unknown(f"failed to compute visual/render AABB: {exc!r}")
	try:
		cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy], useExtentsHint=True)
		box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
		record["collision_aabb"] = _range3d_to_dict(box)
	except Exception as exc:
		record["collision_aabb"] = unknown(f"failed to compute proxy/collision AABB: {exc!r}")
	return record


def _xform_scale(xformable: Any) -> Any:
	for op in xformable.GetOrderedXformOps():
		try:
			if "scale" in str(op.GetOpType()).lower():
				value = op.Get()
				return [float(value[0]), float(value[1]), float(value[2])]
		except Exception:
			continue
	return unknown("no explicit scale op found")


def _range3d_to_dict(box: Any) -> dict[str, Any]:
	min_v = box.GetMin()
	max_v = box.GetMax()
	return {
		"min": [float(min_v[0]), float(min_v[1]), float(min_v[2])],
		"max": [float(max_v[0]), float(max_v[1]), float(max_v[2])],
		"size": [float(max_v[i] - min_v[i]) for i in range(3)],
	}


def _collect_target_pose(unwrapped: Any) -> dict[str, Any]:
	result = {}
	for attr in (
		"gripper_goal_pos",
		"gripper_goal_quat",
		"fixed_pos",
		"fixed_quat",
		"current_insertion_depth_tensor",
		"disassembly_dists",
		"current_insertion_depth",
	):
		if hasattr(unwrapped, attr):
			result[attr] = tensor_like_to_python(getattr(unwrapped, attr))
		else:
			result[attr] = unknown(f"env has no attribute {attr!r}")
	return result


def _collect_success_metrics(unwrapped: Any) -> Any:
	for name in ("_compute_depth_contact_jam", "_compute_srsa_success_metrics"):
		fn = getattr(unwrapped, name, None)
		if callable(fn):
			try:
				if name == "_compute_srsa_success_metrics":
					return fn(update_state=False)
				return fn()
			except Exception as exc:
				return unknown(f"{name} failed: {exc!r}")
	return unknown("env exposes no SRSA success metric function")


def _collect_success_thresholds(unwrapped: Any, cfg: Any | None, success_metrics: Any) -> dict[str, Any]:
	result = {}
	for key in (
		"target_depth",
		"radial_clearance",
		"lateral_tol",
		"relaxed_lateral_tol",
		"keypoint_tol",
		"relaxed_keypoint_tol",
		"yaw_error",
	):
		if isinstance(success_metrics, dict) and key in success_metrics:
			result[key] = tensor_like_to_python(success_metrics[key])
		else:
			result[key] = unknown(f"success metric {key!r} unavailable")
	cfg_task = getattr(unwrapped, "cfg_task", None) or getattr(unwrapped, "cfg", None)
	for attr in ("close_error_thresh", "success_pos_tol", "strict_depth_fraction", "relaxed_depth_fraction"):
		value = getattr(cfg_task, attr, None)
		if value is None and cfg is not None:
			try:
				value = cfg.get(attr, None)
			except Exception:
				value = None
		result[attr] = tensor_like_to_python(value) if value is not None else unknown(f"{attr!r} unavailable")
	return result


def _field_diff(comparable: dict[str, Any], field: str) -> bool:
	item = comparable.get(field) or {}
	return bool(item.get("comparable") and not item.get("same"))


def _prim_any_diff(prim_compare: dict[str, Any], fields: tuple[str, ...]) -> bool:
	for prim_record in prim_compare.values():
		if not isinstance(prim_record, dict):
			continue
		for field in fields:
			item = prim_record.get(field)
			if isinstance(item, dict) and item.get("comparable") and not item.get("same"):
				return True
	return False


def _compare_prim_records(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
	result = {}
	for key in sorted(set(a.keys()) | set(b.keys())):
		result[key] = {}
		a_rec = a.get(key, {})
		b_rec = b.get(key, {})
		if not isinstance(a_rec, dict) or not isinstance(b_rec, dict):
			result[key]["record"] = {"a": a_rec, "b": b_rec, "same": a_rec == b_rec, "comparable": False}
			continue
		for field in ("prim_path", "type_name", "world_transform_translation", "scale", "visual_aabb", "world_aabb", "collision_aabb"):
			av = a_rec.get(field)
			bv = b_rec.get(field)
			result[key][field] = {
				"a": av,
				"b": bv,
				"same": av == bv,
				"comparable": av is not None and bv is not None and not _is_unknown(av) and not _is_unknown(bv),
			}
	return result


def _is_unknown(value: Any) -> bool:
	return isinstance(value, dict) and value.get("status") == "UNKNOWN_WITH_REASON"
