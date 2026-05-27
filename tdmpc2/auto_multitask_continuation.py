from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import torch
from termcolor import colored

from common import set_seed
from common.logger import Logger
from common.multitask_replay import get_active_tasks
from offline_io import load_offline_manifest
from offline_train import (
	_make_hard_case_dataset,
	_make_offline_dataset,
	_offline_gpu_id,
	_prepare_cfg_from_dataset,
	_resolve_offline_dataset_fp,
	_run_multitask_continuation,
	make_agent,
)


COLLECT_OVERRIDE_FIELDS = (
	"eval_success_metric",
	"srsa_eval_success_metric",
	"strict_depth_fraction",
	"strict_success_steps",
	"strict_lateral_tol_min",
	"strict_lateral_tol_max",
	"strict_keypoint_tol_min",
	"strict_keypoint_tol_max",
	"strict_angle_tol_deg",
	"relaxed_depth_fraction",
	"relaxed_success_steps",
	"relaxed_lateral_tol_scale",
	"relaxed_lateral_tol_min",
	"relaxed_lateral_tol_max",
	"relaxed_keypoint_tol_scale",
	"relaxed_keypoint_tol_min",
	"relaxed_keypoint_tol_max",
	"relaxed_angle_tol_deg",
	"relaxed_success_require_official",
	"relaxed_success_require_no_jam",
	"isaaclab_backend",
	"isaaclab_env_id",
	"isaaclab_task_package",
	"task",
	"isaaclab_dir",
	"srsa_dir",
	"srsa_sparse_reward",
	"srsa_align_direct_reward_success",
	"srsa_sil",
	"srsa_if_sbc",
	"srsa_task_template_fp",
	"srsa_task_template_id",
	"srsa_param_template_id",
	"srsa_mesh_geometry_fp",
	"srsa_mesh_geometry_task_id",
	"srsa_mesh_plug_diameter_column",
	"srsa_mesh_hole_diameter_column",
	"srsa_mesh_clearance_column",
	"srsa_mesh_clearance_mode",
	"srsa_mesh_clearance_scale",
	"srsa_mesh_depth_column",
	"srsa_mesh_depth_scale",
	"srsa_mesh_reference_radius_column",
	"srsa_mesh_reference_depth_column",
	"srsa_position_control_only",
	"srsa_policy_action_dim",
	"srsa_env_action_dim",
	"num_envs",
	"gpu_id",
	"num_gpus",
	"model_size",
	"horizon",
	"compile",
	"mpc",
	"eval_terminate_on_success",
	"eval_terminate_success_key",
	"eval_terminate_min_step",
	"isaaclab_headless",
	"isaaclab_use_canonical_obs",
	"isaaclab_gpu_collision_stack_size",
	"isaaclab_disable_imitation_reward",
	"srsa_task_family_name",
	"srsa_task_param_obs",
	"srsa_task_param_obs_mode",
	"srsa_newt_obs",
	"srsa_enable_axial_task_param_sampler",
	"srsa_use_runtime_task_vec",
	"srsa_axial_task_type_id",
	"srsa_axial_scale_range",
	"srsa_axial_fixed_plug_scale",
	"srsa_axial_clearance_range",
	"srsa_axial_clearance_ratio_range",
	"srsa_axial_clearance_base",
	"srsa_axial_clearance_anchor_multipliers",
	"srsa_axial_clearance_anchors",
	"srsa_axial_clearance_depth_templates",
	"srsa_axial_clearance_depth_template_multipliers",
	"srsa_axial_clearance_depth_template_weights",
	"srsa_axial_clearance_jitter_ratio",
	"srsa_axial_clearance_anchor_weights",
	"srsa_axial_depth_range",
	"srsa_axial_target_depth_range",
	"srsa_axial_depth_base",
	"srsa_axial_depth_anchor_multipliers",
	"srsa_axial_depth_anchors",
	"srsa_axial_depth_jitter_ratio",
	"srsa_axial_depth_anchor_weights",
	"srsa_axial_init_error_xy_range",
	"srsa_axial_init_error_z_range",
	"srsa_axial_init_error_yaw_range",
	"srsa_axial_visual_noise_xy_range",
	"srsa_axial_visual_noise_z_range",
	"srsa_axial_yaw_requirement",
	"srsa_axial_reference_radius",
	"srsa_axial_reference_depth",
	"srsa_axial_reference_anchor_assembly_id",
	"srsa_axial_reference_anchor_task_type_id",
	"srsa_enable_flange_force_sensor",
	"isaaclab_canonical_append_force",
	"isaaclab_canonical_append_task_params",
	"isaaclab_canonical_use_visual_noise",
	"task_conditioning",
	"collect_episodes_per_task",
	"collect_weak_task_episodes",
	"collect_screening_fp",
	"collect_weak_success_threshold",
	"collect_defer_success_threshold",
	"collect_defer_depth_threshold",
	"collect_skip_deferred_tasks",
	"collect_match_checkpoint",
	"collect_expected_obs_dim",
	"collect_parallel_workers",
	"collect_parallel_gpu_ids",
	"collect_mpc",
	"collect_max_env_steps",
	"eval_hang_guard_factor",
	"progress_log_interval_sec",
	"eval_task_template_exact",
	"eval_task_template_print",
	"contact_history_enabled",
	"contact_history_len",
	"contact_context_dim",
	"contact_history_hidden_dim",
	"contact_history_layers",
	"contact_force_dim",
	"contact_action_dim",
	"contact_ee_delta_dim",
	"contact_history_use_ee_delta",
)


def _cfg_get(cfg, key, default=None):
	return cfg.get(key, default) if hasattr(cfg, "get") else getattr(cfg, key, default)


def _cfg_set(cfg, key, value):
	if hasattr(cfg, key):
		setattr(cfg, key, value)
	else:
		cfg[key] = value


def _coerce_str_list(raw) -> list[str]:
	if raw is None:
		return []
	if isinstance(raw, str):
		text = raw.strip().strip("'\"")
		if text.startswith("[") and text.endswith("]"):
			try:
				items = json.loads(text)
			except json.JSONDecodeError:
				items = [part for part in text[1:-1].replace(";", ",").replace(" ", ",").split(",") if part]
		else:
			items = [part for part in text.replace(";", ",").replace(" ", ",").split(",") if part]
	else:
		items = list(raw)
	return [str(item).strip().strip("'\"") for item in items if str(item).strip().strip("'\"")]


def _override_value(value):
	if isinstance(value, bool):
		return "true" if value else "false"
	if value is None:
		return "null"
	if isinstance(value, (list, tuple)):
		return json.dumps(list(value))
	if isinstance(value, str) and ("," in value or ";" in value):
		return json.dumps(value)
	return str(value)


def _list_override(values: list[str]) -> str:
	return json.dumps([str(value) for value in values])


def _normalize_task_id(value) -> str:
	text = str(value).strip().strip("'\"")
	return text.zfill(5) if text.isdigit() and len(text) < 5 else text


def _manifest_assembly_ids(manifest_fp: Path) -> set[str]:
	assembly_ids: set[str] = set()
	try:
		entries = load_offline_manifest(manifest_fp)
	except Exception as exc:
		print(colored(
			f"Warning: could not inspect manifest task ids from {manifest_fp}: {exc}",
			"yellow",
		), flush=True)
		return assembly_ids
	for entry in entries:
		assembly_id = entry.get("assembly_id")
		if assembly_id is not None:
			assembly_ids.add(_normalize_task_id(assembly_id))
	return assembly_ids


def _stage_plan(cfg) -> list[dict]:
	task_ids = _coerce_str_list(_cfg_get(cfg, "multitask_task_ids", []))
	if not task_ids:
		raise ValueError("multitask_auto_collect_replay=true requires non-empty multitask_task_ids.")
	total_steps = int(_cfg_get(cfg, "multitask_total_steps", None) or _cfg_get(cfg, "steps", 0))
	if total_steps <= 0:
		raise ValueError(f"Expected positive multitask total steps, got {total_steps}.")
	mode = str(_cfg_get(cfg, "multitask_curriculum_mode", "progressive")).lower()
	if mode == "all_at_once":
		return [{"stage_idx": 0, "start_step": 0, "num_updates": total_steps, "active_tasks": task_ids}]
	stage_steps = int(_cfg_get(cfg, "multitask_stage_steps", 0) or 0)
	if stage_steps <= 0:
		raise ValueError(f"multitask_stage_steps must be positive, got {stage_steps}.")
	plans = []
	start_step = 0
	while start_step < total_steps:
		active_tasks = get_active_tasks(start_step, task_ids, stage_steps, mode)
		num_updates = min(stage_steps, total_steps - start_step)
		plans.append({
			"stage_idx": len(plans),
			"start_step": start_step,
			"num_updates": num_updates,
			"active_tasks": active_tasks,
		})
		start_step += num_updates
	return plans


def _collect_overrides(cfg, *, checkpoint_fp: Path, collect_tasks: list[str], output_dir: Path, manifest_fp: Path, stage_idx: int):
	overrides = [f"checkpoint={checkpoint_fp}"]
	for field in COLLECT_OVERRIDE_FIELDS:
		value = _cfg_get(cfg, field, None)
		if value is None:
			continue
		overrides.append(f"{field}={_override_value(value)}")
	overrides.extend([
		f"collect_assembly_ids={_list_override(collect_tasks)}",
		f"collect_output_dir={output_dir}",
		f"collect_manifest_fp={manifest_fp}",
		"collect_source_assembly_id=null",
		"collect_exclude_source_assembly=false",
		"include_source_anchor_rollouts=true",
		"collect_spawn_per_assembly=true",
		"collect_overwrite=true",
		"enable_wandb=false",
		f"exp_name={_cfg_get(cfg, 'exp_name', 'multitask')}_auto_collect_stage_{stage_idx:02d}",
	])
	return overrides


def _run_collect_stage(cfg, *, checkpoint_fp: Path, collect_tasks: list[str], stage_idx: int) -> Path:
	stage_dir = Path(_cfg_get(cfg, "work_dir")) / "auto_replay" / f"stage_{stage_idx:02d}"
	output_dir = stage_dir / "rollouts"
	manifest_fp = stage_dir / "offline_manifest_stage.json"
	script = Path(__file__).resolve().parent / "collect_eval_rollouts.py"
	cmd = [
		sys.executable,
		str(script),
		*_collect_overrides(
			cfg,
			checkpoint_fp=checkpoint_fp,
			collect_tasks=collect_tasks,
			output_dir=output_dir,
			manifest_fp=manifest_fp,
			stage_idx=stage_idx,
		),
	]
	print(colored(
		f"auto_collect stage={stage_idx} collect_tasks={collect_tasks} checkpoint={checkpoint_fp}",
		"cyan",
		attrs=["bold"],
	), flush=True)
	subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], check=True)
	if not manifest_fp.exists():
		raise FileNotFoundError(f"Collection did not write manifest: {manifest_fp}")
	return manifest_fp


def _merge_manifests(manifest_fps: list[Path], *, work_dir: Path, stage_idx: int) -> Path:
	if len(manifest_fps) == 1:
		return manifest_fps[0]
	merged_dir = work_dir / "auto_replay" / f"stage_{stage_idx:02d}" / "merged_rollouts"
	merged_manifest = work_dir / "auto_replay" / f"stage_{stage_idx:02d}" / "offline_manifest_accumulated.json"
	script = Path(__file__).resolve().parent / "scripts" / "merge_offline_manifests.py"
	cmd = [
		sys.executable,
		str(script),
		*[str(path) for path in manifest_fps],
		"--output-manifest-fp",
		str(merged_manifest),
		"--output-dir",
		str(merged_dir),
		"--overwrite",
	]
	print(colored(
		f"Merging {len(manifest_fps)} replay manifests -> {merged_manifest}",
		"cyan",
		attrs=["bold"],
	), flush=True)
	subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], check=True)
	return merged_manifest


def _train_stage(cfg, *, checkpoint_fp: Path, manifest_fp: Path, active_tasks: list[str], num_updates: int, stage_idx: int) -> Path:
	stage_cfg = deepcopy(cfg)
	_cfg_set(stage_cfg, "checkpoint", str(checkpoint_fp))
	_cfg_set(stage_cfg, "offline_manifest_fp", str(manifest_fp))
	_cfg_set(stage_cfg, "multitask_replay_manifest_fp", str(manifest_fp))
	_cfg_set(stage_cfg, "offline_dataset_fp", None)
	_cfg_set(stage_cfg, "offline_source_fp", None)
	_cfg_set(stage_cfg, "offline_export_fp", str(Path(_cfg_get(cfg, "work_dir")) / "auto_replay" / f"stage_{stage_idx:02d}" / "offline_compact.pt"))
	_cfg_set(stage_cfg, "offline_export_overwrite", True)
	_cfg_set(stage_cfg, "multitask_task_ids", list(active_tasks))
	_cfg_set(stage_cfg, "multitask_eval_task_ids", list(active_tasks))
	_cfg_set(stage_cfg, "multitask_curriculum_mode", "all_at_once")
	_cfg_set(stage_cfg, "multitask_total_steps", int(num_updates))
	_cfg_set(stage_cfg, "steps", int(num_updates))
	_cfg_set(stage_cfg, "enable_wandb", False)
	_cfg_set(stage_cfg, "rank", 0)
	_cfg_set(stage_cfg, "world_size", 1)

	offline_gpu = _offline_gpu_id(stage_cfg)
	torch.cuda.set_device(offline_gpu)
	offline_dataset_fp = _resolve_offline_dataset_fp(stage_cfg)
	offline_device = f"cuda:{offline_gpu}"
	dataset = _make_offline_dataset(
		stage_cfg,
		offline_dataset_fp,
		filter_mode=_cfg_get(stage_cfg, "offline_filter_mode", "all"),
		device=offline_device,
	)
	hard_dataset = _make_hard_case_dataset(stage_cfg, offline_dataset_fp, device=offline_device)
	stage_cfg = _prepare_cfg_from_dataset(stage_cfg, dataset)
	logger = Logger(stage_cfg)
	agent = make_agent(stage_cfg)
	agent.load(str(checkpoint_fp))
	print(colored(
		f"Training shared model stage={stage_idx} updates={num_updates:,} replay_manifest={manifest_fp}",
		"cyan",
		attrs=["bold"],
	))
	_run_multitask_continuation(agent, dataset, hard_dataset, logger, stage_cfg)

	stage_ckpt = Path(_cfg_get(cfg, "work_dir")) / "models" / f"auto_stage_{stage_idx:02d}.pt"
	stage_ckpt.parent.mkdir(parents=True, exist_ok=True)
	agent.save(stage_ckpt)
	latest_fp = stage_ckpt.parent / "latest.pt"
	shutil.copy2(stage_ckpt, latest_fp)
	print(colored(f"Saved auto-collect stage checkpoint: {stage_ckpt}", "green", attrs=["bold"]))
	return stage_ckpt


def run_auto_collect_multitask_continuation(cfg):
	"""
	Stage-wise online collection fallback for Scheme A.

	SRSA/IsaacLab assembly_id is process-static in the current wrapper, so this path
	collects each active task in isolated rollout workers, accumulates replay manifests,
	and then updates the single shared model from mixed replay before moving on.
	"""
	set_seed(_cfg_get(cfg, "seed", 1))
	work_dir = Path(_cfg_get(cfg, "work_dir")).expanduser().resolve()
	work_dir.mkdir(parents=True, exist_ok=True)
	current_checkpoint = Path(_cfg_get(cfg, "checkpoint")).expanduser().resolve()
	if not current_checkpoint.exists():
		raise FileNotFoundError(f"Warm-start checkpoint not found: {current_checkpoint}")
	plans = _stage_plan(cfg)
	stage_manifests: list[Path] = []
	collected_task_ids: set[str] = set()
	recollect_active_tasks = bool(_cfg_get(cfg, "multitask_auto_collect_recollect_active_tasks", False))
	initial_manifest = _cfg_get(cfg, "multitask_replay_manifest_fp", None) or _cfg_get(cfg, "offline_manifest_fp", None)
	if initial_manifest:
		initial_manifest = Path(initial_manifest).expanduser().resolve()
		if not initial_manifest.exists():
			raise FileNotFoundError(f"Initial replay manifest not found: {initial_manifest}")
		stage_manifests.append(initial_manifest)
		collected_task_ids.update(_manifest_assembly_ids(initial_manifest))
		print(colored(f"Seeded auto-collect replay pool with manifest: {initial_manifest}", "cyan", attrs=["bold"]))
	print(colored(
		"Starting auto-collect multitask continuation. "
		"Replay manifest is optional here because rollout data will be collected stage by stage.",
		"cyan",
		attrs=["bold"],
	))
	for plan in plans:
		stage_idx = int(plan["stage_idx"])
		active_tasks = list(plan["active_tasks"])
		num_updates = int(plan["num_updates"])
		if recollect_active_tasks:
			collect_tasks = active_tasks
		else:
			collect_tasks = [
				task_id for task_id in active_tasks
				if _normalize_task_id(task_id) not in collected_task_ids
			]
		print(colored(
			f"stage={stage_idx} active_tasks={active_tasks} collect_tasks={collect_tasks} "
			f"recollect_active_tasks={recollect_active_tasks}",
			"cyan",
			attrs=["bold"],
		), flush=True)
		if collect_tasks:
			stage_manifest = _run_collect_stage(
				cfg,
				checkpoint_fp=current_checkpoint,
				collect_tasks=collect_tasks,
				stage_idx=stage_idx,
			)
			stage_manifests.append(stage_manifest)
			manifest_ids = _manifest_assembly_ids(stage_manifest)
			collected_task_ids.update(manifest_ids or {_normalize_task_id(task_id) for task_id in collect_tasks})
		else:
			print(colored(
				f"Skipping auto_collect stage={stage_idx}: all active tasks already have replay.",
				"yellow",
			), flush=True)
		if not stage_manifests:
			raise ValueError(
				"No replay data is available for multitask continuation. "
				"Provide multitask_replay_manifest_fp or enable collection for at least one task."
			)
		accumulated_manifest = _merge_manifests(stage_manifests, work_dir=work_dir, stage_idx=stage_idx)
		current_checkpoint = _train_stage(
			cfg,
			checkpoint_fp=current_checkpoint,
			manifest_fp=accumulated_manifest,
			active_tasks=active_tasks,
			num_updates=num_updates,
			stage_idx=stage_idx,
		)
	final_fp = work_dir / "models" / "final.pt"
	shutil.copy2(current_checkpoint, final_fp)
	print(colored(f"Auto-collect multitask continuation complete: {final_fp}", "green", attrs=["bold"]))
	return final_fp
