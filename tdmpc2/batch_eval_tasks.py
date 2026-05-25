import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ["TORCH_DISTRIBUTED_TIMEOUT"] = "1800"
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
os.environ['TORCH_LOGS'] = "+recompiles"
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from time import monotonic
import csv
import json
import math
import subprocess
import sys

import hydra
import torch
from hydra.core.config_store import ConfigStore
from termcolor import colored

from common import set_seed
from config import Config, apply_eval_task_template, parse_cfg, safe_run_token
from collect_eval_rollouts import (
	_adapt_obs_to_checkpoint,
	_apply_checkpoint_compat,
	_make_agent,
	_model_task_input,
	_normalize_assembly_id,
	_override_value,
	_read_json,
	_write_json,
)
from envs import make_env
from offline_io import load_offline_manifest


torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

cs = ConfigStore.instance()
cs.store(name="config", node=Config)

SUCCESS_DIAGNOSTIC_KEYS = (
	"official_success_latched",
	"official_success_terminal",
	"process_success_terminal",
	"relaxed_process_success_terminal",
	"relaxed_success_stable",
	"relaxed_success_episode",
	"strict_success_stable",
	"strict_success_episode",
	"official_success",
	"current_official_success",
	"process_success",
	"relaxed_process_success",
	"episode_relaxed_process_success",
	"relaxed_terminal_process_success",
	"episode_process_success",
	"terminal_process_success",
	"dual_success",
	"depth_fraction",
	"lateral_error",
	"angle_error",
	"orientation_error",
	"yaw_error",
	"keypoint_error",
	"jam",
)


def _parse_assembly_ids(raw) -> list[str]:
	if raw is None:
		return []
	if isinstance(raw, str):
		text = raw.strip().strip("'\"")
		if text.startswith("[") and text.endswith("]"):
			text = text[1:-1]
		items = [item for item in text.replace(";", ",").replace(" ", ",").split(",") if item.strip()]
	else:
		items = list(raw)
	return list(dict.fromkeys(_normalize_assembly_id(item) for item in items))


def _resolve_eval_entries(cfg) -> list[dict]:
	manifest_entries = []
	if cfg.offline_manifest_fp:
		manifest_entries = load_offline_manifest(cfg.offline_manifest_fp)
	manifest_by_assembly = {
		_normalize_assembly_id(entry.get("assembly_id")): dict(entry)
		for entry in manifest_entries
		if entry.get("assembly_id") is not None
	}
	selected = _parse_assembly_ids(cfg.get('eval_assembly_ids', None))
	if not selected:
		selected = _parse_assembly_ids(cfg.get('batch_eval_assembly_ids', None))
	if selected:
		entries = []
		for index, assembly_id in enumerate(selected):
			if assembly_id in manifest_by_assembly:
				entry = dict(manifest_by_assembly[assembly_id])
			else:
				entry = {
					"task_id": index if not manifest_entries else len(entries),
					"task_name": f"{cfg.task}-{assembly_id}",
					"assembly_id": assembly_id,
				}
			entry["assembly_id"] = assembly_id
			entry["eval_index"] = index
			entries.append(entry)
	elif manifest_entries:
		entries = manifest_entries
	else:
		raise ValueError("Provide `eval_assembly_ids`, `batch_eval_assembly_ids`, or `offline_manifest_fp`.")
	if not entries:
		raise ValueError("No eval entries resolved from eval/batch assembly ids or offline_manifest_fp.")
	return entries


def _resolve_output_dir(cfg) -> Path:
	if cfg.get('batch_eval_output_dir', None):
		return Path(cfg.batch_eval_output_dir).expanduser().resolve()
	checkpoint_stem = safe_run_token(Path(cfg.checkpoint).stem if cfg.checkpoint else "checkpoint")
	return Path(cfg.work_dir) / "batch_eval" / checkpoint_stem


def _resolve_summary_fp(cfg, output_dir: Path) -> Path:
	if cfg.get('batch_eval_summary_fp', None):
		return Path(cfg.batch_eval_summary_fp).expanduser().resolve()
	return output_dir / "batch_eval_summary.json"


def _child_overrides(cfg, *, entry: dict, output_dir: Path):
	fields = [
		"checkpoint",
		"offline_manifest_fp",
		"eval_assembly_ids",
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
		"srsa_dir",
		"srsa_sparse_reward",
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
		"num_envs",
		"gpu_id",
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
		"srsa_enable_flange_force_sensor",
		"isaaclab_canonical_append_force",
		"isaaclab_canonical_append_task_params",
		"isaaclab_canonical_use_visual_noise",
		"task_conditioning",
		"contact_history_enabled",
		"contact_history_len",
		"contact_context_dim",
		"contact_history_hidden_dim",
		"contact_history_layers",
		"contact_force_dim",
		"contact_action_dim",
		"contact_ee_delta_dim",
		"contact_history_use_ee_delta",
		"learn_task_emb",
		"collect_match_checkpoint",
		"collect_expected_obs_dim",
		"batch_eval_episodes_per_task",
		"batch_eval_overwrite",
		"batch_eval_mpc",
		"batch_eval_max_env_steps",
		"enable_wandb",
		"exp_name",
		"run_id",
		"seed",
		"eval_hang_guard_factor",
		"progress_log_interval_sec",
		"eval_task_template_exact",
		"eval_task_template_print",
	]
	overrides = []
	for field in fields:
		value = cfg.get(field, None)
		if value is None:
			continue
		overrides.append(f"{field}={_override_value(value)}")
	overrides.extend([
		f"batch_eval_worker_assembly_id={entry['assembly_id']}",
		f"batch_eval_worker_task_id={int(entry['task_id'])}",
		f"batch_eval_worker_eval_index={int(entry.get('eval_index', entry['task_id']))}",
		"batch_eval_spawn_per_assembly=false",
		f"batch_eval_output_dir={output_dir}",
	])
	return overrides


def _run_subprocess_eval(cfg, entries: list[dict], output_dir: Path):
	script = Path(__file__).resolve()
	results = []
	for index, entry in enumerate(entries):
		assembly_id = _normalize_assembly_id(entry["assembly_id"])
		entry = dict(entry)
		entry["assembly_id"] = assembly_id
		result_fp = output_dir / assembly_id / "eval_metrics.json"
		if result_fp.exists() and cfg.get('batch_eval_overwrite', False):
			result_fp.unlink()
		cmd = [
			sys.executable,
			str(script),
			*_child_overrides(cfg, entry=entry, output_dir=output_dir),
		]
		print(colored(
			f"Launching isolated eval process for assembly_id={assembly_id} "
			f"task_id={int(entry['task_id'])} ({index + 1}/{len(entries)}).",
			"cyan",
			attrs=["bold"],
		), flush=True)
		subprocess.run(cmd, cwd=hydra.utils.get_original_cwd(), check=True)
		if not result_fp.exists():
			raise FileNotFoundError(f"Worker did not write eval metrics: {result_fp}")
		result = _read_json(result_fp)
		result["eval_index"] = index
		results.append(result)
	return results


def _mean(values):
	return float(sum(values) / max(1, len(values)))


def _metric(item, *names, default=0.0):
	for name in names:
		for key in (name, f"episode_{name}"):
			if key in item:
				return float(item[key])
	return default


def _summary_row(item):
	official_latched = _metric(item, "official_success_latched", "official_success")
	official_terminal = _metric(item, "official_success_terminal", "current_official_success")
	relaxed_success = _metric(item, "relaxed_success_stable", "relaxed_terminal_process_success")
	relaxed_process_success = _metric(item, "relaxed_process_success_terminal", "relaxed_process_success")
	strict_success = _metric(item, "strict_success_stable", "terminal_process_success", "success")
	process_success = _metric(item, "process_success_terminal", "process_success")
	angle_error = _metric(item, "angle_error", "orientation_error", "yaw_error")
	return {
		"task_id": item.get("task_id"),
		"assembly_id": item.get("assembly_id"),
		"task_name": item.get("task_name"),
		"episodes": item.get("episodes"),
		"official_success_latched": official_latched,
		"official_success_terminal": official_terminal,
		"relaxed_success": relaxed_success,
		"relaxed_process_success": relaxed_process_success,
		"strict_success": strict_success,
		"process_success": process_success,
		"mean_depth_fraction": _metric(item, "depth_fraction"),
		"mean_lateral_error_mm": _metric(item, "lateral_error") * 1000.0,
		"mean_angle_error_deg": math.degrees(angle_error),
		"mean_keypoint_error_mm": _metric(item, "keypoint_error") * 1000.0,
		"episode_len_mean": item.get("episode_length"),
		"official_relaxed_gap": official_latched - relaxed_success,
		"relaxed_strict_gap": relaxed_success - strict_success,
		"official_strict_gap": official_latched - strict_success,
		"episode_reward": item.get("episode_reward"),
		"success_count": item.get("success_count"),
		"failure_count": item.get("failure_count"),
	}


def _evaluate_one(cfg, entry: dict, output_dir: Path):
	assembly_id = _normalize_assembly_id(entry["assembly_id"])
	task_id = int(entry["task_id"])
	cfg.assembly_id = assembly_id
	cfg.eval_task_id = task_id
	cfg = apply_eval_task_template(cfg, entry)
	set_seed(cfg.seed)
	torch.cuda.set_device(cfg.device_id)
	task_output_dir = output_dir / assembly_id
	result_fp = task_output_dir / "eval_metrics.json"
	if result_fp.exists() and not cfg.batch_eval_overwrite:
		print(colored(f"Reusing existing eval metrics: {result_fp}", "blue", attrs=["bold"]))
		return _read_json(result_fp)

	env = make_env(cfg)
	try:
		expected_obs_dim = cfg.get('collect_expected_obs_dim', None)
		if expected_obs_dim is not None:
			actual_obs_dim = int(env.observation_space.shape[-1])
			if actual_obs_dim < int(expected_obs_dim):
				raise RuntimeError(
					f"Environment obs_dim={actual_obs_dim} is smaller than checkpoint obs_dim={expected_obs_dim}."
				)
			if actual_obs_dim > int(expected_obs_dim):
				print(colored(
					f"Adapting environment obs_dim={actual_obs_dim} to checkpoint obs_dim={int(expected_obs_dim)}.",
					"yellow",
					attrs=["bold"],
				))
		agent = _make_agent(cfg)
		target_episodes = int(cfg.get('batch_eval_episodes_per_task', cfg.get('eval_trials', 100)) or 100)
		use_mpc = cfg.mpc if cfg.get('batch_eval_mpc', None) is None else bool(cfg.batch_eval_mpc)
		rollout_device = torch.device(f"cuda:{cfg.device_id}")
		if str(cfg.get('task_conditioning', '')).lower() == 'axial_params':
			tasks = torch.zeros(cfg.num_envs, dtype=torch.long, device=rollout_device)
		else:
			tasks = torch.full((cfg.num_envs,), task_id, dtype=torch.long, device=rollout_device)
		episode_return = torch.zeros(cfg.num_envs, dtype=torch.float32, device=rollout_device)
		episode_len = torch.zeros(cfg.num_envs, dtype=torch.int64, device=rollout_device)
		returns = []
		lengths = []
		successes = []
		success_diagnostics = {key: [] for key in SUCCESS_DIAGNOSTIC_KEYS}
		completed = 0
		env_steps = 0
		start_time = monotonic()
		last_log = 0.0
		guard_steps = cfg.get('batch_eval_max_env_steps', None)
		if guard_steps is None:
			waves = math.ceil(target_episodes / max(1, int(cfg.num_envs)))
			guard_steps = int((waves + 2) * max(1, int(cfg.episode_length)) * max(1.0, float(cfg.eval_hang_guard_factor)))

		obs, _ = env.reset()
		obs = _adapt_obs_to_checkpoint(obs, expected_obs_dim)
		print(colored(
			f"Evaluating checkpoint={cfg.checkpoint} assembly_id={assembly_id} "
			f"task_id={task_id} episodes={target_episodes} mpc={use_mpc}.",
			"cyan",
			attrs=["bold"],
		))
		with torch.no_grad():
			while completed < target_episodes:
				t0 = episode_len == 0
				torch.compiler.cudagraph_mark_step_begin()
				model_tasks = _model_task_input(cfg, env, tasks)
				action, _ = agent(
					obs,
					t0=t0,
					step=1 if use_mpc else 0,
					eval_mode=True,
					task=model_tasks,
					mpc=use_mpc,
				)
				raw_obs, reward, terminated, truncated, info = env.step(action)
				done = terminated | truncated
				obs = _adapt_obs_to_checkpoint(raw_obs, expected_obs_dim)
				next_return = episode_return + reward
				next_len = episode_len + 1
				final_info = info.get('final_info', {}) if isinstance(info, dict) else {}
				success_tensor = final_info.get('success', None)
				for env_index in range(cfg.num_envs):
					if not bool(done[env_index].item()):
						continue
					if completed >= target_episodes:
						break
					success = 0.0
					if success_tensor is not None:
						success = float(torch.nan_to_num(success_tensor[env_index], nan=0.0).detach().item())
					returns.append(float(next_return[env_index].detach().item()))
					lengths.append(int(next_len[env_index].detach().item()))
					successes.append(success)
					for metric_key in SUCCESS_DIAGNOSTIC_KEYS:
						if metric_key in final_info:
							value = torch.nan_to_num(final_info[metric_key][env_index], nan=0.0)
							success_diagnostics[metric_key].append(float(value.detach().item()))
					completed += 1
				episode_return = torch.where(done, torch.zeros_like(next_return), next_return)
				episode_len = torch.where(done, torch.zeros_like(next_len), next_len)
				env_steps += 1

				now = monotonic()
				if now - last_log >= float(cfg.progress_log_interval_sec) or completed >= target_episodes:
					last_log = now
					elapsed = int(now - start_time)
					print(colored(
						f"eval progress assembly_id={assembly_id} task_id={task_id} "
						f"episodes={completed}/{target_episodes} env_steps={env_steps} elapsed={elapsed}s",
						"cyan",
						attrs=["bold"],
					), flush=True)
				if env_steps > guard_steps:
					raise RuntimeError(
						f"Evaluation did not finish within guard_steps={guard_steps} for assembly_id={assembly_id}."
					)

		metrics = {
			"assembly_id": assembly_id,
			"task_id": task_id,
			"eval_index": int(entry.get("eval_index", task_id)),
			"task_name": entry.get("task_name", f"{cfg.task}-{assembly_id}"),
			"checkpoint": str(Path(cfg.checkpoint).expanduser().resolve()),
			"episodes": len(returns),
			"episode_reward": _mean(returns),
			"episode_length": _mean(lengths),
			"episode_success": _mean(successes),
			"success_count": int(sum(1 for value in successes if value > 0.5)),
			"failure_count": int(sum(1 for value in successes if value <= 0.5)),
			"mpc": bool(use_mpc),
			"env_steps": int(env_steps),
		}
		for metric_key, values in success_diagnostics.items():
			if values:
				metrics[f"episode_{metric_key}"] = _mean(values)
		_write_json(result_fp, metrics)
		print(colored(
			f"Saved eval metrics for assembly_id={assembly_id}: success={metrics['episode_success']:.4f} "
			f"reward={metrics['episode_reward']:.3f}",
			"green",
			attrs=["bold"],
		))
		return metrics
	finally:
		env.close()


def _write_summary(summary_fp: Path, results: list[dict]):
	summary_fp.parent.mkdir(parents=True, exist_ok=True)
	ordered = sorted(results, key=lambda item: int(item.get("eval_index", item.get("task_id", 0))))
	rows = [_summary_row(item) for item in ordered]
	summary = {
		"num_tasks": len(ordered),
		"episode_success": _mean([item["episode_success"] for item in ordered]),
		"episode_reward": _mean([item["episode_reward"] for item in ordered]),
		"episode_length": _mean([item["episode_length"] for item in ordered]),
		"relaxed_success": _mean([row["relaxed_success"] for row in rows]),
		"relaxed_process_success": _mean([row["relaxed_process_success"] for row in rows]),
		"strict_success": _mean([row["strict_success"] for row in rows]),
		"process_success": _mean([row["process_success"] for row in rows]),
		"official_success_latched": _mean([row["official_success_latched"] for row in rows]),
		"official_relaxed_gap": _mean([row["official_relaxed_gap"] for row in rows]),
		"relaxed_strict_gap": _mean([row["relaxed_strict_gap"] for row in rows]),
		"official_strict_gap": _mean([row["official_strict_gap"] for row in rows]),
		"tasks": ordered,
		"csv_rows": rows,
	}
	for metric_key in SUCCESS_DIAGNOSTIC_KEYS:
		values = [item[f"episode_{metric_key}"] for item in ordered if f"episode_{metric_key}" in item]
		if values:
			summary[f"episode_{metric_key}"] = _mean(values)
	_write_json(summary_fp, summary)
	csv_fp = summary_fp.with_suffix(".csv")
	with open(csv_fp, "w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(
			f,
			fieldnames=[
				"task_id",
				"assembly_id",
				"task_name",
				"episodes",
				"official_success_latched",
				"official_success_terminal",
				"relaxed_success",
				"relaxed_process_success",
				"strict_success",
				"process_success",
				"mean_depth_fraction",
				"mean_lateral_error_mm",
				"mean_angle_error_deg",
				"mean_keypoint_error_mm",
				"episode_len_mean",
				"official_relaxed_gap",
				"relaxed_strict_gap",
				"official_strict_gap",
				"episode_reward",
				"success_count",
				"failure_count",
			],
		)
		writer.writeheader()
		for row in rows:
			writer.writerow({key: row.get(key) for key in writer.fieldnames})
	return summary, csv_fp


@hydra.main(version_base=None, config_name="config")
def launch(cfg: Config):
	assert torch.cuda.is_available()
	if not cfg.checkpoint:
		raise ValueError("`checkpoint` must point to the offline-RL checkpoint to evaluate.")
	checkpoint_fp = Path(hydra.utils.to_absolute_path(str(cfg.checkpoint))).expanduser().resolve()
	if not checkpoint_fp.exists():
		raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_fp}")
	cfg.checkpoint = str(checkpoint_fp)
	if cfg.get('offline_manifest_fp', None):
		cfg.offline_manifest_fp = str(
			Path(hydra.utils.to_absolute_path(str(cfg.offline_manifest_fp))).expanduser().resolve()
		)
	_apply_checkpoint_compat(cfg, checkpoint_fp)
	cfg = parse_cfg(cfg)
	cfg.enable_wandb = False
	cfg.save_agent = False
	cfg.multiproc = False
	cfg.device_id = int(cfg.gpu_id)
	if cfg.get('batch_eval_worker_assembly_id', None) is not None:
		if cfg.get('batch_eval_worker_task_id', None) is None:
			raise ValueError("`batch_eval_worker_task_id` is required when `batch_eval_worker_assembly_id` is set.")
		worker_task_id = int(cfg.batch_eval_worker_task_id)
		worker_assembly_id = _normalize_assembly_id(cfg.batch_eval_worker_assembly_id)
		worker_eval_index = cfg.get('batch_eval_worker_eval_index', worker_task_id)
		entries = []
		if cfg.get('offline_manifest_fp', None):
			for entry in load_offline_manifest(cfg.offline_manifest_fp):
				entry_assembly_id = _normalize_assembly_id(entry.get("assembly_id", ""))
				if int(entry["task_id"]) == worker_task_id and entry_assembly_id == worker_assembly_id:
					entry = dict(entry)
					entry["assembly_id"] = entry_assembly_id
					entries = [entry]
					break
		if not entries:
			entries = [{
				"assembly_id": worker_assembly_id,
				"task_id": worker_task_id,
				"task_name": f"{cfg.task}-{worker_assembly_id}",
			}]
		entries[0]["eval_index"] = int(worker_eval_index)
	else:
		entries = _resolve_eval_entries(cfg)
	output_dir = _resolve_output_dir(cfg)
	summary_fp = _resolve_summary_fp(cfg, output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)
	print(colored(f'Batch eval output dir: {output_dir}', 'yellow', attrs=['bold']))
	print(colored(f'Evaluating {len(entries)} task(s).', 'yellow', attrs=['bold']))

	if cfg.get('batch_eval_worker_assembly_id', None) is not None:
		_evaluate_one(cfg, entries[0], output_dir)
		return

	if len(entries) > 1 and cfg.get('batch_eval_spawn_per_assembly', True):
		results = _run_subprocess_eval(cfg, entries, output_dir)
	else:
		results = [_evaluate_one(cfg, entry, output_dir) for entry in entries]
	summary, csv_fp = _write_summary(summary_fp, results)
	print(colored(f"Saved batch eval summary: {summary_fp}", "green", attrs=["bold"]))
	print(colored(f"Saved batch eval CSV: {csv_fp}", "green", attrs=["bold"]))
	print(colored(
		f"Average success={summary['episode_success']:.4f}, reward={summary['episode_reward']:.3f} "
		f"over {summary['num_tasks']} tasks.",
		"green",
		attrs=["bold"],
	))


if __name__ == '__main__':
	launch()
