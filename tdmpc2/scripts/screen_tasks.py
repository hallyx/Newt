import csv
import json
import subprocess
import sys
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from termcolor import colored

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Config, parse_cfg  # noqa: E402


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


def _normalize_assembly_id(value) -> str:
	value = str(value).strip().strip("'\"")
	if value.isdigit() and len(value) < 5:
		value = value.zfill(5)
	return value


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


def _override_value(value):
	if isinstance(value, bool):
		return "true" if value else "false"
	if value is None:
		return "null"
	if isinstance(value, (list, tuple)):
		return "[" + ",".join(str(item) for item in value) + "]"
	if isinstance(value, str) and ("," in value or ";" in value):
		return json.dumps(value)
	return str(value)


def _resolve_output(path_value, default_path: Path) -> Path:
	if path_value:
		path = Path(path_value).expanduser()
		if not path.is_absolute():
			path = Path(hydra.utils.get_original_cwd()) / path
		return path
	return default_path


def _decision(row, cfg):
	strict = float(row.get("strict_success", 0.0) or 0.0)
	process = float(row.get("process_success", strict) or strict)
	success = min(strict, process) if process > 0.0 else strict
	depth = float(row.get("mean_depth_fraction", 0.0) or 0.0)
	official = float(row.get("official_success_latched", 0.0) or 0.0)
	if success < float(cfg.get("screen_defer_success", 0.03)) and depth < float(cfg.get("screen_low_depth_threshold", 0.35)):
		return "defer_online_boost"
	if official >= 0.70 and strict < 0.20:
		return "official_gap_target"
	if float(cfg.get("screen_hard_min_success", 0.15)) <= success <= float(cfg.get("screen_hard_max_success", 0.45)):
		return "hard_target"
	if float(cfg.get("screen_extra_min_success", 0.05)) <= success < float(cfg.get("screen_extra_max_success", 0.15)) and depth >= float(cfg.get("screen_high_depth_threshold", 0.75)):
		return "hard_target_extra_episodes"
	if strict > float(cfg.get("screen_easy_success", 0.70)):
		return "easy_anchor"
	if success < float(cfg.get("screen_extra_min_success", 0.05)):
		return "hard_target_extra_episodes" if depth >= float(cfg.get("screen_high_depth_threshold", 0.75)) else "defer_online_boost"
	return "hard_target"


def _screen_rows(summary, cfg):
	rows = list(summary.get("csv_rows", []))
	for row in rows:
		row["screen_decision"] = _decision(row, cfg)
	if rows and not any(row["screen_decision"] == "easy_anchor" for row in rows):
		best = max(rows, key=lambda item: float(item.get("strict_success", 0.0) or 0.0))
		best["relative_easy_anchor"] = True
	return rows


@hydra.main(version_base=None, config_name="config")
def launch(cfg: Config):
	if not cfg.checkpoint:
		raise ValueError("`checkpoint` is required for zero-shot screening.")
	if cfg.get("isaaclab_backend", "auto") == "auto":
		cfg.isaaclab_backend = "srsa"
	if cfg.task == "soup":
		cfg.task = "isaaclab-srsa-assembly"
	cfg.eval_success_metric = cfg.get("eval_success_metric", "strict") or "strict"
	cfg.srsa_eval_success_metric = cfg.eval_success_metric
	cfg.enable_wandb = False
	cfg.save_agent = False
	cfg.compile = False
	cfg = parse_cfg(cfg)

	assembly_ids = _parse_assembly_ids(cfg.get("screen_assembly_ids", None))
	if not assembly_ids:
		assembly_ids = _parse_assembly_ids(cfg.get("eval_assembly_ids", None))
	if not assembly_ids:
		assembly_ids = ["00004", "00014", "00062", "00271"]

	output_csv = _resolve_output(cfg.get("screen_output_csv", None), Path(hydra.utils.get_original_cwd()) / "data" / "task_screening_01125_axial_hole.csv")
	output_json = _resolve_output(cfg.get("screen_output_json", None), output_csv.with_suffix(".json"))
	eval_summary_fp = output_json.with_suffix(".batch_eval_summary.json")
	eval_output_dir = output_json.with_suffix("").parent / (output_json.with_suffix("").name + "_batch_eval")

	batch_eval_script = Path(__file__).resolve().parents[1] / "batch_eval_tasks.py"
	cmd = [
		sys.executable,
		str(batch_eval_script),
		f"checkpoint={Path(hydra.utils.to_absolute_path(str(cfg.checkpoint))).expanduser().resolve()}",
		f"eval_assembly_ids=[{','.join(assembly_ids)}]",
		f"batch_eval_episodes_per_task={int(cfg.get('screen_trials', 200))}",
		f"batch_eval_summary_fp={eval_summary_fp}",
		f"batch_eval_output_dir={eval_output_dir}",
		"batch_eval_overwrite=true",
		"enable_wandb=false",
		"compile=false",
	]
	for field in (
		"isaaclab_backend",
		"task",
		"srsa_dir",
		"srsa_sparse_reward",
		"srsa_sil",
		"srsa_if_sbc",
		"num_envs",
		"gpu_id",
		"model_size",
		"horizon",
		"mpc",
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
		"srsa_enable_flange_force_sensor",
		"isaaclab_canonical_append_force",
		"isaaclab_canonical_append_task_params",
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
		"eval_success_metric",
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
		"progress_log_interval_sec",
	):
		value = cfg.get(field, None)
		if value is not None:
			cmd.append(f"{field}={_override_value(value)}")

	print(colored(f"Running zero-shot screening for assembly ids: {assembly_ids}", "cyan", attrs=["bold"]))
	subprocess.run(cmd, cwd=hydra.utils.get_original_cwd(), check=True)
	with open(eval_summary_fp, "r", encoding="utf-8") as f:
		summary = json.load(f)
	rows = _screen_rows(summary, cfg)
	output_csv.parent.mkdir(parents=True, exist_ok=True)
	with open(output_csv, "w", encoding="utf-8", newline="") as f:
		fieldnames = [
			"assembly_id",
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
			"screen_decision",
			"relative_easy_anchor",
		]
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		for row in rows:
			writer.writerow({key: row.get(key, False if key == "relative_easy_anchor" else None) for key in fieldnames})
	with open(output_json, "w", encoding="utf-8") as f:
		json.dump({"checkpoint": str(cfg.checkpoint), "tasks": rows}, f, indent=2, ensure_ascii=True)
	print(colored(f"Saved screening CSV: {output_csv}", "green", attrs=["bold"]))
	print(colored(f"Saved screening JSON: {output_json}", "green", attrs=["bold"]))


if __name__ == "__main__":
	launch()
