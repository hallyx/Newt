#!/usr/bin/env python3
"""Read-only closed-loop task-vector swap eval for Phase 2.2."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from _common import (
	DEFAULT_CONFIG,
	DEFAULT_OUTPUT_DIR,
	DEFAULT_TASK_A_REPLAY,
	DEFAULT_TASK_B_REPLAY,
	add_message,
	load_task_conditions,
	print_status,
	resolve,
	status_from_messages,
	tensor_to_list,
	write_json,
	write_text,
)


DEFAULT_REPAIR_CHECKPOINT = (
	"logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_taskctx_repair_01125_00256/"
	"20260708_taskctx_repair_phase2_stage-2_asm-00256/models/best_step-99840_s-0p9961.pt"
)
DEFAULT_PYTHON = "/home/gpuserver/miniconda3/envs/isaac51/bin/python"
DEFAULT_CONFIG_DIR = "configs/train"
DEFAULT_CONFIG_NAME = "srsa_01125_imitation_relaxed"
DEFAULT_RUN_ROOT = DEFAULT_OUTPUT_DIR / "closed_loop_task_swap_eval_phase2_runs"
DEFAULT_OUTPUT_JSON = DEFAULT_OUTPUT_DIR / "closed_loop_task_swap_eval_phase2.json"
DEFAULT_OUTPUT_MD = DEFAULT_OUTPUT_DIR / "closed_loop_task_swap_eval_phase2.md"

ASSEMBLIES = ("01125", "00256")


def _vec_arg(vec: torch.Tensor | list[float]) -> str:
	values = vec.tolist() if torch.is_tensor(vec) else list(vec)
	return "[" + ",".join(f"{float(value):.8g}" for value in values) + "]"


def _resolve_path(value: str | Path) -> Path:
	return resolve(value)


def _run_root(args) -> Path:
	path = Path(args.run_root).expanduser()
	if not path.is_absolute():
		path = DEFAULT_OUTPUT_DIR.parent.parent / path
	return path.resolve()


def _output_dir(args) -> Path:
	path = Path(args.output_dir).expanduser()
	if not path.is_absolute():
		path = DEFAULT_OUTPUT_DIR.parent.parent / path
	return path.resolve()


def _task_vecs(args) -> dict[str, list[float]]:
	conditions = load_task_conditions(args)
	return {key: tensor_to_list(value) for key, value in conditions.items()}


def _condition_plan(task_vecs: dict[str, list[float]]) -> list[dict[str, Any]]:
	return [
		{"name": "correct", "force": None, "description": "runtime current_task_vec"},
		{"name": "swapped", "force": "swapped", "description": "realistic opposite assembly task_vec"},
		{"name": "zero", "force": "zero", "description": "zero task_vec"},
		{"name": "random", "force": "random", "description": "deterministic random task_vec"},
	]


def _force_vec_for(assembly_id: str, condition: str, task_vecs: dict[str, list[float]]) -> list[float] | None:
	if condition == "correct":
		return None
	if condition == "swapped":
		other = "00256" if assembly_id == "01125" else "01125"
		return task_vecs[other]
	if condition == "zero":
		return task_vecs["zero"]
	if condition == "random":
		return task_vecs["random"]
	raise ValueError(f"Unknown condition: {condition}")


def _base_eval_overrides(args, *, assembly_id: str, label: str, output_dir: Path, summary_fp: Path) -> list[str]:
	repo_root = Path(__file__).resolve().parents[3]
	return [
		f"--config-dir={args.config_dir}",
		f"--config-name={args.config_name}",
		f"checkpoint={_resolve_path(args.checkpoint)}",
		f"eval_assembly_ids=[{assembly_id}]",
		"isaaclab_backend=srsa",
		"task=isaaclab-srsa-assembly",
		f"isaaclab_dir={args.isaaclab_dir}",
		f"srsa_dir={args.srsa_dir}",
		f"srsa_task_template_fp={repo_root / 'data' / 'srsa_axial_task_templates.json'}",
		f"srsa_mesh_geometry_fp={repo_root / 'data' / 'srsa_mesh_geometry_params.csv'}",
		"srsa_param_template_id=2",
		"eval_task_template_exact=true",
		"srsa_axial_reference_anchor_assembly_id=01125",
		"srsa_axial_reference_anchor_task_type_id=0",
		"srsa_axial_recompute_manifest_task_vecs=true",
		"srsa_axial_clearance_depth_templates='1.0:1.0'",
		f"num_envs={int(args.num_envs)}",
		f"gpu_id={int(args.gpu_id)}",
		"model_size=S",
		"horizon=3",
		"compile=false",
		"mpc=true",
		"isaaclab_headless=true",
		"isaaclab_use_canonical_obs=true",
		"srsa_task_family_name=normal_fit",
		"srsa_task_param_obs=false",
		"srsa_task_param_obs_mode=task_vec",
		"srsa_enable_axial_task_param_sampler=true",
		"srsa_axial_fixed_plug_scale=true",
		"srsa_axial_clearance_base=0.000114",
		"srsa_axial_clearance_jitter_ratio=0.10",
		"srsa_axial_depth_base=0.015",
		"srsa_axial_depth_jitter_ratio=0.10",
		'srsa_axial_init_error_xy_range="0.009,0.0010"',
		'srsa_axial_init_error_z_range="0.0010,0.0020"',
		'srsa_axial_init_error_yaw_range="-0.0872665,0.0872665"',
		'srsa_axial_visual_noise_xy_range="0.0,0.0"',
		'srsa_axial_visual_noise_z_range="0.0,0.0"',
		"srsa_enable_flange_force_sensor=true",
		"isaaclab_canonical_append_force=true",
		"isaaclab_canonical_append_task_params=false",
		"srsa_vision_noise_xy_std=0.0",
		"srsa_vision_noise_xy_jitter_std=0.0",
		"srsa_vision_noise_z_std=0.0",
		"srsa_vision_noise_z_jitter_std=0.0",
		"isaaclab_canonical_use_visual_noise=false",
		"task_conditioning=axial_params",
		"contact_history_enabled=true",
		"contact_history_len=4",
		"contact_context_dim=64",
		"contact_history_hidden_dim=128",
		"contact_history_layers=2",
		"contact_force_dim=6",
		"contact_action_dim=3",
		"contact_ee_delta_dim=3",
		"contact_history_use_ee_delta=true",
		"task_context_adapter_enabled=true",
		"task_context_adapter_hidden_dim=128",
		"task_context_adapter_alpha=0.01",
		"task_context_adapter_source=raw_task_vec",
		"task_context_adapter_apply_encoder=true",
		"task_context_adapter_apply_dynamics=true",
		"task_context_adapter_apply_policy=false",
		"task_context_adapter_apply_reward=false",
		"task_context_adapter_apply_q=false",
		"task_context_adapter_lr_scale=0.1",
		"task_context_repair_enabled=true",
		"task_recon_coef=0.1",
		"task_spread_coef=0.01",
		"task_raw_residual_scale=0.1",
		"task_spread_near_threshold=0.3",
		"task_spread_far_threshold=1.0",
		"task_spread_margin=0.5",
		"eval_success_metric=relaxed",
		"srsa_eval_success_metric=relaxed",
		f"batch_eval_episodes_per_task={int(args.episodes)}",
		"batch_eval_spawn_per_assembly=false",
		"batch_eval_overwrite=true",
		f"batch_eval_output_dir={output_dir}",
		f"batch_eval_summary_fp={summary_fp}",
		"enable_wandb=false",
		"exp_name=srsa_phase2_closed_loop_task_swap_eval",
		f"run_id=phase2_closed_loop_{assembly_id}_{label}",
		f"seed={int(args.seed)}",
		"progress_log_interval_sec=30",
	]


def _command_for(args, *, assembly_id: str, condition: str, force_vec: list[float] | None, output_dir: Path) -> list[str]:
	summary_fp = output_dir / "batch_eval_summary.json"
	cmd = [
		str(_resolve_path(args.python)),
		"tdmpc2/batch_eval_tasks.py",
		*_base_eval_overrides(args, assembly_id=assembly_id, label=condition, output_dir=output_dir, summary_fp=summary_fp),
	]
	if force_vec is not None:
		cmd.extend([
			f"batch_eval_force_task_vec_label={assembly_id}_{condition}",
			f"batch_eval_force_task_vec_6={_vec_arg(force_vec)}",
		])
	return cmd


def _read_json(path: Path) -> dict[str, Any]:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def _metric(item: dict[str, Any], *keys: str, default: float | None = None) -> float | None:
	for key in keys:
		if key in item and item[key] is not None:
			return float(item[key])
		episode_key = f"episode_{key}"
		if episode_key in item and item[episode_key] is not None:
			return float(item[episode_key])
	return default


def _extract_metrics(summary: dict[str, Any], *, max_force_reason: str) -> dict[str, Any]:
	tasks = summary.get("tasks") or []
	if not tasks:
		raise RuntimeError("batch_eval_summary.json has no tasks entries.")
	task = tasks[0]
	force_value = _metric(
		task,
		"max_force",
		"flange_force_norm",
		"flange_force_max",
		"contact_force_max",
	)
	return {
		"assembly_id": task.get("assembly_id"),
		"episodes": task.get("episodes"),
		"relaxed_success": _metric(task, "relaxed_success_stable", "relaxed_success_episode", "success", default=0.0),
		"strict_success": _metric(task, "strict_success_stable", "strict_success_episode", default=0.0),
		"process_success": _metric(task, "process_success_terminal", "process_success", default=0.0),
		"lateral_error": _metric(task, "lateral_error", default=0.0),
		"lateral_error_mm": (_metric(task, "lateral_error", default=0.0) or 0.0) * 1000.0,
		"keypoint_error": _metric(task, "keypoint_error", default=0.0),
		"keypoint_error_mm": (_metric(task, "keypoint_error", default=0.0) or 0.0) * 1000.0,
		"reward": float(task.get("episode_reward", 0.0)),
		"max_force": force_value if force_value is not None else {
			"status": "UNKNOWN_WITH_REASON",
			"reason": max_force_reason,
		},
		"episode_length": float(task.get("episode_length", 0.0)),
		"jamming_rate": _metric(task, "jam", default=0.0),
		"failure_count": task.get("failure_count"),
		"raw_metrics": task,
	}


def _failure_modes(metrics: dict[str, Any], correct: dict[str, Any] | None = None) -> list[str]:
	modes: list[str] = []
	if float(metrics.get("relaxed_success") or 0.0) < 0.8:
		modes.append("relaxed_success_drop")
	if float(metrics.get("strict_success") or 0.0) < 0.2:
		modes.append("strict_success_weak")
	if float(metrics.get("process_success") or 0.0) < 0.2:
		modes.append("process_success_weak")
	if float(metrics.get("jamming_rate") or 0.0) > 0.0:
		modes.append("jam")
	if float(metrics.get("lateral_error_mm") or 0.0) > 1.0:
		modes.append("lateral_error_high")
	if float(metrics.get("keypoint_error_mm") or 0.0) > 2.5:
		modes.append("keypoint_error_high")
	if correct is not None:
		if float(metrics.get("reward") or 0.0) < float(correct.get("reward") or 0.0) - 5.0:
			modes.append("reward_drop_vs_correct")
		if float(metrics.get("strict_success") or 0.0) < float(correct.get("strict_success") or 0.0) - 0.05:
			modes.append("strict_drop_vs_correct")
		if float(metrics.get("process_success") or 0.0) < float(correct.get("process_success") or 0.0) - 0.05:
			modes.append("process_drop_vs_correct")
		if float(metrics.get("keypoint_error_mm") or 0.0) > float(correct.get("keypoint_error_mm") or 0.0) + 0.25:
			modes.append("keypoint_worse_vs_correct")
		if float(metrics.get("lateral_error_mm") or 0.0) > float(correct.get("lateral_error_mm") or 0.0) + 0.10:
			modes.append("lateral_worse_vs_correct")
	return modes


def _run_condition(args, *, assembly_id: str, condition: str, force_vec: list[float] | None, run_root: Path) -> dict[str, Any]:
	output_dir = run_root / assembly_id / condition
	summary_fp = output_dir / "batch_eval_summary.json"
	cmd = _command_for(args, assembly_id=assembly_id, condition=condition, force_vec=force_vec, output_dir=output_dir)
	if args.dry_run:
		print("[dry-run] " + " ".join(cmd))
		return {
			"assembly_id": assembly_id,
			"condition": condition,
			"output_dir": str(output_dir),
			"summary_fp": str(summary_fp),
			"command": cmd,
			"status": "DRY_RUN",
		}
	if args.reuse_existing and summary_fp.exists():
		print(f"Reusing existing eval summary: {summary_fp}", flush=True)
	else:
		output_dir.mkdir(parents=True, exist_ok=True)
		log_fp = output_dir / "batch_eval.log"
		print(f"[closed-loop-swap] {assembly_id}/{condition} -> {output_dir}", flush=True)
		start = time.monotonic()
		with open(log_fp, "w", encoding="utf-8") as log_f:
			proc = subprocess.run(
				cmd,
				cwd=str(Path(__file__).resolve().parents[3]),
				stdout=log_f,
				stderr=subprocess.STDOUT,
				text=True,
				check=False,
				timeout=float(args.subprocess_timeout),
			)
		elapsed = time.monotonic() - start
		if proc.returncode != 0:
			return {
				"assembly_id": assembly_id,
				"condition": condition,
				"output_dir": str(output_dir),
				"log_fp": str(log_fp),
				"command": cmd,
				"status": "FAILED",
				"returncode": proc.returncode,
				"elapsed_sec": elapsed,
			}
	if not summary_fp.exists():
		return {
			"assembly_id": assembly_id,
			"condition": condition,
			"output_dir": str(output_dir),
			"summary_fp": str(summary_fp),
			"command": cmd,
			"status": "FAILED",
			"error": "summary file missing after eval",
		}
	summary = _read_json(summary_fp)
	metrics = _extract_metrics(
		summary,
		max_force_reason=(
			"batch_eval_tasks.py does not export flange_force_norm/max_force in eval_metrics.json; "
			"closed_loop_task_swap_eval.py keeps the field explicit instead of modifying the eval trunk."
		),
	)
	return {
		"assembly_id": assembly_id,
		"condition": condition,
		"output_dir": str(output_dir),
		"summary_fp": str(summary_fp),
		"status": "DONE",
		"forced_task_vec_6": force_vec,
		"metrics": metrics,
	}


def _delta_record(item: dict[str, Any], correct: dict[str, Any]) -> dict[str, float]:
	metrics = item.get("metrics") or {}
	base = correct.get("metrics") or {}
	keys = ("relaxed_success", "strict_success", "process_success", "reward", "lateral_error_mm", "keypoint_error_mm", "jamming_rate")
	return {key: float(metrics.get(key) or 0.0) - float(base.get(key) or 0.0) for key in keys}


def _build_interpretation(results: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
	correct_ok = True
	zero_random_effects = []
	swap_effects = []
	for assembly_id, conds in results.items():
		correct = conds.get("correct", {})
		correct_metrics = correct.get("metrics") or {}
		if float(correct_metrics.get("relaxed_success") or 0.0) < 0.8:
			correct_ok = False
		for condition in ("swapped", "zero", "random"):
			item = conds.get(condition, {})
			if item.get("status") != "DONE":
				continue
			delta = _delta_record(item, correct)
			item["delta_vs_correct"] = delta
			item["metrics"]["failure_modes"] = _failure_modes(item["metrics"], correct_metrics)
			effect = (
				abs(delta["reward"]) >= 5.0
				or abs(delta["strict_success"]) >= 0.05
				or abs(delta["process_success"]) >= 0.05
				or abs(delta["lateral_error_mm"]) >= 0.10
				or abs(delta["keypoint_error_mm"]) >= 0.25
				or abs(delta["jamming_rate"]) >= 0.05
			)
			record = {
				"assembly_id": assembly_id,
				"condition": condition,
				"effect": effect,
				"delta_vs_correct": delta,
			}
			if condition in {"zero", "random"}:
				zero_random_effects.append(record)
			else:
				swap_effects.append(record)
		if correct.get("status") == "DONE":
			correct["metrics"]["failure_modes"] = _failure_modes(correct["metrics"])
	zero_random_measurable = any(item["effect"] for item in zero_random_effects)
	swap_measurable = any(item["effect"] for item in swap_effects)
	relaxed_preserved_under_swap = all(
		abs(item["delta_vs_correct"].get("relaxed_success", 0.0)) <= 0.05
		for item in swap_effects
	) if swap_effects else None
	near_pair = bool(relaxed_preserved_under_swap)
	final_acceptance = "PASS_WITH_CAVEAT" if correct_ok and zero_random_measurable else "WARNING"
	if not correct_ok:
		final_acceptance = "FAIL"
	return {
		"correct_relaxed_retention_ok": correct_ok,
		"zero_random_measurable_effect": zero_random_measurable,
		"realistic_swap_measurable_effect": swap_measurable,
		"realistic_swap_relaxed_preserved": relaxed_preserved_under_swap,
		"near_pair_likely": near_pair,
		"phase2_final_acceptance": final_acceptance,
		"zero_random_effects": zero_random_effects,
		"swap_effects": swap_effects,
	}


def _markdown_table(results: dict[str, dict[str, dict[str, Any]]]) -> list[str]:
	lines = [
		"| Env | Condition | Relaxed | Strict | Process | Reward | Lateral mm | Keypoint mm | Jam | Episode len | Failure modes |",
		"| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
	]
	for assembly_id in ASSEMBLIES:
		for condition in ("correct", "swapped", "zero", "random"):
			item = results.get(assembly_id, {}).get(condition, {})
			metrics = item.get("metrics") or {}
			if item.get("status") != "DONE":
				lines.append(f"| `{assembly_id}` | `{condition}` | NA | NA | NA | NA | NA | NA | NA | NA | `{item.get('status')}` |")
				continue
			modes = ", ".join(metrics.get("failure_modes") or []) or "none"
			lines.append(
				"| "
				f"`{assembly_id}` | `{condition}` | "
				f"{float(metrics.get('relaxed_success') or 0.0):.3f} | "
				f"{float(metrics.get('strict_success') or 0.0):.3f} | "
				f"{float(metrics.get('process_success') or 0.0):.3f} | "
				f"{float(metrics.get('reward') or 0.0):.2f} | "
				f"{float(metrics.get('lateral_error_mm') or 0.0):.3f} | "
				f"{float(metrics.get('keypoint_error_mm') or 0.0):.3f} | "
				f"{float(metrics.get('jamming_rate') or 0.0):.3f} | "
				f"{float(metrics.get('episode_length') or 0.0):.1f} | "
				f"{modes} |"
			)
	return lines


def _build_markdown(report: dict[str, Any]) -> str:
	interp = report["interpretation"]
	status = report["status"]
	answers = [
		f"1. closed-loop 中 task_vec 是否影响控制质量：`{bool(interp['zero_random_measurable_effect'] or interp['realistic_swap_measurable_effect'])}`。",
		"2. wrong/zero/random task_vec 是否主要影响 relaxed 还是 strict/process："
		"`strict/process/keypoint/lateral/reward` 优先；relaxed 对 near pair 可能保持不变。",
		f"3. 01125/00256 是否更像 near pair：`{bool(interp['near_pair_likely'])}`。",
		f"4. Phase 2 是否可以最终验收：`{interp['phase2_final_acceptance']}`。",
		"5. 下一步建议：如果 zero/random 已有闭环影响且 correct retention 保持，优先进入 easy third task；"
		"如果 zero/random 仍几乎无差异，再考虑 counterfactual reward。",
	]
	lines = [
		"# SRSA Phase 2.2 Closed-Loop Task-Vector Swap Eval",
		"",
		"本报告只做闭环 task-vector swap 评估；未修改模型、训练流程、sampler、reward、Q、policy 或 MPPI。",
		"",
		f"Status: `{status}`",
		"",
		"## 结论回答",
		"",
		*answers,
		"",
		"## 指标表",
		"",
		*_markdown_table(report["results"]),
		"",
		"## 解释",
		"",
		f"- correct task_vec relaxed/retention gate：`{interp['correct_relaxed_retention_ok']}`。",
		f"- zero/random 是否有可测闭环影响：`{interp['zero_random_measurable_effect']}`。",
		f"- realistic swap 是否有可测闭环影响：`{interp['realistic_swap_measurable_effect']}`。",
		f"- realistic swap relaxed 是否保持：`{interp['realistic_swap_relaxed_preserved']}`。",
		"- `max_force` 字段当前不由 `batch_eval_tasks.py` 导出；JSON 中保留 `UNKNOWN_WITH_REASON`，避免改 eval 主干。",
		"",
		"## 输入",
		"",
		f"- checkpoint: `{report['checkpoint']}`",
		f"- run_root: `{report['run_root']}`",
		f"- episodes_per_condition: `{report['episodes_per_condition']}`",
		f"- num_envs: `{report['num_envs']}`",
	]
	return "\n".join(lines) + "\n"


def build_report(args) -> dict[str, Any]:
	messages: list[dict[str, Any]] = []
	checkpoint = _resolve_path(args.checkpoint)
	if not checkpoint.exists() and not args.dry_run:
		add_message(messages, "FAIL", f"Checkpoint not found: {checkpoint}")
		return {"status": "FAIL", "messages": messages}
	task_vecs = _task_vecs(args)
	run_root = _run_root(args)
	results: dict[str, dict[str, dict[str, Any]]] = {assembly_id: {} for assembly_id in ASSEMBLIES}
	for assembly_id in ASSEMBLIES:
		for condition in _condition_plan(task_vecs):
			name = condition["name"]
			force_vec = _force_vec_for(assembly_id, name, task_vecs)
			result = _run_condition(args, assembly_id=assembly_id, condition=name, force_vec=force_vec, run_root=run_root)
			result["description"] = condition["description"]
			results[assembly_id][name] = result
			if result.get("status") == "FAILED":
				add_message(messages, "FAIL", f"Eval failed for assembly_id={assembly_id} condition={name}.", result=result)
	if args.dry_run:
		add_message(messages, "WARNING", "Dry-run requested; no Isaac eval launched.")
		interpretation = {
			"correct_relaxed_retention_ok": None,
			"zero_random_measurable_effect": None,
			"realistic_swap_measurable_effect": None,
			"realistic_swap_relaxed_preserved": None,
			"near_pair_likely": None,
			"phase2_final_acceptance": "DRY_RUN",
		}
	else:
		interpretation = _build_interpretation(results)
		if interpretation["phase2_final_acceptance"] == "PASS_WITH_CAVEAT":
			add_message(messages, "PASS", "Closed-loop swap eval completed; Phase 2 passes with strict/process caveat.")
		elif interpretation["phase2_final_acceptance"] == "FAIL":
			add_message(messages, "FAIL", "Correct task_vec relaxed/retention gate failed.")
		else:
			add_message(messages, "WARNING", "Closed-loop zero/random task_vec effect is weak or incomplete.")
	status = status_from_messages(messages)
	if status == "PASS" and interpretation.get("phase2_final_acceptance") == "PASS_WITH_CAVEAT":
		status = "PASS_WITH_CAVEAT"
	return {
		"status": status,
		"checkpoint": str(checkpoint),
		"run_root": str(run_root),
		"episodes_per_condition": int(args.episodes),
		"num_envs": int(args.num_envs),
		"task_vecs": task_vecs,
		"results": results,
		"interpretation": interpretation,
		"messages": messages,
	}


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--checkpoint", default=DEFAULT_REPAIR_CHECKPOINT)
	parser.add_argument("--config", default=DEFAULT_CONFIG)
	parser.add_argument("--python", default=DEFAULT_PYTHON)
	parser.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR)
	parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
	parser.add_argument("--isaaclab-dir", default="/home/gpuserver/IsaacLab")
	parser.add_argument("--srsa-dir", default="/home/gpuserver/hx/github/srsa")
	parser.add_argument("--task-a-label", default="01125")
	parser.add_argument("--task-b-label", default="00256")
	parser.add_argument("--task-a-replay", default=DEFAULT_TASK_A_REPLAY)
	parser.add_argument("--task-b-replay", default=DEFAULT_TASK_B_REPLAY)
	parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
	parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
	parser.add_argument("--episodes", type=int, default=20)
	parser.add_argument("--num-envs", type=int, default=256)
	parser.add_argument("--gpu-id", type=int, default=0)
	parser.add_argument("--seed", type=int, default=1)
	parser.add_argument("--subprocess-timeout", type=float, default=1800.0)
	parser.add_argument("--reuse-existing", action="store_true")
	parser.add_argument("--dry-run", action="store_true")
	parser.add_argument("--include-zero", action="store_true", default=True)
	parser.add_argument("--include-random", action="store_true", default=True)
	args = parser.parse_args()

	report = build_report(args)
	output_dir = _output_dir(args)
	json_fp = output_dir / "closed_loop_task_swap_eval_phase2.json"
	md_fp = output_dir / "closed_loop_task_swap_eval_phase2.md"
	write_json(report, json_fp, dry_run=args.dry_run)
	write_text(_build_markdown(report), md_fp, dry_run=args.dry_run)
	print_status(report["status"], report.get("messages", []))
	return 1 if str(report["status"]).upper() == "FAIL" else 0


if __name__ == "__main__":
	raise SystemExit(main())
