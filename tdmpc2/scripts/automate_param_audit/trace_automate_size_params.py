#!/usr/bin/env python3
"""Trace SRSA/AutoMate size parameters from task metadata into env-facing code paths."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from _common import (
	AXIAL_TASK_VEC_FIELDS,
	DEFAULT_OUTPUT_DIR,
	DEFAULT_TASK_HASH_CSV,
	add_message,
	compare_task_vectors,
	decode_task_vec,
	load_task_id_vectors,
	print_status,
	status_from_messages,
	write_json_report,
	write_text_report,
)


def _fmt(value: Any) -> str:
	if isinstance(value, float):
		return f"{value:.8g}"
	return str(value)


def _bool_text(value: bool | str) -> str:
	if isinstance(value, bool):
		return "yes" if value else "no"
	return str(value)


def _row(
	param_name,
	value_a,
	value_b,
	source,
	*,
	used_by_task_vec,
	used_by_geometry,
	used_by_collision,
	used_by_target_pose,
	used_by_reward,
	used_by_success_metric,
	risk_level,
	notes,
):
	return {
		"param_name": param_name,
		"value_01125": value_a,
		"value_00256": value_b,
		"source_file_or_attr": source,
		"used_by_task_vec": used_by_task_vec,
		"used_by_geometry": used_by_geometry,
		"used_by_collision": used_by_collision,
		"used_by_target_pose": used_by_target_pose,
		"used_by_reward": used_by_reward,
		"used_by_success_metric": used_by_success_metric,
		"risk_level": risk_level,
		"notes": notes,
	}


def build_rows(task_a: dict[str, Any], task_b: dict[str, Any]) -> list[dict[str, Any]]:
	vec_a = task_a["task_vec_6"]
	vec_b = task_b["task_vec_6"]
	dec_a = decode_task_vec(vec_a)
	dec_b = decode_task_vec(vec_b)
	rows = [
		_row(
			"assembly_id",
			task_a.get("assembly_id"),
			task_b.get("assembly_id"),
			"tdmpc2/envs/isaaclab.py:_configure_assembly_task",
			used_by_task_vec=False,
			used_by_geometry=True,
			used_by_collision="likely, via loaded assembly USD/assets",
			used_by_target_pose="likely, via assembly disassemble/eval files",
			used_by_reward=False,
			used_by_success_metric=False,
			risk_level="WARNING_RUNTIME_GEOMETRY_NOT_VERIFIED",
			notes="assembly_id is intentionally excluded from task_vec_6; it selects assembly_dir and asset paths.",
		),
		_row(
			"task_hash",
			task_a.get("task_hash"),
			task_b.get("task_hash"),
			"reports/task_consistency/task_id_to_hash.csv",
			used_by_task_vec=True,
			used_by_geometry=False,
			used_by_collision=False,
			used_by_target_pose=False,
			used_by_reward=False,
			used_by_success_metric=False,
			risk_level="PASS",
			notes="Phase 0.5 confirmed replay task hashes differ for 01125 and 00256.",
		),
	]
	for index, field in enumerate(AXIAL_TASK_VEC_FIELDS):
		geometry = field in ("log_scale", "clearance_abs_norm", "clearance_rel_norm", "depth_abs_norm", "task_type_id_float", "yaw_requirement_float")
		collision = field in ("log_scale", "clearance_abs_norm", "clearance_rel_norm")
		target = field in ("depth_abs_norm", "yaw_requirement_float")
		reward = field in ("clearance_abs_norm", "clearance_rel_norm", "depth_abs_norm", "yaw_requirement_float")
		success = reward
		source = "tdmpc2/config.py:make_axial_task_vec"
		notes = "Direct task_vec_6 component."
		risk = "PASS_STATIC_TRACE"
		if field == "log_scale":
			source = "tdmpc2/config.py:make_axial_task_vec + _apply_exact_axial_task_vec_to_sampler"
			notes = "Decoded into srsa_axial_scale_range when eval_task_template_exact is used."
			risk = "WARNING_RUNTIME_GEOMETRY_NOT_VERIFIED"
		elif field in ("clearance_abs_norm", "clearance_rel_norm"):
			source = "tdmpc2/config.py:make_axial_task_vec; tdmpc2/envs/isaaclab.py:_configure_srsa_runtime_env/_compute_srsa_eval_success"
			notes = "Clearance is exported through sampler/env vars and used in lateral/keypoint/jam thresholds if current_task_params expose radial_clearance/clearance."
			risk = "PASS_STATIC_OBJECTIVE_TRACE_WARNING_RUNTIME_GEOMETRY"
		elif field == "depth_abs_norm":
			source = "tdmpc2/config.py:make_axial_task_vec; tdmpc2/envs/isaaclab.py:_configure_srsa_runtime_env/_compute_srsa_eval_success"
			notes = "Depth is exported through sampler/env vars and used as target_depth/current_depth normalization in eval diagnostics."
			risk = "PASS_STATIC_OBJECTIVE_TRACE_WARNING_RUNTIME_GEOMETRY"
		elif field == "yaw_requirement_float":
			source = "tdmpc2/config.py:make_axial_task_vec; tdmpc2/envs/isaaclab.py:_compute_srsa_eval_success"
			notes = "Yaw requirement gates yaw_ok when task params or cfg require yaw."
			risk = "WARNING_RUNTIME_TASK_TYPE_DEPENDENCE_NOT_VERIFIED"
		rows.append(
			_row(
				field,
				vec_a[index],
				vec_b[index],
				source,
				used_by_task_vec=True,
				used_by_geometry=geometry,
				used_by_collision=collision,
				used_by_target_pose=target,
				used_by_reward=reward,
				used_by_success_metric=success,
				risk_level=risk,
				notes=notes,
			)
		)
	rows.extend([
		_row(
			"scale_ratio_decoded",
			dec_a["scale_ratio"],
			dec_b["scale_ratio"],
			"exp(task_vec_6[1]); tdmpc2/config.py:_apply_exact_axial_task_vec_to_sampler",
			used_by_task_vec=True,
			used_by_geometry=True,
			used_by_collision=True,
			used_by_target_pose=False,
			used_by_reward=False,
			used_by_success_metric=False,
			risk_level="WARNING_RUNTIME_GEOMETRY_NOT_VERIFIED",
			notes="Static decode only; actual prim scale/AABB must be checked with inspect_automate_env_geometry.py --launch-env.",
		),
		_row(
			"radial_clearance_m_decoded_with_default_reference",
			dec_a["radial_clearance_m_if_default_reference"],
			dec_b["radial_clearance_m_if_default_reference"],
			"task_vec_6[2] * axial_reference_radius(default 0.003993m)",
			used_by_task_vec=True,
			used_by_geometry=True,
			used_by_collision=True,
			used_by_target_pose=False,
			used_by_reward=True,
			used_by_success_metric=True,
			risk_level="PASS_STATIC_OBJECTIVE_TRACE_WARNING_RUNTIME_GEOMETRY",
			notes="Decoded with default reference radius; manifest-specific reference radius may differ.",
		),
		_row(
			"target_depth_m_decoded_with_default_reference",
			dec_a["target_depth_m_if_default_reference"],
			dec_b["target_depth_m_if_default_reference"],
			"task_vec_6[4] * axial_reference_depth(default 0.015m)",
			used_by_task_vec=True,
			used_by_geometry=False,
			used_by_collision=False,
			used_by_target_pose=True,
			used_by_reward=True,
			used_by_success_metric=True,
			risk_level="PASS_STATIC_OBJECTIVE_TRACE_WARNING_RUNTIME_TARGET",
			notes="Decoded with default reference depth; runtime target depth still needs env inspection.",
		),
	])
	return rows


def build_markdown(report: dict[str, Any]) -> str:
	lines = [
		"# AutoMate/SRSA Size Parameter Static Trace",
		"",
		f"Status: `{report['status']}`",
		"",
		"本报告只做静态代码路径梳理，不启动 Isaac env，不修改模型/训练/sampler/reward。",
		"",
		"## Task Vector Difference",
		"",
		"```json",
		report["task_vector_compare_json"],
		"```",
		"",
		"## Parameter Trace",
		"",
		"| param_name | value_01125 | value_00256 | task_vec | geometry | collision | target | reward | success | risk |",
		"| --- | ---: | ---: | --- | --- | --- | --- | --- | --- | --- |",
	]
	for row in report["parameter_trace"]:
		lines.append(
			"| "
			+ " | ".join([
				str(row["param_name"]),
				_fmt(row["value_01125"]),
				_fmt(row["value_00256"]),
				_bool_text(row["used_by_task_vec"]),
				_bool_text(row["used_by_geometry"]),
				_bool_text(row["used_by_collision"]),
				_bool_text(row["used_by_target_pose"]),
				_bool_text(row["used_by_reward"]),
				_bool_text(row["used_by_success_metric"]),
				str(row["risk_level"]),
			])
			+ " |"
		)
	lines.extend([
		"",
		"## Static Interpretation",
		"",
		"- `task_vec_6` 中 01125 和 00256 明确不同。",
		"- `clearance`/`depth` 不只是 label：静态代码显示它们会进入 sampler/env var，并进入 eval diagnostic 的 target_depth、lateral/keypoint/jam threshold。",
		"- `assembly_id` 不在 `task_vec_6` 中，但会选择 assembly asset 目录；真实 mesh/collision/AABB 是否不同必须用运行时几何 probe 验证。",
		"- 本报告不能证明接触动力学有效，只能证明有静态入口。",
	])
	return "\n".join(lines)


def build_report(args):
	messages: list[dict[str, Any]] = []
	tasks = load_task_id_vectors(args.task_hash_csv)
	task_a = tasks.get(str(args.task_id_a).zfill(5))
	task_b = tasks.get(str(args.task_id_b).zfill(5))
	if task_a is None or task_b is None:
		missing = [task_id for task_id, task in ((args.task_id_a, task_a), (args.task_id_b, task_b)) if task is None]
		add_message(messages, "FAIL", "Missing task vectors in task_id_to_hash.csv.", missing_task_ids=missing)
		return {"status": status_from_messages(messages), "messages": messages}
	compare = compare_task_vectors(task_a, task_b)
	if compare["task_vecs_differ"]:
		add_message(messages, "PASS", "01125 and 00256 task_vec_6 differ.", task_ids=[task_a["task_id"], task_b["task_id"]])
	else:
		add_message(messages, "FAIL", "01125 and 00256 task_vec_6 are identical.", task_ids=[task_a["task_id"], task_b["task_id"]])
	rows = build_rows(task_a, task_b)
	add_message(messages, "WARNING", "Runtime mesh/collision/contact effectiveness is not verified by static trace.")
	report = {
		"status": status_from_messages(messages),
		"task_id_a": task_a["task_id"],
		"task_id_b": task_b["task_id"],
		"task_a": task_a,
		"task_b": task_b,
		"task_vector_compare": compare,
		"task_vector_compare_json": __import__("json").dumps(compare, ensure_ascii=False, indent=2),
		"parameter_trace": rows,
		"static_code_evidence": [
			"tdmpc2/config.py:make_axial_task_vec excludes assembly_id and creates 6D param-only vector.",
			"tdmpc2/config.py:_apply_exact_axial_task_vec_to_sampler decodes scale/clearance/depth/yaw into sampler config.",
			"tdmpc2/envs/isaaclab.py:_configure_srsa_runtime_env exports SRSA_* and SRSA_AXIAL_* env vars.",
			"tdmpc2/envs/isaaclab.py:_configure_assembly_task selects assembly_dir and USD paths from assembly_id.",
			"tdmpc2/envs/isaaclab.py:_compute_srsa_eval_success uses target_depth/radial_clearance/yaw_requirement for diagnostics.",
		],
		"messages": messages,
	}
	report["status"] = status_from_messages(messages)
	return report


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--task-hash-csv", default=str(DEFAULT_TASK_HASH_CSV))
	parser.add_argument("--task-id-a", default="01125")
	parser.add_argument("--task-id-b", default="00256")
	parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
	parser.add_argument("--dry-run", action="store_true")
	args = parser.parse_args()

	report = build_report(args)
	output_dir = Path(args.output_dir).expanduser()
	if not output_dir.is_absolute():
		output_dir = DEFAULT_OUTPUT_DIR.parent.parent / output_dir
	print_status(report["status"], report.get("messages", []))
	write_json_report(report, output_dir / "size_param_trace_report.json", dry_run=args.dry_run)
	write_text_report(build_markdown(report), output_dir / "size_param_trace_report.md", dry_run=args.dry_run)
	return 1 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
	raise SystemExit(main())
