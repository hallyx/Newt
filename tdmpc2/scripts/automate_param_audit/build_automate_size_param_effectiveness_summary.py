#!/usr/bin/env python3
"""Build the Chinese AutoMate size-parameter effectiveness summary."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from _common import DEFAULT_OUTPUT_DIR, add_message, load_json, print_status, status_from_messages, write_text_report


def _path(output_dir: Path, name: str) -> Path:
	return output_dir / name


def _metric_nonzero(report: dict[str, Any] | None) -> bool:
	if not isinstance(report, dict):
		return False
	nonzero = report.get("nonzero_deltas", {})
	return bool(nonzero)


def _trace_task_vec_diff(trace: dict[str, Any] | None) -> bool | None:
	if not isinstance(trace, dict):
		return None
	return bool((trace.get("task_vector_compare") or {}).get("task_vecs_differ"))


def _runtime_geometry_effective(geometry_compare: dict[str, Any] | None) -> bool | None:
	if not isinstance(geometry_compare, dict):
		return None
	if geometry_compare.get("runtime_geometry_check") != "DONE":
		return None
	if geometry_compare.get("geometry_effective") is not None:
		return bool(geometry_compare.get("geometry_effective"))
	geom = geometry_compare.get("geometry_comparison") or {}
	return bool(geom.get("geometry_effective", False))


def _runtime_objective_effective(geometry_compare: dict[str, Any] | None, reward_report: dict[str, Any] | None) -> bool | None:
	if isinstance(geometry_compare, dict) and geometry_compare.get("runtime_geometry_check") == "DONE":
		if geometry_compare.get("objective_effective") is not None:
			return bool(geometry_compare.get("objective_effective"))
		geom = geometry_compare.get("geometry_comparison") or {}
		if geom.get("objective_effective") is not None:
			return bool(geom.get("objective_effective"))
	if isinstance(reward_report, dict) and _metric_nonzero(reward_report):
		return True
	return None


def _runtime_success_threshold_effective(geometry_compare: dict[str, Any] | None) -> bool | None:
	if not isinstance(geometry_compare, dict):
		return None
	if geometry_compare.get("runtime_geometry_check") != "DONE":
		return None
	if geometry_compare.get("success_threshold_differs") is not None:
		return bool(geometry_compare.get("success_threshold_differs"))
	geom = geometry_compare.get("geometry_comparison") or {}
	if geom.get("success_threshold_differs") is not None:
		return bool(geom.get("success_threshold_differs"))
	return None


def _contact_diff(contact: dict[str, Any] | None) -> bool | None:
	if not isinstance(contact, dict):
		return None
	if contact.get("runtime_contact_check") != "DONE":
		return None
	if contact.get("contact_effective") is not None:
		return bool(contact.get("contact_effective"))
	compare = contact.get("metric_compare") or {}
	return any(not item.get("same", True) for item in compare.values() if isinstance(item, dict))


def _unknown_reason(report: dict[str, Any] | None, check_key: str) -> str:
	if not isinstance(report, dict):
		return "report missing"
	if report.get(check_key) == "FAILED":
		messages = report.get("messages") or []
		return "env launch or runtime probe failed: " + "; ".join(str(item.get("message", "")) for item in messages)
	if report.get(check_key) == "SKIPPED":
		return str(report.get("reason") or "script was run without --launch-env")
	return "environment API did not expose enough comparable fields"


def build_summary(args):
	output_dir = Path(args.output_dir).expanduser()
	if not output_dir.is_absolute():
		output_dir = DEFAULT_OUTPUT_DIR.parent.parent / output_dir
	output_dir = output_dir.resolve()
	trace = load_json(_path(output_dir, "size_param_trace_report.json"), default={})
	geometry_runtime = load_json(_path(output_dir, "geometry_compare_01125_00256_runtime.json"), default=None)
	geometry_static = load_json(_path(output_dir, "geometry_compare_01125_00256.json"), default={})
	geometry = geometry_runtime if geometry_runtime is not None else geometry_static
	reward = load_json(_path(output_dir, "reward_size_sensitivity.json"), default={})
	contact_runtime = load_json(_path(output_dir, "contact_size_sensitivity_runtime.json"), default=None)
	contact_static = load_json(_path(output_dir, "contact_size_sensitivity.json"), default={})
	contact = contact_runtime if contact_runtime is not None else contact_static

	messages: list[dict[str, Any]] = []
	task_vec_diff = _trace_task_vec_diff(trace)
	geometry_effective = _runtime_geometry_effective(geometry)
	objective_effective = _runtime_objective_effective(geometry, reward)
	success_threshold_effective = _runtime_success_threshold_effective(geometry)
	reward_diff = _metric_nonzero(reward)
	contact_diff = _contact_diff(contact)

	if task_vec_diff:
		add_message(messages, "PASS", "01125/00256 task_vec differ.")
	else:
		add_message(messages, "FAIL" if task_vec_diff is False else "WARNING", "task_vec difference could not be confirmed.")
	if geometry_effective is None:
		add_message(messages, "WARNING", "Runtime mesh/collision geometry effectiveness is not verified.")
	elif geometry_effective:
		add_message(messages, "PASS", "Runtime geometry records differ.")
	else:
		add_message(messages, "WARNING", "Runtime asset/AABB/scale records did not differ.")
	if objective_effective is None:
		add_message(messages, "WARNING", "Runtime target pose/depth/threshold effectiveness is unknown.")
	elif objective_effective:
		add_message(messages, "PASS", "Target/depth/threshold or surrogate objective metrics are size-sensitive.")
	else:
		add_message(messages, "WARNING", "Runtime target/depth/threshold fields did not differ.")
	if reward_diff:
		add_message(messages, "PASS", "Reward/success surrogate metrics are size-sensitive.")
	else:
		add_message(messages, "FAIL", "Reward/success surrogate metrics did not change.")
	if contact_diff is None:
		add_message(messages, "WARNING", "Runtime contact dynamics check is skipped.")
	elif contact_diff:
		add_message(messages, "PASS", "Contact rollout metrics differ.")
	else:
		add_message(messages, "WARNING", "Contact rollout metrics were accessible but identical.")

	label_only = (
		bool(task_vec_diff)
		and geometry_effective is False
		and objective_effective is False
		and not reward_diff
		and contact_diff is False
	)
	overall = "LABEL_ONLY" if label_only else "SIZE_PARAMETERS_HAVE_NON_LABEL_EFFECTS_OR_RUNTIME_UNKNOWN"
	if geometry_effective is True and objective_effective is True and reward_diff and contact_diff is True:
		overall = "GEOMETRY_OBJECTIVE_REWARD_CONTACT_EFFECTIVE"
	elif geometry_effective is True and (objective_effective is True or reward_diff):
		overall = "GEOMETRY_OBJECTIVE_REWARD_EFFECTIVE_CONTACT_UNKNOWN_OR_WEAK"
	elif geometry_effective is None or contact_diff is None:
		overall = "STATIC_OBJECTIVE_REWARD_EFFECTIVE_WITH_RUNTIME_GEOMETRY_CONTACT_UNKNOWN"
	classification = {
		"geometry_effective": geometry_effective if geometry_effective is not None else "UNKNOWN_RUNTIME",
		"objective_effective": objective_effective if objective_effective is not None else "UNKNOWN_RUNTIME",
		"success_threshold_effective": success_threshold_effective if success_threshold_effective is not None else "UNKNOWN_RUNTIME",
		"reward_effective": bool(reward_diff),
		"contact_effective": contact_diff if contact_diff is not None else "UNKNOWN_RUNTIME_SKIPPED",
		"label_only": bool(label_only),
		"overall": overall,
		"unknown_reasons": {
			"geometry": None if geometry_effective is not None else _unknown_reason(geometry, "runtime_geometry_check"),
			"objective": None if objective_effective is not None else _unknown_reason(geometry, "runtime_geometry_check"),
			"contact": None if contact_diff is not None else _unknown_reason(contact, "runtime_contact_check"),
		},
	}

	lines = [
		"# AutoMate/SRSA Size Parameter Effectiveness Summary Runtime",
		"",
		"本报告对应 Phase 0.7，只读运行时验证，不修改模型、训练逻辑、sampler 或 reward。",
		"",
		f"Status: `{status_from_messages(messages)}`",
		"",
		"## 结论回答",
		"",
		f"1. 01125/00256 的 `task_vec_6` 是否不同：`{task_vec_diff}`。",
		f"2. 真实 asset/mesh/collision 是否不同：`{geometry_effective if geometry_effective is not None else 'UNKNOWN_RUNTIME'}`。",
		f"3. size 是否影响 target depth/pose：`{objective_effective if objective_effective is not None else 'UNKNOWN_RUNTIME'}`。",
		f"4. size 是否影响 reward/success threshold：`success_threshold={success_threshold_effective if success_threshold_effective is not None else 'UNKNOWN_RUNTIME'}, reward_surrogate={bool(reward_diff)}`。",
		f"5. size 是否影响 contact dynamics：`{contact_diff if contact_diff is not None else 'UNKNOWN_RUNTIME_SKIPPED'}`。",
		f"6. 分类：`{classification['overall']}`。",
		"7. 未知项原因：见 `unknown_reasons`。",
		"",
		"## 分类细节",
		"",
		"```json",
		__import__("json").dumps(classification, ensure_ascii=False, indent=2),
		"```",
		"",
		"## 当前可确认的事实",
		"",
		"- Phase 0.5 已证明 replay 中 01125/00256 的 task hash 不同，online-family mixed sample 中 hash_counts 与 label_counts 一致。",
		"- 静态代码路径显示 `clearance`/`depth` 会进入 `task_vec_6`、sampler/env var，并进入 eval diagnostic 的 depth/lateral/keypoint/jam 相关计算。",
		"- Phase 0.7 优先读取 runtime 几何和接触报告；如果 runtime 报告缺失或失败，本 summary 会明确保留 UNKNOWN。",
		"",
		"## Runtime 输入状态",
		"",
		f"- geometry runtime report: `{geometry.get('runtime_geometry_check', 'MISSING') if isinstance(geometry, dict) else 'MISSING'}`",
		f"- contact runtime report: `{contact.get('runtime_contact_check', 'MISSING') if isinstance(contact, dict) else 'MISSING'}`",
		f"- geometry unknown reason: `{classification['unknown_reasons']['geometry']}`",
		f"- contact unknown reason: `{classification['unknown_reasons']['contact']}`",
		"",
		"## 如果后续发现 label-only",
		"",
		"- 让 `assembly_id` 或 size 参数实际选择不同 USD/mesh/collision，或在相同 assembly 下显式改变 plug/hole scale。",
		"- 确认 `current_task_params` 中的 `radial_clearance`、`insertion_depth` 与 env 几何/目标位姿一致。",
		"- 把 reward/success 中的 clearance/depth/yaw 依赖保持为 runtime task params，而不是只依赖 cfg 默认值。",
		"- 在加入第三任务前，先用 `inspect_automate_env_geometry.py --launch-env` 和 `automate_contact_size_sensitivity.py --launch-env` 补齐运行时证据。",
		"",
		"## 输入报告",
		"",
		"- `size_param_trace_report.json`",
		"- `geometry_compare_01125_00256_runtime.json` 或 `geometry_compare_01125_00256.json`",
		"- `reward_size_sensitivity.json`",
		"- `contact_size_sensitivity_runtime.json` 或 `contact_size_sensitivity.json`",
		"- `size_override_probe.json`",
	]
	return "\n".join(lines), messages


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
	parser.add_argument("--runtime", action="store_true", help="Write the Phase 0.7 runtime summary filename.")
	parser.add_argument("--dry-run", action="store_true")
	args = parser.parse_args()

	text, messages = build_summary(args)
	output_dir = Path(args.output_dir).expanduser()
	if not output_dir.is_absolute():
		output_dir = DEFAULT_OUTPUT_DIR.parent.parent / output_dir
	print_status(status_from_messages(messages), messages)
	name = "automate_size_param_effectiveness_summary_runtime.md" if args.runtime else "automate_size_param_effectiveness_summary.md"
	write_text_report(text, output_dir / name, dry_run=args.dry_run)
	return 1 if status_from_messages(messages) == "FAIL" else 0


if __name__ == "__main__":
	raise SystemExit(main())
