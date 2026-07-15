#!/usr/bin/env python3
"""Build a Chinese summary for Phase 1.0 model task-sensitivity audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _common import DEFAULT_OUTPUT_DIR, add_message, output_dir, print_status, safe_mean, status_from_messages, write_text


def _load(path: Path) -> dict[str, Any] | None:
	if not path.exists():
		return None
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def _is_status_bad(report: dict[str, Any] | None) -> bool:
	return not isinstance(report, dict) or str(report.get("status", "")).upper() == "FAIL"


def build_summary(args):
	out_dir = output_dir(args)
	context = _load(out_dir / "task_context_sensitivity.json")
	latent = _load(out_dir / "latent_dynamics_reward_sensitivity.json")
	ranking = _load(out_dir / "mppi_task_ranking_sensitivity.json")
	messages: list[dict[str, Any]] = []
	for name, report in (("task_context", context), ("latent_dynamics_reward", latent), ("mppi_ranking", ranking)):
		if report is None:
			add_message(messages, "FAIL", f"Missing required report: {name}.")
		elif str(report.get("status", "")).upper() == "FAIL":
			add_message(messages, "FAIL", f"{name} report status is FAIL.")
	if _is_status_bad(context) or _is_status_bad(latent) or _is_status_bad(ranking):
		classification = "UNKNOWN_INCOMPLETE"
	else:
		context_l2 = float(context.get("task_context_L2_01125_00256", 0.0))
		context_collapsed = context_l2 <= float(args.context_eps)
		latent_swap = safe_mean(latent.get("latent_delta_correct_swap"), "mean")
		latent_zero = safe_mean(latent.get("latent_delta_correct_zero"), "mean")
		next_swap = safe_mean(latent.get("next_latent_delta_correct_swap"), "mean")
		next_zero = safe_mean(latent.get("next_latent_delta_correct_zero"), "mean")
		reward_swap = safe_mean(latent.get("reward_delta_correct_swap"), "mean")
		reward_zero = safe_mean(latent.get("reward_delta_correct_zero"), "mean")
		q_swap = safe_mean(latent.get("Q_delta_correct_swap"), "mean")
		q_zero = safe_mean(latent.get("Q_delta_correct_zero"), "mean")
		latent_sensitive = max(latent_swap, latent_zero) > float(args.latent_eps)
		next_sensitive = max(next_swap, next_zero) > float(args.latent_eps)
		reward_sensitive = max(reward_swap, reward_zero) > float(args.reward_eps)
		q_sensitive = max(q_swap, q_zero) > float(args.q_eps)
		zero_tau = safe_mean(ranking.get("kendall_tau_correct_zero"), "mean", default=1.0)
		random_tau = safe_mean(ranking.get("kendall_tau_correct_random"), "mean", default=1.0)
		top1_rates = ranking.get("top1_changed_rate") or {}
		ranking_sensitive = (
			min(zero_tau, random_tau) < float(args.ranking_tau_eps)
			or max(float(top1_rates.get("zero", 0.0)), float(top1_rates.get("random", 0.0))) > 0.0
		)
		score = sum(bool(x) for x in (not context_collapsed, latent_sensitive, next_sensitive, reward_sensitive, q_sensitive, ranking_sensitive))
		if score >= 5:
			classification = "task-sensitive"
		elif score >= 2:
			classification = "weakly task-sensitive"
		else:
			classification = "task-insensitive"
		if context_collapsed:
			recommendation = "task_context repair"
		elif not reward_sensitive and not q_sensitive:
			recommendation = "counterfactual reward"
		elif not ranking_sensitive:
			recommendation = "tiny reward residual"
		else:
			recommendation = "继续做更严格闭环/near-pair ablation，暂不修模型"
		add_message(messages, "PASS" if classification != "task-insensitive" else "WARNING", f"Model classified as {classification}.")
	diagnosis = {
		"task_context_collapsed": None,
		"latent_sensitive": None,
		"next_latent_sensitive": None,
		"reward_sensitive": None,
		"q_sensitive": None,
		"mppi_ranking_sensitive": None,
		"classification": classification,
		"recommendation": None,
	}
	if isinstance(context, dict) and isinstance(latent, dict) and isinstance(ranking, dict):
		diagnosis["task_context_collapsed"] = float(context.get("task_context_L2_01125_00256", 0.0)) <= float(args.context_eps)
		diagnosis["latent_sensitive"] = max(
			safe_mean(latent.get("latent_delta_correct_swap"), "mean"),
			safe_mean(latent.get("latent_delta_correct_zero"), "mean"),
		) > float(args.latent_eps)
		diagnosis["next_latent_sensitive"] = max(
			safe_mean(latent.get("next_latent_delta_correct_swap"), "mean"),
			safe_mean(latent.get("next_latent_delta_correct_zero"), "mean"),
		) > float(args.latent_eps)
		diagnosis["reward_sensitive"] = max(
			safe_mean(latent.get("reward_delta_correct_swap"), "mean"),
			safe_mean(latent.get("reward_delta_correct_zero"), "mean"),
		) > float(args.reward_eps)
		diagnosis["q_sensitive"] = max(
			safe_mean(latent.get("Q_delta_correct_swap"), "mean"),
			safe_mean(latent.get("Q_delta_correct_zero"), "mean"),
		) > float(args.q_eps)
		top1_rates = ranking.get("top1_changed_rate") or {}
		diagnosis["mppi_ranking_sensitive"] = (
			min(
				safe_mean(ranking.get("kendall_tau_correct_zero"), "mean", default=1.0),
				safe_mean(ranking.get("kendall_tau_correct_random"), "mean", default=1.0),
			) < float(args.ranking_tau_eps)
			or max(float(top1_rates.get("zero", 0.0)), float(top1_rates.get("random", 0.0))) > 0.0
		)
		if diagnosis["task_context_collapsed"]:
			diagnosis["recommendation"] = "task_context repair"
		elif not diagnosis["reward_sensitive"] and not diagnosis["q_sensitive"]:
			diagnosis["recommendation"] = "counterfactual reward"
		elif not diagnosis["mppi_ranking_sensitive"]:
			diagnosis["recommendation"] = "tiny reward residual"
		else:
			diagnosis["recommendation"] = "继续做更严格闭环/near-pair ablation，暂不修模型"
	lines = [
		"# SRSA Phase 1.0 Model Task Sensitivity Summary",
		"",
		"本报告只汇总模型侧 sensitivity audit；未修改模型、训练流程、sampler 或 reward。",
		"",
		f"Status: `{status_from_messages(messages)}`",
		"",
		"## 结论回答",
		"",
		f"1. task_context 是否塌缩：`{diagnosis['task_context_collapsed']}`。",
		f"2. task_vec 是否影响 encoder latent：`{diagnosis['latent_sensitive']}`。",
		f"3. task_vec 是否影响 dynamics / next latent：`{diagnosis['next_latent_sensitive']}`。",
		f"4. task_vec 是否影响 reward prediction：`{diagnosis['reward_sensitive']}`。",
		f"5. task_vec 是否影响 Q prediction：`{diagnosis['q_sensitive']}`。",
		f"6. task_vec 是否影响 MPPI candidate ranking：`{diagnosis['mppi_ranking_sensitive']}`。",
		f"7. 当前模型分类：`{diagnosis['classification']}`。",
		f"8. 下一步建议：`{diagnosis['recommendation']}`。",
		"",
		"## 分类细节",
		"",
		"```json",
		json.dumps(diagnosis, ensure_ascii=False, indent=2),
		"```",
		"",
		"## 输入报告",
		"",
		"- `task_context_sensitivity.json`",
		"- `latent_dynamics_reward_sensitivity.json`",
		"- `mppi_task_ranking_sensitivity.json`",
	]
	return "\n".join(lines), messages


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
	parser.add_argument("--context-eps", type=float, default=1.0e-5)
	parser.add_argument("--latent-eps", type=float, default=1.0e-5)
	parser.add_argument("--reward-eps", type=float, default=1.0e-5)
	parser.add_argument("--q-eps", type=float, default=1.0e-4)
	parser.add_argument("--ranking-tau-eps", type=float, default=0.999)
	parser.add_argument("--dry-run", action="store_true")
	args = parser.parse_args()
	text, messages = build_summary(args)
	print_status(status_from_messages(messages), messages)
	write_text(text, output_dir(args) / "model_task_sensitivity_summary.md", dry_run=args.dry_run)
	return 1 if status_from_messages(messages) == "FAIL" else 0


if __name__ == "__main__":
	raise SystemExit(main())
