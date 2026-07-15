#!/usr/bin/env python3
"""Build the Phase 3.1 acquisition/retention/representation gate summary."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_ROOT = (
	REPO_ROOT / "logs" / "isaaclab-srsa-assembly" / "1" /
	"srsa_axial_online_family_taskctx_repair_01125_00256_00186"
)
DEFAULT_LAUNCHER = DEFAULT_RUN_ROOT / "20260712_phase3_1_00186_launcher"
DEFAULT_STAGE = DEFAULT_RUN_ROOT / "20260712_phase3_1_00186_stage-3_asm-00186"
DEFAULT_REPORT_DIR = REPO_ROOT / "reports" / "phase3_three_task_pilot"


def _resolve(value: str | Path) -> Path:
	path = Path(value).expanduser()
	if not path.is_absolute():
		path = REPO_ROOT / path
	return path.resolve()


def _load(path: str | Path) -> dict[str, Any]:
	path = _resolve(path)
	if not path.exists():
		raise FileNotFoundError(path)
	return json.loads(path.read_text(encoding="utf-8"))


def _task_rows(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
	out: dict[str, dict[str, Any]] = {}
	for item in summary.get("tasks") or []:
		out[str(item.get("assembly_id")).zfill(5)] = dict(item)
	for item in summary.get("csv_rows") or []:
		out.setdefault(str(item.get("assembly_id")).zfill(5), {}).update(item)
	return out


def _metric(row: dict[str, Any], *keys: str, default: float = 0.0) -> float:
	for key in keys:
		if row.get(key) is not None:
			return float(row[key])
		if row.get(f"episode_{key}") is not None:
			return float(row[f"episode_{key}"])
	return float(default)


def _latest_mixed_metrics(path: Path) -> dict[str, Any]:
	latest = None
	with path.open("r", encoding="utf-8") as f:
		for line in f:
			if not line.strip():
				continue
			item = json.loads(line)
			if item.get("category") == "train" and "online_family_batch_num_tasks" in item:
				latest = item
	if latest is None:
		raise RuntimeError(f"No mixed online-family metrics found in {path}")
	return latest


def _mean(values: list[float]) -> float:
	return sum(values) / max(len(values), 1)


def _sensitivity_summary(report: dict[str, Any], swap_label: str) -> dict[str, float]:
	item = report["comparisons_vs_base"][swap_label]
	return {
		"latent_l2": float(item["latent_l2"]["mean"]),
		"next_latent_l2": float(item["next_latent_l2"]["mean"]),
		"reward_abs": float(item["reward_abs"]["mean"]),
		"q_abs": float(item["q_abs"]["mean"]),
	}


def _mppi_summary(report: dict[str, Any], label: str) -> dict[str, float]:
	item = report["metrics_vs_base"][label]
	return {
		"kendall_tau": float(item["kendall_tau"]["mean"]),
		"top1_changed_rate": float(item["top1_changed_rate"]),
		"topk_overlap": float(item["topk_overlap"]["mean"]),
		"selected_action_l2": float(item["selected_action_L2"]["mean"]),
		"return_margin": float(item["return_margin_correct_vs_wrong"]["mean"]),
	}


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
	runtime = _load(args.runtime_smoke)
	best = _load(args.best_json)
	family = _load(args.family_summary)
	phase2_family = _load(args.phase2_family_summary)
	context = _load(args.context_probe)
	latent_01125 = _load(args.latent_vs_01125)
	latent_00256 = _load(args.latent_vs_00256)
	mppi_01125 = _load(args.mppi_vs_01125)
	mppi_00256 = _load(args.mppi_vs_00256)
	phase2_latent = _load(args.phase2_latent)
	phase2_mppi = _load(args.phase2_mppi)
	task_mix = _load(args.task_mix_report)
	mixed = _latest_mixed_metrics(_resolve(args.metrics_jsonl))

	rows = _task_rows(family)
	baseline_rows = _task_rows(phase2_family)
	needed = {"01125", "00256", "00186"}
	if not needed.issubset(rows):
		raise RuntimeError(f"Family eval is missing tasks: {sorted(needed - set(rows))}")

	task_metrics = {}
	for task_id, row in rows.items():
		task_metrics[task_id] = {
			"relaxed_success": _metric(row, "relaxed_success", "relaxed_success_stable", "episode_success"),
			"strict_success": _metric(row, "strict_success", "strict_success_stable"),
			"process_success": _metric(row, "process_success", "process_success_terminal"),
			"reward": _metric(row, "episode_reward"),
			"lateral_error_mm": _metric(row, "mean_lateral_error_mm", "lateral_error") * (1.0 if row.get("mean_lateral_error_mm") is not None else 1000.0),
			"keypoint_error_mm": _metric(row, "mean_keypoint_error_mm", "keypoint_error") * (1.0 if row.get("mean_keypoint_error_mm") is not None else 1000.0),
			"jam": _metric(row, "jam"),
		}

	acquisition_gate = task_metrics["00186"]["relaxed_success"] >= 0.75
	old_relaxed_gate = all(task_metrics[task]["relaxed_success"] >= 0.90 for task in ("01125", "00256"))
	family_mean = _mean([task_metrics[task]["relaxed_success"] for task in ("01125", "00256", "00186")])
	family_gate = family_mean >= 0.90
	old_quality = {}
	old_quality_gate = True
	for task_id in ("01125", "00256"):
		baseline = baseline_rows[task_id]
		base_strict = _metric(baseline, "strict_success", "strict_success_stable")
		base_process = _metric(baseline, "process_success", "process_success_terminal")
		new = task_metrics[task_id]
		strict_drop = base_strict - new["strict_success"]
		process_drop = base_process - new["process_success"]
		ok = strict_drop <= float(args.old_quality_drop_tol) and process_drop <= float(args.old_quality_drop_tol)
		old_quality_gate = old_quality_gate and ok
		old_quality[task_id] = {
			"phase2_strict": base_strict,
			"phase3_strict": new["strict_success"],
			"strict_drop": strict_drop,
			"phase2_process": base_process,
			"phase3_process": new["process_success"],
			"process_drop": process_drop,
			"gate": ok,
		}

	context_gate = bool(not context["context_collapse"])
	structure_gate = bool(context["near_pair_structure_gate"])
	recon_gate = float(context["task_reconstruction_r2"]) >= 0.8

	phase3_sens_01125 = _sensitivity_summary(latent_01125, "01125")
	phase3_sens_00256 = _sensitivity_summary(latent_00256, "00256")
	phase2_sens = _sensitivity_summary(phase2_latent, str(phase2_latent["swap_label"]))
	phase3_mppi_01125 = _mppi_summary(mppi_01125, "01125")
	phase3_mppi_00256 = _mppi_summary(mppi_00256, "00256")
	phase2_mppi_swap = _mppi_summary(phase2_mppi, str(phase2_mppi["swap_label"]))
	stronger_votes = sum([
		phase3_sens_01125["latent_l2"] > phase2_sens["latent_l2"],
		phase3_sens_01125["next_latent_l2"] > phase2_sens["next_latent_l2"],
		phase3_sens_01125["reward_abs"] > phase2_sens["reward_abs"],
		phase3_sens_01125["q_abs"] > phase2_sens["q_abs"],
		phase3_mppi_01125["top1_changed_rate"] > phase2_mppi_swap["top1_changed_rate"],
		phase3_mppi_01125["kendall_tau"] < phase2_mppi_swap["kendall_tau"],
	])
	stronger_than_near_pair = stronger_votes >= 4

	zero_mppi = _mppi_summary(mppi_01125, "zero")
	random_mppi = _mppi_summary(mppi_01125, "random")
	zero_random_gate = bool(
		max(zero_mppi["top1_changed_rate"], random_mppi["top1_changed_rate"]) >= 0.10
		or min(zero_mppi["kendall_tau"], random_mppi["kendall_tau"]) <= 0.95
	)

	gates = {
		"runtime_smoke": runtime.get("status") == "PASS",
		"three_task_replay_consistency": task_mix.get("status") == "PASS",
		"00186_acquisition_relaxed_ge_0p75": acquisition_gate,
		"old_tasks_relaxed_ge_0p90": old_relaxed_gate,
		"family_relaxed_mean_ge_0p90": family_gate,
		"old_strict_process_not_significantly_worse": old_quality_gate,
		"three_task_context_not_collapsed": context_gate,
		"near_pair_context_ordering": structure_gate,
		"task_reconstruction_r2_ge_0p8": recon_gate,
		"00186_zero_random_mppi_effect": zero_random_gate,
	}
	if acquisition_gate and old_relaxed_gate and family_gate and old_quality_gate:
		status = "PASS"
		decision = "ENTER_BALANCED_THREE_TASK_CONSOLIDATION"
	elif not acquisition_gate:
		if not old_relaxed_gate or not old_quality_gate:
			status = (
				"FAIL_ACQUISITION_AND_QUALITY_RETENTION_WITH_REPRESENTATION_PASS"
				if context_gate and structure_gate and recon_gate else
				"FAIL_ACQUISITION_AND_RETENTION"
			)
		else:
			status = "FAIL_ACQUISITION_WITH_REPRESENTATION_PASS" if context_gate and structure_gate and recon_gate else "FAIL_ACQUISITION"
		decision = "CONTINUE_00186_ACQUISITION_BEFORE_CONSOLIDATION"
	else:
		status = "FAIL_RETENTION"
		decision = "FIX_RETENTION_BEFORE_CONSOLIDATION"

	counts = {
		task: int(round(float(mixed.get(f"online_family_batch_task_count_{task}", 0.0))))
		for task in ("00186", "01125", "00256")
	}
	hashes = {
		key.removeprefix("online_family_batch_task_hash_count_"): int(round(float(value)))
		for key, value in mixed.items() if key.startswith("online_family_batch_task_hash_count_")
	}
	report = {
		"status": status,
		"decision": decision,
		"checkpoint": str(_resolve(args.checkpoint)),
		"runtime_smoke": runtime,
		"training_best": best,
		"replay_counts": counts,
		"replay_task_hash_counts": hashes,
		"replay_condition_entropy_norm": float(mixed.get("online_family_batch_condition_entropy_norm", math.nan)),
		"task_mix_audit": task_mix,
		"task_metrics": task_metrics,
		"family_relaxed_mean": family_mean,
		"old_quality_vs_phase2": old_quality,
		"context": {
			"pairwise_l2": context["task_context_l2"],
			"reconstruction_r2": context["task_reconstruction_r2"],
			"ctx_task_distance_corr": context["ctx_task_distance_corr"],
			"collapse": context["context_collapse"],
			"near_pair_structure_gate": context["near_pair_structure_gate"],
		},
		"sensitivity": {
			"phase3_00186_vs_01125": phase3_sens_01125,
			"phase3_00186_vs_00256": phase3_sens_00256,
			"phase2_00256_vs_01125": phase2_sens,
			"phase3_mppi_00186_vs_01125": phase3_mppi_01125,
			"phase3_mppi_00186_vs_00256": phase3_mppi_00256,
			"phase2_mppi_00256_vs_01125": phase2_mppi_swap,
			"zero_mppi": zero_mppi,
			"random_mppi": random_mppi,
			"stronger_than_phase2_near_pair": stronger_than_near_pair,
		},
		"gates": gates,
	}

	lines = [
		"# SRSA Phase 3.1 Three-Task Pilot Summary",
		"",
		"本报告汇总 `00186` 100k acquisition、三任务 retention、representation 和 task-vector sensitivity。未启用 counterfactual reward、reward residual，也未修改 Q/policy/MPPI。",
		"",
		f"Status: `{status}`",
		"",
		"## 结论回答",
		"",
		f"1. `00186` acquisition 是否成功：`{acquisition_gate}`。relaxed success=`{task_metrics['00186']['relaxed_success']:.4f}`，gate=`0.75`。",
		f"2. `01125/00256` retention 是否保持：relaxed gate=`{old_relaxed_gate}`，strict/process quality gate=`{old_quality_gate}`；overall=`{old_relaxed_gate and old_quality_gate}`。relaxed=`{task_metrics['01125']['relaxed_success']:.3f}/{task_metrics['00256']['relaxed_success']:.3f}`。",
		f"3. 三任务 task_context 是否形成合理结构：`{context_gate and structure_gate and recon_gate}`。near-pair distance 最小，recon R²=`{context['task_reconstruction_r2']:.6f}`。",
		f"4. `00186` 是否比 `01125/00256` near-pair 表现出更强 task-vector sensitivity：`{stronger_than_near_pair}`。",
		f"5. 下一步：`{decision}`。",
		"",
		"## Gate",
		"",
	]
	for name, passed in gates.items():
		lines.append(f"- `{name}`: `{'PASS' if passed else 'FAIL'}`")
	lines.extend([
		"",
		"## 三任务评估",
		"",
		"| Task | Relaxed | Strict | Process | Reward | Lateral mm | Keypoint mm | Jam |",
		"| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
	])
	for task_id in ("01125", "00256", "00186"):
		item = task_metrics[task_id]
		lines.append(
			f"| `{task_id}` | {item['relaxed_success']:.3f} | {item['strict_success']:.3f} | "
			f"{item['process_success']:.3f} | {item['reward']:.2f} | {item['lateral_error_mm']:.3f} | "
			f"{item['keypoint_error_mm']:.3f} | {item['jam']:.3f} |"
		)
	lines.extend([
		"",
		f"Family relaxed mean: `{family_mean:.4f}`。",
		"",
		"## Replay",
		"",
		f"- counts: `{counts}`",
		f"- task hashes: `{hashes}`",
		f"- condition entropy norm: `{report['replay_condition_entropy_norm']:.6f}`",
		f"- independent task-mix audit: `{task_mix.get('status')}`",
		"",
		"## Task Context",
		"",
		f"- pairwise L2: `{context['task_context_l2']}`",
		f"- reconstruction R²: `{context['task_reconstruction_r2']:.6f}`",
		f"- context/task distance correlation: `{context['ctx_task_distance_corr']:.6f}`",
		"",
		"## Sensitivity",
		"",
		f"- `00186 vs 01125`: `{phase3_sens_01125}`",
		f"- `00186 vs 00256`: `{phase3_sens_00256}`",
		f"- Phase 2 near-pair baseline: `{phase2_sens}`",
		f"- MPPI `00186 vs 01125`: `{phase3_mppi_01125}`",
		f"- MPPI zero/random top1 changed: `{zero_mppi['top1_changed_rate']:.4f}/{random_mppi['top1_changed_rate']:.4f}`",
		"",
		"## 决策",
		"",
		"当前 representation 和 task-vector sensitivity 已通过，但只有 acquisition/retention gate 同时通过后才进入三任务均衡 consolidation。若 `00186` acquisition 未过，应继续 current-heavy acquisition，不新增任务，也不启用新方法模块。",
	])
	return report, "\n".join(lines) + "\n"


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--runtime-smoke", default=str(DEFAULT_REPORT_DIR / "00186_runtime_smoke.json"))
	parser.add_argument("--best-json", default=str(DEFAULT_STAGE / "models" / "best.json"))
	parser.add_argument("--checkpoint", default=str(DEFAULT_STAGE / "models" / "latest.pt"))
	parser.add_argument("--metrics-jsonl", default=str(DEFAULT_STAGE / "metrics.jsonl"))
	parser.add_argument("--family-summary", default=str(DEFAULT_LAUNCHER / "family_eval_after_00186" / "batch_eval_summary.json"))
	parser.add_argument("--phase2-family-summary", default=(
		"logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_taskctx_repair_01125_00256/"
		"20260708_taskctx_repair_phase2_launcher/family_eval_after_00256/batch_eval_summary.json"
	))
	parser.add_argument("--context-probe", default=str(DEFAULT_REPORT_DIR / "three_task_context_probe.json"))
	parser.add_argument("--latent-vs-01125", default=str(DEFAULT_REPORT_DIR / "sensitivity_vs_01125" / "latent_dynamics_reward_sensitivity.json"))
	parser.add_argument("--latent-vs-00256", default=str(DEFAULT_REPORT_DIR / "sensitivity_vs_00256" / "latent_dynamics_reward_sensitivity.json"))
	parser.add_argument("--mppi-vs-01125", default=str(DEFAULT_REPORT_DIR / "sensitivity_vs_01125" / "mppi_task_ranking_sensitivity.json"))
	parser.add_argument("--mppi-vs-00256", default=str(DEFAULT_REPORT_DIR / "sensitivity_vs_00256" / "mppi_task_ranking_sensitivity.json"))
	parser.add_argument("--phase2-latent", default="reports/model_task_sensitivity/phase2_1_probe/repair/latent_dynamics_reward_sensitivity.json")
	parser.add_argument("--phase2-mppi", default="reports/model_task_sensitivity/phase2_1_probe/repair/mppi_task_ranking_sensitivity.json")
	parser.add_argument("--task-mix-report", default=str(DEFAULT_REPORT_DIR / "task_consistency" / "online_family_sample_mix_report.json"))
	parser.add_argument("--old-quality-drop-tol", type=float, default=0.10)
	parser.add_argument("--output-json", default=str(DEFAULT_REPORT_DIR / "phase3_1_summary.json"))
	parser.add_argument("--output-md", default=str(DEFAULT_REPORT_DIR / "phase3_1_summary.md"))
	args = parser.parse_args()
	report, markdown = build_report(args)
	output_json = _resolve(args.output_json)
	output_md = _resolve(args.output_md)
	output_json.parent.mkdir(parents=True, exist_ok=True)
	output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
	output_md.write_text(markdown, encoding="utf-8")
	print(f"{report['status']} wrote {output_md}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
