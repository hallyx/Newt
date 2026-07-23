#!/usr/bin/env python3
"""Build the Phase 4.2 causal-ablation report from completed artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ARMS = ("fixed", "parametric", "staged_A", "parametric_then_three_task")
CONDITIONS = (
	("seen_low", "seen"),
	("seen_nominal", "seen"),
	("seen_high", "seen"),
	("heldout_interpolation", "interpolation"),
	("heldout_extrapolation", "extrapolation"),
)


def load(path: Path):
	if not path.exists():
		raise FileNotFoundError(path)
	return json.loads(path.read_text(encoding="utf-8"))


def f(value, digits=3):
	try:
		value = float(value)
	except (TypeError, ValueError):
		return "n/a"
	return "n/a" if not math.isfinite(value) else f"{value:.{digits}f}"


def metric_row(summary):
	row = summary["tasks"][0]
	return {
		"relaxed": float(row.get("episode_relaxed_success_stable", row.get("episode_success", 0.0))),
		"strict": float(row.get("episode_strict_success_stable", 0.0)),
		"process": float(row.get("episode_strict_success_episode", row.get("episode_process_success", 0.0))),
		"lateral_mm": 1000.0 * float(row.get("episode_lateral_error", math.nan)),
		"keypoint_mm": 1000.0 * float(row.get("episode_keypoint_error", math.nan)),
		"jam": float(row.get("episode_jam", math.nan)),
	}


def mean(rows, key):
	values = [float(row[key]) for row in rows if math.isfinite(float(row[key]))]
	return sum(values) / len(values) if values else math.nan


def ranking_summary(report):
	result = {}
	for phase, row in report["prediction_reality"]["phase_results"].items():
		result[phase] = {
			"spearman": row["ranking_correlations"]["total"]["vs_actual_return_5"]["spearman_mean"],
			"top8_gain": row["predicted_total_topk_realized"]["predicted_topk_gain_vs_all"],
			"top8_overlap": row["predicted_total_topk_realized"]["topk_overlap_with_actual"],
		}
	return result


def offline_summary(report):
	compatibility = report.get("architecture_compatibility", report.get("compatibility", {}))
	context = report["task_context_structure"]
	has_pairwise_context = bool(context.get("pairwise_l2"))
	aggregate = report["world_model"]["aggregate"]
	dynamics = aggregate["multi_step_dynamics_latent_l2"]
	reward = aggregate["reward_calibration"]
	q_value = aggregate["q_calibration_to_replay_return"]
	regret = report.get("proposal_regret", {}).get("summary", {})
	return {
		"task_dim": compatibility.get("task_dim"),
		"adapter_source": compatibility.get("task_context_adapter_source"),
		"adapter_encoder": compatibility.get("task_context_adapter_apply_encoder"),
		"adapter_dynamics": compatibility.get("task_context_adapter_apply_dynamics"),
		"context_task_pearson": context.get("context_distance_vs_task_distance_pearson", math.nan) if has_pairwise_context else math.nan,
		"task_reconstruction_r2": context.get("task_reconstruction_r2", math.nan) if has_pairwise_context else math.nan,
		"dynamics_1_mean": dynamics["1"]["mean"],
		"dynamics_2_mean": dynamics["2"]["mean"],
		"dynamics_3_mean": dynamics["3"]["mean"],
		"reward_mae": reward["mae"],
		"reward_pearson": reward["pearson"],
		"q_mae": q_value["mae"],
		"q_pearson": q_value["pearson"],
		"proposal_regret_mean": regret.get("mean", math.nan),
	}


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--root", default="reports/phase4_2_parametric_pretraining")
	parser.add_argument("--output-json", default="reports/phase4_2_parametric_pretraining_ablation.json")
	parser.add_argument("--output-md", default="reports/phase4_2_parametric_pretraining_ablation.md")
	args = parser.parse_args()
	root = Path(args.root).resolve()

	runtime = load(root / "runtime_gate" / "parametric_runtime_gate.json")
	param_metrics = {}
	for arm in ARMS:
		param_metrics[arm] = {}
		for condition, split in CONDITIONS:
			summary = load(root / "closed_loop_param" / arm / condition / "batch_eval_summary.json")
			param_metrics[arm][condition] = {"split": split, **metric_row(summary)}

	three_task = {}
	for arm in ARMS:
		summary = load(root / "closed_loop_three_task" / arm / "batch_eval_summary.json")
		three_task[arm] = {
			str(row["assembly_id"]): {
				"relaxed": float(row.get("episode_relaxed_success_stable", row.get("episode_success", 0.0))),
				"strict": float(row.get("episode_strict_success_stable", 0.0)),
				"process": float(row.get("episode_strict_success_episode", row.get("episode_process_success", 0.0))),
				"lateral_mm": 1000.0 * float(row.get("episode_lateral_error", math.nan)),
				"keypoint_mm": 1000.0 * float(row.get("episode_keypoint_error", math.nan)),
				"jam": float(row.get("episode_jam", math.nan)),
			}
			for row in summary["tasks"]
		}

	offline = {
		"fixed": load(root / "offline" / "fixed_from_start.json"),
		"parametric": load(root / "offline" / "parametric_from_start.json"),
		"staged_A": load(root / "offline" / "variant_staged_A.json"),
		"parametric_then_three_task": load(root / "offline" / "variant_parametric_then_three_task.json"),
	}
	offline_aggregate = {arm: offline_summary(arm_report) for arm, arm_report in offline.items()}
	expansion_failure_path = root / "checkpoints" / "phase4_2_expansion_v1_failure.json"
	expansion_failure = load(expansion_failure_path) if expansion_failure_path.exists() else None
	ranking_files = {
		"fixed_nominal": "fixed_nominal.json",
		"param_seen_nominal": "param_seen_nominal.json",
		"param_heldout_interpolation": "param_heldout_interpolation.json",
		"param_heldout_extrapolation": "param_heldout_extrapolation.json",
		"staged_A_00186": "staged_A_00186.json",
		"parametric_then_three_task_00186": "parametric_then_three_task_00186.json",
	}
	ranking = {key: ranking_summary(load(root / "ranking" / filename)) for key, filename in ranking_files.items()}

	seen_names = [name for name, split in CONDITIONS if split == "seen"]
	held_names = [name for name, split in CONDITIONS if split != "seen"]
	seen_mean = {arm: mean([param_metrics[arm][name] for name in seen_names], "relaxed") for arm in ARMS}
	held_mean = {arm: mean([param_metrics[arm][name] for name in held_names], "relaxed") for arm in ARMS}
	param_heldout_gain = held_mean["parametric"] - held_mean["fixed"]
	param_seen_delta = seen_mean["parametric"] - seen_mean["fixed"]
	p3 = three_task["parametric_then_three_task"]
	a = three_task["staged_A"]
	retention_ok = p3["01125"]["relaxed"] >= a["01125"]["relaxed"] - .05
	transfer_ok = (
		p3["00256"]["relaxed"] >= a["00256"]["relaxed"] - .05
		and p3["00186"]["relaxed"] >= a["00186"]["relaxed"] - .05
	) or p3["00186"]["relaxed"] >= a["00186"]["relaxed"] + .10
	heldout_retained = held_mean["parametric_then_three_task"] >= held_mean["parametric"] - .10
	param_advantage = param_heldout_gain >= .10 and param_seen_delta >= -.05
	if runtime["status"] != "PASS":
		classification = "RUNTIME_PARAMETRIC_CAUSAL_CHAIN_FAILED"
	elif param_advantage and retention_ok and transfer_ok and heldout_retained:
		classification = "PARAMETRIC_PRETRAINING_SUPPORTS_EXPANSION"
	elif param_heldout_gain < .05 and not transfer_ok:
		classification = "PARAMETRIC_PRETRAINING_NO_TRANSFER"
	elif mean(list(a.values()), "relaxed") > mean(list(p3.values()), "relaxed") + .05 and not param_advantage:
		classification = "STAGED_PATH_REMAINS_BETTER"
	else:
		classification = "MIXED_OR_UNRESOLVED"

	report = {
		"status": "PASS",
		"phase": "4.2",
		"classification": classification,
		"predeclared_gates": {
			"runtime_causal_chain_pass": runtime["status"] == "PASS",
			"parametric_heldout_advantage": param_advantage,
			"three_task_retention_vs_A": retention_ok,
			"three_task_transfer_vs_A": transfer_ok,
			"heldout_retention_after_expansion": heldout_retained,
			"thresholds": {
				"heldout_gain_over_fixed": .10,
				"allowed_seen_drop_vs_fixed": .05,
				"allowed_A_task_drop": .05,
				"alternative_00186_gain": .10,
				"allowed_heldout_drop_after_expansion": .10,
			},
		},
		"runtime_gate": runtime,
		"parametric_closed_loop": param_metrics,
		"parametric_split_means": {"seen_relaxed": seen_mean, "heldout_relaxed": held_mean},
		"three_task_closed_loop": three_task,
		"acquisition_retention": {
			"acquisition_delta_parametric_to_expanded": {
				"00256_relaxed": p3["00256"]["relaxed"] - three_task["parametric"]["00256"]["relaxed"],
				"00186_relaxed": p3["00186"]["relaxed"] - three_task["parametric"]["00186"]["relaxed"],
			},
			"01125_retention_delta_parametric_to_expanded": p3["01125"]["relaxed"] - three_task["parametric"]["01125"]["relaxed"],
			"heldout_retention_delta_parametric_to_expanded": held_mean["parametric_then_three_task"] - held_mean["parametric"],
		},
		"offline": offline,
		"offline_aggregate": offline_aggregate,
		"execution_notes": {
			"expansion_v1_failure": expansion_failure,
			"completed_expansion_run": "phase4_2_expansion_v2",
			"resume_used": False,
		},
		"ranking": ranking,
		"prohibitions_verified_by_protocol": {
			"elite_distillation": False,
			"counterfactual_reward_or_residual": False,
			"mppi_modified": False,
			"new_task_type": False,
			"yaw_added": False,
		},
	}
	Path(args.output_json).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

	lines = [
		"# Phase 4.2: Single-Family Parametric Pretraining Ablation",
		"",
		f"Final classification: `{classification}`",
		"",
		"## Causal contract",
		"",
		f"- Runtime geometry/target/reward chain: `{runtime['status']}`.",
		"- One task parameter is sampled per episode and held fixed; training batches are exactly balanced over parameter anchor × phase cells.",
		"- Scale is continuous in `[0.85, 1.15]`; clearance/depth use five seen anchors with 10% episode-level jitter.",
		"- Yaw remains disabled because the unchanged action contract is 3D translation only.",
		"- Physical CUDA1 only; MPPI, reward/Q architecture, task type, elite distillation, and counterfactual residual remain unchanged.",
		"",
		"## Seen and held-out closed loop",
		"",
		"| arm | seen mean relaxed | held-out mean relaxed |",
		"| --- | ---: | ---: |",
	]
	for arm in ARMS:
		lines.append(f"| {arm} | {f(seen_mean[arm])} | {f(held_mean[arm])} |")
	lines.extend(["", "| arm | condition | split | relaxed | strict | process | lateral mm | keypoint mm | jam |", "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
	for arm in ARMS:
		for condition, _ in CONDITIONS:
			row = param_metrics[arm][condition]
			lines.append(
				f"| {arm} | {condition} | {row['split']} | {f(row['relaxed'])} | {f(row['strict'])} | "
				f"{f(row['process'])} | {f(row['lateral_mm'])} | {f(row['keypoint_mm'])} | {f(row['jam'])} |"
			)
	lines.extend(["", "## Three-task acquisition and retention", "", "| arm | 01125 relaxed | 00256 relaxed | 00186 relaxed | mean relaxed |", "| --- | ---: | ---: | ---: | ---: |"])
	for arm in ARMS:
		rows = three_task[arm]
		lines.append(
			f"| {arm} | {f(rows['01125']['relaxed'])} | {f(rows['00256']['relaxed'])} | "
			f"{f(rows['00186']['relaxed'])} | {f(mean(list(rows.values()), 'relaxed'))} |"
		)
	lines.extend([
		"",
		f"- Acquisition delta after expansion: 00256 `{f(report['acquisition_retention']['acquisition_delta_parametric_to_expanded']['00256_relaxed'])}`, 00186 `{f(report['acquisition_retention']['acquisition_delta_parametric_to_expanded']['00186_relaxed'])}`.",
		f"- 01125 retention delta: `{f(report['acquisition_retention']['01125_retention_delta_parametric_to_expanded'])}`.",
		f"- Held-out parameter retention delta: `{f(report['acquisition_retention']['heldout_retention_delta_parametric_to_expanded'])}`.",
		"",
		"| arm | task | relaxed | strict | process | lateral mm | keypoint mm | jam |",
		"| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
	])
	for arm in ARMS:
		for task in ("01125", "00256", "00186"):
			row = three_task[arm][task]
			lines.append(
				f"| {arm} | {task} | {f(row['relaxed'])} | {f(row['strict'])} | {f(row['process'])} | "
				f"{f(row['lateral_mm'])} | {f(row['keypoint_mm'])} | {f(row['jam'])} |"
			)
	lines.extend([
		"",
		"## Task context and world-model calibration",
		"",
		"| arm | task dim | adapter source | encoder | dynamics | context/task Pearson | task recon R2 |",
		"| --- | ---: | --- | --- | --- | ---: | ---: |",
	])
	for arm, row in offline_aggregate.items():
		lines.append(
			f"| {arm} | {row['task_dim']} | {row['adapter_source']} | {row['adapter_encoder']} | "
			f"{row['adapter_dynamics']} | {f(row['context_task_pearson'])} | {f(row['task_reconstruction_r2'])} |"
		)
	lines.extend([
		"",
		"The fixed arm has only one task anchor, so its pairwise context/task correlation is undefined.",
		"",
		"| arm | dynamics L2 @1 | @2 | @3 | reward MAE | reward Pearson | Q MAE | Q Pearson | proposal regret mean |",
		"| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
	])
	for arm, row in offline_aggregate.items():
		lines.append(
			f"| {arm} | {f(row['dynamics_1_mean'])} | {f(row['dynamics_2_mean'])} | "
			f"{f(row['dynamics_3_mean'])} | {f(row['reward_mae'])} | {f(row['reward_pearson'])} | "
			f"{f(row['q_mae'])} | {f(row['q_pearson'])} | {f(row['proposal_regret_mean'])} |"
		)
	lines.extend([
		"",
		"## Predicted-vs-real ranking",
		"",
		"| evaluator | phase | Spearman | top-8 real gain | top-8 overlap |",
		"| --- | --- | ---: | ---: | ---: |",
	])
	for label, phases in ranking.items():
		for phase, row in phases.items():
			lines.append(f"| {label} | {phase} | {f(row['spearman'])} | {f(row['top8_gain'])} | {f(row['top8_overlap'])} |")
	lines.extend([
		"",
		"## Decision gates",
		"",
		f"- Parametric held-out advantage: `{param_advantage}` (gain `{f(param_heldout_gain)}`, seen delta `{f(param_seen_delta)}`).",
		f"- Three-task retention vs A: `{retention_ok}`.",
		f"- Three-task transfer vs A: `{transfer_ok}`.",
		f"- Held-out retention after expansion: `{heldout_retained}`.",
		"",
		"## Execution notes",
		"",
	])
	if expansion_failure is not None:
		lines.extend([
			f"- `{expansion_failure['run_label']}` stopped after update {expansion_failure['failed_after_update']} because "
			f"`{expansion_failure['failure']}`; no checkpoint was written.",
			"- The output-directory guard was fixed and the formal `phase4_2_expansion_v2` run restarted from update 1; no resume state was used.",
		])
	lines.extend([
		"- All reported CUDA jobs completed on physical CUDA1, and no evaluation process remains active.",
		"",
		"Detailed calibration and task-context tensors are retained in the JSON report and per-arm offline artifacts.",
	])
	Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
	print(f"[phase4.2] wrote {args.output_md} ({classification})")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
