#!/usr/bin/env python3
"""Aggregate Phase 4.0 training, offline, ranking, and closed-loop evidence."""

from __future__ import annotations

import argparse
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
VARIANTS = ("A", "B", "C", "D")
TASKS = ("01125", "00256", "00186")
PHASES = ("pre_contact", "contact", "insertion")
ALLOWED = (
	"FULL_RETRAIN_REQUIRED", "DOWNSTREAM_HEAD_RESET_SUFFICIENT",
	"PLANNER_CALIBRATION_IS_PRIMARY", "MIXED_OR_UNRESOLVED",
)


def resolve(value: str | Path) -> Path:
	path = Path(value).expanduser()
	return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def _load(path: Path):
	if not path.exists():
		raise FileNotFoundError(path)
	return json.loads(path.read_text(encoding="utf-8"))


def _mean(values):
	values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
	return float(sum(values) / len(values)) if values else math.nan


def _task_closed_loop(summary):
	result = OrderedDict()
	for item in summary.get("tasks", []):
		task = str(item["assembly_id"])
		result[task] = {
			"relaxed": float(item.get("episode_relaxed_success_stable", item.get("episode_success", math.nan))),
			"strict": float(item.get("episode_strict_success_stable", math.nan)),
			"process": float(item.get("episode_process_success_terminal", item.get("episode_process_success", math.nan))),
			"relaxed_process": float(item.get("episode_relaxed_process_success_terminal", math.nan)),
			"lateral_mm": 1000.0 * float(item.get("episode_lateral_error", math.nan)),
			"keypoint_mm": 1000.0 * float(item.get("episode_keypoint_error", math.nan)),
			"jam": float(item.get("episode_jam", math.nan)),
			"depth_fraction": float(item.get("episode_depth_fraction", math.nan)),
			"episodes": int(item.get("episodes", 0)),
		}
	missing = [task for task in TASKS if task not in result]
	if missing:
		raise RuntimeError(f"Closed-loop summary missing tasks: {missing}")
	return result


def _family(closed):
	return {
		"mean_relaxed": _mean(item["relaxed"] for item in closed.values()),
		"min_relaxed": min(item["relaxed"] for item in closed.values()),
		"mean_strict": _mean(item["strict"] for item in closed.values()),
		"mean_process": _mean(item["process"] for item in closed.values()),
		"mean_lateral_mm": _mean(item["lateral_mm"] for item in closed.values()),
		"mean_keypoint_mm": _mean(item["keypoint_mm"] for item in closed.values()),
		"mean_jam": _mean(item["jam"] for item in closed.values()),
	}


def _ranking_summary(report):
	phase_results = report["prediction_reality"]["phase_results"]
	result = OrderedDict()
	for phase in PHASES:
		item = phase_results[phase]
		corr = item["ranking_correlations"]["total"]["vs_actual_return_5"]
		top = item["predicted_total_topk_realized"]
		result[phase] = {
			"pearson": float(corr["pearson_mean"]), "spearman": float(corr["spearman_mean"]),
			"kendall": float(corr["kendall_tau_mean"]),
			"top8_real_gain": float(top["predicted_topk_gain_vs_all"]),
			"top8_overlap": float(top["topk_overlap_with_actual"]),
			"top8_regret_to_oracle": float(top["predicted_topk_regret_to_oracle"]),
		}
	return result


def _causal_recovery(variant, baseline):
	closed, family = variant["closed_loop"], variant["family"]
	base_closed, base_family = baseline["closed_loop"], baseline["family"]
	deltas = {
		"mean_relaxed": family["mean_relaxed"] - base_family["mean_relaxed"],
		"00186_relaxed": closed["00186"]["relaxed"] - base_closed["00186"]["relaxed"],
		"min_task_relaxed": min(closed[task]["relaxed"] - base_closed[task]["relaxed"] for task in TASKS),
		"mean_strict_process": .5 * (
			family["mean_strict"] + family["mean_process"] - base_family["mean_strict"] - base_family["mean_process"]
		),
		"mean_jam": family["mean_jam"] - base_family["mean_jam"],
		"mean_lateral_ratio": family["mean_lateral_mm"] / max(base_family["mean_lateral_mm"], 1e-8),
		"mean_keypoint_ratio": family["mean_keypoint_mm"] / max(base_family["mean_keypoint_mm"], 1e-8),
	}
	checks = {
		"mean_relaxed_gain_ge_0p10": deltas["mean_relaxed"] >= .10,
		"00186_relaxed_gain_ge_0p10": deltas["00186_relaxed"] >= .10,
		"no_task_relaxed_drop_gt_0p10": deltas["min_task_relaxed"] >= -.10,
		"strict_process_not_worse": deltas["mean_strict_process"] >= -.025,
		"jam_not_worse_gt_0p05": deltas["mean_jam"] <= .05,
		"lateral_not_worse_gt_10pct": deltas["mean_lateral_ratio"] <= 1.10,
		"keypoint_not_worse_gt_10pct": deltas["mean_keypoint_ratio"] <= 1.10,
	}
	return {"passed": all(checks.values()), "deltas": deltas, "checks": checks}


def _classification(evidence):
	baseline = evidence["A"]
	recovery = {variant: _causal_recovery(evidence[variant], baseline) for variant in ("B", "C", "D")}
	closed_score = {
		variant: evidence[variant]["family"]["mean_relaxed"]
		+ .5 * evidence[variant]["family"]["mean_strict"]
		+ .5 * evidence[variant]["family"]["mean_process"]
		for variant in VARIANTS
	}
	reset_best = max(("C", "D"), key=lambda name: closed_score[name])
	if recovery["B"]["passed"] and not recovery["C"]["passed"] and not recovery["D"]["passed"]:
		classification = "FULL_RETRAIN_REQUIRED"
		reason = "只有全模型从零联合训练通过预先声明的闭环 recovery gate。"
	elif recovery[reset_best]["passed"] and closed_score[reset_best] >= closed_score["B"] - .05:
		classification = "DOWNSTREAM_HEAD_RESET_SUFFICIENT"
		reason = f"Reset 臂 {reset_best} 通过 recovery gate，且 composite success score 与全重训相差不超过 0.05。"
	elif not any(item["passed"] for item in recovery.values()):
		poor = []
		for variant in ("B", "C", "D"):
			for phase in ("contact", "insertion"):
				item = evidence[variant]["ranking"][phase]
				poor.append(item["spearman"] <= .10 or item["top8_real_gain"] <= 0.0)
		if sum(poor) >= 4:
			classification = "PLANNER_CALIBRATION_IS_PRIMARY"
			reason = "没有 reset/retrain 臂恢复闭环，且 6 个训练臂 contact/insertion ranking cell 中至少 4 个仍不可预测。"
		else:
			classification = "MIXED_OR_UNRESOLVED"
			reason = "没有训练臂通过 recovery gate，但 planner-ranking failure 也未按预先声明的阈值在训练臂中占主导。"
	else:
		classification = "MIXED_OR_UNRESOLVED"
		reason = "多个初始化机制同时通过，或领先幅度不足以支持唯一因果归因。"
	return {
		"classification": classification, "reason": reason, "causal_recovery": recovery,
		"closed_loop_composite_score": closed_score, "reset_best": reset_best,
		"thresholds": {
			"mean_relaxed_gain": .10, "00186_relaxed_gain": .10, "max_per_task_relaxed_drop": .10,
			"max_jam_increase": .05, "max_lateral_or_keypoint_ratio": 1.10,
			"planner_poor_spearman": .10, "planner_poor_top8_gain": 0.0,
		},
	}


def _fmt(value, digits=3):
	try:
		return f"{float(value):.{digits}f}"
	except Exception:
		return "NA"


def _markdown(report):
	classification = report["classification"]
	lines = [
		"# SRSA Phase 4.0 Multi-Task-From-Start Causal Ablation",
		"",
		f"Final classification: `{classification['classification']}`",
		"",
		"## 结论",
		"",
		f"- {classification['reason']}",
		"- 比较口径为 00186 首次进入三任务训练后的等量 optimizer exposure：Phase 3.1 的 6,086 次加 Phase 3.2 的 2,361 次，共 8,447 次。",
		"- B/C/D 使用相同 episode-disjoint replay split、相同采样 seed；每个 update 的 1,024 样本都覆盖 9 个 task×phase cell，cell 计数最多相差 1，余数轮转。",
		"- C/D 的保留模块仅用于初始化，联合训练时没有冻结参数。A 始终只读。",
		"",
		"## Closed-Loop Results",
		"",
		"| Variant | Task | Relaxed | Strict | Process | Lateral mm | Keypoint mm | Jam |",
		"| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
	]
	for variant, item in report["variants"].items():
		for task, values in item["closed_loop"].items():
			lines.append(
				f"| `{variant}` | `{task}` | {_fmt(values['relaxed'])} | {_fmt(values['strict'])} | "
				f"{_fmt(values['process'])} | {_fmt(values['lateral_mm'])} | {_fmt(values['keypoint_mm'])} | {_fmt(values['jam'])} |"
			)
	lines.extend([
		"",
		"| Variant | Mean relaxed | Min relaxed | Mean strict | Mean process | Mean jam | Recovery gate |",
		"| --- | ---: | ---: | ---: | ---: | ---: | --- |",
	])
	for variant, item in report["variants"].items():
		gate = "baseline" if variant == "A" else str(classification["causal_recovery"][variant]["passed"])
		family = item["family"]
		lines.append(
			f"| `{variant}` | {_fmt(family['mean_relaxed'])} | {_fmt(family['min_relaxed'])} | "
			f"{_fmt(family['mean_strict'])} | {_fmt(family['mean_process'])} | {_fmt(family['mean_jam'])} | `{gate}` |"
		)
	lines.extend([
		"",
		"## Predicted-vs-Real Candidate Ranking",
		"",
		"每个 phase 使用 4 个 cloned base states、每 state 64 个候选（3 policy + 61 Gaussian）、horizon=3、top-8；真实收益为 3 candidate steps 加 2 controller continuation steps。",
		"",
		"| Variant | Phase | Spearman | Pearson | Top-8 real gain | Top-8 overlap | Oracle regret |",
		"| --- | --- | ---: | ---: | ---: | ---: | ---: |",
	])
	for variant, item in report["variants"].items():
		for phase, values in item["ranking"].items():
			lines.append(
				f"| `{variant}` | `{phase}` | {_fmt(values['spearman'])} | {_fmt(values['pearson'])} | "
				f"{_fmt(values['top8_real_gain'])} | {_fmt(values['top8_overlap'])} | {_fmt(values['top8_regret_to_oracle'])} |"
			)
	lines.extend([
		"",
		"## Offline Model Calibration",
		"",
		"| Variant | Dyn L2 h1 | Dyn L2 h3 | Reward MAE | Reward r | Q MAE | Q r | Proposal regret | Context recon R2 |",
		"| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
	])
	for variant, item in report["variants"].items():
		offline = item["offline"]
		world = offline["world_model"]["aggregate"]
		lines.append(
			f"| `{variant}` | {_fmt(world['multi_step_dynamics_latent_l2']['1']['mean'])} | "
			f"{_fmt(world['multi_step_dynamics_latent_l2']['3']['mean'])} | "
			f"{_fmt(world['reward_calibration']['mae'])} | {_fmt(world['reward_calibration']['pearson'])} | "
			f"{_fmt(world['q_calibration_to_replay_return']['mae'])} | {_fmt(world['q_calibration_to_replay_return']['pearson'])} | "
			f"{_fmt(offline['proposal_regret']['summary']['mean'])} | "
			f"{_fmt(offline['task_context_structure'].get('task_reconstruction_r2'))} |"
		)
	lines.extend([
		"",
		"分 task×phase 的 multi-step dynamics、reward/Q calibration 明细保存在配套 JSON 的 `variants.*.offline.world_model.by_task_phase`。task_context 向量和三对 pairwise L2 保存在 `task_context_structure`。",
		"",
		"## Initialization And Causal Contract",
		"",
		"- A: 当前 staged Phase 3.2 checkpoint，不继续训练。",
		"- B: 全模型同架构随机初始化，从第一个 update 开始三任务联合训练。",
		"- C: 保留 task/obs/contact encoder 与 encoder-side task-context adapter；重置 dynamics/reward/Q/pi。",
		"- D: 在 C 基础上保留 dynamics 与 dynamics-side task-context adapter；重置 reward/Q/pi。",
		"- 没有新任务、elite distillation、counterfactual reward/residual；MPPI 配置和候选合同未修改。",
		"- 全部训练和评估固定在 physical CUDA1（`CUDA_VISIBLE_DEVICES=1`，logical `cuda:0`）。",
		"",
		"## Decision Rule",
		"",
		"- `FULL_RETRAIN_REQUIRED`: 仅 B 通过 closed-loop recovery gate。",
		"- `DOWNSTREAM_HEAD_RESET_SUFFICIENT`: C/D 至少一项通过，且 composite score 距 B 不超过 0.05。",
		"- `PLANNER_CALIBRATION_IS_PRIMARY`: 无训练臂通过，且 6 个 trained contact/insertion ranking cell 至少 4 个仍满足 Spearman≤0.10 或 top-8 real gain≤0。",
		"- 其他情况为 `MIXED_OR_UNRESOLVED`。",
	])
	return "\n".join(lines) + "\n"


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--root", default="reports/phase4_0_multitask_origin")
	parser.add_argument("--output-json", default="reports/phase4_0_multitask_origin_ablation.json")
	parser.add_argument("--output-md", default="reports/phase4_0_multitask_origin_ablation.md")
	parser.add_argument("--dry-run", action="store_true")
	args = parser.parse_args()
	root = resolve(args.root)
	paths = OrderedDict()
	for variant in VARIANTS:
		paths[variant] = {
			"offline": root / "offline" / f"variant_{variant}.json",
			"ranking": root / "ranking" / f"variant_{variant}.json",
			"closed_loop": root / "closed_loop" / variant / "batch_eval_summary.json",
		}
		if variant != "A":
			paths[variant]["training"] = root / "checkpoints" / f"variant_{variant}_train.json"
	for variant_paths in paths.values():
		for path in variant_paths.values():
			if not path.exists():
				raise FileNotFoundError(path)
	if args.dry_run:
		print("PASS dry-run: all Phase 4.0 artifacts exist")
		return 0

	evidence = OrderedDict()
	for variant, variant_paths in paths.items():
		offline = _load(variant_paths["offline"])
		ranking_raw = _load(variant_paths["ranking"])
		closed_raw = _load(variant_paths["closed_loop"])
		closed = _task_closed_loop(closed_raw)
		evidence[variant] = {
			"offline": offline, "ranking": _ranking_summary(ranking_raw),
			"closed_loop": closed, "family": _family(closed),
			"training": _load(variant_paths["training"]) if "training" in variant_paths else None,
			"artifact_paths": {key: str(value) for key, value in variant_paths.items()},
		}
	classification = _classification(evidence)
	if classification["classification"] not in ALLOWED:
		raise RuntimeError(classification)
	report = {
		"status": "PASS" if classification["classification"] != "MIXED_OR_UNRESOLVED" else "WARNING",
		"phase": "4.0", "parent_phase3_11_classification": "MIXED_OR_UNRESOLVED",
		"classification": classification, "variants": evidence,
		"prohibitions_observed": {
			"tasks": list(TASKS), "new_task_added": False, "elite_distillation": False,
			"counterfactual_reward_or_residual": False, "mppi_modified": False,
		},
	}
	json_path, md_path = resolve(args.output_json), resolve(args.output_md)
	json_path.parent.mkdir(parents=True, exist_ok=True)
	md_path.parent.mkdir(parents=True, exist_ok=True)
	json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
	md_path.write_text(_markdown(report), encoding="utf-8")
	print(report["status"])
	print(f"Final classification: {classification['classification']}")
	print(f"Wrote {md_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
