#!/usr/bin/env python3
"""Build the Phase 3.2 acquisition rescue and diagnosis summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = REPO_ROOT / "reports" / "phase3_three_task_pilot"
RUN_ROOT = (
	REPO_ROOT / "logs" / "isaaclab-srsa-assembly" / "1" /
	"srsa_axial_online_family_taskctx_repair_01125_00256_00186_rescue"
)


def _resolve(value: str | Path) -> Path:
	path = Path(value).expanduser()
	return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def _load(value: str | Path) -> dict[str, Any]:
	path = _resolve(value)
	if not path.exists():
		raise FileNotFoundError(path)
	return json.loads(path.read_text(encoding="utf-8"))


def _rows(summary: dict[str, Any]) -> dict[str, dict[str, float]]:
	result = {}
	for item in summary.get("tasks") or []:
		task_id = str(item["assembly_id"]).zfill(5)
		result[task_id] = {
			"relaxed": float(item.get("episode_success", 0.0)),
			"strict": float(item.get("episode_strict_success_stable", 0.0)),
			"process": float(item.get("episode_process_success", 0.0)),
			"reward": float(item.get("episode_reward", 0.0)),
			"lateral_mm": 1000.0 * float(item.get("episode_lateral_error", 0.0)),
			"keypoint_mm": 1000.0 * float(item.get("episode_keypoint_error", 0.0)),
			"jam": float(item.get("episode_jam", 0.0)),
		}
	return result


def _latest_mix(path: str | Path) -> dict[str, Any]:
	latest = None
	with _resolve(path).open("r", encoding="utf-8") as f:
		for line in f:
			if not line.strip():
				continue
			item = json.loads(line)
			if item.get("category") == "train" and "online_family_batch_num_tasks" in item:
				latest = item
	if latest is None:
		raise RuntimeError(f"No mixed replay metrics in {path}")
	return latest


def _single_eval(path: str | Path) -> dict[str, float]:
	return _rows(_load(path))["00186"]


def build(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
	phase31 = _load(args.phase31_summary)
	checkpoints = {
		0: phase31["task_metrics"],
		25088: _rows(_load(args.eval_25k)),
		50176: _rows(_load(args.eval_50k)),
		75264: _rows(_load(args.eval_75k)),
		100352: _rows(_load(args.eval_100k)),
	}
	context_50 = _load(args.context_50k)
	context_100 = _load(args.context_100k)
	mix_50 = _latest_mix(args.metrics_50k)
	mix_100 = _latest_mix(args.metrics_100k)
	mix_audit = _load(args.mix_audit)
	diagnosis = _load(args.model_diagnosis)

	baseline = checkpoints[0]["00186"]
	if "relaxed_success" in baseline:
		baseline = {
			"relaxed": float(baseline["relaxed_success"]),
			"strict": float(baseline["strict_success"]),
			"process": float(baseline["process_success"]),
			"reward": float(baseline["reward"]),
			"lateral_mm": float(baseline["lateral_error_mm"]),
			"keypoint_mm": float(baseline["keypoint_error_mm"]),
			"jam": float(baseline["jam"]),
		}
		checkpoints[0] = {
			key: {
				"relaxed": float(value["relaxed_success"]),
				"strict": float(value["strict_success"]),
				"process": float(value["process_success"]),
				"reward": float(value["reward"]),
				"lateral_mm": float(value["lateral_error_mm"]),
				"keypoint_mm": float(value["keypoint_error_mm"]),
				"jam": float(value["jam"]),
			}
			for key, value in checkpoints[0].items()
		}
	first_50 = checkpoints[50176]["00186"]
	final_100 = checkpoints[100352]["00186"]
	first_gate = {
		"00186_relaxed_ge_0p35": first_50["relaxed"] >= 0.35,
		"00186_jam_below_0p70": first_50["jam"] < 0.70,
		"00186_lateral_improved": first_50["lateral_mm"] <= 0.5 * baseline["lateral_mm"],
		"00186_keypoint_improved": first_50["keypoint_mm"] <= 0.5 * baseline["keypoint_mm"],
		"old_tasks_relaxed_ge_0p85": all(checkpoints[50176][task]["relaxed"] >= 0.85 for task in ("01125", "00256")),
		"task_context_not_collapsed": not bool(context_50["context_collapse"]),
	}
	first_gate_pass = all(first_gate.values())
	second_target_pass = final_100["relaxed"] >= 0.75
	trajectory = [checkpoints[step]["00186"]["relaxed"] for step in sorted(checkpoints)]
	monotonic = all(right >= left for left, right in zip(trajectory, trajectory[1:]))

	def mix_summary(item: dict[str, Any]) -> dict[str, Any]:
		return {
			"counts": {
				task: int(item.get(f"online_family_batch_task_count_{task}", 0))
				for task in ("00186", "01125", "00256")
			},
			"hash_counts": {
				key.removeprefix("online_family_batch_task_hash_count_"): int(value)
				for key, value in item.items()
				if key.startswith("online_family_batch_task_hash_count_")
			},
			"condition_entropy_norm": float(item["online_family_batch_condition_entropy_norm"]),
		}

	region_paths = {
		"historical_standalone_default": args.standalone_default,
		"direct_default": args.direct_default,
		"direct_easy": args.direct_easy,
		"direct_hard": args.direct_hard,
		"multitask_easy": args.multitask_easy,
		"multitask_hard": args.multitask_hard,
	}
	regions = {name: _single_eval(path) for name, path in region_paths.items()}
	regions["multitask_default"] = first_50
	pairwise = diagnosis["pairwise"]["direct_finetune_vs_multitask_rescue_best"]
	direct_model = diagnosis["models"]["direct_finetune"]
	multitask_model = diagnosis["models"]["multitask_rescue_best"]

	status = "STOP_AND_DIAGNOSE_ACQUISITION_NON_MONOTONIC"
	decision = "STANDALONE_VS_MULTITASK_ACQUISITION_DIAGNOSIS_BEFORE_CONSOLIDATION"
	report = {
		"status": status,
		"decision": decision,
		"first_50k_gate": first_gate,
		"first_50k_gate_pass": first_gate_pass,
		"second_50k_target_relaxed_ge_0p75": second_target_pass,
		"acquisition_monotonic": monotonic,
		"timeline": {str(step): value for step, value in checkpoints.items()},
		"replay_50k": mix_summary(mix_50),
		"replay_100k": mix_summary(mix_100),
		"task_mix_audit_status": mix_audit.get("status"),
		"context_50k": {
			"pairwise_l2": context_50["task_context_l2"],
			"reconstruction_r2": context_50["task_reconstruction_r2"],
			"collapsed": context_50["context_collapse"],
		},
		"context_100k": {
			"pairwise_l2": context_100["task_context_l2"],
			"reconstruction_r2": context_100["task_reconstruction_r2"],
			"collapsed": context_100["context_collapse"],
		},
		"diagnosis": {
			"historical_standalone_action_dim": diagnosis["checkpoint_compatibility"]["standalone"]["action_dim"],
			"current_action_dim": diagnosis["checkpoint_compatibility"]["multitask_rescue_best"]["action_dim"],
			"regions": regions,
			"direct_vs_multitask": pairwise,
			"direct_reward_calibration": direct_model["reward_calibration"],
			"multitask_reward_calibration": multitask_model["reward_calibration"],
			"direct_q_calibration": direct_model["q_calibration"],
			"multitask_q_calibration": multitask_model["q_calibration"],
		},
	}

	lines = [
		"# SRSA Phase 3.2 00186 Acquisition Rescue Summary",
		"",
		"本报告汇总两个连续 50k current-heavy rescue、每约 25k 的三任务评估、replay/context 审计，以及 standalone/direct/multitask acquisition diagnosis。未修改模型、sampler、reward、Q、policy 或 MPPI。",
		"",
		f"Status: `{status}`",
		"",
		"## 结论回答",
		"",
		"1. `00186` 是否仍有明确 acquisition 上升趋势：`否`。首个 50k 上升，但第二个 50k 回落，整体非单调。",
		f"2. 高 jam 和大 lateral/keypoint error 是否改善：首个 50k `是`，jam `{baseline['jam']:.2f}->{first_50['jam']:.2f}`，lateral `{baseline['lateral_mm']:.2f}->{first_50['lateral_mm']:.2f} mm`，keypoint `{baseline['keypoint_mm']:.2f}->{first_50['keypoint_mm']:.2f} mm`；第二段随后部分恶化。",
		f"3. `01125` strict/process 是否继续退化：`否`。100k 末为 `{checkpoints[100352]['01125']['strict']:.2f}/{checkpoints[100352]['01125']['process']:.2f}`，高于 Phase 3.1 的 `0.10/0.15`。",
		f"4. 是否可以继续到 `00186 relaxed>=0.75`：`否`。100k 末 relaxed=`{final_100['relaxed']:.2f}`，且曲线非单调。",
		"5. 下一步：`standalone-vs-multitask acquisition diagnosis`，不进入 consolidation，不继续追加训练步数。",
		"",
		"## Acquisition 时间线",
		"",
		"| Rescue steps | Task | Relaxed | Strict | Process | Reward | Lateral mm | Keypoint mm | Jam |",
		"| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
	]
	for step in sorted(checkpoints):
		for task in ("01125", "00256", "00186"):
			item = checkpoints[step][task]
			lines.append(
				f"| {step:,} | `{task}` | {item['relaxed']:.2f} | {item['strict']:.2f} | {item['process']:.2f} | "
				f"{item['reward']:.2f} | {item['lateral_mm']:.2f} | {item['keypoint_mm']:.2f} | {item['jam']:.2f} |"
			)
	lines.extend([
		"",
		"## 50k Gate",
		"",
	])
	for name, passed in first_gate.items():
		lines.append(f"- `{name}`: `{'PASS' if passed else 'FAIL'}`")
	lines.extend([
		f"- 首个 50k gate overall: `{'PASS' if first_gate_pass else 'FAIL'}`",
		f"- 第二个 50k `00186 relaxed>=0.75`: `{'PASS' if second_target_pass else 'FAIL'}`",
		"",
		"## Replay 与 Task Context",
		"",
		f"- 50k replay: `{report['replay_50k']}`",
		f"- 100k replay: `{report['replay_100k']}`",
		f"- independent task mix audit: `{report['task_mix_audit_status']}`",
		f"- 50k context pairwise L2: `{report['context_50k']['pairwise_l2']}`",
		f"- 50k reconstruction R²: `{report['context_50k']['reconstruction_r2']:.6f}`",
		f"- 100k context pairwise L2: `{report['context_100k']['pairwise_l2']}`",
		f"- 100k reconstruction R²: `{report['context_100k']['reconstruction_r2']:.6f}`",
		"",
		"## Acquisition Diagnosis",
		"",
		f"- 历史 standalone action dim=`{report['diagnosis']['historical_standalone_action_dim']}`，当前三任务 action dim=`{report['diagnosis']['current_action_dim']}`；历史 `1.0` checkpoint 在当前 exact profile 下 relaxed=`{regions['historical_standalone_default']['relaxed']:.2f}`，不可作为直接可比上限。",
		f"- 可比 direct checkpoint 当前 profile relaxed=`{regions['direct_default']['relaxed']:.2f}`，rescue-best relaxed=`{regions['multitask_default']['relaxed']:.2f}`。",
		f"- direct vs multitask policy action mean L2=`{pairwise['policy_action_mean_l2']['mean']:.3f}`。",
		f"- direct vs multitask fixed-candidate MPPI selected action L2=`{pairwise['mppi_selected_action_l2']['mean']:.3f}`。",
		f"- reward prediction abs delta=`{pairwise['reward_prediction_abs_delta']['mean']:.3f}`，Q prediction abs delta=`{pairwise['q_prediction_abs_delta']['mean']:.3f}`。",
		"",
		"### Initial-State / Jam Region",
		"",
		"| Model/profile | Relaxed | Reward | Lateral mm | Keypoint mm | Jam |",
		"| --- | ---: | ---: | ---: | ---: | ---: |",
	])
	for name in ("historical_standalone_default", "direct_default", "direct_easy", "direct_hard", "multitask_default", "multitask_easy", "multitask_hard"):
		item = regions[name]
		lines.append(
			f"| `{name}` | {item['relaxed']:.2f} | {item['reward']:.2f} | {item['lateral_mm']:.2f} | "
			f"{item['keypoint_mm']:.2f} | {item['jam']:.2f} |"
		)
	lines.extend([
		"",
		"## 决策",
		"",
		"Phase 3.2 证明 75/12.5/12.5 current-heavy replay 可以短暂改善 `00186`，但不能稳定推进到 `0.75`。representation、task hash 和旧任务 relaxed retention 不是当前主阻塞；下一步应围绕可比 direct checkpoint 与 rescue-best 的动作选择、MPPI 选中动作及 initial-state/contact success region 做 acquisition diagnosis，暂不进行 1/3 consolidation。",
	])
	return report, "\n".join(lines) + "\n"


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--phase31-summary", default=str(REPORT_DIR / "phase3_1_summary.json"))
	parser.add_argument("--eval-25k", default=str(REPORT_DIR / "phase3_2_eval_25k" / "batch_eval_summary.json"))
	parser.add_argument("--eval-50k", default=str(RUN_ROOT / "20260713_phase3_2_rescue_00186_launcher" / "family_eval_after_00186" / "batch_eval_summary.json"))
	parser.add_argument("--eval-75k", default=str(REPORT_DIR / "phase3_2_eval_75k" / "batch_eval_summary.json"))
	parser.add_argument("--eval-100k", default=str(RUN_ROOT / "20260713_phase3_2_rescue_00186_part2_launcher" / "family_eval_after_00186" / "batch_eval_summary.json"))
	parser.add_argument("--metrics-50k", default=str(RUN_ROOT / "20260713_phase3_2_rescue_00186_stage-3_asm-00186" / "metrics.jsonl"))
	parser.add_argument("--metrics-100k", default=str(RUN_ROOT / "20260713_phase3_2_rescue_00186_part2_stage-3_asm-00186" / "metrics.jsonl"))
	parser.add_argument("--context-50k", default=str(REPORT_DIR / "phase3_2_context_50k.json"))
	parser.add_argument("--context-100k", default=str(REPORT_DIR / "phase3_2_context_100k.json"))
	parser.add_argument("--mix-audit", default=str(REPORT_DIR / "phase3_2_task_consistency" / "online_family_sample_mix_report.json"))
	parser.add_argument("--model-diagnosis", default=str(REPORT_DIR / "phase3_2_diagnosis" / "standalone_vs_multitask_diagnosis.json"))
	for name in ("standalone_default", "direct_default", "direct_easy", "direct_hard", "multitask_easy", "multitask_hard"):
		default_dir = "direct_finetune_default" if name == "direct_default" else name
		parser.add_argument(f"--{name.replace('_', '-')}", default=str(REPORT_DIR / "phase3_2_diagnosis" / default_dir / "batch_eval_summary.json"))
	parser.add_argument("--output-md", default=str(REPORT_DIR / "phase3_2_acquisition_rescue_summary.md"))
	parser.add_argument("--output-json", default=str(REPORT_DIR / "phase3_2_acquisition_rescue_summary.json"))
	args = parser.parse_args()
	report, markdown = build(args)
	json_path = _resolve(args.output_json)
	md_path = _resolve(args.output_md)
	json_path.parent.mkdir(parents=True, exist_ok=True)
	json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
	md_path.write_text(markdown, encoding="utf-8")
	print(f"{report['status']} wrote {md_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
