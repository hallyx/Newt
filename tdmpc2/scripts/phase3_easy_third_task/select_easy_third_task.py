#!/usr/bin/env python3
"""Read-only Phase 3 candidate selector for an easy SRSA third task.

The script does not launch Isaac, train, or mutate replay/checkpoints. It
combines mesh-derived task vectors with existing best.json evidence from prior
runs and writes a ranked candidate report.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


SRSA_BASE_PLUG_DIAMETER = 0.007986
DEFAULT_EXISTING_TASKS = ("01125", "00256")
DEFAULT_CANDIDATES = ("00186", "00004", "00014", "00062", "00271")
REPORT_DIR = Path("reports/phase3_easy_third_task")


def _as_float(value: Any, default: float | None = None) -> float:
    if value is None or str(value).strip() == "":
        if default is None:
            raise ValueError("missing float value")
        return float(default)
    return float(value)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_mesh_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return {str(row["assembly_id"]).zfill(5): row for row in csv.DictReader(f)}


def _load_template(path: Path, template_id: int) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _load_json(path)
    templates = manifest.get("parameter_templates") or manifest.get("templates") or []
    for template in templates:
        if int(template.get("template_id", template.get("task_id", -1))) == int(template_id):
            return manifest, template
    raise ValueError(f"template_id={template_id} not found in {path}")


def _mesh_col(mesh_cfg: dict[str, Any], key: str, default: str) -> str:
    return str(mesh_cfg.get(key, default))


def _diametral_clearance(value: float, column: str, mode: str) -> float:
    mode = str(mode or "auto").strip().lower()
    if mode == "radial":
        return 2.0 * value
    if mode == "diametral":
        return value
    column_l = str(column or "").lower()
    if "radial" in column_l or "surface_dist" in column_l:
        return 2.0 * value
    return value


def _template_multiplier(template: dict[str, Any], key: str, index: int, default: float) -> float:
    for name in (key, f"{key}_multiplier", f"{key}_gamma"):
        if template.get(name) is not None:
            return float(template[name])
    pair = template.get("clearance_depth_template") or template.get("template") or template.get("multipliers")
    if isinstance(pair, str):
        parts = [part.strip() for part in re.split(r"[:,]", pair) if part.strip()]
        if len(parts) == 2:
            return float(parts[index])
    if isinstance(pair, (list, tuple)) and len(pair) == 2:
        return float(pair[index])
    return float(default)


def _task_entry(
    assembly_id: str,
    rows: dict[str, dict[str, str]],
    manifest: dict[str, Any],
    template: dict[str, Any],
    *,
    reference_anchor: str,
) -> dict[str, Any]:
    mesh_cfg = manifest.get("mesh_geometry", {}) or {}
    row = rows[assembly_id]
    anchor = rows[reference_anchor]

    plug_col = _mesh_col(mesh_cfg, "plug_diameter_column", "plug_xy_bbox_max")
    hole_col = _mesh_col(mesh_cfg, "hole_diameter_column", "socket_xy_bbox_max")
    clearance_col = _mesh_col(mesh_cfg, "clearance_base_column", "plug_to_socket_surface_dist_p05")
    clearance_mode = str(mesh_cfg.get("clearance_mode", "auto"))
    depth_col = _mesh_col(mesh_cfg, "depth_base_column", "plug_bbox_z")
    reference_radius_col = _mesh_col(mesh_cfg, "reference_radius_column", "plug_xy_radius_p95_from_centroid")
    reference_depth_col = _mesh_col(mesh_cfg, "reference_depth_column", depth_col)

    plug_diameter = _as_float(row[plug_col])
    mesh_hole_diameter = _as_float(row[hole_col])
    raw_clearance = _as_float(row[clearance_col])
    clearance_base = max(0.0, _diametral_clearance(raw_clearance, clearance_col, clearance_mode))
    depth_base = max(0.0, _as_float(row[depth_col]))
    reference_radius = max(1.0e-8, _as_float(anchor[reference_radius_col], 0.5 * _as_float(anchor[plug_col])))
    reference_depth = max(1.0e-8, _as_float(anchor[reference_depth_col], _as_float(anchor[depth_col])))

    clearance_multiplier = _template_multiplier(template, "clearance", 0, 1.0)
    depth_multiplier = _template_multiplier(template, "depth", 1, 1.0)
    diametral_clearance = clearance_base * clearance_multiplier
    radial_clearance = 0.5 * diametral_clearance
    target_depth = depth_base * depth_multiplier
    male_radius = max(0.5 * plug_diameter, 1.0e-8)
    scale_ratio = plug_diameter / max(SRSA_BASE_PLUG_DIAMETER, 1.0e-8)
    yaw_requirement = bool(template.get("yaw_requirement", False))
    task_type_id = int(template.get("task_type_id", 0))

    task_vec = [
        float(task_type_id),
        float(math.log(max(scale_ratio, 1.0e-8))),
        float(radial_clearance) / reference_radius,
        float(radial_clearance) / male_radius,
        float(target_depth) / reference_depth,
        1.0 if yaw_requirement else 0.0,
    ]
    plug_obj = row.get("plug_obj", "")
    socket_obj = row.get("socket_obj", "")
    return {
        "assembly_id": assembly_id,
        "template_id": int(template.get("template_id", template.get("task_id", 0))),
        "task_vec_6": task_vec,
        "plug_obj": plug_obj,
        "socket_obj": socket_obj,
        "plug_obj_exists": bool(plug_obj and os.path.exists(plug_obj)),
        "socket_obj_exists": bool(socket_obj and os.path.exists(socket_obj)),
        "plug_diameter": plug_diameter,
        "mesh_hole_diameter": mesh_hole_diameter,
        "hole_diameter_task": plug_diameter + diametral_clearance,
        "radial_clearance": radial_clearance,
        "diametral_clearance": diametral_clearance,
        "target_depth": target_depth,
        "reference_radius": reference_radius,
        "reference_depth": reference_depth,
        "clearance_multiplier": clearance_multiplier,
        "depth_multiplier": depth_multiplier,
        "success_pos_tol": float(template.get("success_pos_tol", 0.015)),
        "mesh_columns": {
            "plug_diameter_column": plug_col,
            "hole_diameter_column": hole_col,
            "clearance_base_column": clearance_col,
            "clearance_mode": clearance_mode,
            "depth_base_column": depth_col,
            "reference_radius_column": reference_radius_col,
            "reference_depth_column": reference_depth_col,
        },
    }


def _l2(a: list[float], b: list[float]) -> float:
    return float(math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b))))


def _best_json_records(log_root: Path) -> dict[str, list[dict[str, Any]]]:
    records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fp_text in glob.glob(str(log_root / "**/models/best.json"), recursive=True):
        fp = Path(fp_text)
        match = re.search(r"asm-(\d{5})", fp_text)
        if not match:
            continue
        assembly_id = match.group(1)
        try:
            payload = _load_json(fp)
        except Exception as exc:  # pragma: no cover - report only.
            records[assembly_id].append({"source_fp": str(fp), "error": str(exc)})
            continue
        run_type = "other"
        p = fp_text
        if "srsa_axial_direct_finetune_from_01125" in p:
            run_type = "direct_finetune_from_01125"
        elif "srsa_axial_finetune_from_01125" in p:
            run_type = "finetune_from_01125"
        elif "srsa_axial_continual_from_01125" in p and "taskvec_smoke" not in p:
            run_type = "continual_from_01125"
        elif "/srsa_axial_online/" in p:
            run_type = "standalone_online"
        elif "online_family" in p:
            run_type = "online_family"
        records[assembly_id].append(
            {
                "source_fp": str(fp),
                "run_type": run_type,
                "metric": payload.get("metric"),
                "value": payload.get("value"),
                "step": payload.get("step"),
                "checkpoint": payload.get("checkpoint"),
            }
        )
    return dict(records)


def _max_by_run_type(records: list[dict[str, Any]], run_types: set[str]) -> dict[str, Any] | None:
    vals = [r for r in records if r.get("run_type") in run_types and isinstance(r.get("value"), (int, float))]
    if not vals:
        return None
    return max(vals, key=lambda item: float(item["value"]))


def _known_evidence(records: list[dict[str, Any]]) -> dict[str, Any]:
    direct = _max_by_run_type(records, {"direct_finetune_from_01125"})
    finetune = _max_by_run_type(records, {"finetune_from_01125"})
    direct_like = _max_by_run_type(records, {"direct_finetune_from_01125", "finetune_from_01125"})
    standalone = _max_by_run_type(records, {"standalone_online"})
    continual = _max_by_run_type(records, {"continual_from_01125"})
    all_best = _max_by_run_type(records, {"direct_finetune_from_01125", "finetune_from_01125", "standalone_online", "continual_from_01125", "online_family", "other"})
    return {
        "direct_success": None if direct is None else float(direct["value"]),
        "direct_source": None if direct is None else direct["source_fp"],
        "finetune_success": None if finetune is None else float(finetune["value"]),
        "finetune_source": None if finetune is None else finetune["source_fp"],
        "direct_like_success": None if direct_like is None else float(direct_like["value"]),
        "direct_like_source": None if direct_like is None else direct_like["source_fp"],
        "standalone_success": None if standalone is None else float(standalone["value"]),
        "standalone_source": None if standalone is None else standalone["source_fp"],
        "continual_success": None if continual is None else float(continual["value"]),
        "continual_source": None if continual is None else continual["source_fp"],
        "known_single_task_success": None if all_best is None else float(all_best["value"]),
        "known_single_task_source": None if all_best is None else all_best["source_fp"],
        "records": records,
    }


def _distance_score(min_distance: float, pair_distance: float) -> tuple[float, str]:
    near_cutoff = max(0.35, 1.5 * pair_distance)
    if min_distance < near_cutoff:
        return 0.15, "near_pair_like"
    if min_distance <= 1.6:
        target = 1.0
        return max(0.65, 1.0 - 0.2 * abs(min_distance - target)), "medium"
    if min_distance <= 2.4:
        return 0.55, "far_but_usable"
    return 0.20, "too_far"


def _learnability_score(evidence: dict[str, Any]) -> float:
    direct_like = evidence.get("direct_like_success")
    standalone = evidence.get("standalone_success")
    continual = evidence.get("continual_success")
    score = 0.0
    if direct_like is not None:
        score += 0.70 * min(1.0, max(0.0, float(direct_like)) / 0.8)
    if standalone is not None:
        score += 0.20 * min(1.0, max(0.0, float(standalone)) / 0.9)
    if continual is not None:
        score += 0.10 * min(1.0, max(0.0, float(continual)) / 0.8)
    return float(min(1.0, score))


def _failure_modes(assembly_id: str, evidence: dict[str, Any], distance_band: str) -> list[str]:
    modes: list[str] = []
    direct_like = evidence.get("direct_like_success")
    direct = evidence.get("direct_success")
    if assembly_id in {"00062", "00271"}:
        modes.append("known_hard_canary")
    if direct is not None and float(direct) <= 0.05:
        modes.append("direct_finetune_failed")
    elif direct_like is not None and float(direct_like) < 0.2:
        modes.append("direct_or_finetune_weak")
    if evidence.get("standalone_success") is None and evidence.get("direct_like_success") is None:
        modes.append("no_single_task_evidence")
    if distance_band == "near_pair_like":
        modes.append("too_near_to_existing_pair")
    if distance_band == "too_far":
        modes.append("task_vec_distance_too_far")
    return modes


def _candidate_record(
    entry: dict[str, Any],
    existing_entries: dict[str, dict[str, Any]],
    evidence: dict[str, Any],
    pair_distance: float,
) -> dict[str, Any]:
    distances = {
        task_id: _l2(entry["task_vec_6"], existing["task_vec_6"])
        for task_id, existing in existing_entries.items()
    }
    min_distance = min(distances.values())
    distance_score, distance_band = _distance_score(min_distance, pair_distance)
    learnability = _learnability_score(evidence)

    assets_exist = bool(entry["plug_obj_exists"] and entry["socket_obj_exists"])
    objective_effective = any(
        abs(entry["target_depth"] - existing["target_depth"]) > 1e-9
        or abs(entry["radial_clearance"] - existing["radial_clearance"]) > 1e-12
        or _l2(entry["task_vec_6"], existing["task_vec_6"]) > 1e-6
        for existing in existing_entries.values()
    )
    reward_effective = bool(objective_effective and entry["success_pos_tol"] > 0)
    contact_evidence_success = max(
        value
        for value in [
            evidence.get("direct_like_success") or 0.0,
            evidence.get("standalone_success") or 0.0,
            evidence.get("continual_success") or 0.0,
        ]
    )
    contact_effective = bool(assets_exist and contact_evidence_success > 0.05)
    geometry_effective = bool(assets_exist)
    validity_score = sum(bool(x) for x in [geometry_effective, objective_effective, reward_effective, contact_effective]) / 4.0

    modes = _failure_modes(entry["assembly_id"], evidence, distance_band)
    hard_penalty = 0.0 if any(m in modes for m in ["known_hard_canary", "direct_finetune_failed"]) else 1.0
    score = 0.50 * learnability + 0.25 * distance_score + 0.15 * validity_score + 0.10 * hard_penalty

    recommendable = bool(
        geometry_effective
        and objective_effective
        and reward_effective
        and contact_effective
        and learnability >= 0.15
        and distance_band in {"medium", "far_but_usable"}
        and "known_hard_canary" not in modes
        and "direct_finetune_failed" not in modes
    )

    out = {
        "candidate_task_id": entry["assembly_id"],
        "task_vec_6": [round(float(v), 9) for v in entry["task_vec_6"]],
        "distance_to_01125": distances.get("01125"),
        "distance_to_00256": distances.get("00256"),
        "distance_to_existing_min": min_distance,
        "distance_band": distance_band,
        "geometry_effective": geometry_effective,
        "objective_effective": objective_effective,
        "reward_effective": reward_effective,
        "contact_effective": contact_effective,
        "asset_load_check": {
            "plug_obj_exists": entry["plug_obj_exists"],
            "socket_obj_exists": entry["socket_obj_exists"],
            "plug_obj": entry["plug_obj"],
            "socket_obj": entry["socket_obj"],
        },
        "runtime_evidence_level": "training_logs_plus_static_assets" if contact_effective else "static_assets_only",
        "known_single_task_success": evidence.get("known_single_task_success"),
        "known_single_task_source": evidence.get("known_single_task_source"),
        "known_direct_like_success": evidence.get("direct_like_success"),
        "known_direct_like_source": evidence.get("direct_like_source"),
        "known_standalone_success": evidence.get("standalone_success"),
        "known_standalone_source": evidence.get("standalone_source"),
        "known_continual_success": evidence.get("continual_success"),
        "known_continual_source": evidence.get("continual_source"),
        "known_failure_modes": modes or ["none"],
        "learnability_score": learnability,
        "distance_score": distance_score,
        "selection_score": score,
        "recommend_for_3task_pilot": recommendable,
        "srsa_params": {
            "plug_diameter": entry["plug_diameter"],
            "mesh_hole_diameter": entry["mesh_hole_diameter"],
            "hole_diameter_task": entry["hole_diameter_task"],
            "radial_clearance": entry["radial_clearance"],
            "diametral_clearance": entry["diametral_clearance"],
            "target_depth": entry["target_depth"],
            "success_pos_tol": entry["success_pos_tol"],
        },
    }
    return out


def _format_bool(value: Any) -> str:
    if isinstance(value, bool):
        return "`True`" if value else "`False`"
    return f"`{value}`"


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    recommended = report["recommended_task"]
    backups = report["backup_tasks"]
    rows = []
    for item in report["ranked_candidates"]:
        rows.append(
            "| `{candidate_task_id}` | `{task_vec}` | {d01125:.3f} | {d00256:.3f} | {geom} | {obj} | {rew} | {contact} | {single} | {rank} | {rec} | {modes} |".format(
                candidate_task_id=item["candidate_task_id"],
                task_vec=", ".join(f"{v:.4g}" for v in item["task_vec_6"]),
                d01125=float(item["distance_to_01125"]),
                d00256=float(item["distance_to_00256"]),
                geom=_format_bool(item["geometry_effective"]),
                obj=_format_bool(item["objective_effective"]),
                rew=_format_bool(item["reward_effective"]),
                contact=_format_bool(item["contact_effective"]),
                single="UNKNOWN" if item["known_single_task_success"] is None else f"{float(item['known_single_task_success']):.3f}",
                rank=item["recommended_rank"] if item.get("recommended_rank") is not None else "-",
                rec="yes" if item["recommend_for_3task_pilot"] else "no",
                modes=", ".join(item["known_failure_modes"]),
            )
        )

    lines = [
        "# SRSA Phase 3.0 Easy Third Task Candidates",
        "",
        "本报告只做任务选择和只读审计；未训练，未修改模型、训练流程、sampler、reward、Q、policy 或 MPPI。",
        "",
        f"Status: `{report['status']}`",
        "",
        "## 结论",
        "",
        f"- 推荐 3-task pilot 第三任务：`{recommended['candidate_task_id'] if recommended else 'NONE'}`。",
        f"- 备选任务：{', '.join(f'`{task}`' for task in backups) if backups else '`NONE`'}。",
        "- `00186` 不是无条件默认项；本次推荐它是因为它同时满足 medium task_vec distance、asset 存在、参数有效、standalone 可学，以及 direct-from-01125 有中等可学证据。",
        "- `00062` / `00271` 保持 hard canary，不建议进入当前 3-task pilot。",
        "- 本次 Phase 3.0 没有重新启动 Isaac env；`runtime/evidence level` 来自本地 asset 存在性和已有训练/eval 日志。若要补强证据，可对推荐项单独跑 Phase 0.7 runtime geometry/contact audit。",
        "",
        "## 关键选择依据",
        "",
        f"- `01125`/`00256` baseline task_vec distance: `{report['existing_pair_distance']:.6f}`。",
        f"- 推荐任务与 `01125` 距离：`{recommended['distance_to_01125']:.6f}`；与 `00256` 距离：`{recommended['distance_to_00256']:.6f}`。",
        f"- 推荐任务 known direct-like success：`{recommended['known_direct_like_success']}`。",
        f"- 推荐任务 known standalone success：`{recommended['known_standalone_success']}`。",
        f"- 推荐任务 runtime/evidence level：`{recommended['runtime_evidence_level']}`。",
        "",
        "## 候选表",
        "",
        "| candidate task_id | task_vec_6 | distance_to_01125 | distance_to_00256 | geometry_effective | objective_effective | reward_effective | contact_effective | known_single_task_success | recommended_rank | 3-task pilot | known_failure_modes |",
        "| --- | --- | ---: | ---: | --- | --- | --- | --- | ---: | ---: | --- | --- |",
        *rows,
        "",
        "## 推荐解释",
        "",
        f"`{recommended['candidate_task_id']}` 是当前最合适的 pilot third task：它不是 `01125/00256` 的 near pair，task_vec 距离处于 `{recommended['distance_band']}`；已有 direct-like success 为 `{recommended['known_direct_like_success']}`，standalone success 为 `{recommended['known_standalone_success']}`，比 `00004/00014` 更有 direct acquisition 可行性证据。",
        "",
        "备选任务中，`00004` 的 task_vec distance 合适且有弱可学证据，但历史 success 低于 `00186`；`00014` 距离较好但 direct-like success 更弱。`00062` 和 `00271` 已有 hard/failing 证据，不适合用于验证 Phase 2 representation repair 的第一轮三任务扩展。",
        "",
        "## 下一步",
        "",
        "进入 3-task pilot 时建议仍保持 acquisition-first：单 active env 训练第三任务，anchor/replay 中保留 `01125` 和 `00256`，先做 acquisition gate，再做 family retention 和 task-vector swap/sensitivity。暂时不要启用 counterfactual reward 或 reward residual。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-csv", default="data/srsa_mesh_geometry_params.csv")
    parser.add_argument("--task-template-fp", default="data/srsa_axial_task_templates.json")
    parser.add_argument("--template-id", type=int, default=2)
    parser.add_argument("--reference-anchor", default="01125")
    parser.add_argument("--existing-task-ids", nargs="+", default=list(DEFAULT_EXISTING_TASKS))
    parser.add_argument("--candidate-ids", nargs="+", default=list(DEFAULT_CANDIDATES))
    parser.add_argument("--log-root", default="logs/isaaclab-srsa-assembly/1")
    parser.add_argument("--out-dir", default=str(REPORT_DIR))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    mesh_csv = Path(args.mesh_csv)
    template_fp = Path(args.task_template_fp)
    out_dir = Path(args.out_dir)

    rows = _load_mesh_rows(mesh_csv)
    manifest, template = _load_template(template_fp, args.template_id)
    reference_anchor = str(args.reference_anchor).zfill(5)
    existing_ids = [str(task_id).zfill(5) for task_id in args.existing_task_ids]
    candidate_ids = [str(task_id).zfill(5) for task_id in args.candidate_ids]

    missing = [task_id for task_id in existing_ids + candidate_ids + [reference_anchor] if task_id not in rows]
    if missing:
        print(f"FAIL missing assembly ids in mesh CSV: {sorted(set(missing))}")
        return 2

    entries = {
        task_id: _task_entry(task_id, rows, manifest, template, reference_anchor=reference_anchor)
        for task_id in sorted(set(existing_ids + candidate_ids))
    }
    existing_entries = {task_id: entries[task_id] for task_id in existing_ids}
    pair_distance = _l2(existing_entries[existing_ids[0]]["task_vec_6"], existing_entries[existing_ids[1]]["task_vec_6"])
    evidence_records = _best_json_records(Path(args.log_root))

    candidates = []
    for task_id in candidate_ids:
        evidence = _known_evidence(evidence_records.get(task_id, []))
        candidates.append(_candidate_record(entries[task_id], existing_entries, evidence, pair_distance))

    ranked = sorted(candidates, key=lambda item: float(item["selection_score"]), reverse=True)
    rank = 1
    for item in ranked:
        if item["recommend_for_3task_pilot"]:
            item["recommended_rank"] = rank
            rank += 1
        else:
            item["recommended_rank"] = None
    recommended = next((item for item in ranked if item["recommend_for_3task_pilot"]), None)
    backups = [item["candidate_task_id"] for item in ranked if item["recommend_for_3task_pilot"] and item is not recommended][:2]

    status = "PASS" if recommended and len(backups) >= 1 else "WARNING"
    messages = []
    if recommended is None:
        messages.append({"level": "FAIL", "message": "No candidate satisfied the 3-task pilot filters."})
        status = "FAIL"
    else:
        messages.append(
            {
                "level": "PASS",
                "message": f"Recommended easy/medium third task is {recommended['candidate_task_id']}.",
            }
        )
    for item in ranked:
        if "known_hard_canary" in item["known_failure_modes"]:
            messages.append(
                {
                    "level": "WARNING",
                    "message": f"{item['candidate_task_id']} is marked as hard canary and excluded from recommendation.",
                }
            )

    report = {
        "status": status,
        "audit_scope": {
            "training_started": False,
            "model_modified": False,
            "sampler_modified": False,
            "reward_modified": False,
            "isaac_env_launched": False,
            "runtime_effectiveness_evidence": (
                "candidate geometry/contact fields use static asset existence plus prior "
                "training/eval best.json evidence; no new Isaac runtime audit was launched."
            ),
        },
        "config": {
            "mesh_csv": str(mesh_csv),
            "task_template_fp": str(template_fp),
            "template_id": args.template_id,
            "reference_anchor": reference_anchor,
            "existing_task_ids": existing_ids,
            "candidate_ids": candidate_ids,
            "log_root": args.log_root,
        },
        "existing_tasks": {task_id: entries[task_id] for task_id in existing_ids},
        "existing_pair_distance": pair_distance,
        "recommended_task": recommended,
        "backup_tasks": backups,
        "ranked_candidates": ranked,
        "messages": messages,
    }

    json_path = out_dir / "third_task_candidates.json"
    md_path = out_dir / "third_task_candidates.md"
    if args.dry_run:
        print(f"{status} dry-run: would write {json_path} and {md_path}")
        if recommended:
            print(f"recommended={recommended['candidate_task_id']} backups={backups}")
        return 0 if status in {"PASS", "WARNING"} else 2

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if recommended is not None:
        _write_markdown(report, md_path)
    print(f"{status} wrote {json_path} and {md_path}")
    if recommended:
        print(f"recommended={recommended['candidate_task_id']} backups={backups}")
    return 0 if status in {"PASS", "WARNING"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
