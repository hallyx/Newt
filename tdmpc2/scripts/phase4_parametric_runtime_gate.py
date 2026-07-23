#!/usr/bin/env python3
"""Runtime causal gate for Phase 4.2 single-family parameter sampling."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
AUDIT_DIR = SCRIPT_DIR / "automate_param_audit"
for path in (SCRIPT_DIR.parent, AUDIT_DIR):
	if str(path) not in sys.path:
		sys.path.insert(0, str(path))

from _common import close_env, launch_probe_env  # noqa: E402


def resolve(value: str | Path) -> Path:
	path = Path(value).expanduser()
	return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def _require_cuda1() -> None:
	if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() != "1":
		raise RuntimeError("Phase 4.2 runtime gate requires physical CUDA1 via CUDA_VISIBLE_DEVICES=1.")
	if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
		raise RuntimeError("Expected exactly one visible CUDA device.")


def _prim_scale(stage, prim_path: str):
	from pxr import UsdGeom

	prim = stage.GetPrimAtPath(prim_path)
	if not prim or not prim.IsValid():
		return None
	for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
		if op.GetOpType() == UsdGeom.XformOp.TypeScale:
			value = op.Get()
			return [float(value[0]), float(value[1]), float(value[2])]
	return [1.0, 1.0, 1.0]


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--output", default="reports/phase4_2_parametric_pretraining/runtime_gate/parametric_runtime_gate.json")
	parser.add_argument("--num-envs", type=int, default=32)
	parser.add_argument("--steps", type=int, default=5)
	parser.add_argument("--seed", type=int, default=4210)
	parser.add_argument("--dry-run", action="store_true")
	args = parser.parse_args()
	if args.dry_run:
		print("PASS dry-run")
		print("single Isaac app; 01125; scale=[0.85,1.15]; five clearance/depth anchors with jitter")
		return 0
	_require_cuda1()
	env = None
	try:
		overrides = {
			"num_envs": args.num_envs,
			"seed": args.seed,
			"srsa_axial_fixed_plug_scale": False,
			"srsa_axial_scale_range": "0.85,1.15",
			"srsa_axial_clearance_base": 0.000114,
			"srsa_axial_depth_base": 0.015,
			"srsa_axial_clearance_depth_templates": "0.5:0.5;0.5:1.0;1.0:1.0;2.0:1.5;4.0:2.0",
			"srsa_axial_clearance_jitter_ratio": 0.10,
			"srsa_axial_depth_jitter_ratio": 0.10,
			"srsa_axial_reference_radius": 0.003993,
			"srsa_axial_reference_depth": 0.015,
			"srsa_align_direct_reward_success": True,
			"srsa_eval_success_metric": "relaxed",
			"eval_success_metric": "relaxed",
			"isaaclab_max_episode_steps": max(32, args.steps + 4),
		}
		cfg, env = launch_probe_env("01125", extra_overrides=overrides)
		obs, _ = env.reset()
		base = env.unwrapped
		task0 = base.current_task_vec.detach().clone()
		params = base.current_task_param_tensors
		expected_scale = torch.exp(task0[:, 1])
		expected_depth = task0[:, 4] * 0.015
		scale_match = float((expected_scale - params["plug_scale_xy"]).abs().max().item())
		depth_match = float((expected_depth - base.current_insertion_depth_tensor).abs().max().item())
		clearance_match = float((task0[:, 2] * 0.003993 - params["radial_clearance"]).abs().max().item())
		metric_depth_match = float((env._srsa_success_metrics(update_state=False)["target_depth"] - expected_depth).abs().max().item())

		stage = base.scene.stage
		prim_rows = []
		for env_id in range(min(args.num_envs, 8)):
			root = str(base.scene.env_prim_paths[env_id])
			prim_rows.append({
				"env_id": env_id,
				"task_vec": [float(x) for x in task0[env_id].cpu().tolist()],
				"held_prim_scale": _prim_scale(stage, f"{root}/HeldAsset"),
				"fixed_prim_scale": _prim_scale(stage, f"{root}/FixedAsset"),
				"expected_plug_scale_xy": float(params["plug_scale_xy"][env_id].item()),
				"expected_hole_scale_xy": float(params["hole_scale_xy"][env_id].item()),
				"geometry_variant_applied": bool(base.current_geometry_variant_applied[env_id].item()),
			})
		prim_scale_error = 0.0
		for row in prim_rows:
			if row["held_prim_scale"] is not None:
				prim_scale_error = max(prim_scale_error, abs(row["held_prim_scale"][0] - row["expected_plug_scale_xy"]))
			if row["fixed_prim_scale"] is not None:
				prim_scale_error = max(prim_scale_error, abs(row["fixed_prim_scale"][0] - row["expected_hole_scale_xy"]))

		for _ in range(args.steps):
			obs, _, terminated, truncated, _ = env.step(torch.zeros((args.num_envs, 3), device=base.device))
			if bool((terminated | truncated).any().item()):
				raise RuntimeError("Episode ended during within-episode invariance check.")
		task_drift = float((base.current_task_vec - task0).abs().max().item())
		unique = torch.unique(torch.round(task0 * 1.0e6) / 1.0e6, dim=0)
		std = task0.std(dim=0, unbiased=False)
		reward_aligned = hasattr(base, "_newt_original_get_rewards")
		checks = {
			"one_parameter_per_episode": task_drift <= 1.0e-7,
			"multiple_runtime_parameters": int(unique.shape[0]) > 1 and bool((std[1:5] > 0).all().item()),
			"task_vec_matches_geometry_params": max(scale_match, clearance_match) <= 1.0e-6,
			"task_vec_matches_target_depth": max(depth_match, metric_depth_match) <= 1.0e-6,
			"usd_prim_scale_matches_runtime_params": prim_scale_error <= 1.0e-5,
			"reward_success_path_uses_runtime_metrics": reward_aligned,
			"geometry_variant_applied": bool(base.current_geometry_variant_applied.any().item()),
		}
		status = "PASS" if all(checks.values()) else "FAIL"
		report = {
			"status": status,
			"phase": "4.2",
			"assembly_id": "01125",
			"num_envs": args.num_envs,
			"steps_without_reset": args.steps,
			"device": {"physical": "cuda1", "visible": os.environ.get("CUDA_VISIBLE_DEVICES"), "logical": "cuda:0"},
			"checks": checks,
			"measurements": {
				"unique_task_vecs": int(unique.shape[0]),
				"task_vec_std": [float(x) for x in std.cpu().tolist()],
				"within_episode_task_vec_linf": task_drift,
				"scale_taskvec_vs_param_linf": scale_match,
				"clearance_taskvec_vs_param_linf": clearance_match,
				"depth_taskvec_vs_tensor_linf": depth_match,
				"depth_taskvec_vs_success_metric_linf": metric_depth_match,
				"prim_scale_vs_param_linf": prim_scale_error,
			},
			"prim_samples": prim_rows,
			"prohibitions": {
				"task_vec_only_override": False,
				"mppi_modified": False,
				"elite_distillation": False,
				"counterfactual_reward_or_residual": False,
			},
		}
		output = resolve(args.output)
		output.parent.mkdir(parents=True, exist_ok=True)
		output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
		print(f"{status}: wrote {output}")
		return 0 if status == "PASS" else 1
	finally:
		if env is not None:
			close_env(env)


if __name__ == "__main__":
	raise SystemExit(main())
