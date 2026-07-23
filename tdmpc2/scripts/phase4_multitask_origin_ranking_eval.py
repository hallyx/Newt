#!/usr/bin/env python3
"""Phase 4.0 predicted-vs-real candidate ranking on cloned simulator states."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PHASE3_DIR = SCRIPT_DIR / "phase3_three_task_pilot"
if str(PHASE3_DIR) not in sys.path:
	sys.path.insert(0, str(PHASE3_DIR))

import offline_closed_loop_gap_diagnosis as gap  # noqa: E402


def resolve(value: str | Path) -> Path:
	path = Path(value).expanduser()
	return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def _sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as handle:
		for block in iter(lambda: handle.read(1024 * 1024), b""):
			digest.update(block)
	return digest.hexdigest()


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--variant", required=True)
	parser.add_argument("--checkpoint", required=True)
	parser.add_argument("--output", required=True)
	parser.add_argument("--config", default="configs/train/srsa_01125_imitation_relaxed.yaml")
	parser.add_argument("--task-template-fp", default="data/srsa_axial_task_templates.json")
	parser.add_argument("--mesh-geometry-fp", default="data/srsa_mesh_geometry_params.csv")
	parser.add_argument("--isaaclab-dir", default="/home/gpuserver/IsaacLab")
	parser.add_argument("--srsa-dir", default="/home/gpuserver/hx/github/srsa")
	parser.add_argument("--gpu-id", type=int, default=0)
	parser.add_argument("--candidate-base-states", type=int, default=4)
	parser.add_argument("--num-candidates", type=int, default=64)
	parser.add_argument("--num-policy-candidates", type=int, default=3)
	parser.add_argument("--num-elites", type=int, default=8)
	parser.add_argument("--candidate-horizon", type=int, default=3)
	parser.add_argument("--real-discount", type=float, default=.99)
	parser.add_argument("--clone-obs-tolerance", type=float, default=1e-4)
	parser.add_argument("--seed", type=int, default=4060)
	parser.add_argument("--assembly-id", default="00186")
	parser.add_argument("--param-scale", type=float, default=None)
	parser.add_argument("--param-template", default=None)
	parser.add_argument("--param-clearance-base", type=float, default=None)
	parser.add_argument("--param-depth-base", type=float, default=None)
	parser.add_argument("--dry-run", action="store_true")
	args = parser.parse_args()

	checkpoint = resolve(args.checkpoint)
	output = resolve(args.output)
	if not checkpoint.exists():
		raise FileNotFoundError(checkpoint)
	if args.num_candidates != 64 or args.num_policy_candidates != 3:
		raise ValueError("Preserve the Phase 3.11 contract: 3 policy + 61 Gaussian candidates.")
	if args.num_elites != 8 or args.candidate_horizon != 3:
		raise ValueError("Preserve top-8 and horizon-3.")
	if args.dry_run:
		print("PASS dry-run")
		print(f"variant={args.variant} checkpoint={checkpoint}")
		print(f"output={output}")
		print("device: physical cuda1 via CUDA_VISIBLE_DEVICES=1, logical cuda:0")
		return 0

	gap._require_physical_cuda1(args)
	hash_before = _sha256(checkpoint)
	result = gap._prediction_reality_report(args, OrderedDict([("original", checkpoint)]))
	hash_after = _sha256(checkpoint)
	if hash_before != hash_after:
		raise RuntimeError(f"Checkpoint mutated during ranking eval: {checkpoint}")
	report = {
		"status": "PASS", "variant": args.variant, "checkpoint": str(checkpoint),
		"checkpoint_sha256_before": hash_before, "checkpoint_sha256_after": hash_after,
		"checkpoint_unchanged": True,
		"device": {"physical": "cuda1", "visible": os.environ.get("CUDA_VISIBLE_DEVICES"), "logical": "cuda:0"},
		"contract": {
			"phases": list(gap.PHASES), "candidate_base_states": args.candidate_base_states,
			"candidates": args.num_candidates, "policy_candidates": args.num_policy_candidates,
			"gaussian_candidates": args.num_candidates - args.num_policy_candidates,
			"horizon": args.candidate_horizon, "top_k": args.num_elites,
			"scorer_and_controller": "the evaluated variant itself; no frozen cross-variant scorer",
		},
		"prediction_reality": result,
	}
	output.parent.mkdir(parents=True, exist_ok=True)
	output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
	print(f"[phase4.0] ranking {args.variant}: wrote {output}", flush=True)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
