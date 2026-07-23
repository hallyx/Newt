#!/usr/bin/env python3
"""Phase 4.2 parametric-pretrain -> balanced three-task expansion."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for path in (SCRIPT_DIR.parent, SCRIPT_DIR, SCRIPT_DIR / "phase3_three_task_pilot"):
	if str(path) not in sys.path:
		sys.path.insert(0, str(path))

import phase4_multitask_origin_train as phase4  # noqa: E402
import task_vec_sensitivity_report as tvsr  # noqa: E402
from common.world_model import WorldModel  # noqa: E402
from tdmpc2 import TDMPC2  # noqa: E402


def resolve(value: str | Path) -> Path:
	path = Path(value).expanduser()
	return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def _write_json(value, path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--pretrain-checkpoint", required=True)
	parser.add_argument("--output", default="reports/phase4_2_parametric_pretraining/checkpoints/parametric_then_three_task.pt")
	parser.add_argument("--report", default="reports/phase4_2_parametric_pretraining/checkpoints/parametric_then_three_task_train.json")
	parser.add_argument("--config", default="configs/train/srsa_01125_imitation_relaxed.yaml")
	parser.add_argument("--replay-01125", default=phase4.DEFAULT_REPLAYS["01125"])
	parser.add_argument("--replay-00256", default=phase4.DEFAULT_REPLAYS["00256"])
	parser.add_argument("--replay-00186", default=phase4.DEFAULT_REPLAYS["00186"])
	parser.add_argument("--updates", type=int, default=8447)
	parser.add_argument("--batch-size", type=int, default=1024)
	parser.add_argument("--horizon", type=int, default=3)
	parser.add_argument("--train-fraction", type=float, default=.80)
	parser.add_argument("--split-seed", type=int, default=4240)
	parser.add_argument("--sample-seed", type=int, default=4241)
	parser.add_argument("--train-seed", type=int, default=4242)
	parser.add_argument("--gpu-id", type=int, default=0)
	parser.add_argument("--log-every", type=int, default=100)
	parser.add_argument("--save-every", type=int, default=500)
	parser.add_argument("--run-label", default="phase4_2_expansion")
	parser.add_argument("--overwrite", action="store_true")
	parser.add_argument("--dry-run", action="store_true")
	args = parser.parse_args()

	pretrain = resolve(args.pretrain_checkpoint)
	output = resolve(args.output)
	report_path = resolve(args.report)
	if args.horizon != 3 or args.batch_size < 9 or args.updates <= 0:
		raise ValueError("Keep horizon=3, positive updates, and batch_size>=9.")
	if args.dry_run:
		print("PASS dry-run")
		print(f"pretrain={pretrain}")
		print(f"output={output}")
		print(f"updates={args.updates} batch={args.batch_size}; all 9 task x phase cells per update")
		print("device: physical cuda1 via CUDA_VISIBLE_DEVICES=1, logical cuda:0")
		return 0
	if not pretrain.exists():
		raise FileNotFoundError(pretrain)
	if output.exists() and not args.overwrite:
		print(f"[phase4.2] expansion checkpoint exists, skipping: {output}")
		return 0
	# Progress checkpoints are written inside the update loop, so their parent
	# must exist before the first save interval rather than only at final save.
	output.parent.mkdir(parents=True, exist_ok=True)
	report_path.parent.mkdir(parents=True, exist_ok=True)

	device = phase4._require_cuda1(args.gpu_id)
	phase4._seed_all(args.train_seed)
	replay_items = phase4._load_replays(args)
	cfg_args = SimpleNamespace(
		config=str(resolve(args.config)), gpu_id=args.gpu_id, batch_size=args.batch_size,
		assembly_id="00186", eval_task_id=2,
	)
	cfg, compat = tvsr._load_config(cfg_args, pretrain)
	cfg.device_id = args.gpu_id
	cfg.gpu_id = args.gpu_id
	cfg.batch_size = args.batch_size
	cfg.num_envs = args.batch_size
	cfg.horizon = args.horizon
	cfg.compile = False
	cfg.lr_schedule = None
	cfg.rank = 0
	cfg.world_size = 1
	cfg.finetune = True
	cfg.multitask_continuation_enabled = False
	cfg.multitask_prox_reg_enabled = False
	cfg.latent_residual_enabled = False
	model = WorldModel(copy.deepcopy(cfg)).to(device)
	model = tvsr._load_world_model(model, pretrain, cfg)
	agent = TDMPC2(model, copy.deepcopy(cfg)).to(device)
	sampler = phase4.TaskPhaseBalancedBuffer(replay_items, args.batch_size, args.horizon, args.sample_seed)
	protocol = {
		"phase": "4.2",
		"run_label": str(args.run_label),
		"arm": "parametric_pretrain_then_three_task_expansion",
		"pretrain_checkpoint": str(pretrain),
		"pretrain_checkpoint_sha256": phase4._sha256(pretrain),
		"updates": args.updates,
		"batch_size": args.batch_size,
		"horizon": args.horizon,
		"batch_balance": "all 3 tasks x pre-contact/contact/insertion cells every update",
		"all_parameters_trainable": True,
		"compatibility": compat,
		"prohibitions": {
			"elite_distillation": False,
			"counterfactual_reward_or_residual": False,
			"mppi_modified": False,
			"new_task_type": False,
		},
		"device": {"physical": "cuda1", "visible": os.environ.get("CUDA_VISIBLE_DEVICES"), "logical": "cuda:0"},
	}
	history = []
	started = time.time()
	for update in range(args.updates):
		metrics = agent.update(sampler)
		completed = update + 1
		if completed == 1 or completed % args.log_every == 0 or completed == args.updates:
			row = {
				"update": completed,
				"elapsed_seconds": time.time() - started,
				"metrics": {key: phase4._metric_float(value) for key, value in metrics.items()},
				"task_counts": sampler.last_batch_task_counts,
				"phase_counts": sampler.last_batch_phase_counts,
				"cell_counts": sampler.last_batch_cell_counts,
			}
			history.append(row)
			print(
				f"[phase4.2] expansion update={completed}/{args.updates} "
				f"loss={row['metrics'].get('total_loss', float('nan')):.4f} elapsed={row['elapsed_seconds']:.1f}s",
				flush=True,
			)
		if completed % args.save_every == 0 and completed < args.updates:
			progress = output.with_suffix(".progress.pt")
			torch.save(phase4._checkpoint_payload(
				agent, variant="P3", updates=completed, sampler=sampler, protocol=protocol, progress=True,
			), progress)

	payload = phase4._checkpoint_payload(
		agent, variant="P3", updates=args.updates, sampler=sampler, protocol=protocol, progress=False,
	)
	payload["metadata"].update({"phase": "4.2", "phase4_2_arm": "parametric_then_three_task"})
	torch.save(payload, output)
	progress = output.with_suffix(".progress.pt")
	if progress.exists():
		progress.unlink()
	_write_json({
		"status": "PASS",
		"checkpoint": str(output),
		"checkpoint_sha256": phase4._sha256(output),
		"protocol": protocol,
		"history": history,
		"final_sample_index": sampler.sample_index,
	}, report_path)
	print(f"[phase4.2] wrote expansion checkpoint: {output}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
