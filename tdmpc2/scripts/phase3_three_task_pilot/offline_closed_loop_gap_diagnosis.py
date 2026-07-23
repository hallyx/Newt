#!/usr/bin/env python3
"""Phase 3.11 read-only offline-to-closed-loop gap diagnosis.

This diagnostic never trains or writes a checkpoint.  It compares the Phase
3.10 original, elite-only, and elite+behavior-anchor-lambda-3 policies using:

1. exact simulator-state cloning and 1/3/5-step action interventions;
2. frozen-scorer candidate rankings versus real short-horizon outcomes;
3. paired closed-loop state-distribution drift from cloned initial states; and
4. old-task policy drift on phase- and success-labelled original rollouts.

The runtime intentionally requires physical CUDA1 to be exposed as logical
cuda:0 via ``CUDA_VISIBLE_DEVICES=1``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import statistics
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
MODEL_AUDIT_DIR = SCRIPT_DIR.parent / "model_task_sensitivity"
if str(MODEL_AUDIT_DIR) not in sys.path:
	sys.path.insert(0, str(MODEL_AUDIT_DIR))

from _common import condition_batch, resolve, tensor_to_list, tvsr, write_json, write_text  # noqa: E402
import collect_eval_rollouts as collector  # noqa: E402
import planner_action_attribution_diagnosis as phase33  # noqa: E402
from common import set_seed  # noqa: E402
from envs import make_env  # noqa: E402


DEFAULT_REPORT_DIR = Path("reports/phase3_three_task_pilot")
DEFAULT_OUTPUT_JSON = DEFAULT_REPORT_DIR / "phase3_11_offline_closed_loop_gap.json"
DEFAULT_OUTPUT_MD = DEFAULT_REPORT_DIR / "phase3_11_offline_closed_loop_gap.md"
DEFAULT_PARTIAL_DIR = DEFAULT_REPORT_DIR / "phase3_11_parts"
DEFAULT_SOURCE_CHECKPOINT = (
	"logs/isaaclab-srsa-assembly/1/"
	"srsa_axial_online_family_taskctx_repair_01125_00256_00186_rescue/"
	"20260713_phase3_2_rescue_00186_stage-3_asm-00186/models/"
	"best_step-50176_s-0p2461.pt"
)
DEFAULT_ELITE_CHECKPOINT = (
	"reports/phase3_three_task_pilot/phase3_10_policy_only_checkpoints/elite_only_100.pt"
)
DEFAULT_ANCHOR_CHECKPOINT = (
	"reports/phase3_three_task_pilot/phase3_10_policy_only_checkpoints/"
	"elite_behavior_anchor_l3_100.pt"
)
DEFAULT_PHASE310_REPORT = (
	"reports/phase3_three_task_pilot/phase3_10_policy_only_anchored_adaptation.json"
)
CHECKPOINT_LABELS = ("original", "elite_only", "anchor_l3")
PHASES = ("pre_contact", "contact", "insertion")
HORIZONS = (1, 3, 5)
OLD_TASKS = ("01125", "00256")
TASK_TEMPLATE_ID = 2


def _sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as handle:
		for block in iter(lambda: handle.read(1024 * 1024), b""):
			digest.update(block)
	return digest.hexdigest()


def _load_json(path_value: str | Path) -> dict[str, Any]:
	path = resolve(path_value)
	if not path.exists():
		raise FileNotFoundError(path)
	return json.loads(path.read_text(encoding="utf-8"))


def _require_physical_cuda1(args: argparse.Namespace) -> None:
	visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
	if visible != "1":
		raise RuntimeError(
			"Phase 3.11 is pinned to physical CUDA1. Launch with CUDA_VISIBLE_DEVICES=1; "
			f"current value is {visible!r}."
		)
	if int(args.gpu_id) != 0:
		raise RuntimeError(
			"With CUDA_VISIBLE_DEVICES=1, physical CUDA1 is logical cuda:0; use --gpu-id 0."
		)
	if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
		raise RuntimeError(
			"Expected exactly one visible CUDA device after CUDA_VISIBLE_DEVICES=1; "
			f"available={torch.cuda.is_available()} count={torch.cuda.device_count()}."
		)
	torch.cuda.set_device(0)


def _checkpoint_paths(args: argparse.Namespace) -> OrderedDict[str, Path]:
	paths = OrderedDict([
		("original", resolve(args.source_checkpoint)),
		("elite_only", resolve(args.elite_checkpoint)),
		("anchor_l3", resolve(args.anchor_checkpoint)),
	])
	for label, path in paths.items():
		if not path.exists():
			raise FileNotFoundError(f"Missing {label} checkpoint: {path}")
	return paths


def _configure_runtime_cfg(
	args: argparse.Namespace,
	checkpoint: Path,
	assembly_id: str,
	num_envs: int,
	*,
	run_id: str,
):
	"""Build the established Phase 2/3 exact-template SRSA eval config."""
	os.chdir(REPO_ROOT)
	cfg_args = SimpleNamespace(
		config=str(resolve(args.config)),
		gpu_id=int(args.gpu_id),
		batch_size=int(num_envs),
		assembly_id=str(assembly_id),
		eval_task_id=TASK_TEMPLATE_ID,
	)
	cfg, _ = tvsr._load_config(cfg_args, checkpoint)
	cfg.num_envs = int(num_envs)
	cfg.gpu_id = int(args.gpu_id)
	cfg.device_id = int(args.gpu_id)
	cfg.isaaclab_dir = str(args.isaaclab_dir)
	cfg.srsa_dir = str(args.srsa_dir)
	cfg.srsa_task_template_fp = str(resolve(args.task_template_fp))
	cfg.srsa_mesh_geometry_fp = str(resolve(args.mesh_geometry_fp))
	cfg.isaaclab_backend = "srsa"
	cfg.task = "isaaclab-srsa-assembly"
	cfg.isaaclab_headless = True
	cfg.isaaclab_use_canonical_obs = True
	cfg.srsa_param_template_id = TASK_TEMPLATE_ID
	cfg.eval_task_template_exact = True
	cfg.srsa_axial_reference_anchor_assembly_id = "01125"
	cfg.srsa_axial_reference_anchor_task_type_id = 0
	cfg.srsa_axial_recompute_manifest_task_vecs = True
	cfg.srsa_axial_clearance_depth_templates = "1.0:1.0"
	cfg.srsa_enable_axial_task_param_sampler = True
	cfg.srsa_axial_fixed_plug_scale = True
	cfg.srsa_axial_clearance_base = 0.000114
	cfg.srsa_axial_clearance_jitter_ratio = 0.10
	cfg.srsa_axial_depth_base = 0.015
	cfg.srsa_axial_depth_jitter_ratio = 0.10
	# Phase 4.2 can reuse the cloned-state evaluator on exact single-family
	# parameter conditions. Defaults preserve the Phase 3.11/4.0 contract.
	param_scale = getattr(args, "param_scale", None)
	param_template = getattr(args, "param_template", None)
	param_clearance_base = getattr(args, "param_clearance_base", None)
	param_depth_base = getattr(args, "param_depth_base", None)
	param_condition = any(
		value is not None
		for value in (param_scale, param_template, param_clearance_base, param_depth_base)
	)
	if param_condition:
		# _task_cfg clears template-derived fields before optionally applying an
		# assembly template. Disable that template here; the physical parameter
		# condition is restored immediately after _task_cfg below.
		cfg.eval_task_template_exact = False
		cfg.eval_task_template_apply_geometry = False
		cfg.eval_task_template_apply_sampler = False
		cfg.srsa_axial_recompute_manifest_task_vecs = False
	# Match the exact-template closed-loop contract used in Phase 2.2/3.10.
	cfg.srsa_axial_init_error_xy_range = "0.009,0.0010"
	cfg.srsa_axial_init_error_z_range = "0.0010,0.0020"
	cfg.srsa_axial_init_error_yaw_range = "-0.0872665,0.0872665"
	cfg.srsa_axial_visual_noise_xy_range = "0.0,0.0"
	cfg.srsa_axial_visual_noise_z_range = "0.0,0.0"
	cfg.srsa_enable_flange_force_sensor = True
	cfg.isaaclab_canonical_append_force = True
	cfg.isaaclab_canonical_append_task_params = False
	cfg.srsa_vision_noise_xy_std = 0.0
	cfg.srsa_vision_noise_xy_jitter_std = 0.0
	cfg.srsa_vision_noise_z_std = 0.0
	cfg.srsa_vision_noise_z_jitter_std = 0.0
	cfg.isaaclab_canonical_use_visual_noise = False
	cfg.task_conditioning = "axial_params"
	cfg.contact_history_enabled = True
	cfg.contact_history_len = 4
	cfg.contact_context_dim = 64
	cfg.contact_history_hidden_dim = 128
	cfg.contact_history_layers = 2
	cfg.contact_force_dim = 6
	cfg.contact_action_dim = 3
	cfg.contact_ee_delta_dim = 3
	cfg.contact_history_use_ee_delta = True
	cfg.compile = False
	cfg.mpc = True
	cfg.eval_terminate_on_success = False
	cfg.enable_wandb = False
	cfg.save_agent = False
	cfg.exp_name = "srsa_phase3_11_offline_closed_loop_gap"
	cfg.run_id = str(run_id)
	cfg.seed = int(args.seed)
	cfg = collector._task_cfg(cfg, str(assembly_id))
	if param_condition:
		cfg.srsa_enable_axial_task_param_sampler = True
		cfg.srsa_axial_reference_radius = 0.003993
		cfg.srsa_axial_reference_depth = 0.015
		cfg.srsa_axial_fixed_plug_scale = param_scale is None
		if param_scale is not None:
			cfg.srsa_axial_scale_range = f"{float(param_scale):.12g},{float(param_scale):.12g}"
		cfg.srsa_axial_clearance_depth_templates = str(param_template or "1.0:1.0")
		cfg.srsa_axial_clearance_base = float(
			0.000114 if param_clearance_base is None else param_clearance_base
		)
		cfg.srsa_axial_depth_base = float(0.015 if param_depth_base is None else param_depth_base)
		cfg.srsa_axial_clearance_jitter_ratio = 0.0
		cfg.srsa_axial_depth_jitter_ratio = 0.0
		cfg.srsa_axial_yaw_requirement = False
	cfg.num_envs = int(num_envs)
	cfg.device_id = int(args.gpu_id)
	cfg.gpu_id = int(args.gpu_id)
	cfg.checkpoint = str(checkpoint)
	return cfg


def _agent_cfg(runtime_cfg, checkpoint: Path, num_envs: int):
	cfg = copy.deepcopy(runtime_cfg)
	cfg.num_envs = int(num_envs)
	cfg.checkpoint = str(checkpoint)
	cfg.device_id = int(runtime_cfg.device_id)
	cfg.gpu_id = int(runtime_cfg.gpu_id)
	cfg.rank = 0
	return cfg


def _make_agents(runtime_cfg, checkpoints: OrderedDict[str, Path], num_envs: int):
	return OrderedDict(
		(label, collector._make_agent(_agent_cfg(runtime_cfg, checkpoint, num_envs)))
		for label, checkpoint in checkpoints.items()
	)


def _reset_agent(agent) -> None:
	agent._prev_mean.zero_()
	agent._residual_force_history = None
	agent._last_latent_residual_info = None


def _matched_agent_action(agent, obs, task, *, t0: bool, seed: int) -> torch.Tensor:
	_reset = torch.full((obs.shape[0],), bool(t0), dtype=torch.bool, device=obs.device)
	with torch.random.fork_rng(devices=[obs.device.index]):
		torch.manual_seed(int(seed))
		torch.cuda.manual_seed_all(int(seed))
		action, _ = agent(
			obs,
			t0=_reset,
			step=1,
			eval_mode=True,
			task=task,
			mpc=True,
		)
	return action.detach()


def _policy_mean(model, obs: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
	with torch.no_grad():
		z = model.encode(obs, task)
		_, info = model.pi(z, task)
	return info["mean"].detach()


def _runtime_task(env, ids: torch.Tensor | None = None) -> torch.Tensor:
	task = getattr(env.unwrapped, "current_task_vec", None)
	if not torch.is_tensor(task) or task.ndim != 2 or task.shape[-1] != 6:
		raise RuntimeError(f"Runtime task_vec has unexpected shape: {getattr(task, 'shape', None)}")
	return task if ids is None else task[ids]


def _obs(env, expected_dim: int) -> torch.Tensor:
	raw = env.unwrapped._get_observations()
	return collector._adapt_obs_to_checkpoint(env._extract_obs(raw), expected_dim)


def _metrics(env) -> dict[str, torch.Tensor]:
	values = env._srsa_success_metrics(update_state=False)
	contact = getattr(env.unwrapped, "flange_force_flag", None)
	if torch.is_tensor(contact):
		values["contact"] = contact.reshape(contact.shape[0], -1).any(dim=1).to(torch.float32)
	else:
		values["contact"] = torch.zeros(env.cfg.num_envs, device=env.unwrapped.device)
	return {
		key: value.detach().clone()
		for key, value in values.items()
		if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == env.cfg.num_envs
	}


def _copy_tensor_rows(value: torch.Tensor, source_ids: torch.Tensor, target_ids: torch.Tensor) -> None:
	if value.ndim == 0 or value.shape[0] <= int(torch.max(torch.cat((source_ids, target_ids))).item()):
		return
	value[target_ids] = value[source_ids].clone()


def _copy_mapping_rows(value: Any, source_ids: torch.Tensor, target_ids: torch.Tensor, num_envs: int) -> None:
	if not isinstance(value, dict):
		return
	for item in value.values():
		if torch.is_tensor(item) and item.ndim > 0 and item.shape[0] == num_envs:
			_copy_tensor_rows(item, source_ids, target_ids)


def _clone_simulator_state(env, source_ids: torch.Tensor, target_ids: torch.Tensor) -> dict[str, float]:
	"""Clone physical and task/control state between parallel Isaac env slots."""
	base = env.unwrapped
	device = base.device
	source_ids = source_ids.to(device=device, dtype=torch.long).reshape(-1)
	target_ids = target_ids.to(device=device, dtype=torch.long).reshape(-1)
	if source_ids.shape != target_ids.shape:
		raise ValueError("source_ids and target_ids must have the same shape.")
	num_envs = int(base.num_envs)
	origins = base.scene.env_origins

	# Task, controller, finite-difference, success, and diagnostic state.  Derived
	# kinematic tensors are refreshed after the physical state is written.
	for owner in (base, env):
		for name, value in vars(owner).items():
			if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == num_envs:
				_copy_tensor_rows(value, source_ids, target_ids)
				if name in {"prev_fingertip_pos", "prev_fingertip_midpoint_pos"}:
					delta = origins[target_ids] - origins[source_ids]
					while delta.ndim < value[target_ids].ndim:
						delta = delta.unsqueeze(1)
					value[target_ids] += delta
			elif isinstance(value, dict):
				_copy_mapping_rows(value, source_ids, target_ids, num_envs)

	# Rigid objects and articulations use world-space root states, so translate
	# positions by the destination/source environment-origin delta.
	for asset in getattr(base.scene, "rigid_objects", {}).values():
		root = asset.data.root_state_w[source_ids].clone()
		root[:, :3] += origins[target_ids] - origins[source_ids]
		asset.write_root_state_to_sim(root, env_ids=target_ids)
	for articulation in getattr(base.scene, "articulations", {}).values():
		root = articulation.data.root_state_w[source_ids].clone()
		root[:, :3] += origins[target_ids] - origins[source_ids]
		articulation.write_root_state_to_sim(root, env_ids=target_ids)
		joint_pos = articulation.data.joint_pos[source_ids].clone()
		joint_vel = articulation.data.joint_vel[source_ids].clone()
		articulation.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=target_ids)
		if hasattr(articulation, "set_joint_position_target"):
			articulation.set_joint_position_target(joint_pos, env_ids=target_ids)

	if hasattr(base, "_apply_geometry_variant"):
		base._apply_geometry_variant(target_ids)
	base.scene.write_data_to_sim()
	if hasattr(base.sim, "forward"):
		base.sim.forward()
	base.scene.update(dt=base.physics_dt)
	if hasattr(base, "_compute_intermediate_values"):
		base._compute_intermediate_values(base.physics_dt)
	if hasattr(base, "_reset_vision_noise_cache"):
		base._reset_vision_noise_cache()

	# Quantify physical residuals after cloning in local coordinates.
	physical = []
	for asset in getattr(base.scene, "rigid_objects", {}).values():
		src = asset.data.root_state_w[source_ids].clone()
		dst = asset.data.root_state_w[target_ids].clone()
		src[:, :3] -= origins[source_ids]
		dst[:, :3] -= origins[target_ids]
		physical.append((src - dst).abs().reshape(src.shape[0], -1))
	for articulation in getattr(base.scene, "articulations", {}).values():
		physical.append((
			articulation.data.joint_pos[source_ids] - articulation.data.joint_pos[target_ids]
		).abs().reshape(source_ids.shape[0], -1))
	max_physical = float(torch.cat(physical, dim=1).max().item()) if physical else math.nan
	return {"max_physical_state_abs_delta": max_physical}


def _clone_groups(env, source_ids: torch.Tensor, target_groups: Iterable[torch.Tensor], expected_dim: int):
	all_targets = []
	all_sources = []
	for targets in target_groups:
		all_targets.append(targets)
		all_sources.append(source_ids)
	result = _clone_simulator_state(env, torch.cat(all_sources), torch.cat(all_targets))
	cloned_obs = _obs(env, expected_dim)
	obs_deltas = []
	for targets in target_groups:
		obs_deltas.append((cloned_obs[source_ids] - cloned_obs[targets]).abs().reshape(source_ids.shape[0], -1))
	packed_raw_delta = torch.cat(obs_deltas, dim=0)
	result["max_raw_observation_abs_delta"] = float(packed_raw_delta.max().item())
	result["raw_observation_max_abs_delta_per_dim"] = tensor_to_list(packed_raw_delta.max(dim=0).values)
	# Contact sensors and finite-difference velocity caches are not part of the
	# writable PhysX root/joint state.  The causal intervention therefore copies
	# the source observation as well as the physical state for the branch's first
	# controller input.  After one simulator step every arm consumes its own real
	# observation again.
	for targets in target_groups:
		cloned_obs[targets] = cloned_obs[source_ids].clone()
	controller_deltas = [
		(cloned_obs[source_ids] - cloned_obs[targets]).abs().reshape(source_ids.shape[0], -1)
		for targets in target_groups
	]
	result["max_controller_input_observation_abs_delta"] = float(
		torch.cat(controller_deltas, dim=1).max().item()
	)
	result["first_input_observation_copied"] = True
	return cloned_obs, result


def _summary(values: torch.Tensor | list[float]) -> dict[str, float | int]:
	tensor = torch.as_tensor(values, dtype=torch.float32).reshape(-1)
	tensor = tensor[torch.isfinite(tensor)]
	if tensor.numel() == 0:
		return {"count": 0, "mean": math.nan, "p50": math.nan, "p95": math.nan, "max": math.nan}
	return {
		"count": int(tensor.numel()),
		"mean": float(tensor.mean().item()),
		"p50": float(torch.quantile(tensor, 0.50).item()),
		"p95": float(torch.quantile(tensor, 0.95).item()),
		"max": float(tensor.max().item()),
	}


def _phase_steps(episode_length: int) -> OrderedDict[str, int]:
	return OrderedDict([
		("pre_contact", 0),
		("contact", max(1, int(round(0.27 * episode_length)))),
		("insertion", max(2, int(round(0.60 * episode_length)))),
	])


def _phase_from_runtime(metrics: dict[str, torch.Tensor], step: int, episode_length: int) -> torch.Tensor:
	depth = metrics["depth_fraction"].reshape(-1)
	contact = metrics["contact"].reshape(-1) > 0.5
	phase = torch.zeros_like(depth, dtype=torch.long)
	phase[(contact | (depth >= 0.25))] = 1
	phase[depth >= 0.65] = 2
	# The force signal can be sparse before collision is fully registered.  Use
	# the established Phase 3.3 temporal fallback only for still-unlabelled rows.
	if step >= int(round(0.60 * episode_length)):
		phase[phase == 0] = 2
	elif step >= int(round(0.27 * episode_length)):
		phase[phase == 0] = 1
	return phase


def _metric_rows(metrics: dict[str, torch.Tensor], ids: torch.Tensor) -> dict[str, torch.Tensor]:
	keys = ("lateral_error", "keypoint_error", "current_depth", "depth_fraction", "contact", "jam")
	return {key: metrics[key][ids].reshape(-1).detach() for key in keys if key in metrics}


def _warm_to_phase(
	env,
	source_agent,
	source_ids: torch.Tensor,
	target_groups: list[torch.Tensor],
	warm_steps: int,
	expected_dim: int,
	seed: int,
):
	obs, _ = env.reset()
	obs = collector._adapt_obs_to_checkpoint(obs, expected_dim)
	obs, initial_clone = _clone_groups(env, source_ids, target_groups, expected_dim)
	_reset_agent(source_agent)
	for step in range(int(warm_steps)):
		task = _runtime_task(env, source_ids)
		action = _matched_agent_action(
			source_agent,
			obs[source_ids],
			task,
			t0=(step == 0),
			seed=int(seed) + step,
		)
		all_action = torch.zeros(
			int(env.cfg.num_envs), action.shape[-1], dtype=action.dtype, device=action.device,
		)
		all_action[source_ids] = action
		for targets in target_groups:
			all_action[targets] = action
		raw_obs, _, _, _, _ = env.step(all_action)
		obs = collector._adapt_obs_to_checkpoint(raw_obs, expected_dim)
	obs, branch_clone = _clone_groups(env, source_ids, target_groups, expected_dim)
	return obs, {"initial": initial_clone, "branch": branch_clone}


def _intervention_report(
	args: argparse.Namespace,
	checkpoints: OrderedDict[str, Path],
	checkpoint_hashes: dict[str, str],
):
	per_policy = int(args.intervention_episodes)
	total_envs = per_policy * len(CHECKPOINT_LABELS)
	runtime_cfg = _configure_runtime_cfg(
		args, checkpoints["original"], "00186", total_envs, run_id="phase3_11_intervention"
	)
	env = make_env(runtime_cfg)
	try:
		expected_dim = int(runtime_cfg.obs_shape["state"][0])
		device = env.unwrapped.device
		source_ids = torch.arange(per_policy, device=device)
		elite_ids = torch.arange(per_policy, 2 * per_policy, device=device)
		anchor_ids = torch.arange(2 * per_policy, 3 * per_policy, device=device)
		groups = [elite_ids, anchor_ids]
		agents = _make_agents(runtime_cfg, checkpoints, per_policy)
		phase_results = OrderedDict()
		clone_checks = []
		for phase_index, (phase, warm_steps) in enumerate(_phase_steps(int(runtime_cfg.episode_length)).items()):
			obs, clone = _warm_to_phase(
				env,
				agents["original"],
				source_ids,
				groups,
				warm_steps,
				expected_dim,
				int(args.seed) + 10_000 * phase_index,
			)
			clone_checks.append({"phase": phase, **clone["branch"]})
			if clone["branch"]["max_controller_input_observation_abs_delta"] > float(args.clone_obs_tolerance):
				raise RuntimeError(f"Observation clone mismatch at {phase}: {clone['branch']}")
			for agent in agents.values():
				_reset_agent(agent)
			initial = _metrics(env)
			initial_rows = {
				"original": _metric_rows(initial, source_ids),
				"elite_only": _metric_rows(initial, elite_ids),
				"anchor_l3": _metric_rows(initial, anchor_ids),
			}
			returns = {label: torch.zeros(per_policy, device=device) for label in CHECKPOINT_LABELS}
			action_drifts: dict[str, list[torch.Tensor]] = {"elite_only": [], "anchor_l3": []}
			horizon_rows: dict[int, dict[str, Any]] = {}
			ids_by_label = OrderedDict([
				("original", source_ids),
				("elite_only", elite_ids),
				("anchor_l3", anchor_ids),
			])
			for local_step in range(1, max(HORIZONS) + 1):
				actions = OrderedDict()
				common_seed = int(args.seed) + 100_000 * phase_index + local_step
				for label, agent in agents.items():
					ids = ids_by_label[label]
					actions[label] = _matched_agent_action(
						agent,
						obs[ids],
						_runtime_task(env, ids),
						t0=(local_step == 1),
						seed=common_seed,
					)
				action_drifts["elite_only"].append(torch.linalg.vector_norm(
					actions["elite_only"] - actions["original"], dim=-1,
				))
				action_drifts["anchor_l3"].append(torch.linalg.vector_norm(
					actions["anchor_l3"] - actions["original"], dim=-1,
				))
				raw_obs, reward, _, _, _ = env.step(torch.cat(list(actions.values()), dim=0))
				obs = collector._adapt_obs_to_checkpoint(raw_obs, expected_dim)
				for label, ids in ids_by_label.items():
					returns[label] += reward[ids]
				if local_step in HORIZONS:
					current = _metrics(env)
					row = OrderedDict()
					for label, ids in ids_by_label.items():
						metrics = _metric_rows(current, ids)
						base = initial_rows[label]
						row[label] = {
							"reward_sum": _summary(returns[label]),
							"depth_progress_mm": _summary(1000.0 * (metrics["current_depth"] - base["current_depth"])),
							"lateral_error_mm": _summary(1000.0 * metrics["lateral_error"]),
							"keypoint_error_mm": _summary(1000.0 * metrics["keypoint_error"]),
							"contact_rate": float(metrics["contact"].float().mean().item()),
							"jam_rate": float(metrics["jam"].float().mean().item()),
						}
						if label != "original":
							row[label]["executed_action_l2_vs_original"] = _summary(
								torch.stack(action_drifts[label], dim=0).reshape(-1)
							)
					horizon_rows[local_step] = row
			phase_results[phase] = {
				"warmup_steps": int(warm_steps),
				"episodes_per_policy": per_policy,
				"horizons": {str(key): value for key, value in horizon_rows.items()},
			}
			print(f"[phase3.11] intervention phase={phase} complete", flush=True)
		return {
			"checkpoint_hashes_before": checkpoint_hashes,
			"clone_checks": clone_checks,
			"phase_results": phase_results,
			"common_random_numbers": True,
			"controller": "full checkpoint TD-MPC2/MPPI with identical RNG seeds per policy arm",
		}
	finally:
		env.close()


def _candidate_bank(model, cfg, obs: torch.Tensor, task: torch.Tensor, args, seed: int):
	count = int(args.num_candidates)
	policy_count = int(args.num_policy_candidates)
	horizon = int(args.candidate_horizon)
	generator = torch.Generator(device=obs.device).manual_seed(int(seed))
	with torch.random.fork_rng(devices=[obs.device.index]):
		torch.manual_seed(int(seed))
		torch.cuda.manual_seed_all(int(seed))
		policy_eps = torch.randn(
			obs.shape[0], horizon, policy_count, int(cfg.action_dim), generator=generator, device=obs.device,
		)
		policy = phase33._policy_proposals(
			model, obs, task[0], policy_eps, horizon=horizon, num_candidates=policy_count,
		)["actions"]
		gaussian = (float(cfg.max_std) * torch.randn(
			obs.shape[0], horizon, count - policy_count, int(cfg.action_dim),
			generator=generator, device=obs.device,
		)).clamp(-1.0, 1.0)
		candidates = torch.cat((policy, gaussian), dim=2)
		scores = phase33._score_candidates(model, cfg, obs, task[0], candidates)
	return candidates, scores


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
	x = x.float().reshape(-1)
	y = y.float().reshape(-1)
	if x.numel() < 2 or float(x.std(unbiased=False).item()) < 1.0e-8 or float(y.std(unbiased=False).item()) < 1.0e-8:
		return math.nan
	return float(torch.corrcoef(torch.stack((x, y)))[0, 1].item())


def _rank(values: torch.Tensor) -> torch.Tensor:
	order = torch.argsort(values, dim=-1)
	ranks = torch.empty_like(order, dtype=torch.float32)
	base = torch.arange(values.shape[-1], device=values.device, dtype=torch.float32)
	ranks.scatter_(-1, order, base.expand_as(order))
	return ranks


def _ranking_metrics(predicted: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
	predicted = predicted.detach().float().cpu()
	actual = actual.detach().float().cpu()
	pearson = [_pearson(left, right) for left, right in zip(predicted, actual)]
	spearman = [_pearson(_rank(left), _rank(right)) for left, right in zip(predicted, actual)]
	kendall = phase33._kendall_tau_batch(predicted, actual)
	return {
		"pearson_mean": float(torch.tensor(pearson).nanmean().item()),
		"spearman_mean": float(torch.tensor(spearman).nanmean().item()),
		"kendall_tau_mean": float(kendall.mean().item()),
	}


def _topk_realized(predicted: torch.Tensor, actual: torch.Tensor, k: int) -> dict[str, float]:
	k = min(int(k), int(predicted.shape[-1]))
	pred_idx = torch.topk(predicted, k=k, dim=1).indices
	actual_idx = torch.topk(actual, k=k, dim=1).indices
	pred_top_actual = actual.gather(1, pred_idx).mean(dim=1)
	oracle_top_actual = actual.gather(1, actual_idx).mean(dim=1)
	all_actual = actual.mean(dim=1)
	overlap = []
	for pred, oracle in zip(pred_idx.cpu(), actual_idx.cpu()):
		overlap.append(len(set(pred.tolist()) & set(oracle.tolist())) / max(1, k))
	return {
		"predicted_topk_actual_return_mean": float(pred_top_actual.mean().item()),
		"all_candidate_actual_return_mean": float(all_actual.mean().item()),
		"predicted_topk_gain_vs_all": float((pred_top_actual - all_actual).mean().item()),
		"oracle_topk_actual_return_mean": float(oracle_top_actual.mean().item()),
		"predicted_topk_regret_to_oracle": float((oracle_top_actual - pred_top_actual).mean().item()),
		"topk_overlap_with_actual": float(sum(overlap) / len(overlap)),
	}


def _prediction_reality_report(args: argparse.Namespace, checkpoints: OrderedDict[str, Path]):
	base_states = int(args.candidate_base_states)
	candidates_per_state = int(args.num_candidates)
	total_envs = base_states * candidates_per_state
	assembly_id = str(getattr(args, "assembly_id", "00186"))
	runtime_cfg = _configure_runtime_cfg(
		args, checkpoints["original"], assembly_id, total_envs, run_id="phase3_11_prediction_reality"
	)
	env = make_env(runtime_cfg)
	try:
		expected_dim = int(runtime_cfg.obs_shape["state"][0])
		device = env.unwrapped.device
		source_ids = torch.arange(base_states, device=device) * candidates_per_state
		target_groups = []
		for candidate_index in range(1, candidates_per_state):
			target_groups.append(source_ids + candidate_index)
		source_agent = collector._make_agent(_agent_cfg(runtime_cfg, checkpoints["original"], base_states))
		continuation_agent = collector._make_agent(_agent_cfg(runtime_cfg, checkpoints["original"], total_envs))
		phase_results = OrderedDict()
		clone_checks = []
		for phase_index, (phase, warm_steps) in enumerate(_phase_steps(int(runtime_cfg.episode_length)).items()):
			obs, clone = _warm_to_phase(
				env,
				source_agent,
				source_ids,
				target_groups,
				warm_steps,
				expected_dim,
				int(args.seed) + 300_000 + 10_000 * phase_index,
			)
			clone_checks.append({"phase": phase, **clone["branch"]})
			if clone["branch"]["max_controller_input_observation_abs_delta"] > float(args.clone_obs_tolerance):
				raise RuntimeError(f"Candidate observation clone mismatch at {phase}: {clone['branch']}")
			base_obs = obs[source_ids]
			base_task = _runtime_task(env, source_ids)
			candidates, predicted = _candidate_bank(
				source_agent.model,
				source_agent.cfg,
				base_obs,
				base_task,
				args,
				int(args.seed) + 400_000 + phase_index,
			)
			initial = _metrics(env)
			initial_depth = initial["current_depth"].reshape(base_states, candidates_per_state)
			actual_return = torch.zeros(base_states, candidates_per_state, device=device)
			actual_return3 = torch.zeros_like(actual_return)
			jam_any = torch.zeros_like(actual_return, dtype=torch.bool)
			discount = 1.0
			for step in range(int(args.candidate_horizon)):
				action = candidates[:, step].reshape(total_envs, int(runtime_cfg.action_dim))
				raw_obs, reward, _, _, _ = env.step(action)
				obs = collector._adapt_obs_to_checkpoint(raw_obs, expected_dim)
				actual_return += discount * reward.reshape(base_states, candidates_per_state)
				discount *= float(args.real_discount)
				jam_any |= _metrics(env)["jam"].reshape(base_states, candidates_per_state) > 0.5
			actual_return3.copy_(actual_return)
			_reset_agent(continuation_agent)
			for continuation_step in range(int(args.candidate_horizon), max(HORIZONS)):
				action = _matched_agent_action(
					continuation_agent,
					obs,
					_runtime_task(env),
					t0=(continuation_step == int(args.candidate_horizon)),
					seed=int(args.seed) + 500_000 + phase_index * 100 + continuation_step,
				)
				raw_obs, reward, _, _, _ = env.step(action)
				obs = collector._adapt_obs_to_checkpoint(raw_obs, expected_dim)
				actual_return += discount * reward.reshape(base_states, candidates_per_state)
				discount *= float(args.real_discount)
				jam_any |= _metrics(env)["jam"].reshape(base_states, candidates_per_state) > 0.5
			final = _metrics(env)
			final_depth = final["current_depth"].reshape(base_states, candidates_per_state)
			lateral = 1000.0 * final["lateral_error"].reshape(base_states, candidates_per_state)
			keypoint = 1000.0 * final["keypoint_error"].reshape(base_states, candidates_per_state)
			predicted_cpu = {key: value.detach().cpu() for key, value in predicted.items() if key in {"reward_sum", "terminal_q", "total"}}
			actual3_cpu = actual_return3.detach().cpu()
			actual5_cpu = actual_return.detach().cpu()
			correlations = OrderedDict()
			for key, value in predicted_cpu.items():
				correlations[key] = {
					"vs_actual_return_3": _ranking_metrics(value, actual3_cpu),
					"vs_actual_return_5": _ranking_metrics(value, actual5_cpu),
				}
			phase_results[phase] = {
				"warmup_steps": int(warm_steps),
				"base_states": base_states,
				"candidates_per_state": candidates_per_state,
				"candidate_composition": {
					"policy": int(args.num_policy_candidates),
					"gaussian": candidates_per_state - int(args.num_policy_candidates),
					"horizon": int(args.candidate_horizon),
					"topk": int(args.num_elites),
				},
				"ranking_correlations": correlations,
				"predicted_total_topk_realized": _topk_realized(
					predicted_cpu["total"], actual5_cpu, int(args.num_elites),
				),
				"real_outcomes": {
					"return_3": _summary(actual3_cpu),
					"return_5": _summary(actual5_cpu),
					"depth_progress_mm": _summary(1000.0 * (final_depth - initial_depth)),
					"lateral_error_mm": _summary(lateral),
					"keypoint_error_mm": _summary(keypoint),
					"jam_any_rate": float(jam_any.float().mean().item()),
				},
			}
			print(f"[phase3.11] prediction-reality phase={phase} complete", flush=True)
		return {
			"clone_checks": clone_checks,
			"phase_results": phase_results,
			"frozen_scorer": "original multitask world model",
			"real_return_definition": (
				"discounted environment reward for 3 candidate steps and 2 source-controller continuation steps"
			),
		}
	finally:
		env.close()


def _trajectory_pair_summary(
	obs_distance: torch.Tensor,
	latent_distance: torch.Tensor,
	lateral_delta_mm: torch.Tensor,
	keypoint_delta_mm: torch.Tensor,
	action_distance: torch.Tensor,
	*,
	args: argparse.Namespace,
) -> dict[str, Any]:
	# Inputs are [T, E].
	diverged = (
		(latent_distance >= float(args.divergence_latent_l2))
		| (lateral_delta_mm >= float(args.divergence_lateral_mm))
		| (keypoint_delta_mm >= float(args.divergence_keypoint_mm))
	)
	first = []
	for env_index in range(diverged.shape[1]):
		idx = torch.nonzero(diverged[:, env_index], as_tuple=False).reshape(-1)
		first.append(float(idx[0].item()) if idx.numel() else math.nan)
	finite_first = [value for value in first if math.isfinite(value)]
	return {
		"obs_l2": _summary(obs_distance),
		"common_source_latent_l2": _summary(latent_distance),
		"lateral_abs_delta_mm": _summary(lateral_delta_mm),
		"keypoint_abs_delta_mm": _summary(keypoint_delta_mm),
		"executed_action_l2": _summary(action_distance),
		"diverged_episode_fraction": len(finite_first) / max(1, len(first)),
		"first_divergence_step": _summary(finite_first),
		"thresholds": {
			"latent_l2": float(args.divergence_latent_l2),
			"lateral_mm": float(args.divergence_lateral_mm),
			"keypoint_mm": float(args.divergence_keypoint_mm),
		},
	}


def _distribution_shift_report(args: argparse.Namespace, checkpoints: OrderedDict[str, Path]):
	per_policy = int(args.distribution_episodes)
	total_envs = per_policy * len(CHECKPOINT_LABELS)
	runtime_cfg = _configure_runtime_cfg(
		args, checkpoints["original"], "00186", total_envs, run_id="phase3_11_distribution_shift"
	)
	env = make_env(runtime_cfg)
	try:
		expected_dim = int(runtime_cfg.obs_shape["state"][0])
		device = env.unwrapped.device
		ids = OrderedDict([
			("original", torch.arange(per_policy, device=device)),
			("elite_only", torch.arange(per_policy, 2 * per_policy, device=device)),
			("anchor_l3", torch.arange(2 * per_policy, 3 * per_policy, device=device)),
		])
		agents = _make_agents(runtime_cfg, checkpoints, per_policy)
		obs, _ = env.reset()
		obs = collector._adapt_obs_to_checkpoint(obs, expected_dim)
		obs, clone = _clone_groups(env, ids["original"], [ids["elite_only"], ids["anchor_l3"]], expected_dim)
		if clone["max_controller_input_observation_abs_delta"] > float(args.clone_obs_tolerance):
			raise RuntimeError(f"Distribution initial clone mismatch: {clone}")
		for agent in agents.values():
			_reset_agent(agent)
		rows: dict[str, list[torch.Tensor]] = {
			"elite_obs": [], "anchor_obs": [], "elite_latent": [], "anchor_latent": [],
			"elite_lateral": [], "anchor_lateral": [], "elite_keypoint": [], "anchor_keypoint": [],
			"elite_action": [], "anchor_action": [],
		}
		terminal = {label: {} for label in CHECKPOINT_LABELS}
		for step in range(int(runtime_cfg.episode_length)):
			common_seed = int(args.seed) + 600_000 + step
			actions = OrderedDict()
			for label, agent in agents.items():
				actions[label] = _matched_agent_action(
					agent,
					obs[ids[label]],
					_runtime_task(env, ids[label]),
					t0=(step == 0),
					seed=common_seed,
				)
			with torch.no_grad():
				base_task = _runtime_task(env, ids["original"])
				z_original = agents["original"].model.encode(obs[ids["original"]], base_task)
				z_elite = agents["original"].model.encode(obs[ids["elite_only"]], base_task)
				z_anchor = agents["original"].model.encode(obs[ids["anchor_l3"]], base_task)
			metrics = _metrics(env)
			rows["elite_obs"].append(torch.linalg.vector_norm(
				obs[ids["elite_only"]] - obs[ids["original"]], dim=-1,
			).cpu())
			rows["anchor_obs"].append(torch.linalg.vector_norm(
				obs[ids["anchor_l3"]] - obs[ids["original"]], dim=-1,
			).cpu())
			rows["elite_latent"].append(torch.linalg.vector_norm(z_elite - z_original, dim=-1).cpu())
			rows["anchor_latent"].append(torch.linalg.vector_norm(z_anchor - z_original, dim=-1).cpu())
			for label, prefix in (("elite_only", "elite"), ("anchor_l3", "anchor")):
				rows[f"{prefix}_lateral"].append((1000.0 * (
					metrics["lateral_error"][ids[label]] - metrics["lateral_error"][ids["original"]]
				).abs()).cpu())
				rows[f"{prefix}_keypoint"].append((1000.0 * (
					metrics["keypoint_error"][ids[label]] - metrics["keypoint_error"][ids["original"]]
				).abs()).cpu())
				rows[f"{prefix}_action"].append(torch.linalg.vector_norm(
					actions[label] - actions["original"], dim=-1,
				).cpu())
			raw_obs, _, terminated, truncated, info = env.step(torch.cat(list(actions.values()), dim=0))
			obs = collector._adapt_obs_to_checkpoint(raw_obs, expected_dim)
			if step == int(runtime_cfg.episode_length) - 1:
				final_info = info.get("final_info", {}) if isinstance(info, dict) else {}
				current = _metrics(env)
				for label, label_ids in ids.items():
					terminal[label] = {}
					for key in ("relaxed_success_stable", "strict_success_stable", "process_success_terminal", "lateral_error", "keypoint_error", "jam"):
						value = final_info.get(key, None)
						if value is None:
							value = current.get(key, torch.zeros(total_envs, device=device))
						selected = torch.nan_to_num(value[label_ids].float(), nan=0.0)
						terminal[label][key] = float(selected.mean().item())
		stacked = {key: torch.stack(value, dim=0) for key, value in rows.items()}
		pairs = OrderedDict()
		for label, prefix in (("elite_only", "elite"), ("anchor_l3", "anchor")):
			pairs[label] = _trajectory_pair_summary(
				stacked[f"{prefix}_obs"],
				stacked[f"{prefix}_latent"],
				stacked[f"{prefix}_lateral"],
				stacked[f"{prefix}_keypoint"],
				stacked[f"{prefix}_action"],
				args=args,
			)
		return {
			"initial_clone": clone,
			"episodes_per_policy": per_policy,
			"episode_steps": int(runtime_cfg.episode_length),
			"pairs_vs_original": pairs,
			"terminal_metrics": terminal,
			"latent_encoder": "frozen original checkpoint encoder for all policy state trajectories",
		}
	finally:
		env.close()


def _success_tensor(final_info, current, key: str, num_envs: int, device: torch.device) -> torch.Tensor:
	value = final_info.get(key, None)
	if value is None:
		value = current.get(key, torch.zeros(num_envs, device=device))
	return torch.nan_to_num(value.reshape(-1).float(), nan=0.0)


def _old_task_anchor_one(
	args: argparse.Namespace,
	checkpoints: OrderedDict[str, Path],
	assembly_id: str,
):
	num_envs = int(args.old_task_episodes)
	runtime_cfg = _configure_runtime_cfg(
		args, checkpoints["original"], assembly_id, num_envs, run_id=f"phase3_11_anchor_{assembly_id}"
	)
	env = make_env(runtime_cfg)
	try:
		expected_dim = int(runtime_cfg.obs_shape["state"][0])
		device = env.unwrapped.device
		agents = _make_agents(runtime_cfg, checkpoints, num_envs)
		for agent in agents.values():
			_reset_agent(agent)
		obs, _ = env.reset()
		obs = collector._adapt_obs_to_checkpoint(obs, expected_dim)
		drift_rows = {"elite_only": [], "anchor_l3": []}
		phase_rows = []
		terminal_strict = torch.zeros(num_envs)
		terminal_process = torch.zeros(num_envs)
		for step in range(int(runtime_cfg.episode_length)):
			task = _runtime_task(env)
			means = {
				label: _policy_mean(agent.model, obs, task)
				for label, agent in agents.items()
			}
			for label in ("elite_only", "anchor_l3"):
				drift_rows[label].append(torch.linalg.vector_norm(
					means[label] - means["original"], dim=-1,
				).cpu())
			phase_rows.append(_phase_from_runtime(
				_metrics(env), step, int(runtime_cfg.episode_length),
			).cpu())
			action = _matched_agent_action(
				agents["original"],
				obs,
				task,
				t0=(step == 0),
				seed=int(args.seed) + 700_000 + 1000 * OLD_TASKS.index(assembly_id) + step,
			)
			raw_obs, _, _, _, info = env.step(action)
			obs = collector._adapt_obs_to_checkpoint(raw_obs, expected_dim)
			if step == int(runtime_cfg.episode_length) - 1:
				final_info = info.get("final_info", {}) if isinstance(info, dict) else {}
				current = _metrics(env)
				terminal_strict = _success_tensor(
					final_info, current, "strict_success_stable", num_envs, device,
				).cpu()
				terminal_process = _success_tensor(
					final_info, current, "process_success_terminal", num_envs, device,
				).cpu()
		phase = torch.stack(phase_rows, dim=0)
		result = OrderedDict()
		for label in ("elite_only", "anchor_l3"):
			drift = torch.stack(drift_rows[label], dim=0)
			groups = OrderedDict([("all", torch.ones_like(phase, dtype=torch.bool))])
			for phase_id, phase_name in enumerate(PHASES):
				groups[f"phase/{phase_name}"] = phase == phase_id
			groups["strict_success_trajectory"] = terminal_strict.reshape(1, -1).expand_as(phase) > 0.5
			groups["process_success_trajectory"] = terminal_process.reshape(1, -1).expand_as(phase) > 0.5
			result[label] = {
				name: _summary(drift[mask])
				for name, mask in groups.items()
				if bool(mask.any().item())
			}
		return {
			"episodes": num_envs,
			"original_rollout_success": {
				"strict_stable_rate": float(terminal_strict.mean().item()),
				"process_terminal_rate": float(terminal_process.mean().item()),
			},
			"policy_prior_action_l2_vs_original": result,
			"phase_definition": "runtime contact/depth thresholds with Phase 3.3 temporal fallback",
		}
	finally:
		env.close()


def _old_task_anchor_report(args: argparse.Namespace, checkpoints: OrderedDict[str, Path]):
	result = OrderedDict()
	for task in OLD_TASKS:
		result[task] = _old_task_anchor_one(args, checkpoints, task)
		print(f"[phase3.11] old-task anchor task={task} complete", flush=True)
	return result


def _phase310_closed_loop(report: dict[str, Any]) -> dict[str, Any]:
	result: dict[str, Any] = {}
	for label, aliases in {
		"original": ("original_checkpoint", "original"),
		"elite_only": ("elite_only",),
		"anchor_l3": ("elite_behavior_anchor_l3", "anchor_l3"),
	}.items():
		variant = None
		variants = report.get("variants", {})
		for alias in aliases:
			if alias in variants:
				variant = variants[alias]
				break
		if variant is not None:
			result[label] = variant.get("closed_loop", variant)
	return result


def _classify(report: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
	prediction = report["prediction_reality"]["phase_results"]
	exploitation_phases = []
	for phase in PHASES:
		item = prediction[phase]
		correlation = float(item["ranking_correlations"]["total"]["vs_actual_return_5"]["spearman_mean"])
		gain = float(item["predicted_total_topk_realized"]["predicted_topk_gain_vs_all"])
		if correlation <= float(args.exploitation_spearman_max) or gain <= float(args.exploitation_topk_gain_max):
			exploitation_phases.append(phase)
	exploitation = any(phase in exploitation_phases for phase in ("contact", "insertion"))

	distribution_pairs = report["distribution_shift"]["pairs_vs_original"]
	distribution_hits = [
		label for label, item in distribution_pairs.items()
		if float(item["diverged_episode_fraction"]) >= float(args.distribution_diverged_fraction_min)
	]
	distribution = "anchor_l3" in distribution_hits

	critical_anchor_hits = []
	for task, task_report in report["old_task_anchor"].items():
		groups = task_report["policy_prior_action_l2_vs_original"]["anchor_l3"]
		for group in ("phase/contact", "phase/insertion", "strict_success_trajectory", "process_success_trajectory"):
			item = groups.get(group)
			if item is None:
				continue
			if float(item["p95"]) >= float(args.anchor_critical_p95) or float(item["max"]) >= float(args.anchor_critical_max):
				critical_anchor_hits.append(f"{task}/{group}")
	anchor_coverage = bool(critical_anchor_hits)

	causes = []
	if exploitation:
		causes.append("ELITE_TARGET_MODEL_EXPLOITATION")
	if distribution:
		causes.append("POLICY_STATE_DISTRIBUTION_SHIFT")
	if anchor_coverage:
		causes.append("ANCHOR_CRITICAL_STATE_COVERAGE_FAILURE")
	classification = causes[0] if len(causes) == 1 else "MIXED_OR_UNRESOLVED"
	return {
		"classification": classification,
		"supported_causes": causes,
		"elite_target_model_exploitation": {
			"supported": exploitation,
			"triggered_phases": exploitation_phases,
		},
		"policy_state_distribution_shift": {
			"supported": distribution,
			"triggered_variants": distribution_hits,
		},
		"anchor_critical_state_coverage_failure": {
			"supported": anchor_coverage,
			"triggered_groups": critical_anchor_hits,
		},
		"reason": (
			"Multiple mechanisms are supported by the read-only diagnostics."
			if len(causes) > 1 else
			("No mechanism cleared its evidence threshold." if not causes else f"Only {causes[0]} cleared its evidence threshold.")
		),
	}


def _fmt(value: Any, digits: int = 3) -> str:
	try:
		value = float(value)
	except (TypeError, ValueError):
		return "NA"
	return "NA" if not math.isfinite(value) else f"{value:.{digits}f}"


def _markdown(report: dict[str, Any]) -> str:
	classification = report["classification"]
	lines = [
		"# SRSA Phase 3.11 Offline-to-Closed-Loop Gap Diagnosis",
		"",
		"本报告仅做只读评估；未训练、未修改 checkpoint、未调 lambda，也未修改 reward、Q、dynamics、task_context、MPPI 或 replay sampler。",
		"",
		f"Status: `{report['status']}`",
		f"Final classification: `{classification['classification']}`",
		"",
		"## Conclusion",
		"",
		classification["reason"],
		"",
		f"- Supported causes: `{classification['supported_causes']}`.",
		f"- Phase 3.10 parent conclusion: `{report['phase310_parent_classification']}`.",
		"- Offline proposal regret and mean old-task action drift are not treated as closed-loop predictors.",
		"",
		"## 1. Same-State Action Intervention",
		"",
		"Each phase starts from cloned writable physical/task state. All three full TD-MPC2/MPPI controllers use matched random numbers.",
		"",
		"| Phase | Horizon | Policy | Reward sum | Depth progress mm | Lateral mm | Keypoint mm | Contact | Jam | Action L2 vs original |",
		"| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
	]
	for phase, phase_item in report["same_state_intervention"]["phase_results"].items():
		for horizon, horizon_item in phase_item["horizons"].items():
			for label, item in horizon_item.items():
				lines.append(
					f"| `{phase}` | {horizon} | `{label}` | {_fmt(item['reward_sum']['mean'])} | "
					f"{_fmt(item['depth_progress_mm']['mean'])} | {_fmt(item['lateral_error_mm']['mean'])} | "
					f"{_fmt(item['keypoint_error_mm']['mean'])} | {_fmt(item['contact_rate'])} | "
					f"{_fmt(item['jam_rate'])} | {_fmt(item.get('executed_action_l2_vs_original', {}).get('mean'))} |"
				)
	lines.extend([
		"",
		"### Clone validation and sensor-cache control",
		"",
		"Writable physical state is cloned to numerical precision. Force-sensor and finite-difference observation caches are not writable PhysX state, so the source observation is copied only for each branch's first controller input; every later input is the branch's live simulator observation.",
		"",
		"| Audit | Phase | Max physical-state residual | Raw observation residual | First controller-input residual |",
		"| --- | --- | ---: | ---: | ---: |",
	])
	for audit, checks in (
		("action_intervention", report["same_state_intervention"]["clone_checks"]),
		("prediction_reality", report["prediction_reality"]["clone_checks"]),
	):
		for check in checks:
			lines.append(
				f"| `{audit}` | `{check['phase']}` | {_fmt(check['max_physical_state_abs_delta'], 9)} | "
				f"{_fmt(check['max_raw_observation_abs_delta'], 3)} | "
				f"{_fmt(check['max_controller_input_observation_abs_delta'], 9)} |"
			)
	initial_clone = report["distribution_shift"]["initial_clone"]
	lines.append(
		f"| `distribution_shift` | `initial` | {_fmt(initial_clone['max_physical_state_abs_delta'], 9)} | "
		f"{_fmt(initial_clone['max_raw_observation_abs_delta'], 3)} | "
		f"{_fmt(initial_clone['max_controller_input_observation_abs_delta'], 9)} |"
	)
	lines.extend([
		"",
		"## 2. Frozen Prediction vs Real Outcome",
		"",
		"Candidate contract: 3 source-policy trajectories + 61 Gaussian trajectories, horizon 3, frozen source scorer; all 64 candidates are restored into cloned simulator states.",
		"",
		"| Phase | Predicted component | Spearman vs real-3 | Spearman vs real-5 | Kendall vs real-5 | Pred top-8 gain vs all | Top-8 overlap with actual |",
		"| --- | --- | ---: | ---: | ---: | ---: | ---: |",
	])
	for phase, item in report["prediction_reality"]["phase_results"].items():
		for component, correlation in item["ranking_correlations"].items():
			topk = item["predicted_total_topk_realized"]
			lines.append(
				f"| `{phase}` | `{component}` | {_fmt(correlation['vs_actual_return_3']['spearman_mean'])} | "
				f"{_fmt(correlation['vs_actual_return_5']['spearman_mean'])} | "
				f"{_fmt(correlation['vs_actual_return_5']['kendall_tau_mean'])} | "
				f"{_fmt(topk['predicted_topk_gain_vs_all']) if component == 'total' else '—'} | "
				f"{_fmt(topk['topk_overlap_with_actual']) if component == 'total' else '—'} |"
			)
	lines.extend([
		"",
		"## 3. Closed-Loop State Distribution Drift",
		"",
		"| Variant vs original | Obs L2 | Common latent L2 | Lateral delta mm | Keypoint delta mm | Action L2 | Diverged episodes | Median first divergence step |",
		"| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
	])
	for label, item in report["distribution_shift"]["pairs_vs_original"].items():
		lines.append(
			f"| `{label}` | {_fmt(item['obs_l2']['mean'])} | {_fmt(item['common_source_latent_l2']['mean'])} | "
			f"{_fmt(item['lateral_abs_delta_mm']['mean'])} | {_fmt(item['keypoint_abs_delta_mm']['mean'])} | "
			f"{_fmt(item['executed_action_l2']['mean'])} | {_fmt(item['diverged_episode_fraction'])} | "
			f"{_fmt(item['first_divergence_step']['p50'], 1)} |"
		)
	lines.extend([
		"",
		"## 4. Old-Task Anchor Critical-State Coverage",
		"",
		"Drift is deterministic policy-prior mean action L2 on states visited by the original closed-loop controller.",
		"",
		"| Task | Variant | State group | Rows | Mean | P95 | Max |",
		"| --- | --- | --- | ---: | ---: | ---: | ---: |",
	])
	for task, task_item in report["old_task_anchor"].items():
		success = task_item["original_rollout_success"]
		lines.insert(
			-2,
			f"- `{task}` original rollouts: strict-stable success `{_fmt(success['strict_stable_rate'])}`, "
			f"process-terminal success `{_fmt(success['process_terminal_rate'])}`.",
		)
	lines.insert(-2, "")
	for task, task_item in report["old_task_anchor"].items():
		for label, groups in task_item["policy_prior_action_l2_vs_original"].items():
			for group, item in groups.items():
				lines.append(
					f"| `{task}` | `{label}` | `{group}` | {item['count']} | {_fmt(item['mean'], 4)} | "
					f"{_fmt(item['p95'], 4)} | {_fmt(item['max'], 4)} |"
				)
	lines.extend([
		"",
		"## Classification Evidence",
		"",
		f"- Elite target/model exploitation: `{classification['elite_target_model_exploitation']}`.",
		f"- Policy state-distribution shift: `{classification['policy_state_distribution_shift']}`.",
		f"- Anchor critical-state coverage failure: `{classification['anchor_critical_state_coverage_failure']}`.",
		"",
		"## Safety And Reproducibility",
		"",
		f"- Device contract: `{report['device_contract']}`.",
		f"- Checkpoints unchanged after the audit: `{report['checkpoint_integrity']['all_unchanged']}`.",
		f"- First controller-input clone tolerance: `{report['settings']['clone_obs_tolerance']}`.",
		"- No checkpoint write path is used. The Phase 3.10 source checkpoint is the frozen scorer and world model.",
		"- The three Phase 3.10 checkpoints differ only in policy tensors; this script performs no optimizer step and no checkpoint save.",
		"- Real candidate return is a short-horizon diagnostic, not a replacement for full-episode closed-loop evaluation.",
	])
	return "\n".join(lines) + "\n"


def _clone_smoke(args: argparse.Namespace, checkpoints: OrderedDict[str, Path]) -> dict[str, Any]:
	num_envs = 6
	cfg = _configure_runtime_cfg(args, checkpoints["original"], "00186", num_envs, run_id="phase3_11_clone_smoke")
	env = make_env(cfg)
	try:
		expected_dim = int(cfg.obs_shape["state"][0])
		device = env.unwrapped.device
		env.reset()
		source = torch.tensor([0, 1], device=device)
		target_a = torch.tensor([2, 3], device=device)
		target_b = torch.tensor([4, 5], device=device)
		_, result = _clone_groups(env, source, [target_a, target_b], expected_dim)
		return result
	finally:
		env.close()


def build_report(args: argparse.Namespace) -> dict[str, Any]:
	_require_physical_cuda1(args)
	checkpoints = _checkpoint_paths(args)
	phase310 = _load_json(args.phase310_report)
	hashes_before = {label: _sha256(path) for label, path in checkpoints.items()}
	set_seed(int(args.seed))

	intervention = _intervention_report(args, checkpoints, hashes_before)
	prediction_reality = _prediction_reality_report(args, checkpoints)
	distribution_shift = _distribution_shift_report(args, checkpoints)
	old_task_anchor = _old_task_anchor_report(args, checkpoints)

	hashes_after = {label: _sha256(path) for label, path in checkpoints.items()}
	checkpoint_integrity = {
		"before": hashes_before,
		"after": hashes_after,
		"unchanged": {label: hashes_before[label] == hashes_after[label] for label in checkpoints},
	}
	checkpoint_integrity["all_unchanged"] = all(checkpoint_integrity["unchanged"].values())
	if not checkpoint_integrity["all_unchanged"]:
		raise RuntimeError(f"Checkpoint integrity failure: {checkpoint_integrity}")

	report = {
		"status": "WARNING",
		"phase310_parent_classification": "OFFLINE_TO_CLOSED_LOOP_TRANSFER_FAILURE",
		"device_contract": {
			"physical_device": "cuda1",
			"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
			"logical_device": "cuda:0",
			"name": torch.cuda.get_device_name(0),
		},
		"inputs": {
			"checkpoints": {label: str(path) for label, path in checkpoints.items()},
			"phase310_report": str(resolve(args.phase310_report)),
			"phase310_closed_loop": _phase310_closed_loop(phase310),
		},
		"settings": {
			"intervention_episodes": int(args.intervention_episodes),
			"distribution_episodes": int(args.distribution_episodes),
			"old_task_episodes": int(args.old_task_episodes),
			"candidate_base_states": int(args.candidate_base_states),
			"num_candidates": int(args.num_candidates),
			"num_policy_candidates": int(args.num_policy_candidates),
			"num_elites": int(args.num_elites),
			"candidate_horizon": int(args.candidate_horizon),
			"clone_obs_tolerance": float(args.clone_obs_tolerance),
			"seed": int(args.seed),
		},
		"same_state_intervention": intervention,
		"prediction_reality": prediction_reality,
		"distribution_shift": distribution_shift,
		"old_task_anchor": old_task_anchor,
		"checkpoint_integrity": checkpoint_integrity,
		"prohibitions_observed": {
			"trained": False,
			"checkpoint_written": False,
			"lambda_tuned": False,
			"consolidation_started": False,
			"task_added": False,
			"model_or_planner_modified": False,
		},
	}
	report["classification"] = _classify(report, args)
	report["status"] = "PASS" if report["classification"]["classification"] != "MIXED_OR_UNRESOLVED" else "WARNING"
	return report


def _partial_path(args: argparse.Namespace, section: str) -> Path:
	return resolve(Path(args.partial_dir) / f"{section}.json")


def _write_runtime_partial(
	args: argparse.Namespace,
	section: str,
	checkpoints: OrderedDict[str, Path],
	result: dict[str, Any],
) -> None:
	hashes_after = {label: _sha256(path) for label, path in checkpoints.items()}
	hashes_before = getattr(args, "_section_hashes_before")
	unchanged = {label: hashes_before[label] == hashes_after[label] for label in checkpoints}
	if not all(unchanged.values()):
		raise RuntimeError(f"Checkpoint integrity failure in section {section}: {unchanged}")
	write_json({
		"section": section,
		"device_contract": {
			"physical_device": "cuda1",
			"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
			"logical_device": "cuda:0",
			"name": torch.cuda.get_device_name(0),
		},
		"checkpoint_hashes_before": hashes_before,
		"checkpoint_hashes_after": hashes_after,
		"checkpoints_unchanged": unchanged,
		"result": result,
	}, _partial_path(args, section))


def _aggregate_partials(args: argparse.Namespace, checkpoints: OrderedDict[str, Path]) -> dict[str, Any]:
	parts = {
		section: _load_json(_partial_path(args, section))
		for section in ("intervention", "prediction_reality", "distribution_shift", "anchor_01125", "anchor_00256")
	}
	current_hashes = {label: _sha256(path) for label, path in checkpoints.items()}
	section_integrity = {}
	for section, part in parts.items():
		before = part.get("checkpoint_hashes_before", {})
		after = part.get("checkpoint_hashes_after", {})
		section_integrity[section] = bool(
			before == after == current_hashes
			and all(part.get("checkpoints_unchanged", {}).values())
		)
	if not all(section_integrity.values()):
		raise RuntimeError(f"Partial checkpoint-integrity mismatch: {section_integrity}")

	phase310 = _load_json(args.phase310_report)
	report = {
		"status": "WARNING",
		"phase310_parent_classification": "OFFLINE_TO_CLOSED_LOOP_TRANSFER_FAILURE",
		"device_contract": parts["intervention"]["device_contract"],
		"inputs": {
			"checkpoints": {label: str(path) for label, path in checkpoints.items()},
			"phase310_report": str(resolve(args.phase310_report)),
			"phase310_closed_loop": _phase310_closed_loop(phase310),
			"partial_reports": {section: str(_partial_path(args, section)) for section in parts},
		},
		"settings": {
			"intervention_episodes": int(args.intervention_episodes),
			"distribution_episodes": int(args.distribution_episodes),
			"old_task_episodes": int(args.old_task_episodes),
			"candidate_base_states": int(args.candidate_base_states),
			"num_candidates": int(args.num_candidates),
			"num_policy_candidates": int(args.num_policy_candidates),
			"num_elites": int(args.num_elites),
			"candidate_horizon": int(args.candidate_horizon),
			"clone_obs_tolerance": float(args.clone_obs_tolerance),
			"seed": int(args.seed),
		},
		"same_state_intervention": parts["intervention"]["result"],
		"prediction_reality": parts["prediction_reality"]["result"],
		"distribution_shift": parts["distribution_shift"]["result"],
		"old_task_anchor": OrderedDict([
			("01125", parts["anchor_01125"]["result"]),
			("00256", parts["anchor_00256"]["result"]),
		]),
		"checkpoint_integrity": {
			"current": current_hashes,
			"section_integrity": section_integrity,
			"all_unchanged": all(section_integrity.values()),
		},
		"prohibitions_observed": {
			"trained": False,
			"checkpoint_written": False,
			"lambda_tuned": False,
			"consolidation_started": False,
			"task_added": False,
			"model_or_planner_modified": False,
		},
	}
	report["classification"] = _classify(report, args)
	report["status"] = "PASS" if report["classification"]["classification"] != "MIXED_OR_UNRESOLVED" else "WARNING"
	return report


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--source-checkpoint", default=DEFAULT_SOURCE_CHECKPOINT)
	parser.add_argument("--elite-checkpoint", default=DEFAULT_ELITE_CHECKPOINT)
	parser.add_argument("--anchor-checkpoint", default=DEFAULT_ANCHOR_CHECKPOINT)
	parser.add_argument("--phase310-report", default=DEFAULT_PHASE310_REPORT)
	parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
	parser.add_argument("--output-md", default=str(DEFAULT_OUTPUT_MD))
	parser.add_argument("--partial-dir", default=str(DEFAULT_PARTIAL_DIR))
	parser.add_argument(
		"--section",
		choices=("all", "intervention", "prediction_reality", "distribution_shift", "anchor_01125", "anchor_00256", "aggregate"),
		default="all",
	)
	parser.add_argument("--config", default="configs/train/srsa_01125_imitation_relaxed.yaml")
	parser.add_argument("--task-template-fp", default="data/srsa_axial_task_templates.json")
	parser.add_argument("--mesh-geometry-fp", default="data/srsa_mesh_geometry_params.csv")
	parser.add_argument("--isaaclab-dir", default="/home/gpuserver/IsaacLab")
	parser.add_argument("--srsa-dir", default="/home/gpuserver/hx/github/srsa")
	parser.add_argument("--gpu-id", type=int, default=0)
	parser.add_argument("--intervention-episodes", type=int, default=8)
	parser.add_argument("--distribution-episodes", type=int, default=8)
	parser.add_argument("--old-task-episodes", type=int, default=20)
	parser.add_argument("--candidate-base-states", type=int, default=4)
	parser.add_argument("--num-candidates", type=int, default=64)
	parser.add_argument("--num-policy-candidates", type=int, default=3)
	parser.add_argument("--num-elites", type=int, default=8)
	parser.add_argument("--candidate-horizon", type=int, default=3)
	parser.add_argument("--real-discount", type=float, default=0.99)
	parser.add_argument("--clone-obs-tolerance", type=float, default=1.0e-4)
	parser.add_argument("--divergence-latent-l2", type=float, default=0.10)
	parser.add_argument("--divergence-lateral-mm", type=float, default=0.50)
	parser.add_argument("--divergence-keypoint-mm", type=float, default=1.00)
	parser.add_argument("--exploitation-spearman-max", type=float, default=0.10)
	parser.add_argument("--exploitation-topk-gain-max", type=float, default=0.0)
	parser.add_argument("--distribution-diverged-fraction-min", type=float, default=0.50)
	parser.add_argument("--anchor-critical-p95", type=float, default=0.05)
	parser.add_argument("--anchor-critical-max", type=float, default=0.10)
	parser.add_argument("--seed", type=int, default=1)
	parser.add_argument("--dry-run", action="store_true")
	parser.add_argument("--clone-smoke", action="store_true")
	args = parser.parse_args()

	if int(args.num_candidates) != 64 or int(args.num_policy_candidates) != 3:
		raise ValueError("Phase 3.11 must preserve the Phase 3.10 candidate contract: 3 policy + 61 Gaussian.")
	if int(args.num_elites) != 8 or int(args.candidate_horizon) != 3:
		raise ValueError("Phase 3.11 must preserve top-8, horizon-3 elite scoring.")
	if any(value <= 0 for value in (
		args.intervention_episodes, args.distribution_episodes, args.old_task_episodes, args.candidate_base_states,
	)):
		raise ValueError("All episode/base-state counts must be positive.")

	checkpoints = _checkpoint_paths(args)
	if args.dry_run:
		print("PASS dry-run")
		print("Required device: physical CUDA1 via CUDA_VISIBLE_DEVICES=1, logical --gpu-id 0")
		for label, path in checkpoints.items():
			print(f"{label}: {path}")
		print(f"Would write: {resolve(args.output_json)}")
		print(f"Would write: {resolve(args.output_md)}")
		return 0

	_require_physical_cuda1(args)
	if args.clone_smoke:
		result = _clone_smoke(args, checkpoints)
		print(json.dumps(result, indent=2))
		return 0 if (
			result["max_physical_state_abs_delta"] <= float(args.clone_obs_tolerance)
			and result["max_controller_input_observation_abs_delta"] <= float(args.clone_obs_tolerance)
		) else 1

	if args.section != "all":
		if args.section == "aggregate":
			report = _aggregate_partials(args, checkpoints)
			write_json(report, args.output_json)
			write_text(_markdown(report), args.output_md)
			print(report["status"])
			print(f"Final classification: {report['classification']['classification']}")
			print(f"Checkpoints unchanged: {report['checkpoint_integrity']['all_unchanged']}")
			return 0 if report["status"] != "FAIL" else 1
		args._section_hashes_before = {label: _sha256(path) for label, path in checkpoints.items()}
		if args.section == "intervention":
			result = _intervention_report(args, checkpoints, args._section_hashes_before)
		elif args.section == "prediction_reality":
			result = _prediction_reality_report(args, checkpoints)
		elif args.section == "distribution_shift":
			result = _distribution_shift_report(args, checkpoints)
		elif args.section == "anchor_01125":
			result = _old_task_anchor_one(args, checkpoints, "01125")
		elif args.section == "anchor_00256":
			result = _old_task_anchor_one(args, checkpoints, "00256")
		else:
			raise AssertionError(args.section)
		_write_runtime_partial(args, args.section, checkpoints, result)
		print(f"PASS section={args.section}")
		return 0

	report = build_report(args)
	write_json(report, args.output_json)
	write_text(_markdown(report), args.output_md)
	print(report["status"])
	print(f"Final classification: {report['classification']['classification']}")
	print(f"Checkpoints unchanged: {report['checkpoint_integrity']['all_unchanged']}")
	return 0 if report["status"] != "FAIL" else 1


if __name__ == "__main__":
	raise SystemExit(main())
