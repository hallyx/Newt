#!/usr/bin/env python3
"""Read-only Phase 3.4 fixed-budget proposal rescue ablation for 00186.

This script never updates checkpoint weights.  It evaluates five candidate-bank
compositions with the multitask world-model scorer, then validates them in
headless 00186 rollouts.  Direct policy proposals are diagnostic oracle inputs
only and are never treated as a deployable controller.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
MODEL_AUDIT_DIR = SCRIPT_DIR.parent / "model_task_sensitivity"
if str(MODEL_AUDIT_DIR) not in sys.path:
	sys.path.insert(0, str(MODEL_AUDIT_DIR))

from _common import (  # noqa: E402
	condition_batch,
	resolve,
	summarize_tensor,
	td_math,
	tensor_to_list,
	tvsr,
	write_json,
	write_text,
)

import collect_eval_rollouts as collector  # noqa: E402
import planner_action_attribution_diagnosis as phase33  # noqa: E402
from common import set_seed  # noqa: E402
from config import apply_eval_task_template, parse_cfg  # noqa: E402
from envs import make_env  # noqa: E402


DEFAULT_PHASE32_DIAGNOSIS = (
	"reports/phase3_three_task_pilot/phase3_2_diagnosis/"
	"standalone_vs_multitask_diagnosis.json"
)
DEFAULT_PHASE33_ROLLOUT_ROOT = "reports/phase3_three_task_pilot/phase3_3_rollouts"
DEFAULT_OUTPUT_JSON = "reports/phase3_three_task_pilot/phase3_4_proposal_rescue_ablation.json"
DEFAULT_OUTPUT_MD = "reports/phase3_three_task_pilot/phase3_4_proposal_rescue_ablation.md"
ASSEMBLY_ID = "00186"
TASK_TEMPLATE_ID = 2
REGIONS = ("easy", "default", "hard")
GROUPS = ("contact", "jam")


@dataclass(frozen=True)
class MethodSpec:
	name: str
	display_name: str
	multitask_policy_count: int
	gaussian_count: int
	random_count: int
	direct_policy_count: int
	policy_std_scale: float = 1.0
	is_oracle: bool = False


@dataclass
class PlannerState:
	prev_mean: torch.Tensor
	generator: torch.Generator
	inference_ms_total: float = 0.0
	inference_calls: int = 0


def _load_json(path: str | Path) -> dict[str, Any]:
	path = resolve(path)
	if not path.exists():
		raise FileNotFoundError(path)
	return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint_paths(args: argparse.Namespace) -> tuple[Path, Path]:
	diagnosis = _load_json(args.phase32_diagnosis)
	checkpoints = diagnosis.get("checkpoints") or {}
	direct = resolve(args.direct_checkpoint or checkpoints.get("direct_finetune", ""))
	multitask = resolve(args.multitask_checkpoint or checkpoints.get("multitask_rescue_best", ""))
	for label, path in (("direct", direct), ("multitask", multitask)):
		if not path.exists():
			raise FileNotFoundError(f"Missing {label} checkpoint: {path}")
	return direct, multitask


def _load_model(checkpoint: Path, args: argparse.Namespace, device: torch.device, batch_size: int):
	cfg_args = SimpleNamespace(
		config=args.config,
		gpu_id=args.gpu_id,
		batch_size=batch_size,
		assembly_id=ASSEMBLY_ID,
		eval_task_id=TASK_TEMPLATE_ID,
	)
	return phase33._load_model(checkpoint, cfg_args, device)


def _method_specs(args: argparse.Namespace) -> OrderedDict[str, MethodSpec]:
	total = int(args.total_candidates)
	baseline_pi = max(1, round(total * int(args.current_num_pi_trajs) / int(args.current_num_samples)))
	more_pi = min(total - 1, int(args.increased_policy_count))
	random_count = (total - baseline_pi) // 2
	gaussian_count = total - baseline_pi - random_count
	specs = OrderedDict([
		("current_mppi_ratio", MethodSpec(
			"current_mppi_ratio", "当前 MPPI 比例", baseline_pi, total - baseline_pi, 0, 0,
		)),
		("more_multitask_policy", MethodSpec(
			"more_multitask_policy", "增加 multitask policy proposals", more_pi, total - more_pi, 0, 0,
		)),
		("wide_multitask_policy_std", MethodSpec(
			"wide_multitask_policy_std", "增大 policy sampling std", baseline_pi, total - baseline_pi, 0, 0,
			policy_std_scale=float(args.wide_policy_std_scale),
		)),
		("multitask_plus_gaussian_random", MethodSpec(
			"multitask_plus_gaussian_random", "multitask + Gaussian/random", baseline_pi, gaussian_count, random_count, 0,
		)),
		("multitask_plus_direct_oracle", MethodSpec(
			"multitask_plus_direct_oracle", "multitask + direct proposals（oracle）", baseline_pi, 0, 0, total - baseline_pi,
			is_oracle=True,
		)),
	])
	for spec in specs.values():
		count = spec.multitask_policy_count + spec.gaussian_count + spec.random_count + spec.direct_policy_count
		if count != total:
			raise AssertionError(f"{spec.name} has {count} candidates, expected {total}.")
	return specs


def _atanh(value: torch.Tensor) -> torch.Tensor:
	value = value.clamp(-0.999999, 0.999999)
	return 0.5 * (torch.log1p(value) - torch.log1p(-value))


@torch.no_grad()
def _policy_stats(model, z: torch.Tensor, task: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
	z = model.task_context_adapt("pi", z, task)
	z = model.task_emb(z, task)
	mean, log_std = model._pi(z).chunk(2, dim=-1)
	log_std = td_math.log_std(log_std, model.log_std_min, model.log_std_dif)
	mask = model.action_mask(task, mean)
	while mask.ndim < mean.ndim:
		mask = mask.unsqueeze(-2)
	mask = mask.expand_as(mean)
	return mean * mask, log_std.exp() * mask


@torch.no_grad()
def _policy_candidates(
	model,
	obs: torch.Tensor,
	task_vec: torch.Tensor,
	noise: torch.Tensor,
	*,
	horizon: int,
	count: int,
	std_scale: float = 1.0,
) -> torch.Tensor:
	if count <= 0:
		return obs.new_empty(obs.shape[0], horizon, 0, 3)
	n = int(obs.shape[0])
	action_dim = int(noise.shape[-1])
	task_obs = condition_batch(task_vec, (n,), obs.device)
	z = model.encode(obs, task_obs)
	z = z.unsqueeze(1).expand(n, count, -1).reshape(n * count, -1)
	task = condition_batch(task_vec, (n * count,), obs.device)
	actions = []
	for step in range(horizon):
		mean, std = _policy_stats(model, z, task)
		eps = noise[:, step, :count].reshape(n * count, action_dim)
		action = torch.tanh(_atanh(mean) + eps * std * float(std_scale))
		actions.append(action.reshape(n, count, action_dim))
		z = model.next(z, action, task)
	return torch.stack(actions, dim=1)


def _noise(shape: tuple[int, ...], state: PlannerState, *, uniform: bool = False) -> torch.Tensor:
	if uniform:
		return torch.empty(*shape, device=state.prev_mean.device).uniform_(-1.0, 1.0, generator=state.generator)
	return torch.randn(*shape, device=state.prev_mean.device, generator=state.generator)


@torch.no_grad()
def _compose_candidates(
	*,
	spec: MethodSpec,
	multitask_model,
	direct_model,
	obs: torch.Tensor,
	task_vec: torch.Tensor,
	mean: torch.Tensor,
	std: torch.Tensor,
	state: PlannerState,
	horizon: int,
	total_candidates: int,
) -> tuple[torch.Tensor, torch.Tensor]:
	n, _, action_dim = mean.shape
	shape = (n, horizon, total_candidates, action_dim)
	multi_noise = _noise(shape, state)
	direct_noise = _noise(shape, state)
	gaussian_noise = _noise(shape, state)
	random_actions = _noise(shape, state, uniform=True)
	multitask_policy = _policy_candidates(
		multitask_model,
		obs,
		task_vec,
		multi_noise,
		horizon=horizon,
		count=spec.multitask_policy_count,
		std_scale=spec.policy_std_scale,
	)
	direct_policy = _policy_candidates(
		direct_model,
		obs,
		task_vec,
		direct_noise,
		horizon=horizon,
		count=spec.direct_policy_count,
	)
	gaussian = (
		mean.unsqueeze(2) + std.unsqueeze(2) * gaussian_noise[:, :, :spec.gaussian_count]
	).clamp(-1.0, 1.0)
	random = random_actions[:, :, :spec.random_count]
	candidates = torch.cat([multitask_policy, gaussian, random, direct_policy], dim=2)
	if int(candidates.shape[2]) != total_candidates:
		raise AssertionError(f"{spec.name} candidate count mismatch: {candidates.shape}")
	return candidates, multitask_policy


def _initial_distribution(
	*,
	multitask_model,
	obs: torch.Tensor,
	task_vec: torch.Tensor,
	t0: torch.Tensor,
	state: PlannerState,
	spec: MethodSpec,
	horizon: int,
	total_candidates: int,
	min_std: float,
	max_std: float,
	constrained: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
	n = int(obs.shape[0])
	action_dim = int(state.prev_mean.shape[-1])
	prior_count = max(1, spec.multitask_policy_count)
	prior_noise = _noise((n, horizon, total_candidates, action_dim), state)
	prior = _policy_candidates(
		multitask_model,
		obs,
		task_vec,
		prior_noise,
		horizon=horizon,
		count=prior_count,
		std_scale=1.0,
	)
	shifted = torch.cat([state.prev_mean[:, 1:], torch.zeros_like(state.prev_mean[:, :1])], dim=1)
	base_mean = torch.where(t0.view(n, 1, 1), torch.zeros_like(shifted), shifted)
	base_std = torch.full_like(base_mean, float(max_std))
	if not constrained:
		return base_mean, base_std
	pi_mean = prior.mean(dim=2)
	pi_std = prior.std(dim=2, unbiased=False).clamp(float(min_std), float(max_std))
	return pi_mean, pi_std


@torch.no_grad()
def _plan_step(
	*,
	spec: MethodSpec,
	multitask_model,
	multitask_cfg,
	direct_model,
	obs: torch.Tensor,
	task_vec: torch.Tensor,
	t0: torch.Tensor,
	state: PlannerState,
	iterations: int,
	total_candidates: int,
	num_elites: int,
) -> dict[str, torch.Tensor]:
	if obs.ndim != 2:
		raise ValueError(f"Expected flat obs [N,O], got {tuple(obs.shape)}")
	if state.prev_mean.shape[0] != obs.shape[0]:
		raise ValueError("Planner state batch does not match observation batch.")
	if obs.is_cuda:
		torch.cuda.synchronize(obs.device)
	start = perf_counter()
	horizon = int(multitask_cfg.horizon)
	mean, std = _initial_distribution(
		multitask_model=multitask_model,
		obs=obs,
		task_vec=task_vec,
		t0=t0,
		state=state,
		spec=spec,
		horizon=horizon,
		total_candidates=total_candidates,
		min_std=float(multitask_cfg.min_std),
		max_std=float(multitask_cfg.max_std),
		constrained=bool(multitask_cfg.constrained_planning),
	)
	for _ in range(iterations):
		candidates, _ = _compose_candidates(
			spec=spec,
			multitask_model=multitask_model,
			direct_model=direct_model,
			obs=obs,
			task_vec=task_vec,
			mean=mean,
			std=std,
			state=state,
			horizon=horizon,
			total_candidates=total_candidates,
		)
		scores = phase33._score_candidates(multitask_model, multitask_cfg, obs, task_vec, candidates)
		value = scores["total"].nan_to_num(0.0)
		elite_idx = torch.topk(value, k=num_elites, dim=1).indices
		elite_value = value.gather(1, elite_idx)
		elite_actions = candidates.gather(
			2,
			elite_idx[:, None, :, None].expand(-1, horizon, -1, candidates.shape[-1]),
		)
		weights = torch.exp(float(multitask_cfg.temperature) * (elite_value - elite_value.max(1, keepdim=True).values))
		weights = weights / (weights.sum(1, keepdim=True) + 1.0e-9)
		weights = weights.unsqueeze(1).unsqueeze(-1)
		mean = (weights * elite_actions).sum(dim=2)
		std = ((weights * (elite_actions - mean.unsqueeze(2)).pow(2)).sum(dim=2) /
			(weights.sum(dim=2) + 1.0e-9)).sqrt().clamp(float(multitask_cfg.min_std), float(multitask_cfg.max_std))
	selected_idx = value.argmax(dim=1)
	selected_action = candidates[:, 0].gather(
		1, selected_idx[:, None, None].expand(-1, 1, candidates.shape[-1]),
	).squeeze(1)
	if obs.is_cuda:
		torch.cuda.synchronize(obs.device)
	state.inference_ms_total += 1000.0 * (perf_counter() - start)
	state.inference_calls += 1
	state.prev_mean = mean.detach()
	topk = torch.topk(value, k=min(5, total_candidates), dim=1).values
	return {
		"action": selected_action.clamp(-1.0, 1.0),
		"selected_score": value.gather(1, selected_idx[:, None]).squeeze(1),
		"top5_score": topk.mean(dim=1),
		"candidates": candidates,
		"scores": value,
		"selected_idx": selected_idx,
	}


def _make_state(device: torch.device, n: int, horizon: int, action_dim: int, seed: int) -> PlannerState:
	generator = torch.Generator(device=device)
	generator.manual_seed(int(seed))
	return PlannerState(
		prev_mean=torch.zeros(n, horizon, action_dim, device=device),
		generator=generator,
	)


def _group_indices(labels: dict[str, list[str]], group: str) -> torch.Tensor:
	if group == "contact":
		mask = [phase == "contact" for phase in labels["phase"]]
	elif group == "jam":
		mask = [outcome == "jam" for outcome in labels["outcome"]]
	else:
		raise ValueError(group)
	return torch.nonzero(torch.tensor(mask, dtype=torch.bool), as_tuple=False).reshape(-1)


def _offline_ablation(args, specs, direct_model, direct_cfg, multitask_model, multitask_cfg, task_vec, device):
	banks, _ = phase33._load_rollout_banks(args)
	states = phase33._sample_states(banks, args)
	n = int(states["obs"].shape[0])
	if n != int(args.expected_state_count):
		raise RuntimeError(f"Expected {args.expected_state_count} Phase 3.3 states, got {n}.")
	obs = states["obs"].to(device)
	t0 = torch.ones(n, dtype=torch.bool, device=device)
	results = OrderedDict()
	baseline_actions = None
	labels = {key: states[key] for key in ("region", "phase", "outcome")}
	for method_index, (name, spec) in enumerate(specs.items()):
		state = _make_state(device, n, int(multitask_cfg.horizon), int(multitask_cfg.action_dim), int(args.seed) + 1000 * method_index)
		output = _plan_step(
			spec=spec,
			multitask_model=multitask_model,
			multitask_cfg=multitask_cfg,
			direct_model=direct_model,
			obs=obs,
			task_vec=task_vec,
			t0=t0,
			state=state,
			iterations=int(args.iterations),
			total_candidates=int(args.total_candidates),
			num_elites=int(args.num_elites),
		)
		direct_scores = phase33._score_candidates(
			direct_model, direct_cfg, obs, task_vec, output["candidates"],
		)["total"]
		direct_reference_state = _make_state(device, n, int(multitask_cfg.horizon), int(multitask_cfg.action_dim), int(args.seed) + 9000)
		direct_spec = MethodSpec("direct_reference", "direct reference", int(args.total_candidates), 0, 0, 0)
		direct_ref = _plan_step(
			spec=direct_spec,
			multitask_model=direct_model,
			multitask_cfg=direct_cfg,
			direct_model=direct_model,
			obs=obs,
			task_vec=task_vec,
			t0=t0,
			state=direct_reference_state,
			iterations=1,
			total_candidates=int(args.total_candidates),
			num_elites=int(args.num_elites),
		)
		direct_ref_scores = phase33._score_candidates(
			direct_model, direct_cfg, obs, task_vec, direct_ref["candidates"],
		)["total"]
		selected = output["selected_idx"]
		direct_top = direct_scores.max(dim=1).values
		direct_selected = direct_scores.gather(1, selected[:, None]).squeeze(1)
		direct_reference_top = direct_ref_scores.max(dim=1).values
		proposal_regret = (direct_reference_top - direct_selected).clamp_min(0.0)
		coverage_regret = (direct_reference_top - direct_top).clamp_min(0.0)
		scoring_regret = (direct_top - direct_selected).clamp_min(0.0)
		if baseline_actions is None:
			baseline_actions = output["action"].detach()
		selected_l2 = torch.linalg.vector_norm(output["action"] - baseline_actions, dim=-1)
		metrics = {
			"proposal_regret": proposal_regret.detach().cpu(),
			"candidate_coverage_regret": coverage_regret.detach().cpu(),
			"scorer_selection_regret": scoring_regret.detach().cpu(),
			"selected_action_l2_vs_current": selected_l2.detach().cpu(),
			"selected_score": output["selected_score"].detach().cpu(),
			"top5_score": output["top5_score"].detach().cpu(),
		}
		result = {
			"candidate_composition": {
				"total": int(args.total_candidates),
				"multitask_policy": spec.multitask_policy_count,
				"gaussian": spec.gaussian_count,
				"random": spec.random_count,
				"direct_policy_oracle": spec.direct_policy_count,
				"policy_std_scale": spec.policy_std_scale,
			},
			"all_states": {key: summarize_tensor(value) for key, value in metrics.items()},
			"inference_ms_per_state": state.inference_ms_total / max(1, n),
			"groups": OrderedDict(),
		}
		for group in GROUPS:
			idx = _group_indices(labels, group)
			if idx.numel() == 0:
				continue
			result["groups"][group] = {
				"count": int(idx.numel()),
				**{key: summarize_tensor(value[idx]) for key, value in metrics.items()},
			}
		results[name] = result
	return {
		"sample_count": n,
		"sample_group_counts": states["group_counts"],
		"methods": results,
	}


def _configure_runtime_cfg(args, multitask_checkpoint: Path, total_envs: int, profile: str):
	# IsaacLab redirects cwd to a per-run runtime directory. The train YAML still
	# contains repo-relative task-template paths, so restore the repository cwd
	# before parse_cfg expands them for each independent profile.
	os.chdir(REPO_ROOT)
	cfg_args = SimpleNamespace(
		config=str(resolve(args.config)),
		gpu_id=args.gpu_id,
		batch_size=total_envs,
		assembly_id=ASSEMBLY_ID,
		eval_task_id=TASK_TEMPLATE_ID,
	)
	cfg, _ = tvsr._load_config(cfg_args, multitask_checkpoint)
	cfg.num_envs = int(total_envs)
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
	cfg.srsa_enable_flange_force_sensor = True
	cfg.isaaclab_canonical_append_force = True
	cfg.isaaclab_canonical_append_task_params = False
	cfg.isaaclab_canonical_use_visual_noise = False
	cfg.srsa_vision_noise_xy_std = 0.0
	cfg.srsa_vision_noise_xy_jitter_std = 0.0
	cfg.srsa_vision_noise_z_std = 0.0
	cfg.srsa_vision_noise_z_jitter_std = 0.0
	cfg.srsa_axial_init_error_z_range = "0.001,0.002"
	cfg.srsa_axial_init_error_yaw_range = "-0.0872665,0.0872665"
	cfg.srsa_axial_visual_noise_xy_range = "0.0,0.0"
	cfg.srsa_axial_visual_noise_z_range = "0.0,0.0"
	profile_xy = {
		"easy": "0.001,0.004",
		"default": "0.001,0.009",
		"hard": "0.009,0.015",
	}
	if profile not in profile_xy:
		raise ValueError(f"Unknown profile: {profile}")
	cfg.srsa_axial_init_error_xy_range = profile_xy[profile]
	cfg = collector._task_cfg(cfg, ASSEMBLY_ID)
	cfg.num_envs = int(total_envs)
	cfg.device_id = int(args.gpu_id)
	cfg.exp_name = "srsa_phase3_4_proposal_rescue_eval"
	cfg.run_id = f"phase3_4_{profile}"
	return cfg


def _terminal_metric(final_info: dict, key: str, env_index: int, default: float = 0.0) -> float:
	value = final_info.get(key, None)
	if value is None:
		return float(default)
	return float(torch.nan_to_num(value[env_index], nan=0.0).detach().item())


def _method_metrics(records: list[dict[str, float]]) -> dict[str, float]:
	if not records:
		return {key: math.nan for key in (
			"relaxed_success", "strict_success", "process_success", "reward", "jamming_rate",
			"lateral_error_mm", "keypoint_error_mm", "max_force", "force_excursion", "episode_length",
		)}
	keys = records[0].keys()
	return {key: float(sum(row[key] for row in records) / len(records)) for key in keys}


def _closed_loop_profile(
	args: argparse.Namespace,
	profile: str,
	specs: OrderedDict[str, MethodSpec],
	direct_model,
	direct_cfg,
	multitask_model,
	multitask_cfg,
	multitask_checkpoint: Path,
	device: torch.device,
) -> dict[str, Any]:
	per_method = int(args.episodes_per_method)
	total_envs = per_method * len(specs)
	set_seed(int(args.seed))
	cfg = _configure_runtime_cfg(args, multitask_checkpoint, total_envs, profile)
	env = make_env(cfg)
	try:
		expected_obs_dim = int(multitask_cfg.obs_shape["state"][0])
		obs, _ = env.reset()
		obs = collector._adapt_obs_to_checkpoint(obs, expected_obs_dim)
		task_vecs = getattr(env.unwrapped, "current_task_vec", None)
		if not torch.is_tensor(task_vecs) or task_vecs.shape != (total_envs, 6):
			raise RuntimeError(f"00186 runtime task_vec has unexpected shape: {getattr(task_vecs, 'shape', None)}")
		task_vec = task_vecs[0].detach().float().to(device)
		if not torch.allclose(task_vecs, task_vecs[:1].expand_as(task_vecs), atol=1.0e-6):
			raise RuntimeError("The Phase 3.4 rollout requires a single shared 00186 task vector across envs.")
		states = OrderedDict()
		for method_index, name in enumerate(specs):
			states[name] = _make_state(
				device,
				per_method,
				int(multitask_cfg.horizon),
				int(multitask_cfg.action_dim),
				int(args.seed) + 10000 * (REGIONS.index(profile) + 1) + 100 * method_index,
			)
		records = {name: [] for name in specs}
		episode_return = torch.zeros(total_envs, device=device)
		episode_length = torch.zeros(total_envs, dtype=torch.long, device=device)
		force_sum = torch.zeros(total_envs, device=device)
		force_count = torch.zeros(total_envs, device=device)
		force_max = torch.zeros(total_envs, device=device)
		completed = {name: 0 for name in specs}
		guard_steps = int(max(1, cfg.episode_length))
		for env_step in range(guard_steps):
			t0 = episode_length == 0
			actions = torch.empty(total_envs, int(multitask_cfg.action_dim), device=device)
			for method_index, (name, spec) in enumerate(specs.items()):
				start = method_index * per_method
				stop = start + per_method
				result = _plan_step(
					spec=spec,
					multitask_model=multitask_model,
					multitask_cfg=multitask_cfg,
					direct_model=direct_model,
					obs=obs[start:stop],
					task_vec=task_vec,
					t0=t0[start:stop],
					state=states[name],
					iterations=int(args.iterations),
					total_candidates=int(args.total_candidates),
					num_elites=int(args.num_elites),
				)
				actions[start:stop] = result["action"]
			prev_obs = obs
			raw_obs, reward, terminated, truncated, info = env.step(actions)
			obs = collector._adapt_obs_to_checkpoint(raw_obs, expected_obs_dim)
			# Some SRSA/IsaacLab builds do not surface the timeout in the returned
			# terminated/truncated tensors, although final_info is emitted at the
			# configured episode boundary. Treat that boundary as eval completion;
			# this only controls report collection and never changes env state.
			time_limit = (episode_length + 1) >= int(cfg.episode_length)
			done = terminated | truncated | time_limit
			episode_return += reward
			if prev_obs.shape[-1] >= 17:
				force = torch.linalg.vector_norm(prev_obs[:, 14:17], dim=-1)
				baseline_mask = episode_length < 5
				force_sum = force_sum + torch.where(baseline_mask, force, torch.zeros_like(force))
				force_count = force_count + baseline_mask.to(force_count.dtype)
				force_max = torch.maximum(force_max, force)
			final_info = info.get("final_info", {}) if isinstance(info, dict) else {}
			success_tensor = final_info.get("success", None)
			for method_index, name in enumerate(specs):
				start = method_index * per_method
				stop = start + per_method
				for env_index in range(start, stop):
					if not bool(done[env_index].item()) or completed[name] >= per_method:
						continue
					fallback_success = float(success_tensor[env_index].detach().item()) if success_tensor is not None else 0.0
					relaxed = _terminal_metric(final_info, "relaxed_success_stable", env_index, fallback_success)
					strict = _terminal_metric(final_info, "strict_success_stable", env_index, 0.0)
					process = _terminal_metric(final_info, "process_success_terminal", env_index, 0.0)
					lateral = _terminal_metric(final_info, "lateral_error", env_index, 0.0)
					keypoint = _terminal_metric(final_info, "keypoint_error", env_index, 0.0)
					base_force = force_sum[env_index] / force_count[env_index].clamp_min(1.0)
					excursion = (force_max[env_index] - base_force).clamp_min(0.0)
					jam = float(
						relaxed <= 0.5 and (
							lateral >= float(args.jam_lateral_threshold)
							or keypoint >= float(args.jam_keypoint_threshold)
							or float(excursion.item()) >= float(args.jam_force_excursion_threshold)
						)
					)
					records[name].append({
						"relaxed_success": relaxed,
						"strict_success": strict,
						"process_success": process,
						"reward": float(episode_return[env_index].detach().item()),
						"jamming_rate": jam,
						"lateral_error_mm": 1000.0 * lateral,
						"keypoint_error_mm": 1000.0 * keypoint,
						"max_force": float(force_max[env_index].detach().item()),
						"force_excursion": float(excursion.detach().item()),
						"episode_length": float(episode_length[env_index].item() + 1),
					})
					completed[name] += 1
			episode_return = torch.where(done, torch.zeros_like(episode_return), episode_return)
			episode_length = torch.where(done, torch.zeros_like(episode_length), episode_length + 1)
			force_sum = torch.where(done, torch.zeros_like(force_sum), force_sum)
			force_count = torch.where(done, torch.zeros_like(force_count), force_count)
			force_max = torch.where(done, torch.zeros_like(force_max), force_max)
			if (env_step + 1) % 10 == 0:
				print(
					f"[phase3.4] profile={profile} env_step={env_step + 1}/{guard_steps} "
					f"completed={completed}",
					flush=True,
				)
			if all(value >= per_method for value in completed.values()):
				break
		else:
			raise RuntimeError(f"{profile} rollout did not finish before guard_steps={guard_steps}: {completed}")
		return {
			"profile": profile,
			"num_envs_per_method": per_method,
			"task_vec_6": tensor_to_list(task_vec),
			"methods": {
				name: {
					"metrics": _method_metrics(records[name]),
					"episodes": len(records[name]),
					"inference_ms_per_step": states[name].inference_ms_total / max(1, states[name].inference_calls * per_method),
				}
				for name in specs
			},
		}
	finally:
		env.close()


def _mean_closed_loop_metric(closed_loop: dict[str, Any], method: str, key: str) -> float:
	values = [
		float(closed_loop[region]["methods"][method]["metrics"][key])
		for region in ("default", "hard")
	]
	return sum(values) / len(values)


def _closed_loop_improved(closed_loop: dict[str, Any], method: str, baseline: str) -> bool:
	base_relaxed = _mean_closed_loop_metric(closed_loop, baseline, "relaxed_success")
	method_relaxed = _mean_closed_loop_metric(closed_loop, method, "relaxed_success")
	base_jam = _mean_closed_loop_metric(closed_loop, baseline, "jamming_rate")
	method_jam = _mean_closed_loop_metric(closed_loop, method, "jamming_rate")
	base_lateral = _mean_closed_loop_metric(closed_loop, baseline, "lateral_error_mm")
	method_lateral = _mean_closed_loop_metric(closed_loop, method, "lateral_error_mm")
	base_keypoint = _mean_closed_loop_metric(closed_loop, baseline, "keypoint_error_mm")
	method_keypoint = _mean_closed_loop_metric(closed_loop, method, "keypoint_error_mm")
	return bool(
		method_relaxed >= base_relaxed + 0.10
		or method_jam <= base_jam - 0.10
		or (method_lateral <= 0.90 * base_lateral and method_keypoint <= 0.90 * base_keypoint)
	)


def _verdict(offline: dict[str, Any], closed_loop: dict[str, Any]) -> dict[str, Any]:
	baseline = "current_mppi_ratio"
	base_regret = float(offline["methods"][baseline]["groups"]["contact"]["proposal_regret"]["mean"])
	if not math.isfinite(base_regret) or base_regret <= 0.0:
		return {"decision": "UNRESOLVED", "reason": "baseline contact proposal regret is invalid"}
	diverse = (
		"more_multitask_policy",
		"wide_multitask_policy_std",
		"multitask_plus_gaussian_random",
	)
	diverse_hits = []
	for method in diverse:
		regret = float(offline["methods"][method]["groups"]["contact"]["proposal_regret"]["mean"])
		reduction = 1.0 - regret / base_regret
		improved = _closed_loop_improved(closed_loop, method, baseline)
		diverse_hits.append({"method": method, "proposal_regret_reduction": reduction, "closed_loop_improved": improved})
	if any(item["proposal_regret_reduction"] >= 0.50 and item["closed_loop_improved"] for item in diverse_hits):
		return {
			"decision": "MPPI_SAMPLING_DIVERSITY_FIX",
			"reason": "At least one non-oracle diverse candidate composition halves contact regret and improves default/hard closed-loop quality.",
			"diverse_results": diverse_hits,
		}
	oracle = "multitask_plus_direct_oracle"
	oracle_regret = float(offline["methods"][oracle]["groups"]["contact"]["proposal_regret"]["mean"])
	oracle_reduction = 1.0 - oracle_regret / base_regret
	oracle_improved = _closed_loop_improved(closed_loop, oracle, baseline)
	if oracle_reduction >= 0.50 and oracle_improved:
		return {
			"decision": "POLICY_PRIOR_DISTILLATION_REQUIRED",
			"reason": "Only the direct-proposal oracle clears the 50% contact-regret and closed-loop gates.",
			"diverse_results": diverse_hits,
			"oracle_proposal_regret_reduction": oracle_reduction,
			"oracle_closed_loop_improved": oracle_improved,
		}
	return {
		"decision": "UNRESOLVED",
		"reason": "Neither diverse sampling nor the direct-proposal oracle clears both the proposal-regret and closed-loop gates.",
		"diverse_results": diverse_hits,
		"oracle_proposal_regret_reduction": oracle_reduction,
		"oracle_closed_loop_improved": oracle_improved,
	}


def _offline_contact_regret_reductions(offline: dict[str, Any]) -> dict[str, float]:
	baseline = "current_mppi_ratio"
	base = float(offline["methods"][baseline]["groups"]["contact"]["proposal_regret"]["mean"])
	if not math.isfinite(base) or base <= 0.0:
		return {}
	return {
		name: 1.0 - float(item["groups"]["contact"]["proposal_regret"]["mean"]) / base
		for name, item in offline["methods"].items()
		if "contact" in item["groups"]
	}


def _markdown(report: dict[str, Any]) -> str:
	decision = report["verdict"]
	lines = [
		"# SRSA Phase 3.4 Proposal Rescue Ablation",
		"",
		"本报告只做固定 candidate budget 的 proposal 组成评估；未训练、未修改 checkpoint 参数，也未修改 task_context、reward、Q、dynamics、MPPI 主逻辑或 sampler。",
		"",
		f"Status: `{report['status']}`",
		f"Final decision: `{decision['decision']}`",
		"",
		"## 设计",
		"",
		f"- 固定总预算：每次控制决策严格 `{report['settings']['total_candidates']}` 条 candidate，horizon=`{report['settings']['horizon']}`，ranking iterations=`{report['settings']['iterations']}`。",
		f"- 当前 MPPI 原比例：`{report['settings']['current_num_pi_trajs']}/{report['settings']['current_num_samples']}`，缩放后 baseline 为 `{report['settings']['baseline_policy_count']}` 条 policy proposal。",
		"- direct proposals 仅用于诊断 oracle，上线 controller 不可依赖它们。",
		"",
		"## Offline 408-State",
		"",
		"| Method | Composition (pi/gauss/random/direct) | Proposal regret | Contact regret | Jam regret | Action L2 vs current | Top-5 score | ms/state |",
		"| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
	]
	for name, item in report["offline"]["methods"].items():
		comp = item["candidate_composition"]
		all_metrics = item["all_states"]
		contact = item["groups"].get("contact", {}).get("proposal_regret", {})
		jam = item["groups"].get("jam", {}).get("proposal_regret", {})
		lines.append(
			f"| `{name}` | {comp['multitask_policy']}/{comp['gaussian']}/{comp['random']}/{comp['direct_policy_oracle']} | "
			f"{all_metrics['proposal_regret']['mean']:.3f} | {float(contact.get('mean', math.nan)):.3f} | "
			f"{float(jam.get('mean', math.nan)):.3f} | {all_metrics['selected_action_l2_vs_current']['mean']:.3f} | "
			f"{all_metrics['top5_score']['mean']:.3f} | {item['inference_ms_per_state']:.3f} |"
		)
	lines.extend([
		"",
		"- Contact proposal-regret reduction vs current MPPI: " + ", ".join(
			f"`{name}`={100.0 * reduction:.1f}%"
			for name, reduction in report["offline_contact_regret_reductions"].items()
		),
	])
	lines.extend([
		"",
		"## Closed Loop",
		"",
	])
	if report["closed_loop"] is None:
		lines.append(
			f"Closed-loop status: `{report['closed_loop_status']}`. "
			f"{report['closed_loop_reason']}"
		)
	else:
		lines.extend([
			"| Profile | Method | Relaxed | Strict | Process | Jam | Lateral mm | Keypoint mm | Reward | ms/step |",
			"| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
		])
		for profile in REGIONS:
			for name, item in report["closed_loop"][profile]["methods"].items():
				metrics = item["metrics"]
				lines.append(
					f"| `{profile}` | `{name}` | {metrics['relaxed_success']:.3f} | {metrics['strict_success']:.3f} | "
					f"{metrics['process_success']:.3f} | {metrics['jamming_rate']:.3f} | {metrics['lateral_error_mm']:.3f} | "
					f"{metrics['keypoint_error_mm']:.3f} | {metrics['reward']:.2f} | {item['inference_ms_per_step']:.3f} |"
				)
	lines.extend([
		"",
		"## 判定",
		"",
		f"- `{decision['decision']}`：{decision['reason']}",
		"- closed-loop 的五个方法处在同一 profile/seed 的并行环境批中；不同 env slot 共享初始状态分布，但不是逐 slot 克隆状态。离线 408-state 结果提供严格的相同状态对比。",
		"- jam 是 failure 加 lateral/keypoint/force-excursion 的只读代理；环境 eval API 不直接导出 episode jam 标签。",
		"",
		"## 输入",
		"",
		f"- direct checkpoint: `{report['inputs']['direct_checkpoint']}`",
		f"- multitask checkpoint: `{report['inputs']['multitask_checkpoint']}`",
		f"- Phase 3.3 rollout root: `{report['inputs']['phase33_rollout_root']}`",
	])
	return "\n".join(lines) + "\n"


def build_report(args: argparse.Namespace) -> dict[str, Any]:
	direct_checkpoint, multitask_checkpoint = _checkpoint_paths(args)
	device = torch.device(f"cuda:{int(args.gpu_id)}" if torch.cuda.is_available() and not args.cpu else "cpu")
	if device.type != "cuda" and not args.offline_only:
		raise RuntimeError("Closed-loop Phase 3.4 evaluation requires CUDA/Isaac. Use --offline-only for model-only diagnosis.")
	if device.type == "cuda":
		torch.cuda.set_device(device)
	multitask_model, multitask_cfg, multitask_compat = _load_model(multitask_checkpoint, args, device, int(args.offline_batch_size))
	direct_model, direct_cfg, direct_compat = _load_model(direct_checkpoint, args, device, int(args.offline_batch_size))
	if int(multitask_compat["obs_dim"]) != 17 or int(multitask_compat["action_dim"]) != 3:
		raise RuntimeError(f"Unexpected multitask checkpoint contract: {multitask_compat}")
	if int(direct_compat["obs_dim"]) != 17 or int(direct_compat["action_dim"]) != 3:
		raise RuntimeError(f"Unexpected direct checkpoint contract: {direct_compat}")
	task_vec = tvsr._unique_task_vec_from_replay(resolve(args.task_replay))[0].float().to(device)
	specs = _method_specs(args)
	offline = _offline_ablation(
		args, specs, direct_model, direct_cfg, multitask_model, multitask_cfg, task_vec, device,
	)
	offline_contact_reductions = _offline_contact_regret_reductions(offline)
	closed_loop = OrderedDict()
	if not args.offline_only:
		for profile in REGIONS:
			print(f"[phase3.4] closed-loop profile={profile} start")
			closed_loop[profile] = _closed_loop_profile(
				args,
				profile,
				specs,
				direct_model,
				direct_cfg,
				multitask_model,
				multitask_cfg,
				multitask_checkpoint,
				device,
			)
			print(f"[phase3.4] closed-loop profile={profile} complete")
	else:
		closed_loop = None
	if closed_loop is None:
		closed_loop_status = "UNKNOWN_RUNTIME_STALL" if args.closed_loop_skip_reason else "SKIPPED_OFFLINE_ONLY"
		closed_loop_reason = args.closed_loop_skip_reason or "Closed-loop ablation was skipped with --offline-only."
		best_non_oracle = max(
			(
				reduction
				for name, reduction in offline_contact_reductions.items()
				if name not in {"current_mppi_ratio", "multitask_plus_direct_oracle"}
			),
			default=math.nan,
		)
		oracle_reduction = offline_contact_reductions.get("multitask_plus_direct_oracle", math.nan)
		verdict = {
			"decision": "UNRESOLVED",
			"reason": (
				f"No non-oracle composition reduced contact proposal regret (best={100.0 * best_non_oracle:.1f}%); "
				f"the direct-proposal oracle reached only {100.0 * oracle_reduction:.1f}% (<50%), and {closed_loop_reason}"
			),
		}
		status = "WARNING"
	else:
		closed_loop_status = "COMPLETE"
		closed_loop_reason = None
		verdict = _verdict(offline, closed_loop)
		status = "PASS" if verdict["decision"] != "UNRESOLVED" else "WARNING"
	return {
		"status": status,
		"verdict": verdict,
		"inputs": {
			"direct_checkpoint": str(direct_checkpoint),
			"multitask_checkpoint": str(multitask_checkpoint),
			"task_replay": str(resolve(args.task_replay)),
			"phase33_rollout_root": str(resolve(args.rollout_root)),
		},
		"settings": {
			"total_candidates": int(args.total_candidates),
			"iterations": int(args.iterations),
			"num_elites": int(args.num_elites),
			"horizon": int(multitask_cfg.horizon),
			"current_num_pi_trajs": int(args.current_num_pi_trajs),
			"current_num_samples": int(args.current_num_samples),
			"baseline_policy_count": specs["current_mppi_ratio"].multitask_policy_count,
			"closed_loop_episodes_per_method": int(args.episodes_per_method),
			"closed_loop_selection": "deterministic top-1 of final fixed-budget CEM candidate bank",
		},
		"task_vec_00186": tensor_to_list(task_vec),
		"checkpoint_compatibility": {"direct": direct_compat, "multitask": multitask_compat},
		"offline": offline,
		"offline_contact_regret_reductions": offline_contact_reductions,
		"closed_loop": closed_loop,
		"closed_loop_status": closed_loop_status,
		"closed_loop_reason": closed_loop_reason,
		"limitations": [
			"Direct proposals are an oracle diagnostic arm, not a deployment candidate source.",
			"Closed-loop method groups share sampler settings and seed but use distinct parallel Isaac env slots; exact state matching is provided by the offline 408-state analysis.",
			"The fixed-budget evaluator uses deterministic final top-1 selection so candidate composition is reproducible; production MPPI uses Gumbel elite sampling.",
		],
	}


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--phase32-diagnosis", default=DEFAULT_PHASE32_DIAGNOSIS)
	parser.add_argument("--direct-checkpoint", default=None)
	parser.add_argument("--multitask-checkpoint", default=None)
	parser.add_argument("--task-replay", default=(
		"logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_taskctx_repair_01125_00256_00186_rescue/"
		"20260713_phase3_2_rescue_00186_launcher/replay/00186.pt"
	))
	parser.add_argument("--rollout-root", default=DEFAULT_PHASE33_ROLLOUT_ROOT)
	parser.add_argument("--config", default="configs/train/srsa_01125_imitation_relaxed.yaml")
	parser.add_argument("--task-template-fp", default="data/srsa_axial_task_templates.json")
	parser.add_argument("--mesh-geometry-fp", default="data/srsa_mesh_geometry_params.csv")
	parser.add_argument("--isaaclab-dir", default="/home/gpuserver/IsaacLab")
	parser.add_argument("--srsa-dir", default="/home/gpuserver/hx/github/srsa")
	parser.add_argument("--total-candidates", type=int, default=64)
	parser.add_argument(
		"--iterations",
		type=int,
		default=1,
		help="Ranking passes per control decision. Default 1 preserves the strict fixed total-candidate budget.",
	)
	parser.add_argument("--num-elites", type=int, default=8)
	parser.add_argument("--current-num-samples", type=int, default=512)
	parser.add_argument("--current-num-pi-trajs", type=int, default=24)
	parser.add_argument("--increased-policy-count", type=int, default=24)
	parser.add_argument("--wide-policy-std-scale", type=float, default=3.0)
	parser.add_argument("--offline-batch-size", type=int, default=12)
	parser.add_argument(
		"--max-states-per-group",
		type=int,
		default=8,
		help="Match Phase 3.3's stratified rollout-state cap for each source/phase/outcome group.",
	)
	parser.add_argument("--expected-state-count", type=int, default=408)
	parser.add_argument("--episodes-per-method", type=int, default=16)
	parser.add_argument("--jam-lateral-threshold", type=float, default=0.008)
	parser.add_argument("--jam-keypoint-threshold", type=float, default=0.012)
	parser.add_argument("--jam-force-excursion-threshold", type=float, default=2.0)
	parser.add_argument("--seed", type=int, default=1)
	parser.add_argument("--gpu-id", type=int, default=0)
	parser.add_argument("--cpu", action="store_true")
	parser.add_argument("--offline-only", action="store_true")
	parser.add_argument(
		"--closed-loop-skip-reason",
		default=None,
		help="Persist an observed runtime limitation when producing an offline-only report.",
	)
	parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
	parser.add_argument("--output-md", default=DEFAULT_OUTPUT_MD)
	parser.add_argument("--dry-run", action="store_true")
	args = parser.parse_args()
	if args.total_candidates < 8:
		raise ValueError("--total-candidates must be at least 8.")
	if args.num_elites <= 0 or args.num_elites > args.total_candidates:
		raise ValueError("--num-elites must be in [1, total_candidates].")
	if args.dry_run:
		direct, multitask = _checkpoint_paths(args)
		specs = _method_specs(args)
		print(f"PASS dry-run: direct={direct} multitask={multitask}")
		for spec in specs.values():
			print(
				f"{spec.name}: pi={spec.multitask_policy_count} gaussian={spec.gaussian_count} "
				f"random={spec.random_count} direct_oracle={spec.direct_policy_count}"
			)
		return 0
	report = build_report(args)
	write_json(report, args.output_json)
	write_text(_markdown(report), args.output_md)
	print(report["status"])
	print(report["verdict"]["decision"])
	return 0 if report["status"] != "FAIL" else 1


if __name__ == "__main__":
	raise SystemExit(main())
