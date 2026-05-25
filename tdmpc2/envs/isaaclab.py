import sys
import os
import importlib
import math
import types
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch


_APP_LAUNCHER = None
_SIMULATION_APP = None
_ISAACLAB_WORKDIR = None


def _canonicalize_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
	quat = quat / torch.linalg.norm(quat, dim=-1, keepdim=True).clamp_min(1.0e-8)
	sign = torch.where(quat[:, :1] < 0.0, -1.0, 1.0)
	return quat * sign


def _build_canonical_obs(env, cfg=None) -> torch.Tensor:
	import isaacsim.core.utils.torch as torch_utils

	frame_quat = env.fixed_quat
	frame_pos = env.fixed_pos_obs_frame
	if cfg is not None and cfg.get('isaaclab_canonical_use_visual_noise', False):
		if hasattr(env, '_vision_noise_world'):
			frame_pos = frame_pos + env._vision_noise_world
	frame_quat_inv, frame_pos_inv = torch_utils.tf_inverse(frame_quat, frame_pos)

	tcp_quat_socket, tcp_pos_socket = torch_utils.tf_combine(
		frame_quat_inv,
		frame_pos_inv,
		env.fingertip_midpoint_quat,
		env.fingertip_midpoint_pos,
	)
	tcp_quat_socket = _canonicalize_quat_wxyz(tcp_quat_socket)
	tcp_linvel_socket = torch_utils.quat_rotate_inverse(frame_quat, env.ee_linvel_fd)
	tcp_angvel_socket = torch_utils.quat_rotate_inverse(frame_quat, env.ee_angvel_fd)
	gripper_width = env.joint_pos[:, 7:9].sum(dim=-1, keepdim=True)
	parts = [
		tcp_pos_socket,
		tcp_quat_socket,
		tcp_linvel_socket,
		tcp_angvel_socket,
		gripper_width,
	]
	if cfg is not None and cfg.get('isaaclab_canonical_append_force', False):
		force_obs = getattr(env, 'flange_force_obs', None)
		if cfg.get('isaaclab_canonical_zero_force', False):
			force_dim = (
				int(force_obs.shape[-1])
				if torch.is_tensor(force_obs) and force_obs.ndim > 0
				else int(cfg.get('isaaclab_canonical_force_dim', 3))
			)
			force_obs = torch.zeros(
				(*tcp_pos_socket.shape[:-1], force_dim),
				dtype=tcp_pos_socket.dtype,
				device=tcp_pos_socket.device,
			)
		if force_obs is not None:
			parts.append(force_obs)
	task_params = _get_srsa_task_param_obs_tensor(env, cfg)
	if cfg is not None and cfg.get('isaaclab_canonical_append_task_params', False) and task_params is not None:
		parts.append(task_params)
	return torch.cat(parts, dim=-1)


def _tensor_debug_summary(name, value, max_items=6):
	if value is None:
		return f"{name}: None"
	if not torch.is_tensor(value):
		try:
			value = torch.as_tensor(value)
		except (TypeError, ValueError):
			return f"{name}: {value!r}"
	tensor = value.detach()
	shape = tuple(tensor.shape)
	dtype = str(tensor.dtype).replace("torch.", "")
	device = str(tensor.device)
	flat = tensor.reshape(-1)
	if flat.numel() == 0:
		return f"{name}: shape={shape} dtype={dtype} device={device} empty"

	if torch.is_complex(tensor):
		numeric = flat.abs().to(torch.float32)
	else:
		numeric = flat.to(torch.float32)
	finite = torch.isfinite(numeric)
	finite_count = int(finite.sum().item())
	if finite_count > 0:
		finite_values = numeric[finite]
		stats = (
			f"finite={finite_count}/{flat.numel()} "
			f"min={finite_values.min().item():.6g} "
			f"max={finite_values.max().item():.6g} "
			f"mean={finite_values.mean().item():.6g}"
		)
	else:
		stats = f"finite=0/{flat.numel()}"
	preview = numeric[:max_items].detach().cpu().tolist()
	preview = ", ".join(f"{item:.6g}" for item in preview)
	return f"{name}: shape={shape} dtype={dtype} device={device} {stats} first=[{preview}]"


def _mapping_debug_summary(mapping, max_items=8):
	if not isinstance(mapping, dict) or len(mapping) == 0:
		return None
	parts = []
	for key, value in list(mapping.items())[:max_items]:
		if torch.is_tensor(value):
			flat = value.detach().reshape(-1)
			if flat.numel() == 0:
				parts.append(f"{key}=empty")
			else:
				parts.append(f"{key}={flat[0].item():.6g}")
		else:
			parts.append(f"{key}={value!r}")
	return ", ".join(parts)


def _add_isaaclab_to_sys_path(isaaclab_dir: str):
	root = Path(isaaclab_dir).expanduser().resolve()
	source_root = root / "source"
	paths = [
		source_root / "isaaclab",
		source_root / "isaaclab_tasks",
		source_root / "isaaclab_assets",
		source_root / "isaaclab_mimic",
		source_root / "isaaclab_rl",
		source_root / "isaaclab_contrib",
	]
	for path in paths:
		if path.exists():
			path_str = str(path)
			if path_str not in sys.path:
				sys.path.insert(0, path_str)


def _add_srsa_to_sys_path(srsa_dir: str | None):
	if not srsa_dir:
		return
	root = Path(srsa_dir).expanduser().resolve()
	paths = [
		root / "source" / "SRSA",
		root / "rl_games_sil",
	]
	for path in paths:
		if path.exists():
			path_str = str(path)
			if path_str not in sys.path:
				sys.path.insert(0, path_str)


def _uses_srsa_backend(cfg):
	env_id = cfg.get('isaaclab_env_id', '')
	return (
		cfg.get('isaaclab_backend', 'auto') == 'srsa' or
		cfg.get('isaaclab_task_package', None) == 'SRSA.tasks' or
		env_id.startswith('Assembly-') or
		env_id.startswith('Disassembly-')
	)


def _srsa_position_control_only(cfg):
	return _uses_srsa_backend(cfg) and bool(cfg.get('srsa_position_control_only', True))


def _srsa_policy_action_dim(cfg):
	return int(cfg.get('srsa_policy_action_dim', 3))


def _srsa_env_action_dim(cfg):
	return int(cfg.get('srsa_env_action_dim', 6))


def _bool_env(value):
	return "1" if bool(value) else "0"


def _set_env_if_value(name, value):
	if value is None:
		return
	if isinstance(value, bool):
		os.environ[name] = _bool_env(value)
	else:
		os.environ[name] = str(value)


def _normalize_srsa_task_param_obs_mode(mode):
	normalized = str(mode or "task_vec").strip().lower().replace("-", "_")
	if normalized in {"task_vec", "newt", "newt_task", "newt_task_vec", "axial", "axial_task_vec"}:
		return "task_vec"
	if normalized in {"legacy", "legacy_9d", "task_param", "task_param_tensor"}:
		return "legacy"
	raise ValueError(
		"srsa_task_param_obs_mode must be one of: task_vec, newt, axial, legacy, legacy_9d. "
		f"Got {mode!r}."
	)


def _get_srsa_task_param_obs_tensor(env, cfg=None):
	mode = _normalize_srsa_task_param_obs_mode(
		cfg.get('srsa_task_param_obs_mode', 'task_vec') if cfg is not None else getattr(env, 'task_param_obs_mode', 'task_vec')
	)
	if mode == "task_vec":
		task_vec = getattr(env, 'current_task_vec', None)
		if task_vec is not None:
			return task_vec
	return getattr(env, 'current_task_param_tensor', None)


def _get_srsa_current_task_vec(env):
	task_vec = getattr(env, 'current_task_vec', None)
	if torch.is_tensor(task_vec) and task_vec.ndim >= 2 and task_vec.shape[-1] == 6:
		return task_vec.detach().to(dtype=torch.float32)
	current_task_params = getattr(env, 'current_task_params', None)
	if isinstance(current_task_params, dict):
		explicit = current_task_params.get('task_vec', None)
		if explicit is not None:
			try:
				return torch.as_tensor(explicit, device=env.device, dtype=torch.float32).reshape(1, 6)
			except (TypeError, ValueError, RuntimeError):
				return None
	return None


def _normalize_srsa_eval_success_metric(metric):
	normalized = str(metric or "strict").strip().lower().replace("-", "_")
	aliases = {
		"automate": "official",
		"auto_mate": "official",
		"official_success": "official",
		"official_success_latched": "official",
		"current": "current_official",
		"current_success": "current_official",
		"current_official_success": "current_official",
		"official_terminal": "current_official",
		"official_success_terminal": "current_official",
		"process_success": "process",
		"process_success_terminal": "process",
		"episode_process_success": "episode_process",
		"terminal": "terminal_process",
		"terminal_success": "terminal_process",
		"terminal_process_success": "terminal_process",
		"strict": "terminal_process",
		"strict_success": "terminal_process",
		"strict_success_stable": "terminal_process",
		"relaxed": "relaxed_terminal_process",
		"relaxed_success": "relaxed_terminal_process",
		"relaxed_success_stable": "relaxed_terminal_process",
		"relaxed_terminal": "relaxed_terminal_process",
		"relaxed_terminal_success": "relaxed_terminal_process",
		"relaxed_terminal_process_success": "relaxed_terminal_process",
		"terminal_relaxed_process_success": "relaxed_terminal_process",
		"relaxed_process_success": "relaxed_process",
		"relaxed_process_success_terminal": "relaxed_process",
		"relaxed_episode": "relaxed_episode_process",
		"relaxed_episode_success": "relaxed_episode_process",
		"relaxed_success_episode": "relaxed_episode_process",
		"episode_relaxed_process_success": "relaxed_episode_process",
		"dual_success": "dual",
	}
	normalized = aliases.get(normalized, normalized)
	if normalized not in {
		"official",
		"current_official",
		"process",
		"episode_process",
		"terminal_process",
		"relaxed_process",
		"relaxed_episode_process",
		"relaxed_terminal_process",
		"dual",
	}:
		raise ValueError(
			"srsa_eval_success_metric must be one of: official, current_official, "
			"process, episode_process, terminal_process/strict, relaxed, dual."
		)
	return normalized


def _env_vector(env, value, *, dtype=torch.float32, default=0.0):
	num_envs = int(getattr(env, 'num_envs', 1))
	device = getattr(env, 'device', None)
	if value is None:
		value = default
	if torch.is_tensor(value):
		tensor = value.detach().to(device=device, dtype=dtype)
	else:
		tensor = torch.as_tensor(value, device=device, dtype=dtype)
	if tensor.ndim == 0:
		tensor = tensor.reshape(1)
	if tensor.shape[0] == 1 and num_envs > 1:
		tensor = tensor.expand(num_envs, *tensor.shape[1:])
	return tensor.reshape(num_envs, *tensor.shape[1:])


def _task_param_vector(env, key, *, dtype=torch.float32, default=0.0):
	params = getattr(env, 'current_task_param_tensors', None)
	if isinstance(params, dict) and key in params:
		return _env_vector(env, params[key], dtype=dtype, default=default).reshape(int(getattr(env, 'num_envs', 1)))
	params = getattr(env, 'current_task_params', None)
	if isinstance(params, dict) and key in params:
		return _env_vector(env, params[key], dtype=dtype, default=default).reshape(int(getattr(env, 'num_envs', 1)))
	return _env_vector(env, default, dtype=dtype, default=default).reshape(int(getattr(env, 'num_envs', 1)))


def _task_or_attr_vector(env, key, attr_candidates, *, dtype=torch.float32, default=0.0):
	params = getattr(env, 'current_task_param_tensors', None)
	if isinstance(params, dict) and key in params:
		return _env_vector(env, params[key], dtype=dtype, default=default).reshape(int(getattr(env, 'num_envs', 1)))
	for attr in attr_candidates:
		if hasattr(env, attr):
			return _env_vector(env, getattr(env, attr), dtype=dtype, default=default).reshape(int(getattr(env, 'num_envs', 1)))
	params = getattr(env, 'current_task_params', None)
	if isinstance(params, dict) and key in params:
		return _env_vector(env, params[key], dtype=dtype, default=default).reshape(int(getattr(env, 'num_envs', 1)))
	return _env_vector(env, default, dtype=dtype, default=default).reshape(int(getattr(env, 'num_envs', 1)))


def _clamped_tolerance(base, scale, min_value, max_value):
	tol = base * float(scale)
	if min_value is not None:
		tol = torch.maximum(tol, torch.full_like(tol, float(min_value)))
	if max_value is not None:
		tol = torch.minimum(tol, torch.full_like(tol, float(max_value)))
	return tol


def _compute_srsa_current_official_success(env):
	num_envs = int(getattr(env, 'num_envs', 1))
	device = getattr(env, 'device', None)
	zeros = torch.zeros((num_envs,), dtype=torch.bool, device=device)
	required = ("held_pos", "fixed_pos", "keypoints_held", "keypoints_fixed")
	if not all(hasattr(env, name) for name in required):
		return zeros
	try:
		from isaaclab_tasks.direct.automate import automate_algo_utils as automate_algo

		insertion_depth = _task_or_attr_vector(
			env,
			"insertion_depth",
			("current_insertion_depth_tensor", "disassembly_dists", "current_insertion_depth"),
			default=0.0,
		)
		close_error_thresh = _task_or_attr_vector(
			env,
			"success_pos_tol",
			("current_close_error_thresh_tensor", "current_close_error_thresh"),
			default=float(getattr(getattr(env, 'cfg_task', None), 'close_error_thresh', 0.015)),
		)
		return automate_algo.check_plug_inserted_in_socket(
			env.held_pos,
			env.fixed_pos,
			insertion_depth,
			env.keypoints_held,
			env.keypoints_fixed,
			close_error_thresh,
			env.episode_length_buf,
		).to(dtype=torch.bool)
	except Exception:
		if hasattr(env, 'ep_succeeded'):
			return _env_vector(env, env.ep_succeeded, dtype=torch.bool, default=False).reshape(num_envs)
		return zeros


def _compute_srsa_success_metrics(
	env,
	cfg,
	process_streak,
	episode_process_success,
	relaxed_process_streak=None,
	episode_relaxed_process_success=None,
):
	num_envs = int(getattr(env, 'num_envs', 1))
	device = getattr(env, 'device', None)
	zeros = torch.zeros((num_envs,), dtype=torch.float32, device=device)
	false = torch.zeros((num_envs,), dtype=torch.bool, device=device)
	if relaxed_process_streak is None:
		relaxed_process_streak = torch.zeros((num_envs,), dtype=torch.int64, device=device)
	if episode_relaxed_process_success is None:
		episode_relaxed_process_success = false.clone()

	current_official = _compute_srsa_current_official_success(env)
	if hasattr(env, 'ep_succeeded'):
		official = _env_vector(env, env.ep_succeeded, dtype=torch.bool, default=False).reshape(num_envs)
	else:
		official = current_official

	target_depth = _task_or_attr_vector(
		env,
		"insertion_depth",
		("current_insertion_depth_tensor", "disassembly_dists", "current_insertion_depth"),
		default=0.0,
	)
	radial_clearance = _task_param_vector(env, "radial_clearance", default=0.0)
	if torch.all(radial_clearance <= 0.0):
		diametral_clearance = _task_param_vector(env, "clearance", default=0.0)
		radial_clearance = 0.5 * diametral_clearance.clamp_min(0.0)

	rel_socket = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
	if all(hasattr(env, name) for name in ("held_pos", "fixed_pos", "fixed_quat")):
		try:
			import isaacsim.core.utils.torch as torch_utils

			rel_socket = torch_utils.quat_apply(
				torch_utils.quat_conjugate(env.fixed_quat),
				env.held_pos - env.fixed_pos,
			).to(dtype=torch.float32)
		except Exception:
			rel_socket = (env.held_pos - env.fixed_pos).to(dtype=torch.float32)

	current_depth = (target_depth - rel_socket[:, 2]).clamp_min(0.0)
	depth_fraction = current_depth / target_depth.clamp_min(1.0e-6)
	height_window_ok = (rel_socket[:, 2] > 0.0) & (rel_socket[:, 2] < target_depth.clamp_min(1.0e-6))
	depth_ok = height_window_ok & (depth_fraction >= float(cfg.get('strict_depth_fraction', cfg.get('srsa_process_success_depth_ratio', 0.90))))

	lateral_error = torch.linalg.norm(rel_socket[:, :2], dim=-1)
	lateral_tol = _clamped_tolerance(
		radial_clearance,
		cfg.get('srsa_process_success_lateral_tol_scale', 2.0),
		cfg.get('strict_lateral_tol_min', cfg.get('srsa_process_success_lateral_tol_min', 0.0005)),
		cfg.get('strict_lateral_tol_max', cfg.get('srsa_process_success_lateral_tol_max', 0.0020)),
	)
	lateral_ok = lateral_error <= lateral_tol

	orientation_error = zeros.clone()
	yaw_error = zeros.clone()
	if all(hasattr(env, name) for name in ("held_quat", "fixed_quat")):
		try:
			import isaacsim.core.utils.torch as torch_utils

			rel_quat = torch_utils.quat_mul(env.held_quat, torch_utils.quat_conjugate(env.fixed_quat))
			rel_quat = torch.where(rel_quat[:, :1] < 0.0, -rel_quat, rel_quat)
			rel_euler = torch.stack(torch_utils.get_euler_xyz(rel_quat), dim=1)
			rel_euler = torch.atan2(torch.sin(rel_euler), torch.cos(rel_euler))
			orientation_error = torch.amax(torch.abs(rel_euler[:, :2]), dim=-1).to(dtype=torch.float32)
			yaw_error = torch.abs(rel_euler[:, 2]).to(dtype=torch.float32)
		except Exception:
			pass
	angle_tol_rad = math.radians(float(cfg.get('strict_angle_tol_deg', 3.0)))
	orientation_ok = orientation_error <= float(cfg.get('srsa_process_success_orientation_tol_rad', angle_tol_rad))
	yaw_required = _task_param_vector(env, "yaw_requirement_float", default=0.0) > 0.5
	if cfg.get('srsa_axial_yaw_requirement', None) is not None:
		yaw_required = torch.full_like(yaw_required, bool(cfg.get('srsa_axial_yaw_requirement', False)))
	yaw_ok = (~yaw_required) | (yaw_error <= float(cfg.get('srsa_process_success_yaw_tol_rad', angle_tol_rad)))
	angle_error = torch.maximum(orientation_error, yaw_error)

	keypoint_error = zeros.clone()
	if all(hasattr(env, name) for name in ("keypoints_held", "keypoints_fixed")):
		keypoint_error = torch.linalg.norm(env.keypoints_fixed - env.keypoints_held, dim=-1).mean(dim=-1).to(dtype=torch.float32)
	keypoint_tol = _clamped_tolerance(
		radial_clearance,
		cfg.get('srsa_process_success_keypoint_tol_scale', 2.0),
		cfg.get('strict_keypoint_tol_min', cfg.get('srsa_process_success_keypoint_tol_min', 0.0010)),
		cfg.get('strict_keypoint_tol_max', cfg.get('srsa_process_success_keypoint_tol_max', 0.0030)),
	)
	keypoint_ok = keypoint_error <= keypoint_tol

	contact = false
	if hasattr(env, 'flange_force_flag'):
		contact = _env_vector(env, env.flange_force_flag, dtype=torch.bool, default=False).reshape(num_envs)
	jam_lateral_thresh = torch.maximum(radial_clearance, lateral_tol)
	jam = contact & (~current_official) & (lateral_error > jam_lateral_thresh)

	process = depth_ok & lateral_ok & orientation_ok & yaw_ok & keypoint_ok
	if bool(cfg.get('srsa_process_success_require_no_jam', True)):
		process = process & (~jam)
	if bool(cfg.get('srsa_process_success_require_official', False)):
		process = process & current_official

	stable_steps = max(1, int(cfg.get('strict_success_steps', cfg.get('srsa_process_success_stable_steps', 10))))
	process_streak = torch.where(process, process_streak + 1, torch.zeros_like(process_streak))
	terminal_process = process & (process_streak >= stable_steps)
	episode_process_success = episode_process_success | terminal_process

	relaxed_depth_ok = height_window_ok & (depth_fraction >= float(cfg.get('relaxed_depth_fraction', 0.85)))
	relaxed_lateral_tol = _clamped_tolerance(
		radial_clearance,
		cfg.get('relaxed_lateral_tol_scale', 2.0),
		cfg.get('relaxed_lateral_tol_min', 0.0010),
		cfg.get('relaxed_lateral_tol_max', 0.0030),
	)
	relaxed_lateral_ok = lateral_error <= relaxed_lateral_tol
	relaxed_angle_tol_rad = math.radians(float(cfg.get('relaxed_angle_tol_deg', 5.0)))
	relaxed_orientation_ok = orientation_error <= relaxed_angle_tol_rad
	relaxed_yaw_ok = (~yaw_required) | (yaw_error <= relaxed_angle_tol_rad)
	relaxed_keypoint_tol = _clamped_tolerance(
		radial_clearance,
		cfg.get('relaxed_keypoint_tol_scale', 2.0),
		cfg.get('relaxed_keypoint_tol_min', 0.0010),
		cfg.get('relaxed_keypoint_tol_max', 0.0030),
	)
	relaxed_keypoint_ok = keypoint_error <= relaxed_keypoint_tol
	relaxed_jam_lateral_thresh = torch.maximum(radial_clearance, relaxed_lateral_tol)
	relaxed_jam = contact & (~current_official) & (lateral_error > relaxed_jam_lateral_thresh)
	relaxed_process = relaxed_depth_ok & relaxed_lateral_ok
	if bool(cfg.get('relaxed_success_require_no_jam', True)):
		relaxed_process = relaxed_process & (~relaxed_jam)
	if bool(cfg.get('relaxed_success_require_official', False)):
		relaxed_process = relaxed_process & current_official
	relaxed_stable_steps = max(1, int(cfg.get('relaxed_success_steps', 3)))
	relaxed_process_streak = torch.where(
		relaxed_process,
		relaxed_process_streak + 1,
		torch.zeros_like(relaxed_process_streak),
	)
	relaxed_terminal_process = relaxed_process & (relaxed_process_streak >= relaxed_stable_steps)
	episode_relaxed_process_success = episode_relaxed_process_success | relaxed_terminal_process
	dual = official & terminal_process

	return {
		"success_metric": _normalize_srsa_eval_success_metric(cfg.get('eval_success_metric', cfg.get('srsa_eval_success_metric', 'strict'))),
		"official_success_latched": official,
		"official_success_terminal": current_official,
		"process_success_terminal": process,
		"relaxed_process_success_terminal": relaxed_process,
		"relaxed_success_stable": relaxed_terminal_process,
		"relaxed_success_episode": episode_relaxed_process_success,
		"strict_success_stable": terminal_process,
		"strict_success_episode": episode_process_success,
		"official_success": official,
		"current_official_success": current_official,
		"process_success": process,
		"relaxed_process_success": relaxed_process,
		"episode_relaxed_process_success": episode_relaxed_process_success,
		"relaxed_terminal_process_success": relaxed_terminal_process,
		"episode_process_success": episode_process_success,
		"terminal_process_success": terminal_process,
		"dual_success": dual,
		"depth_ok": depth_ok,
		"lateral_ok": lateral_ok,
		"orientation_ok": orientation_ok,
		"yaw_ok": yaw_ok,
		"keypoint_ok": keypoint_ok,
		"relaxed_depth_ok": relaxed_depth_ok,
		"relaxed_lateral_ok": relaxed_lateral_ok,
		"relaxed_orientation_ok": relaxed_orientation_ok,
		"relaxed_yaw_ok": relaxed_yaw_ok,
		"relaxed_keypoint_ok": relaxed_keypoint_ok,
		"jam": jam,
		"relaxed_jam": relaxed_jam,
		"depth_fraction": depth_fraction,
		"current_depth": current_depth,
		"target_depth": target_depth,
		"lateral_error": lateral_error,
		"lateral_tol": lateral_tol,
		"relaxed_lateral_tol": relaxed_lateral_tol,
		"angle_error": angle_error,
		"orientation_error": orientation_error,
		"yaw_error": yaw_error,
		"keypoint_error": keypoint_error,
		"keypoint_tol": keypoint_tol,
		"relaxed_keypoint_tol": relaxed_keypoint_tol,
		"radial_clearance": radial_clearance,
		"process_success_streak": process_streak.to(dtype=torch.float32),
		"relaxed_process_success_streak": relaxed_process_streak.to(dtype=torch.float32),
		"_process_success_streak": process_streak,
		"_relaxed_process_success_streak": relaxed_process_streak,
	}


def _select_srsa_success(metrics):
	metric = metrics["success_metric"]
	if metric == "official":
		return metrics["official_success"]
	if metric == "current_official":
		return metrics["current_official_success"]
	if metric == "process":
		return metrics["process_success"]
	if metric == "episode_process":
		return metrics["episode_process_success"]
	if metric == "terminal_process":
		return metrics["terminal_process_success"]
	if metric == "relaxed_process":
		return metrics["relaxed_process_success"]
	if metric == "relaxed_episode_process":
		return metrics["episode_relaxed_process_success"]
	if metric == "relaxed_terminal_process":
		return metrics["relaxed_terminal_process_success"]
	if metric == "dual":
		return metrics["dual_success"]
	raise AssertionError(f"Unhandled SRSA success metric: {metric}")


def _float_metric(value):
	if torch.is_tensor(value):
		if value.dtype == torch.bool:
			return value.detach().to(dtype=torch.float32)
		if value.dtype.is_floating_point:
			return value.detach().to(dtype=torch.float32)
		return value.detach().to(dtype=torch.float32)
	return value


def _configure_srsa_runtime_env(cfg):
	if not _uses_srsa_backend(cfg):
		return
	_set_env_if_value("SRSA_ASSEMBLY_ID", cfg.get('assembly_id', None))
	_set_env_if_value("SRSA_TASK_FAMILY_NAME", cfg.get('srsa_task_family_name', None))
	_set_env_if_value("SRSA_TASK_FAMILY_ID", cfg.get('srsa_task_family_id', None))
	_set_env_if_value("SRSA_PLUG_DIAMETER", cfg.get('srsa_plug_diameter', None))
	_set_env_if_value("SRSA_HOLE_DIAMETER", cfg.get('srsa_hole_diameter', None))
	_set_env_if_value("SRSA_CLEARANCE", cfg.get('srsa_clearance', None))
	_set_env_if_value("SRSA_CLEARANCE_RATIO", cfg.get('srsa_clearance_ratio', None))
	_set_env_if_value("SRSA_INSERTION_DEPTH", cfg.get('srsa_insertion_depth', None))
	_set_env_if_value("SRSA_SUCCESS_POS_TOL", cfg.get('srsa_success_pos_tol', None))
	_set_env_if_value("SRSA_TASK_PARAM_OBS", cfg.get('srsa_task_param_obs', False))
	_set_env_if_value("SRSA_TASK_PARAM_OBS_MODE", _normalize_srsa_task_param_obs_mode(cfg.get('srsa_task_param_obs_mode', 'task_vec')))
	_set_env_if_value("SRSA_NEWT_OBS", cfg.get('srsa_newt_obs', False))
	_set_env_if_value("SRSA_ENABLE_AXIAL_TASK_PARAM_SAMPLER", cfg.get('srsa_enable_axial_task_param_sampler', True))
	_set_env_if_value("SRSA_AXIAL_TASK_TYPE_ID", cfg.get('srsa_axial_task_type_id', cfg.get('axial_task_type_id', None)))
	_set_env_if_value("SRSA_AXIAL_SCALE_RANGE", cfg.get('srsa_axial_scale_range', None))
	_set_env_if_value("SRSA_AXIAL_FIXED_PLUG_SCALE", cfg.get('srsa_axial_fixed_plug_scale', None))
	_set_env_if_value("SRSA_AXIAL_CLEARANCE_RANGE", cfg.get('srsa_axial_clearance_range', None))
	_set_env_if_value("SRSA_AXIAL_CLEARANCE_RATIO_RANGE", cfg.get('srsa_axial_clearance_ratio_range', None))
	_set_env_if_value("SRSA_AXIAL_CLEARANCE_BASE", cfg.get('srsa_axial_clearance_base', None))
	_set_env_if_value("SRSA_AXIAL_CLEARANCE_ANCHOR_MULTIPLIERS", cfg.get('srsa_axial_clearance_anchor_multipliers', None))
	_set_env_if_value("SRSA_AXIAL_CLEARANCE_ANCHORS", cfg.get('srsa_axial_clearance_anchors', None))
	_set_env_if_value("SRSA_AXIAL_CLEARANCE_JITTER_RATIO", cfg.get('srsa_axial_clearance_jitter_ratio', None))
	_set_env_if_value("SRSA_AXIAL_CLEARANCE_ANCHOR_WEIGHTS", cfg.get('srsa_axial_clearance_anchor_weights', None))
	_set_env_if_value("SRSA_AXIAL_DEPTH_RANGE", cfg.get('srsa_axial_depth_range', None))
	_set_env_if_value("SRSA_AXIAL_TARGET_DEPTH_RANGE", cfg.get('srsa_axial_target_depth_range', None))
	_set_env_if_value("SRSA_AXIAL_DEPTH_BASE", cfg.get('srsa_axial_depth_base', None))
	_set_env_if_value("SRSA_AXIAL_DEPTH_ANCHOR_MULTIPLIERS", cfg.get('srsa_axial_depth_anchor_multipliers', None))
	_set_env_if_value("SRSA_AXIAL_DEPTH_ANCHORS", cfg.get('srsa_axial_depth_anchors', None))
	_set_env_if_value("SRSA_AXIAL_DEPTH_JITTER_RATIO", cfg.get('srsa_axial_depth_jitter_ratio', None))
	_set_env_if_value("SRSA_AXIAL_DEPTH_ANCHOR_WEIGHTS", cfg.get('srsa_axial_depth_anchor_weights', None))
	_set_env_if_value("SRSA_AXIAL_CLEARANCE_DEPTH_TEMPLATE_MULTIPLIERS", cfg.get('srsa_axial_clearance_depth_template_multipliers', None))
	_set_env_if_value("SRSA_AXIAL_CLEARANCE_DEPTH_TEMPLATES", cfg.get('srsa_axial_clearance_depth_templates', None))
	_set_env_if_value("SRSA_AXIAL_CLEARANCE_DEPTH_TEMPLATE_WEIGHTS", cfg.get('srsa_axial_clearance_depth_template_weights', None))
	_set_env_if_value("SRSA_AXIAL_INIT_ERROR_XY_RANGE", cfg.get('srsa_axial_init_error_xy_range', None))
	_set_env_if_value("SRSA_AXIAL_INIT_ERROR_Z_RANGE", cfg.get('srsa_axial_init_error_z_range', None))
	_set_env_if_value("SRSA_AXIAL_INIT_ERROR_YAW_RANGE", cfg.get('srsa_axial_init_error_yaw_range', None))
	_set_env_if_value("SRSA_AXIAL_VISUAL_NOISE_XY_RANGE", cfg.get('srsa_axial_visual_noise_xy_range', None))
	_set_env_if_value("SRSA_AXIAL_VISUAL_NOISE_Z_RANGE", cfg.get('srsa_axial_visual_noise_z_range', None))
	_set_env_if_value("SRSA_AXIAL_YAW_REQUIREMENT", cfg.get('srsa_axial_yaw_requirement', None))
	_set_env_if_value("SRSA_AXIAL_REFERENCE_RADIUS", cfg.get('srsa_axial_reference_radius', cfg.get('axial_reference_radius', None)))
	_set_env_if_value("SRSA_AXIAL_REFERENCE_DEPTH", cfg.get('srsa_axial_reference_depth', cfg.get('axial_reference_depth', None)))
	_set_env_if_value("SRSA_IF_SBC", cfg.get('srsa_if_sbc', None))
	_set_env_if_value("SRSA_IF_LOGGING_EVAL", cfg.get('srsa_if_logging_eval', False))
	eval_filename = cfg.get('srsa_eval_filename', None)
	if eval_filename is None and cfg.get('srsa_if_logging_eval', False):
		eval_filename = f"evaluation_{cfg.assembly_id}.h5"
	_set_env_if_value("SRSA_EVAL_FILENAME", eval_filename)
	_set_env_if_value("SRSA_NUM_EVAL_TRIALS", cfg.get('srsa_num_eval_trials', 100))
	_set_env_if_value("VISION_NOISE_XY_STD", cfg.get('srsa_vision_noise_xy_std', 0.0))
	_set_env_if_value("VISION_NOISE_XY_JITTER_STD", cfg.get('srsa_vision_noise_xy_jitter_std', 0.0))
	_set_env_if_value("VISION_NOISE_Z_STD", cfg.get('srsa_vision_noise_z_std', 0.0))
	_set_env_if_value("VISION_NOISE_Z_JITTER_STD", cfg.get('srsa_vision_noise_z_jitter_std', 0.0))
	_set_env_if_value("SRSA_ENABLE_FLANGE_FORCE_SENSOR", cfg.get('srsa_enable_flange_force_sensor', False))
	_set_env_if_value("SRSA_FLANGE_FORCE_SENSOR_BODY_NAME", cfg.get('srsa_flange_force_sensor_body_name', 'panda_hand'))
	_set_env_if_value("SRSA_FLANGE_FORCE_SENSOR_SOURCE", cfg.get('srsa_flange_force_sensor_source', 'held_sensor'))
	_set_env_if_value("SRSA_FLANGE_FORCE_SENSOR_OBS_FRAME", cfg.get('srsa_flange_force_sensor_obs_frame', 'socket'))
	_set_env_if_value("SRSA_FLANGE_FORCE_SENSOR_OBS_SCALE", cfg.get('srsa_flange_force_sensor_obs_scale', 50.0))
	_set_env_if_value(
		"SRSA_FLANGE_FORCE_SENSOR_FORCE_THRESHOLD",
		cfg.get('srsa_flange_force_sensor_force_threshold', 1.0),
	)


def _import_task_packages(cfg):
	if _uses_srsa_backend(cfg):
		_add_srsa_to_sys_path(cfg.get('srsa_dir', None))
	modules = []
	if cfg.get('isaaclab_task_package', None):
		modules.append(cfg.isaaclab_task_package)
	if _uses_srsa_backend(cfg) and 'SRSA.tasks' not in modules:
		modules.append('SRSA.tasks')
	for module_name in dict.fromkeys(modules):
		importlib.import_module(module_name)


def _prepare_isaaclab_workdir(cfg):
	global _ISAACLAB_WORKDIR
	if _ISAACLAB_WORKDIR is not None:
		return _ISAACLAB_WORKDIR
	base_dir = Path(cfg.work_dir) / "isaaclab_runtime" / f"rank_{cfg.rank}"
	base_dir.mkdir(parents=True, exist_ok=True)
	os.chdir(base_dir)
	_ISAACLAB_WORKDIR = base_dir
	print(f'[Rank {cfg.rank}] Using Isaac Lab runtime dir: {base_dir}')
	return _ISAACLAB_WORKDIR


class TorchSoftDTW:
	"""
	A small SoftDTW implementation that stays entirely inside PyTorch.
	This avoids Numba CUDA context issues inside Isaac Sim processes.
	"""

	def __init__(self, device, gamma=1.0):
		self.device = device
		self.gamma = float(gamma)
		self.use_cuda = str(device).startswith('cuda')

	@staticmethod
	def _pairwise_sq_dist(x, y):
		n = x.size(1)
		m = y.size(1)
		d = x.size(2)
		x = x.unsqueeze(2).expand(-1, n, m, d)
		y = y.unsqueeze(1).expand(-1, n, m, d)
		return torch.pow(x - y, 2).sum(dim=3)

	@torch.no_grad()
	def __call__(self, x, y):
		dists = self._pairwise_sq_dist(x, y)
		batch, n, m = dists.shape
		r = torch.full(
			(batch, n + 1, m + 1),
			float('inf'),
			device=dists.device,
			dtype=dists.dtype,
		)
		r[:, 0, 0] = 0.0

		if self.gamma <= 0.0:
			for i in range(1, n + 1):
				for j in range(1, m + 1):
					prev = torch.minimum(torch.minimum(r[:, i - 1, j - 1], r[:, i - 1, j]), r[:, i, j - 1])
					r[:, i, j] = dists[:, i - 1, j - 1] + prev
			return r[:, -1, -1]

		gamma = self.gamma
		for i in range(1, n + 1):
			for j in range(1, m + 1):
				r0 = -r[:, i - 1, j - 1] / gamma
				r1 = -r[:, i - 1, j] / gamma
				r2 = -r[:, i, j - 1] / gamma
				rmax = torch.maximum(torch.maximum(r0, r1), r2)
				rsum = torch.exp(r0 - rmax) + torch.exp(r1 - rmax) + torch.exp(r2 - rmax)
				softmin = -gamma * (torch.log(rsum) + rmax)
				r[:, i, j] = dists[:, i - 1, j - 1] + softmin
		return r[:, -1, -1]


def _launch_isaaclab_app(cfg):
	global _APP_LAUNCHER, _SIMULATION_APP
	if _APP_LAUNCHER is not None:
		return
	_add_isaaclab_to_sys_path(cfg.isaaclab_dir)
	_prepare_isaaclab_workdir(cfg)
	from isaaclab.app import AppLauncher

	enable_cameras = cfg.isaaclab_enable_cameras or cfg.save_video or cfg.obs == 'rgb'
	device = f"cuda:{cfg.device_id}" if torch.cuda.is_available() else "cpu"
	_APP_LAUNCHER = AppLauncher(
		headless=cfg.isaaclab_headless,
		enable_cameras=enable_cameras,
		device=device,
	)
	_SIMULATION_APP = _APP_LAUNCHER.app


def _configure_assembly_task(env_cfg, cfg):
	task_name = cfg.isaaclab_task_name
	if task_name not in env_cfg.tasks:
		raise ValueError(f'Unknown Isaac Lab task "{task_name}". Available tasks: {list(env_cfg.tasks.keys())}')
	task = env_cfg.tasks[task_name]
	base_dir = task.assembly_dir.rstrip("/").rsplit("/", 1)[0]
	assembly_dir = f"{base_dir}/{cfg.assembly_id}"

	task.assembly_id = cfg.assembly_id
	task.assembly_dir = f"{assembly_dir}/"
	task.disassembly_path_json = f"{assembly_dir}/disassemble_traj.json"
	task.eval_filename = f"evaluation_{cfg.assembly_id}.h5"
	task.if_logging_eval = False
	task.fixed_asset.spawn.usd_path = f"{assembly_dir}/{task.fixed_asset_cfg.usd_path}"
	task.held_asset.spawn.usd_path = f"{assembly_dir}/{task.held_asset_cfg.usd_path}"


def _configure_physx_buffers(env_cfg, cfg):
	value = cfg.get('isaaclab_gpu_collision_stack_size', None)
	if value is None:
		return
	sim_cfg = getattr(env_cfg, 'sim', None)
	physx_cfg = getattr(sim_cfg, 'physx', None)
	if physx_cfg is None or not hasattr(physx_cfg, 'gpu_collision_stack_size'):
		if int(getattr(cfg, 'rank', 0)) == 0:
			print('[isaaclab-warning] Env config has no physx.gpu_collision_stack_size field; ignoring override.')
		return
	value = int(value)
	if value <= 0:
		return
	physx_cfg.gpu_collision_stack_size = value
	if int(getattr(cfg, 'rank', 0)) == 0:
		print(f'[Rank {cfg.rank}] Set PhysX gpu_collision_stack_size={value}.')


def _configure_soft_dtw(env, cfg):
	env_unwrapped = env.unwrapped
	cfg_task = getattr(env_unwrapped, 'cfg_task', None)
	if cfg_task is None:
		return

	if cfg.isaaclab_disable_imitation_reward:
		cfg_task.imitation_rwd_scale = 0.0
		if hasattr(env_unwrapped, 'soft_dtw_criterion'):
			env_unwrapped.soft_dtw_criterion.use_cuda = False
		print(f'[Rank {cfg.rank}] Disabled AutoMate imitation reward for assembly_id={cfg.assembly_id}.')
		return

	if not hasattr(env_unwrapped, 'soft_dtw_criterion'):
		return

	if cfg.isaaclab_force_cpu_softdtw:
		env_unwrapped.soft_dtw_criterion.use_cuda = False
		print(f'[Rank {cfg.rank}] Using Isaac Lab CPU SoftDTW for assembly_id={cfg.assembly_id}.')
		return

	env_unwrapped.soft_dtw_criterion = TorchSoftDTW(
		device=env_unwrapped.device,
		gamma=cfg_task.soft_dtw_gamma,
	)
	print(f'[Rank {cfg.rank}] Replaced AutoMate SoftDTW with torch backend on {env_unwrapped.device}.')


class IsaacLabWrapper(gym.Wrapper):
	"""
	Adapter from Isaac Lab's native batched torch env to Newt's expected interface.
	"""

	def __init__(self, env, cfg):
		super().__init__(env)
		self.env = env
		self.cfg = cfg
		if cfg.obs != 'state':
			raise ValueError('Isaac Lab integration currently supports state observations only.')
		self._use_canonical_obs = bool(getattr(cfg, 'isaaclab_use_canonical_obs', False))
		if self._use_canonical_obs:
			canonical_dim = 14
			if cfg.get('isaaclab_canonical_append_force', False):
				force = getattr(self.env.unwrapped, 'flange_force_obs', None)
				if torch.is_tensor(force) and force.ndim > 0:
					canonical_dim += int(force.shape[-1])
				elif cfg.get('isaaclab_canonical_zero_force', False):
					canonical_dim += int(cfg.get('isaaclab_canonical_force_dim', 3))
			task_params = _get_srsa_task_param_obs_tensor(self.env.unwrapped, cfg)
			if cfg.get('isaaclab_canonical_append_task_params', False) and task_params is not None:
				canonical_dim += int(task_params.shape[-1])
			self.observation_space = gym.spaces.Box(
				low=-np.inf,
				high=np.inf,
				shape=(canonical_dim,),
				dtype=np.float32,
			)
		else:
			self.observation_space = self.env.unwrapped.single_observation_space['policy']
		self._env_action_dim = int(self.env.unwrapped.single_action_space.shape[0])
		self._position_control_only = _srsa_position_control_only(cfg)
		self._policy_action_dim = _srsa_policy_action_dim(cfg) if self._position_control_only else self._env_action_dim
		if self._position_control_only:
			configured_env_action_dim = _srsa_env_action_dim(cfg)
			if self._env_action_dim != configured_env_action_dim and int(getattr(self.cfg, 'rank', 0)) == 0:
				print(
					f"[isaaclab-warning] SRSA env action_dim={self._env_action_dim}, "
					f"configured srsa_env_action_dim={configured_env_action_dim}; using env action_dim."
				)
			if not (0 < self._policy_action_dim <= self._env_action_dim):
				raise ValueError(
					f"Expected 0 < srsa_policy_action_dim <= env_action_dim, got "
					f"{self._policy_action_dim} and {self._env_action_dim}."
				)
		self.action_space = gym.spaces.Box(
			low=-1.0,
			high=1.0,
			shape=(self._policy_action_dim,),
			dtype=np.float32,
		)
		# AutoMate times out when episode_length_buf >= max_episode_length - 1.
		self.max_episode_steps = max(1, int(self.env.unwrapped.max_episode_length) - 1)
		self._debug_io = bool(cfg.get('isaaclab_debug_io', False))
		self._debug_io_steps = max(0, int(cfg.get('isaaclab_debug_io_steps', 3)))
		self._debug_io_every = max(1, int(cfg.get('isaaclab_debug_io_every', 1)))
		self._debug_step_index = 0
		self._debug_reset_printed = False
		self._srsa_process_success_streak = torch.zeros(
			(self.cfg.num_envs,),
			dtype=torch.int64,
			device=self.env.unwrapped.device,
		)
		self._srsa_relaxed_process_success_streak = torch.zeros(
			(self.cfg.num_envs,),
			dtype=torch.int64,
			device=self.env.unwrapped.device,
		)
		self._srsa_episode_process_success = torch.zeros(
			(self.cfg.num_envs,),
			dtype=torch.bool,
			device=self.env.unwrapped.device,
		)
		self._srsa_episode_relaxed_process_success = torch.zeros(
			(self.cfg.num_envs,),
			dtype=torch.bool,
			device=self.env.unwrapped.device,
		)
		self._configure_srsa_direct_reward_success()
		self._warned_missing_ep_success = False
		self._warned_missing_eval_terminate_key = False
		if self._debug_io and int(getattr(self.cfg, 'rank', 0)) == 0:
			print(
				"[isaaclab-debug] enabled "
				f"steps={self._debug_io_steps} every={self._debug_io_every} "
				f"canonical={self._use_canonical_obs} "
				f"append_force={cfg.get('isaaclab_canonical_append_force', False)} "
				f"zero_force={cfg.get('isaaclab_canonical_zero_force', False)} "
				f"append_task_params={cfg.get('isaaclab_canonical_append_task_params', False)} "
				f"task_param_mode={cfg.get('srsa_task_param_obs_mode', 'task_vec')} "
				f"visual_noise={cfg.get('isaaclab_canonical_use_visual_noise', False)}"
			)

	def _debug_event_index(self, phase):
		if not self._debug_io or int(getattr(self.cfg, 'rank', 0)) != 0:
			return None
		if phase == "reset":
			if self._debug_reset_printed:
				return None
			self._debug_reset_printed = True
			return 0
		self._debug_step_index += 1
		if self._debug_step_index > self._debug_io_steps:
			return None
		if (self._debug_step_index - 1) % self._debug_io_every != 0:
			return None
		return self._debug_step_index

	def _debug_canonical_obs(self, obs):
		if not self._use_canonical_obs or obs is None:
			return
		env = self.env.unwrapped
		obs_dim = int(obs.shape[-1])
		offset = 0
		parts = [
			("tcp_pos_socket", 3),
			("tcp_quat_socket_wxyz", 4),
			("tcp_linvel_socket", 3),
			("tcp_angvel_socket", 3),
			("gripper_width", 1),
		]
		if self.cfg.get('isaaclab_canonical_append_force', False):
			force = getattr(env, 'flange_force_obs', None)
			if torch.is_tensor(force) and force.ndim > 0:
				force_dim = int(force.shape[-1])
			elif self.cfg.get('isaaclab_canonical_zero_force', False):
				force_dim = int(self.cfg.get('isaaclab_canonical_force_dim', 3))
			else:
				force_dim = max(0, min(3, obs_dim - offset))
			parts.append(("flange_force_obs", force_dim))
		if self.cfg.get('isaaclab_canonical_append_task_params', False):
			task_params = _get_srsa_task_param_obs_tensor(env, self.cfg)
			task_dim = int(task_params.shape[-1]) if torch.is_tensor(task_params) and task_params.ndim > 0 else max(0, obs_dim - offset)
			parts.append(("task_params", task_dim))
		for name, dim in parts:
			if dim <= 0 or offset >= obs_dim:
				continue
			end = min(offset + dim, obs_dim)
			print("[isaaclab-debug] " + _tensor_debug_summary(f"obs.{name}[{offset}:{end}]", obs[..., offset:end]))
			offset = end
		if offset != obs_dim:
			print(f"[isaaclab-debug] obs layout consumed {offset}/{obs_dim} dims")

	def _debug_runtime_tensors(self):
		env = self.env.unwrapped
		for label, attr in [
			("runtime.flange_force_obs", "flange_force_obs"),
			("runtime.flange_force_world", "flange_force_world"),
			("runtime.flange_force_socket", "flange_force_socket"),
			("runtime.flange_force_norm", "flange_force_norm"),
			("runtime.flange_force_flag", "flange_force_flag"),
			("runtime.task_vec", "current_task_vec"),
			("runtime.task_param_tensor", "current_task_param_tensor"),
			("runtime.vision_noise_world", "_vision_noise_world"),
			("runtime.vision_noise_episode_local", "_vision_noise_episode_local"),
		]:
			if hasattr(env, attr):
				print("[isaaclab-debug] " + _tensor_debug_summary(label, getattr(env, attr)))
		task_params = _mapping_debug_summary(getattr(env, 'current_task_params', None))
		if task_params:
			print(f"[isaaclab-debug] runtime.current_task_params: {task_params}")

	def _maybe_debug_io(self, phase, obs=None, action=None, reward=None, terminated=None, truncated=None, info=None, obs_dict=None):
		event_index = self._debug_event_index(phase)
		if event_index is None:
			return
		env = self.env.unwrapped
		print(
			f"[isaaclab-debug] phase={phase} index={event_index} "
			f"env_id={self.cfg.isaaclab_env_id} device={env.device} "
			f"obs_space={self.observation_space.shape} action_space={self.action_space.shape}"
		)
		if obs_dict is not None and isinstance(obs_dict, dict) and 'policy' in obs_dict:
			print("[isaaclab-debug] " + _tensor_debug_summary("raw.policy_obs", obs_dict['policy']))
		if action is not None:
			print("[isaaclab-debug] " + _tensor_debug_summary("input.action", action))
		if obs is not None:
			print("[isaaclab-debug] " + _tensor_debug_summary("output.obs", obs))
			self._debug_canonical_obs(obs)
		if reward is not None:
			print("[isaaclab-debug] " + _tensor_debug_summary("output.reward", reward))
		if terminated is not None:
			print("[isaaclab-debug] " + _tensor_debug_summary("output.terminated", terminated))
		if truncated is not None:
			print("[isaaclab-debug] " + _tensor_debug_summary("output.truncated", truncated))
		if isinstance(info, dict) and 'success' in info:
			print("[isaaclab-debug] " + _tensor_debug_summary("info.success", info['success']))
			for key in (
				"official_success",
				"relaxed_terminal_process_success",
				"terminal_process_success",
				"depth_fraction",
				"lateral_error",
				"jam",
			):
				if key in info:
					print("[isaaclab-debug] " + _tensor_debug_summary(f"info.{key}", info[key]))
		self._debug_runtime_tensors()

	def _reset_srsa_success_state(self, env_ids=None):
		if env_ids is None:
			self._srsa_process_success_streak.zero_()
			self._srsa_relaxed_process_success_streak.zero_()
			self._srsa_episode_process_success.zero_()
			self._srsa_episode_relaxed_process_success.zero_()
			return
		if torch.is_tensor(env_ids) and env_ids.numel() > 0:
			self._srsa_process_success_streak[env_ids] = 0
			self._srsa_relaxed_process_success_streak[env_ids] = 0
			self._srsa_episode_process_success[env_ids] = False
			self._srsa_episode_relaxed_process_success[env_ids] = False

	def _srsa_success_metrics(self, *, update_state):
		env = self.env.unwrapped
		metrics = _compute_srsa_success_metrics(
			env,
			self.cfg,
			self._srsa_process_success_streak,
			self._srsa_episode_process_success,
			self._srsa_relaxed_process_success_streak,
			self._srsa_episode_relaxed_process_success,
		)
		process_streak = metrics.pop("_process_success_streak")
		relaxed_process_streak = metrics.pop("_relaxed_process_success_streak")
		if update_state:
			self._srsa_process_success_streak = process_streak
			self._srsa_relaxed_process_success_streak = relaxed_process_streak
			self._srsa_episode_process_success = metrics["episode_process_success"].detach().clone()
			self._srsa_episode_relaxed_process_success = metrics["episode_relaxed_process_success"].detach().clone()
		return metrics

	def _configure_srsa_direct_reward_success(self):
		if not _uses_srsa_backend(self.cfg):
			return
		if not bool(self.cfg.get('srsa_align_direct_reward_success', False)):
			return
		if bool(self.cfg.get('srsa_sparse_reward', False)):
			return

		env = self.env.unwrapped
		if not hasattr(env, '_update_rew_buf'):
			if int(getattr(self.cfg, 'rank', 0)) == 0:
				print("[isaaclab-warning] SRSA direct reward success alignment requested, but env has no _update_rew_buf.")
			return

		wrapper = self

		def _get_rewards_with_newt_success(env_self):
			current_official = _compute_srsa_current_official_success(env_self)
			if hasattr(env_self, 'ep_succeeded'):
				env_self.ep_succeeded = torch.logical_or(
					env_self.ep_succeeded,
					current_official.to(dtype=torch.bool),
				)

			success_metrics = wrapper._srsa_success_metrics(update_state=False)
			reward_success = _select_srsa_success(success_metrics).to(dtype=torch.bool)
			rew_buf = env_self._update_rew_buf(reward_success)

			if hasattr(env_self, "_update_task_param_extras"):
				env_self._update_task_param_extras()
			if hasattr(env_self, "_update_flange_force_extras"):
				env_self._update_flange_force_extras()
			if hasattr(env_self, "_update_srsa_success_extras"):
				env_self._update_srsa_success_extras(success_metrics)
			if hasattr(env_self, "_update_newt_task_extras"):
				env_self._update_newt_task_extras()
			if hasattr(env_self, "extras") and isinstance(env_self.extras, dict):
				env_self.extras["logs_rew_selected_success"] = reward_success.float().mean()
				if current_official is not None:
					env_self.extras["logs_rew_official_curr_successes"] = current_official.float().mean()

			if torch.any(env_self.reset_buf):
				if hasattr(env_self, "extras") and isinstance(env_self.extras, dict):
					env_self.extras["successes"] = torch.count_nonzero(reward_success) / env_self.num_envs
					if hasattr(env_self, 'ep_succeeded'):
						env_self.extras["official_successes"] = torch.count_nonzero(env_self.ep_succeeded) / env_self.num_envs

				from isaaclab_tasks.direct.automate import automate_algo_utils as automate_algo

				sbc_rwd_scale = automate_algo.get_curriculum_reward_scale(
					curr_max_disp=env_self.curr_max_disp,
					curriculum_height_bound=env_self.curriculum_height_bound,
				)
				rew_buf *= sbc_rwd_scale

				if env_self.cfg_task.if_sbc:
					env_self.curr_max_disp = automate_algo.get_new_max_disp(
						curr_success=torch.count_nonzero(reward_success) / env_self.num_envs,
						cfg_task=env_self.cfg_task,
						curriculum_height_bound=env_self.curriculum_height_bound,
						curriculum_height_step=env_self.curriculum_height_step,
						curr_max_disp=env_self.curr_max_disp,
					)

				if hasattr(env_self, "extras") and isinstance(env_self.extras, dict):
					env_self.extras["curr_max_disp"] = env_self.curr_max_disp

				if env_self.cfg_task.if_logging_eval:
					from isaaclab_tasks.direct.automate import automate_log_utils as automate_log

					success_log = reward_success.reshape((env_self.num_envs, 1))
					env_self.success_log = torch.cat([env_self.success_log, success_log], dim=0)

					if env_self.success_log.shape[0] >= env_self.cfg_task.num_eval_trials:
						automate_log.write_log_to_hdf5(
							env_self.held_asset_pose_log,
							env_self.fixed_asset_pose_log,
							env_self.success_log,
							env_self.cfg_task.eval_filename,
						)
						exit(0)

			env_self.prev_actions = env_self.actions.clone()
			return rew_buf

		env._newt_original_get_rewards = env._get_rewards
		env._get_rewards = types.MethodType(_get_rewards_with_newt_success, env)
		if int(getattr(self.cfg, 'rank', 0)) == 0:
			print(
				"[Rank {rank}] Aligned SRSA direct reward success bonus with eval_success_metric={metric}.".format(
					rank=getattr(self.cfg, 'rank', 0),
					metric=self.cfg.get('eval_success_metric', self.cfg.get('srsa_eval_success_metric', 'strict')),
				)
			)

	def _compute_success_info(self):
		env = self.env.unwrapped
		if _uses_srsa_backend(self.cfg):
			metrics = self._srsa_success_metrics(update_state=True)
			success = _select_srsa_success(metrics).to(dtype=torch.float32)
			info = {
				key: _float_metric(value)
				for key, value in metrics.items()
				if key != "success_metric"
			}
			info["success"] = success
			info["score"] = success.clone()
			return info

		if hasattr(env, 'ep_succeeded'):
			success = env.ep_succeeded.clone().to(torch.float32)
		else:
			success = torch.zeros(self.cfg.num_envs, dtype=torch.float32, device=env.device)
			if not self._warned_missing_ep_success and int(getattr(self.cfg, 'rank', 0)) == 0:
				print("[isaaclab-warning] env has no ep_succeeded; reporting episode success as 0.")
				self._warned_missing_ep_success = True
		return {
			"success": success,
			"score": success.clone(),
		}

	def _final_info_for_done(self, success_info, done, final_obs):
		final_info = {}
		for key, value in success_info.items():
			if not torch.is_tensor(value):
				continue
			tensor = _float_metric(value)
			if tensor.ndim == 0 or tensor.shape[0] != self.cfg.num_envs:
				continue
			final_value = torch.full((self.cfg.num_envs,), float('nan'), dtype=torch.float32, device=done.device)
			final_value[done] = tensor[done]
			final_info[key] = final_value
		return {
			"final_observation": final_obs,
			"final_info": final_info,
		}

	def rand_act(self):
		return torch.rand(
			(self.cfg.num_envs, *self.action_space.shape),
			dtype=torch.float32,
			device=self.env.unwrapped.device,
		) * 2 - 1

	def _expand_policy_action(self, action):
		action = action.to(self.env.unwrapped.device, non_blocking=True)
		if not self._position_control_only:
			return action
		if action.shape[-1] == self._env_action_dim:
			env_action = action.clone()
			env_action[..., self._policy_action_dim:] = 0.0
			return env_action
		if action.shape[-1] != self._policy_action_dim:
			raise ValueError(
				f"Expected SRSA position-control action dim {self._policy_action_dim} "
				f"or env action dim {self._env_action_dim}, got {action.shape[-1]}."
			)
		pad_shape = (*action.shape[:-1], self._env_action_dim - self._policy_action_dim)
		padding = torch.zeros(pad_shape, dtype=action.dtype, device=action.device)
		return torch.cat([action, padding], dim=-1)

	def _extract_obs(self, obs_dict):
		if self._use_canonical_obs:
			return _build_canonical_obs(self.env.unwrapped, self.cfg).detach().to(torch.float32)
		return obs_dict['policy'].detach().to(torch.float32)

	def reset(self, **kwargs):
		obs, _ = self.env.reset(**kwargs)
		self._reset_srsa_success_state()
		extracted_obs = self._extract_obs(obs)
		info = {
			'success': torch.zeros(self.cfg.num_envs, dtype=torch.float32, device=self.env.unwrapped.device)
		}
		self._maybe_debug_io("reset", obs=extracted_obs, info=info, obs_dict=obs)
		return extracted_obs, info

	def _step_with_final_obs(self, action):
		env = self.env.unwrapped
		action = self._expand_policy_action(action)

		if env.cfg.action_noise_model:
			action = env._action_noise_model(action)
		env._pre_physics_step(action)

		is_rendering = env.sim.has_gui() or env.sim.has_rtx_sensors()
		for _ in range(env.cfg.decimation):
			env._sim_step_counter += 1
			env._apply_action()
			env.scene.write_data_to_sim()
			env.sim.step(render=False)
			if env._sim_step_counter % env.cfg.sim.render_interval == 0 and is_rendering:
				env.sim.render()
			env.scene.update(dt=env.physics_dt)

		env.episode_length_buf += 1
		env.common_step_counter += 1
		env.reset_terminated[:], env.reset_time_outs[:] = env._get_dones()
		done = env.reset_terminated | env.reset_time_outs
		env.reset_buf = done
		env.reward_buf = env._get_rewards()

		final_obs = None
		success_info = self._compute_success_info()
		if bool(self.cfg.get('eval_terminate_on_success', False)) and _uses_srsa_backend(self.cfg):
			success_key = str(self.cfg.get('eval_terminate_success_key', 'terminal_process_success'))
			success_value = success_info.get(success_key, None)
			if success_value is None:
				if not self._warned_missing_eval_terminate_key and int(getattr(self.cfg, 'rank', 0)) == 0:
					print(f"[isaaclab-warning] eval_terminate_success_key={success_key!r} not found; ignoring success termination.")
					self._warned_missing_eval_terminate_key = True
			else:
				success_done = success_value.to(device=done.device, dtype=torch.float32) > 0.5
				min_step = max(0, int(self.cfg.get('eval_terminate_min_step', 0)))
				if min_step > 0:
					success_done = success_done & (env.episode_length_buf >= min_step)
				done = done | success_done
				env.reset_buf = done
		if done.any():
			final_obs_dict = env._get_observations()
			final_obs = self._extract_obs(final_obs_dict)[done].clone()

		reset_env_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
		if len(reset_env_ids) > 0:
			env._reset_idx(reset_env_ids)
			self._reset_srsa_success_state(reset_env_ids)
			if env.sim.has_rtx_sensors() and env.cfg.num_rerenders_on_reset > 0:
				for _ in range(env.cfg.num_rerenders_on_reset):
					env.sim.render()

		if env.cfg.events and "interval" in env.event_manager.available_modes:
			env.event_manager.apply(mode="interval", dt=env.step_dt)

		env.obs_buf = env._get_observations()
		if env.cfg.observation_noise_model:
			env.obs_buf["policy"] = env._observation_noise_model(env.obs_buf["policy"])

		return env.obs_buf, env.reward_buf, done, final_obs, success_info

	def step(self, action):
		obs_dict, reward, done, final_obs, success_info = self._step_with_final_obs(action)
		obs = self._extract_obs(obs_dict)
		reward = reward.detach().to(torch.float32)
		done = done.detach()
		terminated = torch.zeros_like(done)
		truncated = done.clone()

		info = {key: value.detach() if torch.is_tensor(value) else value for key, value in success_info.items()}
		if done.any():
			info.update(self._final_info_for_done(success_info, done, final_obs))
		self._maybe_debug_io(
			"step",
			obs=obs,
			action=action,
			reward=reward,
			terminated=terminated,
			truncated=truncated,
			info=info,
			obs_dict=obs_dict,
		)
		return obs, reward, terminated, truncated, info

	def render(self, *args, **kwargs):
		frame = self.env.render(*args, **kwargs)
		return None if frame is None else frame.copy()

	def close(self):
		return self.env.close()


def _maybe_update_axial_task_vector_from_env(cfg, env):
	if cfg.get('task_conditioning', 'axial_params') != 'axial_params':
		return
	current_task_vec = _get_srsa_current_task_vec(env.unwrapped)
	if current_task_vec is not None:
		task_vec = current_task_vec.reshape(-1, current_task_vec.shape[-1])[0].detach().cpu().tolist()
	else:
		current_task_params = getattr(env.unwrapped, 'current_task_params', None)
		if not current_task_params:
			return
		try:
			from config import make_axial_task_vec
		except ImportError:
			return
		task_vec = make_axial_task_vec(cfg, current_task_params)
	if not getattr(cfg, 'task_vectors', None):
		cfg.task_vectors = [task_vec]
	elif len(cfg.task_vectors) == 1:
		cfg.task_vectors[0] = task_vec
	elif int(getattr(cfg, 'num_global_tasks', 1)) == 1:
		cfg.task_vectors = [task_vec for _ in cfg.task_vectors]
	else:
		return
	if int(getattr(cfg, 'rank', 0)) == 0:
		pretty_vec = ", ".join(f"{value:.6g}" for value in task_vec)
		print(f"[Rank {cfg.rank}] Updated axial task_vec_6 from SRSA runtime params: [{pretty_vec}]")


def make_env(cfg):
	"""
	Make an Isaac Lab AutoMate environment.
	"""
	if not (
		cfg.task.startswith('isaaclab-') or
		cfg.task.startswith('Isaac-') or
		cfg.isaaclab_env_id.startswith('Isaac-') or
		_uses_srsa_backend(cfg)
	):
		raise ValueError('Unknown task:', cfg.task)
	_launch_isaaclab_app(cfg)
	import isaaclab_tasks  # noqa: F401
	from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

	_configure_srsa_runtime_env(cfg)
	_import_task_packages(cfg)
	env_cfg = parse_env_cfg(
		cfg.isaaclab_env_id,
		device=f"cuda:{cfg.device_id}" if torch.cuda.is_available() else "cpu",
		num_envs=cfg.num_envs,
		use_fabric=cfg.isaaclab_use_fabric,
	)
	env_cfg.seed = cfg.seed + cfg.rank
	if hasattr(env_cfg, 'task_name'):
		env_cfg.task_name = cfg.isaaclab_task_name
	if cfg.isaaclab_env_id == "Isaac-AutoMate-Assembly-Direct-v0":
		_configure_assembly_task(env_cfg, cfg)
	_configure_physx_buffers(env_cfg, cfg)
	render_mode = 'rgb_array' if (cfg.save_video or cfg.obs == 'rgb') else None
	env = gym.make(cfg.isaaclab_env_id, cfg=env_cfg, render_mode=render_mode)
	_configure_soft_dtw(env, cfg)
	_maybe_update_axial_task_vector_from_env(cfg, env)
	env = IsaacLabWrapper(env, cfg)
	print(f'[Rank {cfg.rank}] Created Isaac Lab env {cfg.isaaclab_env_id} for assembly_id={cfg.assembly_id}')
	if cfg.isaaclab_use_canonical_obs:
		print(f'[Rank {cfg.rank}] Using canonical Isaac Lab observations with shape={env.observation_space.shape}.')
	return env
