import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ["TORCH_DISTRIBUTED_TIMEOUT"] = "1800"
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
os.environ['TORCH_LOGS'] = "+recompiles"
import warnings
warnings.filterwarnings('ignore')

from copy import deepcopy
from pathlib import Path
from time import monotonic
import json
import math
import subprocess
import sys

import hydra
import torch
from hydra.core.config_store import ConfigStore
from tensordict import TensorDict
from termcolor import colored

from common import set_seed
from common import MODEL_SIZE
from common.world_model import WorldModel
from config import Config, SRSA_SAMPLER_CFG_KEYS, make_axial_task_vec, parse_cfg, safe_run_token
from envs import make_env
from offline_io import load_offline_manifest, summarize_compact_dataset
from tdmpc2 import TDMPC2


torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

cs = ConfigStore.instance()
cs.store(name="config", node=Config)


def _checkpoint_state_dict(checkpoint_fp: Path):
	obj = torch.load(checkpoint_fp, map_location="cpu", weights_only=False)
	return obj["model"] if isinstance(obj, dict) and "model" in obj else obj


def _get_state_tensor(state_dict, key: str):
	for candidate in (key, f"module.{key}"):
		if candidate in state_dict:
			return state_dict[candidate]
	return None


def _infer_model_size(enc_dim: int):
	for name, values in MODEL_SIZE.items():
		if int(values.get("enc_dim", -1)) == int(enc_dim):
			return name
	return None


def _infer_checkpoint_compat(checkpoint_fp: Path):
	state_dict = _checkpoint_state_dict(checkpoint_fp)
	enc_weight = _get_state_tensor(state_dict, "_encoder.state.0.weight")
	if enc_weight is None:
		return None
	task_emb = _get_state_tensor(state_dict, "_task_emb.weight")
	task_vecs = _get_state_tensor(state_dict, "_task_vecs")
	task_encoder = _get_state_tensor(state_dict, "_task_encoder.type_encoder.weight")
	if task_emb is not None:
		task_conditioning = "id_embedding"
		task_dim = int(task_emb.shape[-1])
	elif task_vecs is not None or task_encoder is not None:
		task_conditioning = "axial_params"
		task_dim = 64
	else:
		task_conditioning = "none"
		task_dim = 0
	obs_dim = int(enc_weight.shape[1]) - task_dim
	if obs_dim <= 0:
		raise ValueError(
			f"Could not infer a positive observation dim from checkpoint={checkpoint_fp}: "
			f"encoder_in={int(enc_weight.shape[1])}, task_dim={task_dim}."
		)
	return {
		"model_size": _infer_model_size(int(enc_weight.shape[0])),
		"enc_dim": int(enc_weight.shape[0]),
		"obs_dim": obs_dim,
		"task_dim": task_dim,
		"task_conditioning": task_conditioning,
	}


def _normalize_assembly_id(value) -> str:
	value = str(value).strip().strip("'\"")
	if value.isdigit() and len(value) < 5:
		value = value.zfill(5)
	return value


def _parse_assembly_ids(raw) -> list[str]:
	if raw is None:
		return []
	if isinstance(raw, str):
		text = raw.strip()
		if text.startswith("[") and text.endswith("]"):
			text = text[1:-1]
		items = [item for item in text.replace(";", ",").replace(" ", ",").split(",") if item.strip()]
	else:
		items = list(raw)
	ids = [_normalize_assembly_id(item) for item in items]
	return list(dict.fromkeys(ids))


def _resolve_assembly_ids(cfg) -> list[str]:
	assembly_ids = _parse_assembly_ids(cfg.get('collect_assembly_ids', None))
	if not assembly_ids and cfg.get('offline_manifest_fp', None):
		entries = load_offline_manifest(cfg.offline_manifest_fp)
		assembly_ids = [
			_normalize_assembly_id(entry["assembly_id"])
			for entry in entries
			if entry.get("assembly_id") is not None
		]
	source_id = cfg.get('collect_source_assembly_id', None)
	if source_id is not None and cfg.get('collect_exclude_source_assembly', True):
		source_id = _normalize_assembly_id(source_id)
		assembly_ids = [assembly_id for assembly_id in assembly_ids if assembly_id != source_id]
	if not assembly_ids:
		raise ValueError(
			"Provide target ids with `collect_assembly_ids=[00141,00211,...]` "
			"or provide `offline_manifest_fp` containing assembly_id values."
		)
	return assembly_ids


def _resolve_output_dir(cfg) -> Path:
	if cfg.get('collect_output_dir', None):
		return Path(cfg.collect_output_dir).expanduser().resolve()
	source_raw = cfg.get('collect_source_assembly_id', '00186')
	source_id = safe_run_token(_normalize_assembly_id(source_raw) if source_raw is not None else "all")
	return Path(cfg.work_dir) / "data" / f"policy_rollouts_from_{source_id}"


def _resolve_manifest_fp(cfg, output_dir: Path) -> Path:
	if cfg.get('collect_manifest_fp', None):
		return Path(cfg.collect_manifest_fp).expanduser().resolve()
	return output_dir / "offline_manifest_eval_rollouts.json"


def _apply_checkpoint_compat(cfg, checkpoint_fp: Path):
	if not cfg.get('collect_match_checkpoint', True):
		return
	compat = _infer_checkpoint_compat(checkpoint_fp)
	if compat is None:
		return
	if compat["model_size"] is not None:
		cfg.model_size = compat["model_size"]
	cfg.task_conditioning = compat["task_conditioning"]
	cfg.collect_expected_obs_dim = compat["obs_dim"]
	if compat["task_conditioning"] == "id_embedding":
		cfg.isaaclab_use_canonical_obs = False
		cfg.isaaclab_canonical_append_force = False
		cfg.isaaclab_canonical_append_task_params = False
		cfg.isaaclab_canonical_use_visual_noise = False
		cfg.srsa_enable_flange_force_sensor = False
		cfg.task_dim = compat["task_dim"]
	print(colored(
		"Matched collection config to checkpoint: "
		f"model_size={cfg.get('model_size', None)} "
		f"task_conditioning={cfg.task_conditioning} "
		f"obs_dim={compat['obs_dim']} task_dim={compat['task_dim']} "
		f"srsa_force_sensor={cfg.get('srsa_enable_flange_force_sensor', None)}.",
		"cyan",
		attrs=["bold"],
	))


def _adapt_obs_to_checkpoint(obs: torch.Tensor, expected_obs_dim: int | None) -> torch.Tensor:
	if expected_obs_dim is None:
		return obs
	actual_obs_dim = int(obs.shape[-1])
	expected_obs_dim = int(expected_obs_dim)
	if actual_obs_dim == expected_obs_dim:
		return obs
	if actual_obs_dim > expected_obs_dim:
		return obs[..., :expected_obs_dim].contiguous()
	raise RuntimeError(
		f"Observation dim {actual_obs_dim} is smaller than checkpoint obs_dim={expected_obs_dim}; cannot adapt safely."
	)


def _task_cfg(base_cfg, assembly_id: str):
	cfg = deepcopy(base_cfg)
	cfg.assembly_id = assembly_id
	cfg.rank = 0
	cfg.world_size = 1
	cfg.device_id = int(cfg.gpu_id)
	cfg.multiproc = False
	cfg.enable_wandb = False
	cfg.save_agent = False
	return cfg


def _make_agent(cfg):
	model = WorldModel(cfg).to(f"cuda:{cfg.device_id}")
	target_task_vecs = getattr(model, "_task_vecs", None)
	if target_task_vecs is not None:
		target_task_vecs = target_task_vecs.detach().clone()
	agent = TDMPC2(model, cfg)
	try:
		agent.load(cfg.checkpoint)
	except Exception as exc:
		raise RuntimeError(
			"Failed to load checkpoint into the collection model. "
			"Check that the checkpoint was trained with the same architecture as this command "
			"(model_size, observation shape, and task_conditioning). "
			f"checkpoint={cfg.checkpoint}, task={cfg.task}, "
			f"model_size={cfg.get('model_size', None)}, task_conditioning={cfg.get('task_conditioning', None)}, "
			f"obs_shape={cfg.get('obs_shape', None)}. "
			"For the current SRSA canonical axial command, use an SRSA axial checkpoint rather than an older "
			"isaaclab-automate/id-embedding checkpoint."
		) from exc
	if target_task_vecs is not None:
		agent.model._task_vecs.copy_(target_task_vecs.to(agent.model._task_vecs.device))
	agent.eval()
	agent.model.eval()
	return agent


def _current_task_vec_6(cfg, env):
	current_task_vec = getattr(env.unwrapped, "current_task_vec", None)
	if torch.is_tensor(current_task_vec) and current_task_vec.ndim >= 2 and current_task_vec.shape[-1] == 6:
		return current_task_vec.reshape(-1, current_task_vec.shape[-1])[0].detach().cpu().tolist()
	current_task_params = getattr(env.unwrapped, "current_task_params", None)
	if not current_task_params:
		return None
	try:
		return make_axial_task_vec(cfg, current_task_params)
	except Exception:
		return None


def _model_task_input(cfg, env, fallback_tasks):
	if (
		not bool(cfg.get('srsa_use_runtime_task_vec', True)) or
		str(cfg.get('task_conditioning', '')).lower() != 'axial_params'
	):
		return fallback_tasks
	current_task_vec = getattr(env.unwrapped, "current_task_vec", None)
	if not torch.is_tensor(current_task_vec):
		return fallback_tasks
	if current_task_vec.ndim != 2 or current_task_vec.shape[0] != cfg.num_envs or current_task_vec.shape[-1] != 6:
		return fallback_tasks
	return current_task_vec.detach().to(fallback_tasks.device, dtype=torch.float32, non_blocking=True).clone()


def _empty_columns():
	return {
		"obs": [],
		"next_obs": [],
		"action": [],
		"reward": [],
		"done": [],
		"terminated": [],
		"truncated": [],
		"episode": [],
		"step_id": [],
		"task": [],
		"episode_return_running": [],
		"episode_return_final": [],
		"episode_success_final": [],
		"episode_failure_final": [],
		"terminal_success": [],
		"terminal_failure": [],
		"success_episode": [],
	}


def _append_finished_episode(columns, rows, *, episode_id: int, task_id: int, final_return: float, final_success: float):
	final_failure = 1.0 - final_success
	last_index = len(rows) - 1
	for index, row in enumerate(rows):
		is_terminal = index == last_index
		columns["obs"].append(row["obs"])
		columns["next_obs"].append(row["next_obs"])
		columns["action"].append(row["action"])
		columns["reward"].append(row["reward"])
		columns["done"].append(row["done"])
		columns["terminated"].append(row["terminated"])
		columns["truncated"].append(row["truncated"])
		columns["episode"].append(torch.tensor(episode_id, dtype=torch.int64))
		columns["step_id"].append(row["step_id"])
		columns["task"].append(torch.tensor(task_id, dtype=torch.int64))
		columns["episode_return_running"].append(row["episode_return_running"])
		columns["episode_return_final"].append(torch.tensor(final_return, dtype=torch.float32))
		columns["episode_success_final"].append(torch.tensor(final_success, dtype=torch.float32))
		columns["episode_failure_final"].append(torch.tensor(final_failure, dtype=torch.float32))
		columns["terminal_success"].append(torch.tensor(bool(is_terminal and final_success > 0.5), dtype=torch.bool))
		columns["terminal_failure"].append(torch.tensor(bool(is_terminal and final_success <= 0.5), dtype=torch.bool))
		columns["success_episode"].append(torch.tensor(final_success, dtype=torch.float32))


def _stack_columns(columns):
	if len(columns["obs"]) == 0:
		raise RuntimeError("No completed rollout episodes were collected.")
	data = {}
	for key, values in columns.items():
		data[key] = torch.stack(values, dim=0).contiguous()
	return TensorDict(data, batch_size=(data["obs"].shape[0],))


def _write_json(path: Path, obj):
	path.parent.mkdir(parents=True, exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		json.dump(obj, f, indent=2, ensure_ascii=True)


def _read_json(path: Path):
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def _json_safe(value):
	if torch.is_tensor(value):
		value = value.detach().cpu()
		if value.numel() == 1:
			return float(value.item())
		return value.tolist()
	if isinstance(value, dict):
		return {key: _json_safe(item) for key, item in value.items()}
	if isinstance(value, (list, tuple)):
		return [_json_safe(item) for item in value]
	if isinstance(value, Path):
		return str(value)
	return value


def _srsa_sampler_config(cfg):
	config = {}
	for field in SRSA_SAMPLER_CFG_KEYS:
		value = cfg.get(field, None)
		if value is not None:
			config[field] = _json_safe(value)
	return config


def _current_srsa_task_params(env):
	params = getattr(env.unwrapped, "current_task_params", None)
	if not isinstance(params, dict) or len(params) == 0:
		return None
	return _json_safe(params)


def _override_value(value):
	if isinstance(value, bool):
		return "true" if value else "false"
	if value is None:
		return "null"
	if isinstance(value, str) and ("," in value or ";" in value):
		return json.dumps(value)
	return str(value)


def _child_overrides(cfg, *, assembly_id: str, output_dir: Path):
	fields = [
		"checkpoint",
		"isaaclab_backend",
		"isaaclab_env_id",
		"isaaclab_task_package",
		"task",
		"srsa_dir",
		"srsa_sparse_reward",
		"srsa_sil",
		"srsa_if_sbc",
		"srsa_task_template_fp",
		"srsa_task_template_id",
		"srsa_param_template_id",
		"srsa_mesh_geometry_fp",
		"srsa_mesh_geometry_task_id",
		"srsa_mesh_plug_diameter_column",
		"srsa_mesh_hole_diameter_column",
		"srsa_mesh_clearance_column",
		"srsa_mesh_clearance_mode",
		"srsa_mesh_clearance_scale",
		"srsa_mesh_depth_column",
		"srsa_mesh_depth_scale",
		"srsa_mesh_reference_radius_column",
		"srsa_mesh_reference_depth_column",
		"num_envs",
		"gpu_id",
		"model_size",
		"horizon",
		"compile",
		"mpc",
		"isaaclab_headless",
		"isaaclab_use_canonical_obs",
		"isaaclab_disable_imitation_reward",
		"srsa_task_family_name",
		"srsa_task_param_obs",
		"srsa_task_param_obs_mode",
		"srsa_newt_obs",
		"srsa_enable_axial_task_param_sampler",
		"srsa_use_runtime_task_vec",
		"srsa_axial_task_type_id",
		"srsa_axial_scale_range",
		"srsa_axial_fixed_plug_scale",
		"srsa_axial_clearance_range",
		"srsa_axial_clearance_ratio_range",
		"srsa_axial_clearance_base",
		"srsa_axial_clearance_anchor_multipliers",
		"srsa_axial_clearance_anchors",
		"srsa_axial_clearance_depth_templates",
		"srsa_axial_clearance_depth_template_multipliers",
		"srsa_axial_clearance_depth_template_weights",
		"srsa_axial_clearance_jitter_ratio",
		"srsa_axial_clearance_anchor_weights",
		"srsa_axial_depth_range",
		"srsa_axial_target_depth_range",
		"srsa_axial_depth_base",
		"srsa_axial_depth_anchor_multipliers",
		"srsa_axial_depth_anchors",
		"srsa_axial_depth_jitter_ratio",
		"srsa_axial_depth_anchor_weights",
		"srsa_axial_init_error_xy_range",
		"srsa_axial_init_error_z_range",
		"srsa_axial_init_error_yaw_range",
		"srsa_axial_visual_noise_xy_range",
		"srsa_axial_visual_noise_z_range",
		"srsa_axial_yaw_requirement",
		"srsa_axial_reference_radius",
		"srsa_axial_reference_depth",
		"srsa_enable_flange_force_sensor",
		"isaaclab_canonical_append_force",
		"isaaclab_canonical_append_task_params",
		"isaaclab_canonical_use_visual_noise",
		"task_conditioning",
		"collect_episodes_per_task",
		"collect_source_assembly_id",
		"collect_match_checkpoint",
		"collect_expected_obs_dim",
		"collect_overwrite",
		"collect_mpc",
		"collect_max_env_steps",
		"enable_wandb",
		"exp_name",
		"run_id",
		"seed",
		"eval_hang_guard_factor",
		"progress_log_interval_sec",
		"eval_task_template_exact",
		"eval_task_template_print",
	]
	overrides = []
	for field in fields:
		value = cfg.get(field, None)
		if value is None:
			continue
		overrides.append(f"{field}={_override_value(value)}")
	overrides.extend([
		f"collect_assembly_ids=[{assembly_id}]",
		f"collect_worker_assembly_id={assembly_id}",
		"collect_spawn_per_assembly=false",
		f"collect_output_dir={output_dir}",
	])
	return overrides


def _collect_via_subprocesses(cfg, assembly_ids: list[str], output_dir: Path):
	script = Path(__file__).resolve()
	entries = []
	for task_id, assembly_id in enumerate(assembly_ids):
		result_fp = output_dir / assembly_id / "manifest_entry.json"
		if result_fp.exists() and cfg.get('collect_overwrite', False):
			result_fp.unlink()
		cmd = [
			sys.executable,
			str(script),
			*_child_overrides(cfg, assembly_id=assembly_id, output_dir=output_dir),
		]
		print(colored(
			f"Launching isolated collection process for assembly_id={assembly_id} "
			f"({task_id + 1}/{len(assembly_ids)}).",
			"cyan",
			attrs=["bold"],
		), flush=True)
		subprocess.run(cmd, cwd=hydra.utils.get_original_cwd(), check=True)
		if not result_fp.exists():
			raise FileNotFoundError(f"Worker did not write manifest entry: {result_fp}")
		entry = _read_json(result_fp)
		entry["task_id"] = int(task_id)
		entries.append(entry)
	return entries


def _collect_for_assembly(base_cfg, assembly_id: str, output_dir: Path, task_id: int):
	cfg = _task_cfg(base_cfg, assembly_id)
	task_dir = output_dir / assembly_id
	output_fp = task_dir / "policy_eval_rollouts.pt"
	metadata_fp = task_dir / "policy_eval_rollouts.pt.json"
	if output_fp.exists() and not cfg.collect_overwrite:
		raise FileExistsError(f"Output already exists: {output_fp}. Use collect_overwrite=true to replace it.")
	set_seed(cfg.seed)
	torch.cuda.set_device(cfg.device_id)
	env = make_env(cfg)
	try:
		expected_obs_dim = cfg.get('collect_expected_obs_dim', None)
		if expected_obs_dim is not None:
			actual_obs_shape = tuple(env.observation_space.shape)
			if len(actual_obs_shape) != 1:
				raise RuntimeError(
					f"Environment observation shape {actual_obs_shape} does not match checkpoint obs_dim={expected_obs_dim}. "
					"Expected a flat state observation."
				)
			actual_obs_dim = int(actual_obs_shape[0])
			if actual_obs_dim < int(expected_obs_dim):
				raise RuntimeError(
					f"Environment observation shape {actual_obs_shape} is smaller than checkpoint obs_dim={expected_obs_dim}."
				)
			if actual_obs_dim > int(expected_obs_dim):
				print(colored(
					f"Adapting environment obs_dim={actual_obs_dim} to checkpoint obs_dim={int(expected_obs_dim)} "
					"by keeping the leading checkpoint dimensions.",
					"yellow",
					attrs=["bold"],
				))
		task_vec = _current_task_vec_6(cfg, env)
		task_params = _current_srsa_task_params(env)
		srsa_sampler = _srsa_sampler_config(cfg)
		agent = _make_agent(cfg)
		target_episodes = int(cfg.get('collect_episodes_per_task', cfg.get('eval_trials', 100)) or 100)
		if target_episodes <= 0:
			raise ValueError(f"collect_episodes_per_task must be positive, got {target_episodes}.")
		use_mpc = cfg.mpc if cfg.get('collect_mpc', None) is None else bool(cfg.collect_mpc)
		rollout_device = torch.device(f"cuda:{cfg.device_id}")
		tasks = torch.zeros(cfg.num_envs, dtype=torch.long, device=rollout_device)
		episode_rows = [[] for _ in range(cfg.num_envs)]
		episode_return = torch.zeros(cfg.num_envs, dtype=torch.float32, device=rollout_device)
		episode_len = torch.zeros(cfg.num_envs, dtype=torch.int64, device=rollout_device)
		columns = _empty_columns()
		episode_returns = []
		episode_successes = []
		completed = 0
		env_steps = 0
		start_time = monotonic()
		last_log = 0.0
		guard_steps = cfg.get('collect_max_env_steps', None)
		if guard_steps is None:
			waves = math.ceil(target_episodes / max(1, int(cfg.num_envs)))
			guard_steps = int((waves + 2) * max(1, int(cfg.episode_length)) * max(1.0, float(cfg.eval_hang_guard_factor)))

		obs, _ = env.reset()
		obs = _adapt_obs_to_checkpoint(obs, expected_obs_dim)
		print(colored(
			f"Collecting {target_episodes} eval episodes on assembly_id={assembly_id} "
			f"with checkpoint={cfg.checkpoint} mpc={use_mpc}.",
			"cyan",
			attrs=["bold"],
		))
		with torch.no_grad():
			while completed < target_episodes:
				t0 = episode_len == 0
				torch.compiler.cudagraph_mark_step_begin()
				model_tasks = _model_task_input(cfg, env, tasks)
				action, _ = agent(
					obs,
					t0=t0,
					step=1 if use_mpc else 0,
					eval_mode=True,
					task=model_tasks,
					mpc=use_mpc,
				)
				prev_obs = obs
				raw_obs, reward, terminated, truncated, info = env.step(action)
				done = terminated | truncated
				obs = _adapt_obs_to_checkpoint(raw_obs, expected_obs_dim)
				next_obs = obs.clone()
				if 'final_observation' in info:
					final_observation = _adapt_obs_to_checkpoint(info['final_observation'], expected_obs_dim)
					next_obs[done] = final_observation

				next_return = episode_return + reward
				step_id = episode_len.to(torch.int32)

				prev_obs_cpu = prev_obs.detach().cpu()
				next_obs_cpu = next_obs.detach().cpu()
				action_cpu = action.detach().cpu()
				reward_cpu = reward.detach().cpu().to(torch.float32)
				terminated_cpu = terminated.detach().cpu().to(torch.bool)
				truncated_cpu = truncated.detach().cpu().to(torch.bool)
				done_cpu = done.detach().cpu().to(torch.bool)
				step_id_cpu = step_id.detach().cpu()
				running_return_cpu = next_return.detach().cpu().to(torch.float32)

				for env_index in range(cfg.num_envs):
					episode_rows[env_index].append({
						"obs": prev_obs_cpu[env_index],
						"next_obs": next_obs_cpu[env_index],
						"action": action_cpu[env_index],
						"reward": reward_cpu[env_index],
						"done": done_cpu[env_index],
						"terminated": terminated_cpu[env_index],
						"truncated": truncated_cpu[env_index],
						"step_id": step_id_cpu[env_index],
						"episode_return_running": running_return_cpu[env_index],
					})

				env_steps += 1
				if 'final_info' in info:
					success_tensor = info['final_info'].get('success', None)
				else:
					success_tensor = None
				for env_index in range(cfg.num_envs):
					if not bool(done[env_index].item()):
						continue
					final_success = 0.0
					if success_tensor is not None:
						value = success_tensor[env_index]
						final_success = float(torch.nan_to_num(value, nan=0.0).detach().item())
					final_return = float(next_return[env_index].detach().item())
					if completed < target_episodes:
						_append_finished_episode(
							columns,
							episode_rows[env_index],
							episode_id=completed,
							task_id=0,
							final_return=final_return,
							final_success=final_success,
						)
						episode_returns.append(final_return)
						episode_successes.append(final_success)
						completed += 1
					episode_rows[env_index] = []

				episode_return = torch.where(done, torch.zeros_like(next_return), next_return)
				episode_len = torch.where(done, torch.zeros_like(episode_len), episode_len + 1)

				now = monotonic()
				if now - last_log >= float(cfg.progress_log_interval_sec) or completed >= target_episodes:
					last_log = now
					elapsed = int(now - start_time)
					print(colored(
						f"collect progress assembly_id={assembly_id} "
						f"episodes={completed}/{target_episodes} env_steps={env_steps} elapsed={elapsed}s",
						"cyan",
						attrs=["bold"],
					), flush=True)
				if env_steps > guard_steps:
					raise RuntimeError(
						f"Collection did not finish within guard_steps={guard_steps} for assembly_id={assembly_id}. "
						f"completed={completed}/{target_episodes}, env_steps={env_steps}."
					)

		dataset = _stack_columns(columns)
		task_dir.mkdir(parents=True, exist_ok=True)
		torch.save(dataset, output_fp)

		summary = summarize_compact_dataset(dataset)
		if getattr(agent.model, "_task_vecs", None) is not None:
			task_vec = agent.model._task_vecs[0].detach().cpu().tolist()
		metadata = {
			"assembly_id": assembly_id,
			"checkpoint": str(Path(cfg.checkpoint).expanduser().resolve()),
			"source_assembly_id": cfg.get('collect_source_assembly_id', None),
			"output": str(output_fp),
			"task_vec_6": task_vec,
			"srsa_params": task_params,
			"srsa_sampler": srsa_sampler,
			"num_envs": int(cfg.num_envs),
			"episodes_requested": target_episodes,
			"episodes_collected": len(episode_returns),
			"episode_return_mean": float(sum(episode_returns) / max(1, len(episode_returns))),
			"episode_success_mean": float(sum(episode_successes) / max(1, len(episode_successes))),
			"mpc": bool(use_mpc),
			"summary": summary,
		}
		_write_json(metadata_fp, metadata)
		print(colored(
			f"Saved assembly_id={assembly_id}: {output_fp} "
			f"success={metadata['episode_success_mean']:.4f} transitions={summary['num_transitions']:,}",
			"green",
			attrs=["bold"],
		))
		entry = {
			"task_id": int(task_id),
			"task_name": f"{cfg.task}-{assembly_id}",
			"assembly_id": assembly_id,
			"source_fp": str(output_fp),
			"action_dim": int(cfg.action_dim),
			"max_episode_steps": int(cfg.episode_length),
			"task_vec_6": task_vec,
			"num_episodes": len(episode_returns),
			"num_transitions": int(summary["num_transitions"]),
			"success_count": int(sum(1 for value in episode_successes if value > 0.5)),
			"failure_count": int(sum(1 for value in episode_successes if value <= 0.5)),
			"success_rate": metadata["episode_success_mean"],
		}
		if task_params:
			entry["srsa_params"] = task_params
		if srsa_sampler:
			entry["srsa_sampler"] = srsa_sampler
		return entry
	finally:
		env.close()


@hydra.main(version_base=None, config_name="config")
def launch(cfg: Config):
	assert torch.cuda.is_available()
	worker_assembly_id = cfg.get('collect_worker_assembly_id', None)
	if worker_assembly_id is not None:
		assembly_ids = [_normalize_assembly_id(worker_assembly_id)]
	else:
		assembly_ids = _resolve_assembly_ids(cfg)
	if cfg.get('collect_source_assembly_id', None) is not None:
		cfg.assembly_id = _normalize_assembly_id(cfg.collect_source_assembly_id)
	if not cfg.checkpoint:
		raise ValueError("`checkpoint` must point to the 00186 trained model checkpoint.")
	checkpoint_fp = Path(hydra.utils.to_absolute_path(str(cfg.checkpoint))).expanduser().resolve()
	if not checkpoint_fp.exists():
		raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_fp}")
	cfg.checkpoint = str(checkpoint_fp)
	_apply_checkpoint_compat(cfg, checkpoint_fp)
	cfg.offline_manifest_fp = None
	cfg = parse_cfg(cfg)
	if cfg.get('collect_source_assembly_id', None) is not None:
		cfg.collect_source_assembly_id = _normalize_assembly_id(cfg.collect_source_assembly_id)
	cfg.enable_wandb = False
	cfg.save_agent = False
	cfg.multiproc = False
	output_dir = _resolve_output_dir(cfg)
	manifest_fp = _resolve_manifest_fp(cfg, output_dir)
	if manifest_fp.exists() and not cfg.collect_overwrite:
		raise FileExistsError(f"Manifest already exists: {manifest_fp}. Use collect_overwrite=true to replace it.")
	output_dir.mkdir(parents=True, exist_ok=True)

	print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)
	print(colored(f'Rollout output dir: {output_dir}', 'yellow', attrs=['bold']))
	print(colored(f'Target assembly ids: {assembly_ids}', 'yellow', attrs=['bold']))

	if worker_assembly_id is not None:
		assembly_id = assembly_ids[0]
		entry = _collect_for_assembly(cfg, assembly_id, output_dir, 0)
		result_fp = output_dir / assembly_id / "manifest_entry.json"
		_write_json(result_fp, entry)
		print(colored(f"Saved worker manifest entry: {result_fp}", "green", attrs=["bold"]))
		return

	if len(assembly_ids) > 1 and cfg.get('collect_spawn_per_assembly', True):
		entries = _collect_via_subprocesses(cfg, assembly_ids, output_dir)
	else:
		entries = []
		for task_id, assembly_id in enumerate(assembly_ids):
			entries.append(_collect_for_assembly(cfg, assembly_id, output_dir, task_id))

	manifest = {
		"source_assembly_id": cfg.get('collect_source_assembly_id', None),
		"checkpoint": str(Path(cfg.checkpoint).expanduser().resolve()),
		"tasks": entries,
	}
	_write_json(manifest_fp, manifest)
	total_episodes = sum(entry["num_episodes"] for entry in entries)
	total_success = sum(entry["success_count"] for entry in entries)
	print(colored(f"Saved offline manifest: {manifest_fp}", "green", attrs=["bold"]))
	print(colored(
		f"Collected {total_episodes} episodes across {len(entries)} tasks; "
		f"success={total_success / max(1, total_episodes):.4f}.",
		"green",
		attrs=["bold"],
	))


if __name__ == '__main__':
	launch()
