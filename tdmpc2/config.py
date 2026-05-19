from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

import datetime
import json
import math
import re
import hydra
from termcolor import colored
from omegaconf import OmegaConf

from common import MODEL_SIZE, TASK_SET
from common.math import discount_heuristic


@dataclass
class Config:
	"""
	Config for experiments.
	"""

	# environment
	task: str = "soup"
	obs: str = "state"
	episodic: bool = False
	num_envs: int = 10
	env_mode: str = "async"
	tasks_fp: str = "/path/to/your/tasks.json"
	isaaclab_dir: str = "/home/gpuserver/IsaacLab"
	isaaclab_backend: str = "auto"
	isaaclab_task_package: Optional[str] = None
	isaaclab_env_id: str = "Isaac-AutoMate-Assembly-Direct-v0"
	isaaclab_task_name: str = "insertion"
	assembly_id: str = "00004"
	isaaclab_headless: bool = True
	isaaclab_enable_cameras: bool = False
	isaaclab_use_fabric: Optional[bool] = None
	isaaclab_use_canonical_obs: bool = False
	isaaclab_canonical_append_force: bool = False
	isaaclab_canonical_append_task_params: bool = False
	isaaclab_canonical_use_visual_noise: bool = False
	isaaclab_action_dim: int = 6
	isaaclab_max_episode_steps: int = 75
	isaaclab_force_cpu_softdtw: bool = False
	isaaclab_disable_imitation_reward: bool = False
	isaaclab_debug_io: bool = False
	isaaclab_debug_io_steps: int = 3
	isaaclab_debug_io_every: int = 1
	srsa_dir: str = "/home/gpuserver/hx/github/srsa"
	srsa_task_family_name: Optional[str] = None
	srsa_task_family_id: Optional[int] = None
	srsa_plug_diameter: Optional[float] = None
	srsa_hole_diameter: Optional[float] = None
	srsa_clearance: Optional[float] = None
	srsa_clearance_ratio: Optional[float] = None
	srsa_insertion_depth: Optional[float] = None
	srsa_success_pos_tol: Optional[float] = None
	srsa_task_param_obs: bool = False
	srsa_if_sbc: Optional[bool] = None
	srsa_if_logging_eval: bool = False
	srsa_eval_filename: Optional[str] = None
	srsa_num_eval_trials: int = 100
	srsa_vision_noise_xy_std: float = 0.0
	srsa_vision_noise_xy_jitter_std: float = 0.0
	srsa_vision_noise_z_std: float = 0.0
	srsa_vision_noise_z_jitter_std: float = 0.0
	srsa_enable_flange_force_sensor: bool = False
	srsa_flange_force_sensor_body_name: str = "panda_hand"
	srsa_flange_force_sensor_source: str = "held_sensor"
	srsa_flange_force_sensor_obs_frame: str = "socket"
	srsa_flange_force_sensor_obs_scale: float = 50.0
	srsa_flange_force_sensor_force_threshold: float = 1.0

	# evaluation
	checkpoint: Optional[str] = None
	eval_episodes: int = 2
	eval_trials: Optional[int] = None
	eval_task_id: Optional[int] = None
	eval_freq: Optional[int] = None
	skip_initial_eval: bool = False

	# offline training
	offline_only: bool = False
	offline_dataset_fp: Optional[str] = None
	offline_source_fp: Optional[str] = None
	offline_manifest_fp: Optional[str] = None
	offline_export_fp: Optional[str] = None
	offline_export_overwrite: bool = False
	offline_gpu_id: Optional[int] = None
	offline_obs_key: str = "obs"
	offline_next_obs_key: str = "next_obs"
	offline_action_key: str = "action"
	offline_obs_dim: int = 14
	offline_filter_mode: str = "all"
	offline_bc_steps: int = 50_000
	offline_wm_steps: int = 100_000
	offline_log_freq: int = 200
	offline_save_freq: int = 5_000
	offline_eval_freq: int = 0

	# training
	steps: int = 100_000_000
	batch_size: int = 1024
	utd: float = 0.075
	reward_coef: float = 0.1
	value_coef: float = 0.1
	consistency_coef: float = 20.0
	prior_coef: float = 10.0
	rho: float = 0.5
	lr: float = 3e-4
	enc_lr_scale: float = 0.3
	grad_clip_norm: float = 20.0
	tau: float = 0.01
	discount_denom: int = 5
	discount_min: float = 0.95
	discount_max: float = 0.995
	buffer_size: int = 10_000_000
	use_demos: bool = True
	no_demo_buffer: bool = False
	demo_steps: int = 200_000
	lr_schedule: Optional[str] = None
	warmup_steps: int = 5_000
	seeding_coef: int = 5
	progress_log_interval_sec: float = 30.0
	eval_hang_guard_factor: float = 2.0
	exp_name: str = "default"
	finetune: bool = False

	# planning
	mpc: bool = True
	iterations: int = 6
	num_samples: int = 512
	num_elites: int = 64
	num_pi_trajs: int = 24
	horizon: int = 3
	min_std: float = 0.05
	max_std: float = 2.0
	temperature: float = 0.5
	constrained_planning: bool = True
	constraint_start_step: int = 2_000_000
	constraint_final_step: int = 10_000_000
	constraint_min_weight: float = 0.0

	# actor
	log_std_min: float = -10
	log_std_max: float = 2.0
	entropy_coef: float = 1e-4
	use_scaled_entropy: bool = True

	# critic
	num_bins: int = 101
	vmin: float = -10.0
	vmax: float = +10.0

	# architecture
	model_size: Optional[str] = None
	num_channels: int = 32
	num_enc_layers: int = 3
	enc_dim: int = 1024
	mlp_dim: int = 1024
	latent_dim: int = 512
	task_dim: int = 512
	task_conditioning: str = "axial_params"
	axial_task_dim: int = 64
	axial_task_vec_dim: int = 6
	axial_task_type: str = "peg_in_hole"
	axial_task_type_id: Optional[int] = None
	axial_task_vec_6: Any = None
	axial_reference_radius: float = 0.003993
	axial_reference_depth: float = 0.015
	axial_target_insertion_depth: Optional[float] = None
	axial_scale_ratio: Optional[float] = None
	axial_yaw_requirement: bool = False
	num_q: int = 5
	simnorm_dim: int = 8
	disable_task_emb: bool = False
	learn_task_emb: Optional[bool] = None

	# logging
	wandb_project: str = "project"
	wandb_entity: str = "entity"
	wandb_silent: bool = False
	enable_wandb: bool = True
	run_id: Optional[str] = None

	# misc
	multiproc: bool = False
	gpu_id: int = 0
	num_gpus: Optional[int] = None
	rank: int = 0
	world_size: int = 1
	port: Optional[str] = None
	compile: bool = True
	save_video: bool = False
	render_size: int = 224
	save_agent: bool = True
	save_freq: Optional[int] = None
	save_best: bool = True
	save_best_metric: str = "episode_success"
	save_buffer: bool = False
	data_dir: str = "/path/to/your/data"
	seed: int = 1

	# convenience (filled at runtime)
	work_dir: Optional[str] = None
	task_title: Optional[str] = None
	tasks: Any = None
	global_tasks: Any = None
	num_tasks: Optional[int] = None
	num_global_tasks: Optional[int] = None
	task_embeddings: Any = None
	task_vectors: Any = None
	obs_shape: Any = None
	action_dim: Optional[int] = None
	episode_length: Optional[int] = None
	obs_shapes: Any = None
	action_dims: Any = None
	episode_lengths: Any = None
	discounts: Any = None
	bin_size: Optional[float] = None
	child_env: bool = False

	get = lambda self, val, default=None: getattr(self, val, default)


def safe_run_token(value, fallback="na"):
	value = str(value if value is not None else fallback).strip()
	value = re.sub(r"[^0-9a-zA-Z._-]+", "-", value)
	value = value.strip("-_.")
	return value or fallback


def make_run_id(cfg):
	stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
	parts = [stamp]
	if cfg.get('offline_manifest_fp', None):
		if cfg.get('eval_task_id', None) is not None:
			parts.append(f"tid-{int(cfg.eval_task_id)}")
		else:
			num_tasks = int(cfg.get('num_global_tasks', 0) or 0)
			if num_tasks > 0:
				parts.append(f"tids-0-{num_tasks - 1}")
	else:
		assembly_id = cfg.get('assembly_id', None)
		if assembly_id:
			parts.append(f"asm-{safe_run_token(assembly_id)}")
		if cfg.get('eval_task_id', None) is not None:
			parts.append(f"tid-{int(cfg.eval_task_id)}")
	return "_".join(parts)


AXIAL_TASK_CONDITIONING_MODES = {"axial", "axial_params", "param", "param_only"}
ID_TASK_CONDITIONING_MODES = {"id", "id_embedding", "language", "language_embedding"}
NO_TASK_CONDITIONING_MODES = {"none", "disabled"}

AXIAL_TASK_TYPE_IDS = {
	"peg_in_hole": 0,
	"shaft_in_hole": 0,
	"pin_in_hole": 0,
	"axis_into_hole": 0,
	"sleeve_on_shaft": 1,
	"hole_on_shaft": 1,
	"socket_on_pin": 1,
}

SRSA_BASE_PLUG_DIAMETER = 0.007986
SRSA_BASE_HOLE_DIAMETER = 0.008100
SRSA_TASK_FAMILY_PRESETS = {
	"baseline": {"task_family_id": -1, "plug_diameter": SRSA_BASE_PLUG_DIAMETER, "hole_diameter": SRSA_BASE_HOLE_DIAMETER},
	"normal_fit": {"task_family_id": 1, "plug_diameter": SRSA_BASE_PLUG_DIAMETER, "hole_diameter": SRSA_BASE_HOLE_DIAMETER},
	"loose_fit": {"task_family_id": 0, "plug_diameter": SRSA_BASE_PLUG_DIAMETER, "hole_diameter": 0.008386},
	"tight_fit": {"task_family_id": 2, "plug_diameter": SRSA_BASE_PLUG_DIAMETER, "hole_diameter": 0.008036},
}


def task_conditioning_mode(cfg):
	return str(cfg.get('task_conditioning', 'axial_params')).strip().lower()


def uses_axial_task_encoder(cfg):
	return task_conditioning_mode(cfg) in AXIAL_TASK_CONDITIONING_MODES


def uses_id_task_embedding(cfg):
	return task_conditioning_mode(cfg) in ID_TASK_CONDITIONING_MODES


def uses_no_task_conditioning(cfg):
	return task_conditioning_mode(cfg) in NO_TASK_CONDITIONING_MODES


def _get_value(source, key, default=None):
	if source is None:
		return default
	if isinstance(source, dict):
		return source.get(key, default)
	return getattr(source, key, default)


def _first_value(*values):
	for value in values:
		if value is not None:
			return value
	return None


def _optional_float(value):
	if value is None:
		return None
	return float(value)


def _parse_vector(raw, *, expected_dim=6):
	if raw is None:
		return None
	if isinstance(raw, str):
		raw = raw.strip()
		if raw.startswith("[") and raw.endswith("]"):
			raw = raw[1:-1]
		raw = [item for item in raw.replace(";", ",").split(",") if item.strip()]
	values = [float(item) for item in raw]
	if len(values) != expected_dim:
		raise ValueError(f"Expected axial task vector with {expected_dim} values, got {len(values)}: {values}")
	return values


def _resolve_axial_task_type_id(cfg, item=None):
	raw = _first_value(
		_get_value(item, "task_type_id", None),
		_get_value(item, "task_type_id_float", None),
		cfg.get("axial_task_type_id", None),
	)
	if raw is not None:
		return int(float(raw))
	task_type = str(_first_value(_get_value(item, "task_type", None), cfg.get("axial_task_type", "peg_in_hole")))
	if task_type not in AXIAL_TASK_TYPE_IDS:
		raise ValueError(f"Unknown axial task_type={task_type!r}. Expected one of {sorted(AXIAL_TASK_TYPE_IDS)}.")
	return AXIAL_TASK_TYPE_IDS[task_type]


def _resolve_srsa_family_preset(cfg, item=None):
	family_name = _first_value(_get_value(item, "task_family_name", None), cfg.get("srsa_task_family_name", None))
	if family_name in SRSA_TASK_FAMILY_PRESETS:
		return SRSA_TASK_FAMILY_PRESETS[family_name]
	family_id = _first_value(_get_value(item, "task_family_id", None), cfg.get("srsa_task_family_id", None))
	if family_id is None:
		return SRSA_TASK_FAMILY_PRESETS["normal_fit"]
	for preset in SRSA_TASK_FAMILY_PRESETS.values():
		if int(preset["task_family_id"]) == int(family_id):
			return preset
	return SRSA_TASK_FAMILY_PRESETS["normal_fit"]


def make_axial_task_vec(cfg, item=None):
	"""
	Create the param-only axial task vector.

	The vector intentionally excludes initial pose error, visual noise, task_id, and assembly_id.
	"""
	explicit = _first_value(
		_get_value(item, "task_vec_6", None),
		_get_value(item, "axial_task_vec", None),
		_get_value(item, "axial_task_vec_6", None),
		cfg.get("axial_task_vec_6", None),
	)
	parsed = _parse_vector(explicit, expected_dim=int(cfg.get("axial_task_vec_dim", 6)))
	if parsed is not None:
		return parsed

	direct_fields = [
		"task_type_id_float",
		"log_scale",
		"clearance_abs_norm",
		"clearance_rel_norm",
		"depth_abs_norm",
		"yaw_requirement_float",
	]
	if item is not None and all(_get_value(item, key, None) is not None for key in direct_fields):
		return [float(_get_value(item, key)) for key in direct_fields]

	preset = _resolve_srsa_family_preset(cfg, item)
	male_diameter = _optional_float(_first_value(
		_get_value(item, "male_diameter", None),
		_get_value(item, "plug_diameter", None),
		cfg.get("srsa_plug_diameter", None),
		preset.get("plug_diameter"),
		SRSA_BASE_PLUG_DIAMETER,
	))
	female_diameter = _optional_float(_first_value(
		_get_value(item, "female_diameter", None),
		_get_value(item, "hole_diameter", None),
		cfg.get("srsa_hole_diameter", None),
		preset.get("hole_diameter"),
		SRSA_BASE_HOLE_DIAMETER,
	))
	male_radius = max(0.5 * float(male_diameter), 1.0e-8)
	reference_radius = max(float(cfg.get("axial_reference_radius", male_radius)), 1.0e-8)
	reference_depth = max(float(cfg.get("axial_reference_depth", 1.0)), 1.0e-8)

	radial_clearance = _optional_float(_get_value(item, "radial_clearance", None))
	if radial_clearance is None:
		diametral_clearance = _optional_float(_first_value(
			_get_value(item, "diametral_clearance", None),
			_get_value(item, "clearance", None),
			cfg.get("srsa_clearance", None),
		))
		if diametral_clearance is not None:
			radial_clearance = 0.5 * diametral_clearance
	if radial_clearance is None:
		clearance_ratio = _optional_float(_first_value(
			_get_value(item, "clearance_ratio", None),
			cfg.get("srsa_clearance_ratio", None),
		))
		if clearance_ratio is not None:
			radial_clearance = 0.5 * clearance_ratio * float(male_diameter)
	if radial_clearance is None:
		radial_clearance = 0.5 * max(0.0, float(female_diameter) - float(male_diameter))

	target_depth = _optional_float(_first_value(
		_get_value(item, "target_insertion_depth", None),
		_get_value(item, "insertion_depth", None),
		cfg.get("srsa_insertion_depth", None),
		cfg.get("axial_target_insertion_depth", None),
		reference_depth,
	))
	scale_ratio = _optional_float(_first_value(
		_get_value(item, "scale_ratio", None),
		cfg.get("axial_scale_ratio", None),
		male_radius / reference_radius,
	))
	yaw_requirement = _first_value(
		_get_value(item, "yaw_requirement", None),
		_get_value(item, "yaw_requirement_float", None),
		cfg.get("axial_yaw_requirement", False),
	)

	return [
		float(_resolve_axial_task_type_id(cfg, item)),
		float(math.log(max(scale_ratio, 1.0e-8))),
		float(radial_clearance) / reference_radius,
		float(radial_clearance) / male_radius,
		float(target_depth) / reference_depth,
		1.0 if bool(yaw_requirement) else 0.0,
	]


def split_by_rank(global_list, rank, world_size):
	"""Split a global list into sublists for each rank."""
	return [global_list[i] for i in range(len(global_list)) if i % world_size == rank]


def is_isaaclab_task(cfg):
	"""Return True if the task should be created through Isaac Lab."""
	task = getattr(cfg, 'task', '')
	env_id = getattr(cfg, 'isaaclab_env_id', '')
	backend = getattr(cfg, 'isaaclab_backend', 'auto')
	return (
		backend in ('isaaclab', 'srsa') or
		bool(getattr(cfg, 'isaaclab_task_package', None)) or
		task.startswith('isaaclab-') or
		task.startswith('Isaac-') or
		env_id.startswith('Isaac-')
	)


def make_isaaclab_task_info(cfg):
	"""Create synthetic task metadata for Isaac Lab single-task training."""
	embedding = [0.0] * max(int(cfg.task_dim), 0)
	info = {}
	for task in dict.fromkeys(cfg.tasks):
		info[task] = {
			'text_embedding': embedding,
			'task_vec_6': make_axial_task_vec(cfg, {'task_name': task}),
			'max_episode_steps': int(cfg.isaaclab_max_episode_steps),
			'action_dim': int(cfg.isaaclab_action_dim),
		}
	return info


def make_offline_manifest_task_info(cfg, manifest_tasks):
	"""Create task metadata from an offline multitask manifest."""
	info = {}
	for item in manifest_tasks:
		task_name = item.get('task_name', f"{cfg.task}-{item.get('assembly_id', item['task_id'])}")
		embedding = item.get('text_embedding')
		if embedding is None:
			embedding = [0.0] * max(int(cfg.task_dim), 0)
		info[task_name] = {
			'text_embedding': embedding,
			'task_vec_6': make_axial_task_vec(cfg, item),
			'max_episode_steps': int(item.get('max_episode_steps', cfg.isaaclab_max_episode_steps)),
			'action_dim': int(item.get('action_dim', cfg.isaaclab_action_dim)),
		}
		if 'discount_factor' in item:
			info[task_name]['discount_factor'] = float(item['discount_factor'])
	return info


def parse_cfg(cfg):
	"""
	Parses the experiment config dataclass. Mostly for convenience.
	"""
	if cfg.get('isaaclab_backend', 'auto') == 'srsa':
		if cfg.task == 'soup':
			cfg.task = 'isaaclab-srsa-assembly'
		if cfg.isaaclab_env_id == 'Isaac-AutoMate-Assembly-Direct-v0':
			cfg.isaaclab_env_id = 'Assembly-Direct-v0'
		if cfg.isaaclab_task_package is None:
			cfg.isaaclab_task_package = 'SRSA.tasks'

	# Convenience
	cfg.task_title = cfg.task.replace("-", " ").title()
	cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins-1)  # Bin size for discrete regression

	# Model size
	if cfg.get('model_size', None) is not None:
		assert cfg.model_size in MODEL_SIZE.keys(), \
			f'Invalid model size {cfg.model_size}. Must be one of {list(MODEL_SIZE.keys())}'
		for k, v in MODEL_SIZE[cfg.model_size].items():
			cfg[k] = v
	if uses_axial_task_encoder(cfg):
		cfg.task_conditioning = "axial_params"
		cfg.task_dim = int(cfg.axial_task_dim)
	elif uses_no_task_conditioning(cfg):
		cfg.task_conditioning = "none"
		cfg.task_dim = 0
	elif uses_id_task_embedding(cfg):
		cfg.task_conditioning = "id_embedding"
	else:
		raise ValueError(
			f"Unknown task_conditioning={cfg.task_conditioning!r}. "
			"Use axial_params, id_embedding, or none."
		)

	# Set defaults
	manifest_tasks = None
	if cfg.offline_manifest_fp:
		with open(Path(cfg.offline_manifest_fp).expanduser().resolve(), "r", encoding="utf-8") as f:
			manifest = json.load(f)
		manifest_tasks = manifest.get("tasks")
		if not isinstance(manifest_tasks, list) or len(manifest_tasks) == 0:
			raise ValueError(f"Offline manifest at {cfg.offline_manifest_fp} must contain a non-empty 'tasks' list.")
		manifest_tasks = sorted(manifest_tasks, key=lambda item: int(item["task_id"]))
		task_ids = [int(item["task_id"]) for item in manifest_tasks]
		expected = list(range(len(manifest_tasks)))
		assert task_ids == expected, f'Offline manifest task ids must be consecutive starting at 0, got {task_ids}.'
		cfg.tasks = [
			item.get('task_name', f"{cfg.task}-{item.get('assembly_id', item['task_id'])}")
			for item in manifest_tasks
		]
		cfg.num_tasks = len(cfg.tasks)
		cfg.global_tasks = deepcopy(cfg.tasks)
		cfg.num_global_tasks = cfg.num_tasks
	else:
		cfg.tasks = TASK_SET.get(cfg.task, [cfg.task] * cfg.num_envs)
		cfg.num_tasks = len(dict.fromkeys(cfg.tasks))  # Unique tasks
		cfg.global_tasks = deepcopy(cfg.tasks)
		cfg.num_global_tasks = cfg.num_tasks
	if cfg.task == 'soup':
		cfg.num_envs = cfg.num_tasks
		print(colored(f'Number of tasks in soup: {cfg.num_global_tasks}', 'green', attrs=['bold']))
	if cfg.learn_task_emb is None:
		cfg.learn_task_emb = cfg.offline_manifest_fp is not None and uses_id_task_embedding(cfg)
	if cfg.eval_freq is None:
		cfg.eval_freq = 20 * 500 * cfg.num_envs
	if cfg.save_freq is None:
		cfg.save_freq = 5 * cfg.eval_freq

	# Isaac Lab single-task runs typically start without demonstrations.
	if is_isaaclab_task(cfg) and cfg.data_dir == "/path/to/your/data":
		cfg.use_demos = False

	# Load task metadata. The main method uses task_vec_6 -> AxialTaskEncoder;
	# text embeddings remain only for the id/language embedding ablation path.
	if manifest_tasks is not None:
		task_info = make_offline_manifest_task_info(cfg, manifest_tasks)
	elif is_isaaclab_task(cfg) and not Path(cfg.tasks_fp).expanduser().exists():
		print(colored(
			f'No tasks metadata found at {cfg.tasks_fp}; using synthetic Isaac Lab task metadata.',
			'yellow',
			attrs=['bold'],
		))
		task_info = make_isaaclab_task_info(cfg)
	else:
		with open(cfg.tasks_fp, "r") as f:
			task_info = json.load(f)
	cfg.task_embeddings = []
	cfg.task_vectors = []
	cfg.episode_lengths = []
	cfg.discounts = []
	cfg.action_dims = []
	for task in cfg.tasks:
		assert task in task_info, f'Task {task} not found in task embeddings.'
		cfg.task_embeddings.append(task_info[task].get('text_embedding', [0.0] * max(int(cfg.task_dim), 0)))
		if uses_axial_task_encoder(cfg):
			cfg.task_vectors.append(task_info[task].get('task_vec_6', make_axial_task_vec(cfg, task_info[task])))
		cfg.episode_lengths.append(task_info[task]['max_episode_steps'])
		if 'discount_factor' in task_info[task]:
			cfg.discounts.append(task_info[task]['discount_factor'])
		else:
			cfg.discounts.append(discount_heuristic(cfg, task_info[task]['max_episode_steps']))
		cfg.action_dims.append(task_info[task]['action_dim'])

	if cfg.eval_task_id is not None:
		assert 0 <= cfg.eval_task_id < cfg.num_global_tasks, \
			f'eval_task_id={cfg.eval_task_id} is out of range for {cfg.num_global_tasks} tasks.'
	if cfg.run_id is None:
		cfg.run_id = make_run_id(cfg)
	else:
		cfg.run_id = safe_run_token(cfg.run_id)
	cfg.work_dir = Path(hydra.utils.get_original_cwd()) / 'logs' / cfg.task / str(cfg.seed) / cfg.exp_name / cfg.run_id

	return OmegaConf.to_object(cfg)
