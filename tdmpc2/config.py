from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

import json
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
	isaaclab_env_id: str = "Isaac-AutoMate-Assembly-Direct-v0"
	isaaclab_task_name: str = "insertion"
	assembly_id: str = "00004"
	isaaclab_headless: bool = True
	isaaclab_enable_cameras: bool = False
	isaaclab_use_fabric: Optional[bool] = None
	isaaclab_use_canonical_obs: bool = False
	isaaclab_action_dim: int = 6
	isaaclab_max_episode_steps: int = 75
	isaaclab_force_cpu_softdtw: bool = False
	isaaclab_disable_imitation_reward: bool = False

	# evaluation
	checkpoint: Optional[str] = None
	eval_episodes: int = 2
	eval_trials: Optional[int] = None
	eval_task_id: Optional[int] = None
	eval_freq: Optional[int] = None

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
	num_q: int = 5
	simnorm_dim: int = 8
	disable_task_emb: bool = False
	learn_task_emb: Optional[bool] = None

	# logging
	wandb_project: str = "project"
	wandb_entity: str = "entity"
	wandb_silent: bool = False
	enable_wandb: bool = True

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


def split_by_rank(global_list, rank, world_size):
	"""Split a global list into sublists for each rank."""
	return [global_list[i] for i in range(len(global_list)) if i % world_size == rank]


def is_isaaclab_task(cfg):
	"""Return True if the task should be created through Isaac Lab."""
	task = getattr(cfg, 'task', '')
	env_id = getattr(cfg, 'isaaclab_env_id', '')
	return task.startswith('isaaclab-') or task.startswith('Isaac-') or env_id.startswith('Isaac-')


def make_isaaclab_task_info(cfg):
	"""Create synthetic task metadata for Isaac Lab single-task training."""
	embedding = [0.0] * max(int(cfg.task_dim), 0)
	info = {}
	for task in dict.fromkeys(cfg.tasks):
		info[task] = {
			'text_embedding': embedding,
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
	# Convenience
	cfg.work_dir = Path(hydra.utils.get_original_cwd()) / 'logs' / cfg.task / str(cfg.seed) / cfg.exp_name
	cfg.task_title = cfg.task.replace("-", " ").title()
	cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins-1)  # Bin size for discrete regression

	# Model size
	if cfg.get('model_size', None) is not None:
		assert cfg.model_size in MODEL_SIZE.keys(), \
			f'Invalid model size {cfg.model_size}. Must be one of {list(MODEL_SIZE.keys())}'
		for k, v in MODEL_SIZE[cfg.model_size].items():
			cfg[k] = v

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
		cfg.learn_task_emb = cfg.offline_manifest_fp is not None
	if cfg.eval_freq is None:
		cfg.eval_freq = 20 * 500 * cfg.num_envs
	if cfg.save_freq is None:
		cfg.save_freq = 5 * cfg.eval_freq

	# Isaac Lab single-task runs typically start without demonstrations.
	if is_isaaclab_task(cfg) and cfg.data_dir == "/path/to/your/data":
		cfg.use_demos = False

	# Load task embeddings
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
	cfg.episode_lengths = []
	cfg.discounts = []
	cfg.action_dims = []
	for task in cfg.tasks:
		assert task in task_info, f'Task {task} not found in task embeddings.'
		cfg.task_embeddings.append(task_info[task]['text_embedding'])
		cfg.episode_lengths.append(task_info[task]['max_episode_steps'])
		if 'discount_factor' in task_info[task]:
			cfg.discounts.append(task_info[task]['discount_factor'])
		else:
			cfg.discounts.append(discount_heuristic(cfg, task_info[task]['max_episode_steps']))
		cfg.action_dims.append(task_info[task]['action_dim'])

	if cfg.eval_task_id is not None:
		assert 0 <= cfg.eval_task_id < cfg.num_global_tasks, \
			f'eval_task_id={cfg.eval_task_id} is out of range for {cfg.num_global_tasks} tasks.'

	return OmegaConf.to_object(cfg)
