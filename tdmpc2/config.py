from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any

import csv
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
	isaaclab_canonical_zero_force: bool = False
	isaaclab_canonical_force_dim: int = 3
	isaaclab_canonical_append_task_params: bool = False
	isaaclab_canonical_use_visual_noise: bool = False
	isaaclab_action_dim: int = 6
	isaaclab_gpu_collision_stack_size: Optional[int] = None
	srsa_position_control_only: bool = True
	srsa_policy_action_dim: int = 3
	srsa_env_action_dim: int = 6
	isaaclab_max_episode_steps: int = 75
	isaaclab_force_cpu_softdtw: bool = False
	isaaclab_disable_imitation_reward: bool = False
	isaaclab_debug_io: bool = False
	isaaclab_debug_io_steps: int = 3
	isaaclab_debug_io_every: int = 1
	srsa_dir: str = "/home/gpuserver/hx/github/srsa"
	srsa_sparse_reward: bool = False
	srsa_sil: bool = False
	srsa_task_template_fp: Optional[str] = None
	srsa_task_template_id: Optional[int] = None
	srsa_task_template_applied_id: Optional[int] = None
	srsa_param_template_id: Optional[int] = None
	srsa_mesh_geometry_fp: Optional[str] = None
	srsa_mesh_geometry_task_id: Optional[str] = None
	srsa_mesh_plug_diameter_column: Optional[str] = None
	srsa_mesh_hole_diameter_column: Optional[str] = None
	srsa_mesh_clearance_column: Optional[str] = None
	srsa_mesh_clearance_mode: Optional[str] = None
	srsa_mesh_clearance_scale: float = 1.0
	srsa_mesh_depth_column: Optional[str] = None
	srsa_mesh_depth_scale: float = 1.0
	srsa_mesh_reference_radius_column: Optional[str] = None
	srsa_mesh_reference_depth_column: Optional[str] = None
	srsa_task_family_name: Optional[str] = None
	srsa_task_family_id: Optional[int] = None
	srsa_plug_diameter: Optional[float] = None
	srsa_hole_diameter: Optional[float] = None
	srsa_clearance: Optional[float] = None
	srsa_clearance_ratio: Optional[float] = None
	srsa_insertion_depth: Optional[float] = None
	srsa_success_pos_tol: Optional[float] = None
	srsa_task_param_obs: bool = False
	srsa_task_param_obs_mode: str = "task_vec"
	srsa_newt_obs: bool = False
	srsa_enable_axial_task_param_sampler: bool = True
	srsa_use_runtime_task_vec: bool = True
	srsa_axial_task_type_id: Optional[int] = None
	srsa_axial_scale_range: Any = None
	srsa_axial_fixed_plug_scale: Optional[bool] = None
	srsa_axial_clearance_range: Any = None
	srsa_axial_clearance_ratio_range: Any = None
	srsa_axial_clearance_base: Optional[float] = None
	srsa_axial_clearance_anchor_multipliers: Any = None
	srsa_axial_clearance_anchors: Any = None
	srsa_axial_clearance_jitter_ratio: Optional[float] = None
	srsa_axial_clearance_anchor_weights: Any = None
	srsa_axial_depth_range: Any = None
	srsa_axial_target_depth_range: Any = None
	srsa_axial_depth_base: Optional[float] = None
	srsa_axial_depth_anchor_multipliers: Any = None
	srsa_axial_depth_anchors: Any = None
	srsa_axial_depth_jitter_ratio: Optional[float] = None
	srsa_axial_depth_anchor_weights: Any = None
	srsa_axial_clearance_depth_template_multipliers: Any = None
	srsa_axial_clearance_depth_templates: Any = None
	srsa_axial_clearance_depth_template_weights: Any = None
	srsa_axial_init_error_xy_range: Any = None
	srsa_axial_init_error_z_range: Any = None
	srsa_axial_init_error_yaw_range: Any = None
	srsa_axial_visual_noise_xy_range: Any = None
	srsa_axial_visual_noise_z_range: Any = None
	srsa_axial_yaw_requirement: Optional[bool] = None
	srsa_axial_reference_radius: Optional[float] = None
	srsa_axial_reference_depth: Optional[float] = None
	srsa_axial_reference_anchor_assembly_id: Optional[str] = None
	srsa_axial_reference_anchor_task_type_id: Optional[int] = None
	srsa_axial_recompute_manifest_task_vecs: bool = False
	srsa_if_sbc: Optional[bool] = None
	srsa_if_logging_eval: bool = False
	srsa_eval_filename: Optional[str] = None
	srsa_num_eval_trials: int = 100
	srsa_align_direct_reward_success: bool = False
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
	eval_success_metric: str = "strict"
	srsa_eval_success_metric: str = "strict"
	strict_depth_fraction: float = 0.90
	strict_success_steps: int = 10
	strict_lateral_tol_min: float = 0.0005
	strict_lateral_tol_max: Optional[float] = 0.0020
	strict_keypoint_tol_min: float = 0.0010
	strict_keypoint_tol_max: Optional[float] = 0.0030
	strict_angle_tol_deg: float = 3.0
	relaxed_depth_fraction: float = 0.85
	relaxed_success_steps: int = 3
	relaxed_lateral_tol_scale: float = 2.0
	relaxed_lateral_tol_min: float = 0.0010
	relaxed_lateral_tol_max: Optional[float] = 0.0030
	relaxed_keypoint_tol_scale: float = 2.0
	relaxed_keypoint_tol_min: float = 0.0010
	relaxed_keypoint_tol_max: Optional[float] = 0.0030
	relaxed_angle_tol_deg: float = 5.0
	relaxed_success_require_official: bool = False
	relaxed_success_require_no_jam: bool = True
	srsa_process_success_depth_ratio: float = 0.90
	srsa_process_success_lateral_tol_scale: float = 2.0
	srsa_process_success_lateral_tol_min: float = 0.0005
	srsa_process_success_lateral_tol_max: Optional[float] = 0.0020
	srsa_process_success_orientation_tol_rad: float = 0.05235987755982989
	srsa_process_success_yaw_tol_rad: float = 0.05235987755982989
	srsa_process_success_keypoint_tol_scale: float = 2.0
	srsa_process_success_keypoint_tol_min: float = 0.0010
	srsa_process_success_keypoint_tol_max: Optional[float] = 0.0030
	srsa_process_success_stable_steps: int = 10
	srsa_process_success_require_official: bool = False
	srsa_process_success_require_no_jam: bool = True

	# evaluation
	checkpoint: Optional[str] = None
	eval_assembly_ids: Any = "[01125,00004,00014,00062,00271]"
	eval_episodes: int = 2
	eval_trials: Optional[int] = None
	eval_task_id: Optional[int] = None
	eval_freq: Optional[int] = None
	skip_initial_eval: bool = False
	eval_task_template_exact: bool = True
	eval_task_template_print: bool = True
	eval_terminate_on_success: bool = False
	eval_terminate_success_key: str = "terminal_process_success"
	eval_terminate_min_step: int = 0
	eval_mode: str = "sim"
	eval_zmq_enabled: bool = False
	eval_zmq_server: str = "tcp://localhost:5555"
	eval_zmq_env_index: int = 0
	eval_zmq_rate: float = 0.0
	eval_zmq_action_scale: float = 1.0
	eval_zmq_max_trans_delta: Optional[float] = None
	eval_zmq_max_rot_delta: Optional[float] = None
	eval_zmq_warmup_steps: int = 0
	eval_zmq_send_timeout_ms: int = 0
	eval_zmq_send_done: bool = True
	eval_real_mode: str = "stream"
	eval_real_obs_server: str = "tcp://localhost:5556"
	eval_real_obs_socket_type: str = "sub"
	eval_real_obs_connect: bool = True
	eval_real_obs_timeout_ms: int = 1000
	eval_real_obs_key: str = "obs"
	eval_real_task_vec_key: str = "task_vec_6"
	eval_real_done_key: str = "done"
	eval_real_use_msg_task_vec: bool = True
	eval_real_steps: Optional[int] = None
	eval_real_state_format: str = "auto"
	eval_real_socket_pos: Any = None
	eval_real_socket_quat_wxyz: Any = None
	eval_real_socket_quat_xyzw: Any = None
	eval_real_socket_euler_xyz: Any = None
	eval_real_socket_euler_degrees: bool = False
	eval_real_tcp_offset_ee: Any = None
	eval_real_use_initial_pose_as_socket: bool = False
	eval_real_gripper_width_default: float = 0.0
	eval_real_force_scale: float = 50.0
	eval_real_zero_missing_force: bool = True
	eval_real_debug_log: bool = True
	eval_real_debug_log_fp: Optional[str] = None
	eval_zmq_action_frame: str = "socket"
	eval_zmq_command_frame: Optional[str] = None
	eval_zmq_action_order: str = "dx,dy,dz,droll,dpitch,dyaw"
	eval_trace_enabled: bool = False
	eval_trace_steps: int = 16
	eval_trace_env_index: Optional[int] = None
	eval_trace_fp: Optional[str] = None
	eval_trace_include_next_obs: bool = True
	eval_trace_include_action_info: bool = True
	eval_trace_include_raw_msg: bool = False

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
	offline_wm_filter_mode: str = "all"
	offline_bc_filter_mode: str = "success_or_high_depth"
	offline_rl_filter_mode: str = "all"
	task_balanced_sampling: bool = True
	offline_high_depth_threshold: float = 0.75
	offline_high_depth_lateral_tol_m: float = 0.0020
	offline_bc_steps: int = 50_000
	offline_wm_steps: int = 100_000
	offline_rl_steps: int = 0
	offline_log_freq: int = 200
	offline_save_freq: int = 5_000
	offline_eval_freq: int = 0

	# scheme A: single shared model continual multi-task fine-tuning
	multitask_continuation_enabled: bool = False
	multitask_replay_manifest_fp: Optional[str] = None
	multitask_auto_collect_replay: bool = False
	multitask_task_ids: Any = field(default_factory=lambda: ["01125", "00004", "00014", "00062", "00271"])
	multitask_anchor_task_id: str = "01125"
	multitask_curriculum_mode: str = "progressive"
	multitask_stage_steps: int = 200_000
	multitask_total_steps: Optional[int] = None
	multitask_sampling_mode: str = "balanced"
	multitask_task_sampling_weights: Any = None
	multitask_anchor_min_ratio: float = 0.2
	multitask_new_task_min_ratio: float = 0.2
	multitask_hard_case_ratio: float = 0.2
	multitask_eval_task_ids: Any = field(default_factory=lambda: ["01125", "00004", "00014", "00062", "00271"])
	multitask_eval_interval: int = 50_000
	multitask_save_per_task_metrics: bool = True
	multitask_forgetting_metric_enabled: bool = True
	multitask_reference_checkpoint_path: str = ""
	multitask_prox_reg_enabled: bool = False
	multitask_prox_reg_coef: float = 1.0e-4
	multitask_distill_old_policy_enabled: bool = False
	multitask_distill_coef: float = 1.0e-3
	multitask_no_forgetting_max_forgetting: float = 0.05

	# policy rollout collection for offline RL
	collect_assembly_ids: Any = "[00004,00014,00062,00271]"
	collect_source_assembly_id: Optional[str] = "01125"
	collect_exclude_source_assembly: bool = True
	include_source_anchor_rollouts: bool = False
	collect_episodes_per_task: int = 300
	collect_weak_task_episodes: int = 600
	collect_screening_fp: Optional[str] = None
	collect_weak_success_threshold: float = 0.10
	collect_defer_success_threshold: float = 0.03
	collect_defer_depth_threshold: float = 0.35
	collect_skip_deferred_tasks: bool = True
	collect_output_dir: Optional[str] = None
	collect_manifest_fp: Optional[str] = None
	collect_overwrite: bool = False
	collect_mpc: Optional[bool] = None
	collect_max_env_steps: Optional[int] = None
	collect_match_checkpoint: bool = True
	collect_expected_obs_dim: Optional[int] = None
	collect_spawn_per_assembly: bool = True
	collect_worker_assembly_id: Optional[str] = None

	# real-robot human-in-the-loop rollout collection
	hil_collect_episodes: int = 10
	hil_collect_output_fp: Optional[str] = None
	hil_collect_manifest_fp: Optional[str] = None
	hil_collect_overwrite: bool = False
	hil_collect_mpc: Optional[bool] = None
	hil_collect_max_steps: Optional[int] = None
	hil_collect_reward_key: str = "reward"
	hil_collect_success_key: str = "success"
	hil_collect_action_keys: str = "executed_action,actual_action,applied_action,intervene_action"
	hil_collect_intervened_key: str = "intervened"
	hil_collect_require_actual_action: bool = False

	# batch evaluation
	batch_eval_assembly_ids: Any = None
	batch_eval_episodes_per_task: int = 100
	batch_eval_output_dir: Optional[str] = None
	batch_eval_summary_fp: Optional[str] = None
	batch_eval_spawn_per_assembly: bool = True
	batch_eval_worker_assembly_id: Optional[str] = None
	batch_eval_worker_task_id: Optional[int] = None
	batch_eval_worker_eval_index: Optional[int] = None
	batch_eval_overwrite: bool = False
	batch_eval_mpc: Optional[bool] = None
	batch_eval_max_env_steps: Optional[int] = None

	# zero-shot task screening
	screen_assembly_ids: Any = "[00004,00014,00062,00271]"
	screen_trials: int = 200
	screen_output_csv: Optional[str] = "data/task_screening_01125_axial_hole.csv"
	screen_output_json: Optional[str] = None
	screen_high_depth_threshold: float = 0.75
	screen_low_depth_threshold: float = 0.35
	screen_hard_min_success: float = 0.15
	screen_hard_max_success: float = 0.45
	screen_extra_min_success: float = 0.05
	screen_extra_max_success: float = 0.15
	screen_easy_success: float = 0.70
	screen_defer_success: float = 0.03

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
	contact_history_enabled: bool = False
	contact_history_len: int = 4
	contact_force_dim: int = 6
	contact_action_dim: int = 6
	contact_ee_delta_dim: int = 6
	contact_history_use_ee_delta: bool = True
	contact_context_dim: int = 64
	contact_history_hidden_dim: int = 128
	contact_history_layers: int = 2
	latent_residual_enabled: bool = False
	latent_residual_freeze_base_wm: bool = True
	latent_residual_hidden_dim: int = 256
	latent_residual_num_layers: int = 2
	latent_residual_alpha: float = 0.1
	latent_residual_alpha_warmup_steps: int = 0
	latent_residual_clip: float = 0.1
	latent_residual_gate_mode: str = "always"
	latent_residual_contact_force_threshold: float = 0.0
	latent_residual_use_force: bool = True
	latent_residual_use_task_vec: bool = True
	latent_residual_use_z_next_base: bool = True
	latent_residual_train_only_contact_phase: bool = False
	latent_residual_reg_coef: float = 1.0e-4
	latent_residual_depth_loss_coef: float = 0.0
	latent_residual_radial_loss_coef: float = 0.0
	latent_residual_jam_loss_coef: float = 0.0
	latent_residual_force_loss_coef: float = 0.0
	latent_residual_force_history_len: Optional[int] = None
	latent_residual_force_dim: Optional[int] = None
	latent_residual_contact_feature_dim: int = 0
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
	template_fp = cfg.get('offline_manifest_fp', None) or cfg.get('srsa_task_template_fp', None)
	if template_fp:
		assembly_id = cfg.get('assembly_id', None)
		if assembly_id:
			parts.append(f"asm-{safe_run_token(assembly_id)}")
		task_id = cfg.get('eval_task_id', None)
		if task_id is None:
			task_id = cfg.get('srsa_task_template_id', None)
		if task_id is not None:
			parts.append(f"tid-{int(task_id)}")
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
SIM_EVAL_MODES = {"sim", "simulation", "isaac", "isaaclab"}
REAL_EVAL_MODES = {"real", "robot", "hardware"}

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


def eval_mode(cfg):
	return str(cfg.get('eval_mode', 'sim')).strip().lower()


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
		_get_value(item, "task_vec", None),
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
	reference_radius = max(float(_first_value(
		_get_value(item, "reference_radius", None),
		cfg.get("srsa_axial_reference_radius", None),
		cfg.get("axial_reference_radius", None),
		male_radius,
	)), 1.0e-8)
	reference_depth = max(float(_first_value(
		_get_value(item, "reference_depth", None),
		cfg.get("srsa_axial_reference_depth", None),
		cfg.get("axial_reference_depth", None),
		1.0,
	)), 1.0e-8)

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


def is_srsa_task(cfg):
	"""Return True if the Isaac Lab task should use the SRSA adapter."""
	task = getattr(cfg, 'task', '')
	env_id = getattr(cfg, 'isaaclab_env_id', '')
	return (
		getattr(cfg, 'isaaclab_backend', 'auto') == 'srsa' or
		getattr(cfg, 'isaaclab_task_package', None) == 'SRSA.tasks' or
		task.startswith('isaaclab-srsa') or
		env_id.startswith('Assembly-') or
		env_id.startswith('Disassembly-')
	)


def srsa_policy_action_dim(cfg, fallback=None):
	if is_srsa_task(cfg) and bool(getattr(cfg, 'srsa_position_control_only', True)):
		return int(getattr(cfg, 'srsa_policy_action_dim', 3))
	if fallback is not None:
		return int(fallback)
	return int(getattr(cfg, 'isaaclab_action_dim', 6))


def make_isaaclab_task_info(cfg):
	"""Create synthetic task metadata for Isaac Lab single-task training."""
	embedding = [0.0] * max(int(cfg.task_dim), 0)
	info = {}
	for task in dict.fromkeys(cfg.tasks):
		info[task] = {
			'text_embedding': embedding,
			'task_vec_6': make_axial_task_vec(cfg, {'task_name': task}),
			'max_episode_steps': int(cfg.isaaclab_max_episode_steps),
			'action_dim': srsa_policy_action_dim(cfg),
		}
	return info


def _strip_explicit_task_vec_fields(item):
	item = dict(item)
	for key in ("task_vec", "task_vec_6", "axial_task_vec", "axial_task_vec_6", "task_param_vec"):
		item.pop(key, None)
	return item


def _manifest_param_template_id(cfg, item):
	for key in ("srsa_param_template_id", "srsa_task_template_id", "param_template_id", "template_id"):
		value = _get_value(item, key, None)
		if value is not None:
			return int(value)
	for key in ("srsa_param_template_id", "srsa_task_template_id", "eval_task_id"):
		value = cfg.get(key, None)
		if value is not None:
			return int(value)
	return None


def _should_recompute_manifest_task_vec(cfg):
	return (
		bool(cfg.get("srsa_axial_recompute_manifest_task_vecs", False)) or
		cfg.get("srsa_axial_reference_anchor_assembly_id", None) is not None
	)


def _manifest_task_template_entry(cfg, item):
	assembly_id = _normalize_srsa_assembly_id(_get_value(item, "assembly_id", None))
	template_fp = cfg.get("srsa_task_template_fp", None)
	template_id = _manifest_param_template_id(cfg, item)
	if assembly_id is None or template_fp is None or template_id is None:
		return None

	template_cfg = deepcopy(cfg)
	_set_cfg_value(template_cfg, "assembly_id", assembly_id)
	_set_cfg_value(template_cfg, "offline_manifest_fp", None)
	templates = _load_srsa_task_template_tasks(template_cfg, template_fp)
	for template in templates:
		if int(template["task_id"]) == int(template_id):
			template = _merge_template_params(template)
			template["task_id"] = int(_get_value(item, "task_id", template.get("task_id", 0)))
			template["assembly_id"] = assembly_id
			if _get_value(item, "task_name", None) is not None:
				template["task_name"] = _get_value(item, "task_name")
			if _get_value(item, "eval_index", None) is not None:
				template["eval_index"] = _get_value(item, "eval_index")
			return template
	raise ValueError(
		f"Task template id {template_id} for assembly_id={assembly_id} was not found in {template_fp}."
	)


def _manifest_task_vec_from_template(cfg, item):
	template = _manifest_task_template_entry(cfg, item)
	if template is None:
		return None
	return make_axial_task_vec(cfg, template)


def _manifest_task_vec(cfg, item):
	if not _should_recompute_manifest_task_vec(cfg):
		return make_axial_task_vec(cfg, item)

	task_vec = _manifest_task_vec_from_template(cfg, item)
	if task_vec is not None:
		return task_vec

	item_without_vec = _strip_explicit_task_vec_fields(item)
	if _entry_has_task_params(item_without_vec):
		return make_axial_task_vec(cfg, item_without_vec)

	raise ValueError(
		"Cannot recompute axial task_vec_6 for offline manifest entry. "
		"Provide structured SRSA params in the manifest, or provide assembly_id with "
		"srsa_task_template_fp and srsa_param_template_id/srsa_task_template_id. "
		f"Entry keys: {sorted(item.keys())}"
	)


def make_offline_manifest_task_info(cfg, manifest_tasks):
	"""Create task metadata from an offline multitask manifest."""
	info = {}
	for raw_item in manifest_tasks:
		item = _merge_template_params(raw_item)
		task_name = item.get('task_name', f"{cfg.task}-{item.get('assembly_id', item['task_id'])}")
		embedding = item.get('text_embedding')
		if embedding is None:
			embedding = [0.0] * max(int(cfg.task_dim), 0)
		info[task_name] = {
			'text_embedding': embedding,
			'task_vec_6': _manifest_task_vec(cfg, item),
			'max_episode_steps': int(item.get('max_episode_steps', cfg.isaaclab_max_episode_steps)),
			'action_dim': srsa_policy_action_dim(cfg, item.get('action_dim', cfg.isaaclab_action_dim)),
		}
		if 'discount_factor' in item:
			info[task_name]['discount_factor'] = float(item['discount_factor'])
	return info


def _load_manifest_tasks(manifest_fp):
	with open(Path(manifest_fp).expanduser().resolve(), "r", encoding="utf-8") as f:
		manifest = json.load(f)
	manifest_tasks = manifest.get("tasks") if isinstance(manifest, dict) else manifest
	if not isinstance(manifest_tasks, list) or len(manifest_tasks) == 0:
		raise ValueError(f"Offline manifest at {manifest_fp} must contain a non-empty 'tasks' list.")
	return sorted(manifest_tasks, key=lambda item: int(item["task_id"]))


def _load_template_tasks(cfg):
	template_fp = _cfg_value(cfg, "offline_manifest_fp", None)
	if template_fp:
		return _load_manifest_tasks(template_fp)
	template_fp = _cfg_value(cfg, "srsa_task_template_fp", None)
	if not template_fp:
		return None
	return _load_srsa_task_template_tasks(cfg, template_fp)


def _set_cfg_value(cfg, key, value):
	if hasattr(cfg, key):
		setattr(cfg, key, value)
	else:
		cfg[key] = value


def _cfg_value(cfg, key, default=None):
	if hasattr(cfg, 'get'):
		return cfg.get(key, default)
	return getattr(cfg, key, default)


def _absolute_cfg_path(path_value):
	if path_value is None or str(path_value).strip() == "":
		return None
	path_text = str(path_value).strip().strip("'\"")
	try:
		return Path(hydra.utils.to_absolute_path(path_text)).expanduser().resolve()
	except Exception:
		path = Path(path_text).expanduser()
		return (Path.cwd() / path).resolve() if not path.is_absolute() else path.resolve()


def _same_cfg_path(left, right) -> bool:
	return _absolute_cfg_path(left) == _absolute_cfg_path(right)


def _looks_like_model_checkpoint_path(path: Path) -> bool:
	parts = {part.lower() for part in path.parts}
	name = path.name.lower()
	return path.suffix.lower() == ".pt" or "models" in parts or name in {"best.pt", "latest.pt", "final.pt"}


def _sync_multitask_replay_manifest_alias(cfg):
	alias_fp = _cfg_value(cfg, "multitask_replay_manifest_fp", None)
	offline_fp = _cfg_value(cfg, "offline_manifest_fp", None)
	if alias_fp and offline_fp and not _same_cfg_path(alias_fp, offline_fp):
		raise ValueError(
			"Both multitask_replay_manifest_fp and offline_manifest_fp were provided, but they point to "
			"different files. Use multitask_replay_manifest_fp for Scheme A replay manifests and reserve "
			"checkpoint for the warm-start model .pt."
		)
	manifest_fp = alias_fp or offline_fp
	if manifest_fp:
		manifest_path = _absolute_cfg_path(manifest_fp)
		_set_cfg_value(cfg, "multitask_replay_manifest_fp", str(manifest_path))
		_set_cfg_value(cfg, "offline_manifest_fp", str(manifest_path))
	return _cfg_value(cfg, "multitask_replay_manifest_fp", None)


def _validate_multitask_continuation_paths(cfg):
	if not _cfg_value(cfg, "multitask_continuation_enabled", False):
		if _cfg_value(cfg, "multitask_replay_manifest_fp", None):
			_sync_multitask_replay_manifest_alias(cfg)
		return
	manifest_fp = _sync_multitask_replay_manifest_alias(cfg)

	checkpoint_fp = _cfg_value(cfg, "checkpoint", None)
	if not checkpoint_fp:
		raise ValueError(
			"multitask_continuation_enabled=true requires checkpoint=<warm-start model .pt>. "
			"checkpoint is only the initialization model, for example /.../models/best.pt."
		)
	checkpoint_path = _absolute_cfg_path(checkpoint_fp)
	if checkpoint_path.suffix.lower() != ".pt":
		raise ValueError(
			f"checkpoint={checkpoint_path} is invalid: checkpoint must point to an initialization model .pt file, "
			"for example /.../models/best.pt."
		)
	if not checkpoint_path.is_file():
		raise FileNotFoundError(f"checkpoint model .pt file not found: {checkpoint_path}")
	_set_cfg_value(cfg, "checkpoint", str(checkpoint_path))
	if not _cfg_value(cfg, "multitask_reference_checkpoint_path", None):
		_set_cfg_value(cfg, "multitask_reference_checkpoint_path", str(checkpoint_path))

	if not manifest_fp:
		if bool(_cfg_value(cfg, "multitask_auto_collect_replay", False)):
			return
		raise ValueError(
			"Offline multitask continuation requires a replay manifest. To train without an existing manifest, "
			"enable online rollout collection or run the replay collection script first."
		)
	manifest_path = _absolute_cfg_path(manifest_fp)
	if _looks_like_model_checkpoint_path(manifest_path) or manifest_path.suffix.lower() != ".json":
		raise ValueError(
			f"multitask_replay_manifest_fp={manifest_path} is invalid: manifest should point to a replay/data "
			"manifest json, not a model checkpoint. Use checkpoint=/.../models/best.pt for the warm-start model "
			"and multitask_replay_manifest_fp=/.../data/offline_manifest_family.json for replay data."
		)
	if not manifest_path.is_file():
		raise FileNotFoundError(f"multitask replay manifest .json file not found: {manifest_path}")
	_set_cfg_value(cfg, "multitask_replay_manifest_fp", str(manifest_path))
	_set_cfg_value(cfg, "offline_manifest_fp", str(manifest_path))


def _normalize_srsa_assembly_id(value):
	if value is None:
		return None
	text = str(value).strip()
	if len(text) == 0:
		return None
	return text.zfill(5) if text.isdigit() else text


def _resolve_existing_path(path_value, *, cfg=None, base_dir=None):
	path = Path(str(path_value)).expanduser()
	if path.is_absolute():
		return path
	candidates = []
	if base_dir is not None:
		candidates.append(Path(base_dir) / path)
	try:
		candidates.append(Path(hydra.utils.get_original_cwd()) / path)
	except Exception:
		candidates.append(Path.cwd() / path)
	srsa_dir = _cfg_value(cfg, "srsa_dir", None) if cfg is not None else None
	if srsa_dir:
		candidates.append(Path(str(srsa_dir)).expanduser() / path)
	candidates.append(path)
	for candidate in candidates:
		candidate = candidate.expanduser()
		if candidate.exists():
			return candidate.resolve()
	return candidates[0].expanduser().resolve()


def _load_json_document(fp, *, cfg=None):
	path = _resolve_existing_path(fp, cfg=cfg)
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f), path


def _mesh_option(cfg, mesh_cfg, cfg_suffix, json_key, default):
	value = _cfg_value(cfg, f"srsa_mesh_{cfg_suffix}", None)
	if value is not None:
		return value
	if isinstance(mesh_cfg, dict) and mesh_cfg.get(json_key, None) is not None:
		return mesh_cfg[json_key]
	return default


def _read_mesh_geometry_row(cfg, mesh_cfg, template_path):
	assembly_id = _normalize_srsa_assembly_id(
		_first_value(
			_cfg_value(cfg, "srsa_mesh_geometry_task_id", None),
			_cfg_value(cfg, "assembly_id", None),
			mesh_cfg.get("default_assembly_id", None) if isinstance(mesh_cfg, dict) else None,
		)
	)
	if assembly_id is None:
		raise ValueError("SRSA mesh geometry templates require assembly_id or srsa_mesh_geometry_task_id.")
	mesh_fp = _first_value(
		_cfg_value(cfg, "srsa_mesh_geometry_fp", None),
		mesh_cfg.get("fp", None) if isinstance(mesh_cfg, dict) else None,
		mesh_cfg.get("csv", None) if isinstance(mesh_cfg, dict) else None,
	)
	if mesh_fp is None:
		raise ValueError("SRSA mesh geometry templates require srsa_mesh_geometry_fp or mesh_geometry.fp.")
	mesh_path = _resolve_existing_path(mesh_fp, cfg=cfg, base_dir=template_path.parent)
	if not mesh_path.exists():
		raise FileNotFoundError(
			f"SRSA mesh geometry CSV not found: {mesh_path}. "
			"Set srsa_mesh_geometry_fp=/absolute/path/to/srsa_mesh_geometry_params.csv, "
			"or copy the SRSA-exported CSV to the template mesh_geometry.fp path. "
			"Use assembly_id for the SRSA mesh/task id, and srsa_param_template_id or "
			"srsa_task_template_id for the clearance/depth parameter template id."
		)
	with open(mesh_path, "r", encoding="utf-8", newline="") as f:
		for row in csv.DictReader(f):
			if _normalize_srsa_assembly_id(row.get("assembly_id")) == assembly_id:
				return row, assembly_id, mesh_path
	raise ValueError(f"assembly_id={assembly_id} was not found in mesh geometry CSV: {mesh_path}")


def _float_row_value(row, key, *, default=None):
	if key is None:
		if default is None:
			raise ValueError("Missing mesh geometry column name.")
		return float(default)
	if key not in row:
		raise ValueError(f"Mesh geometry CSV does not contain column {key!r}.")
	raw = row[key]
	if raw is None or str(raw).strip() == "":
		if default is None:
			raise ValueError(f"Mesh geometry column {key!r} is empty.")
		return float(default)
	return float(raw)


def _diametral_clearance_from_mesh(value, column, mode):
	mode = str(mode or "auto").strip().lower()
	if mode not in {"auto", "diametral", "radial"}:
		raise ValueError("srsa_mesh_clearance_mode must be auto, diametral, or radial.")
	if mode == "radial":
		return 2.0 * float(value)
	if mode == "diametral":
		return float(value)
	column_text = str(column or "").lower()
	if "radial" in column_text or "surface_dist" in column_text:
		return 2.0 * float(value)
	return float(value)


def _template_multiplier(template, key, index, default):
	for name in (key, f"{key}_multiplier", f"{key}_gamma"):
		if isinstance(template, dict) and template.get(name, None) is not None:
			return float(template[name])
	pair = _first_value(
		template.get("clearance_depth_template", None) if isinstance(template, dict) else None,
		template.get("template", None) if isinstance(template, dict) else None,
		template.get("multipliers", None) if isinstance(template, dict) else None,
	)
	if isinstance(pair, str):
		parts = [part.strip() for part in re.split(r"[:,]", pair) if part.strip()]
		if len(parts) == 2:
			return float(parts[index])
	if isinstance(pair, (list, tuple)) and len(pair) == 2:
		return float(pair[index])
	return float(default)


def _fixed_pair_from_float(value):
	value = float(value)
	return f"{value:.12g},{value:.12g}"


def _type_reference_anchor(cfg, template, mesh_path, plug_col, reference_radius_col, reference_depth_col):
	anchor_assembly_id = _cfg_value(cfg, "srsa_axial_reference_anchor_assembly_id", None)
	if anchor_assembly_id is None:
		return None
	task_type_id = int(template.get("task_type_id", _resolve_axial_task_type_id(cfg, template)))
	anchor_task_type_id = _cfg_value(cfg, "srsa_axial_reference_anchor_task_type_id", None)
	if anchor_task_type_id is None:
		anchor_task_type_id = _resolve_axial_task_type_id(cfg, None)
	anchor_task_type_id = int(anchor_task_type_id)
	if task_type_id != anchor_task_type_id:
		return None

	anchor_assembly_id = _normalize_srsa_assembly_id(anchor_assembly_id)
	with open(mesh_path, "r", encoding="utf-8", newline="") as f:
		for anchor_row in csv.DictReader(f):
			if _normalize_srsa_assembly_id(anchor_row.get("assembly_id")) != anchor_assembly_id:
				continue
			anchor_plug_diameter = _float_row_value(anchor_row, plug_col)
			return {
				"assembly_id": anchor_assembly_id,
				"task_type_id": anchor_task_type_id,
				"reference_radius": max(
					1.0e-8,
					_float_row_value(anchor_row, reference_radius_col, default=0.5 * anchor_plug_diameter),
				),
				"reference_depth": max(
					1.0e-8,
					_float_row_value(anchor_row, reference_depth_col, default=0.015),
				),
			}
	raise ValueError(
		f"Reference anchor assembly_id={anchor_assembly_id} was not found in mesh geometry CSV: {mesh_path}"
	)


def _build_mesh_template_entry(cfg, manifest, template, row, assembly_id, mesh_path):
	mesh_cfg = manifest.get("mesh_geometry", {}) if isinstance(manifest, dict) else {}
	plug_col = _mesh_option(cfg, mesh_cfg, "plug_diameter_column", "plug_diameter_column", "plug_xy_bbox_max")
	hole_col = _mesh_option(cfg, mesh_cfg, "hole_diameter_column", "hole_diameter_column", "socket_xy_bbox_max")
	clearance_col = _mesh_option(cfg, mesh_cfg, "clearance_column", "clearance_base_column", "plug_to_socket_surface_dist_p05")
	clearance_mode = _mesh_option(cfg, mesh_cfg, "clearance_mode", "clearance_mode", "auto")
	depth_col = _mesh_option(cfg, mesh_cfg, "depth_column", "depth_base_column", "plug_bbox_z")
	reference_radius_col = _mesh_option(
		cfg, mesh_cfg, "reference_radius_column", "reference_radius_column", "plug_xy_radius_p95_from_centroid"
	)
	reference_depth_col = _mesh_option(cfg, mesh_cfg, "reference_depth_column", "reference_depth_column", depth_col)

	plug_diameter = _float_row_value(row, plug_col)
	mesh_hole_diameter = _float_row_value(row, hole_col)
	raw_clearance = _float_row_value(row, clearance_col)
	clearance_base = _diametral_clearance_from_mesh(raw_clearance, clearance_col, clearance_mode)
	clearance_base = max(0.0, clearance_base * float(_cfg_value(cfg, "srsa_mesh_clearance_scale", 1.0)))
	depth_base = max(0.0, _float_row_value(row, depth_col) * float(_cfg_value(cfg, "srsa_mesh_depth_scale", 1.0)))
	local_reference_radius = max(1.0e-8, _float_row_value(row, reference_radius_col, default=0.5 * plug_diameter))
	local_reference_depth = max(1.0e-8, _float_row_value(row, reference_depth_col, default=depth_base or 0.015))
	reference_anchor = _type_reference_anchor(
		cfg,
		template,
		mesh_path,
		plug_col,
		reference_radius_col,
		reference_depth_col,
	)
	if reference_anchor is None:
		reference_radius = local_reference_radius
		reference_depth = local_reference_depth
	else:
		reference_radius = reference_anchor["reference_radius"]
		reference_depth = reference_anchor["reference_depth"]
	clearance_multiplier = _template_multiplier(template, "clearance", 0, 1.0)
	depth_multiplier = _template_multiplier(template, "depth", 1, 1.0)
	diametral_clearance = max(0.0, clearance_base * clearance_multiplier)
	radial_clearance = 0.5 * diametral_clearance
	target_depth = max(0.0, depth_base * depth_multiplier)
	hole_diameter = plug_diameter + diametral_clearance
	task_id = int(template.get("template_id", template.get("task_id", 0)))
	scale_ratio = plug_diameter / max(SRSA_BASE_PLUG_DIAMETER, 1.0e-8)
	template_name = template.get("template_name", template.get("task_name", f"c{clearance_multiplier:g}-d{depth_multiplier:g}"))

	entry = {
		"task_id": task_id,
		"template_id": task_id,
		"assembly_id": assembly_id,
		"task_name": f"srsa-{assembly_id}-{template_name}",
		"action_dim": srsa_policy_action_dim(cfg, template.get("action_dim", _cfg_value(cfg, "isaaclab_action_dim", 6))),
		"max_episode_steps": int(template.get("max_episode_steps", _cfg_value(cfg, "isaaclab_max_episode_steps", 75) - 1)),
		"mesh_geometry": {
			"csv": str(mesh_path),
			"plug_diameter_column": plug_col,
			"hole_diameter_column": hole_col,
			"clearance_base_column": clearance_col,
			"clearance_mode": clearance_mode,
			"depth_base_column": depth_col,
			"reference_radius_column": reference_radius_col,
			"reference_depth_column": reference_depth_col,
			"mesh_hole_diameter": mesh_hole_diameter,
			"raw_clearance_proxy": raw_clearance,
			"clearance_base_diametral": clearance_base,
			"depth_base": depth_base,
			"local_reference_radius": local_reference_radius,
			"local_reference_depth": local_reference_depth,
			"reference_anchor": reference_anchor,
		},
		"srsa_params": {
			"task_family_name": template.get("task_family_name", "normal_fit"),
			"task_family_id": int(template.get("task_family_id", 1)),
			"task_type_id": int(template.get("task_type_id", 0)),
			"plug_diameter": plug_diameter,
			"hole_diameter": hole_diameter,
			"mesh_hole_diameter": mesh_hole_diameter,
			"clearance": diametral_clearance,
			"diametral_clearance": diametral_clearance,
			"radial_clearance": radial_clearance,
			"clearance_ratio": diametral_clearance / max(plug_diameter, 1.0e-8),
			"clearance_multiplier": clearance_multiplier,
			"insertion_depth": target_depth,
			"target_insertion_depth": target_depth,
			"depth_multiplier": depth_multiplier,
			"success_pos_tol": float(template.get("success_pos_tol", 0.015)),
			"scale_ratio": scale_ratio,
			"yaw_requirement": bool(template.get("yaw_requirement", False)),
			"reference_radius": reference_radius,
			"reference_depth": reference_depth,
		},
		"srsa_sampler": {
			"srsa_enable_axial_task_param_sampler": True,
			"srsa_axial_task_type_id": int(template.get("task_type_id", 0)),
			"srsa_axial_fixed_plug_scale": False,
			"srsa_axial_scale_range": _fixed_pair_from_float(scale_ratio),
			"srsa_axial_clearance_base": clearance_base,
			"srsa_axial_clearance_depth_templates": f"{clearance_multiplier:.12g}:{depth_multiplier:.12g}",
			"srsa_axial_clearance_jitter_ratio": 0.0,
			"srsa_axial_depth_base": depth_base,
			"srsa_axial_depth_jitter_ratio": 0.0,
			"srsa_axial_reference_radius": reference_radius,
			"srsa_axial_reference_depth": reference_depth,
			"srsa_axial_yaw_requirement": bool(template.get("yaw_requirement", False)),
		},
	}
	entry["task_vec_6"] = make_axial_task_vec(cfg, _merge_template_params(entry))
	return entry


def _load_srsa_task_template_tasks(cfg, template_fp):
	manifest, template_path = _load_json_document(template_fp, cfg=cfg)
	if isinstance(manifest, dict) and isinstance(manifest.get("tasks"), list):
		return sorted(manifest["tasks"], key=lambda item: int(item["task_id"]))
	if not isinstance(manifest, dict):
		raise ValueError(f"SRSA task template file at {template_fp} must be a JSON object.")
	templates = manifest.get("parameter_templates", manifest.get("templates", None))
	if not isinstance(templates, list) or len(templates) == 0:
		raise ValueError(
			f"SRSA task template file at {template_fp} must contain either 'tasks' or non-empty 'parameter_templates'."
		)
	row, assembly_id, mesh_path = _read_mesh_geometry_row(cfg, manifest.get("mesh_geometry", {}) or {}, template_path)
	entries = [_build_mesh_template_entry(cfg, manifest, template, row, assembly_id, mesh_path) for template in templates]
	return sorted(entries, key=lambda item: int(item["task_id"]))


SRSA_SAMPLER_CFG_KEYS = (
	"srsa_enable_axial_task_param_sampler",
	"srsa_axial_task_type_id",
	"srsa_axial_scale_range",
	"srsa_axial_fixed_plug_scale",
	"srsa_axial_clearance_range",
	"srsa_axial_clearance_ratio_range",
	"srsa_axial_clearance_base",
	"srsa_axial_clearance_anchor_multipliers",
	"srsa_axial_clearance_anchors",
	"srsa_axial_clearance_jitter_ratio",
	"srsa_axial_clearance_anchor_weights",
	"srsa_axial_clearance_depth_template_multipliers",
	"srsa_axial_clearance_depth_templates",
	"srsa_axial_clearance_depth_template_weights",
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
)


def _sampler_source_keys(cfg_key):
	keys = [cfg_key]
	if cfg_key.startswith("srsa_"):
		keys.append(cfg_key[len("srsa_"):])
	if cfg_key.startswith("srsa_axial_"):
		keys.append(cfg_key[len("srsa_axial_"):])
	if cfg_key == "srsa_enable_axial_task_param_sampler":
		keys.append("enable_axial_task_param_sampler")
	if cfg_key == "srsa_axial_task_type_id":
		keys.extend(["task_type_id", "task_type_id_float"])
	return tuple(dict.fromkeys(keys))


def _template_section(item, key):
	if not isinstance(item, dict):
		return None
	section = item.get(key, None)
	return section if isinstance(section, dict) else None


def _template_value(item, *keys):
	for key in keys:
		if isinstance(item, dict) and key in item and item[key] is not None:
			return item[key]
	params = item.get("srsa_params", None) if isinstance(item, dict) else None
	if isinstance(params, dict):
		for key in keys:
			if key in params and params[key] is not None:
				return params[key]
	return None


def _merge_template_params(item):
	merged = dict(item)
	params = merged.get("srsa_params", None)
	if isinstance(params, dict):
		for key, value in params.items():
			merged.setdefault(key, value)
	return merged


def _apply_template_sampler_config(cfg, item):
	for section in (_template_section(item, "srsa_sampler"), item):
		if not isinstance(section, dict):
			continue
		for cfg_key in SRSA_SAMPLER_CFG_KEYS:
			for source_key in _sampler_source_keys(cfg_key):
				if source_key in section and section[source_key] is not None:
					_set_cfg_value(cfg, cfg_key, section[source_key])
					break


def _set_from_template_if_empty(cfg, item, cfg_key, *template_keys):
	if _cfg_value(cfg, cfg_key, None) is not None:
		return
	value = _template_value(item, *template_keys)
	if value is not None:
		_set_cfg_value(cfg, cfg_key, value)


def _set_from_template(cfg, item, cfg_key, *template_keys):
	value = _template_value(item, *template_keys)
	if value is not None:
		_set_cfg_value(cfg, cfg_key, value)


def _fixed_pair(value):
	value = float(value)
	return f"{value:.12g},{value:.12g}"


def _apply_exact_axial_task_vec_to_sampler(cfg, task_vec):
	task_vec = _parse_vector(task_vec, expected_dim=int(_cfg_value(cfg, "axial_task_vec_dim", 6)))
	reference_radius = float(_cfg_value(
		cfg,
		"srsa_axial_reference_radius",
		_cfg_value(cfg, "axial_reference_radius", SRSA_BASE_PLUG_DIAMETER * 0.5),
	) or _cfg_value(cfg, "axial_reference_radius", SRSA_BASE_PLUG_DIAMETER * 0.5))
	reference_depth = float(_cfg_value(
		cfg,
		"srsa_axial_reference_depth",
		_cfg_value(cfg, "axial_reference_depth", 0.015),
	) or _cfg_value(cfg, "axial_reference_depth", 0.015))
	scale = math.exp(float(task_vec[1]))
	radial_clearance = float(task_vec[2]) * max(reference_radius, 1.0e-8)
	diametral_clearance = 2.0 * max(0.0, radial_clearance)
	target_depth = float(task_vec[4]) * max(reference_depth, 1.0e-8)

	_set_cfg_value(cfg, "axial_task_vec_6", task_vec)
	_set_cfg_value(cfg, "srsa_enable_axial_task_param_sampler", True)
	_set_cfg_value(cfg, "srsa_axial_task_type_id", int(round(float(task_vec[0]))))
	_set_cfg_value(cfg, "srsa_axial_fixed_plug_scale", False)
	_set_cfg_value(cfg, "srsa_axial_scale_range", _fixed_pair(scale))
	_set_cfg_value(cfg, "srsa_axial_clearance_range", _fixed_pair(diametral_clearance))
	_set_cfg_value(cfg, "srsa_axial_clearance_ratio_range", None)
	_set_cfg_value(cfg, "srsa_axial_clearance_base", diametral_clearance)
	_set_cfg_value(cfg, "srsa_axial_clearance_anchor_multipliers", None)
	_set_cfg_value(cfg, "srsa_axial_clearance_anchors", None)
	_set_cfg_value(cfg, "srsa_axial_clearance_anchor_weights", None)
	_set_cfg_value(cfg, "srsa_axial_clearance_depth_template_multipliers", None)
	_set_cfg_value(cfg, "srsa_axial_clearance_depth_templates", None)
	_set_cfg_value(cfg, "srsa_axial_clearance_depth_template_weights", None)
	_set_cfg_value(cfg, "srsa_axial_clearance_jitter_ratio", 0.0)
	_set_cfg_value(cfg, "srsa_axial_depth_range", _fixed_pair(target_depth))
	_set_cfg_value(cfg, "srsa_axial_target_depth_range", _fixed_pair(target_depth))
	_set_cfg_value(cfg, "srsa_axial_depth_base", target_depth)
	_set_cfg_value(cfg, "srsa_axial_depth_anchor_multipliers", None)
	_set_cfg_value(cfg, "srsa_axial_depth_anchors", None)
	_set_cfg_value(cfg, "srsa_axial_depth_anchor_weights", None)
	_set_cfg_value(cfg, "srsa_axial_depth_jitter_ratio", 0.0)
	_set_cfg_value(cfg, "srsa_axial_yaw_requirement", bool(float(task_vec[5]) > 0.5))
	return task_vec


def _update_selected_task_vector(cfg, task_id, task_vec):
	if task_id is None or task_vec is None:
		return
	task_vectors = _cfg_value(cfg, "task_vectors", None)
	if not task_vectors:
		_set_cfg_value(cfg, "task_vectors", [task_vec])
		return
	task_id = int(task_id)
	if 0 <= task_id < len(task_vectors):
		task_vectors[task_id] = task_vec
	elif len(task_vectors) == 1:
		task_vectors[0] = task_vec
	_set_cfg_value(cfg, "task_vectors", task_vectors)


def _update_selected_task_name(cfg, task_id, task_name):
	if task_id is None or task_name is None:
		return
	task_id = int(task_id)
	for key in ("tasks", "global_tasks"):
		values = _cfg_value(cfg, key, None)
		if not values or not isinstance(values, list):
			continue
		if 0 <= task_id < len(values):
			values[task_id] = str(task_name)
			_set_cfg_value(cfg, key, values)


def _entry_has_task_params(entry):
	if not isinstance(entry, dict):
		return False
	for key in (
		"task_vec",
		"task_vec_6",
		"axial_task_vec",
		"axial_task_vec_6",
		"srsa_params",
		"srsa_sampler",
		"plug_diameter",
		"male_diameter",
		"hole_diameter",
		"female_diameter",
		"clearance",
		"diametral_clearance",
		"radial_clearance",
		"clearance_ratio",
		"insertion_depth",
		"target_insertion_depth",
		"scale_ratio",
	):
		if entry.get(key, None) is not None:
			return True
	return False


def resolve_eval_task_template(cfg, entry=None):
	if entry is not None:
		entry = _merge_template_params(entry)
		if _should_recompute_manifest_task_vec(cfg):
			template = _manifest_task_template_entry(cfg, entry)
			if template is not None:
				return template
		if _entry_has_task_params(entry) or _cfg_value(cfg, "srsa_task_template_fp", None) is None:
			return entry
		task_id = _first_value(
			_get_value(entry, "srsa_param_template_id", None),
			_get_value(entry, "srsa_task_template_id", None),
			_cfg_value(cfg, "srsa_param_template_id", None),
			_cfg_value(cfg, "srsa_task_template_id", None),
			_cfg_value(cfg, "eval_task_id", None),
			_get_value(entry, "task_id", None),
		)
		template_cfg = deepcopy(cfg)
		if entry.get("assembly_id", None) is not None:
			_set_cfg_value(template_cfg, "assembly_id", str(entry["assembly_id"]).zfill(5))
		_set_cfg_value(template_cfg, "axial_task_vec_6", None)
		_set_cfg_value(template_cfg, "srsa_task_template_applied_id", None)
		template_tasks = _load_template_tasks(template_cfg)
		if template_tasks is None or task_id is None:
			return entry
		for item in template_tasks:
			if int(item["task_id"]) != int(task_id):
				continue
			template = _merge_template_params(item)
			template["task_id"] = int(entry.get("task_id", template.get("task_id", 0)))
			template["assembly_id"] = entry.get("assembly_id", template.get("assembly_id"))
			template["task_name"] = entry.get("task_name", template.get("task_name"))
			if entry.get("eval_index", None) is not None:
				template["eval_index"] = entry["eval_index"]
			return template
		raise ValueError(f"task_id={task_id} was not found in the configured SRSA task templates.")
	task_id = _cfg_value(cfg, "eval_task_id", None)
	if task_id is None:
		task_id = _cfg_value(cfg, "srsa_task_template_id", None)
	if (
		task_id is not None and
		_should_recompute_manifest_task_vec(cfg) and
		_cfg_value(cfg, "offline_manifest_fp", None)
	):
		for item in _load_manifest_tasks(_cfg_value(cfg, "offline_manifest_fp")):
			if int(item["task_id"]) != int(task_id):
				continue
			template = _manifest_task_template_entry(cfg, _merge_template_params(item))
			if template is not None:
				return template
			break
	template_tasks = _load_template_tasks(cfg)
	if template_tasks is None or task_id is None:
		return None
	for item in template_tasks:
		if int(item["task_id"]) == int(task_id):
			return _merge_template_params(item)
	raise ValueError(f"task_id={task_id} was not found in the configured SRSA task templates.")


def apply_eval_task_template(cfg, entry=None):
	"""
	Apply a manifest/template task entry to eval-time SRSA config.

	The selected task_id remains the model-side selector. When a template provides
	task_vec_6 and eval_task_template_exact=true, the vector is decoded into fixed
	SRSA sampler ranges so the environment and AxialTaskEncoder see the same task.
	"""
	template = resolve_eval_task_template(cfg, entry)
	if template is None:
		return cfg

	task_id = int(template.get("task_id", _cfg_value(cfg, "eval_task_id", 0) or 0))
	if entry is None and _cfg_value(cfg, "srsa_task_template_applied_id", None) == task_id:
		return cfg
	_set_cfg_value(cfg, "eval_task_id", task_id)
	_set_cfg_value(cfg, "srsa_task_template_id", task_id)
	if template.get("assembly_id", None) is not None:
		_set_cfg_value(cfg, "assembly_id", str(template["assembly_id"]).zfill(5))

	_set_from_template(cfg, template, "srsa_task_family_name", "task_family_name")
	_set_from_template(cfg, template, "srsa_task_family_id", "task_family_id")
	_set_from_template(cfg, template, "srsa_plug_diameter", "plug_diameter", "male_diameter")
	_set_from_template(cfg, template, "srsa_hole_diameter", "hole_diameter", "female_diameter")
	_set_from_template(cfg, template, "srsa_clearance", "clearance", "diametral_clearance")
	_set_from_template(cfg, template, "srsa_clearance_ratio", "clearance_ratio")
	_set_from_template(cfg, template, "srsa_insertion_depth", "insertion_depth", "target_insertion_depth")
	_set_from_template(cfg, template, "srsa_success_pos_tol", "success_pos_tol")
	_set_from_template(cfg, template, "srsa_axial_task_type_id", "task_type_id", "task_type_id_float")
	_set_from_template(cfg, template, "srsa_axial_reference_radius", "reference_radius")
	_set_from_template(cfg, template, "srsa_axial_reference_depth", "reference_depth")
	_apply_template_sampler_config(cfg, template)

	task_vec = _template_value(template, "task_vec", "task_vec_6", "axial_task_vec", "axial_task_vec_6")
	if task_vec is not None and bool(_cfg_value(cfg, "eval_task_template_exact", True)):
		task_vec = _apply_exact_axial_task_vec_to_sampler(cfg, task_vec)
	elif task_vec is not None:
		task_vec = _parse_vector(task_vec, expected_dim=int(_cfg_value(cfg, "axial_task_vec_dim", 6)))
		_set_cfg_value(cfg, "axial_task_vec_6", task_vec)
	else:
		_set_cfg_value(cfg, "axial_task_vec_6", None)
		task_vec = make_axial_task_vec(cfg, template)
		_set_cfg_value(cfg, "axial_task_vec_6", task_vec)
	_update_selected_task_vector(cfg, task_id, task_vec)
	_update_selected_task_name(cfg, task_id, template.get("task_name", None))

	if bool(_cfg_value(cfg, "eval_task_template_print", True)) and int(_cfg_value(cfg, "rank", 0) or 0) == 0:
		vec_text = ", ".join(f"{float(value):.6g}" for value in task_vec) if task_vec is not None else "n/a"
		print(colored(
			f"Applied eval task template: task_id={task_id} "
			f"assembly_id={_cfg_value(cfg, 'assembly_id', None)} task_vec_6=[{vec_text}]",
			"cyan",
			attrs=["bold"],
		))
	_set_cfg_value(cfg, "srsa_task_template_applied_id", task_id)
	return cfg


def parse_cfg(cfg):
	"""
	Parses the experiment config dataclass. Mostly for convenience.
	"""
	if cfg.get('isaaclab_backend', 'auto') == 'srsa':
		if cfg.task == 'soup':
			cfg.task = 'isaaclab-srsa-assembly'
		if cfg.isaaclab_env_id == 'Isaac-AutoMate-Assembly-Direct-v0':
			if cfg.get('srsa_sparse_reward', False):
				cfg.isaaclab_env_id = 'Assembly-Sparse-Sil-v0' if cfg.get('srsa_sil', False) else 'Assembly-Sparse-v0'
			else:
				cfg.isaaclab_env_id = 'Assembly-Direct-Sil-v0' if cfg.get('srsa_sil', False) else 'Assembly-Direct-v0'
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
	cfg.latent_residual_gate_mode = str(cfg.get('latent_residual_gate_mode', 'always')).lower()
	if cfg.latent_residual_gate_mode not in {"always", "contact"}:
		raise ValueError(
			f"Unknown latent_residual_gate_mode={cfg.latent_residual_gate_mode!r}. "
			"Use always or contact."
		)
	if cfg.get('latent_residual_force_history_len', None) is None:
		cfg.latent_residual_force_history_len = int(cfg.get('contact_history_len', 4))
	if cfg.get('latent_residual_force_dim', None) is None:
		cfg.latent_residual_force_dim = int(cfg.get('contact_force_dim', 6))
	mode = eval_mode(cfg)
	if mode in REAL_EVAL_MODES:
		cfg.eval_mode = "real"
		cfg.eval_zmq_enabled = True
	elif mode in SIM_EVAL_MODES:
		cfg.eval_mode = "sim"
	elif mode not in SIM_EVAL_MODES | REAL_EVAL_MODES:
		raise ValueError(
			f"Unknown eval_mode={cfg.eval_mode!r}. Use sim or real."
		)
	if cfg.get('eval_success_metric', None) is not None:
		cfg.srsa_eval_success_metric = cfg.eval_success_metric
	_validate_multitask_continuation_paths(cfg)
	if cfg.get('multitask_continuation_enabled', False):
		if bool(cfg.get('latent_residual_enabled', False)):
			raise ValueError(
				"multitask_continuation_enabled is for the shared family model path; "
				"leave latent_residual_enabled=false for this simulation multitask stage."
			)
		curriculum_mode = str(cfg.get('multitask_curriculum_mode', 'progressive')).lower()
		if curriculum_mode not in {"progressive", "all_at_once"}:
			raise ValueError("multitask_curriculum_mode must be one of: progressive, all_at_once.")
		sampling_mode = str(cfg.get('multitask_sampling_mode', 'balanced')).lower()
		if sampling_mode not in {"balanced", "weighted", "proportional"}:
			raise ValueError("multitask_sampling_mode must be one of: balanced, weighted, proportional.")
		if bool(cfg.get('multitask_distill_old_policy_enabled', False)):
			raise ValueError(
				"multitask_distill_old_policy_enabled is intentionally not part of the default "
				"Scheme A path. Use replay/proximal regularization for this implementation."
			)
	angle_tol_rad = math.radians(float(cfg.get('strict_angle_tol_deg', 3.0)))
	cfg.srsa_process_success_depth_ratio = float(cfg.get('strict_depth_fraction', 0.90))
	cfg.srsa_process_success_stable_steps = int(cfg.get('strict_success_steps', 10))
	cfg.srsa_process_success_lateral_tol_min = float(cfg.get('strict_lateral_tol_min', 0.0005))
	cfg.srsa_process_success_lateral_tol_max = cfg.get('strict_lateral_tol_max', 0.0020)
	cfg.srsa_process_success_keypoint_tol_min = float(cfg.get('strict_keypoint_tol_min', 0.0010))
	cfg.srsa_process_success_keypoint_tol_max = cfg.get('strict_keypoint_tol_max', 0.0030)
	cfg.srsa_process_success_orientation_tol_rad = angle_tol_rad
	cfg.srsa_process_success_yaw_tol_rad = angle_tol_rad

	# Set defaults
	manifest_tasks = None
	if cfg.get('srsa_param_template_id', None) is not None:
		cfg.srsa_task_template_id = int(cfg.srsa_param_template_id)
	if cfg.get('srsa_task_template_id', None) is not None and cfg.get('eval_task_id', None) is None:
		cfg.eval_task_id = int(cfg.srsa_task_template_id)
	if cfg.offline_manifest_fp:
		manifest_tasks = _load_manifest_tasks(cfg.offline_manifest_fp)
	elif cfg.srsa_task_template_fp:
		manifest_tasks = _load_template_tasks(cfg)
	if manifest_tasks is not None:
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

	if manifest_tasks is not None and cfg.get('eval_task_id', None) is not None:
		cfg = apply_eval_task_template(cfg)

	if cfg.eval_task_id is not None:
		assert 0 <= cfg.eval_task_id < cfg.num_global_tasks, \
			f'eval_task_id={cfg.eval_task_id} is out of range for {cfg.num_global_tasks} tasks.'
	if cfg.run_id is None:
		cfg.run_id = make_run_id(cfg)
	else:
		cfg.run_id = safe_run_token(cfg.run_id)
	try:
		original_cwd = Path(hydra.utils.get_original_cwd())
	except ValueError:
		original_cwd = Path.cwd()
	cfg.work_dir = original_cwd / 'logs' / cfg.task / str(cfg.seed) / cfg.exp_name / cfg.run_id

	return OmegaConf.to_object(cfg)
