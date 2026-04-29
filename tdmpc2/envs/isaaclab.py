import sys
import os
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


def _build_canonical_obs(env) -> torch.Tensor:
	import isaacsim.core.utils.torch as torch_utils

	frame_quat = env.fixed_quat
	frame_pos = env.fixed_pos_obs_frame
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
	return torch.cat(
		[
			tcp_pos_socket,
			tcp_quat_socket,
			tcp_linvel_socket,
			tcp_angvel_socket,
			gripper_width,
		],
		dim=-1,
	)


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
			self.observation_space = gym.spaces.Box(
				low=-np.inf,
				high=np.inf,
				shape=(14,),
				dtype=np.float32,
			)
		else:
			self.observation_space = self.env.unwrapped.single_observation_space['policy']
		self.action_space = gym.spaces.Box(
			low=-1.0,
			high=1.0,
			shape=self.env.unwrapped.single_action_space.shape,
			dtype=np.float32,
		)
		# AutoMate times out when episode_length_buf >= max_episode_length - 1.
		self.max_episode_steps = max(1, int(self.env.unwrapped.max_episode_length) - 1)

	def rand_act(self):
		return torch.rand(
			(self.cfg.num_envs, *self.action_space.shape),
			dtype=torch.float32,
			device=self.env.unwrapped.device,
		) * 2 - 1

	def _extract_obs(self, obs_dict):
		if self._use_canonical_obs:
			return _build_canonical_obs(self.env.unwrapped).detach().to(torch.float32)
		return obs_dict['policy'].detach().to(torch.float32)

	def reset(self, **kwargs):
		obs, _ = self.env.reset(**kwargs)
		return self._extract_obs(obs), {
			'success': torch.zeros(self.cfg.num_envs, dtype=torch.float32, device=self.env.unwrapped.device)
		}

	def _step_with_final_obs(self, action):
		env = self.env.unwrapped
		action = action.to(env.device, non_blocking=True)

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
		episode_success = env.ep_succeeded.clone().to(torch.float32) if hasattr(env, 'ep_succeeded') else done.to(torch.float32)
		if done.any():
			final_obs_dict = env._get_observations()
			final_obs = self._extract_obs(final_obs_dict)[done].clone()

		reset_env_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
		if len(reset_env_ids) > 0:
			env._reset_idx(reset_env_ids)
			if env.sim.has_rtx_sensors() and env.cfg.num_rerenders_on_reset > 0:
				for _ in range(env.cfg.num_rerenders_on_reset):
					env.sim.render()

		if env.cfg.events and "interval" in env.event_manager.available_modes:
			env.event_manager.apply(mode="interval", dt=env.step_dt)

		env.obs_buf = env._get_observations()
		if env.cfg.observation_noise_model:
			env.obs_buf["policy"] = env._observation_noise_model(env.obs_buf["policy"])

		return env.obs_buf, env.reward_buf, done, final_obs, episode_success

	def step(self, action):
		obs_dict, reward, done, final_obs, episode_success = self._step_with_final_obs(action)
		obs = self._extract_obs(obs_dict)
		reward = reward.detach().to(torch.float32)
		done = done.detach()
		terminated = torch.zeros_like(done)
		truncated = done.clone()

		info = {
			'success': episode_success.detach(),
		}
		if done.any():
			success = torch.full((self.cfg.num_envs,), float('nan'), dtype=torch.float32, device=done.device)
			success[done] = episode_success.detach()[done]
			info['final_observation'] = final_obs
			info['final_info'] = {
				'success': success,
				'score': success.clone(),
			}
		return obs, reward, terminated, truncated, info

	def render(self, *args, **kwargs):
		frame = self.env.render(*args, **kwargs)
		return None if frame is None else frame.copy()

	def close(self):
		return self.env.close()


def make_env(cfg):
	"""
	Make an Isaac Lab AutoMate environment.
	"""
	if not (cfg.task.startswith('isaaclab-') or cfg.task.startswith('Isaac-') or cfg.isaaclab_env_id.startswith('Isaac-')):
		raise ValueError('Unknown task:', cfg.task)
	_launch_isaaclab_app(cfg)
	import isaaclab_tasks  # noqa: F401
	from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

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
	render_mode = 'rgb_array' if (cfg.save_video or cfg.obs == 'rgb') else None
	env = gym.make(cfg.isaaclab_env_id, cfg=env_cfg, render_mode=render_mode)
	_configure_soft_dtw(env, cfg)
	env = IsaacLabWrapper(env, cfg)
	print(f'[Rank {cfg.rank}] Created Isaac Lab env {cfg.isaaclab_env_id} for assembly_id={cfg.assembly_id}')
	if cfg.isaaclab_use_canonical_obs:
		print(f'[Rank {cfg.rank}] Using canonical 14D Isaac Lab observations.')
	return env
