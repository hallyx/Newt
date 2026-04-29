import warnings
warnings.filterwarnings('ignore')

import gymnasium as gym

from config import is_isaaclab_task
from envs.wrappers.vectorized_multitask import make_vectorized_multitask_env
from envs.wrappers.render import Render


def make_isaaclab_env(cfg):
	from envs.isaaclab import make_env as _make_env
	return _make_env(cfg)


def make_dm_control_env(cfg):
	from envs.dmcontrol import make_env as _make_env
	return _make_env(cfg)


def make_maniskill_env(cfg):
	from envs.maniskill import make_env as _make_env
	return _make_env(cfg)


def make_metaworld_env(cfg):
	from envs.metaworld import make_env as _make_env
	return _make_env(cfg)


def make_mujoco_env(cfg):
	from envs.mujoco import make_env as _make_env
	return _make_env(cfg)


def make_box2d_env(cfg):
	from envs.box2d import make_env as _make_env
	return _make_env(cfg)


def make_robodesk_env(cfg):
	from envs.robodesk import make_env as _make_env
	return _make_env(cfg)


def make_ogbench_env(cfg):
	from envs.ogbench import make_env as _make_env
	return _make_env(cfg)


def make_pygame_env(cfg):
	from envs.pygame import make_env as _make_env
	return _make_env(cfg)


def make_atari_env(cfg):
	from envs.atari import make_env as _make_env
	return _make_env(cfg)


def make_env(cfg):
	"""
	Make an environment for Newt experiments.
	"""
	if hasattr(gym.logger, 'set_level'):
		gym.logger.set_level(40)
	else:
		gym.logger.min_level = 40
	if not cfg.child_env and not is_isaaclab_task(cfg):
		env = make_vectorized_multitask_env(cfg, make_env)
	else:
		env = None
		for fn in [
			make_isaaclab_env,
			make_dm_control_env, make_maniskill_env, make_metaworld_env,
			make_mujoco_env, make_box2d_env, make_robodesk_env,
			make_ogbench_env, make_pygame_env, make_atari_env,
		]:
			try:
				env = fn(cfg)
				break
			except ValueError as e:
				if 'Unknown task' in str(e):
					continue
				else:
					raise e
		if env is None:
			raise ValueError(f'Failed to make environment "{cfg.task}": please verify that dependencies are installed and that the task exists.')
		assert cfg.num_envs == 1 or cfg.get('obs', 'state') == 'state', \
			'Vectorized environments only support state observations.'
		if cfg.save_video and cfg.get('num_demos', 0) > 0:
			env = Render(env, cfg)
		print(f'[Rank {cfg.rank}] Created env for task {cfg.task}')
	try: # Dict
		cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
	except: # Box
		cfg.obs_shape = {cfg.get('obs', 'state'): env.observation_space.shape}
	cfg.action_dim = env.action_space.shape[0]
	cfg.episode_length = env.max_episode_steps
	if is_isaaclab_task(cfg) and cfg.task != 'soup':
		cfg.episode_lengths = [env.max_episode_steps for _ in cfg.episode_lengths]
	return env
