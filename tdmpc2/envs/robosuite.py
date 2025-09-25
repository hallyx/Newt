import logging
import gymnasium as gym
import numpy as np
import robosuite

from envs.wrappers.timeout import Timeout


ROBOSUITE_TASKS = {
	'rs-lift-block': dict(
		env='Lift',
		robot='Panda',
		max_episode_steps=50,
	),
	'rs-stack': dict(
		env='Stack',
		robot='Panda',
		max_episode_steps=50,
	),
}


class RobosuiteWrapper(gym.Wrapper):
	def __init__(self, env, cfg):
		super().__init__(env)
		self.env = env
		self.cfg = cfg
		obs, _ = self.reset()
		self.observation_space = gym.spaces.Box(
			low=-np.inf, high=np.inf, shape=(obs.shape[0],), dtype=np.float32)
		self.action_space = gym.spaces.Box(
			low=-1.0, high=1.0, shape=env.action_spec[0].shape, dtype=np.float32)
		print('Observation space:', self.observation_space)
		print('Action space:', self.action_space)
	
	def _extract_info(self, info):
		info = {
			'terminated': info.get('terminated', False),
			'truncated': info.get('truncated', False),
			'success': info.get('success', float(self.env._check_success())),
		}
		info['score'] = info['success']
		return info

	def preprocess_obs(self, obs):
		proprio = obs['robot0_proprio-state']
		object = obs['object-state']
		return np.concatenate([
			proprio,
			object,
		]).astype(np.float32)
	
	def reset(self, **kwargs):
		obs = self.env.reset(**kwargs)
		obs = self.preprocess_obs(obs)
		return obs, self._extract_info({})
	
	def step(self, action):
		obs, reward, truncated, info = self.env.step(action)
		obs = self.preprocess_obs(obs)
		info = self._extract_info(info)
		return obs, reward, False, truncated, info
	
	@property
	def unwrapped(self):
		return self.env.unwrapped
	
	@property
	def metadata(self):
		return {}

	def render(self, *args, **kwargs):
		frame = self.env.sim.render(
			height=224, width=224, camera_name='frontview').copy()
		frame = frame[::-1, :, :]  # flip frame vertically
		return frame


def make_env(cfg):
	"""
	Make Robosuite environment.
	"""
	if not cfg.task in ROBOSUITE_TASKS:
		raise ValueError('Unknown task:', cfg.task)
	task_cfg = ROBOSUITE_TASKS[cfg.task]
	logging.getLogger("robosuite_logs").setLevel(logging.WARNING)
	env = robosuite.make(
		env_name=task_cfg['env'],
		robots=task_cfg['robot'],
		has_renderer=False,
		has_offscreen_renderer=True,
		use_camera_obs=cfg.obs=='rgb',
		# cameras=['frontview'],
		# camera_heights=64,
		# camera_widths=64,
		control_freq=10,
		horizon=task_cfg['max_episode_steps'],
		reward_shaping=True,
	)
	env = RobosuiteWrapper(env, cfg)
	env = Timeout(env, max_episode_steps=task_cfg['max_episode_steps'])
	return env
