import os
import datetime
from collections import defaultdict, OrderedDict
from contextlib import nullcontext
from time import time

import torch
import numpy as np
from termcolor import colored
from tqdm import tqdm
from tensordict.tensordict import TensorDict

from common import barrier


def split_by_rank(global_list, rank, world_size):
	"""Split a global list into sublists for each rank."""
	return [global_list[i] for i in range(len(global_list)) if i % world_size == rank]


def empty_metrics():
	return {'reward': [], 'length': [], 'success': [], 'score': []}


class Trainer():
	"""Trainer class for MMBench experiments."""

	def __init__(self, cfg, env, agent, buffer, logger):
		self.cfg = cfg
		self.env = env
		self.agent = agent
		self.buffer = buffer
		self.logger = logger
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()
		self._last_progress_log = 0.0
		self._progress_log_interval = max(0.0, float(cfg.get('progress_log_interval_sec', 30.0)))
		self._rollout_device = torch.device(f'cuda:{cfg.device_id}') if (
			cfg.task.startswith('isaaclab-') or
			cfg.get('isaaclab_env_id', '').startswith('Isaac-') or
			cfg.get('isaaclab_backend', 'auto') == 'srsa' or
			cfg.get('isaaclab_task_package', None) == 'SRSA.tasks'
		) else torch.device('cpu')
		if cfg.task == 'soup':
			self._tasks = torch.tensor(split_by_rank(range(cfg.num_global_tasks), cfg.rank, cfg.world_size),
				dtype=torch.int32, device=self._rollout_device)
			self._episode_lengths = torch.tensor(split_by_rank(cfg.episode_lengths, cfg.rank, cfg.world_size),
				dtype=torch.int32, device=self._rollout_device)
		elif cfg.num_global_tasks > 1:
			task_id = int(cfg.eval_task_id) if cfg.eval_task_id is not None else 0
			if cfg.eval_task_id is None and cfg.rank == 0:
				print(colored(
					'Warning: multi-task config without eval_task_id; defaulting evaluation task to task_id=0.',
					'yellow',
					attrs=['bold'],
				))
			self._tasks = torch.full((cfg.num_envs,), task_id, dtype=torch.int32, device=self._rollout_device)
			self._episode_lengths = torch.full(
				(cfg.num_envs,),
				int(cfg.episode_lengths[task_id]),
				dtype=torch.int32,
				device=self._rollout_device,
			)
		else:
			self._tasks = torch.tensor([0], dtype=torch.int32, device=self._rollout_device)
			self._episode_lengths = torch.tensor([cfg.episode_lengths[0]], dtype=torch.int32, device=self._rollout_device)
			self._tasks = self._tasks.repeat_interleave(cfg.num_envs)
			self._episode_lengths = self._episode_lengths.repeat_interleave(cfg.num_envs)
		assert int(self._episode_lengths.max().item()) <= int(self.cfg.episode_length), \
			f'[Rank {cfg.rank}] Expected configured episode_length={self.cfg.episode_length} to cover task lengths {self._episode_lengths.tolist()}.'
		self._tds = TensorDict({}, batch_size=(self.cfg.episode_length+1, self.cfg.num_envs), device=self._rollout_device)
		self._update_freq = self.cfg.num_envs * self.cfg.episode_length * self.cfg.world_size
		self._update_tokens = 0
		self._eps_per_update_freq = int((cfg.episode_length / np.array(cfg.episode_lengths)).sum())
		if cfg.rank == 0:
			print('Architecture:', self.agent.model)
			print(f'Update frequency: {self._update_freq:,}')
			print(f'Episodes per update frequency: {self._eps_per_update_freq:,}')
			if self._progress_log_interval > 0:
				print(f'Progress heartbeat: every {self._progress_log_interval:.0f}s during eval/rollout/update.')

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		elapsed_time = time() - self._start_time
		return dict(
			step=self._step,
			episode=self._ep_idx,
			elapsed_time=elapsed_time,
			steps_per_second=self._step / elapsed_time
		)

	def _uses_runtime_task_vec(self):
		return (
			bool(self.cfg.get('srsa_use_runtime_task_vec', True)) and
			str(self.cfg.get('task_conditioning', '')).lower() == 'axial_params'
		)

	def _runtime_task_vec(self):
		if not self._uses_runtime_task_vec():
			return None
		env = getattr(self.env, 'unwrapped', None)
		task_vec = getattr(env, 'current_task_vec', None)
		if task_vec is None or not torch.is_tensor(task_vec):
			return None
		if task_vec.ndim != 2 or task_vec.shape[0] != self.cfg.num_envs:
			return None
		if int(task_vec.shape[-1]) != int(self.cfg.get('axial_task_vec_dim', 6)):
			return None
		return task_vec.detach().to(self._rollout_device, dtype=torch.float32, non_blocking=True).clone()

	def _model_tasks(self):
		task_vec = self._runtime_task_vec()
		return self._tasks if task_vec is None else task_vec

	def _elapsed_str(self):
		return str(datetime.timedelta(seconds=int(time() - self._start_time)))

	def _maybe_log_progress(self, phase, extra="", force=False):
		if self.cfg.rank != 0 or self._progress_log_interval <= 0:
			return
		now = time()
		if not force and now - self._last_progress_log < self._progress_log_interval:
			return
		self._last_progress_log = now
		msg = (
			f"{phase:<8} progress "
			f"E: {self._ep_idx:,} I: {self._step:,} "
			f"T: {self._elapsed_str()}"
		)
		if extra:
			msg += f" {extra}"
		print(colored(msg, 'cyan', attrs=['bold']), flush=True)

	def eval(self):
		"""Evaluate agent and aggregate all completed episodes per unique task name."""
		task_results = defaultdict(empty_metrics)

		obs, info = self.env.reset()
		episode_reward = torch.zeros(self.cfg.num_envs, device=self._rollout_device)
		episode_len = torch.zeros(self.cfg.num_envs, device=self._rollout_device)
		episodes_completed = torch.zeros(self.cfg.num_envs, dtype=torch.int32, device=self._rollout_device)

		if self.cfg.save_video:
			self.logger.video.init(self.env, enabled=self.cfg.rank==0)

		eval_env_steps = 0
		eval_guard_steps = int(
			max(1, self.cfg.eval_episodes) *
			max(1, self.cfg.episode_length) *
			max(1.0, float(self.cfg.get('eval_hang_guard_factor', 2.0)))
		)
		self._maybe_log_progress(
			'eval',
			extra=f"start target_eps={self.cfg.eval_episodes} envs={self.cfg.num_envs} guard_steps={eval_guard_steps}",
			force=True,
		)
		use_eval_zmq = bool(self.cfg.get('eval_zmq_enabled', False)) and self.cfg.rank == 0
		if use_eval_zmq:
			from zmq_action_publisher import make_eval_zmq_publisher
			publisher_context = make_eval_zmq_publisher(self.cfg)
		else:
			publisher_context = nullcontext(None)
		with publisher_context as action_publisher:
			while (episodes_completed < self.cfg.eval_episodes).any():
				use_mpc = self._step > 0 or self.cfg.finetune
				torch.compiler.cudagraph_mark_step_begin()
				model_tasks = self._model_tasks()
				action, _ = self.agent(obs, t0=episode_len==0, step=self._step, eval_mode=True, task=model_tasks, mpc=use_mpc)
				if action_publisher is not None:
					env_index = int(self.cfg.get('eval_zmq_env_index', 0))
					action_publisher.send_action(
						action,
						step=self._step,
						episode_step=int(episode_len[env_index].item()),
						task_id=int(self._tasks[env_index].item()),
					)
				obs, reward, terminated, truncated, info = self.env.step(action)
				eval_env_steps += 1

				done = terminated | truncated
				episode_reward += reward
				episode_len += 1

				if action_publisher is not None:
					env_index = int(self.cfg.get('eval_zmq_env_index', 0))
					if bool(done[env_index].item()):
						action_publisher.send_done(
							step=self._step,
							episode_step=int(episode_len[env_index].item()),
							task_id=int(self._tasks[env_index].item()),
						)

				if 'final_info' in info:
					for i in range(self.cfg.num_envs):
						if done[i]:
							task_id = self._tasks[i].item()
							task_name = self.cfg.global_tasks[task_id]

							task_results[task_name]['reward'].append(episode_reward[i].item())
							task_results[task_name]['length'].append(episode_len[i].item())
							task_results[task_name]['success'].append(info['final_info']['success'][i].item())
							task_results[task_name]['score'].append(info['final_info']['score'][i].item())

							episode_reward[i] = 0.0
							episode_len[i] = 0.0
							episodes_completed[i] += 1
				self._maybe_log_progress(
					'eval',
					extra=(
						f"env_steps={eval_env_steps} "
						f"done_this_step={int(done.sum().item())}/{self.cfg.num_envs} "
						f"eps_done_min={int(episodes_completed.min().item())} "
						f"eps_done_max={int(episodes_completed.max().item())} "
						f"ep_len_max={int(episode_len.max().item())}"
					),
				)
				if eval_env_steps > eval_guard_steps and (episodes_completed < self.cfg.eval_episodes).any():
					raise RuntimeError(
						"Evaluation did not finish within the configured guard. "
						f"eval_env_steps={eval_env_steps}, guard_steps={eval_guard_steps}, "
						f"episodes_completed_min={int(episodes_completed.min().item())}, "
						f"episodes_completed_max={int(episodes_completed.max().item())}, "
						f"episode_len_max={int(episode_len.max().item())}. "
						"Check whether the environment is returning truncated/final_info."
					)

				if self.cfg.save_video and episodes_completed.min() == 0:
					self.logger.video.record(self.env)

		if self.cfg.save_video:
			self.logger.video.save(self._step)

		barrier()  # Ensure all processes have completed evaluation

		if self.cfg.world_size > 1:
			# Gather results from all ranks
			gathered_results = [None for _ in range(self.cfg.world_size)] if self.cfg.rank == 0 else None
			torch.distributed.gather_object(task_results, gathered_results, dst=0)
			if self.cfg.rank == 0:
				# Combine results from all ranks
				for rank_results in gathered_results:
					for task_name, metrics in rank_results.items():
						for metric_name, values in metrics.items():
							task_results[task_name][metric_name].extend(values)

		results = {}
		if self.cfg.task == 'soup' and self.cfg.rank == 0:
			assert len(task_results) == self.cfg.num_global_tasks, \
				f'Expected results for {self.cfg.num_global_tasks} tasks, but got {len(task_results)}.'

		# Sort tasks by order in cfg.global_tasks
		task_results = OrderedDict(sorted(task_results.items(), key=lambda x: self.cfg.global_tasks.index(x[0])))

		# Compute per-task averages
		for task_name, metrics in task_results.items():
			n = len(metrics['reward'])
			results[f'episode_reward+{task_name}'] = sum(metrics['reward']) / n
			results[f'episode_length+{task_name}'] = sum(metrics['length']) / n
			results[f'episode_success+{task_name}'] = sum(metrics['success']) / n
			results[f'episode_score+{task_name}'] = sum(metrics['score']) / n

		# Compute unweighted averages *across tasks*
		num_tasks = len(task_results)
		if self.cfg.rank == 0 and self.cfg.task == 'soup':
			assert num_tasks == self.cfg.num_global_tasks, \
				f'Number of eval tasks ({num_tasks}) does not match expected ({self.cfg.num_global_tasks})'
		results['episode_reward'] = sum(
			sum(m['reward']) / len(m['reward']) for m in task_results.values()
		) / num_tasks
		results['episode_success'] = sum(
			sum(m['success']) / len(m['success']) for m in task_results.values()
		) / num_tasks
		results['episode_score'] = sum(
			sum(m['score']) / len(m['score']) for m in task_results.values()
		) / num_tasks

		return results

	def to_td(self, obs, action=None, reward=None, terminated=None, task=None):
		"""Creates a TensorDict for a new episode."""
		if isinstance(obs, dict):
			obs = TensorDict(obs, batch_size=(), device=self._rollout_device)
		else:
			obs = obs.to(self._rollout_device, non_blocking=True)
		if action is None:
			action = torch.full_like(self.env.rand_act(), float('nan'))
		else:
			action = action.to(self._rollout_device, non_blocking=True)
		if reward is None:
			reward = torch.tensor(float('nan'), device=self._rollout_device).repeat(self.cfg.num_envs)
		else:
			reward = reward.to(self._rollout_device, non_blocking=True)
		if terminated is None:
			terminated = torch.tensor(False, device=self._rollout_device).repeat(self.cfg.num_envs)
		elif not isinstance(terminated, torch.Tensor):
			terminated = torch.stack(terminated.tolist()).to(self._rollout_device, non_blocking=True)
		else:
			terminated = terminated.to(self._rollout_device, non_blocking=True)
		if task is None:
			task = self._model_tasks()
		else:
			task = task.to(self._rollout_device, non_blocking=True)
		td = TensorDict(
			obs=obs,
			action=action,
			reward=reward,
			terminated=terminated,
			task=task,
			batch_size=(self.cfg.num_envs,))
		return td

	def _reset_train_rollout(self):
		obs, info = self.env.reset()
		ep_reward = torch.zeros((self.cfg.num_envs,), device=self._rollout_device)
		ep_len = torch.zeros((self.cfg.num_envs,), dtype=torch.int32, device=self._rollout_device)
		done = torch.full((self.cfg.num_envs,), True, dtype=torch.bool, device=self._rollout_device)
		action_infos = []
		self._next_action = None
		self._tds[ep_len] = self.to_td(obs)
		return obs, ep_reward, ep_len, done, action_infos
	
	def train(self):
		"""Train a Newt agent."""
		# Load demonstrations
		use_demos = self.cfg.get('use_demos', False)
		
		# Load checkpoint
		checkpoint = self.cfg.get('checkpoint', None)
		if checkpoint:
			if not os.path.exists(checkpoint):
				raise FileNotFoundError(f'Checkpoint file not found: {checkpoint}')
			self.agent.load(self.cfg.checkpoint)
			if self.cfg.rank == 0:
				print(colored(f'Loaded checkpoint from {self.cfg.checkpoint}.', 'blue', attrs=['bold']))
		else:
			checkpoint = None
			if self.cfg.rank == 0:
				print(colored(f'No checkpoint found, training from scratch.', 'yellow', attrs=['bold']))
		
		# Pretrain agent on demonstrations if available
		if use_demos and not checkpoint and self.cfg.demo_steps > 0:
			if self.cfg.rank == 0:
				print('Pretraining agent on demonstrations...')
			self.agent.maxq_pi = False  # Disable max-Q for pretraining
			print(f'prior_coef is {self.agent.cfg.prior_coef}, setting to 1.0 for pretraining.')
			self.agent.cfg.prior_coef = 1.0  # Use only behavior cloning loss
			iterator = tqdm(range(self.cfg.demo_steps), desc='Pretraining') if self.cfg.rank == 0 else range(self.cfg.demo_steps)
			for i in iterator:
				pretrain_metrics = self.agent.update(self.buffer)
				if i % int(self.cfg.demo_steps / 50) == 0:
					self.logger.pprint_pretrain(pretrain_metrics)
			pretrain_metrics.update({
				'step': 0,
				'elapsed_time': time() - self._start_time,
			})
			self.agent.maxq_pi = True
			self.agent.cfg.prior_coef = self.cfg.prior_coef
			print(f'Set prior_coef to {self.agent.cfg.prior_coef} after pretraining.')
			if self.cfg.rank == 0:
				print('Pretraining complete.')
			self.logger.save_agent(self.agent, f'{self._step:,}'.replace(',', '_'), metrics=pretrain_metrics)

		# Training loop
		if self.cfg.rank == 0:
			print(f'Training agent for {self.cfg.steps:,} steps...')
			if self._progress_log_interval > 0:
				print(
					colored(
						f'If the console is quiet, heartbeat lines will appear every '
						f'{self._progress_log_interval:.0f}s.',
						'cyan',
						attrs=['bold'],
					),
					flush=True,
				)
		train_metrics = defaultdict(list)
		obs = ep_reward = ep_len = done = action_infos = None
		while self._step <= self.cfg.steps:

			# Evaluate agent periodically
			should_eval = self.cfg.eval_freq and self._step % self.cfg.eval_freq == 0
			if self._step == 0 and self.cfg.get('skip_initial_eval', False):
				should_eval = False
				if self.cfg.rank == 0:
					print(colored('Skipping initial evaluation; starting rollout immediately.', 'cyan', attrs=['bold']))
			if should_eval:
				eval_metrics = self.eval()
				eval_metrics.update(self.common_metrics())
				if self.cfg.task == 'soup':
					self.logger.pprint_multitask(eval_metrics, self.cfg)
				self.logger.log(eval_metrics, 'eval')
				if self._step > 0:
					self.logger.save_latest_agent(
						self.agent,
						eval_metrics,
					)
				if self.cfg.save_best and self._step > 0:
					self.logger.maybe_save_best_agent(
						self.agent,
						eval_metrics,
						self._step,
					)

				# Save agent
				if self._step % self.cfg.save_freq == 0 and self._step > 0:
					save_metrics = dict(eval_metrics)
					save_metrics['step'] = self._step
					self.logger.save_agent(self.agent, f'{self._step:,}'.replace(',', '_'), metrics=save_metrics)

				# Reset environment and metrics
				obs, ep_reward, ep_len, done, action_infos = self._reset_train_rollout()
			if obs is None:
				obs, ep_reward, ep_len, done, action_infos = self._reset_train_rollout()

			# Collect experience
			model_tasks = self._model_tasks()
			if self.cfg.finetune:
				torch.compiler.cudagraph_mark_step_begin()
				action, action_info = self.agent(obs, t0=done, step=self._step, task=model_tasks, mpc=True)
			elif use_demos and self.cfg.demo_steps > 0:
				use_mpc = self._step >= self.cfg.seeding_coef * self._update_freq
				torch.compiler.cudagraph_mark_step_begin()
				action, action_info = self.agent(obs, t0=done, step=self._step, task=model_tasks, mpc=use_mpc)
			elif self._step >= self.cfg.seeding_coef * self._update_freq:
				torch.compiler.cudagraph_mark_step_begin()
				action, action_info = self.agent(obs, t0=done, step=self._step, task=model_tasks)
			else:
				action, action_info = self.env.rand_act(), None

			obs, reward, terminated, truncated, info = self.env.step(action)
			assert not terminated.any(), \
				f'Unexpected termination signal received.'
			ep_reward += reward
			ep_len += 1
			done = terminated | truncated
			action_infos.append(action_info)
			self._step += self.cfg.num_envs * self.cfg.world_size
			self._maybe_log_progress(
				'rollout',
				extra=(
					f"done_this_step={int(done.sum().item())}/{self.cfg.num_envs} "
					f"ep_len_min={int(ep_len.min().item())} "
					f"ep_len_max={int(ep_len.max().item())} "
					f"buffer_eps={self.buffer.num_eps}"
				),
			)

			# Store experience
			_obs = obs.clone()
			if 'final_observation' in info:
				_obs[done] = info['final_observation']
			td = self.to_td(_obs, action, reward, terminated, task=model_tasks)
			self._tds[ep_len] = td
			if done.any():
				max_ep_len = ep_len.max()

				for i in range(self.cfg.num_envs):
					if done[i]:
						assert ep_len[i] == self._episode_lengths[i], \
							f'Episode length {ep_len[i]} does not match expected length {self._episode_lengths[i]}.'

						# Add to buffer
						_td = self._tds[:ep_len[i]+1, i].unsqueeze(0)
						self.buffer.add(_td, self.cfg.world_size, self.cfg.rank)

						# Save metrics
						train_metrics['episode_reward'].append(ep_reward[i].item())
						train_metrics['episode_success'].append(info['final_info']['success'][i].item())
						train_metrics['episode_score'].append(info['final_info']['score'][i].item())
						train_metrics['episode_length'].append(ep_len[i].item())
						train_metrics['episode_terminated'].append(terminated[i].item())

						# Reset episode metrics
						ep_reward[i] = 0.0
						ep_len[i] = 0

				reset_td = self.to_td(obs)
				self._tds[0, done] = reset_td[done]
				
				# Log and reset metrics if enough data is collected
				if max_ep_len >= self.cfg.episode_length:
					self._ep_idx += self._eps_per_update_freq
					for key in ['episode_reward', 'episode_success', 'episode_score', 'episode_length', 'episode_terminated']:
						train_metrics[key] = torch.tensor(train_metrics[key], dtype=torch.float32).nanmean().item()
					if not (None in action_infos):
						train_metrics.update(torch.stack(action_infos).mean())
					train_metrics.update(self.common_metrics())
					self.logger.log(train_metrics, 'train')
					train_metrics = defaultdict(list)
			
			# Update agent
			if self._step >= self.cfg.seeding_coef * self._update_freq:
				self._update_tokens += self.cfg.num_envs * self.cfg.world_size * self.cfg.utd
				if self._update_tokens >= 1.0:
					num_updates = int(self._update_tokens)
					self._maybe_log_progress('update', extra=f"start updates={num_updates}")
					for _ in range(num_updates):
						_train_metrics = self.agent.update(self.buffer)
					train_metrics.update(_train_metrics)
					self._update_tokens -= num_updates
					self._maybe_log_progress(
						'update',
						extra=f"done updates={num_updates} remaining_tokens={self._update_tokens:.3f}",
						force=True,
					)
		
		self.logger.finish()
