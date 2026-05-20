import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ["TORCH_DISTRIBUTED_TIMEOUT"] = "1800"
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
os.environ['TORCH_LOGS'] = "+recompiles"
import warnings
warnings.filterwarnings('ignore')

from collections import defaultdict

import torch
import hydra
from hydra.core.config_store import ConfigStore
from termcolor import colored

from common import barrier, set_seed
from common.logger import Logger
from common.world_model import WorldModel
from config import Config, parse_cfg
from envs import make_env
from tdmpc2 import TDMPC2
from trainer import Trainer
from zmq_action_publisher import make_eval_zmq_publisher

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

cs = ConfigStore.instance()
cs.store(name="config", node=Config)


def setup(rank, world_size, port):
	os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
	os.environ["MASTER_PORT"] = port
	torch.distributed.init_process_group(
		backend="nccl",
		rank=rank,
		world_size=world_size
	)
	return port


def empty_metrics():
	return {'reward': [], 'length': [], 'success': [], 'score': []}


def eval_by_trials(trainer: Trainer, total_trials: int):
	"""
	Evaluate for an exact total number of completed episodes across all envs.
	This matches AutoMate's success definition by using final_info.success,
	which is derived from ep_succeeded over the episode.
	"""
	local_target = total_trials
	if trainer.cfg.world_size > 1:
		local_target = total_trials // trainer.cfg.world_size
		if trainer.cfg.rank < (total_trials % trainer.cfg.world_size):
			local_target += 1

	task_results = defaultdict(empty_metrics)
	obs, info = trainer.env.reset()
	episode_reward = torch.zeros(trainer.cfg.num_envs, device=trainer._rollout_device)
	episode_len = torch.zeros(trainer.cfg.num_envs, device=trainer._rollout_device)
	completed = 0

	if trainer.cfg.save_video:
		trainer.logger.video.init(trainer.env, enabled=trainer.cfg.rank == 0)

	with make_eval_zmq_publisher(trainer.cfg) as action_publisher:
		while completed < local_target:
			use_mpc = trainer._step > 0 or trainer.cfg.finetune
			torch.compiler.cudagraph_mark_step_begin()
			model_tasks = trainer._model_tasks()
			action, _ = trainer.agent(obs, t0=episode_len == 0, step=trainer._step, eval_mode=True, task=model_tasks, mpc=use_mpc)
			if trainer.cfg.rank == 0:
				env_index = int(trainer.cfg.get('eval_zmq_env_index', 0))
				action_publisher.send_action(
					action,
					step=trainer._step,
					episode_step=int(episode_len[env_index].item()),
					task_id=int(trainer._tasks[env_index].item()),
				)
			obs, reward, terminated, truncated, info = trainer.env.step(action)

			done = terminated | truncated
			episode_reward += reward
			episode_len += 1

			if trainer.cfg.rank == 0:
				env_index = int(trainer.cfg.get('eval_zmq_env_index', 0))
				if bool(done[env_index].item()):
					action_publisher.send_done(
						step=trainer._step,
						episode_step=int(episode_len[env_index].item()),
						task_id=int(trainer._tasks[env_index].item()),
					)

			if 'final_info' in info:
				for i in range(trainer.cfg.num_envs):
					if not done[i]:
						continue
					if completed >= local_target:
						break
					task_id = trainer._tasks[i].item()
					task_name = trainer.cfg.global_tasks[task_id]
					task_results[task_name]['reward'].append(episode_reward[i].item())
					task_results[task_name]['length'].append(episode_len[i].item())
					task_results[task_name]['success'].append(info['final_info']['success'][i].item())
					task_results[task_name]['score'].append(info['final_info']['score'][i].item())
					episode_reward[i] = 0.0
					episode_len[i] = 0.0
					completed += 1

			if trainer.cfg.save_video and completed == 0:
				trainer.logger.video.record(trainer.env)

	if trainer.cfg.save_video:
		trainer.logger.video.save(trainer._step)

	barrier()

	if trainer.cfg.world_size > 1:
		gathered_results = [None for _ in range(trainer.cfg.world_size)] if trainer.cfg.rank == 0 else None
		torch.distributed.gather_object(task_results, gathered_results, dst=0)
		if trainer.cfg.rank == 0:
			merged_results = defaultdict(empty_metrics)
			for rank_results in gathered_results:
				for task_name, metrics in rank_results.items():
					for metric_name, values in metrics.items():
						merged_results[task_name][metric_name].extend(values)
			task_results = merged_results
		else:
			return None

	if trainer.cfg.rank != 0:
		return None

	metrics = {}
	total_count = 0
	total_success = 0.0
	for task_name, values in task_results.items():
		if len(values['reward']) == 0:
			continue
		task_count = len(values['reward'])
		total_count += task_count
		total_success += sum(values['success'])
		prefix = f'eval/{task_name}'
		metrics[f'{prefix}/episode_reward'] = sum(values['reward']) / task_count
		metrics[f'{prefix}/episode_length'] = sum(values['length']) / task_count
		metrics[f'{prefix}/episode_success'] = sum(values['success']) / task_count
		metrics[f'{prefix}/episode_score'] = sum(values['score']) / task_count

	if total_count == 0:
		raise RuntimeError('No completed evaluation episodes were collected.')

	metrics['episode_reward'] = sum(sum(v['reward']) for v in task_results.values()) / total_count
	metrics['episode_length'] = sum(sum(v['length']) for v in task_results.values()) / total_count
	metrics['episode_success'] = total_success / total_count
	metrics['episode_score'] = sum(sum(v['score']) for v in task_results.values()) / total_count
	metrics['eval_trials'] = total_count
	return metrics


def evaluate(rank: int, cfg: dict):
	"""
	Script for checkpoint evaluation.
	Loads a trained model, runs Trainer.eval(), logs metrics, and exits.
	"""
	if cfg.world_size > 1:
		setup(rank, cfg.world_size, cfg.port)
		print(colored('Rank:', 'yellow', attrs=['bold']), rank)
	set_seed(cfg.seed + rank)
	cfg.rank = rank
	cfg.device_id = cfg.gpu_id + rank
	torch.cuda.set_device(cfg.device_id)

	if not cfg.checkpoint:
		raise ValueError('`checkpoint` must be provided for evaluation.')
	if not os.path.exists(cfg.checkpoint):
		raise FileNotFoundError(f'Checkpoint file not found: {cfg.checkpoint}')
	if cfg.num_global_tasks > 1 and cfg.task != 'soup' and cfg.eval_task_id is None:
		raise ValueError(
			'`eval_task_id` must be provided when evaluating a multitask checkpoint outside of soup mode.'
		)

	def make_agent(cfg):
		model = WorldModel(cfg).to(f"cuda:{cfg.device_id}")
		agent = TDMPC2(model, cfg)
		agent.load(cfg.checkpoint)
		return agent

	cfg.save_agent = False
	trainer = Trainer(
		cfg=cfg,
		env=make_env(cfg),
		agent=make_agent(cfg),
		buffer=None,
		logger=Logger(cfg),
	)
	barrier()
	try:
		if cfg.rank == 0:
			print(colored(f'Evaluating checkpoint: {cfg.checkpoint}', 'blue', attrs=['bold']))
			if cfg.eval_task_id is not None:
				print(colored(f'Evaluation task_id: {cfg.eval_task_id}', 'blue', attrs=['bold']))
			if cfg.eval_zmq_enabled:
				print(colored(f'Sending eval actions over ZMQ to {cfg.eval_zmq_server}', 'blue', attrs=['bold']))
		if cfg.mpc:
			trainer._step = 1
		if cfg.eval_trials is not None:
			eval_metrics = eval_by_trials(trainer, cfg.eval_trials)
		else:
			eval_metrics = trainer.eval()
		eval_metrics.update(trainer.common_metrics())
		if cfg.task == 'soup':
			trainer.logger.pprint_multitask(eval_metrics, cfg)
		trainer.logger.log(eval_metrics, 'eval')
		if cfg.rank == 0 and cfg.eval_trials is not None:
			print(colored(
				f"AutoMate-style success over {int(eval_metrics['eval_trials'])} trials: {float(eval_metrics['episode_success']):.4f}",
				'green',
				attrs=['bold'],
			))
		trainer.logger.finish()
		if cfg.rank == 0:
			print(colored('Evaluation completed successfully.', 'green', attrs=['bold']))
	except Exception as e:
		print(colored(f'[Rank {cfg.rank}] Evaluation crashed with exception: {repr(e)}', 'red', attrs=['bold']))
		raise
	finally:
		if torch.distributed.is_initialized():
			torch.distributed.destroy_process_group()


@hydra.main(version_base=None, config_name="config")
def launch(cfg: Config):
	assert torch.cuda.is_available()
	cfg = parse_cfg(cfg)
	cfg.enable_wandb = cfg.enable_wandb
	print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)

	available_gpus = torch.cuda.device_count() - cfg.gpu_id
	assert available_gpus > 0, \
		f'gpu_id={cfg.gpu_id} leaves no visible CUDA devices (total={torch.cuda.device_count()}).'
	if cfg.multiproc:
		requested_gpus = cfg.num_gpus if cfg.num_gpus is not None else available_gpus
		assert requested_gpus > 0, f'num_gpus must be positive, got {requested_gpus}.'
		assert requested_gpus <= available_gpus, \
			f'Requested num_gpus={requested_gpus}, but only {available_gpus} GPUs are available from gpu_id={cfg.gpu_id}.'
		cfg.world_size = requested_gpus
	else:
		cfg.world_size = 1
	if cfg.world_size > 1:
		gpu_range = f'{cfg.gpu_id}-{cfg.gpu_id + cfg.world_size - 1}'
		print(colored(f'Using {cfg.world_size} GPUs for evaluation (cuda:{gpu_range})', 'green', attrs=['bold']))

	if cfg.world_size > 1:
		cfg.port = os.getenv("MASTER_PORT", str(12355 + int(os.getpid()) % 1000))
		torch.multiprocessing.spawn(
			evaluate,
			args=(cfg,),
			nprocs=cfg.world_size,
			join=True,
		)
	else:
		evaluate(0, cfg)


if __name__ == '__main__':
	launch()
