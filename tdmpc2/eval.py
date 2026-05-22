import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ["TORCH_DISTRIBUTED_TIMEOUT"] = "1800"
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
os.environ['TORCH_LOGS'] = "+recompiles"
import warnings
warnings.filterwarnings('ignore')

from collections import defaultdict
from pathlib import Path
from time import monotonic

import torch
import hydra
from hydra.core.config_store import ConfigStore
from termcolor import colored

from common import barrier, set_seed
from common.logger import Logger
from common.world_model import WorldModel
from config import Config, apply_eval_task_template, parse_cfg
from envs import make_env
from tdmpc2 import TDMPC2
from trainer import Trainer
from zmq_action_publisher import make_eval_zmq_observation_receiver, make_eval_zmq_publisher

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


def _real_eval_mode(cfg) -> str:
	return str(cfg.get('eval_real_mode', 'stream')).strip().lower().replace('-', '_')


def _is_real_closed_loop(cfg) -> bool:
	return cfg.get('eval_mode', 'sim') == 'real' and _real_eval_mode(cfg) in {
		'closed_loop',
		'robot_closed_loop',
		'obs_closed_loop',
	}


def _checkpoint_state_dict(checkpoint_fp):
	obj = torch.load(checkpoint_fp, map_location="cpu", weights_only=False)
	return obj["model"] if isinstance(obj, dict) and "model" in obj else obj


def _get_state_tensor(state_dict, key: str):
	for candidate in (key, f"module.{key}"):
		if candidate in state_dict:
			return state_dict[candidate]
	return None


def _infer_checkpoint_io(checkpoint_fp):
	state_dict = _checkpoint_state_dict(checkpoint_fp)
	enc_weight = _get_state_tensor(state_dict, "_encoder.state.0.weight")
	if enc_weight is None:
		raise KeyError(f"Could not find `_encoder.state.0.weight` in checkpoint={checkpoint_fp}.")
	task_emb = _get_state_tensor(state_dict, "_task_emb.weight")
	task_vecs = _get_state_tensor(state_dict, "_task_vecs")
	task_encoder = _get_state_tensor(state_dict, "_task_encoder.type_encoder.weight")
	if task_emb is not None:
		task_dim = int(task_emb.shape[-1])
		task_conditioning = "id_embedding"
	elif task_vecs is not None or task_encoder is not None:
		task_dim = 64
		task_conditioning = "axial_params"
	else:
		task_dim = 0
		task_conditioning = "none"
	obs_dim = int(enc_weight.shape[1]) - task_dim
	if obs_dim <= 0:
		raise ValueError(
			f"Could not infer positive obs_dim from checkpoint={checkpoint_fp}: "
			f"encoder_in={int(enc_weight.shape[1])}, task_dim={task_dim}."
		)
	action_masks = _get_state_tensor(state_dict, "_action_masks")
	action_dim = int(action_masks.shape[-1]) if action_masks is not None else 6
	return {
		"obs_dim": obs_dim,
		"action_dim": action_dim,
		"task_dim": task_dim,
		"task_conditioning": task_conditioning,
	}


def _configure_real_closed_loop_cfg(cfg):
	if cfg.world_size != 1:
		raise ValueError("`eval_real_mode=closed_loop` only supports a single process/GPU.")
	if cfg.num_envs != 1:
		print(colored("Forcing num_envs=1 for real closed-loop eval.", "yellow", attrs=["bold"]))
		cfg.num_envs = 1
	compat = _infer_checkpoint_io(cfg.checkpoint)
	if str(cfg.task_conditioning).lower() != compat["task_conditioning"]:
		raise ValueError(
			"Real closed-loop config does not match checkpoint task conditioning: "
			f"cfg={cfg.task_conditioning}, checkpoint={compat['task_conditioning']}."
		)
	cfg.obs = 'state'
	cfg.obs_shape = {'state': (int(compat["obs_dim"]),)}
	cfg.action_dim = int(compat["action_dim"])
	if not cfg.action_dims:
		cfg.action_dims = [cfg.action_dim]
	else:
		cfg.action_dims = [
			cfg.action_dim if int(dim) <= 0 else min(int(dim), cfg.action_dim)
			for dim in cfg.action_dims
		]
	cfg.episode_length = int(
		cfg.get('eval_real_steps', None) or
		cfg.get('episode_length', None) or
		cfg.get('isaaclab_max_episode_steps', 75)
	)
	cfg.episode_lengths = [cfg.episode_length for _ in cfg.episode_lengths]
	return compat


def _real_task_id(cfg) -> int:
	if cfg.get('eval_task_id', None) is not None:
		return int(cfg.eval_task_id)
	if cfg.get('srsa_task_template_id', None) is not None:
		return int(cfg.srsa_task_template_id)
	return 0


def _real_task_input(cfg, obs_receiver, message, device):
	if (
		bool(cfg.get('eval_real_use_msg_task_vec', True)) and
		str(cfg.get('task_conditioning', '')).lower() == 'axial_params'
	):
		task_vec = obs_receiver.task_vec_tensor(message, device=device)
		if task_vec is not None:
			return task_vec
	task_id = _real_task_id(cfg)
	if str(cfg.get('task_conditioning', '')).lower() == 'axial_params':
		task_vectors = cfg.get('task_vectors', None) or []
		if len(task_vectors) == 1:
			return torch.tensor(task_vectors, dtype=torch.float32, device=device)
		if 0 <= task_id < len(task_vectors):
			return torch.tensor([task_vectors[task_id]], dtype=torch.float32, device=device)
	return torch.tensor([task_id], dtype=torch.long, device=device)


@torch.no_grad()
def eval_real_closed_loop(agent: TDMPC2, cfg, logger: Logger):
	"""
	Closed-loop real-robot inference.

	Robot side publishes the latest canonical observation over ZMQ. Newt consumes
	that observation, runs the current policy/planner, and sends one 6D delta
	action back to the robot action receiver.
	"""
	device = torch.device(f"cuda:{cfg.device_id}")
	obs_dim = int(cfg.obs_shape['state'][0])
	max_steps = int(cfg.get('eval_real_steps', None) or cfg.episode_length)
	use_mpc = bool(cfg.get('mpc', True))
	task_id = _real_task_id(cfg)
	step_count = 0
	last_log = None
	start_time = monotonic()

	print(colored(
		"Starting real closed-loop inference: "
		f"obs_dim={obs_dim}, action_dim={cfg.action_dim}, max_steps={max_steps}, "
		f"obs_endpoint={cfg.eval_real_obs_server} ({cfg.eval_real_obs_socket_type}), "
		f"action_endpoint={cfg.eval_zmq_server}.",
		"cyan",
		attrs=["bold"],
	))
	print(colored(
		"Robot observation can be either direct `obs` or libfranka robot_state. "
		"Direct obs layout: [tcp_pos_socket(3), tcp_quat_wxyz(4), tcp_linvel_socket(3), "
		"tcp_angvel_socket(3), gripper_width(1), optional force/wrench]. "
		"libfranka mode needs --full-state plus socket pose calibration and force fields for force checkpoints.",
		"cyan",
		attrs=["bold"],
	))

	with make_eval_zmq_publisher(cfg) as action_publisher, make_eval_zmq_observation_receiver(cfg) as obs_receiver:
		message = obs_receiver.recv()
		for step_idx in range(max_steps):
			if obs_receiver.is_done(message):
				action_publisher.send_done(step=step_idx, episode_step=step_idx, task_id=task_id)
				break
			obs = obs_receiver.obs_tensor(message, obs_dim=obs_dim, device=device)
			model_tasks = _real_task_input(cfg, obs_receiver, message, device)
			t0 = torch.tensor([step_idx == 0], dtype=torch.bool, device=device)
			torch.compiler.cudagraph_mark_step_begin()
			action, info = agent(
				obs,
				t0=t0,
				step=1 if use_mpc else 0,
				eval_mode=True,
				task=model_tasks,
				mpc=use_mpc,
			)
			episode_step = int(message.get("episode_step", step_idx))
			action_publisher.send_action(
				action,
				step=step_idx,
				episode_step=episode_step,
				task_id=task_id,
				state_seq=message.get("seq", None),
				state_timestamp=message.get("timestamp", None),
			)
			step_count += 1
			elapsed_s = monotonic() - start_time
			if (
				last_log is None or
				elapsed_s - last_log >= float(cfg.get('progress_log_interval_sec', 30.0)) or
				step_idx == max_steps - 1
			):
				last_log = elapsed_s
				action_max = float(action.detach().abs().max().item())
				pi_std = info.get("pi_std", None) if info is not None else None
				pi_std_text = "n/a" if pi_std is None else f"{float(torch.as_tensor(pi_std).detach().cpu().item()):.4g}"
				print(colored(
					f"real progress step={step_count}/{max_steps} "
					f"elapsed={elapsed_s:.1f}s action_abs_max={action_max:.4g} pi_std={pi_std_text}",
					"cyan",
					attrs=["bold"],
				), flush=True)
			if step_idx + 1 >= max_steps:
				break
			message = obs_receiver.recv()
		action_publisher.send_done(step=step_count, episode_step=step_count, task_id=task_id)

	elapsed = monotonic() - start_time
	return {
		"step": step_count,
		"episode": 1,
		"episode_reward": 0.0,
		"episode_score": 0.0,
		"episode_length": step_count,
		"episode_success": 0.0,
		"eval_real_steps": step_count,
		"elapsed_time": elapsed,
		"steps_per_second": step_count / max(elapsed, 1.0e-6),
	}


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
	real_closed_loop = _is_real_closed_loop(cfg)
	real_compat = None
	if real_closed_loop:
		real_compat = _configure_real_closed_loop_cfg(cfg)

	def make_agent(cfg):
		model = WorldModel(cfg).to(f"cuda:{cfg.device_id}")
		agent = TDMPC2(model, cfg)
		agent.load(cfg.checkpoint)
		agent.eval()
		agent.model.eval()
		return agent

	cfg.save_agent = False
	if real_closed_loop:
		logger = Logger(cfg)
		agent = make_agent(cfg)
		try:
			if cfg.rank == 0:
				print(colored(f'Evaluating checkpoint: {cfg.checkpoint}', 'blue', attrs=['bold']))
				print(colored('Evaluation mode: real closed_loop', 'blue', attrs=['bold']))
				print(colored(
					f"Checkpoint I/O: obs_dim={real_compat['obs_dim']} "
					f"action_dim={real_compat['action_dim']} task_dim={real_compat['task_dim']}",
					'blue',
					attrs=['bold'],
				))
			eval_metrics = eval_real_closed_loop(agent, cfg, logger)
			logger.log(eval_metrics, 'eval')
			logger.finish()
			if cfg.rank == 0:
				print(colored('Real closed-loop inference completed successfully.', 'green', attrs=['bold']))
			return
		except Exception as e:
			print(colored(f'[Rank {cfg.rank}] Real closed-loop eval crashed with exception: {repr(e)}', 'red', attrs=['bold']))
			raise
		finally:
			if torch.distributed.is_initialized():
				torch.distributed.destroy_process_group()

	env = make_env(cfg)
	logger = Logger(cfg)
	agent = make_agent(cfg)
	trainer = Trainer(
		cfg=cfg,
		env=env,
		agent=agent,
		buffer=None,
		logger=logger,
	)
	barrier()
	try:
		if cfg.rank == 0:
			print(colored(f'Evaluating checkpoint: {cfg.checkpoint}', 'blue', attrs=['bold']))
			print(colored(f'Evaluation mode: {cfg.eval_mode}', 'blue', attrs=['bold']))
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
	if cfg.checkpoint:
		cfg.checkpoint = str(
			Path(hydra.utils.to_absolute_path(str(cfg.checkpoint))).expanduser().resolve()
		)
	cfg = parse_cfg(cfg)
	cfg = apply_eval_task_template(cfg)
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
