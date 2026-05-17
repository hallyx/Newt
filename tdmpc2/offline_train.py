import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ["TORCH_DISTRIBUTED_TIMEOUT"] = "1800"
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
os.environ['TORCH_LOGS'] = "+recompiles"
import warnings
warnings.filterwarnings('ignore')
from dataclasses import asdict
from pathlib import Path

import hydra
import torch
from hydra.core.config_store import ConfigStore
from termcolor import colored

from common import set_seed
from common.logger import Logger
from common.world_model import WorldModel
from config import Config, parse_cfg
from offline_dataset import OfflineSequenceDataset
from offline_io import export_compact_dataset, export_multitask_compact_dataset
from tdmpc2 import TDMPC2

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

cs = ConfigStore.instance()
cs.store(name="config", node=Config)


def make_agent(cfg):
	model = WorldModel(cfg).to(f"cuda:{cfg.device_id}")
	return TDMPC2(model, cfg)


def _offline_gpu_id(cfg) -> int:
	return cfg.offline_gpu_id if cfg.offline_gpu_id is not None else cfg.gpu_id


def _dataset_episode_length(dataset: OfflineSequenceDataset) -> int:
	if dataset.step_id is None:
		return int(dataset._horizon + 1)
	return int(dataset.step_id.max().item()) + 1


def _prepare_cfg_from_dataset(cfg, dataset: OfflineSequenceDataset):
	cfg.rank = 0
	cfg.world_size = 1
	cfg.device_id = _offline_gpu_id(cfg)
	cfg.num_envs = 1
	cfg.obs = 'state'
	cfg.obs_shape = {'state': tuple(dataset.stats.obs_shape)}
	cfg.action_dim = max([int(dataset.action.shape[-1]), *[int(dim) for dim in (cfg.action_dims or [])]])
	if not cfg.action_dims:
		cfg.action_dims = [cfg.action_dim]
	dataset_episode_length = _dataset_episode_length(dataset)
	if cfg.episode_lengths:
		cfg.episode_length = max(dataset_episode_length, *[int(length) for length in cfg.episode_lengths])
	else:
		cfg.episode_length = dataset_episode_length
		cfg.episode_lengths = [cfg.episode_length]
	return cfg


def _resolve_offline_dataset_fp(cfg):
	if cfg.offline_dataset_fp:
		return Path(cfg.offline_dataset_fp).expanduser().resolve()
	if cfg.offline_manifest_fp:
		manifest_fp = Path(cfg.offline_manifest_fp).expanduser().resolve()
		if cfg.offline_export_fp:
			output_fp = Path(cfg.offline_export_fp).expanduser().resolve()
		else:
			output_fp = Path(cfg.work_dir) / 'data' / f'{manifest_fp.stem}_compact.pt'
		if output_fp.exists() and not cfg.offline_export_overwrite:
			print(colored(f'Reusing existing compact offline dataset: {output_fp}', 'blue', attrs=['bold']))
			return output_fp
		output_fp, metadata_fp, summary = export_multitask_compact_dataset(
			manifest_fp,
			output_fp,
			obs_key=cfg.offline_obs_key,
			next_obs_key=cfg.offline_next_obs_key,
			action_key=cfg.offline_action_key,
			overwrite=cfg.offline_export_overwrite,
		)
		print(colored(f'Prepared compact multitask offline dataset: {output_fp}', 'blue', attrs=['bold']))
		print(colored(f'Offline dataset metadata: {metadata_fp}', 'blue', attrs=['bold']))
		print(colored(
			f"Offline dataset transitions: {summary['num_transitions']:,} across {summary['num_tasks']} tasks",
			'blue',
			attrs=['bold'],
		))
		return output_fp
	if not cfg.offline_source_fp:
		raise ValueError(
			'Provide one of `offline_dataset_fp`, `offline_manifest_fp`, or `offline_source_fp` for offline training.'
		)
	source_fp = Path(cfg.offline_source_fp).expanduser().resolve()
	if cfg.offline_export_fp:
		output_fp = Path(cfg.offline_export_fp).expanduser().resolve()
	else:
		output_fp = Path(cfg.work_dir) / 'data' / f'{source_fp.stem}_compact.pt'
	if output_fp.exists() and not cfg.offline_export_overwrite:
		print(colored(f'Reusing existing compact offline dataset: {output_fp}', 'blue', attrs=['bold']))
		return output_fp
	output_fp, metadata_fp, summary = export_compact_dataset(
		source_fp,
		output_fp,
		obs_key=cfg.offline_obs_key,
		next_obs_key=cfg.offline_next_obs_key,
		action_key=cfg.offline_action_key,
		overwrite=cfg.offline_export_overwrite,
	)
	print(colored(f'Prepared compact offline dataset: {output_fp}', 'blue', attrs=['bold']))
	print(colored(f'Offline dataset metadata: {metadata_fp}', 'blue', attrs=['bold']))
	print(colored(f"Offline dataset transitions: {summary['num_transitions']:,}", 'blue', attrs=['bold']))
	return output_fp


def _stage_cfg(agent: TDMPC2, *, consistency: float, reward: float, value: float, prior: float, maxq_pi: bool):
	agent.maxq_pi = maxq_pi
	agent.cfg.consistency_coef = float(consistency)
	agent.cfg.reward_coef = float(reward)
	agent.cfg.value_coef = float(value)
	agent.cfg.prior_coef = float(prior)


def _run_stage(agent: TDMPC2, dataset: OfflineSequenceDataset, logger: Logger, cfg, *, stage_name: str, num_steps: int):
	if num_steps <= 0:
		return
	log_freq = max(1, int(cfg.offline_log_freq))
	save_freq = max(1, int(cfg.offline_save_freq))
	if logger.rank == 0:
		print(colored(
			f"Starting offline stage '{stage_name}' for {num_steps:,} updates "
			f"(filter={dataset.stats.filter_mode}, horizon={dataset.stats.horizon}, valid_starts={dataset.stats.num_valid_starts:,}).",
			'cyan',
			attrs=['bold'],
		))
	for update in range(1, num_steps + 1):
		metrics = dict(agent.update(dataset).items())
		metrics.update({
			'iteration': update,
			'step': update,
			'stage': stage_name,
		})
		if update == 1 or update % log_freq == 0 or update == num_steps:
			logger.log(metrics, 'pretrain')
		if logger.rank == 0 and (update % save_freq == 0 or update == num_steps):
			logger.save_agent(agent, f'{stage_name}_{update:,}'.replace(',', '_'), metrics=metrics)


@hydra.main(version_base=None, config_name="config")
def launch(cfg: Config):
	assert torch.cuda.is_available()
	cfg = parse_cfg(cfg)
	print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)
	set_seed(cfg.seed)
	offline_gpu = _offline_gpu_id(cfg)
	torch.cuda.set_device(offline_gpu)
	print(colored(f'Offline training device: cuda:{offline_gpu}', 'yellow', attrs=['bold']))
	offline_dataset_fp = _resolve_offline_dataset_fp(cfg)

	dataset = OfflineSequenceDataset(
		path=offline_dataset_fp,
		batch_size=cfg.batch_size,
		horizon=cfg.horizon,
		filter_mode=cfg.offline_filter_mode,
		device=f'cuda:{offline_gpu}',
	)
	cfg = _prepare_cfg_from_dataset(cfg, dataset)
	logger = Logger(cfg)
	if logger.rank == 0:
		print(colored('Offline dataset stats:', 'yellow', attrs=['bold']), asdict(dataset.stats))

	agent = make_agent(cfg)
	if cfg.checkpoint:
		if not os.path.exists(cfg.checkpoint):
			raise FileNotFoundError(f'Checkpoint file not found: {cfg.checkpoint}')
		agent.load(cfg.checkpoint)
		print(colored(f'Loaded checkpoint from {cfg.checkpoint}.', 'blue', attrs=['bold']))

	# Stage 1: BC-only sanity check.
	_stage_cfg(agent, consistency=0.0, reward=0.0, value=0.0, prior=1.0, maxq_pi=False)
	_run_stage(agent, dataset, logger, cfg, stage_name='bc', num_steps=cfg.offline_bc_steps)

	# Stage 2: state-only WM pretraining with BC retained.
	_stage_cfg(
		agent,
		consistency=cfg.consistency_coef,
		reward=cfg.reward_coef,
		value=cfg.value_coef,
		prior=cfg.prior_coef,
		maxq_pi=False,
	)
	_run_stage(agent, dataset, logger, cfg, stage_name='wm', num_steps=cfg.offline_wm_steps)

	logger.finish(agent)
	print(colored('Offline training completed successfully.', 'green', attrs=['bold']))


if __name__ == '__main__':
	launch()
