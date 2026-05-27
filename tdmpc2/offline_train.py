import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ["TORCH_DISTRIBUTED_TIMEOUT"] = "1800"
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
os.environ['TORCH_LOGS'] = "+recompiles"
import warnings
warnings.filterwarnings('ignore')
import csv
import json
import math
from dataclasses import asdict
from copy import deepcopy
from pathlib import Path

import hydra
import torch
from hydra.core.config_store import ConfigStore
from termcolor import colored

from common import set_seed
from common.logger import Logger
from common.multitask_replay import (
	MultiTaskReplayPool,
	OfflineTaskReplayBuffer,
	get_active_tasks,
)
from common.world_model import WorldModel
from config import Config, parse_cfg
from offline_dataset import OfflineSequenceDataset
from offline_io import export_compact_dataset, export_multitask_compact_dataset, load_offline_manifest
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
		for task_id, count in sorted(getattr(dataset, "last_sample_task_counts", {}).items()):
			metrics[f"sample_task_frac/{stage_name}/{task_id}"] = float(count) / max(1, int(cfg.batch_size))
		metrics.update({
			'iteration': update,
			'step': update,
			'stage': stage_name,
		})
		if update == 1 or update % log_freq == 0 or update == num_steps:
			logger.log(metrics, 'pretrain')
		if logger.rank == 0 and (update % save_freq == 0 or update == num_steps):
			logger.save_agent(agent, f'{stage_name}_{update:,}'.replace(',', '_'), metrics=metrics)


def _make_offline_dataset(cfg, offline_dataset_fp, *, filter_mode: str, device: str):
	return OfflineSequenceDataset(
		path=offline_dataset_fp,
		batch_size=cfg.batch_size,
		horizon=cfg.horizon,
		filter_mode=filter_mode,
		task_balanced_sampling=bool(cfg.get('task_balanced_sampling', True)),
		high_depth_threshold=float(cfg.get('offline_high_depth_threshold', 0.75)),
		high_depth_lateral_tol_m=float(cfg.get('offline_high_depth_lateral_tol_m', 0.0020)),
		device=device,
	)


def _coerce_str_list(raw) -> list[str]:
	if raw is None:
		return []
	if isinstance(raw, str):
		text = raw.strip().strip("'\"")
		if not text:
			return []
		try:
			parsed = json.loads(text)
			if isinstance(parsed, list):
				raw = parsed
			else:
				raw = [parsed]
		except json.JSONDecodeError:
			if text.startswith("[") and text.endswith("]"):
				text = text[1:-1]
			raw = [item for item in text.replace(";", ",").replace(" ", ",").split(",") if item.strip()]
	elif isinstance(raw, (list, tuple)):
		raw = list(raw)
	else:
		raw = [raw]
	return [str(item).strip().strip("'\"") for item in raw if str(item).strip().strip("'\"")]


def _coerce_weight_dict(raw) -> dict[str, float]:
	if raw is None:
		return {}
	if isinstance(raw, dict):
		return {str(key).strip().strip("'\""): float(value) for key, value in raw.items()}
	if isinstance(raw, str):
		text = raw.strip()
		if not text:
			return {}
		try:
			parsed = json.loads(text)
			if isinstance(parsed, dict):
				return {str(key).strip().strip("'\""): float(value) for key, value in parsed.items()}
		except json.JSONDecodeError:
			pass
		out = {}
		for item in text.strip("{}").replace(";", ",").split(","):
			if not item.strip():
				continue
			if ":" not in item:
				raise ValueError(f"Invalid multitask weight item {item!r}; expected task_id:weight.")
			key, value = item.split(":", 1)
			out[key.strip().strip("'\"")] = float(value)
		return out
	raise TypeError(f"Unsupported multitask_task_sampling_weights type: {type(raw)}")


def _normalize_assembly_like(value) -> str:
	text = str(value).strip().strip("'\"")
	if text.isdigit() and len(text) < 5:
		return text.zfill(5)
	return text


def _entry_label(entry: dict) -> str:
	assembly_id = entry.get("assembly_id", None)
	if assembly_id is not None:
		return _normalize_assembly_like(assembly_id)
	return str(entry["task_id"])


def _entry_aliases(entry: dict) -> list[str]:
	aliases = {str(entry["task_id"]), _entry_label(entry)}
	task_name = entry.get("task_name", None)
	if task_name is not None:
		aliases.add(str(task_name))
	assembly_id = entry.get("assembly_id", None)
	if assembly_id is not None:
		aliases.add(str(assembly_id).strip().strip("'\""))
		aliases.add(_normalize_assembly_like(assembly_id))
	return [alias for alias in aliases if alias]


def _resolve_alias(raw_id, alias_map: dict[str, dict], *, field_name: str) -> dict:
	text = str(raw_id).strip().strip("'\"")
	candidates = [text]
	if text.isdigit():
		candidates.append(_normalize_assembly_like(text))
	for candidate in candidates:
		if candidate in alias_map:
			return alias_map[candidate]
	raise ValueError(
		f"{field_name}={raw_id!r} does not match any task in the offline manifest/dataset. "
		f"Available ids: {sorted(alias_map.keys())}"
	)


def _multitask_manifest_entries(cfg, dataset: OfflineSequenceDataset) -> tuple[list[dict], dict[str, dict]]:
	if cfg.get('offline_manifest_fp', None):
		entries = [dict(entry) for entry in load_offline_manifest(cfg.offline_manifest_fp)]
	else:
		entries = []
		for task_id in dataset.task_ids:
			task_name = cfg.global_tasks[task_id] if 0 <= task_id < len(cfg.global_tasks) else f"task_{task_id}"
			entries.append({
				"task_id": int(task_id),
				"task_name": task_name,
			})
	available_task_ids = set(int(task_id) for task_id in dataset.task_ids)
	entries = [entry for entry in entries if int(entry["task_id"]) in available_task_ids]
	if not entries:
		raise ValueError("No offline manifest entries match tasks available in the compact dataset.")
	alias_map = {}
	for entry in entries:
		entry["_multitask_label"] = _entry_label(entry)
		for alias in _entry_aliases(entry):
			alias_map.setdefault(alias, entry)
	return entries, alias_map


def _resolve_multitask_training_entries(cfg, dataset: OfflineSequenceDataset) -> list[dict]:
	entries, alias_map = _multitask_manifest_entries(cfg, dataset)
	requested = _coerce_str_list(cfg.get('multitask_task_ids', []))
	if not requested:
		selected = entries
	else:
		selected = [_resolve_alias(task_id, alias_map, field_name="multitask_task_ids") for task_id in requested]
	deduped = []
	seen = set()
	for entry in selected:
		label = entry["_multitask_label"]
		if label in seen:
			continue
		seen.add(label)
		deduped.append(entry)
	anchor_entry = _resolve_alias(
		cfg.get('multitask_anchor_task_id', deduped[0]["_multitask_label"]),
		alias_map,
		field_name="multitask_anchor_task_id",
	)
	anchor_label = anchor_entry["_multitask_label"]
	if anchor_label not in {entry["_multitask_label"] for entry in deduped}:
		raise ValueError(
			f"multitask_anchor_task_id={cfg.multitask_anchor_task_id!r} must be included in multitask_task_ids."
		)
	cfg.multitask_task_ids = [entry["_multitask_label"] for entry in deduped]
	cfg.multitask_anchor_task_id = anchor_label

	eval_ids = _coerce_str_list(cfg.get('multitask_eval_task_ids', []))
	if eval_ids:
		cfg.multitask_eval_task_ids = [
			_resolve_alias(task_id, alias_map, field_name="multitask_eval_task_ids")["_multitask_label"]
			for task_id in eval_ids
		]
	else:
		cfg.multitask_eval_task_ids = list(cfg.multitask_task_ids)
	return deduped


def _normalized_weight_dict(cfg, alias_map: dict[str, dict]) -> dict[str, float]:
	weights = _coerce_weight_dict(cfg.get('multitask_task_sampling_weights', None))
	out = {}
	for raw_key, value in weights.items():
		entry = _resolve_alias(raw_key, alias_map, field_name="multitask_task_sampling_weights")
		out[entry["_multitask_label"]] = float(value)
	return out


def _make_hard_case_dataset(cfg, offline_dataset_fp, *, device: str):
	if float(cfg.get('multitask_hard_case_ratio', 0.0)) <= 0.0:
		return None
	try:
		return _make_offline_dataset(
			cfg,
			offline_dataset_fp,
			filter_mode="failure_only",
			device=device,
		)
	except Exception as exc:
		print(colored(
			f"Hard-case replay disabled because no failure-only dataset could be built: {exc}",
			"yellow",
			attrs=["bold"],
		))
		return None


def _make_multitask_pool(cfg, dataset: OfflineSequenceDataset, hard_dataset: OfflineSequenceDataset | None):
	training_entries = _resolve_multitask_training_entries(cfg, dataset)
	_entries, alias_map = _multitask_manifest_entries(cfg, dataset)
	weights = _normalized_weight_dict(cfg, alias_map)
	pool = MultiTaskReplayPool(
		task_ids=cfg.multitask_task_ids,
		anchor_task_id=cfg.multitask_anchor_task_id,
		sampling_mode=cfg.multitask_sampling_mode,
		task_sampling_weights=weights,
		anchor_min_ratio=cfg.multitask_anchor_min_ratio,
		new_task_min_ratio=cfg.multitask_new_task_min_ratio,
		hard_case_ratio=cfg.multitask_hard_case_ratio,
		batch_size=cfg.batch_size,
	)
	for entry in training_entries:
		internal_task_id = int(entry["task_id"])
		label = entry["_multitask_label"]
		task_vec = cfg.task_vectors[internal_task_id]
		task_buffer = OfflineTaskReplayBuffer(dataset, internal_task_id, task_vec, task_label=label)
		hard_buffer = None
		if hard_dataset is not None and internal_task_id in set(hard_dataset.task_ids):
			hard_buffer = OfflineTaskReplayBuffer(hard_dataset, internal_task_id, task_vec, task_label=label)
		pool.add_task_buffer(label, task_buffer, hard_case_buffer=hard_buffer)
	return pool


def _mean(values):
	values = [float(value) for value in values if value is not None and not math.isnan(float(value))]
	return float(sum(values) / max(1, len(values)))


def _write_multitask_eval_records(cfg, *, step: int, results: list[dict], aggregate: dict):
	if not bool(cfg.get('multitask_save_per_task_metrics', True)) or cfg.rank != 0:
		return
	work_dir = Path(cfg.work_dir)
	csv_fp = work_dir / "multitask_eval.csv"
	jsonl_fp = work_dir / "multitask_eval.jsonl"
	csv_fp.parent.mkdir(parents=True, exist_ok=True)
	fieldnames = [
		"global_step",
		"task_id",
		"assembly_id",
		"success_rate",
		"mean_return",
		"mean_episode_length",
		"mean_progress",
		"jam_rate",
		"mean_force",
		"max_force",
		"collapse_error_optional",
	]
	write_header = not csv_fp.exists()
	with open(csv_fp, "a", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		if write_header:
			writer.writeheader()
		for item in results:
			task_label = item.get("assembly_id", item.get("task_id"))
			row = {
				"global_step": int(step),
				"task_id": task_label,
				"assembly_id": item.get("assembly_id", None),
				"success_rate": item.get("episode_success", None),
				"mean_return": item.get("episode_reward", None),
				"mean_episode_length": item.get("episode_length", None),
				"mean_progress": item.get("episode_depth_fraction", None),
				"jam_rate": item.get("episode_jam", None),
				"mean_force": item.get("episode_mean_force", None),
				"max_force": item.get("episode_max_force", None),
				"collapse_error_optional": item.get("collapse_error_optional", None),
			}
			writer.writerow(row)
	with open(jsonl_fp, "a", encoding="utf-8") as f:
		for item in results:
			record = dict(item)
			record["global_step"] = int(step)
			record["record_type"] = "task"
			f.write(json.dumps(record, ensure_ascii=True) + "\n")
		aggregate_record = dict(aggregate)
		aggregate_record["global_step"] = int(step)
		aggregate_record["record_type"] = "aggregate"
		f.write(json.dumps(aggregate_record, ensure_ascii=True) + "\n")


def _multitask_eval_metrics(cfg, results: list[dict], eval_state: dict):
	by_label = {}
	for item in results:
		label = item.get("assembly_id", item.get("task_id"))
		label = _normalize_assembly_like(label) if label is not None else str(item.get("task_id"))
		by_label[label] = float(item.get("episode_success", 0.0))
	metrics = {
		f"{task_id}_success": by_label.get(task_id, 0.0)
		for task_id in cfg.multitask_eval_task_ids
	}
	family_avg = _mean(list(metrics.values()))
	anchor_success = by_label.get(cfg.multitask_anchor_task_id, 0.0)
	main_task_id = cfg.multitask_task_ids[-1] if cfg.multitask_task_ids else cfg.multitask_anchor_task_id
	main_task_success = by_label.get(main_task_id, 0.0)
	if eval_state.get("anchor_success_initial", None) is None:
		eval_state["anchor_success_initial"] = anchor_success
	forgetting = float(eval_state["anchor_success_initial"]) - anchor_success
	metrics.update({
		"family_avg_success": family_avg,
		"anchor_success": anchor_success,
		"main_task_success": main_task_success,
		f"forgetting_{cfg.multitask_anchor_task_id}": forgetting,
		"step": eval_state.get("step", 0),
	})
	return metrics


def _maybe_save_multitask_bests(agent: TDMPC2, logger: Logger, cfg, metrics: dict, eval_state: dict, *, step: int):
	if logger.rank != 0:
		return
	family_avg = float(metrics.get("family_avg_success", float("nan")))
	main_success = float(metrics.get("main_task_success", float("nan")))
	forgetting = float(metrics.get(f"forgetting_{cfg.multitask_anchor_task_id}", float("inf")))
	if not math.isnan(family_avg) and family_avg > eval_state.get("best_family_avg", -float("inf")):
		eval_state["best_family_avg"] = family_avg
		logger.save_agent(agent, "best_family_avg", metrics={"step": step, "family_avg_success": family_avg})
	if not math.isnan(main_success) and main_success > eval_state.get("best_main_task", -float("inf")):
		eval_state["best_main_task"] = main_success
		logger.save_agent(agent, "best_main_task", metrics={"step": step, "main_task_success": main_success})
	if (
		not math.isnan(family_avg) and
		forgetting <= float(cfg.get("multitask_no_forgetting_max_forgetting", 0.05)) and
		family_avg > eval_state.get("best_no_forgetting", -float("inf"))
	):
		eval_state["best_no_forgetting"] = family_avg
		logger.save_agent(
			agent,
			"best_no_forgetting",
			metrics={
				"step": step,
				"family_avg_success": family_avg,
				f"forgetting_{cfg.multitask_anchor_task_id}": forgetting,
			},
		)


def evaluate_all_tasks(agent: TDMPC2, cfg, logger: Logger, *, step: int, eval_state: dict):
	"""Evaluate one shared checkpoint on every configured multitask eval task."""
	if logger.rank != 0:
		return {}
	save_metrics = {"step": int(step)}
	ckpt_fp = logger.save_latest_agent(agent, save_metrics)
	if ckpt_fp is None:
		return {}
	from batch_eval_tasks import _evaluate_one, _resolve_eval_entries, _write_summary

	eval_cfg = deepcopy(cfg)
	eval_cfg.checkpoint = str(ckpt_fp)
	eval_cfg.eval_assembly_ids = list(cfg.multitask_eval_task_ids)
	eval_cfg.batch_eval_spawn_per_assembly = False
	eval_cfg.batch_eval_output_dir = str(Path(cfg.work_dir) / "multitask_eval" / f"step_{int(step):08d}")
	eval_cfg.batch_eval_summary_fp = str(Path(eval_cfg.batch_eval_output_dir) / "multitask_eval_summary.json")
	eval_cfg.enable_wandb = False
	eval_cfg.save_agent = False
	eval_cfg.multiproc = False
	eval_cfg.rank = 0
	eval_cfg.world_size = 1
	eval_cfg.device_id = int(eval_cfg.gpu_id)
	entries = _resolve_eval_entries(eval_cfg)
	output_dir = Path(eval_cfg.batch_eval_output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	results = [_evaluate_one(eval_cfg, entry, output_dir) for entry in entries]
	summary, csv_fp = _write_summary(Path(eval_cfg.batch_eval_summary_fp), results)
	eval_state["step"] = int(step)
	metrics = _multitask_eval_metrics(cfg, results, eval_state)
	metrics.update({
		"episode_success": summary.get("episode_success", metrics["family_avg_success"]),
		"episode_reward": summary.get("episode_reward", 0.0),
		"episode_length": summary.get("episode_length", 0.0),
		"multitask_eval_summary_fp": str(eval_cfg.batch_eval_summary_fp),
		"multitask_eval_csv_fp": str(csv_fp),
	})
	_write_multitask_eval_records(cfg, step=step, results=results, aggregate=metrics)
	logger.log(metrics, "eval")
	_maybe_save_multitask_bests(agent, logger, cfg, metrics, eval_state, step=step)
	return metrics


def _multitask_train_metrics(pool: MultiTaskReplayPool, cfg, update: int, active_tasks: list[str], agent_metrics: dict):
	metrics = dict(agent_metrics)
	for task_id, ratio in sorted(pool.last_batch_task_distribution.items()):
		metrics[f"batch_ratio_{task_id}"] = float(ratio)
	for task_id, stats in sorted(pool.last_task_vec_stats.items()):
		for index, value in enumerate(stats.get("mean", [])):
			metrics[f"task_vec_mean_{task_id}_{index}"] = float(value)
		for index, value in enumerate(stats.get("std", [])):
			metrics[f"task_vec_std_{task_id}_{index}"] = float(value)
	metrics.update({
		"iteration": int(update),
		"step": int(update),
		"stage": "multitask_continuation",
		"active_task_count": len(active_tasks),
		"active_tasks": ",".join(active_tasks),
		"batch_task_distribution": dict(pool.last_batch_task_distribution),
	})
	return metrics


def _run_multitask_continuation(
	agent: TDMPC2,
	dataset: OfflineSequenceDataset,
	hard_dataset: OfflineSequenceDataset | None,
	logger: Logger,
	cfg,
):
	if not cfg.get('checkpoint', None):
		raise ValueError("multitask_continuation_enabled=true requires checkpoint=<01125 warm-start checkpoint>.")
	pool = _make_multitask_pool(cfg, dataset, hard_dataset)
	if bool(cfg.get('multitask_prox_reg_enabled', False)):
		agent.set_proximal_reference(module_filter=("_encoder", "_dynamics"))
	total_steps = int(cfg.get('multitask_total_steps', None) or cfg.get('steps', 0))
	if total_steps <= 0:
		raise ValueError(f"Expected positive multitask total steps, got {total_steps}.")
	log_freq = max(1, int(cfg.offline_log_freq))
	save_freq = max(1, int(cfg.offline_save_freq))
	eval_enabled = bool(cfg.get('multitask_eval_enabled', True))
	eval_interval = int(cfg.get('multitask_eval_interval', 0) or 0) if eval_enabled else 0
	if eval_enabled and eval_interval <= 0:
		eval_interval = max(1, min(50_000, total_steps))
		if logger.rank == 0:
			print(colored(
				f"multitask_eval_enabled=true but multitask_eval_interval<=0; "
				f"using eval_interval={eval_interval}. Set multitask_eval_enabled=false to disable success eval.",
				"yellow",
				attrs=["bold"],
			))
	eval_state = {
		"anchor_success_initial": None,
		"best_family_avg": -float("inf"),
		"best_main_task": -float("inf"),
		"best_no_forgetting": -float("inf"),
	}
	if logger.rank == 0:
		print(colored(
			"Starting Scheme A multitask continuation: "
			f"tasks={cfg.multitask_task_ids}, anchor={cfg.multitask_anchor_task_id}, "
			f"mode={cfg.multitask_curriculum_mode}, stage_steps={cfg.multitask_stage_steps}, "
			f"total_updates={total_steps:,}.",
			"cyan",
			attrs=["bold"],
		))
		print(colored("Only one shared task-conditioned model will be updated.", "cyan", attrs=["bold"]))
	last_active_tasks = None
	if eval_interval > 0 and bool(cfg.get('multitask_forgetting_metric_enabled', True)):
		evaluate_all_tasks(agent, cfg, logger, step=0, eval_state=eval_state)
	for update in range(1, total_steps + 1):
		active_tasks = get_active_tasks(
			update - 1,
			cfg.multitask_task_ids,
			cfg.multitask_stage_steps,
			cfg.multitask_curriculum_mode,
		)
		current_new_task_id = active_tasks[-1] if active_tasks else None
		pool.set_active_tasks(active_tasks, current_new_task_id=current_new_task_id)
		cfg.multitask_current_active_tasks = list(active_tasks)
		agent.cfg.multitask_current_active_tasks = list(active_tasks)
		active_changed = active_tasks != last_active_tasks
		if logger.rank == 0 and active_changed:
			print(colored(f"active_tasks = {active_tasks}", "cyan", attrs=["bold"]))
			last_active_tasks = list(active_tasks)
		agent_metrics = dict(agent.update(pool).items())
		metrics = _multitask_train_metrics(pool, cfg, update, active_tasks, agent_metrics)
		if logger.rank == 0 and (active_changed or update == 1 or update % log_freq == 0 or update == total_steps):
			print(colored(
				f"batch_task_distribution = {pool.last_batch_task_distribution}",
				"cyan",
				attrs=["bold"],
			))
			logger.log(metrics, "train")
		if logger.rank == 0 and (update % save_freq == 0 or update == total_steps):
			logger.save_latest_agent(agent, {"step": update, "family_avg_success": None})
		if eval_interval > 0 and (update % eval_interval == 0 or update == total_steps):
			evaluate_all_tasks(agent, cfg, logger, step=update, eval_state=eval_state)
	logger.finish(agent)


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
	offline_device = f'cuda:{offline_gpu}'

	if cfg.get('multitask_continuation_enabled', False):
		dataset = _make_offline_dataset(
			cfg,
			offline_dataset_fp,
			filter_mode=cfg.get('offline_filter_mode', 'all'),
			device=offline_device,
		)
		hard_dataset = _make_hard_case_dataset(cfg, offline_dataset_fp, device=offline_device)
		cfg = _prepare_cfg_from_dataset(cfg, dataset)
		logger = Logger(cfg)
		if logger.rank == 0:
			print(colored('Multitask continuation dataset stats:', 'yellow', attrs=['bold']), asdict(dataset.stats))
			if hard_dataset is not None:
				print(colored('Hard-case dataset stats:', 'yellow', attrs=['bold']), asdict(hard_dataset.stats))
		agent = make_agent(cfg)
		if not cfg.checkpoint:
			raise ValueError('multitask_continuation_enabled=true requires `checkpoint` for the 01125 warm start.')
		if not os.path.exists(cfg.checkpoint):
			raise FileNotFoundError(f'Checkpoint file not found: {cfg.checkpoint}')
		agent.load(cfg.checkpoint)
		print(colored(f'Loaded shared warm-start checkpoint from {cfg.checkpoint}.', 'blue', attrs=['bold']))
		_run_multitask_continuation(agent, dataset, hard_dataset, logger, cfg)
		print(colored('Multitask continuation completed successfully.', 'green', attrs=['bold']))
		return

	wm_dataset = _make_offline_dataset(
		cfg,
		offline_dataset_fp,
		filter_mode=cfg.get('offline_wm_filter_mode', cfg.offline_filter_mode),
		device=offline_device,
	)
	bc_dataset = _make_offline_dataset(
		cfg,
		offline_dataset_fp,
		filter_mode=cfg.get('offline_bc_filter_mode', cfg.offline_filter_mode),
		device=offline_device,
	)
	rl_dataset = _make_offline_dataset(
		cfg,
		offline_dataset_fp,
		filter_mode=cfg.get('offline_rl_filter_mode', cfg.offline_filter_mode),
		device=offline_device,
	)
	cfg = _prepare_cfg_from_dataset(cfg, wm_dataset)
	logger = Logger(cfg)
	if logger.rank == 0:
		print(colored('Offline WM dataset stats:', 'yellow', attrs=['bold']), asdict(wm_dataset.stats))
		print(colored('Offline BC dataset stats:', 'yellow', attrs=['bold']), asdict(bc_dataset.stats))
		print(colored('Offline RL dataset stats:', 'yellow', attrs=['bold']), asdict(rl_dataset.stats))

	agent = make_agent(cfg)
	if cfg.checkpoint:
		if not os.path.exists(cfg.checkpoint):
			raise FileNotFoundError(f'Checkpoint file not found: {cfg.checkpoint}')
		agent.load(cfg.checkpoint)
		print(colored(f'Loaded checkpoint from {cfg.checkpoint}.', 'blue', attrs=['bold']))

	# Stage 1: BC-only sanity check.
	_stage_cfg(agent, consistency=0.0, reward=0.0, value=0.0, prior=1.0, maxq_pi=False)
	_run_stage(agent, bc_dataset, logger, cfg, stage_name='bc', num_steps=cfg.offline_bc_steps)

	# Stage 2: state-only WM pretraining with BC retained.
	_stage_cfg(
		agent,
		consistency=cfg.consistency_coef,
		reward=cfg.reward_coef,
		value=cfg.value_coef,
		prior=cfg.prior_coef,
		maxq_pi=False,
	)
	_run_stage(agent, wm_dataset, logger, cfg, stage_name='wm', num_steps=cfg.offline_wm_steps)

	# Stage 3: offline Max-Q fine-tuning with the learned world model and behavior prior.
	_stage_cfg(
		agent,
		consistency=cfg.consistency_coef,
		reward=cfg.reward_coef,
		value=cfg.value_coef,
		prior=cfg.prior_coef,
		maxq_pi=True,
	)
	_run_stage(agent, rl_dataset, logger, cfg, stage_name='rl', num_steps=cfg.offline_rl_steps)

	logger.finish(agent)
	print(colored('Offline training completed successfully.', 'green', attrs=['bold']))


if __name__ == '__main__':
	launch()
