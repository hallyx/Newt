import os
import datetime
import re
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from termcolor import colored

from common import TASK_SET


CONSOLE_FORMAT = [
	("iteration", "I", "int"),
	("episode", "E", "int"),
	("step", "I", "int"),
	("episode_reward", "R", "float"),
	("episode_score", "S", "float"),
	("elapsed_time", "T", "time"),
]

CAT_TO_COLOR = {
	"pretrain": "yellow",
	"train": "blue",
	"eval": "green",
}


def make_dir(dir_path):
	"""Create directory if it does not already exist."""
	try:
		os.makedirs(dir_path)
	except OSError:
		pass
	return dir_path


def print_run(cfg):
	"""
	Pretty-printing of current run information.
	Logger calls this method at initialization.
	"""
	prefix, color, attrs = "  ", "green", ["bold"]

	def _limstr(s, maxlen=36):
		return str(s[:maxlen]) + "..." if len(str(s)) > maxlen else s

	def _pprint(k, v):
		print(
			prefix + colored(f'{k.capitalize()+":":<15}', color, attrs=attrs), _limstr(v)
		)

	observations  = ", ".join([str(v) for v in cfg.obs_shape.values()])
	kvs = [
		("task", cfg.task_title),
		("envs", cfg.num_envs*cfg.world_size),
		("steps", f"{int(cfg.steps):,}"),
		("observations", observations),
		("actions", cfg.action_dim),
		("experiment", cfg.exp_name),
	]
	if cfg.task == "soup":
		kvs[0] = ("tasks", cfg.num_global_tasks)
		kvs[1] = ("world size", cfg.world_size)
	w = np.max([len(_limstr(str(kv[1]))) for kv in kvs]) + 25
	div = "-" * w
	print(div)
	for k, v in kvs:
		_pprint(k, v)
	print(div)


def cfg_to_group(cfg, return_list=False):
	"""
	Return a wandb-safe group name for logging.
	Optionally returns group name as list.
	"""
	lst = [cfg.task, re.sub("[^0-9a-zA-Z]+", "-", cfg.exp_name)]
	return lst if return_list else "-".join(lst)


def _safe_token(value, fallback="na"):
	value = str(value if value is not None else fallback).strip()
	value = re.sub(r"[^0-9a-zA-Z._-]+", "-", value)
	value = value.strip("-_.")
	return value or fallback


def _format_metric_token(value):
	if isinstance(value, torch.Tensor):
		value = value.item() if value.numel() == 1 else float("nan")
	elif isinstance(value, np.generic):
		value = value.item()
	try:
		value = float(value)
	except (TypeError, ValueError):
		return _safe_token(value)
	if np.isnan(value):
		return "nan"
	return f"{value:.4f}".replace(".", "p").replace("-", "m")


class VideoRecorder:
	"""Utility class for logging evaluation videos."""

	def __init__(self, cfg, wandb, fps=15):
		self.cfg = cfg
		self._save_dir = make_dir(Path(cfg.work_dir) / 'eval_video')
		self._wandb = wandb
		self.fps = fps
		self.frames = []
		self.enabled = False

	def init(self, env, enabled=True):
		self.frames = []
		self.enabled = self._save_dir and self._wandb and enabled
		self.record(env)

	def record(self, env):
		if self.enabled:
			self.frames.append(env.render())

	def save(self, step, key='videos/eval_video'):
		if self.enabled and len(self.frames) > 1:
			frames = np.stack(self.frames[:-1])
			return self._wandb.log(
				{key: self._wandb.Video(frames.transpose(0, 3, 1, 2), fps=self.fps, format='mp4')}, step=step
			)


class Logger:
	"""Primary logging object. Logs either locally or using wandb."""

	def __init__(self, cfg):
		self.rank = cfg.rank
		self.project = self._resolve_project(cfg)
		self.entity = self._resolve_entity(cfg)
		self._log_dir = Path(make_dir(cfg.work_dir))
		self._model_dir = make_dir(self._log_dir / "models")
		self._local_log_fp = self._log_dir / "metrics.jsonl"
		self._save_agent = cfg.save_agent and self.rank == 0
		self._group = cfg_to_group(cfg)
		self._seed = cfg.seed
		self._run_id = cfg.get("run_id", None)
		self._eval = []
		self._wandb = None
		self._video = None
		self._best_metric_name = cfg.get("save_best_metric", "episode_success")
		self._best_metric_value = None
		self._best_step = None

		if self.rank > 0:
			print(colored(f"Logging disabled for rank {self.rank}.", "blue", attrs=["bold"]))
			cfg.save_video = False
			return

		self._write_run_metadata(cfg)
		print_run(cfg)
		if not cfg.enable_wandb or self.project is None:
			print(colored("Wandb disabled. Using local logging only.", "blue", attrs=["bold"]))
			cfg.save_video = False
			return

		os.environ["WANDB_SILENT"] = "true" if cfg.wandb_silent else "false"
		try:
			import wandb
			wandb.init(
				project=self.project,
				entity=self.entity,
				name=str(self._run_id or cfg.seed),
				group=self._group,
				tags=cfg_to_group(cfg, return_list=True) + [f"seed:{cfg.seed}"],
				dir=self._log_dir,
				config=cfg,
			)
			dest = self.project if self.entity is None else f"{self.entity}/{self.project}"
			print(colored(f"Logs will be synced with wandb: {dest}.", "blue", attrs=["bold"]))
			self._wandb = wandb
			self._video = VideoRecorder(cfg, self._wandb) if cfg.save_video else None
		except Exception as exc:
			cfg.save_video = False
			print(colored(f"Wandb init failed; continuing with local logging only. {exc}", "yellow", attrs=["bold"]))

	@staticmethod
	def _normalize_wandb_field(value):
		if value is None:
			return None
		value = str(value).strip()
		if value.lower() in {"", "none", "null", "entity", "project"}:
			return None
		return value

	def _resolve_project(self, cfg):
		project = self._normalize_wandb_field(os.getenv("WANDB_PROJECT"))
		if project is not None:
			return project
		project = self._normalize_wandb_field(cfg.get("wandb_project", None))
		if project is not None:
			return project
		return re.sub("[^0-9a-zA-Z._-]+", "-", cfg.task)

	def _resolve_entity(self, cfg):
		entity = self._normalize_wandb_field(os.getenv("WANDB_ENTITY"))
		if entity is not None:
			return entity
		return self._normalize_wandb_field(cfg.get("wandb_entity", None))

	def _write_run_metadata(self, cfg):
		meta = {
			"run_id": cfg.get("run_id", None),
			"task": cfg.task,
			"exp_name": cfg.exp_name,
			"seed": int(cfg.seed),
			"assembly_id": cfg.get("assembly_id", None),
			"eval_task_id": cfg.get("eval_task_id", None),
			"offline_manifest_fp": cfg.get("offline_manifest_fp", None),
			"num_global_tasks": cfg.get("num_global_tasks", None),
			"work_dir": str(self._log_dir),
			"model_dir": str(self._model_dir),
			"metrics_fp": str(self._local_log_fp),
		}
		with open(self._log_dir / "run.json", "w", encoding="utf-8") as f:
			json.dump(meta, f, ensure_ascii=True, indent=2)

	def _checkpoint_filename(self, identifier, metrics=None):
		identifier = _safe_token(identifier)
		parts = [identifier]
		metrics = metrics or {}
		step = metrics.get("step", None)
		if step is not None and identifier in {"best", "final"}:
			try:
				parts.append(f"step-{int(step)}")
			except (TypeError, ValueError):
				pass
		for key in ("episode_success", self._best_metric_name, "success_rate"):
			if key in metrics:
				parts.append(f"s-{_format_metric_token(metrics[key])}")
				break
		return "_".join(parts) + ".pt"

	@property
	def video(self):
		return self._video

	def save_agent(self, agent=None, identifier='final', metrics=None, aliases=None):
		if self._save_agent and agent:
			fp = self._model_dir / self._checkpoint_filename(identifier, metrics)
			agent.save(fp)
			for alias in aliases or []:
				alias_fp = self._model_dir / f"{_safe_token(alias)}.pt"
				shutil.copy2(fp, alias_fp)
			if self._wandb:
				artifact = self._wandb.Artifact(
					self._group + '-' + str(self._seed) + '-' + fp.stem,
					type='model',
				)
				artifact.add_file(fp)
				self._wandb.log_artifact(artifact)
			print(colored(f"Saved checkpoint: {fp}", "green", attrs=["bold"]))
			return fp
		return None

	def maybe_save_best_agent(self, agent, metrics, step):
		if not self._save_agent or not agent:
			return False
		metric_name = self._best_metric_name
		if metric_name not in metrics:
			return False
		value = metrics[metric_name]
		if isinstance(value, torch.Tensor):
			value = value.item()
		elif isinstance(value, np.generic):
			value = value.item()
		value = float(value)
		if np.isnan(value):
			return False
		is_better = self._best_metric_value is None or value > self._best_metric_value
		if not is_better:
			return False
		self._best_metric_value = value
		self._best_step = int(step)
		save_metrics = dict(metrics)
		save_metrics['step'] = int(step)
		fp = self.save_agent(agent, 'best', metrics=save_metrics, aliases=['best'])
		meta = {
			'metric': metric_name,
			'value': value,
			'step': int(step),
			'checkpoint': str(fp) if fp is not None else None,
			'alias': str(self._model_dir / 'best.pt'),
		}
		with open(self._model_dir / 'best.json', 'w', encoding='utf-8') as f:
			json.dump(meta, f, ensure_ascii=True, indent=2)
		print(colored(
			f"Saved new best checkpoint: {metric_name}={value:.4f} at step {int(step):,}.",
			'green',
			attrs=['bold'],
		))
		return True

	def finish(self, agent=None):
		if agent is not None:
			self.save_agent(agent)
		if self._wandb:
			self._wandb.finish()

	def _to_serializable(self, value):
		if isinstance(value, torch.Tensor):
			if value.numel() == 1:
				return value.item()
			return value.detach().cpu().tolist()
		if isinstance(value, np.generic):
			return value.item()
		if isinstance(value, np.ndarray):
			return value.tolist()
		if isinstance(value, dict):
			return {k: self._to_serializable(v) for k, v in value.items()}
		if isinstance(value, (list, tuple)):
			return [self._to_serializable(v) for v in value]
		return value

	def _write_local(self, d, category):
		record = {"category": category}
		for k, v in d.items():
			record[k] = self._to_serializable(v)
		with open(self._local_log_fp, "a", encoding="utf-8") as f:
			f.write(json.dumps(record, ensure_ascii=True) + "\n")

	def _format(self, key, value, ty):
		if ty == "int":
			return f'{colored(key+":", "blue")} {int(value):,}'
		elif ty == "float":
			return f'{colored(key+":", "blue")} {value:.03f}'
		elif ty == "time":
			value = str(datetime.timedelta(seconds=int(value)))
			return f'{colored(key+":", "blue")} {value}'
		else:
			raise f"invalid log format type: {ty}"

	def _print(self, d, category):
		category = colored(category, CAT_TO_COLOR[category])
		pieces = [f" {category:<14}"]
		for k, disp_k, ty in CONSOLE_FORMAT:
			if k in d:
				pieces.append(f"{self._format(disp_k, d[k], ty):<22}")
		print("   ".join(pieces))

	def pprint_multitask(self, d, cfg):
		"""Pretty-print evaluation metrics for multi-task training."""
		if self.rank > 0:
			return
		print(colored(f'Evaluated agent on {cfg.num_global_tasks} tasks:', 'yellow', attrs=['bold']))
		scores = defaultdict(list)
		domains = [k for k in TASK_SET.keys() if k != 'soup']
		for k, v in d.items():
			if '+' not in k:
				continue
			task = k.split('+')[1]
			if k.startswith('episode_score'):
				for domain in domains:
					if task in TASK_SET[domain]:
						scores[f'avg_score_{domain}'].append(v)
						print(colored(f'  {task:<34}\tS: {v:.03f}', 'yellow'))
						break
				scores['avg_score'].append(v)

		# Normalized score
		for domain, score in scores.items():
			scores[domain] = np.mean(score) if len(score) > 0 else float('nan')
	
		# Print summary
		for domain, score in scores.items():
			if domain.startswith('avg_score_'):
				print(colored(f'{domain[10:]:<34}\tS: {score:.03f}', 'yellow', attrs=['bold']))
		print(colored(f'{"unweighted score":<34}\tS: {scores["avg_score"]:.03f}', 'yellow', attrs=['bold']))
		scores['avg_score_weighted'] = np.nanmean([scores[domain] for domain in scores if domain.startswith('avg_score_')])
		print(colored(f'{"weighted score":<34}\tS: {scores["avg_score_weighted"]:.03f}', 'yellow', attrs=['bold']))
		d.update(scores)

	def pprint_pretrain(self, d):
		if self.rank > 0:
			return
		print(colored('-'*30 + '\nPretraining metrics:', 'yellow', attrs=['bold']))
		for k, v in d.items():
			print(colored(f' {k:<22}{v:.05f}', 'yellow'))
		print(colored('-'*30, 'yellow'))

	def log(self, d, category="train"):
		if self.rank > 0:
			return
		assert category in CAT_TO_COLOR.keys(), f"invalid category: {category}"
		self._write_local(d, category)
		if self._wandb:
			_d = dict()
			for k, v in d.items():
				_d[category + "/" + k] = v
			self._wandb.log(_d, step=d["step"])
		if category in {'train', 'eval'}:
			self._print(d, category)
