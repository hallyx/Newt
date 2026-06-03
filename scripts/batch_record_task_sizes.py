#!/usr/bin/env python3
"""Record Newt eval videos across SRSA clearance/depth task sizes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


DEFAULT_PYTHON = "/home/gpuserver/miniconda3/envs/isaac51/bin/python"
DEFAULT_ISAACLAB_DIR = "/home/gpuserver/IsaacLab"
DEFAULT_SRSA_DIR = "/home/gpuserver/hx/github/srsa"
DEFAULT_SIZE_TEMPLATES = "0.5:0.5;0.5:1.0;1.0:1.0;2.0:1.5;4.0:2.0"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_python() -> str:
    env_python = os.environ.get("PYTHON")
    if env_python:
        return env_python
    if Path(DEFAULT_PYTHON).exists():
        return DEFAULT_PYTHON
    return sys.executable


def _parse_template_list(value: str) -> list[tuple[float, float]]:
    normalized = value.strip()
    for char in "()[]{}":
        normalized = normalized.replace(char, "")
    parts = [item.strip() for item in re.split(r"[,;\s]+", normalized) if item.strip()]
    templates: list[tuple[float, float]] = []
    for part in parts:
        pair = [item.strip() for item in re.split(r"[:/xX]", part) if item.strip()]
        if len(pair) != 2:
            raise ValueError(f"Invalid size template {part!r}; expected CLEARANCE_MULT:DEPTH_MULT.")
        templates.append((float(pair[0]), float(pair[1])))
    if not templates:
        raise ValueError("At least one size template is required.")
    return templates


def _safe_number(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value).strip("_")


def _template_key(clearance_mult: float, depth_mult: float) -> tuple[float, float]:
    return (round(float(clearance_mult), 12), round(float(depth_mult), 12))


def _resolve_repo_path(value: str, repo_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _load_template_id_map(template_fp: str, repo_root: Path) -> dict[tuple[float, float], int]:
    path = _resolve_repo_path(template_fp, repo_root)
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    templates = doc.get("parameter_templates", doc.get("templates", []))
    mapping: dict[tuple[float, float], int] = {}
    for item in templates:
        if not isinstance(item, dict):
            continue
        if "clearance_multiplier" not in item or "depth_multiplier" not in item:
            continue
        template_id = item.get("template_id", item.get("task_id", None))
        if template_id is None:
            continue
        key = _template_key(item["clearance_multiplier"], item["depth_multiplier"])
        mapping[key] = int(template_id)
    return mapping


def _gpu_id_from_device(device: str) -> int:
    match = re.fullmatch(r"cuda:(\d+)", str(device).strip())
    if match:
        return int(match.group(1))
    if str(device).strip().isdigit():
        return int(str(device).strip())
    raise ValueError(f"Unsupported device {device!r}; use cuda:N or pass --gpu_id.")


def _hydra_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value)
    if any(char in text for char in [",", " ", ";", ":", "[", "]"]):
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return text


def _override(key: str, value) -> str:
    return f"{key}={_hydra_value(value)}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record videos with Newt tdmpc2/eval.py over SRSA task-size templates.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Example:\n"
            "  scripts/batch_record_task_sizes.py \\\n"
            "    --assembly_id 00783 \\\n"
            "    --checkpoint logs/isaaclab-srsa-assembly/1/.../models/best.pt \\\n"
            "    --device cuda:0 \\\n"
            "    --video_length 300 \\\n"
            "    --templates \"0.5:0.5;0.5:1.0;1.0:1.0;2.0:1.5;4.0:2.0\"\n\n"
            "Extra arguments like model_size=S or contact_history_enabled=false are forwarded as Hydra overrides."
        ),
    )
    parser.add_argument("--assembly_id", required=True, help="SRSA assembly id used for every recording.")
    parser.add_argument("--checkpoint", required=True, help="Newt checkpoint used by tdmpc2/eval.py.")
    parser.add_argument("--templates", default=DEFAULT_SIZE_TEMPLATES, help="CLEARANCE_MULT:DEPTH_MULT pairs.")
    parser.add_argument("--python", default=_default_python(), help="Python executable for Newt eval.")
    parser.add_argument("--isaaclab_dir", default=os.environ.get("ISAACLAB_DIR", DEFAULT_ISAACLAB_DIR))
    parser.add_argument("--srsa_dir", default=os.environ.get("SRSA_DIR", DEFAULT_SRSA_DIR))
    parser.add_argument("--device", default="cuda:0", help="CUDA device, e.g. cuda:0.")
    parser.add_argument("--gpu_id", type=int, default=None, help="Override GPU id instead of parsing --device.")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--eval_trials", type=int, default=1)
    parser.add_argument("--video_length", type=int, default=300, help="Maximum recorded frames per template.")
    parser.add_argument("--video_dir", default="data/video_eval_task_sizes", help="Video output directory.")
    parser.add_argument("--video_fps", type=int, default=15)
    parser.add_argument("--video_format", default="mp4")
    parser.add_argument("--video_record_every", type=int, default=1)
    parser.add_argument("--video_prefix", default=None, help="Filename prefix. Defaults to assembly_id.")
    parser.add_argument("--model_size", default="S")
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--srsa_task_template_fp", default="data/srsa_axial_task_templates.json")
    parser.add_argument("--srsa_mesh_geometry_fp", default="data/srsa_mesh_geometry_params.csv")
    parser.add_argument("--clearance_base", type=float, default=0.000114)
    parser.add_argument("--depth_base", type=float, default=0.015)
    parser.add_argument("--clearance_jitter_ratio", type=float, default=0.0)
    parser.add_argument("--depth_jitter_ratio", type=float, default=0.0)
    parser.add_argument("--success_metric", default="relaxed")
    parser.add_argument("--socket_camera_profile_fp", default="camera_profiles/socket_camera_offset.json")
    parser.add_argument("--socket_camera_env_index", type=int, default=0)
    parser.add_argument("--no_socket_camera_follow", action="store_true")
    parser.add_argument("--interactive", action="store_true", help="Use a visible Isaac window.")
    parser.add_argument("--no_contact_preset", action="store_true", help="Do not add 17D/contact-history defaults.")
    parser.add_argument("--dry_run", action="store_true", help="Print Newt eval commands without running them.")
    return parser


def _base_overrides(args: argparse.Namespace, gpu_id: int) -> list[str]:
    overrides = [
        _override("checkpoint", args.checkpoint),
        _override("eval_mode", "sim"),
        _override("isaaclab_backend", "srsa"),
        _override("task", "isaaclab-srsa-assembly"),
        _override("assembly_id", args.assembly_id),
        _override("isaaclab_dir", args.isaaclab_dir),
        _override("srsa_dir", args.srsa_dir),
        _override("srsa_task_template_fp", args.srsa_task_template_fp),
        _override("srsa_mesh_geometry_fp", args.srsa_mesh_geometry_fp),
        _override("eval_task_template_exact", True),
        _override("eval_task_template_print", True),
        _override("srsa_sparse_reward", False),
        _override("isaaclab_disable_imitation_reward", False),
        _override("srsa_align_direct_reward_success", True),
        _override("srsa_if_sbc", False),
        _override("num_envs", args.num_envs),
        _override("gpu_id", gpu_id),
        _override("multiproc", False),
        _override("model_size", args.model_size),
        _override("horizon", args.horizon),
        _override("compile", False),
        _override("mpc", True),
        _override("isaaclab_headless", not args.interactive),
        _override("isaaclab_use_canonical_obs", True),
        _override("srsa_task_family_name", "normal_fit"),
        _override("srsa_task_param_obs", False),
        _override("srsa_task_param_obs_mode", "task_vec"),
        _override("srsa_enable_axial_task_param_sampler", True),
        _override("srsa_axial_fixed_plug_scale", True),
        _override("srsa_axial_clearance_base", args.clearance_base),
        _override("srsa_axial_clearance_jitter_ratio", args.clearance_jitter_ratio),
        _override("srsa_axial_depth_base", args.depth_base),
        _override("srsa_axial_depth_jitter_ratio", args.depth_jitter_ratio),
        _override("srsa_axial_init_error_xy_range", "0.009,0.0010"),
        _override("srsa_axial_init_error_z_range", "0.0010,0.0020"),
        _override("srsa_axial_init_error_yaw_range", "-0.0872665,0.0872665"),
        _override("srsa_axial_visual_noise_xy_range", "0.0,0.0"),
        _override("srsa_axial_visual_noise_z_range", "0.0,0.0"),
        _override("srsa_vision_noise_xy_std", 0.0),
        _override("srsa_vision_noise_xy_jitter_std", 0.0),
        _override("srsa_vision_noise_z_std", 0.0),
        _override("srsa_vision_noise_z_jitter_std", 0.0),
        _override("isaaclab_canonical_use_visual_noise", False),
        _override("task_conditioning", "axial_params"),
        _override("eval_success_metric", args.success_metric),
        _override("srsa_eval_success_metric", args.success_metric),
        _override("eval_trials", args.eval_trials),
        _override("eval_terminate_on_success", True),
        _override("eval_terminate_success_key", "terminal_process_success"),
        _override("save_video", True),
        _override("eval_video_dir", args.video_dir),
        _override("eval_video_fps", args.video_fps),
        _override("eval_video_format", args.video_format),
        _override("eval_video_max_frames", args.video_length),
        _override("eval_video_max_episodes", 1),
        _override("eval_video_record_every", args.video_record_every),
        _override("eval_video_env_index", 0),
        _override("enable_wandb", False),
        _override("save_agent", False),
        _override("exp_name", "batch_record_task_sizes"),
        _override("seed", 1),
    ]
    if not args.no_socket_camera_follow:
        overrides.extend([
            _override("srsa_socket_camera_follow", True),
            _override("srsa_socket_camera_profile_fp", args.socket_camera_profile_fp),
            _override("srsa_socket_camera_env_index", args.socket_camera_env_index),
        ])
    if not args.no_contact_preset:
        overrides.extend([
            _override("srsa_enable_flange_force_sensor", True),
            _override("isaaclab_canonical_append_force", True),
            _override("isaaclab_canonical_append_task_params", False),
            _override("contact_history_enabled", True),
            _override("contact_history_len", 4),
            _override("contact_context_dim", 64),
            _override("contact_history_hidden_dim", 128),
            _override("contact_history_layers", 2),
            _override("contact_force_dim", 6),
            _override("contact_action_dim", 3),
            _override("contact_ee_delta_dim", 3),
            _override("contact_history_use_ee_delta", True),
        ])
    return overrides


def main() -> int:
    parser = _build_parser()
    args, extra_overrides = parser.parse_known_args()
    extra_overrides = [arg for arg in extra_overrides if arg != "--"]
    repo_root = _repo_root()
    eval_script = repo_root / "tdmpc2" / "eval.py"
    templates = _parse_template_list(args.templates)
    template_id_map = _load_template_id_map(args.srsa_task_template_fp, repo_root)
    gpu_id = int(args.gpu_id if args.gpu_id is not None else _gpu_id_from_device(args.device))
    video_prefix = _safe_name(args.video_prefix or args.assembly_id)
    base_overrides = _base_overrides(args, gpu_id)

    for index, (clearance_mult, depth_mult) in enumerate(templates, start=1):
        template = f"{clearance_mult:g}:{depth_mult:g}"
        template_id = template_id_map.get(_template_key(clearance_mult, depth_mult), None)
        if template_id is None:
            available = ", ".join(f"{c:g}:{d:g}" for c, d in sorted(template_id_map.keys()))
            raise ValueError(
                f"Template {template!r} is not listed in {args.srsa_task_template_fp}. "
                f"Available templates: {available}"
            )
        video_name = f"{video_prefix}_c{_safe_number(clearance_mult)}_d{_safe_number(depth_mult)}"
        command = [
            args.python,
            str(eval_script),
            *base_overrides,
            _override("srsa_param_template_id", template_id),
            _override("srsa_axial_clearance_depth_templates", template),
            _override("eval_video_name", video_name),
            _override("run_id", video_name),
            *extra_overrides,
        ]
        print(f"[newt] ({index}/{len(templates)}) Newt eval template {template} -> {video_name}", flush=True)
        print(f"[newt] command={shlex.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=repo_root, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
