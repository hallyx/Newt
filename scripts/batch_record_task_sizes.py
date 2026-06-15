#!/usr/bin/env python3
"""Record Newt eval videos across SRSA clearance/depth task sizes."""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
import re
import selectors
import shlex
import signal
import subprocess
import sys
import time


DEFAULT_PYTHON = "/home/gpuserver/miniconda3/envs/isaac51/bin/python"
DEFAULT_ISAACLAB_DIR = "/home/gpuserver/IsaacLab"
DEFAULT_SRSA_DIR = "/home/gpuserver/hx/github/srsa"
DEFAULT_SIZE_TEMPLATES = "0.5:0.5;0.5:1.0;1.0:1.0;2.0:1.5;4.0:2.0"
SUCCESS_MARKERS = (
    "Evaluation artifacts saved successfully.",
    "Evaluation completed successfully.",
    "Real closed-loop inference completed successfully.",
)


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


def _normalize_extra_overrides(values: list[str]) -> list[str]:
    normalized: list[str] = []
    index = 0
    while index < len(values):
        value = values[index]
        if value == "--":
            index += 1
            continue
        if value.startswith("--") and len(value) > 2:
            key_value = value[2:]
            if "=" in key_value:
                key, raw = key_value.split("=", 1)
                normalized.append(_override(key.replace("-", "_"), raw))
                index += 1
                continue
            if index + 1 < len(values) and not values[index + 1].startswith("--"):
                normalized.append(_override(key_value.replace("-", "_"), values[index + 1]))
                index += 2
                continue
            normalized.append(_override(key_value.replace("-", "_"), True))
            index += 1
            continue
        normalized.append(value)
        index += 1
    return normalized


def _terminate_process(proc: subprocess.Popen, *, kill_after_s: float = 10.0) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            proc.terminate()
        else:
            os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        proc.terminate()
    try:
        proc.wait(timeout=max(0.1, float(kill_after_s)))
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        if os.name == "nt":
            proc.kill()
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except Exception:
        proc.kill()
    proc.wait()


def _artifact_outputs_ready(
    *,
    required_paths: list[Path],
    required_globs: list[str],
    started_at_wall_s: float,
) -> bool:
    min_mtime = float(started_at_wall_s) - 1.0
    for path in required_paths:
        try:
            stat = path.stat()
        except FileNotFoundError:
            return False
        if stat.st_size <= 0 or stat.st_mtime < min_mtime:
            return False
    for pattern in required_globs:
        matches = [Path(path) for path in glob.glob(pattern)]
        fresh_matches = []
        for path in matches:
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            if stat.st_size > 0 and stat.st_mtime >= min_mtime:
                fresh_matches.append(path)
        if not fresh_matches:
            return False
    return True


def _run_eval_command(
    command: list[str],
    *,
    cwd: Path,
    exit_grace_s: float | None,
    artifact_required_paths: list[Path] | None = None,
    artifact_required_globs: list[str] | None = None,
) -> None:
    started_at_wall_s = time.time()
    proc = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=(os.name != "nt"),
    )
    assert proc.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)
    success_seen_at = None
    stream_open = True
    try:
        while True:
            if stream_open:
                for key, _ in selector.select(timeout=0.5):
                    line = key.fileobj.readline()
                    if line == "":
                        selector.unregister(key.fileobj)
                        stream_open = False
                        break
                    print(line, end="", flush=True)
                    if any(marker in line for marker in SUCCESS_MARKERS):
                        success_seen_at = time.monotonic()
            else:
                time.sleep(0.1)

            if (
                success_seen_at is None
                and (artifact_required_paths or artifact_required_globs)
                and _artifact_outputs_ready(
                    required_paths=artifact_required_paths or [],
                    required_globs=artifact_required_globs or [],
                    started_at_wall_s=started_at_wall_s,
                )
            ):
                success_seen_at = time.monotonic()
                print(
                    "[newt] eval video/summary artifacts detected; treating eval as complete "
                    "for teardown timeout handling.",
                    flush=True,
                )

            returncode = proc.poll()
            if returncode is not None:
                if stream_open:
                    for line in proc.stdout:
                        print(line, end="", flush=True)
                if returncode != 0:
                    if success_seen_at is not None:
                        print(
                            "[newt-warning] eval printed success but exited nonzero during teardown "
                            f"(returncode={returncode}); continuing.",
                            flush=True,
                        )
                        return
                    raise subprocess.CalledProcessError(returncode, command)
                return

            if (
                success_seen_at is not None
                and exit_grace_s is not None
                and float(exit_grace_s) >= 0.0
                and time.monotonic() - success_seen_at >= float(exit_grace_s)
            ):
                print(
                    "[newt-warning] eval printed success but IsaacSim did not exit within "
                    f"{float(exit_grace_s):g}s; terminating the child process and continuing.",
                    flush=True,
                )
                _terminate_process(proc)
                return
    except KeyboardInterrupt:
        _terminate_process(proc)
        raise
    finally:
        selector.close()


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
    parser.add_argument(
        "--video_length",
        type=int,
        default=300,
        help="Maximum total recorded frames per template. Use 0 to disable the frame cap.",
    )
    parser.add_argument(
        "--video_episodes",
        type=int,
        default=1,
        help="Number of completed episodes to include in each video. eval_trials is raised to at least this value.",
    )
    parser.add_argument("--video_dir", default="data/video_eval_task_sizes", help="Video output directory.")
    parser.add_argument("--video_fps", type=int, default=15)
    parser.add_argument("--video_format", default="mp4")
    parser.add_argument("--video_record_every", type=int, default=1)
    parser.add_argument("--video_prefix", default=None, help="Filename prefix. Defaults to assembly_id.")
    parser.add_argument("--no_force_trace", action="store_true", help="Do not save frame-aligned force CSV traces.")
    parser.add_argument("--model_size", default="S")
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--srsa_task_template_fp", default="data/srsa_axial_task_templates.json")
    parser.add_argument("--srsa_mesh_geometry_fp", default="data/srsa_mesh_geometry_params.csv")
    parser.add_argument(
        "--size_source",
        choices=("fixed", "mesh"),
        default="fixed",
        help=(
            "fixed matches SRSA scripts/batch_record_task_sizes.py using clearance_base/depth_base; "
            "mesh uses Newt mesh-derived task templates."
        ),
    )
    parser.add_argument("--clearance_base", type=float, default=0.000114)
    parser.add_argument("--depth_base", type=float, default=0.015)
    parser.add_argument("--clearance_jitter_ratio", type=float, default=0.0)
    parser.add_argument("--depth_jitter_ratio", type=float, default=0.0)
    parser.add_argument(
        "--init_error_xy_range",
        default="0.009,0.0010",
        help="Initial lateral error magnitude range in meters. SRSA treats positive ranges as signed +/- magnitudes.",
    )
    parser.add_argument(
        "--init_error_z_range",
        default="0.0010,0.0020",
        help="SRSA sampled z-error range in meters. The visible start height is controlled by --freespace_range.",
    )
    parser.add_argument(
        "--init_error_yaw_range",
        default="-0.0872665,0.0872665",
        help="Initial yaw error range in radians.",
    )
    parser.add_argument(
        "--no_terminate_on_success",
        action="store_true",
        help="Keep rolling after success instead of ending the episode early.",
    )
    parser.add_argument(
        "--freespace_range",
        type=float,
        default=None,
        help="Socket-top free-space height in meters. This is the visible vertical start offset above insertion depth.",
    )
    parser.add_argument(
        "--episode_length_s",
        type=float,
        default=None,
        help="IsaacLab episode length in seconds. Default SRSA/AutoMate value is 5s.",
    )
    parser.add_argument("--success_metric", default="relaxed")
    parser.add_argument("--socket_camera_profile_fp", default="camera_profiles/socket_camera_offset.json")
    parser.add_argument("--socket_camera_env_index", type=int, default=0)
    parser.add_argument("--no_socket_camera_follow", action="store_true")
    display_group = parser.add_mutually_exclusive_group()
    display_group.add_argument(
        "--interactive",
        dest="interactive",
        action="store_true",
        default=True,
        help="Use a visible Isaac window. This is the default.",
    )
    display_group.add_argument(
        "--headless",
        dest="interactive",
        action="store_false",
        help="Run IsaacLab headless.",
    )
    parser.add_argument(
        "--isaaclab_multi_gpu",
        action="store_true",
        help="Allow IsaacSim renderer multi-GPU mode. Default is disabled so --device cuda:N stays on one GPU.",
    )
    parser.add_argument("--no_contact_preset", action="store_true", help="Do not add 17D/contact-history defaults.")
    parser.add_argument(
        "--eval_exit_grace_s",
        type=float,
        default=30.0,
        help=(
            "After eval prints a success marker, wait this many seconds for IsaacSim to exit. "
            "If it is still stuck in plugin teardown, terminate it and continue. Use -1 to disable."
        ),
    )
    parser.add_argument("--dry_run", action="store_true", help="Print Newt eval commands without running them.")
    return parser


def _base_overrides(args: argparse.Namespace, gpu_id: int) -> list[str]:
    eval_trials = max(int(args.eval_trials), int(args.video_episodes))
    fixed_size_source = args.size_source == "fixed"
    overrides = [
        _override("checkpoint", args.checkpoint),
        _override("eval_mode", "sim"),
        _override("isaaclab_backend", "srsa"),
        _override("task", "isaaclab-srsa-assembly"),
        _override("assembly_id", args.assembly_id),
        _override("isaaclab_dir", args.isaaclab_dir),
        _override("srsa_dir", args.srsa_dir),
        _override("eval_task_template_exact", not fixed_size_source),
        _override("eval_task_template_apply_geometry", not fixed_size_source),
        _override("eval_task_template_apply_sampler", not fixed_size_source),
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
        _override("isaaclab_multi_gpu", args.isaaclab_multi_gpu),
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
        _override("srsa_axial_init_error_xy_range", args.init_error_xy_range),
        _override("srsa_axial_init_error_z_range", args.init_error_z_range),
        _override("srsa_axial_init_error_yaw_range", args.init_error_yaw_range),
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
        _override("eval_trials", eval_trials),
        _override("eval_terminate_on_success", not args.no_terminate_on_success),
        _override("eval_terminate_success_key", "terminal_process_success"),
        _override("save_video", True),
        _override("eval_video_dir", args.video_dir),
        _override("eval_video_fps", args.video_fps),
        _override("eval_video_format", args.video_format),
        _override("eval_video_max_episodes", args.video_episodes),
        _override("eval_video_record_every", args.video_record_every),
        _override("eval_video_env_index", 0),
        _override("eval_video_force_trace", not args.no_force_trace),
        _override("eval_video_force_trace_env_index", 0),
        _override("enable_wandb", False),
        _override("save_agent", False),
        _override("exp_name", "batch_record_task_sizes"),
        _override("seed", 1),
    ]
    if args.freespace_range is not None:
        overrides.append(_override("srsa_curriculum_freespace_range", args.freespace_range))
    if args.episode_length_s is not None:
        overrides.extend([
            _override("isaaclab_episode_length_s", args.episode_length_s),
            _override("isaaclab_max_episode_steps", max(1, int(round(float(args.episode_length_s) * 15.0)))),
        ])
    if int(args.video_length) > 0:
        overrides.append(_override("eval_video_max_frames", args.video_length))
    if not fixed_size_source:
        overrides.extend([
            _override("srsa_task_template_fp", args.srsa_task_template_fp),
            _override("srsa_mesh_geometry_fp", args.srsa_mesh_geometry_fp),
        ])
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


def _expected_artifacts(args: argparse.Namespace, repo_root: Path, video_name: str) -> tuple[list[Path], list[str]]:
    work_dir = repo_root / "logs" / "isaaclab-srsa-assembly" / "1" / "batch_record_task_sizes" / video_name
    video_dir = Path(args.video_dir).expanduser()
    if not video_dir.is_absolute():
        video_dir = work_dir / video_dir
    fmt = str(args.video_format or "mp4").strip().lower().lstrip(".") or "mp4"
    required_paths = [work_dir / "eval_summary" / "eval_summary.json"]
    required_globs = [str(video_dir / f"{video_name}_step-*.{fmt}")]
    return required_paths, required_globs


def main() -> int:
    parser = _build_parser()
    args, extra_overrides = parser.parse_known_args()
    extra_overrides = _normalize_extra_overrides(extra_overrides)
    repo_root = _repo_root()
    eval_script = repo_root / "tdmpc2" / "eval.py"
    templates = _parse_template_list(args.templates)
    template_id_map = (
        _load_template_id_map(args.srsa_task_template_fp, repo_root)
        if args.size_source == "mesh"
        else {}
    )
    gpu_id = int(args.gpu_id if args.gpu_id is not None else _gpu_id_from_device(args.device))
    video_prefix = _safe_name(args.video_prefix or args.assembly_id)
    base_overrides = _base_overrides(args, gpu_id)
    if int(args.video_length) > 0 and int(args.video_episodes) > 1:
        episode_steps = int(round(float(args.episode_length_s) * 15.0)) if args.episode_length_s else 75
        expected_frames = int(args.video_episodes) * max(1, episode_steps) + 1
        if int(args.video_length) < expected_frames:
            print(
                "[newt-warning] --video_length is a total frame cap, not per-episode. "
                f"video_length={args.video_length} may stop before recording "
                f"video_episodes={args.video_episodes}; estimated_full_frames={expected_frames}. "
                "Increase --video_length or use --video_length 0.",
                flush=True,
            )

    for index, (clearance_mult, depth_mult) in enumerate(templates, start=1):
        template = f"{clearance_mult:g}:{depth_mult:g}"
        template_id = template_id_map.get(_template_key(clearance_mult, depth_mult), None)
        if args.size_source == "mesh" and template_id is None:
            available = ", ".join(f"{c:g}:{d:g}" for c, d in sorted(template_id_map.keys()))
            raise ValueError(
                f"Template {template!r} is not listed in {args.srsa_task_template_fp}. "
                f"Available templates: {available}"
            )
        video_name = f"{video_prefix}_c{_safe_number(clearance_mult)}_d{_safe_number(depth_mult)}"
        template_overrides = []
        if args.size_source == "mesh":
            template_overrides.append(_override("srsa_param_template_id", template_id))
        template_overrides.extend([
            _override("srsa_axial_clearance_depth_templates", template),
            _override("eval_video_name", video_name),
            _override("run_id", video_name),
        ])
        command = [
            args.python,
            str(eval_script),
            *base_overrides,
            *template_overrides,
            *extra_overrides,
        ]
        if args.size_source == "fixed":
            clearance = float(args.clearance_base) * float(clearance_mult)
            depth = float(args.depth_base) * float(depth_mult)
            size_text = f"diametral_clearance={clearance * 1.0e3:.3f}mm target_depth={depth * 1.0e3:.3f}mm"
        else:
            size_text = "mesh-derived size from Newt task template"
        print(f"[newt] ({index}/{len(templates)}) Newt eval template {template} -> {video_name} ({size_text})", flush=True)
        print(f"[newt] command={shlex.join(command)}", flush=True)
        if not args.dry_run:
            exit_grace_s = None if float(args.eval_exit_grace_s) < 0.0 else float(args.eval_exit_grace_s)
            artifact_paths, artifact_globs = _expected_artifacts(args, repo_root, video_name)
            _run_eval_command(
                command,
                cwd=repo_root,
                exit_grace_s=exit_grace_s,
                artifact_required_paths=artifact_paths,
                artifact_required_globs=artifact_globs,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
