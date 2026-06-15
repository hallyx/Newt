#!/usr/bin/env python3
"""Record Newt eval videos for multiple SRSA assembly ids and task sizes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys


DEFAULT_VIDEO_ROOT = "data/video_eval_ids_task_sizes"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value).strip("_")


def _resolve_path(value: str, *, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _split_items(value: str) -> list[str]:
    return [item.strip() for item in re.split(r"[;；\n]+", value) if item.strip()]


def _parse_item(value: str) -> tuple[str, str]:
    if "=" in value:
        assembly_id, checkpoint = value.split("=", 1)
    elif ":" in value:
        assembly_id, checkpoint = value.split(":", 1)
    else:
        parts = value.split()
        if len(parts) != 2:
            raise ValueError(
                f"Invalid item {value!r}; expected ASSEMBLY_ID=CHECKPOINT or ASSEMBLY_ID CHECKPOINT."
            )
        assembly_id, checkpoint = parts
    assembly_id = str(assembly_id).strip().zfill(5)
    checkpoint = str(checkpoint).strip()
    if not assembly_id or not checkpoint:
        raise ValueError(f"Invalid empty assembly/checkpoint item: {value!r}")
    return assembly_id, checkpoint


def _load_targets_from_items(items: str) -> list[tuple[str, str]]:
    return [_parse_item(item) for item in _split_items(items)]


def _load_targets_from_csv(path: Path) -> list[tuple[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV manifest is empty: {path}")
        targets = []
        for row in reader:
            assembly_id = row.get("assembly_id") or row.get("id") or row.get("asm")
            checkpoint = row.get("checkpoint") or row.get("checkpoint_path") or row.get("pt")
            if not assembly_id or not checkpoint:
                raise ValueError(
                    f"CSV manifest rows need assembly_id/id and checkpoint/checkpoint_path columns: {path}"
                )
            targets.append((str(assembly_id).zfill(5), str(checkpoint)))
    return targets


def _load_targets_from_json(path: Path) -> list[tuple[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    if isinstance(doc, dict):
        if "targets" in doc:
            doc = doc["targets"]
        else:
            return [(str(key).zfill(5), str(value)) for key, value in doc.items()]
    if not isinstance(doc, list):
        raise ValueError(f"JSON manifest must be a list or mapping: {path}")
    targets = []
    for item in doc:
        if isinstance(item, str):
            targets.append(_parse_item(item))
            continue
        if not isinstance(item, dict):
            raise ValueError(f"Unsupported JSON manifest entry: {item!r}")
        assembly_id = item.get("assembly_id") or item.get("id") or item.get("asm")
        checkpoint = item.get("checkpoint") or item.get("checkpoint_path") or item.get("pt")
        if not assembly_id or not checkpoint:
            raise ValueError(f"JSON manifest entry needs assembly_id/id and checkpoint: {item!r}")
        targets.append((str(assembly_id).zfill(5), str(checkpoint)))
    return targets


def _load_targets(args: argparse.Namespace, repo_root: Path) -> list[tuple[str, Path]]:
    raw_targets: list[tuple[str, str]] = []
    if args.items:
        raw_targets.extend(_load_targets_from_items(args.items))
    if args.manifest:
        manifest_fp = _resolve_path(args.manifest, base=repo_root)
        suffix = manifest_fp.suffix.lower()
        if suffix == ".json":
            raw_targets.extend(_load_targets_from_json(manifest_fp))
        else:
            raw_targets.extend(_load_targets_from_csv(manifest_fp))
    if not raw_targets:
        raise ValueError("Provide --items or --manifest.")

    targets = []
    seen = set()
    for assembly_id, checkpoint in raw_targets:
        if assembly_id in seen and not args.allow_duplicate_ids:
            raise ValueError(f"Duplicate assembly_id={assembly_id}; pass --allow_duplicate_ids to keep duplicates.")
        seen.add(assembly_id)
        targets.append((assembly_id, _resolve_path(checkpoint, base=repo_root)))
    return targets


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Batch record videos for multiple assembly ids/checkpoints. Unknown options are forwarded "
            "to scripts/batch_record_task_sizes.py."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  scripts/batch_record_ids_task_sizes.py \\\n"
            "    --items \"00186=logs/.../00186/models/best.pt;00271=logs/.../00271/models/best.pt\" \\\n"
            "    --templates \"0.5:0.5;1.0:1.0;4.0:2.0\" \\\n"
            "    --freespace_range 0.10 --episode_length_s 10 --video_fps 15\n\n"
            "CSV manifest columns: assembly_id,checkpoint\n"
            "JSON manifest: {\"00186\": \"path/to/best.pt\"} or [{\"assembly_id\": \"00186\", \"checkpoint\": \"...\"}]"
        ),
    )
    parser.add_argument(
        "--items",
        default=None,
        help="Semicolon/newline separated ASSEMBLY_ID=CHECKPOINT pairs.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="CSV/JSON manifest with assembly_id and checkpoint columns/fields.",
    )
    parser.add_argument(
        "--record_script",
        default="scripts/batch_record_task_sizes.py",
        help="Single-assembly recording script to call.",
    )
    parser.add_argument(
        "--video_root",
        default=DEFAULT_VIDEO_ROOT,
        help="Root directory for videos. Each assembly id gets a subdirectory.",
    )
    parser.add_argument(
        "--allow_duplicate_ids",
        action="store_true",
        help="Allow repeated assembly ids in --items/--manifest.",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue with later ids if one recording command fails.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running eval.")
    return parser


def main() -> int:
    parser = _build_parser()
    args, forwarded_args = parser.parse_known_args()
    repo_root = _repo_root()
    record_script = _resolve_path(args.record_script, base=repo_root)
    video_root = _resolve_path(args.video_root, base=repo_root)
    targets = _load_targets(args, repo_root)

    if not record_script.exists():
        raise FileNotFoundError(f"record script not found: {record_script}")
    if not args.dry_run:
        video_root.mkdir(parents=True, exist_ok=True)

    print(f"[newt-multi] repo_root={repo_root}", flush=True)
    print(f"[newt-multi] record_script={record_script}", flush=True)
    print(f"[newt-multi] video_root={video_root}", flush=True)
    print(f"[newt-multi] targets={len(targets)}", flush=True)

    failures: list[tuple[str, int]] = []
    for index, (assembly_id, checkpoint) in enumerate(targets, start=1):
        if not checkpoint.exists():
            message = f"[newt-multi] checkpoint not found for assembly_id={assembly_id}: {checkpoint}"
            if args.dry_run:
                print(f"{message} (dry-run warning)", flush=True)
            else:
                raise FileNotFoundError(message)

        asm_video_dir = video_root / _safe_name(assembly_id)
        command = [
            str(record_script),
            "--assembly_id",
            assembly_id,
            "--checkpoint",
            str(checkpoint),
            "--video_dir",
            str(asm_video_dir),
            "--video_prefix",
            assembly_id,
            *forwarded_args,
        ]
        if args.dry_run and "--dry_run" not in forwarded_args:
            command.append("--dry_run")

        print(f"[newt-multi] ({index}/{len(targets)}) assembly_id={assembly_id}", flush=True)
        print(f"[newt-multi] command={shlex.join(command)}", flush=True)
        if args.dry_run:
            continue
        try:
            subprocess.run(command, cwd=repo_root, check=True)
        except subprocess.CalledProcessError as exc:
            failures.append((assembly_id, int(exc.returncode)))
            print(
                f"[newt-multi] failed assembly_id={assembly_id} returncode={exc.returncode}",
                flush=True,
            )
            if not args.continue_on_error:
                raise

    if failures:
        summary = ", ".join(f"{assembly_id}:{code}" for assembly_id, code in failures)
        print(f"[newt-multi] failures: {summary}", flush=True)
        return 1
    print("[newt-multi] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
