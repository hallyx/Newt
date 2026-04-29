#!/usr/bin/env python3
"""
Export a compact state-only offline dataset for Phase 1 canonical Newt training.

This script reads a TensorDict-style rollout file exported from SRSA teacher
collection and writes a reduced dataset that keeps only the fields needed for
14D canonical offline training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tdmpc2.offline_io import export_compact_dataset, export_multitask_compact_dataset


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Export a compact state-only offline dataset for canonical 14D Newt training."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to the source teacher_rollouts_newt.pt file.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Optional offline multitask manifest JSON. When provided, merges multiple source rollout files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the compact output .pt file.",
    )
    parser.add_argument(
        "--obs-key",
        type=str,
        default="obs",
        help="Source key to use as obs. Default: obs",
    )
    parser.add_argument(
        "--next-obs-key",
        type=str,
        default="next_obs",
        help="Source key to use as next_obs. Default: next_obs",
    )
    parser.add_argument(
        "--action-key",
        type=str,
        default="action",
        help="Source key to use as action. Default: action",
    )
    parser.add_argument(
        "--metadata-out",
        type=str,
        default=None,
        help="Optional path for a sidecar metadata JSON. Default: <output>.json",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output if it already exists.",
    )
    return parser.parse_args()

def main():
    args = _parse_args()
    if bool(args.input) == bool(args.manifest):
        raise ValueError("Provide exactly one of --input or --manifest.")
    if args.manifest:
        output_fp, metadata_fp, summary = export_multitask_compact_dataset(
            args.manifest,
            args.output,
            obs_key=args.obs_key,
            next_obs_key=args.next_obs_key,
            action_key=args.action_key,
            metadata_fp=args.metadata_out,
            overwrite=args.overwrite,
        )
    else:
        output_fp, metadata_fp, summary = export_compact_dataset(
            args.input,
            args.output,
            obs_key=args.obs_key,
            next_obs_key=args.next_obs_key,
            action_key=args.action_key,
            metadata_fp=args.metadata_out,
            overwrite=args.overwrite,
        )

    print(f"Saved compact offline dataset to: {output_fp}")
    print(f"Saved metadata to: {metadata_fp}")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
