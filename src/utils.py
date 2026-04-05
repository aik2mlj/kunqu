"""Shared helpers for the Kunqu analysis pipeline."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml


def load_config(path: str | Path) -> dict:
    """Load a YAML config file and return as dict."""
    path = Path(path)
    if not path.exists():
        print(f"Error: config file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f)


def get_video_entry(cfg: dict, video_id: str) -> dict:
    """Look up a video entry from the config by id."""
    for v in cfg["videos"]:
        if v["id"] == video_id:
            return v
    print(f"Error: video_id '{video_id}' not found in config.", file=sys.stderr)
    sys.exit(1)


def get_video_ids(cfg: dict) -> list[str]:
    """Return all video ids listed in config."""
    return [v["id"] for v in cfg["videos"]]


def require_file(path: str | Path, hint: str = "") -> Path:
    """Check that a file exists, exit with a message if not."""
    path = Path(path)
    if not path.exists():
        msg = f"Error: {path} not found."
        if hint:
            msg += f" {hint}"
        print(msg, file=sys.stderr)
        sys.exit(1)
    return path


def require_ffmpeg() -> None:
    """Check that ffmpeg is available on PATH."""
    if shutil.which("ffmpeg") is None:
        print(
            "Error: ffmpeg not found. Install it via your system package manager "
            "(e.g., apt install ffmpeg, brew install ffmpeg).",
            file=sys.stderr,
        )
        sys.exit(1)


def save_signals(npz_path: str | Path, metadata: dict, **arrays: np.ndarray) -> None:
    """Save numpy arrays to .npz and metadata to a companion .json sidecar."""
    npz_path = Path(npz_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **arrays)
    json_path = npz_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2, default=_json_default)


def load_signals(npz_path: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    """Load arrays from .npz and metadata from companion .json sidecar."""
    npz_path = Path(npz_path)
    require_file(npz_path)
    arrays = dict(np.load(npz_path))
    json_path = npz_path.with_suffix(".json")
    require_file(json_path)
    with open(json_path) as f:
        metadata = json.load(f)
    return arrays, metadata


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def base_argparser(description: str) -> argparse.ArgumentParser:
    """Create a base argument parser with --config, --video_id, and --all."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML file",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video_id", type=str, help="Process a single video by id")
    group.add_argument(
        "--all", action="store_true", help="Process all videos listed in config"
    )
    return parser


def resolve_video_ids(cfg: dict, args: argparse.Namespace) -> list[str]:
    """Return the list of video ids to process based on CLI args."""
    if args.all:
        return get_video_ids(cfg)
    return [args.video_id]


def get_joint_groups(cfg: dict, pose_model: str) -> dict:
    """Return the joint group definitions for the given pose model."""
    motion_cfg = cfg["motion"]
    key = f"joint_groups_{pose_model}"
    if key not in motion_cfg:
        print(
            f"Error: no joint group definitions for pose model '{pose_model}' in config.",
            file=sys.stderr,
        )
        sys.exit(1)
    return motion_cfg[key]
