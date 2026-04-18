"""Step 2: Extract text rhythm signals from character-level annotation JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from utils import (
    base_argparser,
    load_config,
    get_video_entry,
    resolve_video_ids,
    require_file,
    save_signals,
    PipelineError,
)


def load_annotation(path: Path) -> dict:
    """Load and return the annotation JSON project data."""
    with open(path) as f:
        data = json.load(f)
    return data["project"]


def parse_characters(project: dict) -> list[dict]:
    """Extract characterAnnotations sorted by startTime."""
    chars = project.get("characterAnnotations", [])
    if not chars:
        raise PipelineError("No characterAnnotations found in annotation file.")
    return sorted(chars, key=lambda c: c["startTime"])


def parse_breath_points(project: dict) -> list[float]:
    """Extract breath mark timestamps from the 呼吸轨 track."""
    for track in project.get("builtinTracks", []):
        for apt in track.get("attachedPointTracks", []):
            if "呼吸" in apt.get("name", ""):
                points = apt.get("points", [])
                return sorted(p["time"] for p in points)
    return []


def validate_timing(chars: list[dict], video_id: str) -> None:
    """Validate character timing: spans valid, check for overlaps."""
    for c in chars:
        if c["endTime"] <= c["startTime"]:
            print(
                f"  WARNING [{video_id}]: char '{c['char']}' at {c['startTime']:.3f}s "
                f"has endTime <= startTime",
                file=sys.stderr,
            )

    # Check for overlapping spans (consecutive chars only)
    overlap_count = 0
    for i in range(len(chars) - 1):
        if chars[i]["endTime"] > chars[i + 1]["startTime"] + 0.05:
            overlap_count += 1
    if overlap_count > 0:
        print(
            f"  WARNING [{video_id}]: {overlap_count} overlapping character span(s) detected",
            file=sys.stderr,
        )


def compute_text_features(
    chars: list[dict], breath_times: list[float], cfg: dict, total_duration: float
) -> tuple[dict[str, np.ndarray], dict]:
    """Compute all text rhythm signals from parsed annotations."""
    fps = cfg["common_fps"]
    text_cfg = cfg["text"]
    density_sigma = text_cfg["density_sigma"]
    breath_window = text_cfg["breath_window"]

    T = int(np.ceil(total_duration * fps))
    times = np.arange(T) / fps

    # Per-character arrays
    char_onsets = np.array([c["startTime"] for c in chars])
    char_offsets = np.array([c["endTime"] for c in chars])
    char_durations = char_offsets - char_onsets
    char_labels = np.array([c["char"] for c in chars])
    char_line_ids = np.array([c["lineId"] for c in chars])
    char_singing_styles = np.array([c.get("singingStyle", "") for c in chars])
    breath_times_arr = np.array(breath_times)

    # --- Frame-level signals ---

    # 5a. onset_signal: binary impulse at each character onset
    onset_signal = np.zeros(T, dtype=np.float32)
    for t in char_onsets:
        idx = int(round(t * fps))
        if 0 <= idx < T:
            onset_signal[idx] += 1.0

    # 5b. char_density: Gaussian-smoothed onset density (chars/sec)
    sigma_frames = density_sigma * fps
    char_density = gaussian_filter1d(onset_signal.astype(np.float64), sigma=sigma_frames)
    # Scale so that the integral matches: density in chars/sec
    char_density = (char_density * fps).astype(np.float32)

    # 5c. char_duration_signal: per-frame character duration
    char_duration_signal = np.zeros(T, dtype=np.float32)
    for onset, offset, dur in zip(char_onsets, char_offsets, char_durations):
        i_start = int(round(onset * fps))
        i_end = int(round(offset * fps))
        i_start = max(0, i_start)
        i_end = min(T, i_end)
        char_duration_signal[i_start:i_end] = dur

    # 5d. inter_onset_interval: time since previous onset, interpolated
    inter_onset_interval = np.full(T, np.nan, dtype=np.float32)
    if len(char_onsets) >= 2:
        # Compute IOI at each onset
        ioi_at_onsets = np.diff(char_onsets)
        onset_frames = np.array([int(round(t * fps)) for t in char_onsets])
        # Assign IOI values at onset frames (starting from 2nd onset)
        for i in range(1, len(onset_frames)):
            idx = onset_frames[i]
            if 0 <= idx < T:
                inter_onset_interval[idx] = ioi_at_onsets[i - 1]
        # Linearly interpolate between onset frames
        valid = ~np.isnan(inter_onset_interval)
        if np.sum(valid) >= 2:
            valid_indices = np.where(valid)[0]
            valid_values = inter_onset_interval[valid]
            # Interpolate only between first and last onset
            interp_range = np.arange(valid_indices[0], valid_indices[-1] + 1)
            inter_onset_interval[interp_range] = np.interp(
                interp_range, valid_indices, valid_values
            )

    # 5e. breath_signal: Gaussian pulse at each breath mark
    breath_signal = np.zeros(T, dtype=np.float32)
    sigma_breath = breath_window * fps
    for t in breath_times:
        idx = int(round(t * fps))
        if 0 <= idx < T:
            breath_signal[idx] = 1.0
    if np.any(breath_signal > 0):
        breath_signal = gaussian_filter1d(
            breath_signal.astype(np.float64), sigma=sigma_breath
        ).astype(np.float32)

    # 5f. silence_mask: True for frames not within any character span
    silence_mask = np.ones(T, dtype=bool)
    for onset, offset in zip(char_onsets, char_offsets):
        i_start = max(0, int(round(onset * fps)))
        i_end = min(T, int(round(offset * fps)))
        silence_mask[i_start:i_end] = False

    # --- Metadata ---
    singing_duration = float(np.sum(char_durations))
    silence_fraction = float(np.sum(silence_mask)) / T if T > 0 else 0.0

    # Tally singing styles
    styles, counts = np.unique(char_singing_styles, return_counts=True)
    singing_style_counts = {s: int(c) for s, c in zip(styles, counts)}

    # Count unique lines
    num_lines = len(np.unique(char_line_ids))

    arrays = {
        "char_onsets": char_onsets,
        "char_offsets": char_offsets,
        "char_durations": char_durations,
        "char_labels": char_labels,
        "char_line_ids": char_line_ids,
        "char_singing_styles": char_singing_styles,
        "breath_times": breath_times_arr,
        "onset_signal": onset_signal,
        "char_density": char_density,
        "char_duration_signal": char_duration_signal,
        "inter_onset_interval": inter_onset_interval,
        "breath_signal": breath_signal,
        "silence_mask": silence_mask,
        "times": times.astype(np.float32),
    }

    metadata = {
        "video_id": None,  # filled by caller
        "num_characters": len(chars),
        "num_breaths": len(breath_times),
        "num_lines": num_lines,
        "total_singing_duration_sec": round(singing_duration, 3),
        "silence_fraction": round(silence_fraction, 4),
        "singing_style_counts": singing_style_counts,
        "total_duration_sec": round(total_duration, 3),
        "common_fps": fps,
        "num_frames": T,
    }

    return arrays, metadata


def get_total_duration(chars: list[dict], breath_times: list[float], video_id: str) -> float:
    """Determine total duration from annotation data and any existing pipeline outputs."""
    # Start with annotation-derived duration
    duration = chars[-1]["endTime"]
    if breath_times:
        duration = max(duration, breath_times[-1])

    # Try to get video duration from audio metadata if available
    audio_meta_path = Path("data/processed") / f"{video_id}_audio_features.json"
    if audio_meta_path.exists():
        with open(audio_meta_path) as f:
            audio_meta = json.load(f)
        if "duration_sec" in audio_meta:
            duration = max(duration, audio_meta["duration_sec"])

    return duration


def process_video(video_id: str, cfg: dict) -> None:
    entry = get_video_entry(cfg, video_id)
    annotation_path = require_file(
        entry["annotation"],
        hint="Create annotation JSON or check config.",
    )

    print(f"[{video_id}] Loading annotation from {annotation_path}...")
    project = load_annotation(annotation_path)

    chars = parse_characters(project)
    breath_times = parse_breath_points(project)
    print(
        f"[{video_id}] Parsed {len(chars)} characters, "
        f"{len(breath_times)} breath marks"
    )

    validate_timing(chars, video_id)

    total_duration = get_total_duration(chars, breath_times, video_id)

    print(f"[{video_id}] Computing text features...")
    arrays, metadata = compute_text_features(chars, breath_times, cfg, total_duration)
    metadata["video_id"] = video_id

    out_path = Path("data/processed") / f"{video_id}_text_features.npz"
    save_signals(out_path, metadata, **arrays)

    print(
        f"[{video_id}] Text features saved: {metadata['num_frames']} frames, "
        f"{metadata['total_duration_sec']}s, {metadata['common_fps']} fps"
    )
    print(
        f"[{video_id}]   Characters: {metadata['num_characters']}, "
        f"Lines: {metadata['num_lines']}, "
        f"Breaths: {metadata['num_breaths']}"
    )
    print(
        f"[{video_id}]   Singing duration: {metadata['total_singing_duration_sec']}s, "
        f"Silence fraction: {metadata['silence_fraction']:.1%}"
    )
    print(f"[{video_id}]   Singing styles: {metadata['singing_style_counts']}")


def main():
    parser = base_argparser("Extract text rhythm signals from annotation JSON")
    args = parser.parse_args()
    cfg = load_config(args.config)

    for vid in resolve_video_ids(cfg, args):
        process_video(vid, cfg)


if __name__ == "__main__":
    main()
