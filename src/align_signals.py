"""Step 5: Resample audio and motion signals to a common time axis."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from utils import (
    base_argparser,
    load_config,
    resolve_video_ids,
    require_file,
    load_signals,
    save_signals,
)


def _resample_preserving_nan(times_old: np.ndarray, signal: np.ndarray, times_new: np.ndarray) -> np.ndarray:
    """Resample a signal to a new time grid, preserving NaN positions."""
    nan_mask = np.isnan(signal)
    filled = signal.copy()
    filled[nan_mask] = 0.0

    resampled = np.interp(times_new, times_old, filled)

    # Re-apply NaN: if a new time point falls near an old NaN region, mark it NaN
    # For each new time, find the nearest old time and check if it was NaN
    nan_resampled = np.interp(times_new, times_old, nan_mask.astype(float))
    resampled[nan_resampled > 0.5] = np.nan

    return resampled


def align_signals(video_id: str, cfg: dict) -> None:
    common_fps = cfg["common_fps"]

    # Load audio features
    audio_path = Path("data/processed") / f"{video_id}_audio_features.npz"
    audio_arrays, audio_meta = load_signals(audio_path)

    # Load motion signals
    motion_path = Path("data/processed") / f"{video_id}_motion_signals.npz"
    motion_arrays, motion_meta = load_signals(motion_path)

    # Load shot boundaries for cut mask
    sb_path = require_file(
        Path("data/processed") / f"{video_id}_shot_boundaries.json",
        hint="Run detect_cuts.py first.",
    )
    with open(sb_path) as f:
        shot_data = json.load(f)
    cut_frames_orig = shot_data["cuts"]
    nan_margin = cfg["cuts"]["nan_margin"]

    # Determine common duration — trim to shortest
    audio_duration = audio_meta["duration_sec"]
    motion_times = motion_arrays["times"]
    motion_duration = float(motion_times[-1]) if len(motion_times) > 0 else 0

    if abs(audio_duration - motion_duration) > 0.5:
        print(
            f"[{video_id}] Warning: audio duration ({audio_duration:.2f}s) and "
            f"motion duration ({motion_duration:.2f}s) differ by "
            f"{abs(audio_duration - motion_duration):.2f}s"
        )

    common_duration = min(audio_duration, motion_duration)
    N = int(common_duration * common_fps)
    times_common = np.arange(N) / common_fps

    # Resample audio signals
    audio_times = audio_arrays["times"]
    audio_onset = _resample_preserving_nan(audio_times, audio_arrays["onset_env"], times_common)
    audio_rms = _resample_preserving_nan(audio_times, audio_arrays["rms"], times_common)
    audio_f0 = _resample_preserving_nan(audio_times, audio_arrays["f0"], times_common)
    audio_pitch_delta = _resample_preserving_nan(audio_times, audio_arrays["pitch_delta"], times_common)

    # Resample motion signals
    motion_keys = [
        ("motion_total", "total_motion"),
        ("motion_hand", "hand_motion"),
        ("motion_hand_left", "hand_left_motion"),
        ("motion_hand_right", "hand_right_motion"),
        ("motion_torso", "torso_motion"),
        ("motion_upper_body", "upper_body_motion"),
        ("motion_head", "head_motion"),
        ("motion_root_displacement", "root_displacement"),
    ]

    motion_resampled = {}
    for out_name, src_name in motion_keys:
        motion_resampled[out_name] = _resample_preserving_nan(
            motion_times, motion_arrays[src_name], times_common
        )

    # Generate cut mask on the common timeline
    video_fps = shot_data["video_fps"]
    cut_mask = np.zeros(N, dtype=bool)
    for cf in cut_frames_orig:
        cut_time = cf / video_fps
        # Mark frames within nan_margin of the cut
        for i in range(N):
            frame_time = times_common[i]
            margin_sec = nan_margin / common_fps
            if abs(frame_time - cut_time) <= margin_sec + 0.5 / common_fps:
                cut_mask[i] = True

    # NaN fraction in motion
    nan_frac_motion = np.isnan(motion_resampled["motion_total"]).sum() / N if N > 0 else 0

    # Merged metadata
    metadata = {
        "video_id": video_id,
        "common_fps": common_fps,
        "total_duration_sec": round(common_duration, 3),
        "total_frames": N,
        "nan_fraction_motion": round(float(nan_frac_motion), 4),
        "audio": audio_meta,
        "motion": motion_meta,
    }

    # Save
    out_path = Path("data/processed") / f"{video_id}_aligned_signals.npz"
    save_signals(
        out_path, metadata,
        times=times_common,
        audio_onset=audio_onset,
        audio_rms=audio_rms,
        audio_f0=audio_f0,
        audio_pitch_delta=audio_pitch_delta,
        cut_mask=cut_mask,
        **motion_resampled,
    )

    # Summary
    n_audio = 4
    n_motion = len(motion_keys)
    print(f"\n[{video_id}] Aligned signals saved: {N} frames, {common_duration:.1f} sec, {common_fps} fps")
    print(f"  Audio signals: {n_audio} channels (onset, rms, f0, pitch_delta)")
    print(f"  Motion signals: {n_motion} channels (total, hand, hand_left, hand_right, torso, upper_body, head, root_displacement)")
    print(f"  NaN coverage in motion: {nan_frac_motion*100:.1f}%")


def process_video(video_id: str, cfg: dict) -> None:
    align_signals(video_id, cfg)


def main():
    parser = base_argparser("Align audio and motion signals to common timeline")
    args = parser.parse_args()
    cfg = load_config(args.config)

    for vid in resolve_video_ids(cfg, args):
        process_video(vid, cfg)


if __name__ == "__main__":
    main()
