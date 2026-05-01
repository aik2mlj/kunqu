"""Step 6: Generate QA plots for sanity-checking pipeline output."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils import (
    base_argparser,
    load_config,
    resolve_video_ids,
    require_file,
    load_signals,
)


def _add_cut_lines(ax, cut_times: list[float], nan_margin_sec: float):
    """Add vertical red dashed lines and NaN-margin shading at cuts."""
    for ct in cut_times:
        ax.axvline(ct, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.axvspan(ct - nan_margin_sec, ct + nan_margin_sec, color="red", alpha=0.08)


def _shade_mask(ax, times: np.ndarray, mask: np.ndarray, color: str = "grey", alpha: float = 0.15):
    """Shade contiguous True regions of a boolean mask."""
    if len(mask) == 0:
        return
    changes = np.diff(mask.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1
    # Handle edge cases
    if mask[0]:
        starts = np.concatenate([[0], starts])
    if mask[-1]:
        ends = np.concatenate([ends, [len(mask)]])
    for s, e in zip(starts, ends):
        ax.axvspan(times[s], times[min(e, len(times) - 1)], color=color, alpha=alpha)


def _has_text(aligned: dict) -> bool:
    """Check if aligned signals contain text modality."""
    return "text_density" in aligned


def plot_signals_overview(
    video_id: str, aligned: dict, meta: dict, cut_times: list, nan_margin_sec: float, out_dir: Path
):
    """Plot 1: Stacked signal overview (7 rows with text, 5 without)."""
    times = aligned["times"]
    has_text = _has_text(aligned)
    n_rows = 7 if has_text else 5

    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 2.4 * n_rows), sharex=True)
    fig.suptitle(f"{video_id} — Signal Overview", fontsize=14)

    row = 0

    # Row 1: onset
    axes[row].plot(times, aligned["audio_onset"], linewidth=0.5, color="C0")
    axes[row].set_ylabel("Onset strength")
    row += 1

    # Row 2: RMS
    axes[row].plot(times, aligned["audio_rms"], linewidth=0.5, color="C1")
    axes[row].set_ylabel("RMS energy")
    row += 1

    # Row 3: pitch
    axes[row].plot(times, aligned["audio_f0"], linewidth=0.5, color="C2")
    axes[row].set_ylabel("Pitch (Hz)")
    row += 1

    if has_text:
        # Row 4: text density + onset ticks
        density = aligned["text_density"]
        axes[row].plot(times, density, linewidth=0.5, color="C4")
        # Add onset ticks along the top
        onset = aligned["text_onset"]
        onset_times = times[onset > 0]
        tick_y = float(np.nanmax(density)) if np.any(density > 0) else 1.0
        axes[row].scatter(onset_times, np.full(len(onset_times), tick_y),
                          marker="|", s=10, color="C4", alpha=0.5)
        axes[row].set_ylabel("Char density\n(chars/sec)")
        row += 1

        # Row 5: breath marks + silence shading
        axes[row].plot(times, aligned["text_breath"], linewidth=0.5, color="C5")
        # Grey shading for silence regions
        silence = aligned["text_silence_mask"].astype(bool)
        _shade_mask(axes[row], times, silence, color="grey", alpha=0.15)
        axes[row].set_ylabel("Breath marks")
        row += 1

    # Row 6 (or 4): total motion
    axes[row].plot(times, aligned["motion_total"], linewidth=0.5, color="C3")
    axes[row].set_ylabel("Total motion")
    row += 1

    # Row 7 (or 5): hand vs torso
    axes[row].plot(times, aligned["motion_hand"], linewidth=0.5, color="C0", label="Hand")
    axes[row].plot(times, aligned["motion_torso"], linewidth=0.5, color="C1", label="Torso")
    axes[row].set_ylabel("Motion")
    axes[row].legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (seconds)")

    for ax in axes:
        _add_cut_lines(ax, cut_times, nan_margin_sec)

    plt.tight_layout()
    path = out_dir / "01_signals_overview.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_pose_quality(
    video_id: str, cfg: dict, out_dir: Path
):
    """Plot 2: Pose quality report."""
    kp_path = Path("data/poses") / f"{video_id}_keypoints.npz"
    kp_arrays, kp_meta = load_signals(kp_path)

    confidence = kp_arrays["confidence"]  # (T, J)
    T, J = confidence.shape
    fps = kp_meta["video_fps"]
    times = np.arange(T) / fps

    pose_model = kp_meta["model_name"]
    from utils import get_joint_groups
    jg = get_joint_groups(cfg, pose_model)

    # Upper vs lower body validity
    ub_indices = jg.get("upper_body", [])
    leg_indices = jg.get("legs", [])

    conf_thresh = cfg["pose"]["confidence_threshold"]

    def valid_fraction(indices, window=30):
        if indices is None or len(indices) == 0:
            return np.full(T, np.nan)
        idx = [i for i in indices if i < J]
        valid = (confidence[:, idx] >= conf_thresh).mean(axis=1)
        # Smooth with rolling mean
        if T > window:
            kernel = np.ones(window) / window
            valid = np.convolve(valid, kernel, mode="same")
        return valid

    ub_valid = valid_fraction(ub_indices)
    leg_valid = valid_fraction(leg_indices)

    # Average confidence (upper body)
    ub_idx = [i for i in (ub_indices or []) if i < J]
    avg_conf = confidence[:, ub_idx].mean(axis=1) if ub_idx else np.full(T, np.nan)

    # Scale reference
    motion_path = Path("data/processed") / f"{video_id}_motion_signals.npz"
    motion_arrays, motion_meta = load_signals(motion_path)
    scale = motion_arrays["scale_reference"]

    fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
    fig.suptitle(f"{video_id} — Pose Quality Report", fontsize=14)

    axes[0].plot(times, ub_valid * 100, linewidth=0.5, label="Upper body", color="C0")
    axes[0].plot(times, leg_valid * 100, linewidth=0.5, label="Legs", color="C1")
    axes[0].set_ylabel("Valid keypoints (%)")
    axes[0].legend(fontsize=8)

    axes[1].plot(times, avg_conf, linewidth=0.5, color="C2")
    axes[1].set_ylabel("Avg confidence\n(upper body)")

    scale_times = np.arange(len(scale)) / fps
    axes[2].plot(scale_times, scale, linewidth=0.5, color="C3")
    axes[2].set_ylabel("Scale reference\n(pixels)")
    axes[2].set_xlabel("Time (seconds)")

    plt.tight_layout()
    path = out_dir / "02_pose_quality.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_clip_detail(
    video_id: str, aligned: dict, meta: dict, cut_times: list, nan_margin_sec: float,
    out_dir: Path, clip_start: float | None = None, clip_end: float | None = None,
):
    """Plot 3: 30-second detailed clip view."""
    times = aligned["times"]
    total_motion = aligned["motion_total"]
    common_fps = meta.get("common_fps", 30)

    if clip_start is None:
        # Find the 30s window with most motion activity
        window_frames = int(30 * common_fps)
        if len(total_motion) > window_frames:
            valid = np.nan_to_num(total_motion, nan=0.0)
            cumsum = np.cumsum(valid)
            windowed = cumsum[window_frames:] - cumsum[:-window_frames]
            best_start = int(np.argmax(windowed))
            clip_start = times[best_start]
            clip_end = clip_start + 30.0
        else:
            clip_start = 0.0
            clip_end = float(times[-1])

    time_mask = (times >= clip_start) & (times <= clip_end)
    t = times[time_mask]
    has_text = _has_text(aligned)
    n_rows = 7 if has_text else 5

    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 2 * n_rows), sharex=True)
    fig.suptitle(f"{video_id} — Detail [{clip_start:.1f}s – {clip_end:.1f}s]", fontsize=14)

    row = 0

    axes[row].plot(t, aligned["audio_onset"][time_mask], linewidth=0.8, color="C0")
    axes[row].set_ylabel("Onset strength")
    row += 1

    axes[row].plot(t, aligned["audio_rms"][time_mask], linewidth=0.8, color="C1")
    axes[row].set_ylabel("RMS energy")
    row += 1

    axes[row].plot(t, aligned["audio_f0"][time_mask], linewidth=0.8, color="C2")
    axes[row].set_ylabel("Pitch (Hz)")
    row += 1

    if has_text:
        density_clip = aligned["text_density"][time_mask]
        axes[row].plot(t, density_clip, linewidth=0.8, color="C4")
        onset = aligned["text_onset"][time_mask]
        onset_t = t[onset > 0]
        if len(onset_t) > 0:
            tick_y = float(np.nanmax(density_clip)) if np.any(density_clip > 0) else 1.0
            axes[row].scatter(onset_t, np.full(len(onset_t), tick_y),
                              marker="|", s=20, color="C4", alpha=0.7)
        axes[row].set_ylabel("Char density\n(chars/sec)")
        row += 1

        axes[row].plot(t, aligned["text_breath"][time_mask], linewidth=0.8, color="C5")
        silence = aligned["text_silence_mask"][time_mask].astype(bool)
        _shade_mask(axes[row], t, silence, color="grey", alpha=0.15)
        axes[row].set_ylabel("Breath marks")
        row += 1

    axes[row].plot(t, aligned["motion_total"][time_mask], linewidth=0.8, color="C3")
    axes[row].set_ylabel("Total motion")
    row += 1

    axes[row].plot(t, aligned["motion_hand"][time_mask], linewidth=0.8, color="C0", label="Hand")
    axes[row].plot(t, aligned["motion_torso"][time_mask], linewidth=0.8, color="C1", label="Torso")
    axes[row].set_ylabel("Motion")
    axes[row].legend(fontsize=8)

    axes[-1].set_xlabel("Time (seconds)")

    for ax in axes:
        _add_cut_lines(ax, cut_times, nan_margin_sec)
        ax.set_xlim(clip_start, clip_end)

    plt.tight_layout()
    path = out_dir / "03_clip_detail.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_camera_diagnostic(
    video_id: str, out_dir: Path
):
    """Plot 4: Camera motion diagnostic."""
    motion_path = Path("data/processed") / f"{video_id}_motion_signals.npz"
    motion_arrays, motion_meta = load_signals(motion_path)

    fps = motion_meta["fps"]
    root_disp = motion_arrays["root_displacement"]
    scale = motion_arrays["scale_reference"]

    root_times = np.arange(len(root_disp)) / fps
    scale_times = np.arange(len(scale)) / fps

    fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True)
    fig.suptitle(f"{video_id} — Camera Diagnostic", fontsize=14)

    axes[0].plot(root_times, root_disp, linewidth=0.5, color="C0")
    axes[0].set_ylabel("Root displacement\n(px/sec)")

    valid_scale = scale[~np.isnan(scale)]
    mean_s = np.mean(valid_scale) if len(valid_scale) > 0 else 0
    std_s = np.std(valid_scale) if len(valid_scale) > 0 else 0

    axes[1].plot(scale_times, scale, linewidth=0.5, color="C3")
    axes[1].axhline(mean_s, color="gray", linestyle="--", linewidth=0.8)
    axes[1].axhspan(mean_s - std_s, mean_s + std_s, color="gray", alpha=0.1)
    axes[1].set_ylabel("Scale reference\n(pixels)")
    axes[1].set_xlabel("Time (seconds)")

    cv = motion_meta.get("scale_reference_cv", 0)
    mode = motion_meta.get("root_mode_resolved", "?")
    axes[1].set_title(f"Root mode: {mode} | Scale CV: {cv:.3f}", fontsize=10, loc="right")

    plt.tight_layout()
    path = out_dir / "04_camera_diagnostic.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def process_video(video_id: str, cfg: dict, clip_start: float | None = None, clip_end: float | None = None) -> None:
    aligned_path = Path("data/processed") / f"{video_id}_aligned_signals.npz"
    aligned, meta = load_signals(aligned_path)

    sb_path = require_file(Path("data/processed") / f"{video_id}_shot_boundaries.json")
    with open(sb_path) as f:
        shot_data = json.load(f)

    cut_frames = shot_data["cuts"]
    video_fps = shot_data["video_fps"]
    cut_times = [cf / video_fps for cf in cut_frames]
    nan_margin_sec = cfg["cuts"]["nan_margin"] / video_fps

    out_dir = Path("outputs/figures") / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{video_id}] Generating QA plots...")
    plot_signals_overview(video_id, aligned, meta, cut_times, nan_margin_sec, out_dir)
    plot_pose_quality(video_id, cfg, out_dir)
    plot_clip_detail(video_id, aligned, meta, cut_times, nan_margin_sec, out_dir, clip_start, clip_end)
    plot_camera_diagnostic(video_id, out_dir)
    print(f"[{video_id}] All plots saved to {out_dir}/")


def main():
    parser = base_argparser("Generate QA plots for pipeline output")
    parser.add_argument("--clip_start", type=float, default=None, help="Start time (sec) for detail plot")
    parser.add_argument("--clip_end", type=float, default=None, help="End time (sec) for detail plot")
    args = parser.parse_args()
    cfg = load_config(args.config)

    for vid in resolve_video_ids(cfg, args):
        process_video(vid, cfg, clip_start=args.clip_start, clip_end=args.clip_end)


if __name__ == "__main__":
    main()
