"""Step 4: Compute camera-invariant motion signals from pose keypoints."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt

from utils import (
    base_argparser,
    load_config,
    get_video_entry,
    get_joint_groups,
    resolve_video_ids,
    require_file,
    load_signals,
    save_signals,
)


def _interpolate_short_gaps(arr: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly interpolate NaN gaps of <= max_gap frames, per joint per dim."""
    out = arr.copy()
    T = len(out)
    if T < 2:
        return out

    valid = ~np.isnan(out)
    # Find gap runs
    i = 0
    while i < T:
        if not valid[i]:
            # Start of a gap
            j = i
            while j < T and not valid[j]:
                j += 1
            gap_len = j - i
            if gap_len <= max_gap and i > 0 and j < T:
                # Interpolate
                t_start = i - 1
                t_end = j
                for k in range(i, j):
                    alpha = (k - t_start) / (t_end - t_start)
                    out[k] = out[t_start] * (1 - alpha) + out[t_end] * alpha
            i = j
        else:
            i += 1
    return out


def _medfilt_with_nan(arr: np.ndarray, kernel: int) -> np.ndarray:
    """Apply median filter, skipping NaN segments."""
    out = arr.copy()
    valid = ~np.isnan(arr)

    # Find contiguous valid segments and filter each
    i = 0
    T = len(arr)
    while i < T:
        if valid[i]:
            j = i
            while j < T and valid[j]:
                j += 1
            seg_len = j - i
            if seg_len >= kernel:
                out[i:j] = medfilt(arr[i:j], kernel_size=kernel)
            i = j
        else:
            i += 1
    return out


def _smooth_with_nan(signal: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smooth a signal, preserving NaN positions."""
    nan_mask = np.isnan(signal)
    filled = signal.copy()
    filled[nan_mask] = 0.0
    smoothed = gaussian_filter1d(filled, sigma=sigma)
    smoothed[nan_mask] = np.nan
    return smoothed


def compute_motion(video_id: str, cfg: dict) -> None:
    # Load keypoints
    kp_path = Path("data/poses") / f"{video_id}_keypoints.npz"
    kp_arrays, kp_meta = load_signals(kp_path)

    keypoints = kp_arrays["keypoints"]          # (T, J, 2)
    kp_confidence = kp_arrays["confidence"]      # (T, J)
    frame_valid = kp_arrays["frame_valid"]       # (T,)

    pose_model = kp_meta["model_name"]
    fps = kp_meta["video_fps"]
    T, J, _ = keypoints.shape

    # Load shot boundaries
    sb_path = require_file(
        Path("data/processed") / f"{video_id}_shot_boundaries.json",
        hint="Run detect_cuts.py first.",
    )
    with open(sb_path) as f:
        shot_data = json.load(f)
    cut_frames = shot_data["cuts"]
    nan_margin = cfg["cuts"]["nan_margin"]

    # Get joint group definitions
    joint_groups = get_joint_groups(cfg, pose_model)
    motion_cfg = cfg["motion"]

    # Verify joint count
    expected_joints = {"dwpose": 133, "mediapipe": 33}
    if pose_model in expected_joints and J != expected_joints[pose_model]:
        print(
            f"Warning: expected {expected_joints[pose_model]} joints for {pose_model}, "
            f"got {J}",
            file=sys.stderr,
        )

    # --- Step 2: Mask low-confidence keypoints ---
    conf_thresh = cfg["pose"]["confidence_threshold"]
    low_conf = kp_confidence < conf_thresh
    keypoints[low_conf] = np.nan

    # --- Step 3: Resolve root mode ---
    root_mode = motion_cfg["root_mode"]
    sl = joint_groups["shoulder_left"]
    sr = joint_groups["shoulder_right"]
    hl = joint_groups["hip_left"]
    hr = joint_groups["hip_right"]
    neck = joint_groups["neck_proxy"]

    hip_nan = np.isnan(keypoints[:, hl, 0]) | np.isnan(keypoints[:, hr, 0])
    hip_nan_frac = hip_nan.sum() / T

    if root_mode == "auto":
        if hip_nan_frac > 0.5:
            root_mode_resolved = "shoulder"
            print(
                f"[{video_id}] Auto root selection: using shoulders "
                f"(hips unavailable in {hip_nan_frac*100:.0f}% of frames)"
            )
        else:
            root_mode_resolved = "hip"
            print(
                f"[{video_id}] Auto root selection: using hips "
                f"(available in {(1-hip_nan_frac)*100:.0f}% of frames)"
            )
    else:
        root_mode_resolved = root_mode

    # --- Step 4: Interpolate short gaps ---
    max_interp = motion_cfg["gap_max_interpolate"]
    for j in range(J):
        for d in range(2):
            keypoints[:, j, d] = _interpolate_short_gaps(keypoints[:, j, d], max_interp)

    # --- Step 5: Median filter ---
    smooth_win = motion_cfg["smoothing_window"]
    if smooth_win % 2 == 0:
        smooth_win += 1  # medfilt requires odd kernel
    for j in range(J):
        for d in range(2):
            keypoints[:, j, d] = _medfilt_with_nan(keypoints[:, j, d], smooth_win)

    # --- Step 6: Body-centric coordinates ---
    if root_mode_resolved == "shoulder":
        root = (keypoints[:, sl] + keypoints[:, sr]) / 2  # (T, 2)
    else:
        root = (keypoints[:, hl] + keypoints[:, hr]) / 2  # (T, 2)

    root_pixel = root.copy()  # save for diagnostic before subtraction

    keypoints_local = keypoints.copy()
    for j in range(J):
        keypoints_local[:, j] -= root

    # --- Step 7: Scale normalization ---
    if root_mode_resolved == "shoulder":
        scale = np.linalg.norm(keypoints[:, sl] - keypoints[:, sr], axis=1)  # (T,)
    else:
        neck_pos = keypoints[:, neck]
        scale = np.linalg.norm(neck_pos - root, axis=1)  # (T,)

    # Invalidate bad scale values
    bad_scale = np.isnan(scale) | (scale < 5.0)
    scale[bad_scale] = np.nan

    keypoints_norm = keypoints_local.copy()
    for j in range(J):
        keypoints_norm[:, j, 0] /= scale
        keypoints_norm[:, j, 1] /= scale

    # Where scale was bad, all keypoints are NaN (already handled by division by NaN)

    # --- Step 8: Compute velocity respecting shot boundaries ---
    # Build cut mask: True at frames within nan_margin of any cut
    cut_mask = np.zeros(T, dtype=bool)
    for cf in cut_frames:
        lo = max(0, cf - nan_margin)
        hi = min(T, cf + nan_margin + 1)
        cut_mask[lo:hi] = True

    # Per-joint velocity: (T-1, J)
    diff = keypoints_norm[1:] - keypoints_norm[:-1]  # (T-1, J, 2)
    v = np.linalg.norm(diff, axis=2) * fps  # (T-1, J) in body-proportions/sec

    # Set velocity to NaN at/near cut boundaries
    for t in range(T - 1):
        if cut_mask[t] or cut_mask[t + 1]:
            v[t, :] = np.nan

    # Root displacement in pixel space (diagnostic)
    root_diff = root_pixel[1:] - root_pixel[:-1]
    root_displacement = np.linalg.norm(root_diff, axis=1) * fps

    # --- Step 9: Aggregate by body region ---
    def aggregate_group(joint_indices):
        if joint_indices is None:
            # full_body: use all joints
            return np.nanmean(v, axis=1)
        idx = [i for i in joint_indices if i < J]
        if not idx:
            return np.full(T - 1, np.nan)
        return np.nanmean(v[:, idx], axis=1)

    # Region groups to compute
    region_keys = {
        "total_motion": joint_groups.get("full_body"),
        "hand_left_motion": joint_groups.get("hand_left"),
        "hand_right_motion": joint_groups.get("hand_right"),
        "torso_motion": joint_groups.get("torso"),
        "upper_body_motion": joint_groups.get("upper_body"),
        "head_motion": joint_groups.get("head"),
    }

    signals = {}
    for name, indices in region_keys.items():
        signals[name] = aggregate_group(indices)

    # Combined hand motion = mean of left and right
    signals["hand_motion"] = np.nanmean(
        np.stack([signals["hand_left_motion"], signals["hand_right_motion"]]),
        axis=0,
    )

    # --- Step 10: Gaussian smooth ---
    sigma_frames = motion_cfg["velocity_sigma"] * fps
    for name in signals:
        signals[name] = _smooth_with_nan(signals[name], sigma_frames)

    # Times for velocity signals
    times = (np.arange(T - 1) + 0.5) / fps

    # --- Step 11: Diagnostics ---
    total_nan_frac = np.isnan(signals["total_motion"]).sum() / (T - 1)
    scale_valid = scale[~np.isnan(scale)]
    scale_cv = float(np.std(scale_valid) / np.mean(scale_valid)) if len(scale_valid) > 0 else float("nan")

    print(f"\n[{video_id}] Camera analysis:")
    print(f"  - Root mode: {root_mode_resolved}" +
          (f" (auto — hips unavailable in {hip_nan_frac*100:.0f}% of frames)" if root_mode == "auto" else ""))
    print(f"  - Detected cuts: {len(cut_frames)}")
    print(f"  - Scale reference CV: {scale_cv:.3f}" +
          (" (low -> camera distance is stable)" if scale_cv < 0.1 else ""))
    mean_root_disp = np.nanmean(root_displacement)
    print(f"  - Mean root displacement: {mean_root_disp:.1f} px/sec")
    print(f"  - Motion NaN fraction: {total_nan_frac*100:.1f}%")
    if total_nan_frac > 0.15:
        print(f"  WARNING: NaN fraction exceeds 15%")
    print(f"  Recommendation: body-centric signals are {'reliable' if total_nan_frac < 0.15 else 'noisy — inspect QA plots'}.")

    # Save
    metadata = {
        "video_id": video_id,
        "fps": fps,
        "pose_model": pose_model,
        "joint_groups_used": pose_model,
        "smoothing_params": {
            "smoothing_window": smooth_win,
            "velocity_sigma": motion_cfg["velocity_sigma"],
            "gap_max_interpolate": max_interp,
        },
        "coordinate_mode": "body_centric_normalized",
        "root_mode_resolved": root_mode_resolved,
        "root_joint": "shoulder_midpoint" if root_mode_resolved == "shoulder" else "hip_midpoint",
        "scale_reference_type": "shoulder_width" if root_mode_resolved == "shoulder" else "torso_length",
        "scale_reference_cv": round(scale_cv, 4),
        "num_cuts": len(cut_frames),
        "cut_frames": cut_frames,
        "nan_fraction": round(float(total_nan_frac), 4),
        "hip_nan_fraction": round(float(hip_nan_frac), 4),
    }

    out_path = Path("data/processed") / f"{video_id}_motion_signals.npz"
    save_signals(
        out_path, metadata,
        times=times,
        root_displacement=root_displacement,
        scale_reference=scale,
        **signals,
    )
    print(f"[{video_id}] Motion signals saved to {out_path}")


def process_video(video_id: str, cfg: dict) -> None:
    compute_motion(video_id, cfg)


def main():
    parser = base_argparser("Compute motion signals from pose keypoints")
    args = parser.parse_args()
    cfg = load_config(args.config)

    for vid in resolve_video_ids(cfg, args):
        process_video(vid, cfg)


if __name__ == "__main__":
    main()
