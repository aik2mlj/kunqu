"""Step 3: Run pose estimation on every frame of the video."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from utils import (
    base_argparser,
    load_config,
    get_video_entry,
    resolve_video_ids,
    require_file,
    save_signals,
)


# ---------------------------------------------------------------------------
# MediaPipe backend
# ---------------------------------------------------------------------------

def _extract_mediapipe(video_path: Path, cfg: dict, video_id: str):
    """Extract poses using MediaPipe PoseLandmarker (Tasks API)."""
    try:
        import mediapipe as mp
        from mediapipe.tasks.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
        from mediapipe.tasks import BaseOptions
    except ImportError:
        print(
            "Error: mediapipe not installed. Run: uv sync --extra mediapipe",
            file=sys.stderr,
        )
        sys.exit(1)

    pose_cfg = cfg["pose"]
    conf_thresh = pose_cfg["confidence_threshold"]

    # Open video to get metadata
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    num_joints = 33  # MediaPipe pose landmarks
    keypoints = np.full((total_frames, num_joints, 2), np.nan, dtype=np.float32)
    confidence = np.full((total_frames, num_joints), 0.0, dtype=np.float32)
    frame_valid = np.zeros(total_frames, dtype=bool)

    # Find model file — check common locations
    model_paths = [
        Path("models/pose_landmarker_heavy.task"),
        Path("pose_landmarker_heavy.task"),
    ]
    model_path = None
    for p in model_paths:
        if p.exists():
            model_path = str(p)
            break

    if model_path is None:
        print(
            "Warning: pose_landmarker_heavy.task not found, using lite model via default.",
            file=sys.stderr,
        )
        # Fall back to creating landmarker without explicit model path
        # User should download the model file
        print(
            "Error: Please download a MediaPipe pose landmarker model .task file "
            "and place it in models/ directory.",
            file=sys.stderr,
        )
        sys.exit(1)

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        for t in tqdm(range(total_frames), desc=f"[{video_id}] MediaPipe poses"):
            ret, frame = cap.read()
            if not ret:
                break

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamp_ms = int(t * 1000 / fps)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                landmarks = result.pose_landmarks[0]
                frame_valid[t] = True
                for j, lm in enumerate(landmarks):
                    if j >= num_joints:
                        break
                    keypoints[t, j, 0] = lm.x * width
                    keypoints[t, j, 1] = lm.y * height
                    vis = lm.visibility if hasattr(lm, "visibility") and lm.visibility is not None else 1.0
                    confidence[t, j] = vis
                    if vis < conf_thresh:
                        keypoints[t, j] = np.nan

                if len(result.pose_landmarks) > 1:
                    print(f"[{video_id}] Warning: {len(result.pose_landmarks)} people detected at frame {t}")
            # else: frame_valid stays False, keypoints stay NaN

    cap.release()

    metadata = {
        "video_id": video_id,
        "model_name": "mediapipe",
        "num_joints": num_joints,
        "video_fps": fps,
        "total_frames": total_frames,
        "frame_width": width,
        "frame_height": height,
    }

    return keypoints, confidence, frame_valid, metadata


# ---------------------------------------------------------------------------
# DWPose backend
# ---------------------------------------------------------------------------

def _extract_dwpose(video_path: Path, cfg: dict, video_id: str):
    """Extract poses using DWPose (MMPose)."""
    try:
        from mmpose.apis import MMPoseInferencer
    except ImportError:
        print(
            "Error: mmpose not installed. Install DWPose dependencies:\n"
            "  pip install openmim && mim install mmengine mmcv mmdet mmpose",
            file=sys.stderr,
        )
        sys.exit(1)

    pose_cfg = cfg["pose"]
    conf_thresh = pose_cfg["confidence_threshold"]
    batch_size = pose_cfg["batch_size"]
    strategy = pose_cfg["multi_person_strategy"]

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    num_joints = 133  # COCO-WholeBody
    keypoints = np.full((total_frames, num_joints, 2), np.nan, dtype=np.float32)
    confidence = np.full((total_frames, num_joints), 0.0, dtype=np.float32)
    frame_valid = np.zeros(total_frames, dtype=bool)

    inferencer = MMPoseInferencer(
        pose2d=pose_cfg["dwpose_pose_config"],
        pose2d_weights=pose_cfg["dwpose_pose_checkpoint"],
        det_model=pose_cfg["dwpose_det_config"],
        det_weights=pose_cfg["dwpose_det_checkpoint"],
    )

    # Read frames in batches
    frames_buffer = []
    frame_indices = []

    pbar = tqdm(total=total_frames, desc=f"[{video_id}] DWPose")

    def process_batch(frames, indices):
        results = inferencer(frames, batch_size=len(frames))
        for idx, result in zip(indices, results):
            preds = result.get("predictions", [])
            if not preds or len(preds) == 0:
                continue

            # Select person based on strategy
            if len(preds) > 1:
                print(f"[{video_id}] Warning: {len(preds)} people at frame {idx}")
                if strategy == "largest_bbox":
                    areas = []
                    for p in preds:
                        bbox = p.get("bbox", [[0, 0, 0, 0]])[0]
                        areas.append((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                    person = preds[np.argmax(areas)]
                else:  # highest_confidence
                    avg_confs = [np.mean(p["keypoint_scores"]) for p in preds]
                    person = preds[np.argmax(avg_confs)]
            else:
                person = preds[0]

            kps = np.array(person["keypoints"])
            scores = np.array(person["keypoint_scores"])

            n = min(num_joints, len(kps))
            frame_valid[idx] = True
            keypoints[idx, :n] = kps[:n, :2]
            confidence[idx, :n] = scores[:n]
            # Mask low-confidence
            low = scores[:n] < conf_thresh
            keypoints[idx, :n][low] = np.nan

    for t in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames_buffer.append(frame)
        frame_indices.append(t)
        pbar.update(1)

        if len(frames_buffer) >= batch_size:
            process_batch(frames_buffer, frame_indices)
            frames_buffer = []
            frame_indices = []

    if frames_buffer:
        process_batch(frames_buffer, frame_indices)

    pbar.close()
    cap.release()

    metadata = {
        "video_id": video_id,
        "model_name": "dwpose",
        "num_joints": num_joints,
        "video_fps": fps,
        "total_frames": total_frames,
        "frame_width": width,
        "frame_height": height,
    }

    return keypoints, confidence, frame_valid, metadata


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_video(video_id: str, cfg: dict, model_override: str | None = None) -> None:
    entry = get_video_entry(cfg, video_id)
    video_path = require_file(entry["path"])

    model = model_override or cfg["pose"]["model"]
    print(f"[{video_id}] Extracting poses with {model}...")

    if model == "mediapipe":
        keypoints, conf, frame_valid, metadata = _extract_mediapipe(video_path, cfg, video_id)
    elif model == "dwpose":
        keypoints, conf, frame_valid, metadata = _extract_dwpose(video_path, cfg, video_id)
    else:
        print(f"Error: unknown pose model '{model}'", file=sys.stderr)
        sys.exit(1)

    out_path = Path("data/poses") / f"{video_id}_keypoints.npz"
    save_signals(
        out_path, metadata,
        keypoints=keypoints,
        confidence=conf,
        frame_valid=frame_valid,
    )

    valid_pct = frame_valid.sum() / len(frame_valid) * 100
    print(
        f"[{video_id}] Pose extraction complete: {metadata['total_frames']} frames, "
        f"{valid_pct:.1f}% valid detections"
    )


def main():
    parser = base_argparser("Extract pose keypoints from video(s)")
    parser.add_argument(
        "--model", type=str, default=None,
        choices=["mediapipe", "dwpose"],
        help="Override pose model from config",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    for vid in resolve_video_ids(cfg, args):
        process_video(vid, cfg, model_override=args.model)


if __name__ == "__main__":
    main()
