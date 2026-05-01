"""Step 1: Detect camera cuts / shot boundaries in a video."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

from utils import base_argparser, load_config, get_video_entry, resolve_video_ids, require_file


def detect_cuts(video_path: Path, cfg: dict, video_id: str) -> dict:
    cuts_cfg = cfg["cuts"]
    threshold = cuts_cfg["threshold"]

    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    print(f"[{video_id}] Detecting cuts (threshold={threshold})...")
    scene_manager.detect_scenes(video, show_progress=True)
    scene_list = scene_manager.get_scene_list()

    video_fps = video.frame_rate
    total_frames = video.duration.get_frames()

    # Build cut frame indices (the first frame of each new scene, except the very first)
    cut_frames = []
    segments = []

    if len(scene_list) == 0:
        # No scenes detected at all — treat as single shot
        segments.append({
            "start_frame": 0,
            "end_frame": total_frames - 1,
            "duration_sec": round(total_frames / video_fps, 2),
        })
    else:
        for i, (start, end) in enumerate(scene_list):
            sf = start.get_frames()
            ef = end.get_frames() - 1  # end is exclusive
            if i > 0:
                cut_frames.append(sf)
            segments.append({
                "start_frame": sf,
                "end_frame": ef,
                "duration_sec": round((ef - sf + 1) / video_fps, 2),
            })

    result = {
        "video_id": video_id,
        "cuts": cut_frames,
        "segments": segments,
        "total_frames": total_frames,
        "video_fps": video_fps,
        "num_cuts": len(cut_frames),
    }

    # Logging
    if len(cut_frames) == 0:
        print(f"[{video_id}] No cuts detected — video is a single continuous shot.")
    else:
        print(f"[{video_id}] Detected {len(cut_frames)} cuts at frames: {cut_frames}")
        avg_seg_dur = sum(s["duration_sec"] for s in segments) / len(segments)
        if avg_seg_dur < 60:
            print(
                f"[{video_id}] WARNING: Average segment duration is {avg_seg_dur:.1f}s "
                f"(<60s). Video may be too heavily edited for reliable motion analysis."
            )

    return result


def process_video(video_id: str, cfg: dict) -> None:
    entry = get_video_entry(cfg, video_id)
    video_path = require_file(entry["path"], hint=f"Place the video file for '{video_id}' there.")

    result = detect_cuts(video_path, cfg, video_id)

    out_path = Path("data/processed") / f"{video_id}_shot_boundaries.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{video_id}] Shot boundaries saved to {out_path}")


def main():
    parser = base_argparser("Detect camera cuts in video(s)")
    args = parser.parse_args()
    cfg = load_config(args.config)

    for vid in resolve_video_ids(cfg, args):
        process_video(vid, cfg)


if __name__ == "__main__":
    main()
