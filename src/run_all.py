"""Batch runner: execute the full pipeline for all configured videos."""

from __future__ import annotations

import sys
import traceback

from utils import base_argparser, load_config, resolve_video_ids, PipelineError

# Import each step's process_video function
from detect_cuts import process_video as step_detect_cuts
from extract_audio import process_video as step_extract_audio
from extract_poses import process_video as step_extract_poses
from compute_motion import process_video as step_compute_motion
from align_signals import process_video as step_align_signals
from visualize import process_video as step_visualize

STEPS = [
    ("detect_cuts", step_detect_cuts),
    ("extract_audio", step_extract_audio),
    ("extract_poses", step_extract_poses),
    ("compute_motion", step_compute_motion),
    ("align_signals", step_align_signals),
    ("visualize", step_visualize),
]


def main():
    parser = base_argparser("Run full pipeline for video(s)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    video_ids = resolve_video_ids(cfg, args)

    results = {}

    for vid in video_ids:
        print(f"\n{'='*60}")
        print(f"Processing: {vid}")
        print(f"{'='*60}")

        failed_step = None
        for step_name, step_fn in STEPS:
            try:
                print(f"\n--- {step_name} ---")
                step_fn(vid, cfg)
            except PipelineError as e:
                print(f"  Error: {e}", file=sys.stderr)
                failed_step = step_name
                break
            except Exception:
                traceback.print_exc()
                failed_step = step_name
                break

        if failed_step:
            results[vid] = f"FAILED at {failed_step}"
        else:
            results[vid] = "all steps passed"

    # Summary
    print(f"\n{'='*60}")
    print("Pipeline complete.")
    for vid, status in results.items():
        marker = "OK" if "passed" in status else "FAIL"
        print(f"  {vid}: [{marker}] {status}")

    if any("FAILED" in s for s in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
