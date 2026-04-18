"""Batch runner: execute the full pipeline for all configured videos."""

from __future__ import annotations

import sys
import traceback

from utils import base_argparser, load_config, resolve_video_ids, PipelineError

# Import each step's process_video function
from detect_cuts import process_video as step_detect_cuts
from extract_text import process_video as step_extract_text
from extract_audio import process_video as step_extract_audio
from extract_poses import process_video as step_extract_poses
from compute_motion import process_video as step_compute_motion
from align_signals import process_video as step_align_signals
from visualize import process_video as step_visualize

# Steps 1-3 (detect_cuts, extract_text, extract_audio) have no dependencies
# on each other but run sequentially here for simplicity.
# extract_text is optional — skipped if no annotation file is configured.
STEPS = [
    ("detect_cuts", step_detect_cuts, False),
    ("extract_text", step_extract_text, True),   # optional
    ("extract_audio", step_extract_audio, False),
    ("extract_poses", step_extract_poses, False),
    ("compute_motion", step_compute_motion, False),
    ("align_signals", step_align_signals, False),
    ("visualize", step_visualize, False),
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
        n_passed = 0
        n_skipped = 0
        for step_name, step_fn, optional in STEPS:
            try:
                print(f"\n--- {step_name} ---")
                step_fn(vid, cfg)
                n_passed += 1
            except PipelineError as e:
                if optional:
                    print(f"  Skipped (optional): {e}")
                    n_skipped += 1
                    continue
                print(f"  Error: {e}", file=sys.stderr)
                failed_step = step_name
                break
            except Exception:
                if optional:
                    traceback.print_exc()
                    print(f"  Skipped (optional step failed)")
                    n_skipped += 1
                    continue
                traceback.print_exc()
                failed_step = step_name
                break

        if failed_step:
            results[vid] = f"FAILED at {failed_step}"
        else:
            skip_note = f", {n_skipped} skipped" if n_skipped else ""
            results[vid] = f"all steps passed ({n_passed}/{len(STEPS)}{skip_note})"

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
