#!/usr/bin/env python3
"""
Merge 6 annotation JSON parts into a single unified annotation file.

The 6 files (xunmeng_0.json through xunmeng_5.json) are segments annotated by
different people. Annotators started/ended at sentence boundaries, so consecutive
segments have overlapping characters. This script detects those overlaps by matching
character text sequences and keeps the earlier annotator's version.

Special cases:
  - Files 3 and 5 contain an identical placeholder "intro" section (~12-120s)
    with slow per-character annotations (春江花月夜...). This is discarded.
  - Action annotations from the intro time range are also discarded.
  - Breath marks are deduplicated by time proximity across all files.

Usage:
    uv run python src/merge_annotations.py [input_dir] [output_path]

    Defaults:
        input_dir:   data/annotations
        output_path: data/annotations/xunmeng_annotation.json
"""

import json
import sys
from pathlib import Path


def load(path):
    with open(path) as f:
        return json.load(f)


def split_by_gap(chars, gap_threshold=100):
    """Split a sorted char list into groups separated by time gaps > threshold."""
    if not chars:
        return []
    groups = [[chars[0]]]
    for i in range(1, len(chars)):
        if chars[i]["startTime"] - chars[i - 1]["endTime"] > gap_threshold:
            groups.append([])
        groups[-1].append(chars[i])
    return groups


def find_overlap_count(earlier_chars, later_chars):
    """Find how many chars at the start of later_chars duplicate the end of earlier_chars.

    Matches by character text sequence in the time-overlapping region.
    Returns the number of later_chars to skip.
    """
    if not earlier_chars or not later_chars:
        return 0

    earlier_end = earlier_chars[-1]["endTime"]
    later_start = later_chars[0]["startTime"]

    # No time overlap — nothing to dedup
    if later_start >= earlier_end + 5.0:
        return 0

    # Count later chars whose startTime falls in the overlap zone
    n_candidates = 0
    for c in later_chars:
        if c["startTime"] < earlier_end + 2.0:
            n_candidates += 1
        else:
            break

    if n_candidates == 0:
        return 0

    # Try matching char text sequences, longest first
    for length in range(n_candidates, 0, -1):
        later_seq = [c["char"] for c in later_chars[:length]]
        search_start = max(0, len(earlier_chars) - length - 10)
        for start in range(search_start, len(earlier_chars) - length + 1):
            earlier_seq = [c["char"] for c in earlier_chars[start : start + length]]
            if earlier_seq == later_seq:
                return length

    return 0


def get_breath_points(project):
    """Extract breath track points from builtinTracks."""
    for track in project.get("builtinTracks", []):
        for apt in track.get("attachedPointTracks", []):
            if "呼吸" in apt.get("name", ""):
                return sorted(apt.get("points", []), key=lambda p: p["time"])
    return []


def dedup_breaths(all_points, tolerance=1.0):
    """Remove duplicate breath points within tolerance seconds of each other."""
    if not all_points:
        return []
    sorted_pts = sorted(all_points, key=lambda p: p["time"])
    result = [sorted_pts[0]]
    for pt in sorted_pts[1:]:
        if pt["time"] - result[-1]["time"] > tolerance:
            result.append(pt)
    return result


def build_subtitle_lines(chars):
    """Reconstruct subtitle lines from merged characters, grouped by lineId."""
    line_chars = {}  # lineId -> list of chars
    line_order = []  # preserve first-seen order

    for c in chars:
        lid = c["lineId"]
        if lid not in line_chars:
            line_chars[lid] = []
            line_order.append(lid)
        line_chars[lid].append(c)

    lines = []
    for lid in line_order:
        group = line_chars[lid]
        lines.append(
            {
                "id": lid,
                "text": "".join(c["char"] for c in group),
                "startTime": group[0]["startTime"],
                "endTime": group[-1]["endTime"],
            }
        )
    # Sort by startTime for consistency
    lines.sort(key=lambda l: l["startTime"])
    return lines


def main():
    input_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/annotations")
    output_path = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else input_dir / "xunmeng_annotation.json"
    )

    # Load all 6 files
    parts = []
    for i in range(6):
        path = input_dir / f"xunmeng_{i}.json"
        data = load(path)
        parts.append(data["project"])
        chars = sorted(
            data["project"].get("characterAnnotations", []),
            key=lambda c: c["startTime"],
        )
        print(
            f"Loaded xunmeng_{i}.json: {len(chars)} chars "
            f"[{chars[0]['startTime']:.1f}-{chars[-1]['endTime']:.1f}s]"
        )

    # ── Phase 1: separate intro from main segments ──────────────────────
    # Files 3 and 5 contain a placeholder "intro" section (~12-120s) with slow
    # per-char annotations (春江花月夜...) separated from the main content by a
    # large time gap. Discard the intro; keep only the main section.
    main_segments = []  # (file_idx, chars)

    for i, p in enumerate(parts):
        chars = sorted(
            p.get("characterAnnotations", []), key=lambda c: c["startTime"]
        )

        if i in (3, 5):
            groups = split_by_gap(chars, gap_threshold=100)
            if len(groups) == 2:
                intro_text = "".join(c["char"] for c in groups[0])
                print(
                    f"  file_{i}: discarding intro placeholder "
                    f"({len(groups[0])} chars, \"{intro_text}\")"
                )
                main_segments.append((i, groups[1]))
            else:
                # No gap found — treat entire file as main
                main_segments.append((i, chars))
        else:
            main_segments.append((i, chars))

    # Sort main segments by start time
    main_segments.sort(key=lambda s: s[1][0]["startTime"])

    # ── Phase 2: merge main segments, deduplicating overlaps ────────────
    print("\nMerging main segments:")
    merged_main = []

    for file_idx, chars in main_segments:
        label = f"file_{file_idx}"
        text_head = "".join(c["char"] for c in chars[:5])
        text_tail = "".join(c["char"] for c in chars[-3:])

        skip = find_overlap_count(merged_main, chars)

        if skip > 0:
            skipped_text = "".join(c["char"] for c in chars[:skip])
            print(
                f"  {label} [{chars[0]['startTime']:.1f}-{chars[-1]['endTime']:.1f}s]: "
                f"skipped {skip} overlapping chars \"{skipped_text}\", "
                f"added {len(chars) - skip}"
            )
        else:
            print(
                f"  {label} [{chars[0]['startTime']:.1f}-{chars[-1]['endTime']:.1f}s]: "
                f"no overlap, added all {len(chars)} chars"
            )

        merged_main.extend(chars[skip:])

    # ── Phase 3: finalize character list ───────────────────────────────
    all_chars = merged_main
    all_chars.sort(key=lambda c: c["startTime"])

    # Reassign IDs: new sequential char IDs, remap lineIds for consistency
    line_id_map = {}
    for i, c in enumerate(all_chars):
        c["id"] = f"char-{i + 1}"
        old_line = c["lineId"]
        if old_line not in line_id_map:
            line_id_map[old_line] = f"line-{len(line_id_map) + 1}"
        c["lineId"] = line_id_map[old_line]

    # ── Phase 4: merge breath marks ─────────────────────────────────────
    all_breaths = []
    for p in parts:
        all_breaths.extend(get_breath_points(p))
    total_raw = len(all_breaths)
    merged_breaths = dedup_breaths(all_breaths, tolerance=1.0)
    print(f"\nBreath marks: {total_raw} raw → {len(merged_breaths)} after dedup")

    # ── Phase 5: merge action annotations ───────────────────────────────
    # Discard actions from the intro time range (< 130s), which are placeholders
    # present in files 3 and 5.
    char_start = all_chars[0]["startTime"] if all_chars else 0
    all_actions = []
    seen = set()
    skipped_actions = 0
    for p in parts:
        for a in p.get("actionAnnotations", []):
            if a["endTime"] < char_start:
                skipped_actions += 1
                continue
            key = (round(a["startTime"], 1), round(a["endTime"], 1), a["label"])
            if key not in seen:
                seen.add(key)
                all_actions.append(a)
    all_actions.sort(key=lambda a: a["startTime"])
    for i, a in enumerate(all_actions):
        a["id"] = f"action-{i + 1}"
    if skipped_actions:
        print(f"  Discarded {skipped_actions} action annotations from intro range")

    # ── Phase 6: build subtitle lines from merged chars ─────────────────
    merged_lines = build_subtitle_lines(all_chars)

    # ── Phase 7: write output ───────────────────────────────────────────
    output = {
        "version": 2,
        "project": {
            "video": parts[0]["video"],
            "subtitleLines": merged_lines,
            "characterAnnotations": all_chars,
            "actionAnnotations": all_actions,
            "builtinTracks": [
                {
                    "id": "character-track",
                    "name": "逐字文字轨",
                    "type": "character",
                    "options": [
                        "普通唱",
                        "拖腔",
                        "顿音",
                        "装饰音",
                        "念白式",
                        "其他",
                    ],
                    "attachedPointTracks": [
                        {"name": "呼吸轨", "points": merged_breaths}
                    ],
                    "attachedPointTracksExpanded": True,
                    "snapToWaveformKeypoints": False,
                },
                {
                    "id": "hand-action",
                    "name": "手部动作轨",
                    "type": "action",
                    "options": [
                        "抬手",
                        "落手",
                        "指向",
                        "翻腕",
                        "水袖动作",
                        "其他",
                    ],
                    "attachedPointTracks": [],
                    "attachedPointTracksExpanded": False,
                    "snapToWaveformKeypoints": False,
                },
                {
                    "id": "body-action",
                    "name": "肢体动作轨",
                    "type": "action",
                    "options": [
                        "转身",
                        "移步",
                        "屈伸",
                        "亮相",
                        "前倾",
                        "后仰",
                        "其他",
                    ],
                    "attachedPointTracks": [],
                    "attachedPointTracksExpanded": False,
                    "snapToWaveformKeypoints": False,
                },
            ],
            "customTracks": [],
            "activeTrackOrder": [
                "character-track",
                "hand-action",
                "body-action",
            ],
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Merged annotation written to {output_path}")
    print(f"  Characters:   {len(all_chars)}")
    print(f"  Lines:        {len(merged_lines)}")
    print(f"  Breath marks: {len(merged_breaths)}")
    print(f"  Actions:      {len(all_actions)}")
    print(
        f"  Time range:   {all_chars[0]['startTime']:.1f} - "
        f"{all_chars[-1]['endTime']:.1f}s"
    )


if __name__ == "__main__":
    main()
