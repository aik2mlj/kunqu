"""
Convert SOFA TextGrid forced-alignment output back into the annotation JSON
format used by the labeling tool.

Reads an original annotation JSON (for subtitleLines text and timing) and a
directory of TextGrid files, producing a new JSON with character-level
annotations derived from the TextGrid word-tier intervals.

Usage:
    python data/textgrid_to_annotation.py \
        --annotation data/xunmeng/xunmeng_annotation.json \
        --textgrid-dir segments/xunmeng_vocals/TextGrid \
        --output data/xunmeng/xunmeng_vocals_annotation.json \
        --padding 0.1
"""

import argparse
import json
import os
import re
import uuid


def parse_textgrid(path):
    """
    Parse a Praat long-format TextGrid file.

    Returns a dict mapping tier name -> list of (xmin, xmax, text) tuples.
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    tiers = {}
    tier_blocks = re.split(r'item\s*\[\d+\]\s*:', content)

    for block in tier_blocks[1:]:
        name_match = re.search(r'name\s*=\s*"([^"]+)"', block)
        if not name_match:
            continue
        tier_name = name_match.group(1)

        intervals = []
        interval_blocks = re.split(r'intervals\s*\[\d+\]\s*:', block)
        for iblock in interval_blocks[1:]:
            xmin_m = re.search(r'xmin\s*=\s*([\d.eE+-]+)', iblock)
            xmax_m = re.search(r'xmax\s*=\s*([\d.eE+-]+)', iblock)
            text_m = re.search(r'text\s*=\s*"([^"]*)"', iblock)
            if xmin_m and xmax_m and text_m:
                intervals.append((
                    float(xmin_m.group(1)),
                    float(xmax_m.group(1)),
                    text_m.group(1),
                ))

        tiers[tier_name] = intervals

    return tiers


def extract_pinyin_intervals(tiers):
    """
    From parsed TextGrid tiers, return the word-tier intervals that are
    actual pinyin syllables (filtering out AP and SP markers).
    """
    words = tiers.get("words", [])
    return [(xmin, xmax, text) for xmin, xmax, text in words
            if text not in ("AP", "SP")]


def is_chinese_char(ch):
    cp = ord(ch)
    return (0x4E00 <= cp <= 0x9FFF or      # CJK Unified
            0x3400 <= cp <= 0x4DBF or      # CJK Extension A
            0x20000 <= cp <= 0x2A6DF or    # CJK Extension B
            0xF900 <= cp <= 0xFAFF)        # CJK Compatibility


def match_chars_to_pinyin(text, pinyin_intervals, line_id):
    """
    Walk through Chinese characters in text and pair each with the next
    pinyin interval. Returns list of (char, pinyin, xmin, xmax) and warnings.
    """
    chinese_chars = [ch for ch in text if is_chinese_char(ch)]
    warnings = []

    if len(chinese_chars) != len(pinyin_intervals):
        warnings.append(
            f"{line_id}: character count ({len(chinese_chars)}) != "
            f"pinyin count ({len(pinyin_intervals)}): "
            f"chars={''.join(chinese_chars)}, "
            f"pinyin={' '.join(p for _, _, p in pinyin_intervals)}"
        )

    pairs = []
    n = min(len(chinese_chars), len(pinyin_intervals))
    for i in range(n):
        xmin, xmax, pinyin = pinyin_intervals[i]
        pairs.append((chinese_chars[i], pinyin, xmin, xmax))

    return pairs, warnings


def find_textgrid_for_line(textgrid_dir, line_id):
    """
    Find the TextGrid file matching a given line_id.

    Supports both old naming ({NNN}_{lineId}.TextGrid) and new naming
    with embedded timestamps ({NNN}_{lineId}_t{ms}.TextGrid).
    """
    for fname in os.listdir(textgrid_dir):
        if not fname.endswith(".TextGrid"):
            continue
        base = fname[:-len(".TextGrid")]
        parts = base.split("_", 1)
        if len(parts) < 2:
            continue
        rest = parts[1]
        # rest is either "line-N" or "line-N_t12345"
        rest_line_id = rest.split("_t")[0]
        if rest_line_id == line_id:
            return os.path.join(textgrid_dir, fname)
    return None


def extract_seg_start_from_filename(filename):
    """
    If the filename contains an embedded timestamp (_t{ms}), return
    the start time in seconds. Otherwise return None.
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    m = re.search(r'_t(\d+)$', base)
    if m:
        return int(m.group(1)) / 1000.0
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Convert SOFA TextGrid output to annotation JSON")
    parser.add_argument("--annotation", "-a", required=True,
                        help="Path to original JSON annotation file")
    parser.add_argument("--textgrid-dir", "-t", required=True,
                        help="Directory containing TextGrid files")
    parser.add_argument("--output", "-o", required=True,
                        help="Output JSON file path")
    parser.add_argument("--padding", "-p", type=float, default=0.0,
                        help="Padding (seconds) that was used during segmentation. "
                             "Needed to correct TextGrid times back to absolute. "
                             "Default 0.0 (no padding).")
    args = parser.parse_args()

    with open(args.annotation, "r", encoding="utf-8") as f:
        data = json.load(f)

    lines = data["project"]["subtitleLines"]
    print(f"Annotation: {args.annotation}")
    print(f"TextGrid dir: {args.textgrid_dir}")
    print(f"Subtitle lines: {len(lines)}")
    print(f"Padding correction: {args.padding}s")
    print()

    # Pre-index TextGrid files
    textgrid_files = {}
    for line in lines:
        tg_path = find_textgrid_for_line(args.textgrid_dir, line["id"])
        if tg_path:
            textgrid_files[line["id"]] = tg_path

    print(f"TextGrid files found: {len(textgrid_files)} / {len(lines)}")
    print()

    character_annotations = []
    all_warnings = []
    processed = 0
    skipped = 0

    for i, line in enumerate(lines):
        line_id = line["id"]
        text = line["text"]
        line_start = line["startTime"]

        tg_path = textgrid_files.get(line_id)
        if not tg_path:
            print(f"  [{i+1:3d}/{len(lines)}] {line_id}: no TextGrid found, skipping")
            skipped += 1
            continue

        # Determine the absolute time of TextGrid t=0
        embedded_start = extract_seg_start_from_filename(tg_path)
        if embedded_start is not None:
            seg_start = embedded_start
        else:
            seg_start = line_start - args.padding

        tiers = parse_textgrid(tg_path)
        pinyin_intervals = extract_pinyin_intervals(tiers)
        pairs, warnings = match_chars_to_pinyin(text, pinyin_intervals, line_id)
        all_warnings.extend(warnings)

        for char, pinyin, xmin, xmax in pairs:
            character_annotations.append({
                "id": f"char-{uuid.uuid4()}",
                "lineId": line_id,
                "char": char,
                "pinyin": pinyin,
                "startTime": seg_start + xmin,
                "endTime": seg_start + xmax,
            })

        n_chars = len([ch for ch in text if is_chinese_char(ch)])
        print(f"  [{i+1:3d}/{len(lines)}] {line_id}: \"{text}\" -> "
              f"{len(pairs)}/{n_chars} characters matched")
        processed += 1

    # Build output JSON preserving the original structure
    output = {
        "version": data.get("version", 2),
        "project": {
            "video": data["project"].get("video", {}),
            "subtitleLines": lines,
            "characterAnnotations": character_annotations,
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nProcessed: {processed} lines")
    print(f"Skipped (no TextGrid): {skipped} lines")
    print(f"Character annotations: {len(character_annotations)}")
    print(f"Output: {args.output}")

    if all_warnings:
        print(f"\nWarnings ({len(all_warnings)}):")
        for w in all_warnings:
            print(f"  {w}")


if __name__ == "__main__":
    main()
