#!/usr/bin/env python
"""
Convert an OCR .srt of burnt-in lyrics into the exact xunmeng annotation JSON,
auto-merging duplicate cue runs and writing a cleanup report.

OCR over-segments: a single displayed phrase flickers into many near-identical
cues (e.g. "话到其间..." -> 33 variants, "又素之平生半面" -> 3). This tool collapses
each run of consecutive near-duplicates into one phrase by per-position majority
vote, then emits annotation JSON whose structure is byte-compatible with the
labeling tool's format (cloned from a --template). Character-level annotations are
left empty; the SOFA aligner fills them in.

Non-lyric cues (credits / role names) are kept in the output but flagged in the
report for you to prune later. Less-obvious OCR errors (single wrong characters,
two genuinely different halves of a long phrase that did not merge) are also only
flagged, not corrected.

Usage (from kunqu/videocr/):
    python srt_to_annotation.py xunmeng.srt
    python srt_to_annotation.py xunmeng.srt --sim 0.6 --gap 2.0 \
        --template ../SOFA/data/xunmeng/xunmeng_annotation.json \
        --out ../SOFA/data/xunmeng/xunmeng_ocr_annotation.json
"""

import argparse
import difflib
import json
import os
import re
from collections import Counter


# ----------------------------------------------------------------------------
# Parsing + text helpers
# ----------------------------------------------------------------------------
def srt_ts_to_sec(ts):
    """'HH:MM:SS,mmm' -> float seconds."""
    hh, mm, rest = ts.split(":")
    ss, ms = rest.split(",")
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0


def normalize_text(s):
    """Strip all whitespace (incl. full-width 　) so OCR spacing noise doesn't matter."""
    return re.sub(r"\s+", "", s)


def is_cjk(ch):
    cp = ord(ch)
    return (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or
            0x20000 <= cp <= 0x2A6DF or 0xF900 <= cp <= 0xFAFF)


def parse_srt(path):
    """Return list of cues: {'start','end','texts'(list),'text'(normalized join)}."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    cues = []
    for block in re.split(r"\n\s*\n", content.strip()):
        lines = [ln for ln in block.splitlines() if ln.strip() != ""]
        if len(lines) < 2:
            continue
        m = re.search(r"(\d\d:\d\d:\d\d,\d+)\s*-->\s*(\d\d:\d\d:\d\d,\d+)", lines[1])
        if not m:
            continue
        texts = lines[2:]
        cues.append({
            "start": srt_ts_to_sec(m.group(1)),
            "end": srt_ts_to_sec(m.group(2)),
            "texts": texts,
            "text": normalize_text("".join(texts)),
            "multiline": len(texts) > 1,
        })
    return cues


# ----------------------------------------------------------------------------
# Duplicate merge (per-position majority vote)
# ----------------------------------------------------------------------------
def majority_merge(texts):
    """
    Merge OCR variants of the same phrase. Group variants by length, keep the
    modal length, and majority-vote each character position within that group
    (tiebreak: char from the longest variant). Cancels random OCR flips.
    """
    texts = [t for t in texts if t]
    if not texts:
        return ""
    if len(texts) == 1:
        return texts[0]

    modal_len = Counter(len(t) for t in texts).most_common(1)[0][0]
    pool = [t for t in texts if len(t) == modal_len] or texts
    longest = max(pool, key=len)
    length = max(len(t) for t in pool)

    out = []
    for i in range(length):
        col = [t[i] for t in pool if i < len(t)]
        if not col:
            continue
        counts = Counter(col).most_common()
        top = counts[0][1]
        cands = [c for c, n in counts if n == top]
        if len(cands) == 1:
            out.append(cands[0])
        elif i < len(longest) and longest[i] in cands:
            out.append(longest[i])
        else:
            out.append(cands[0])
    return "".join(out)


def merge_runs(cues, sim, gap):
    """
    Collapse consecutive near-duplicate cues. Two adjacent cues join a run when
    SequenceMatcher.ratio > sim AND the time gap between them < gap seconds.
    Returns (merged_cues, merge_records).
    """
    merged = []
    records = []
    i = 0
    while i < len(cues):
        j = i
        while j + 1 < len(cues):
            ratio = difflib.SequenceMatcher(None, cues[j]["text"], cues[j + 1]["text"]).ratio()
            g = cues[j + 1]["start"] - cues[j]["end"]
            if ratio > sim and g < gap:
                j += 1
            else:
                break
        run = cues[i:j + 1]
        text = majority_merge([c["text"] for c in run]) if len(run) > 1 else run[0]["text"]
        merged.append({
            "start": run[0]["start"],
            "end": run[-1]["end"],
            "text": text,
            "multiline": any(c["multiline"] for c in run),
        })
        if len(run) > 1:
            records.append({
                "n": len(run),
                "start": run[0]["start"],
                "end": run[-1]["end"],
                "variants": [c["text"] for c in run],
                "chosen": text,
            })
        i = j + 1
    return merged, records


def _containment(a, b, sub_dur):
    """
    If one of two adjacent cues is a short substring artifact of the other, return
    (keep, drop) where `keep` is the full phrase and `drop` is the artifact; else
    None. The artifact must be a strict substring (shorter text) AND shorter than
    `sub_dur` seconds — that duration floor is what separates a mid-render flicker
    from a genuine short line that merely shares characters.
    """
    ta, tb = a["text"], b["text"]
    if ta and ta != tb and ta in tb and (a["end"] - a["start"]) < sub_dur:
        return b, a
    if tb and ta != tb and tb in ta and (b["end"] - b["start"]) < sub_dur:
        return a, b
    return None


def _fuse(keep, drop, records):
    """Merge `drop` into `keep`: keep the full phrase's text, span both times."""
    fused = {
        "start": min(keep["start"], drop["start"]),
        "end": max(keep["end"], drop["end"]),
        "text": keep["text"],
        "multiline": keep.get("multiline", False) or drop.get("multiline", False),
    }
    records.append({
        "kept": keep["text"], "dropped": drop["text"],
        "start": fused["start"], "end": fused["end"],
    })
    return fused


def merge_contained(cues, sub_dur, gap):
    """
    Absorb a short cue whose text is a substring of an adjacent cue into that
    neighbor. Catches subtitle mid-render artifacts the ratio-based dup merge
    misses: a growing prefix like '凭栏仍' (0.2s) right before '凭栏仍是玉栏杆'
    scores only ~0.6 symmetric similarity (the long phrase inflates the
    denominator), so it never joins the run — but it is clearly the same line
    caught mid-render. Requires the artifact be < `sub_dur` seconds, a strict
    substring, and within `gap` seconds of the neighbor. Growing-prefix chains
    fold into the final full phrase. Returns (merged_cues, records).
    """
    cues = [dict(c) for c in cues]
    out = []
    i = 0
    records = []
    while i < len(cues):
        cur = cues[i]
        # absorb into the next cue (leading prefix render); replace next so chains fold
        if i + 1 < len(cues) and cues[i + 1]["start"] - cur["end"] < gap:
            cont = _containment(cur, cues[i + 1], sub_dur)
            if cont:
                cues[i + 1] = _fuse(cont[0], cont[1], records)
                i += 1
                continue
        # absorb into the previous kept cue (trailing flicker)
        if out and cur["start"] - out[-1]["end"] < gap:
            cont = _containment(out[-1], cur, sub_dur)
            if cont:
                out[-1] = _fuse(cont[0], cont[1], records)
                i += 1
                continue
        out.append(cur)
        i += 1
    return out, records


def flags_for(cue):
    f = []
    if cue["multiline"]:
        f.append("multiline/credit?")
    if any(not is_cjk(ch) for ch in cue["text"]):
        f.append("non-CJK chars")
    if (cue["end"] - cue["start"]) < 1.0:
        f.append("<1s (possible split/dup)")
    return f


# ----------------------------------------------------------------------------
# JSON assembly (clone template structure for byte-compatible format)
# ----------------------------------------------------------------------------
def build_annotation(template_path, merged, video_name=None):
    with open(template_path, "r", encoding="utf-8") as f:
        data = json.loads(f.read())  # fresh deep copy

    subtitle_lines = [{
        "id": f"line-{idx + 1}",
        "text": c["text"],
        "startTime": round(c["start"], 3),
        "endTime": round(c["end"], 3),
    } for idx, c in enumerate(merged)]

    data["project"]["subtitleLines"] = subtitle_lines
    data["project"]["characterAnnotations"] = []
    data["project"]["actionAnnotations"] = []
    # The stale --template predates the labeling tool's gongche feature; without
    # this key the tool errors while a newer export (which has it) loads. Emit an
    # empty array so the output matches the current schema.
    data["project"]["gongcheAnnotations"] = []
    if video_name is not None:
        data["project"].setdefault("video", {})["name"] = video_name
    return data


def write_report(path, srt_path, template_path, sim, gap, sub_dur, n_cues, merged,
                 records, crecords):
    lines = []
    lines.append("SRT -> annotation cleanup report")
    lines.append(f"input    : {srt_path}")
    lines.append(f"template : {template_path}")
    lines.append(f"params   : sim>{sim}  gap<{gap}s  sub_dur<{sub_dur}s")
    lines.append(f"cues parsed     : {n_cues}")
    lines.append(f"subtitleLines   : {len(merged)}  (collapsed {n_cues - len(merged)})")
    lines.append("")
    lines.append(f"=== merges ({len(records)}) ===")
    for r in records:
        variants = " | ".join(r["variants"])
        lines.append(f"[{r['start']:.1f}-{r['end']:.1f}s] n={r['n']}  {variants}")
        lines.append(f"    -> {r['chosen']}")
    lines.append("")
    lines.append(f"=== prefix/containment merges ({len(crecords)}) ===")
    for r in crecords:
        lines.append(f"[{r['start']:.1f}-{r['end']:.1f}s] dropped \"{r['dropped']}\"  ->  \"{r['kept']}\"")
    lines.append("")
    flagged = [(idx + 1, c, flags_for(c)) for idx, c in enumerate(merged) if flags_for(c)]
    lines.append(f"=== flagged (kept in output, review later) ({len(flagged)}) ===")
    for line_no, c, fl in flagged:
        lines.append(f"line-{line_no} [{c['start']:.1f}-{c['end']:.1f}s] \"{c['text']}\"  ->  {', '.join(fl)}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return flagged


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="OCR SRT -> xunmeng annotation JSON + cleanup")
    ap.add_argument("srt", help="input .srt")
    ap.add_argument("--template", default="../SOFA/data/xunmeng/xunmeng_annotation.json",
                    help="annotation JSON whose structure is cloned (default: xunmeng GT)")
    ap.add_argument("--out", default=None,
                    help="output JSON (default: <template_dir>/<srt_stem>_ocr_annotation.json)")
    ap.add_argument("--report", default=None, help="report path (default: <out>.report.txt)")
    ap.add_argument("--sim", type=float, default=0.6, help="dup similarity threshold (0-1)")
    ap.add_argument("--gap", type=float, default=2.0, help="max inter-cue gap to merge (s)")
    ap.add_argument("--sub-dur", type=float, default=1.0,
                    help="max duration (s) of a short substring cue to absorb into its neighbor")
    ap.add_argument("--video-name", default=None, help="override project.video.name")
    args = ap.parse_args()

    cues = parse_srt(args.srt)
    if not cues:
        raise SystemExit(f"ERROR: no cues parsed from {args.srt}")

    merged, records = merge_runs(cues, args.sim, args.gap)
    merged, crecords = merge_contained(merged, args.sub_dur, args.gap)
    data = build_annotation(args.template, merged, args.video_name)

    if args.out is None:
        stem = os.path.splitext(os.path.basename(args.srt))[0]
        args.out = os.path.join(os.path.dirname(args.template), f"{stem}_ocr_annotation.json")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    report_path = args.report or (args.out + ".report.txt")
    flagged = write_report(report_path, args.srt, args.template, args.sim, args.gap,
                           args.sub_dur, len(cues), merged, records, crecords)

    print(f"cues parsed     : {len(cues)}")
    print(f"subtitleLines   : {len(merged)} (merged {len(records)} runs, "
          f"{len(crecords)} prefix, collapsed {len(cues) - len(merged)} cues)")
    print(f"flagged later   : {len(flagged)}")
    print(f"output JSON     : {args.out}")
    print(f"cleanup report  : {report_path}")


if __name__ == "__main__":
    main()
