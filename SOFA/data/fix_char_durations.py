"""
Fix melisma duration-collapse in a character-level annotation, cosmetically.

SOFA collapses passing/melismatic syllables to ~0ms and dumps the held time into
a neighbor, so those chars stack at one instant and render overlapping in the
labeling tool. This post-process redistributes time WITHIN each contiguous run of
characters so every char gets at least --min-dur seconds, taking the deficit from
the abnormally long neighbor(s) in the same run, proportional to their surplus.

Guarantees:
  * every char ends up with duration >= min_dur,
  * each contiguous run's total span is preserved exactly (start of the first
    char and end of the last char in a run are unchanged),
  * boundaries stay monotonic and contiguous within a run,
  * silences (gaps > --gap between chars) are never touched — runs are split at
    them and handled independently.

This does NOT improve alignment accuracy; it only removes the zero-width stacking.
It's the shipping fix after finetuning was rejected (it worsened collapse).

Usage (cwd = kunqu/SOFA):
    python data/fix_char_durations.py \
        data/shihuajiaohua/shihuajiaohua_roformer_annotation.json \
        --min-dur 0.05 --out data/shihuajiaohua/shihuajiaohua_roformer_fixed_annotation.json
"""

import argparse
import json
from collections import defaultdict


def redistribute(durs, m):
    """
    Given a contiguous run's per-char durations, return new durations where every
    entry >= m, taking the deficit proportionally from entries with surplus (> m).
    Total is preserved when feasible; if the run is too short (sum < n*m) every
    char gets an equal share (best effort). Returns (new_durs, feasible).
    """
    S = sum(durs)
    n = len(durs)
    if n == 0:
        return durs, True
    if S < n * m:                      # physically can't give everyone m
        return [S / n] * n, False
    need = sum(m - d for d in durs if d < m)
    if need <= 0:
        return durs, True
    surplus = sum(d - m for d in durs if d > m)
    scale = need / surplus             # <= 1 since need <= surplus here
    return [m if d < m else d - (d - m) * scale for d in durs], True


def fix_line(chars, m, gap, line_start, line_end):
    """
    Two passes over a line's chars:
      1. within each contiguous run, redistribute so every char >= m (steal from
         the long neighbor; run span preserved) — fixes the stacked clusters;
      2. any char still < m (isolated / short run with no long neighbor) expands
         into the adjacent silence, bounded by its neighbors and the line edges.
    """
    fixed = 0
    n = len(chars)

    # pass 1: per-run redistribution
    i = 0
    while i < n:
        j = i
        while j + 1 < n and (chars[j + 1]["startTime"] - chars[j]["endTime"]) <= gap:
            j += 1
        run = chars[i:j + 1]
        durs = [c["endTime"] - c["startTime"] for c in run]
        if any(d < m for d in durs):
            new_durs, _ = redistribute(durs, m)
            t = run[0]["startTime"]            # anchor: run start unchanged
            for c, d in zip(run, new_durs):
                c["startTime"] = t
                t += d
                c["endTime"] = t
            fixed += sum(1 for d in durs if d < m)
        i = j + 1

    # pass 2: fill remaining short chars from surrounding silence
    for k, c in enumerate(chars):
        deficit = m - (c["endTime"] - c["startTime"])
        if deficit <= 1e-9:
            continue
        right_bound = chars[k + 1]["startTime"] if k + 1 < n else line_end
        left_bound = chars[k - 1]["endTime"] if k > 0 else line_start
        take_r = min(deficit, max(0.0, right_bound - c["endTime"]))
        c["endTime"] += take_r
        deficit -= take_r
        take_l = min(deficit, max(0.0, c["startTime"] - left_bound))
        c["startTime"] -= take_l

    return fixed


def main():
    ap = argparse.ArgumentParser(description="Redistribute near-zero char durations (fix melisma collapse)")
    ap.add_argument("annotation", help="input annotation JSON")
    ap.add_argument("--min-dur", type=float, default=0.05, help="min char duration in seconds (default 0.05)")
    ap.add_argument("--gap", type=float, default=0.005, help="max inter-char gap treated as contiguous (s)")
    ap.add_argument("--out", default=None, help="output JSON (default: <input>_fixed.json)")
    args = ap.parse_args()

    with open(args.annotation, encoding="utf-8") as f:
        data = json.load(f)
    ca = data["project"]["characterAnnotations"]

    before = sum(1 for c in ca if c["endTime"] - c["startTime"] < args.min_dur)

    line_span = {l["id"]: (l["startTime"], l["endTime"])
                 for l in data["project"].get("subtitleLines", [])}
    by_line = defaultdict(list)
    for c in ca:
        by_line[c["lineId"]].append(c)
    total_fixed = 0
    for lid, chars in by_line.items():
        chars.sort(key=lambda x: x["startTime"])
        ls, le = line_span.get(lid, (chars[0]["startTime"], chars[-1]["endTime"]))
        total_fixed += fix_line(chars, args.min_dur, args.gap, ls, le)

    after = sum(1 for c in ca if c["endTime"] - c["startTime"] < args.min_dur - 1e-9)

    out = args.out or args.annotation.rsplit(".json", 1)[0] + "_fixed.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    n = len(ca)
    print(f"chars                : {n}")
    print(f"collapsed <{args.min_dur}s before: {before} ({100*before/n:.2f}%)")
    print(f"collapsed <{args.min_dur}s after : {after} ({100*after/n:.2f}%)")
    print(f"chars adjusted       : {total_fixed}")
    print(f"output               : {out}")


if __name__ == "__main__":
    main()
