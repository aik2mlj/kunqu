"""
Two-pass split for subtitle lines longer than a threshold (default 55s).

Direct SOFA inference is accurate up to ~55s but drifts badly in the interior of
longer lines (out-of-distribution: the training data is all <=45s). This tool
splits only the long lines and re-aligns each piece in-distribution.

It is driven entirely by SOFA's OWN first-pass output -- never by the manual
characterAnnotations test oracle. The pipeline stays phrase-level.

Flow (wired into scripts/align_pipeline.sh):
  1. prepare_segments.py            -> per-line .wav + .lab           (all lines)
  2. infer.py -f <segments>         -> <segments>/TextGrid/*.TextGrid (pass 1)
  3. two_pass_split.py split        -> <split>/ pieces + manifest.json
  4. infer.py -f <split>            -> <split>/TextGrid/*.TextGrid    (pass 2)
  5. two_pass_split.py stitch       -> merged TextGrids overwrite the long lines'
                                       pass-1 TextGrids in <segments>/TextGrid/
  6. textgrid_to_annotation.py      -> character-level annotation JSON (unchanged)

Subcommands:
  split : for each per-line .wav longer than --threshold, read its pass-1
          TextGrid, pick cut points at the largest inter-syllable gaps
          (recursively bisecting until every piece <= --threshold), cut the
          .wav and split the pinyin .lab to match, and emit the pieces plus a
          manifest. Char->piece assignment follows the cut indices, so each
          piece's audio carries exactly its own syllables.
  stitch: after pass-2 inference on the pieces, offset each piece's word-tier
          intervals back to line-relative time and concatenate them into one
          TextGrid per line, overwriting that line's pass-1 TextGrid.
"""

import argparse
import glob
import json
import os

import soundfile as sf

# Run as `python data/two_pass_split.py ...`, so the data/ dir is on sys.path[0]
# and these sibling modules import directly.
from prepare_segments import extract_wav_segment
from textgrid_to_annotation import parse_textgrid


def choose_cuts(intervals, seg_dur, threshold, alpha=0.1):
    """
    Pick cut indices so every resulting piece spans <= threshold seconds.

    intervals: list of (xmin, xmax, pinyin) for the line's syllables, in order,
               relative to the line start (the pass-1 TextGrid word tier with
               AP/SP removed).
    Returns a sorted list of cut indices. A cut index k means "cut after
    syllable k": the left piece keeps syllables [..k], the right keeps [k+1..].
    The cut time is the midpoint of the gap between syllables k and k+1, so the
    audio is sliced inside a silence the model itself found.

    Cuts are chosen by recursive bisection: at each step pick the gap that
    maximises  gap_size - alpha*|cut_time - piece_midpoint|  (largest gap,
    tie-broken toward the middle), then recurse into any half still too long.
    """
    cuts = []

    def rec(lo, hi, t0, t1):
        # intervals[lo..hi] inclusive span the piece covering [t0, t1).
        if (t1 - t0) <= threshold:
            return
        if hi - lo < 1:
            return  # a single syllable longer than threshold: cannot split it
        mid = 0.5 * (t0 + t1)
        best_k = None
        best_score = None
        best_t = None
        for k in range(lo, hi):
            gap = intervals[k + 1][0] - intervals[k][1]
            tcut = 0.5 * (intervals[k][1] + intervals[k + 1][0])
            score = gap - alpha * abs(tcut - mid)
            if best_score is None or score > best_score:
                best_score, best_k, best_t = score, k, tcut
        cuts.append(best_k)
        rec(lo, best_k, t0, best_t)
        rec(best_k + 1, hi, best_t, t1)

    rec(0, len(intervals) - 1, 0.0, seg_dur)
    return sorted(cuts)


def _pinyin_intervals(tg_path):
    """Word-tier intervals that are real syllables (AP/SP removed), in order."""
    tiers = parse_textgrid(tg_path)
    words = tiers.get("words", [])
    return [(xmin, xmax, text) for xmin, xmax, text in words
            if text not in ("AP", "SP")]


def cmd_split(args):
    seg_dir = args.segments
    tg_dir = os.path.join(seg_dir, "TextGrid")
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    manifest = {"pieces": {}, "lines": {}}
    n_split = 0
    n_pieces = 0

    for wav_path in sorted(glob.glob(os.path.join(seg_dir, "*.wav"))):
        base = os.path.splitext(os.path.basename(wav_path))[0]
        info = sf.info(wav_path)
        seg_dur = info.frames / info.samplerate
        if seg_dur <= args.threshold:
            continue

        tg_path = os.path.join(tg_dir, base + ".TextGrid")
        lab_path = os.path.join(seg_dir, base + ".lab")
        if not (os.path.exists(tg_path) and os.path.exists(lab_path)):
            print(f"  {base}: missing pass-1 TextGrid or .lab, leaving on direct")
            continue

        intervals = _pinyin_intervals(tg_path)
        with open(lab_path, "r", encoding="utf-8") as f:
            syllables = f.read().split()

        if len(syllables) != len(intervals):
            print(f"  {base}: lab syllables ({len(syllables)}) != TextGrid "
                  f"syllables ({len(intervals)}), leaving on direct")
            continue

        cuts = choose_cuts(intervals, seg_dur, args.threshold)
        if not cuts:
            print(f"  {base}: no valid cut found, leaving on direct")
            continue

        # Cut times (line-relative) -> piece boundaries [0, t1, t2, ..., seg_dur].
        cut_times = [0.5 * (intervals[k][1] + intervals[k + 1][0]) for k in cuts]
        bounds = [0.0] + cut_times + [seg_dur]
        # Syllable index boundaries aligned with the cuts.
        idx_bounds = [0] + [k + 1 for k in cuts] + [len(syllables)]

        manifest["lines"][base] = {"seg_dur": seg_dur, "n_pieces": len(bounds) - 1}
        for p in range(len(bounds) - 1):
            t0, t1 = bounds[p], bounds[p + 1]
            syl_slice = syllables[idx_bounds[p]:idx_bounds[p + 1]]
            piece_base = f"{base}__p{p}"
            piece_wav = os.path.join(out_dir, piece_base + ".wav")
            piece_lab = os.path.join(out_dir, piece_base + ".lab")
            extract_wav_segment(wav_path, piece_wav, t0, t1)
            with open(piece_lab, "w", encoding="utf-8") as f:
                f.write(" ".join(syl_slice))
            manifest["pieces"][piece_base] = {"line": base, "order": p, "offset": t0}
            n_pieces += 1

        print(f"  {base}: {seg_dur:.1f}s -> {len(bounds) - 1} pieces "
              f"at cut times {[round(t, 2) for t in cut_times]}")
        n_split += 1

    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\nSplit {n_split} long lines into {n_pieces} pieces -> {out_dir}/")
    print(f"Manifest: {os.path.join(out_dir, 'manifest.json')}")
    if n_pieces == 0:
        print("No lines exceeded the threshold; nothing to do in pass 2.")


def _write_textgrid(path, intervals, xmax):
    """Write a minimal Praat long-format TextGrid with one 'words' tier."""
    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        "xmin = 0",
        f"xmax = {xmax}",
        "tiers? <exists>",
        "size = 1",
        "item []:",
        "    item [1]:",
        '        class = "IntervalTier"',
        '        name = "words"',
        "        xmin = 0",
        f"        xmax = {xmax}",
        f"        intervals: size = {len(intervals)}",
    ]
    for i, (xmin, xmax_i, text) in enumerate(intervals, start=1):
        text = text.replace('"', "")
        lines += [
            f"        intervals [{i}]:",
            f"            xmin = {xmin}",
            f"            xmax = {xmax_i}",
            f'            text = "{text}"',
        ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def cmd_stitch(args):
    split_dir = args.split_dir
    tg_in_dir = os.path.join(split_dir, "TextGrid")
    tg_out_dir = args.textgrid_out
    os.makedirs(tg_out_dir, exist_ok=True)

    with open(os.path.join(split_dir, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    by_line = {}
    for piece_base, meta in manifest["pieces"].items():
        by_line.setdefault(meta["line"], []).append((meta["order"], piece_base, meta["offset"]))

    stitched = 0
    for line_base, pieces in by_line.items():
        pieces.sort(key=lambda x: x[0])
        merged = []
        for _, piece_base, offset in pieces:
            tg_path = os.path.join(tg_in_dir, piece_base + ".TextGrid")
            if not os.path.exists(tg_path):
                print(f"  {line_base}: missing piece TextGrid {piece_base}, skipping line")
                merged = None
                break
            tiers = parse_textgrid(tg_path)
            for xmin, xmax, text in tiers.get("words", []):
                merged.append((xmin + offset, xmax + offset, text))
        if not merged:
            continue
        seg_dur = manifest["lines"][line_base]["seg_dur"]
        out_path = os.path.join(tg_out_dir, line_base + ".TextGrid")
        _write_textgrid(out_path, merged, seg_dur)
        print(f"  {line_base}: stitched {len(pieces)} pieces -> {out_path}")
        stitched += 1

    print(f"\nStitched {stitched} long lines back into {tg_out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Two-pass split for long lines")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("split", help="cut long lines into pieces using pass-1 TextGrids")
    sp.add_argument("--segments", "-s", required=True, help="per-line segments dir (with TextGrid/)")
    sp.add_argument("--out", "-o", required=True, help="output dir for pieces + manifest")
    sp.add_argument("--threshold", "-t", type=float, default=55.0, help="max piece length (s)")
    sp.set_defaults(func=cmd_split)

    st = sub.add_parser("stitch", help="merge pass-2 piece TextGrids back per line")
    st.add_argument("--split-dir", "-s", required=True, help="dir with pieces + manifest + TextGrid/")
    st.add_argument("--textgrid-out", "-o", required=True, help="TextGrid dir to overwrite (pass-1 dir)")
    st.set_defaults(func=cmd_stitch)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
