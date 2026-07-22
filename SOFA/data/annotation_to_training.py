"""
Build SOFA full_label training data from a manually-corrected char-level
annotation JSON (the "human GT").

SOFA training wants, per clip, a wav (<=45s) plus a row in transcriptions.csv:
    name, ph_seq, ph_dur
where ph_seq is space-separated phonemes (each syllable split into initial+final
per the dictionary, plus SP for silence) and ph_dur is one duration (seconds) per
phoneme (see https://github.com/qiuqiao/SOFA/discussions/5). full_label requires
len(ph_dur) == len(ph_seq), else the binarizer downgrades the item to weak_label.

Our GT gives per-*syllable* (character) boundaries; SOFA wants per-*phoneme*
durations. Each syllable is split into its phones two ways:

  --align-ckpt <ckpt>  (constrained pretrained align, recommended, needs GPU):
      slice each character's audio to its own tiny clip, run `infer.py
      --out_formats transcriptions` on those clips, and read back the phone
      durations. A single-syllable clip cannot collapse across syllables, so the
      pretrained model yields a realistic short-initial / long-final split that is
      anchored to the human character boundaries.

  heuristic (default, no model needed):
      initial consonant = min(80ms, 40% of the syllable), remainder -> final;
      single-phone syllable -> full span.

Either way, the phones of a syllable always sum to the human character duration
(the melisma signal we are trying to teach), and SP fills the gaps.

Output layout:
    <out_root>/full_label/<play>/wavs/<name>.wav
    <out_root>/full_label/<play>/transcriptions.csv

Usage (from kunqu/SOFA/):
    # local logic check, no audio, no model:
    python data/annotation_to_training.py \
        --annotation data/shihuajiaohua/shihuajiaohua_roformer_annotation_human_gt.json \
        --play shihuajiaohua --out-root data/finetune_B --dry-run

    # server, full build with constrained align:
    python data/annotation_to_training.py \
        --annotation data/shihuajiaohua/shihuajiaohua_roformer_annotation_human_gt.json \
        --wav data/shihuajiaohua/shihuajiaohua_vocals_mel-band-roformer.wav \
        --play shihuajiaohua --out-root data/finetune_B \
        --align-ckpt ckpt/pretrained_mandarin_singing/v1.0.0_mandarin_singing.ckpt
"""

import argparse
import csv
import json
import os
import pathlib
import subprocess
import sys
import tempfile
from collections import defaultdict

# reuse the audio slicer from the inference-prep script
from prepare_segments import extract_wav_segment  # noqa: E402

from pypinyin import lazy_pinyin, Style

SP = "SP"                 # silence token
SP_MIN = 0.03             # gaps shorter than this are absorbed, not emitted as SP
INIT_MAX = 0.08           # heuristic: max initial-consonant duration (s)
INIT_FRAC = 0.40          # heuristic: initial consonant <= this fraction of syllable
MAX_CLIP = 40.0           # sub-split lines longer than this (margin under SOFA's 45s)
SILENCE_SPLIT = 2.0       # always cut a clip at any inter-char silence >= this (s)


# ---------------------------------------------------------------------------
# dictionary + g2p
# ---------------------------------------------------------------------------
def load_dict_mapping(dict_path):
    """opencpop-extension.txt -> {pinyin: [phone, ...]}."""
    mapping = {}
    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            key, _, phones = line.partition("\t")
            key = key.strip()
            phones = phones.split()
            if key and phones:
                mapping[key] = phones
    return mapping


def chars_to_pinyin(chars):
    """
    Context-aware pinyin for a run of characters. Join first so pypinyin sees the
    word context (better for polyphones), returning one pinyin per input char.
    Returns list aligned 1:1 with `chars` (empty string where not convertible).
    """
    joined = "".join(chars)
    py = lazy_pinyin(joined, style=Style.NORMAL)
    if len(py) != len(chars):        # punctuation / surrogate mismatch: fall back per-char
        py = [ (lazy_pinyin(c, style=Style.NORMAL) or [""])[0] for c in chars ]
    return [p.strip().lower() for p in py]


# ---------------------------------------------------------------------------
# per-syllable phone-duration split
# ---------------------------------------------------------------------------
def heuristic_split(phones, dur):
    """Short fixed initial, remainder -> final(s). Sums to `dur`."""
    n = len(phones)
    if n == 0:
        return []
    if n == 1:
        return [dur]
    init = min(INIT_MAX, INIT_FRAC * dur)
    rest = max(dur - init, 0.0)
    tail = [rest / (n - 1)] * (n - 1)
    return [init, *tail]


def normalize_to(durs, target):
    """Scale a positive duration vector to sum exactly to `target`."""
    s = sum(durs)
    if s <= 0:
        n = len(durs)
        return [target / n] * n if n else []
    return [d * target / s for d in durs]


# ---------------------------------------------------------------------------
# constrained pretrained align: per-char clips -> infer transcriptions -> durs
# ---------------------------------------------------------------------------
def build_align_table(chars, pinyins, mapping, wav_path, ckpt, dict_path):
    """
    For every character, align its own audio slice to its phones with the
    pretrained model and return {char_index: [phone_durs]} (already the intra-
    syllable split). Characters that fail to align are omitted (caller falls back
    to the heuristic). Requires a real wav + ckpt.
    """
    tmp = tempfile.mkdtemp(prefix="sofa_syll_")
    index_of = {}
    for i, (c, py) in enumerate(zip(chars, pinyins)):
        phones = mapping.get(py)
        if not phones or len(phones) < 2:
            continue                       # nothing to split (1 phone) -> heuristic
        dur = c["endTime"] - c["startTime"]
        if dur <= 0:
            continue
        name = f"c{i:05d}"
        extract_wav_segment(wav_path, os.path.join(tmp, name + ".wav"),
                            c["startTime"], c["endTime"])
        with open(os.path.join(tmp, name + ".lab"), "w", encoding="utf-8") as f:
            f.write(py)
        index_of[name] = i

    if not index_of:
        return {}

    subprocess.run(
        [sys.executable, "infer.py", "--ckpt", ckpt, "--folder", tmp,
         "--out_formats", "transcriptions", "--dictionary", dict_path],
        check=True,
    )

    table = {}
    trans = pathlib.Path(tmp) / "transcriptions.csv"
    if not trans.exists():                 # some SOFA versions write per-subfolder
        hits = list(pathlib.Path(tmp).rglob("transcriptions.csv"))
        trans = hits[0] if hits else None
    if trans and trans.exists():
        with open(trans, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                i = index_of.get(str(row["name"]))
                if i is None:
                    continue
                seq = row["ph_seq"].split()
                dur = [float(x) for x in row["ph_dur"].split()]
                # drop SP/AP infer pads at the edges; keep the real phones
                keep = [d for p, d in zip(seq, dur) if p not in ("SP", "AP", "")]
                if keep:
                    table[i] = keep
    return table


# ---------------------------------------------------------------------------
# assemble one clip (a run of chars) into ph_seq / ph_dur with SP gaps
# ---------------------------------------------------------------------------
def assemble_clip(run, pinyins, mapping, seg_start, seg_end, align_table, warns):
    """
    run: list of (char_index, char_obj). Returns (ph_seq, ph_dur) covering
    [seg_start, seg_end]; SP fills leading/inter/trailing silence.
    """
    ph_seq, ph_dur = [], []

    def add_sp(gap):
        if gap >= SP_MIN:
            ph_seq.append(SP)
            ph_dur.append(gap)

    prev_end = seg_start
    for idx, c in run:
        add_sp(c["startTime"] - prev_end)               # silence before this char
        py = pinyins[idx]
        phones = mapping.get(py)
        dur = c["endTime"] - c["startTime"]
        if not phones:
            warns.append(f"no phones for '{c.get('char')}' (pinyin '{py}') -> treated as SP")
            add_sp(dur)
            prev_end = c["endTime"]
            continue
        durs = align_table.get(idx)
        if durs and len(durs) == len(phones):
            durs = normalize_to(durs, dur)              # anchor to human syllable span
        else:
            durs = heuristic_split(phones, dur)
        ph_seq.extend(phones)
        ph_dur.extend(durs)
        prev_end = c["endTime"]
    add_sp(seg_end - prev_end)                          # trailing silence
    return ph_seq, ph_dur


# ---------------------------------------------------------------------------
# split a line's chars into <=MAX_CLIP runs at the widest inter-char gap
# ---------------------------------------------------------------------------
def split_runs(chars, max_clip):
    """
    Yield (seg_start, seg_end, [(idx,char)...]) runs with TIGHT boundaries
    ([first.start, last.end]). Cuts (1) at every inter-char silence >=
    SILENCE_SPLIT (instrumental interludes, dropped from the clips) and then
    (2) recursively at the largest remaining internal gap until every run's span
    is <= max_clip. Tight boundaries mean no clip ever straddles a big silence.
    """
    # 1. mandatory cuts at large silences
    groups, cur = [], [chars[0]]
    for prev, c in zip(chars, chars[1:]):
        gap = c[1]["startTime"] - prev[1]["endTime"]
        if gap >= SILENCE_SPLIT:
            groups.append(cur)
            cur = [c]
        else:
            cur.append(c)
    groups.append(cur)

    # 2. recursively halve any group still longer than max_clip at its widest gap
    out = []

    def rec(items):
        span = items[-1][1]["endTime"] - items[0][1]["startTime"]
        if span <= max_clip or len(items) < 2:
            out.append(items)
            return
        best_k, best_gap = 1, None
        for k in range(1, len(items)):
            gap = items[k][1]["startTime"] - items[k - 1][1]["endTime"]
            if best_gap is None or gap > best_gap:
                best_gap, best_k = gap, k
        rec(items[:best_k])
        rec(items[best_k:])

    for g in groups:
        rec(g)

    # 3. tight clip boundaries
    return [(g[0][1]["startTime"], g[-1][1]["endTime"], g) for g in out]


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="GT annotation JSON -> SOFA full_label data")
    ap.add_argument("--annotation", "-a", required=True)
    ap.add_argument("--play", "-p", required=True, help="singer/play folder name")
    ap.add_argument("--out-root", "-o", required=True, help="e.g. data/finetune_B")
    ap.add_argument("--wav", "-w", default=None, help="source vocals wav (omit with --dry-run)")
    ap.add_argument("--dictionary", "-d", default="dictionary/opencpop-extension.txt")
    ap.add_argument("--align-ckpt", default=None,
                    help="pretrained ckpt -> constrained per-syllable align (needs --wav + GPU)")
    ap.add_argument("--max-clip", type=float, default=MAX_CLIP)
    ap.add_argument("--dry-run", action="store_true", help="write CSV only, skip wav slicing")
    args = ap.parse_args()

    with open(args.annotation, encoding="utf-8") as f:
        proj = json.load(f)["project"]
    mapping = load_dict_mapping(args.dictionary)

    # group characters by line, in time order
    by_line = defaultdict(list)
    for c in proj["characterAnnotations"]:
        by_line[c["lineId"]].append(c)
    line_order = [s["id"] for s in proj["subtitleLines"]]

    out_dir = pathlib.Path(args.out_root) / "full_label" / args.play
    wav_dir = out_dir / "wavs"
    (wav_dir if not args.dry_run else out_dir).mkdir(parents=True, exist_ok=True)

    # optional constrained-align table over ALL chars (one infer pass)
    align_table = {}
    all_chars, all_py = [], []
    if args.align_ckpt:
        if not args.wav:
            ap.error("--align-ckpt requires --wav")
        # build a global index so align rows map back to (line, char)
        flat = []
        for lid in line_order:
            for c in sorted(by_line.get(lid, []), key=lambda x: x["startTime"]):
                flat.append(c)
        all_py = chars_to_pinyin([c["char"] for c in flat])
        gtab = build_align_table(flat, all_py, mapping, args.wav,
                                 args.align_ckpt, args.dictionary)
        # remap: build_align_table keyed by position in `flat`
        align_table = {id(flat[i]): d for i, d in gtab.items()}

    rows = []
    warns = []
    n_clips = 0
    for lid in line_order:
        chars = sorted(by_line.get(lid, []), key=lambda x: x["startTime"])
        if not chars:
            continue
        pys = chars_to_pinyin([c["char"] for c in chars])
        # local pinyin index for this line
        pin = {i: pys[i] for i in range(len(chars))}
        items = list(enumerate(chars))
        for j, (seg_start, seg_end, run) in enumerate(split_runs(items, args.max_clip)):
            # per-run align lookup keyed by object identity (align_table above)
            local_align = {}
            for idx, c in run:
                d = align_table.get(id(c))
                if d:
                    local_align[idx] = d
            ph_seq, ph_dur = assemble_clip(run, pin, mapping, seg_start, seg_end,
                                           local_align, warns)
            if not ph_seq:
                continue
            name = f"{args.play}_{lid}_{j}"
            if not args.dry_run:
                extract_wav_segment(args.wav, str(wav_dir / (name + ".wav")),
                                    seg_start, seg_end)
            rows.append({
                "name": name,
                "ph_seq": " ".join(ph_seq),
                "ph_dur": " ".join(f"{d:.6f}" for d in ph_dur),
                "_dur": seg_end - seg_start,
                "_np": len(ph_seq),
            })
            n_clips += 1

    csv_path = out_dir / "transcriptions.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "ph_seq", "ph_dur"])
        for r in rows:
            w.writerow([r["name"], r["ph_seq"], r["ph_dur"]])

    # report
    over = [r for r in rows if r["_dur"] > 45.0]
    print(f"play={args.play}  clips={n_clips}  csv={csv_path}")
    print(f"  align: {'constrained('+str(len(align_table))+' chars)' if args.align_ckpt else 'heuristic'}"
          f"  wavs={'SKIPPED (dry-run)' if args.dry_run else wav_dir}")
    print(f"  clips >45s (would be skipped by binarize): {len(over)}")
    if warns:
        print(f"  warnings: {len(warns)} (first 10)")
        for wln in warns[:10]:
            print("   ", wln)


if __name__ == "__main__":
    main()
