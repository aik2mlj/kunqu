#!/usr/bin/env python
"""
Phrase-level (subtitleLine) timing eval: OCR-derived annotation vs a manual
ground-truth annotation. Play-specific — only for plays that have a GT in
example/ (e.g. xunmeng). Analogous to the char-level eval (SOFA/eval/evaluate.py)
but on phrase start/end timestamps.

OCR over-segments and mislabels, so the OCR and GT subtitleLine sets differ in
count, ids, and text. We therefore MATCH phrases first (hybrid: time-overlap IoU
primary, text similarity as tiebreak, greedy one-to-one with a min-IoU floor),
then measure |Δstart|/|Δend|/IoU on matched pairs and report unmatched lines
(GT with no match = missed; OCR with no match = spurious).

Usage (from kunqu/videocr/):
    python eval_phrase_alignment.py xunmeng
    python eval_phrase_alignment.py xunmeng --min-iou 0.1 --thresholds 0.5,1.0,2.0 \
        --pred ../SOFA/data/xunmeng/xunmeng_ocr_annotation.json \
        --gt example/xunmeng_annotation.json
"""

import argparse
import difflib
import json
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Songti SC",
                                   "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", context="talk")


def load_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return d["project"]["subtitleLines"]


def _norm(s):
    return re.sub(r"\s+", "", s or "")


def iou(a, b):
    s = max(a["startTime"], b["startTime"])
    e = min(a["endTime"], b["endTime"])
    inter = max(0.0, e - s)
    union = (a["endTime"] - a["startTime"]) + (b["endTime"] - b["startTime"]) - inter
    return inter / union if union > 0 else 0.0


def text_sim(a, b):
    return difflib.SequenceMatcher(None, _norm(a["text"]), _norm(b["text"])).ratio()


def match_hybrid(gt, ocr, min_iou):
    """
    Greedy one-to-one matching. Candidates are (gt,ocr) pairs with IoU >= min_iou,
    ranked by (IoU, text_sim) descending so IoU drives matching and text breaks
    ties. Returns (pairs, unmatched_gt_idx, unmatched_ocr_idx) where each pair is
    (gi, oi, iou, text_sim).
    """
    cands = []
    for gi, g in enumerate(gt):
        for oi, o in enumerate(ocr):
            v = iou(g, o)
            if v >= min_iou:
                cands.append((v, text_sim(g, o), gi, oi))
    cands.sort(key=lambda x: (round(x[0], 6), x[1]), reverse=True)

    g_used, o_used, pairs = set(), set(), []
    for v, ts, gi, oi in cands:
        if gi in g_used or oi in o_used:
            continue
        g_used.add(gi)
        o_used.add(oi)
        pairs.append((gi, oi, v, ts))
    unmatched_gt = [i for i in range(len(gt)) if i not in g_used]
    unmatched_ocr = [i for i in range(len(ocr)) if i not in o_used]
    return pairs, unmatched_gt, unmatched_ocr


def build_pairs_df(gt, ocr, pairs):
    rows = []
    for gi, oi, v, ts in pairs:
        g, o = gt[gi], ocr[oi]
        rows.append({
            "gt_id": g["id"], "ocr_id": o["id"],
            "gt_text": g["text"], "ocr_text": o["text"],
            "gt_start": round(g["startTime"], 3), "ocr_start": round(o["startTime"], 3),
            "gt_end": round(g["endTime"], 3), "ocr_end": round(o["endTime"], 3),
            "start_diff": round(abs(o["startTime"] - g["startTime"]), 3),
            "end_diff": round(abs(o["endTime"] - g["endTime"]), 3),
            "iou": round(v, 3), "text_sim": round(ts, 3),
        })
    df = pd.DataFrame(rows).sort_values("gt_start").reset_index(drop=True)
    return df


def stats_block(series, thresholds):
    s = series.dropna()
    out = {"mean": round(s.mean(), 3), "median": round(s.median(), 3),
           "std": round(s.std(), 3)}
    for t in thresholds:
        out[f"<{t}s%"] = round((s < t).mean() * 100, 1)
    return out


def make_plots(df, out_dir, play, thresholds):
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    box = pd.concat([
        pd.DataFrame({"type": "startTime", "error": df["start_diff"]}),
        pd.DataFrame({"type": "endTime", "error": df["end_diff"]}),
    ], ignore_index=True)
    sns.boxplot(data=box, x="type", y="error", hue="type", palette="Set2",
                legend=False, showfliers=False, width=0.35, ax=axes[0])
    axes[0].set_title(f"{play}: phrase Δstart/Δend (box)")
    axes[0].set_ylabel("Absolute Error (s)")
    axes[0].set_xlabel("")

    sns.ecdfplot(df["end_diff"], linewidth=3, ax=axes[1])
    axes[1].set_title(f"{play}: Δend ECDF (higher=better)")
    axes[1].set_xlabel("Error Threshold (s)")
    axes[1].set_ylabel("Proportion of matched phrases")
    for t in thresholds:
        axes[1].axvline(x=t, color="red", linestyle="--", alpha=0.5)

    sns.ecdfplot(df["iou"], linewidth=3, ax=axes[2], color="green")
    axes[2].set_title(f"{play}: IoU ECDF (higher=better)")
    axes[2].set_xlabel("IoU")
    axes[2].set_ylabel("Proportion of matched phrases")

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{play}_phrase_eval.{ext}"),
                    bbox_inches="tight", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Phrase-level OCR-vs-GT timing eval")
    ap.add_argument("play", help="play name, e.g. xunmeng")
    ap.add_argument("--pred", default=None,
                    help="OCR annotation JSON (default: ../SOFA/data/<play>/<play>_ocr_annotation.json)")
    ap.add_argument("--gt", default=None,
                    help="ground-truth annotation JSON (default: example/<play>_annotation.json)")
    ap.add_argument("--min-iou", type=float, default=0.1, help="min IoU to consider a match")
    ap.add_argument("--thresholds", default="0.5,1.0,2.0", help="comma list of seconds")
    ap.add_argument("--out", default="output", help="output root (default: output/, i.e. output/<play>/)")
    args = ap.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(",")]
    pred = args.pred or f"../SOFA/data/xunmeng/{args.play}_ocr_annotation.json"
    gt = args.gt or f"example/{args.play}_annotation.json"
    for p in (pred, gt):
        if not os.path.exists(p):
            raise SystemExit(f"ERROR: not found: {p}")

    gt_lines = load_lines(gt)
    ocr_lines = load_lines(pred)
    pairs, um_gt, um_ocr = match_hybrid(gt_lines, ocr_lines, args.min_iou)
    df = build_pairs_df(gt_lines, ocr_lines, pairs)

    out_dir = os.path.join(args.out, args.play)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "matched_pairs.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame([{"gt_id": gt_lines[i]["id"], "gt_text": gt_lines[i]["text"],
                   "gt_start": round(gt_lines[i]["startTime"], 3),
                   "gt_end": round(gt_lines[i]["endTime"], 3)} for i in um_gt]
                 ).to_csv(os.path.join(out_dir, "unmatched_gt.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame([{"ocr_id": ocr_lines[i]["id"], "ocr_text": ocr_lines[i]["text"],
                   "ocr_start": round(ocr_lines[i]["startTime"], 3),
                   "ocr_end": round(ocr_lines[i]["endTime"], 3)} for i in um_ocr]
                 ).to_csv(os.path.join(out_dir, "unmatched_ocr.csv"), index=False, encoding="utf-8-sig")

    n_gt, n_ocr, n_match = len(gt_lines), len(ocr_lines), len(pairs)
    precision = n_match / n_ocr * 100 if n_ocr else 0.0
    recall = n_match / n_gt * 100 if n_gt else 0.0
    sstart = stats_block(df["start_diff"], thresholds)
    send = stats_block(df["end_diff"], thresholds)
    worst = df.sort_values("end_diff", ascending=False).head(5)

    lines = []
    lines.append(f"Phrase-level alignment eval: {args.play}")
    lines.append(f"pred (OCR): {pred}  ({n_ocr} lines)")
    lines.append(f"gt        : {gt}  ({n_gt} lines)")
    lines.append(f"matching  : hybrid IoU>={args.min_iou} + text tiebreak, greedy 1:1")
    lines.append("")
    lines.append(f"matched   : {n_match}   precision={precision:.1f}% (matched/OCR)   "
                 f"recall={recall:.1f}% (matched/GT)")
    lines.append(f"missed GT (no OCR match)   : {len(um_gt)}")
    lines.append(f"spurious OCR (no GT match) : {len(um_ocr)}")
    lines.append(f"mean IoU (matched)         : {df['iou'].mean():.3f}")
    lines.append("")
    lines.append(f"Δstart: mean={sstart['mean']} median={sstart['median']} "
                 + " ".join(f"<{t}s={sstart[f'<{t}s%']}%" for t in thresholds))
    lines.append(f"Δend  : mean={send['mean']} median={send['median']} "
                 + " ".join(f"<{t}s={send[f'<{t}s%']}%" for t in thresholds))
    lines.append("")
    lines.append("worst 5 matched by Δend:")
    for _, r in worst.iterrows():
        lines.append(f"  {r['gt_id']}~{r['ocr_id']} Δend={r['end_diff']}s iou={r['iou']} "
                     f"gt=\"{r['gt_text']}\" ocr=\"{r['ocr_text']}\"")
    summary = "\n".join(lines)
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary + "\n")

    make_plots(df, out_dir, args.play, thresholds)

    print(summary)
    print(f"\nOutputs -> {out_dir}/")


if __name__ == "__main__":
    main()
