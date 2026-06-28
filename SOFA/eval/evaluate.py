#!/usr/bin/env python
"""
Character-level alignment evaluation: manual ground truth vs machine annotations.

Ports kunqu/xunmeng_SOFAevaluation_demucs_roformer.ipynb into a reusable CLI and
writes every output into a per-play folder. This is DISTINCT from the existing
kunqu/SOFA/evaluate.py, which computes phoneme-boundary metrics
(BoundaryEditRatio / VlabelerEditRatio / IoU) on TextGrid pred-vs-target dirs for
model development. This script instead compares the character-level annotation
JSONs directly: it pairs each character (zip by index within a subtitle line) and
measures absolute start/end timing error against the manual ground truth.

Naming convention (from scripts/align_pipeline.sh):
  ground truth : data/<play>/<play>_annotation.json
  machine      : data/<play>/<play>_<tag>_annotation.json   (tag e.g. roformer)

Usage (cwd = kunqu/SOFA):
  python eval/evaluate.py xunmeng
  python eval/evaluate.py xunmeng --tags roformer,demucs --thresholds 0.1,0.5,1.0
"""

import argparse
import glob
import json
import os
from collections import defaultdict

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

MISSING = "-缺字-"  # "-缺字-"


# ----------------------------------------------------------------------------
# Loading + per-character alignment (ported from notebook cells 3-4)
# ----------------------------------------------------------------------------
def load_grouped(path, key="characterAnnotations"):
    """Load a JSON annotation, group its characters by lineId, sort by startTime."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    grouped = defaultdict(list)
    for item in data["project"][key]:
        grouped[item["lineId"]].append(item)
    for lid in grouped:
        grouped[lid].sort(key=lambda x: x["startTime"])
    return data, grouped


def line_sort_key(line_id):
    try:
        return int(line_id.split("-")[1])
    except (IndexError, ValueError):
        return 0


def build_aligned_df(manual_grouped, machine_grouped, line_dur):
    """
    Pair manual vs machine characters per line (zip by index) and compute
    start/end absolute error. Unpaired positions (one side shorter) are marked
    MISSING with NaN diffs. Each row carries its line duration for short/long
    splitting.
    """
    rows = []
    all_ids = sorted(set(list(manual_grouped) + list(machine_grouped)), key=line_sort_key)
    for lid in all_ids:
        M = manual_grouped.get(lid, [])
        G = machine_grouped.get(lid, [])
        for i in range(max(len(M), len(G))):
            row = {"lineId": lid, "char_index": i + 1, "line_dur": line_dur.get(lid, np.nan)}
            if i < len(M):
                row["manual_char"] = M[i].get("char", "")
                row["manual_start"] = M[i].get("startTime")
                row["manual_end"] = M[i].get("endTime")
            else:
                row["manual_char"] = MISSING
            if i < len(G):
                row["machine_char"] = G[i].get("char", "")
                row["machine_pinyin"] = G[i].get("pinyin", "")
                row["machine_start"] = G[i].get("startTime")
                row["machine_end"] = G[i].get("endTime")
            else:
                row["machine_char"] = MISSING
                row["machine_pinyin"] = "-"
            paired = i < len(M) and i < len(G)
            row["paired"] = paired
            if paired:
                row["start_diff"] = abs(row["machine_start"] - row["manual_start"])
                row["end_diff"] = abs(row["machine_end"] - row["manual_end"])
            rows.append(row)
    cols = ["lineId", "char_index", "line_dur", "paired",
            "manual_char", "machine_char", "machine_pinyin",
            "manual_start", "machine_start", "start_diff",
            "manual_end", "machine_end", "end_diff"]
    df = pd.DataFrame(rows)
    return df[[c for c in cols if c in df.columns]]


# ----------------------------------------------------------------------------
# Stats (ported from notebook cell 6, extended: arbitrary thresholds + unaligned)
# ----------------------------------------------------------------------------
def _pct(x):
    return f"{x * 100:.2f}%"


def stats_table(df, thresholds):
    v = df.dropna(subset=["end_diff"]).copy()
    s, e = v["start_diff"], v["end_diff"]
    c = pd.concat([s, e], ignore_index=True)

    metrics = ["平均误差 (秒)", "中位数 (秒)", "标准差 (秒)"]
    metrics += [f"< {t}s 占比" for t in thresholds]

    def col(x):
        return [f"{x.mean():.4f}", f"{x.median():.4f}", f"{x.std():.4f}"] + \
               [_pct((x < t).mean()) for t in thresholds]

    both = ["—", "—", "—"] + [_pct(((s < t) & (e < t)).mean()) for t in thresholds]
    tbl = pd.DataFrame({
        "指标": metrics,
        "起始时间": col(s),
        "结束时间": col(e),
        "合并 (起始+结束)": col(c),
        "同时满足 (起始且结束)": both,
    })
    meta = {
        "N_total": len(df),
        "N_paired": int(df["paired"].sum()),
        "N_unaligned": int(len(df) - df["paired"].sum()),
    }
    return tbl, meta


def metric_series(df, thresholds):
    v = df.dropna(subset=["end_diff"])
    s, e = v["start_diff"], v["end_diff"]
    d = {"start_mean": s.mean(), "start_median": s.median(),
         "end_mean": e.mean(), "end_median": e.median(), "N": len(e)}
    for t in thresholds:
        d[f"start_<{t}s%"] = (s < t).mean() * 100
        d[f"end_<{t}s%"] = (e < t).mean() * 100
    return pd.Series(d)


def short_long_rows(df, tag, thresholds, split=45.0):
    v = df.dropna(subset=["end_diff"]).copy()
    out = []
    groups = [("all", v["line_dur"].notna() | v["line_dur"].isna()),
              (f"short<={split}s", v["line_dur"] <= split),
              (f"long>{split}s", v["line_dur"] > split)]
    for label, mask in groups:
        vv = v[mask]
        if len(vv) == 0:
            continue
        rec = {"tag": tag, "group": label, "N": len(vv),
               "start_mean": round(vv["start_diff"].mean(), 4),
               "start_median": round(vv["start_diff"].median(), 4),
               "end_mean": round(vv["end_diff"].mean(), 4),
               "end_median": round(vv["end_diff"].median(), 4)}
        for t in thresholds:
            rec[f"start_<{t}s%"] = round((vv["start_diff"] < t).mean() * 100, 1)
            rec[f"end_<{t}s%"] = round((vv["end_diff"] < t).mean() * 100, 1)
        out.append(rec)
    return out


# ----------------------------------------------------------------------------
# Error rankings (ported from notebook cell 15)
# ----------------------------------------------------------------------------
def worst_rankings(df, tag, out_dir):
    dpv = df.dropna(subset=["end_diff", "machine_pinyin"]).copy()
    pr = dpv.groupby("machine_pinyin")["end_diff"].agg(
        count="count", mean="mean", median="median", max="max").reset_index()
    pr = pr[pr["count"] > 3].sort_values("mean", ascending=False)
    pr.to_csv(os.path.join(out_dir, f"{tag}_worst_pinyin.csv"),
              index=False, encoding="utf-8-sig")

    def join_sentence(chars):
        return "".join(str(c) for c in chars if str(c) not in [MISSING, "nan"])

    dsv = df.dropna(subset=["end_diff"]).copy()
    sr = dsv.groupby("lineId").agg(
        sentence=("manual_char", join_sentence),
        n_chars=("manual_char", "count"),
        mean_end=("end_diff", "mean"),
        max_end=("end_diff", "max")).reset_index()
    sr = sr.sort_values("mean_end", ascending=False).round(4)
    sr.to_csv(os.path.join(out_dir, f"{tag}_worst_lines.csv"),
              index=False, encoding="utf-8-sig")


# ----------------------------------------------------------------------------
# Plots (ported from notebook cells 8 + 11, seaborn)
# ----------------------------------------------------------------------------
def plot_box_ecdf(df, tag, out_dir):
    v = df.dropna(subset=["end_diff"])
    df_box = pd.concat([
        pd.DataFrame({"type": "startTime", "error": v["start_diff"]}),
        pd.DataFrame({"type": "endTime", "error": v["end_diff"]}),
    ], ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.boxplot(data=df_box, x="type", y="error", hue="type", palette="Set2",
                legend=False, showfliers=False, width=0.35, ax=axes[0])
    axes[0].set_title(f"{tag}: Start/End Time Error (Boxplot)")
    axes[0].set_ylabel("Absolute Error (Seconds)")
    axes[0].set_xlabel("")
    axes[0].set_xlim(-0.8, 1.8)

    sns.ecdfplot(data=df_box, x="error", hue="type", linewidth=3, ax=axes[1])
    axes[1].set_title(f"{tag}: Cumulative Error Ratio (Higher is Better)")
    axes[1].set_xlabel("Error Threshold (Seconds)")
    axes[1].set_ylabel("Proportion of Characters")
    axes[1].set_xlim(0, 0.5)
    axes[1].axvline(x=0.1, color="red", linestyle="--", alpha=0.7)
    axes[1].text(0.12, 0.1, "0.1s Threshold", color="red", fontsize=12)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{tag}_box_ecdf.{ext}"),
                    bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_combined(dfs, out_dir, thresholds):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for tag, df in dfs.items():
        e = df.dropna(subset=["end_diff"])["end_diff"]
        sns.ecdfplot(e, label=tag, linewidth=3, ax=axes[0])
    axes[0].set_title("End-time Error ECDF (Higher is Better)")
    axes[0].set_xlabel("Error Threshold (Seconds)")
    axes[0].set_ylabel("Proportion of Characters")
    axes[0].set_xlim(0, 0.5)
    axes[0].axvline(x=0.1, color="red", linestyle="--", alpha=0.7)
    axes[0].legend(title="model")

    rates = pd.DataFrame({
        tag: {f"<{t}s": (df.dropna(subset=["end_diff"])["end_diff"] < t).mean() * 100
              for t in thresholds}
        for tag, df in dfs.items()
    }).T
    rates.plot(kind="bar", ax=axes[1], rot=0)
    axes[1].set_title("End-time Accuracy by Threshold")
    axes[1].set_ylabel("% of Characters")
    axes[1].legend(title="threshold")

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"combined_plots.{ext}"),
                    bbox_inches="tight", dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Discovery + driver
# ----------------------------------------------------------------------------
def discover_tags(data_dir, play, gt_path):
    """Find every data/<play>/<play>_<tag>_annotation.json (skip GT + legacy vocals)."""
    pat = os.path.join(data_dir, play, f"{play}_*_annotation.json")
    tags = []
    for p in sorted(glob.glob(pat)):
        base = os.path.basename(p)
        mid = base[len(play) + 1:-len("_annotation.json")]
        if not mid or mid == "vocals":
            continue
        if os.path.abspath(p) == os.path.abspath(gt_path):
            continue
        tags.append(mid)
    return tags


def main():
    ap = argparse.ArgumentParser(description="Character-level alignment eval vs manual GT")
    ap.add_argument("play", help="play name, e.g. xunmeng")
    ap.add_argument("--data-dir", default="data", help="root containing <play>/ (default: data)")
    ap.add_argument("--gt", default=None, help="override ground-truth JSON path")
    ap.add_argument("--tags", default=None, help="comma list of machine tags (default: auto-discover)")
    ap.add_argument("--out", default=None, help="output root (default: <script_dir>/outputs)")
    ap.add_argument("--thresholds", default="0.1,0.5,1.0", help="comma list of seconds")
    args = ap.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(",")]
    gt_path = args.gt or os.path.join(args.data_dir, args.play, f"{args.play}_annotation.json")
    if not os.path.exists(gt_path):
        raise SystemExit(f"ERROR: ground truth not found: {gt_path}")

    tags = args.tags.split(",") if args.tags else discover_tags(args.data_dir, args.play, gt_path)
    if not tags:
        raise SystemExit(f"ERROR: no machine annotations found for play '{args.play}' "
                         f"(expected {args.data_dir}/{args.play}/{args.play}_<tag>_annotation.json)")

    out_root = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    out_dir = os.path.join(out_root, args.play)
    os.makedirs(out_dir, exist_ok=True)

    gt_data, manual = load_grouped(gt_path)
    line_dur = {l["id"]: l["endTime"] - l["startTime"]
                for l in gt_data["project"]["subtitleLines"]}
    n_gt = len(gt_data["project"]["characterAnnotations"])

    print(f"Play         : {args.play}")
    print(f"Ground truth : {gt_path}  ({len(line_dur)} lines, {n_gt} chars)")
    print(f"Tags         : {tags}")
    print(f"Output       : {out_dir}/\n")

    dfs = {}
    short_long_all = []
    summary_lines = [f"Play: {args.play}   GT: {n_gt} chars, {len(line_dur)} lines",
                     f"Thresholds: {thresholds}", ""]

    for tag in tags:
        mpath = os.path.join(args.data_dir, args.play, f"{args.play}_{tag}_annotation.json")
        if not os.path.exists(mpath):
            print(f"[skip] {tag}: not found ({mpath})")
            summary_lines.append(f"[{tag}] SKIPPED (file not found)")
            continue

        _, machine = load_grouped(mpath)
        df = build_aligned_df(manual, machine, line_dur)
        df.to_csv(os.path.join(out_dir, f"{tag}_aligned_comparison.csv"),
                  index=False, encoding="utf-8-sig")

        tbl, meta = stats_table(df, thresholds)
        tbl.to_csv(os.path.join(out_dir, f"{tag}_stats.csv"),
                   index=False, encoding="utf-8-sig")
        with open(os.path.join(out_dir, f"{tag}_stats.tex"), "w", encoding="utf-8") as f:
            f.write(tbl.to_latex(index=False, column_format="lcccc"))

        worst_rankings(df, tag, out_dir)
        plot_box_ecdf(df, tag, out_dir)
        short_long_all.extend(short_long_rows(df, tag, thresholds))
        dfs[tag] = df

        ms = metric_series(df, thresholds)
        line = (f"[{tag}] N={meta['N_paired']} unaligned={meta['N_unaligned']}  "
                f"start: median={ms['start_median']:.3f} "
                f"<0.5s={ms.get('start_<0.5s%', float('nan')):.1f}% "
                f"<1.0s={ms.get('start_<1.0s%', float('nan')):.1f}%  "
                f"end: median={ms['end_median']:.3f} "
                f"<0.5s={ms.get('end_<0.5s%', float('nan')):.1f}% "
                f"<1.0s={ms.get('end_<1.0s%', float('nan')):.1f}%")
        summary_lines.append(line)
        print("  " + line)

    if short_long_all:
        pd.DataFrame(short_long_all).to_csv(
            os.path.join(out_dir, "short_vs_long.csv"), index=False, encoding="utf-8-sig")

    if len(dfs) >= 2:
        combined = pd.DataFrame({tag: metric_series(df, thresholds) for tag, df in dfs.items()})
        combined.round(4).to_csv(os.path.join(out_dir, "combined_comparison.csv"),
                                 encoding="utf-8-sig")
        plot_combined(dfs, out_dir, thresholds)
    else:
        summary_lines.append("\n(combined comparison skipped: need >=2 tags)")

    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    print(f"\nDone. All outputs in {out_dir}/")


if __name__ == "__main__":
    main()
