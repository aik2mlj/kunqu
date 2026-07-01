#!/usr/bin/env bash
#
# run_play.sh — one-command videocr pipeline for a single play, with auto-probe.
#
# Two forms:
#   ./run_play.sh probe <play> <video> [time]       # only make the probe image
#   ./run_play.sh <play> <video> [time_start] [time_end]   # full: probe+OCR+convert
#
# What it does (full form):
#   1. PROBE  — print resolution; grab a frame and draw the CURRENT crop box on it
#               -> output/<play>/probe.jpg and output/<play>/probe_crop.jpg
#   2. OCR    — run_videocr_udocker.sh -> output/<play>/<play>.srt
#   3. CONVERT (full pass only, i.e. no time window) — srt_to_annotation.py
#               -> ../SOFA/data/<play>/<play>_ocr_annotation.json
#               -> cleanup report in output/<play>/
#
# Crop tuning loop (no prompt): the crop is set by CROP_* env vars. Re-run on a
# short window, eyeball output/<play>/probe_crop.jpg (red box should hug the
# lyrics) and the .srt, adjust CROP_* and re-run; when right, run with no window
# to do the full pass + convert.
#
#   CROP_Y=600 CROP_HEIGHT=110 ./run_play.sh shihuajiaohua "<video>" 2:00 3:00   # tune
#   CROP_Y=600 CROP_HEIGHT=110 ./run_play.sh shihuajiaohua "<video>"             # full
#
# Requires ffprobe/ffmpeg + udocker (see README). Run from kunqu/videocr/.
set -euo pipefail

# --- crop config: CLI env > plays/<play>.env > defaults ---------------------
# Capture crop values passed on the command line; these win over the per-play
# config file (sourced below, once the play name is known).
_CLI_CROP="${CROP:-}"; _CLI_CROP_X="${CROP_X:-}"; _CLI_CROP_Y="${CROP_Y:-}"
_CLI_CROP_W="${CROP_WIDTH:-}"; _CLI_CROP_H="${CROP_HEIGHT:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- subcommand dispatch -----------------------------------------------------
MODE="full"
if [ "${1:-}" = "probe" ]; then
  MODE="probe"
  shift
fi

if [ "$#" -lt 2 ]; then
  echo "usage: $0 [probe] <play> <video> [time_start] [time_end]" >&2
  exit 2
fi
PLAY="$1"
VIDEO="$2"
TIME_START="${3:-}"
TIME_END="${4:-}"

command -v ffprobe >/dev/null 2>&1 || { echo "error: ffprobe not found (install ffmpeg)" >&2; exit 1; }
command -v ffmpeg  >/dev/null 2>&1 || { echo "error: ffmpeg not found" >&2; exit 1; }
[ -f "$VIDEO" ] || { echo "error: video not found: $VIDEO" >&2; exit 1; }

# per-play crop config (version-controlled): plays/<play>.env
if [ -f "plays/$PLAY.env" ]; then
  # shellcheck disable=SC1090
  . "plays/$PLAY.env"
  echo "==> loaded crop config plays/$PLAY.env"
fi
# precedence: command-line env > plays/<play>.env > built-in defaults
CROP="${_CLI_CROP:-${CROP:-on}}"
CROP_X="${_CLI_CROP_X:-${CROP_X:-140}}"
CROP_Y="${_CLI_CROP_Y:-${CROP_Y:-580}}"
CROP_WIDTH="${_CLI_CROP_W:-${CROP_WIDTH:-960}}"
CROP_HEIGHT="${_CLI_CROP_H:-${CROP_HEIGHT:-95}}"
export CROP CROP_X CROP_Y CROP_WIDTH CROP_HEIGHT

OUT="output/$PLAY"
mkdir -p "$OUT"

# --- 1. PROBE ---------------------------------------------------------------
RES="$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 "$VIDEO" || true)"
PROBE_T="${TIME_START:-120}"   # ffmpeg -ss accepts seconds or HH:MM:SS / MM:SS
echo "==> [$PLAY] resolution: ${RES:-unknown}   probe frame at: $PROBE_T"

ffmpeg -nostdin -loglevel error -ss "$PROBE_T" -i "$VIDEO" -frames:v 1 -y "$OUT/probe.jpg"
if [ "$CROP" = "on" ]; then
  ffmpeg -nostdin -loglevel error -ss "$PROBE_T" -i "$VIDEO" -frames:v 1 \
    -vf "drawbox=x=${CROP_X}:y=${CROP_Y}:w=${CROP_WIDTH}:h=${CROP_HEIGHT}:color=red:thickness=3" \
    -y "$OUT/probe_crop.jpg"
  echo "==> crop box ${CROP_X},${CROP_Y} ${CROP_WIDTH}x${CROP_HEIGHT} drawn -> $OUT/probe_crop.jpg"
else
  echo "==> CROP=off (full frame); probe -> $OUT/probe.jpg"
fi

if [ "$MODE" = "probe" ]; then
  echo "Probe only. Open $OUT/probe_crop.jpg, adjust CROP_* if needed, then re-run without 'probe'."
  exit 0
fi

# --- 2. OCR -----------------------------------------------------------------
SRT="$OUT/$PLAY.srt"
echo "==> [$PLAY] OCR -> $SRT"
./run_videocr_udocker.sh "$VIDEO" "$SRT" ${TIME_START:+"$TIME_START"} ${TIME_END:+"$TIME_END"}

# --- 3. CONVERT (full pass only) --------------------------------------------
if [ -n "$TIME_START" ]; then
  echo "==> windowed test run (start=$TIME_START) — skipping convert."
  echo "    Inspect $OUT/probe_crop.jpg + $SRT; when the crop looks right, re-run with no time window."
  exit 0
fi

JSON="../SOFA/data/$PLAY/${PLAY}_ocr_annotation.json"
REPORT="$OUT/${PLAY}_ocr_cleanup_report.txt"
echo "==> [$PLAY] convert -> $JSON"
python srt_to_annotation.py "$SRT" \
  --out "$JSON" \
  --report "$REPORT" \
  --video-name "$(basename "$VIDEO")"

echo
echo "Done [$PLAY]:"
echo "  probe : $OUT/probe_crop.jpg"
echo "  srt   : $SRT"
echo "  json  : $JSON"
echo "  report: $REPORT"
