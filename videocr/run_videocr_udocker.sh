#!/usr/bin/env bash
#
# run_videocr_udocker.sh — rootless variant of run_videocr.sh for clusters/HPC
# boxes where you CANNOT install Docker (no root, no daemon).
#
# It runs the same timminator/VideOCR GPU image (PaddleOCR PP-OCRv5) entirely in
# userspace via udocker. Same arguments and tunables as run_videocr.sh.
#
# One-time setup (in any conda env on the server):
#   pip install udocker
#
# Usage (from kunqu/videocr/):
#   ./run_videocr_udocker.sh <video> [output.srt] [time_start] [time_end]
#
# Examples:
#   # 1-minute sanity window first (check crop + language):
#   ./run_videocr_udocker.sh "../SOFA/data/xunmeng/央视_顾卫英《寻梦》.mp4" xunmeng.srt 3:00 4:00
#   # then the full pass:
#   ./run_videocr_udocker.sh "../SOFA/data/xunmeng/央视_顾卫英《寻梦》.mp4" xunmeng.srt
#
# Tunables can be overridden inline, e.g.:
#   CONF_THRESHOLD=60 ./run_videocr_udocker.sh video.mp4 out.srt
set -euo pipefail

# ----------------------------------------------------------------------------
# Tunables (override via environment) — same as run_videocr.sh
# ----------------------------------------------------------------------------
IMAGE="${IMAGE:-ghcr.io/timminator/videocr-cli-gpu-cuda12.9:latest}"
CONTAINER="${UDOCKER_CONTAINER:-videocr}"      # udocker container name (created once)
OCR_LANG="${OCR_LANG:-ch}"                      # ch = Simplified+Traditional Chinese
USE_GPU="${USE_GPU:-true}"
USE_SERVER_MODEL="${USE_SERVER_MODEL:-false}"  # true = larger/more accurate, slower
CONF_THRESHOLD="${CONF_THRESHOLD:-75}"         # 0-100; lower keeps more low-confidence text
FRAMES_TO_SKIP="${FRAMES_TO_SKIP:-2}"          # higher = faster, coarser timing
SUBTITLE_POSITION="${SUBTITLE_POSITION:-center}"

# Crop = the lyric band only (see run_videocr.sh / README for why). CROP=off => full frame.
CROP="${CROP:-on}"
CROP_X="${CROP_X:-140}"
CROP_Y="${CROP_Y:-580}"
CROP_WIDTH="${CROP_WIDTH:-960}"
CROP_HEIGHT="${CROP_HEIGHT:-95}"

# ----------------------------------------------------------------------------
# Args
# ----------------------------------------------------------------------------
if [ "$#" -lt 1 ]; then
  echo "usage: $0 <video> [output.srt] [time_start] [time_end]" >&2
  exit 2
fi
VIDEO="$1"
OUTPUT="${2:-}"
TIME_START="${3:-}"
TIME_END="${4:-}"

# Portable absolute-path helper (no realpath needed).
abspath() {
  local p="$1"
  if [ -d "$p" ]; then (cd "$p" && pwd)
  else (cd "$(dirname "$p")" && printf '%s/%s\n' "$(pwd)" "$(basename "$p")"); fi
}

if [ ! -f "$VIDEO" ]; then
  echo "error: video not found: $VIDEO" >&2
  exit 1
fi
if ! command -v udocker >/dev/null 2>&1; then
  echo "error: udocker not found on PATH. In your conda env run: pip install udocker" >&2
  exit 1
fi

VIDEO_ABS="$(abspath "$VIDEO")"
VIDEO_DIR="$(dirname "$VIDEO_ABS")"
VIDEO_BASE="$(basename "$VIDEO_ABS")"

if [ -z "$OUTPUT" ]; then
  OUTPUT="${VIDEO_BASE%.*}.srt"
fi
OUTPUT_DIR="$(abspath "$(dirname "$OUTPUT")")"
OUTPUT_BASE="$(basename "$OUTPUT")"

# ----------------------------------------------------------------------------
# One-time udocker bootstrap (all idempotent / tolerant of "already done")
# ----------------------------------------------------------------------------
udocker install >/dev/null 2>&1 || true
if ! udocker inspect "$CONTAINER" >/dev/null 2>&1; then
  echo "==> first run: pulling image + creating container '$CONTAINER' (large, one-time)"
  udocker pull "$IMAGE"
  udocker create --name="$CONTAINER" "$IMAGE" || true
fi
if [ "$USE_GPU" = "true" ]; then
  # Wire host NVIDIA libs into the container. Safe to repeat.
  udocker setup --nvidia "$CONTAINER" >/dev/null 2>&1 \
    || udocker setup --nvidia --force "$CONTAINER" >/dev/null 2>&1 || true
fi

# ----------------------------------------------------------------------------
# Build the CLI args passed to the image entrypoint (same set as docker variant)
# ----------------------------------------------------------------------------
cli=(
  --video_path "/in/$VIDEO_BASE"
  --output "/out/$OUTPUT_BASE"
  --lang "$OCR_LANG"
  --use_gpu "$USE_GPU"
  --use_server_model "$USE_SERVER_MODEL"
  --conf_threshold "$CONF_THRESHOLD"
  --frames_to_skip "$FRAMES_TO_SKIP"
  --subtitle_position "$SUBTITLE_POSITION"
)
if [ "$CROP" = "on" ]; then
  cli+=(--crop_x "$CROP_X" --crop_y "$CROP_Y" --crop_width "$CROP_WIDTH" --crop_height "$CROP_HEIGHT")
else
  cli+=(--use_fullframe true)
fi
[ -n "$TIME_START" ] && cli+=(--time_start "$TIME_START")
[ -n "$TIME_END" ]   && cli+=(--time_end "$TIME_END")

echo "==> Input : $VIDEO_ABS"
echo "==> Output: $OUTPUT_DIR/$OUTPUT_BASE"
echo "==> udocker container: $CONTAINER  (image $IMAGE)"
echo "==> lang=$OCR_LANG gpu=$USE_GPU server_model=$USE_SERVER_MODEL conf=$CONF_THRESHOLD skip=$FRAMES_TO_SKIP crop=$CROP" \
     "${CROP_X:-}:${CROP_Y:-}:${CROP_WIDTH:-}:${CROP_HEIGHT:-}" \
     "${TIME_START:+time=$TIME_START-$TIME_END}"
echo

udocker run \
  -v "$VIDEO_DIR":/in \
  -v "$OUTPUT_DIR":/out \
  "$CONTAINER" "${cli[@]}"

echo
echo "Done -> $OUTPUT_DIR/$OUTPUT_BASE"
