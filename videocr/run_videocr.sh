#!/usr/bin/env bash
#
# run_videocr.sh — extract hardcoded (burnt-in) subtitles from a video into an
# .srt using the timminator/VideOCR GPU Docker image (PaddleOCR PP-OCRv5).
#
# Why Docker: the `videocr` package only orchestrates; the actual OCR is a
# precompiled `paddleocr` binary + bundled models that the Docker image ships.
# A plain `pip install` does NOT give you working OCR. Run this on an
# x86_64 + NVIDIA host (e.g. the RTX 4090 server), not on Apple Silicon.
#
# Usage:
#   ./run_videocr.sh <video> [output.srt] [time_start] [time_end]
#
# Examples (run from kunqu/videocr/):
#   # 1-minute test window first (fast sanity check of crop + language):
#   ./run_videocr.sh "../SOFA/data/xunmeng/央视_顾卫英《寻梦》.mp4" xunmeng.srt 3:00 4:00
#   # then the full pass:
#   ./run_videocr.sh "../SOFA/data/xunmeng/央视_顾卫英《寻梦》.mp4" xunmeng.srt
#
# Tunables can be overridden inline, e.g.:
#   CROP_Y=560 CROP_HEIGHT=120 ./run_videocr.sh video.mp4 out.srt
set -euo pipefail

# ----------------------------------------------------------------------------
# Tunables (override via environment, e.g. `OCR_LANG=ch CONF_THRESHOLD=60 ...`)
# ----------------------------------------------------------------------------
IMAGE="${IMAGE:-ghcr.io/timminator/videocr-cli-gpu-cuda12.9:latest}"
OCR_LANG="${OCR_LANG:-ch}"          # ch = Simplified+Traditional Chinese
USE_GPU="${USE_GPU:-true}"
USE_SERVER_MODEL="${USE_SERVER_MODEL:-false}"  # true = larger/more accurate, slower
CONF_THRESHOLD="${CONF_THRESHOLD:-75}"         # 0-100; lower keeps more low-confidence text
FRAMES_TO_SKIP="${FRAMES_TO_SKIP:-2}"          # higher = faster, coarser timing
SUBTITLE_POSITION="${SUBTITLE_POSITION:-center}"

# Crop = the lyric band only. Defaults tuned for this 1280x720 CCTV-11 source:
# horizontal blue lyrics sit bottom-center; this box excludes the right-side
# vertical title column, the top banners/logos and the bottom-right logo.
# Set CROP=off to OCR the full frame instead.
CROP="${CROP:-on}"
CROP_X="${CROP_X:-140}"
CROP_Y="${CROP_Y:-580}"
CROP_WIDTH="${CROP_WIDTH:-960}"
CROP_HEIGHT="${CROP_HEIGHT:-95}"

# Optional: run the container as your uid:gid so outputs aren't root-owned.
# Leave empty to use the image default (root). e.g. DOCKER_USER="$(id -u):$(id -g)"
DOCKER_USER="${DOCKER_USER:-}"

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

# Portable absolute-path helper (works on macOS + Linux, no realpath needed).
abspath() {
  local p="$1"
  if [ -d "$p" ]; then (cd "$p" && pwd)
  else (cd "$(dirname "$p")" && printf '%s/%s\n' "$(pwd)" "$(basename "$p")"); fi
}

if [ ! -f "$VIDEO" ]; then
  echo "error: video not found: $VIDEO" >&2
  exit 1
fi
if ! command -v docker >/dev/null 2>&1; then
  echo "error: docker not found on PATH. Install Docker + nvidia-container-toolkit." >&2
  exit 1
fi

VIDEO_ABS="$(abspath "$VIDEO")"
VIDEO_DIR="$(dirname "$VIDEO_ABS")"
VIDEO_BASE="$(basename "$VIDEO_ABS")"

# Default output: <video name>.srt in the current directory.
if [ -z "$OUTPUT" ]; then
  OUTPUT="${VIDEO_BASE%.*}.srt"
fi
# Make sure the output directory exists, then absolutize.
OUTPUT_DIR="$(abspath "$(dirname "$OUTPUT")")"
OUTPUT_BASE="$(basename "$OUTPUT")"

# ----------------------------------------------------------------------------
# Build the container command
# ----------------------------------------------------------------------------
docker_args=(run --rm)
[ -t 1 ] && docker_args+=(-t)
[ "$USE_GPU" = "true" ] && docker_args+=(--gpus all)
[ -n "$DOCKER_USER" ] && docker_args+=(--user "$DOCKER_USER")
docker_args+=(
  -v "$VIDEO_DIR":/in:ro
  -v "$OUTPUT_DIR":/out
  "$IMAGE"
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
  docker_args+=(--crop_x "$CROP_X" --crop_y "$CROP_Y" --crop_width "$CROP_WIDTH" --crop_height "$CROP_HEIGHT")
else
  docker_args+=(--use_fullframe true)
fi
[ -n "$TIME_START" ] && docker_args+=(--time_start "$TIME_START")
[ -n "$TIME_END" ]   && docker_args+=(--time_end "$TIME_END")

echo "==> Input : $VIDEO_ABS"
echo "==> Output: $OUTPUT_DIR/$OUTPUT_BASE"
echo "==> Image : $IMAGE"
echo "==> lang=$OCR_LANG gpu=$USE_GPU server_model=$USE_SERVER_MODEL conf=$CONF_THRESHOLD skip=$FRAMES_TO_SKIP crop=$CROP" \
     "${CROP_X:-}:${CROP_Y:-}:${CROP_WIDTH:-}:${CROP_HEIGHT:-}" \
     "${TIME_START:+time=$TIME_START-$TIME_END}"
echo
docker "${docker_args[@]}"

echo
echo "Done -> $OUTPUT_DIR/$OUTPUT_BASE"
