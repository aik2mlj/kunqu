#!/usr/bin/env bash
#
# Core SOFA forced-alignment pipeline for ONE source-separated vocal track.
#
# Given a long vocal .wav and a short tag (e.g. "roformer" / "demucs"), it runs
# the exact same three steps end to end:
#
#   1. Segment the long vocal wav into per-phrase clips (ALL lines are kept;
#      SOFA has no hard length cap -- 45s is only a training-data filter, and
#      direct inference stays accurate well past it) and emit matching .lab
#      pinyin transcriptions  ->  segments/<play>_<tag>/
#   2. Run SOFA forced alignment on that folder              ->  .../TextGrid/
#   3. Convert the TextGrid output back into a character-level annotation JSON
#                                          ->  data/<play>/<play>_<tag>_annotation.json
#
# Usage:
#   scripts/align_pipeline.sh <wav_path> <tag>          # PLAY defaults to xunmeng
#   PLAY=<play> ANNOTATION=<json> scripts/align_pipeline.sh <wav_path> <tag>
#
# Example:
#   scripts/align_pipeline.sh data/xunmeng/xunmeng_vocals_mel-band-roformer.wav roformer
#   PLAY=shihuajiaohua ANNOTATION=data/shihuajiaohua/shihuajiaohua_ocr_annotation.json \
#     scripts/align_pipeline.sh data/shihuajiaohua/shihuajiaohua_vocals_mel-band-roformer.wav roformer
#
# Override any of these via environment variables:
#   PLAY          play slug; keys all in/out paths (default: xunmeng)
#   CKPT          path to the SOFA .ckpt model   (default: ckpt/pretrained_mandarin_singing/v1.0.0_mandarin_singing.ckpt)
#   ANNOTATION    source subtitle annotation     (default: data/<play>/<play>_annotation.json)
#   DICT          pronunciation dictionary       (default: dictionary/opencpop-extension.txt)
#   MAX_DURATION  warn-only threshold in seconds (default: 45; long lines kept)
#   PYTHON        python interpreter             (default: python)

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <wav_path> <tag>" >&2
  echo "  e.g. $0 data/xunmeng/xunmeng_vocals_mel-band-roformer.wav roformer" >&2
  exit 1
fi

WAV="$1"
TAG="$2"

# Resolve the SOFA project root (the parent of this scripts/ folder) and cd into
# it so every relative path below resolves no matter where this is invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOFA_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SOFA_ROOT"

PYTHON="${PYTHON:-python}"
PLAY="${PLAY:-xunmeng}"
CKPT="${CKPT:-ckpt/pretrained_mandarin_singing/v1.0.0_mandarin_singing.ckpt}"
ANNOTATION="${ANNOTATION:-data/${PLAY}/${PLAY}_annotation.json}"
DICT="${DICT:-dictionary/opencpop-extension.txt}"
MAX_DURATION="${MAX_DURATION:-45}"

SEGMENTS_DIR="segments/${PLAY}_${TAG}"
TEXTGRID_DIR="${SEGMENTS_DIR}/TextGrid"
OUT_JSON="data/${PLAY}/${PLAY}_${TAG}_annotation.json"

echo "============================================================"
echo "SOFA alignment pipeline: ${PLAY} / ${TAG}"
echo "  SOFA root   : ${SOFA_ROOT}"
echo "  wav         : ${WAV}"
echo "  annotation  : ${ANNOTATION}"
echo "  ckpt        : ${CKPT}"
echo "  segments    : ${SEGMENTS_DIR}"
echo "  output json : ${OUT_JSON}"
echo "  max dur     : ${MAX_DURATION}s"
echo "============================================================"

# Fail early with actionable messages if an input is missing.
[[ -f "$WAV" ]]        || { echo "ERROR: wav not found: $WAV" >&2; exit 1; }
[[ -f "$ANNOTATION" ]] || { echo "ERROR: annotation not found: $ANNOTATION" >&2; exit 1; }
[[ -f "$DICT" ]]       || { echo "ERROR: dictionary not found: $DICT" >&2; exit 1; }
[[ -f "$CKPT" ]]       || { echo "ERROR: checkpoint not found: $CKPT" >&2; \
                            echo "       Set CKPT=/path/to/model.ckpt and re-run." >&2; exit 1; }

# ---- 1. Segment -------------------------------------------------------------
echo
echo ">>> [1/3] Segmenting $WAV -> $SEGMENTS_DIR (all lines; warn if > ${MAX_DURATION}s)"
"$PYTHON" data/prepare_segments.py \
  --annotation "$ANNOTATION" \
  --wav "$WAV" \
  --output "$SEGMENTS_DIR" \
  --dictionary "$DICT" \
  --max-duration "$MAX_DURATION"

# ---- 2. Forced alignment ----------------------------------------------------
echo
echo ">>> [2/3] Running SOFA forced alignment on $SEGMENTS_DIR"
"$PYTHON" infer.py \
  --ckpt "$CKPT" \
  --folder "$SEGMENTS_DIR" \
  --dictionary "$DICT" \
  --out_formats textgrid

# ---- 3. TextGrid -> annotation JSON ----------------------------------------
echo
echo ">>> [3/3] Converting TextGrid output -> $OUT_JSON"
"$PYTHON" data/textgrid_to_annotation.py \
  --annotation "$ANNOTATION" \
  --textgrid-dir "$TEXTGRID_DIR" \
  --output "$OUT_JSON" \
  --padding 0

echo
echo "Done. Character-level annotation written to: $OUT_JSON"
