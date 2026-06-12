#!/usr/bin/env bash
#
# Run the full SOFA alignment pipeline on the demucs separated vocals.
#   wav  : data/xunmeng/xunmeng_vocals_demucs.wav   (formerly xunmeng_vocals.wav)
#   out  : segments/xunmeng_demucs/  +  data/xunmeng/xunmeng_demucs_annotation.json
#
# CKPT can be overridden, e.g.:
#   CKPT=/abs/path/model.ckpt scripts/run_demucs.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/align_pipeline.sh" \
  data/xunmeng/xunmeng_vocals_demucs.wav \
  demucs
