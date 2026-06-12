#!/usr/bin/env bash
#
# Run the full SOFA alignment pipeline on the mel-band-roformer separated vocals.
#   wav  : data/xunmeng/xunmeng_vocals_mel-band-roformer.wav
#   out  : segments/xunmeng_roformer/  +  data/xunmeng/xunmeng_roformer_annotation.json
#
# CKPT can be overridden, e.g.:
#   CKPT=/abs/path/model.ckpt scripts/run_roformer.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/align_pipeline.sh" \
  data/xunmeng/xunmeng_vocals_mel-band-roformer.wav \
  roformer
