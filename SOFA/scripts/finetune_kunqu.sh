#!/usr/bin/env bash
#
# finetune_kunqu.sh — one cross-play fold of the kunqu SOFA finetune, end to end.
#
#   Fold A: train on xunmeng,       held-out test = shihuajiaohua
#   Fold B: train on shihuajiaohua, held-out test = xunmeng
#
# Stages (all on the GPU server, conda env with SOFA deps):
#   1. build   full_label data from the train play's human GT (constrained align)
#   2. binarize (per-fold data_folder; copies the hardcoded global_config across)
#   3. train    from the pretrained ckpt, backbone frozen (train_config_finetune)
#   4. infer    the finetuned model on the held-out play (align_pipeline.sh)
#   5. eval     collapse-rate + Δ vs GT, finetuned tag next to the roformer baseline
#
# Usage (cwd anywhere):
#   scripts/finetune_kunqu.sh A          # or B
#   START_STAGE=4 scripts/finetune_kunqu.sh A     # resume from a later stage
#
# Requires the separated vocals wavs to exist:
#   data/<play>/<play>_vocals_mel-band-roformer.wav
set -euo pipefail

FOLD="${1:-}"
case "$FOLD" in A|B|ALL) ;; *) echo "usage: $0 A|B|ALL" >&2; exit 2;; esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOFA_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SOFA_ROOT"

PYTHON="${PYTHON:-python}"
CKPT="${CKPT:-ckpt/pretrained_mandarin_singing/v1.0.0_mandarin_singing.ckpt}"
DICT="${DICT:-dictionary/opencpop-extension.txt}"
START_STAGE="${START_STAGE:-1}"

# --- per-play ground-truth + wav paths --------------------------------------
gt_of() { case "$1" in
  xunmeng)       echo data/xunmeng/xunmeng_annotation.json ;;
  shihuajiaohua) echo data/shihuajiaohua/shihuajiaohua_roformer_annotation_human_gt.json ;;
esac; }
wav_of() { echo "data/$1/$1_vocals_mel-band-roformer.wav"; }

# --- fold -> train plays + held-out test ------------------------------------
#   A/B = cross-play validation (train one, test the other, held-out).
#   ALL = production model: train on both, no held-out (ship this after A/B pass).
case "$FOLD" in
  A)   TRAIN_PLAYS=(xunmeng);              TEST_PLAY=shihuajiaohua ;;
  B)   TRAIN_PLAYS=(shihuajiaohua);        TEST_PLAY=xunmeng ;;
  ALL) TRAIN_PLAYS=(xunmeng shihuajiaohua); TEST_PLAY="" ;;
esac
TEST_GT=""; TEST_WAV=""
if [ -n "$TEST_PLAY" ]; then TEST_GT="$(gt_of "$TEST_PLAY")"; TEST_WAV="$(wav_of "$TEST_PLAY")"; fi

DATA_ROOT="data/finetune_${FOLD}"
MODEL_NAME="kunqu_finetune_${FOLD}"
TAG="ft_${FOLD}"
GEN_BIN_CFG="${DATA_ROOT}/binarize_config.yaml"
GEN_TRAIN_CFG="${DATA_ROOT}/train_config.yaml"

echo "=========================================================="
echo "Fold ${FOLD}:  train=${TRAIN_PLAYS[*]}  test=${TEST_PLAY:-<none (production)>}"
echo "  data root : ${DATA_ROOT}"
echo "  ckpt      : ${CKPT}"
echo "  from stage: ${START_STAGE}"
echo "=========================================================="
[ -f "$CKPT" ] || { echo "ERROR: pretrained ckpt not found: $CKPT" >&2; exit 1; }

# --- 1. build full_label data (constrained per-syllable align) --------------
if [ "$START_STAGE" -le 1 ]; then
  echo ">>> [1/5] build full_label data -> ${DATA_ROOT}/full_label/"
  for p in "${TRAIN_PLAYS[@]}"; do
    w="$(wav_of "$p")"; g="$(gt_of "$p")"
    [ -f "$w" ] || { echo "ERROR: train vocals wav not found: $w" >&2; exit 1; }
    "$PYTHON" data/annotation_to_training.py \
      --annotation "$g" --play "$p" --out-root "$DATA_ROOT" \
      --wav "$w" --align-ckpt "$CKPT" --dictionary "$DICT"
  done
fi

# --- 2. binarize (per-fold data_folder + global_config fixup) ---------------
if [ "$START_STAGE" -le 2 ]; then
  echo ">>> [2/5] binarize"
  mkdir -p "$DATA_ROOT"
  sed "s#^data_folder:.*#data_folder: ${DATA_ROOT}#" \
    configs/binarize_config_finetune.yaml > "$GEN_BIN_CFG"
  "$PYTHON" binarize.py -c "$GEN_BIN_CFG"
  # binarize.py writes global_config.yaml to a hardcoded data/binary/; train.py
  # reads it from <data_folder>/binary/ — copy it across.
  mkdir -p "${DATA_ROOT}/binary"
  cp data/binary/global_config.yaml "${DATA_ROOT}/binary/global_config.yaml"
fi

# --- 3. train (freeze backbone, from pretrained) ----------------------------
if [ "$START_STAGE" -le 3 ]; then
  echo ">>> [3/5] train (frozen backbone) -> ckpt/${MODEL_NAME}/"
  sed "s#^model_name:.*#model_name: ${MODEL_NAME}#" \
    configs/train_config_finetune.yaml > "$GEN_TRAIN_CFG"
  "$PYTHON" train.py -c "$GEN_TRAIN_CFG" --data_folder "$DATA_ROOT" -p "$CKPT"
fi

# newest finetuned checkpoint (search recursively; lightning may nest under version_*/)
FT_CKPT="$(find "ckpt/${MODEL_NAME}" -type f -name '*.ckpt' 2>/dev/null | xargs -r ls -t 2>/dev/null | head -1 || true)"

# --- 4. infer finetuned model on the held-out play --------------------------
if [ "$START_STAGE" -le 4 ] && [ -n "$TEST_PLAY" ]; then
  echo ">>> [4/5] infer finetuned model on held-out ${TEST_PLAY} (tag=${TAG})"
  [ -n "$FT_CKPT" ] || { echo "ERROR: no finetuned ckpt under ckpt/${MODEL_NAME}/" >&2; exit 1; }
  [ -f "$TEST_WAV" ] || { echo "ERROR: test vocals wav not found: $TEST_WAV" >&2; exit 1; }
  echo "    finetuned ckpt: $FT_CKPT"
  CKPT="$FT_CKPT" PLAY="$TEST_PLAY" ANNOTATION="$TEST_GT" \
    scripts/align_pipeline.sh "$TEST_WAV" "$TAG"
fi

# --- 5. eval: collapse-rate + Δ vs GT, finetuned next to roformer baseline ---
if [ "$START_STAGE" -le 5 ] && [ -n "$TEST_PLAY" ]; then
  echo ">>> [5/5] eval on held-out ${TEST_PLAY}"
  TAGS="$TAG"
  [ -f "data/${TEST_PLAY}/${TEST_PLAY}_roformer_annotation.json" ] && TAGS="${TAG},roformer"
  "$PYTHON" eval/evaluate.py "$TEST_PLAY" --gt "$TEST_GT" --tags "$TAGS"
  echo
  echo "Compare collapse.csv: ${TAG} (finetuned) vs roformer (pretrained baseline)."
  echo "Success = ${TAG} <50ms% below roformer, end<0.5s% not worse."
fi

if [ "$FOLD" = "ALL" ]; then
  echo
  echo "Production model trained on both plays. Checkpoint:"
  echo "    ${FT_CKPT:-ckpt/${MODEL_NAME}/ (see newest *.ckpt)}"
  echo "Use it to align new plays:  CKPT=<that ckpt> scripts/align_pipeline.sh <wav> <tag>"
fi

echo "Fold ${FOLD} done."
