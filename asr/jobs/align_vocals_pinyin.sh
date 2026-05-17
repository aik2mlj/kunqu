#!/bin/bash
#SBATCH --job-name=align-voc-py
#SBATCH --gres=gpu:nvidia_a100_3g.40gb:1
#SBATCH --cpus-per-gpu=16
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "GPU:      $CUDA_VISIBLE_DEVICES"
echo "Started:  $(date)"

uv run python scripts/align_chunked_pinyin.py vocals_output/xunmeng_vocals.wav

echo "Finished align: $(date)"

uv run python scripts/to_srt.py aligned/xunmeng_vocals.aligned_chunked_pinyin.json aligned/xunmeng_vocals.aligned_chunked_pinyin.srt

echo "SRT generated: $(date)"
