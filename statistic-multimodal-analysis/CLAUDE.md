# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kunqu-analysis is a computational pipeline for analyzing rhythmic synchronization between audio, text, and motion in Kunqu opera (昆曲) performances. It extracts frame-level signals from video and quantifies cross-modal synchrony at various temporal scales.

## Setup & Running

```bash
# Install dependencies (uses uv + hatchling)
uv sync --extra notebooks --extra mediapipe

# Run full pipeline for a video
uv run python src/run_all.py --config configs/default.yaml --video_id xunmeng

# Run individual steps
uv run python src/detect_cuts.py --config configs/default.yaml --video_id xunmeng
uv run python src/extract_audio.py --config configs/default.yaml --video_id xunmeng
uv run python src/extract_poses.py --config configs/default.yaml --video_id xunmeng --model mediapipe
uv run python src/compute_motion.py --config configs/default.yaml --video_id xunmeng
uv run python src/align_signals.py --config configs/default.yaml --video_id xunmeng
uv run python src/visualize.py --config configs/default.yaml --video_id xunmeng

# Process all configured videos
uv run python src/run_all.py --config configs/default.yaml --all
```

DWPose requires additional setup: `pip install openmim && mim install mmengine mmcv mmdet mmpose`

No test suite or linter is configured.

## Architecture

### Pipeline Flow

```
detect_cuts → extract_audio → extract_poses → compute_motion → align_signals → visualize
```

Each step is a standalone CLI script in `src/` that reads from and writes to `data/processed/` using NPZ + JSON sidecar pairs. All steps share config via YAML files in `configs/`.

### Key Source Files

- **`src/run_all.py`** — Orchestrates the full pipeline
- **`src/utils.py`** — Shared helpers: config loading, I/O, CLI arg parsing, `PipelineError` exception
- **`src/detect_cuts.py`** — Shot boundary detection via PySceneDetect
- **`src/extract_audio.py`** — Librosa-based audio feature extraction (onset strength, RMS, PYIN pitch)
- **`src/extract_poses.py`** — Pose estimation supporting DWPose (133 keypoints) and MediaPipe (33 landmarks)
- **`src/compute_motion.py`** — Camera-invariant motion signals from raw keypoints
- **`src/align_signals.py`** — Resamples audio + motion to a common timeline (default 30fps)
- **`src/visualize.py`** — QA visualization plots

### Coordinate Transform Chain

Raw pixel coords → body-centric (root joint subtraction) → scale-normalized (divided by shoulder width or torso length) → resampled to common fps

### Design Decisions

- **NaN-aware everywhere**: Uses `nanmean` instead of `sum` for robust aggregation during pose occlusions
- **Shot boundary respect**: Velocity is never computed across camera cuts
- **Config-driven**: All hyperparameters live in YAML configs, including joint group definitions per pose model
- **Audio tuning**: PYIN frequency bounds (80–1000 Hz) are set for Kunqu vocal range

## Data Layout

- `data/raw/` — Source video files
- `data/audio/` — Extracted WAV files
- `data/poses/` — Raw pose keypoints
- `data/processed/` — Pipeline outputs (NPZ + JSON per step)
- `configs/` — YAML pipeline configs (`default.yaml` uses shoulder root; `shoulder.yaml` is an alias)
- `notebooks/` — Jupyter exploration and analysis
- `reports/` — Generated analysis reports and slides
- `kunqu-pipeline-plan.md` — Detailed technical spec for the pipeline (reference doc)
