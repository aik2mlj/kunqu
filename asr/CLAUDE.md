# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Kunqu (昆曲) ASR pipeline — transcribe, force-align, and extract vocals from Chinese opera audio using Qwen3 models and MelBand Roformer.

## Common commands

```bash
# Run any script (venv + deps managed by uv)
uv run python scripts/<name>.py [args...]

# Submit a SLURM job (all jobs in jobs/)
sbatch jobs/<name>.sh

# Add a dependency
uv add <package>
```

## Pipeline

```
data/xunmeng.wav
  ├─[extract_vocals.py]──> vocals_output/xunmeng_vocals.wav
  ├─[transcribe.py]──> data/xunmeng.json (ASR timestamps)
  └─[align*.py]──────> aligned/xunmeng*.aligned*.json
                            └─[to_srt.py]──> aligned/*.srt
```

**Vocal extraction** (`extract_vocals.py`): uses MelBand Roformer to separate vocals from instrumental. Models auto-download to `/storage/external/lejun/`. Output: `*_vocals.wav` + `*_instrumental.wav`.

**ASR** (`transcribe.py`): transcribes audio with Qwen3-ASR-1.7B, outputs character-level timestamps to JSON.

**Force alignment** — four variants, all using Qwen3-ForcedAligner-0.6B:

| Script | Method | Input text |
|---|---|---|
| `align.py` | Whole-file | Hanzi from libretto |
| `align_pinyin.py` | Whole-file | Pinyin transliteration |
| `align_chunked.py` | Chunked by subtitle line | Hanzi |
| `align_chunked_pinyin.py` | Chunked + pinyin | Pinyin |

Chunked variants split audio by subtitle line timestamps (from `statistic-multimodal-analysis/data/annotations/xunmeng_annotation.json`) and align each chunk independently — avoids drift on long audio but requires annotation JSON.

**SRT generation** (`to_srt.py`): converts any aligned JSON to character-level SRT subtitles.

## Models on disk

All models stored at `/storage/external/lejun/`:

| Path | Purpose |
|---|---|
| `Qwen3-ASR-0.6B`, `Qwen3-ASR-1.7B` | ASR transcription |
| `Qwen3-ForcedAligner-0.6B` | Force alignment |
| `melband-roformer-kim-vocals/` | Vocal extraction (auto-downloaded) |

## SLURM

GPU type: `nvidia_a100_3g.40gb:1` (MIG slice). Job scripts discover project root from their own location — can be submitted from any directory.
