# VideOCR — extracting burnt-in subtitles to SRT

Extract hardcoded (burned-in) subtitles from kunqu videos that have no `.srt`,
using [timminator/VideOCR](https://github.com/timminator/VideOCR) (PaddleOCR
PP-OCRv5). Output is a standard `.srt` with timestamps.

## How it actually works (read this first)

VideOCR's `videocr` Python package is only a thin orchestrator. At runtime it
**shells out to a precompiled `paddleocr` binary** and a bundled
`PaddleOCR.PP-OCRv5.support.files/` model folder. Its `pyproject.toml` ships
**zero** packages and the project is compiled with Nuitka — so **`pip install`
does not give you working OCR**. The two supported setups are the **Docker GPU
image** and the **precompiled Linux GPU release**. We use the Docker image.

Because the engine is an x86_64 + CUDA binary, **run this on the RTX 4090
server, not on Apple Silicon.** (No native macOS/arm build exists; some deps
such as `cpuid` are x86-only.)

## Prerequisites (on the server)

- NVIDIA GPU + recent driver
- Docker
- [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  (so `docker run --gpus all` works)

Verify GPU access:

```bash
docker run --rm --gpus all nvidia/cuda:12.9.0-base-ubuntu22.04 nvidia-smi
```

## Quick start

```bash
docker pull ghcr.io/timminator/videocr-cli-gpu-cuda12.9:latest

cd kunqu/videocr

# 1) Fast sanity check on a 1-minute window (verify crop + language are right):
./run_videocr.sh "../SOFA/data/xunmeng/央视_顾卫英《寻梦》.mp4" xunmeng.srt 3:00 4:00

# 2) Inspect xunmeng.srt. If the lyrics look right, run the full pass:
./run_videocr.sh "../SOFA/data/xunmeng/央视_顾卫英《寻梦》.mp4" xunmeng.srt
```

`run_videocr.sh <video> [output.srt] [time_start] [time_end]` mounts the video's
folder into the container, runs OCR on the GPU, and writes the SRT next to where
you point `output`.

Compose alternative:

```bash
DATA_DIR=../SOFA/data/xunmeng docker compose run --rm videocr \
  --video_path "/data/央视_顾卫英《寻梦》.mp4" --output "/data/xunmeng.srt" \
  --lang ch --use_gpu true \
  --crop_x 140 --crop_y 580 --crop_width 960 --crop_height 95
```

## Cluster / no-Docker (udocker) — for the GPU server

Shared HPC boxes usually **can't run Docker** (it needs a root daemon). Use
[`udocker`](https://github.com/indigo-dc/udocker) instead: it runs the same image
rootless, in userspace, with GPU support. Use `run_videocr_udocker.sh` (same args
and tunables as `run_videocr.sh`).

```bash
# one-time, in any conda env on the server
pip install udocker

cd kunqu/videocr

# 1) 1-minute sanity window (first run also pulls the image + sets up GPU — slow once):
./run_videocr_udocker.sh "../SOFA/data/xunmeng/央视_顾卫英《寻梦》.mp4" xunmeng.srt 3:00 4:00

# 2) if the lyrics look right, full pass:
./run_videocr_udocker.sh "../SOFA/data/xunmeng/央视_顾卫英《寻梦》.mp4" xunmeng.srt
```

The script bootstraps udocker automatically (`udocker install`, `pull`, `create`,
`setup --nvidia`) on first run; later runs reuse the `videocr` container. Force a
CPU run with `USE_GPU=false`. If apptainer/singularity is available instead,
prefer it (`apptainer run --nv docker://<image> ...`).

## One command per play (`run_play.sh`)

`run_play.sh` orchestrates the whole pipeline for one play and keeps every
artifact under `output/<play>/`. The OCR engine is still udocker (it calls
`run_videocr_udocker.sh`).

```bash
cd kunqu/videocr

# Probe only: write the frame with the crop box drawn, so you can tune the crop.
./run_play.sh probe shihuajiaohua "../SOFA/data/shihuajiaohua/东方大剧院_岳美缇《拾画叫画》.mp4"
#   -> open output/shihuajiaohua/probe_crop.jpg ; the red box should hug the lyrics

# Tune the crop on a short window (no convert on a windowed run):
CROP_Y=600 CROP_HEIGHT=110 ./run_play.sh shihuajiaohua "<video>" 2:00 3:00

# Happy with the crop? Full pass — also converts SRT -> annotation JSON:
CROP_Y=600 CROP_HEIGHT=110 ./run_play.sh shihuajiaohua "<video>"
```

The crop is set by `CROP_*` env vars (see Tuning); the probe image redraws with
whatever you pass, so you iterate by re-running. No prompt — a windowed run stops
after OCR (so a bad crop never writes a JSON); a full run (no time window) also
converts.

**Per-play crop is version-controlled** in `plays/<play>.env` (e.g.
`plays/shihuajiaohua.env`). `run_play.sh` auto-loads it, so once tuned the crop
is committed and every run reuses it — no need to remember `CROP_*` on the
command line. Precedence: command-line `CROP_*` > `plays/<play>.env` > built-in
defaults. To tune a new play: run `probe`, eyeball `output/<play>/probe_crop.jpg`,
edit `plays/<play>.env`, repeat; commit the `.env` when happy.

**Output layout** (`output/<play>/`): `probe.jpg`, `probe_crop.jpg`,
`<play>.srt`, `<play>_ocr_cleanup_report.txt`, and phrase-eval files. Only the
`.srt` is committed; images/csv/plots/reports are gitignored. The converted
annotation JSON lands in its SOFA home: `../SOFA/data/<play>/<play>_ocr_annotation.json`.

For a new play you only ever re-tune the crop; everything else is just the play
name and video path.

## Why these defaults (the 寻梦 source)

The target video is 1280×720, 30 fps, ~24.9 min, from CCTV-11 (戏曲). The frame
has several text regions; only the **lyrics** should be OCR'd:

- **Lyrics** — horizontal blue Chinese text, bottom-center (≈ y 600–660). ← want this
- Right-side **vertical** title column: 昆曲 / 牡丹亭 / 选段 / 寻梦 ← exclude
- Top: CCTV-11 戏曲 logo, "多剧种折子戏专场二(下)" banner, CCTV.com watermark ← exclude
- Bottom-right: 九州大戏台 logo ← exclude

So the defaults are `--lang ch` plus a crop limited to the bottom lyric band:

```
--crop_x 140 --crop_y 580 --crop_width 960 --crop_height 95
```

This box (x∈[140,1100], y∈[580,675]) keeps the centered lyrics and drops the
vertical column and corner logos. Without a crop, the vertical title would be
read on every frame and pollute the output.

## Tuning

Override any tunable inline, e.g. `CROP_Y=560 CROP_HEIGHT=120 ./run_videocr.sh ...`:

| Variable | Default | Notes |
|---|---|---|
| `OCR_LANG` | `ch` | PaddleOCR language code |
| `USE_GPU` | `true` | set `false` to force CPU |
| `USE_SERVER_MODEL` | `false` | `true` = larger, more accurate, slower model |
| `CONF_THRESHOLD` | `75` | lower (e.g. 60) keeps more low-confidence text |
| `FRAMES_TO_SKIP` | `2` | higher = faster, coarser timing |
| `CROP` | `on` | set `off` to OCR the full frame |
| `CROP_X/Y/WIDTH/HEIGHT` | `140/580/960/95` | the lyric band |
| `IMAGE` | cuda12.9 image | switch to a CUDA 11.8 tag if needed |
| `DOCKER_USER` | _(empty)_ | set to `$(id -u):$(id -g)` to avoid root-owned output |

The same `CROP_*` vars work for `run_play.sh`, which additionally draws the box on
the probe frame. Easiest way to tune for a new video:

```bash
./run_play.sh probe <play> "<video>"        # -> output/<play>/probe_crop.jpg (box drawn)
# adjust CROP_* until the red box hugs the lyrics, then run the full pass
```

Or dump a raw frame manually:

```bash
ffmpeg -ss 200 -i "<video>" -frames:v 1 probe.jpg
```

## Notes

- **Do not commit the video** (373 MB) or other large media — the repo
  `.gitignore` excludes `*.mp4 / *.wav / *.7z`. The SRT output is small and fine
  to commit.
- `requirements.txt` here lists the `videocr` package's Python deps for the
  from-source/dev path only; it is **not** enough to run OCR (the engine +
  models come from the Docker image).
