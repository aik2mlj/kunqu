# Kunqu Opera — Multimodal Analysis Pipeline

## Project Context

We are building a computational analysis pipeline for Kunqu opera (昆曲) performances. The ultimate research goal is to quantify the rhythmic synchronization between three modalities — text (lyrics), audio (singing/music), and motion (body movement) — across performances.

**This plan covers all three modalities: text (character-level lyrics alignment), audio, and motion.**

The pipeline is designed to be **performance-agnostic**: it processes any input video without assumptions about specific plays, roles, or staging. It handles common real-world conditions including moving cameras, camera angle switches, upper-body-only framing, and traditional costume occlusion.

### What we're extracting

For every frame of the video (resampled to a common fps), we want three parallel time-series:

1. **Text rhythm signals**: character onset pulses, character density, inter-character intervals, breath marks
2. **Audio rhythm signals**: onset strength, RMS energy envelope, pitch contour & pitch change rate
3. **Motion rhythm signals**: per-frame whole-body velocity, hand velocity, torso velocity (from pose estimation keypoints), computed in body-centric coordinates to be camera-invariant

These will later be analyzed for cross-modal synchrony (cross-correlation, windowed Pearson, wavelet coherence) — but that analysis step is NOT part of this plan. This plan is purely about robust extraction and storage of the signals.

---

## Directory Structure

The pipeline supports processing multiple videos. Each video is identified by a short `video_id` (e.g., `xunmeng`, `youyuan`, `jingmeng_zhangv2`). All per-video outputs go into subdirectories keyed by this ID.

```
kunqu-analysis/
├── data/
│   ├── raw/                          # Original video files
│   │   ├── xunmeng.mp4
│   │   └── youyuan.mp4
│   ├── annotations/                  # Text alignment JSON files
│   │   └── {video_id}_annotation.json
│   ├── audio/                        # Extracted audio
│   │   └── {video_id}.wav
│   ├── poses/                        # Per-frame keypoint arrays
│   │   └── {video_id}_keypoints.npz
│   └── processed/                    # Final outputs
│       ├── {video_id}_shot_boundaries.json
│       ├── {video_id}_text_features.npz
│       ├── {video_id}_text_features.json    # metadata sidecar
│       ├── {video_id}_audio_features.npz
│       ├── {video_id}_audio_features.json   # metadata sidecar
│       ├── {video_id}_motion_signals.npz
│       ├── {video_id}_motion_signals.json   # metadata sidecar
│       ├── {video_id}_aligned_signals.npz
│       └── {video_id}_aligned_signals.json  # metadata sidecar
├── src/
│   ├── detect_cuts.py                # Video → shot boundary detection
│   ├── extract_text.py               # Annotation JSON → text rhythm signals
│   ├── extract_audio.py              # Video → WAV + audio feature extraction
│   ├── extract_poses.py              # Video → per-frame pose keypoints
│   ├── compute_motion.py             # Keypoints + shot boundaries → motion signals
│   ├── align_signals.py              # Resample all signals to common timeline
│   ├── visualize.py                  # Sanity-check plots
│   ├── run_all.py                    # Run full pipeline for one or more videos
│   └── utils.py                      # Shared helpers (I/O, filtering, config)
├── notebooks/
│   └── exploration.ipynb             # Interactive exploration & QA
├── configs/
│   └── default.yaml                  # All hyperparameters in one place
├── outputs/
│   └── figures/
│       └── {video_id}/              # Per-video QA plots
├── .python-version                   # Pins Python 3.12 (read by uv automatically)
├── pyproject.toml                    # Project metadata & dependencies (managed by uv)
├── uv.lock                           # Locked dependency versions (committed to git)
└── README.md
```

---

## Config File: `configs/default.yaml`

Centralize all parameters here so nothing is hardcoded in scripts. Videos to process are listed explicitly; all other settings apply globally.

```yaml
# Videos to process — list of {id, path, annotation} entries
# Each video is processed independently through the full pipeline
videos:
  - id: "xunmeng"
    path: "data/raw/xunmeng.mp4"
    annotation: "data/annotations/xunmeng_annotation.json"
  # - id: "youyuan"
  #   path: "data/raw/youyuan.mp4"
  #   annotation: "data/annotations/youyuan_annotation.json"

# Common timeline
common_fps: 30 # All signals resampled to this rate

# Shot / cut detection
cuts:
  threshold: 27.0 # ContentDetector threshold (lower = more sensitive)
  nan_margin: 2 # Frames of NaN padding on each side of a cut

# Text rhythm extraction
text:
  density_sigma: 0.5 # Gaussian smoothing sigma (seconds) for character density curve
  breath_window: 0.15 # Half-window (seconds) around each breath mark for the breath pulse signal

# Audio extraction
audio:
  sample_rate: 22050
  hop_length: 512 # ~23ms per frame at 22050 Hz
  onset_env_fmax: 8000 # Frequency ceiling for onset detection
  rms_frame_length: 2048
  pyin_fmin: 80 # Kunqu voice lower bound (Hz) — covers dan and sheng roles
  pyin_fmax: 1000 # Upper bound — Kunqu can go very high

# Pose extraction
pose:
  model: "dwpose" # Options: "dwpose", "mediapipe"
  confidence_threshold: 0.3 # Drop keypoints below this
  batch_size: 32 # Frames per GPU batch (DWPose only)
  # If multiple people are detected in a frame:
  #   "highest_confidence" — pick the person with highest avg keypoint confidence
  #   "largest_bbox" — pick the person with the largest bounding box
  multi_person_strategy: "highest_confidence"
  # DWPose specific (ignored if model=mediapipe)
  dwpose_det_config: "rtmdet_m_640-8xb32_coco-person.py"
  dwpose_det_checkpoint: "rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
  dwpose_pose_config: "rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py"
  dwpose_pose_checkpoint: "dw-ll_ucoco_384.pth"

# Motion computation
motion:
  smoothing_window: 5 # Median filter window (frames) for keypoint stabilization
  velocity_sigma: 0.1 # Gaussian smoothing sigma (seconds) for velocity curves
  gap_max_interpolate: 3 # Max consecutive NaN frames to linearly interpolate

  # Root joint and scale reference strategy
  # The root joint defines the body-centric origin (cancels camera translation).
  # The scale reference normalizes for camera zoom / distance.
  #
  # "shoulder" — root = shoulder midpoint, scale = shoulder width
  #     Best when framing is upper-body or when hips are frequently occluded
  #     (e.g., wide sleeves, tables, upper-body camera framing).
  #
  # "hip" — root = hip midpoint, scale = torso length (neck-to-hip)
  #     Best when the full body is reliably visible.
  #
  # "auto" — start with hip; if >50% of frames have NaN hips, fall back to shoulder.
  #     Recommended default.
  root_mode: "auto"

  # Joint group definitions — two presets, selected automatically based on pose model.
  # Groups where joints are frequently out of frame will naturally produce NaN;
  # downstream aggregation (nanmean) handles this gracefully.

  joint_groups_dwpose: # DWPose: 133 keypoints (COCO-WholeBody)
    hand_left:
      [
        91,
        92,
        93,
        94,
        95,
        96,
        97,
        98,
        99,
        100,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        108,
        109,
        110,
        111,
      ]
    hand_right:
      [
        112,
        113,
        114,
        115,
        116,
        117,
        118,
        119,
        120,
        121,
        122,
        123,
        124,
        125,
        126,
        127,
        128,
        129,
        130,
        131,
        132,
      ]
    wrist_only: [9, 10] # Coarse hand tracking fallback
    shoulders: [5, 6] # Left + right shoulder
    upper_arms: [5, 6, 7, 8] # Shoulders + elbows
    forearms: [7, 8, 9, 10] # Elbows + wrists
    torso: [5, 6, 11, 12] # Shoulders + hips
    upper_body: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # Head through wrists
    head: [0, 1, 2, 3, 4] # Nose + eyes + ears
    face:
      [
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        83,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
      ]
    legs: [13, 14, 15, 16] # Knees + ankles
    full_body: null # null = use all available keypoints
    # Reference joints for root/scale computation
    shoulder_left: 5
    shoulder_right: 6
    hip_left: 11
    hip_right: 12
    neck_proxy: 0 # Nose as neck proxy (COCO has no neck keypoint)

  joint_groups_mediapipe: # MediaPipe Holistic: 33 pose + 21+21 hand landmarks
    hand_left:
      [
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
      ]
    hand_right:
      [
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        73,
        74,
      ]
    wrist_only: [15, 16]
    shoulders: [11, 12]
    upper_arms: [11, 12, 13, 14]
    forearms: [13, 14, 15, 16]
    torso: [11, 12, 23, 24]
    upper_body: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    head: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    legs: [25, 26, 27, 28]
    full_body: null
    shoulder_left: 11
    shoulder_right: 12
    hip_left: 23
    hip_right: 24
    neck_proxy: 0
```

---

## Step 1: Cut Detection — `src/detect_cuts.py`

### What it does

Detect camera angle switches (hard cuts) in the video. This must run before motion computation because velocity is undefined across cuts.

### Implementation details

```
Input:  data/raw/{video_id}.mp4
Output: data/processed/{video_id}_shot_boundaries.json
          {
            "video_id": str,
            "cuts": [1823, 4501, 7220],           # Frame indices where cuts occur
            "segments": [
              {"start_frame": 0, "end_frame": 1822, "duration_sec": 60.7},
              {"start_frame": 1824, "end_frame": 4500, "duration_sec": 89.2},
              ...
            ],
            "total_frames": int,
            "video_fps": float,
            "num_cuts": int
          }
```

### Key implementation notes

- Use PySceneDetect with `ContentDetector`. Default threshold of 27.0 is good for hard cuts; lower to ~20 if the video has dissolves or fades.
- Log the number of cuts found and their timestamps prominently. If 0 cuts are found, the video is a single continuous shot — ideal case, print a confirmation.
- **Warning heuristic**: If average segment duration is less than 60 seconds, log a warning that the video may be too heavily edited for reliable motion analysis.

### CLI interface

```bash
uv run python src/detect_cuts.py --config configs/default.yaml --video_id xunmeng
# Or process all videos listed in config:
uv run python src/detect_cuts.py --config configs/default.yaml --all
```

---

## Step 2: Text Rhythm Extraction — `src/extract_text.py`

### What it does

Parse the annotation JSON and convert character-level timing + breath marks into frame-level rhythm signals.

### Input format

The annotation JSON (produced by a separate annotation tool) has this structure:

```json
{
  "project": {
    "characterAnnotations": [
      {
        "id": "line-1-char-1",
        "lineId": "line-1",
        "char": "一",
        "startTime": 58.199,
        "endTime": 58.683,
        "singingStyle": "普通唱"
      }
    ],
    "subtitleLines": [
      {
        "id": "line-1",
        "text": "一",
        "startTime": 58.199,
        "endTime": 58.683
      }
    ],
    "builtinTracks": [
      {
        "attachedPointTracks": [
          {
            "name": "呼吸轨",
            "points": [{ "time": 61.917, "label": "呼吸" }]
          }
        ]
      }
    ]
  }
}
```

Key fields:

- **`characterAnnotations`**: One entry per sung character. `startTime`/`endTime` give the exact time span. `lineId` groups characters into lyric lines. `singingStyle` is a categorical label (e.g., `"普通唱"`, `"拖腔"`, `"念白式"`).
- **`subtitleLines`**: Line-level groupings. The `text` field may contain a single character or a multi-character phrase.
- **`builtinTracks[0].attachedPointTracks`**: Look for the track with `name == "呼吸轨"`. Its `points` array contains breath mark timestamps.

### Implementation details

```
Input:  data/annotations/{video_id}_annotation.json
Output: data/processed/{video_id}_text_features.npz
          - char_onsets: (N_chars,) array of character start times in seconds
          - char_offsets: (N_chars,) array of character end times in seconds
          - char_durations: (N_chars,) duration of each character in seconds
          - char_labels: (N_chars,) array of character strings (e.g., ["一", "径", "行", ...])
          - char_line_ids: (N_chars,) line ID for each character
          - char_singing_styles: (N_chars,) singing style label for each character
          - breath_times: (N_breaths,) array of breath mark timestamps in seconds
          - onset_signal: (T,) binary impulse signal at common_fps, 1 at each char onset frame
          - char_density: (T,) Gaussian-smoothed character density at common_fps (chars/sec)
          - char_duration_signal: (T,) per-frame character duration — for each frame that falls
              within a character span, the value is that character's duration; 0 during silences.
              This captures tempo: short durations = fast, long durations = slow.
          - inter_onset_interval: (T,) for each character onset frame, the time since the
              previous onset; interpolated linearly between onsets. Captures local pacing.
          - breath_signal: (T,) impulse signal at breath marks (Gaussian pulse, sigma = breath_window)
          - silence_mask: (T,) boolean, True for frames not covered by any character span
          - times: (T,) time axis in seconds at common_fps
        data/processed/{video_id}_text_features.json   (metadata sidecar)
          - video_id, num_characters, num_breaths, num_lines,
            total_singing_duration_sec, silence_fraction,
            singing_style_counts (e.g., {"普通唱": 45, "拖腔": 3})
```

### Processing steps

1. **Parse JSON**: Load the annotation file. Extract `characterAnnotations`, sorted by `startTime`. Extract breath points from the attached point track named `"呼吸轨"`.

2. **Validate timing**: Check that character spans don't overlap (warn if they do — could indicate annotation errors). Check that all `startTime < endTime`. Check that times are within the video duration (if known from other pipeline steps).

3. **Build character arrays**: Create the sorted arrays `char_onsets`, `char_offsets`, `char_durations`, `char_labels`, `char_line_ids`, `char_singing_styles` directly from the parsed annotations.

4. **Determine total duration**: The frame-level signals need a total length T. Use the maximum of: the last character's `endTime`, the last breath mark's `time`, or (if available from other pipeline outputs) the video duration from audio/motion metadata. This ensures the text signals span the same duration as the other modalities.

5. **Generate frame-level signals** at `common_fps`:

   a. **`onset_signal`**: Binary impulse — for each character, set `signal[round(startTime * fps)] = 1`.

   b. **`char_density`**: Gaussian-smooth the onset_signal with `sigma = density_sigma * common_fps`. This gives a continuous "characters per second" curve. Peaks indicate dense singing; valleys indicate pauses or long sustained notes.

   c. **`char_duration_signal`**: For each frame `t`, find if it falls within any character's `[startTime, endTime)` span. If yes, set the value to that character's duration. If no, set to 0. This is a step function that directly encodes singing tempo.

   d. **`inter_onset_interval`**: At each character onset, compute the time gap from the previous onset. Linearly interpolate between onsets to fill all frames. First onset gets NaN. This captures pacing rhythm — regular IOI = steady pace, variable IOI = rubato.

   e. **`breath_signal`**: Place a Gaussian pulse (`sigma = breath_window * common_fps`) at each breath timestamp. This marks phrasing boundaries.

   f. **`silence_mask`**: True for frames not within any character span. Useful for downstream analysis to distinguish "no singing" from "sustained note".

6. **Compute metadata**: Count characters, breaths, lines. Compute silence fraction (% of total annotated time span that is silence). Tally singing styles. Write to the JSON sidecar.

### CLI interface

```bash
uv run python src/extract_text.py --config configs/default.yaml --video_id xunmeng
uv run python src/extract_text.py --config configs/default.yaml --all
```

---

## Step 3: Audio Extraction — `src/extract_audio.py`

### What it does

1. Extract audio track from video → save as WAV
2. Compute frame-level audio features
3. Save as numpy arrays

### Implementation details

```
Input:  data/raw/{video_id}.mp4
Output: data/audio/{video_id}.wav
        data/processed/{video_id}_audio_features.npz
          - onset_env: (N_audio_frames,) onset strength envelope
          - rms: (N_audio_frames,) RMS energy
          - f0: (N_audio_frames,) fundamental frequency (NaN for unvoiced)
          - pitch_delta: (N_audio_frames,) absolute frame-to-frame f0 change
          - times: (N_audio_frames,) timestamp in seconds for each frame
        data/processed/{video_id}_audio_features.json   (metadata sidecar)
          - video_id, sample_rate, hop_length, duration_sec,
            audio_frame_rate (= sample_rate / hop_length)
```

### Key implementation notes

- **Audio extraction**: Use `ffmpeg` via subprocess, not moviepy (lighter dependency). Command: `ffmpeg -i input.mp4 -vn -acodec pcm_s16le -ar 22050 -ac 1 output.wav`. Check return code and that the output file is non-empty. **Note**: `ffmpeg` is a system dependency — verify it is installed before running this step (e.g., `shutil.which("ffmpeg")`) and print a clear error message if missing.
- **Onset envelope**: `librosa.onset.onset_strength()` — captures note attacks in the singing and percussion hits from accompaniment. Set `fmax=8000` to avoid high-frequency noise.
- **RMS energy**: `librosa.feature.rms()` — use `frame_length=2048` for smoothness.
- **Pitch (f0)**: `librosa.pyin()` — critical for Kunqu since melodic contour is a major rhythmic signal. `fmin`/`fmax` are set in config to cover both dan and sheng roles. Store NaN for unvoiced frames as-is — do NOT interpolate (that's an analysis decision for later).
- **Pitch delta**: Compute as `np.abs(np.diff(f0, prepend=np.nan))`, then explicitly set to 0.0 wherever either the current or previous frame's f0 is NaN (unvoiced). This ensures we only measure pitched melodic movement and avoids the edge case where `f0[0]` is itself NaN.
- **All arrays share the same time axis**: `librosa.times_like(onset_env, sr=sr, hop_length=hop)`.
- Audio features are **not affected by camera cuts** — the soundtrack is continuous regardless of visual editing. No special handling needed here.

### CLI interface

```bash
uv run python src/extract_audio.py --config configs/default.yaml --video_id xunmeng
uv run python src/extract_audio.py --config configs/default.yaml --all
```

---

## Step 4: Pose Extraction — `src/extract_poses.py`

### What it does

1. Run pose estimation on every frame of the video
2. Save per-frame keypoints with confidence scores
3. Apply basic quality filtering

### Implementation details

```
Input:  data/raw/{video_id}.mp4
Output: data/poses/{video_id}_keypoints.npz
          - keypoints: (T, J, 2) x,y coordinates — T=total frames, J=num joints
          - confidence: (T, J) confidence score per keypoint per frame
          - frame_valid: (T,) boolean, True if a person was detected
        data/poses/{video_id}_keypoints.json   (metadata sidecar)
          - video_id, model_name, num_joints, video_fps,
            total_frames, frame_width, frame_height
```

### DWPose setup (preferred)

DWPose gives 133 keypoints (COCO-WholeBody): 17 body + 6 feet + 68 face + 42 hands. The hand keypoints are important for Kunqu gesture analysis.

**Installation**: DWPose uses MMPose under the hood. Install the optional dependency group:

```bash
uv sync --extra dwpose
# Then download DWPose checkpoints (see config for filenames)
```

If DWPose setup is too painful (MMPose dependency hell), **fall back to MediaPipe** — see below.

### MediaPipe fallback implementation

**Note**: The legacy `mp.solutions.holistic` API was deprecated and removed. Use the **MediaPipe Tasks API** (`mediapipe.tasks.vision.PoseLandmarker`) instead.

MediaPipe Pose gives 33 pose landmarks. Hand landmarks require a separate `HandLandmarker` task. For simplicity, the fallback uses pose-only (33 joints); hand keypoints will be NaN, and downstream motion analysis will rely on wrist keypoints (indices 15, 16) for coarse hand tracking.

If full hand detail is needed, run `HandLandmarker` as a second pass and concatenate to get 33 + 21 + 21 = 75 joints matching the `joint_groups_mediapipe` config.

Implementation approach:

```python
from mediapipe.tasks.vision import PoseLandmarker, PoseLandmarkerOptions
from mediapipe.tasks import BaseOptions
import mediapipe as mp

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker_heavy.task"),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_poses=1,
)
landmarker = PoseLandmarker.create_from_options(options)
# For each frame: result = landmarker.detect_for_video(mp_image, timestamp_ms)
# result.pose_landmarks[0] → 33 NormalizedLandmark objects
# Convert to pixel coords using frame width/height
```

**Important**: The code must read `pose.model` from config and branch accordingly. Joint group indices are also selected from the matching config preset (`joint_groups_dwpose` or `joint_groups_mediapipe`). This ensures downstream code (compute_motion.py) works identically regardless of which model was used.

### Key implementation notes

- **Frame reading**: Use `cv2.VideoCapture` to iterate frames. Read FPS from `cap.get(cv2.CAP_PROP_FPS)` and total frames from `cap.get(cv2.CAP_PROP_FRAME_COUNT)`. Store both in metadata.
- **Batch processing**: For DWPose, batch frames (e.g. 32 at a time) for GPU efficiency. For MediaPipe, process frame-by-frame (it doesn't support batching).
- **Multi-person handling**: Use the strategy specified in config (`multi_person_strategy`). Log a warning with the frame number and detection count whenever >1 person is detected — useful for spotting secondary performers or audience.
- **Missing detections**: If no person is detected in a frame, mark `frame_valid[t] = False` and fill keypoints with NaN. Do NOT forward-fill — that's handled in compute_motion.py.
- **Coordinate system**: Store raw pixel coordinates. Normalization happens in compute_motion.py.
- **Progress bar**: Use tqdm. For a 30-min video at 30fps that's ~54,000 frames — takes ~30–60 min on a decent GPU with DWPose, ~15 min with MediaPipe on CPU.

### CLI interface

```bash
uv run python src/extract_poses.py --config configs/default.yaml --video_id xunmeng
uv run python src/extract_poses.py --config configs/default.yaml --all
# Override model:
uv run python src/extract_poses.py --config configs/default.yaml --video_id xunmeng --model mediapipe
```

---

## Step 5: Motion Signals — `src/compute_motion.py`

### What it does

1. Load raw keypoints and shot boundaries
2. Filter and stabilize keypoints
3. Convert to body-centric, scale-normalized coordinates (camera-invariant)
4. Compute velocity-based motion signals for different body regions
5. Respect shot boundaries — never compute velocity across camera cuts

### Implementation details

```
Input:  data/poses/{video_id}_keypoints.npz
        data/processed/{video_id}_shot_boundaries.json
        configs/default.yaml
Output: data/processed/{video_id}_motion_signals.npz
          - total_motion: (T-1,) whole-body avg joint velocity (body-centric, normalized)
          - hand_motion: (T-1,) combined hand motion (mean of left + right)
          - hand_left_motion: (T-1,)
          - hand_right_motion: (T-1,)
          - torso_motion: (T-1,)
          - upper_body_motion: (T-1,)
          - head_motion: (T-1,)
          - root_displacement: (T-1,) root joint pixel displacement per frame (camera diagnostic)
          - scale_reference: (T,) per-frame scale value in pixels (shoulder width or torso length)
          - times: (T-1,) timestamps in seconds
        data/processed/{video_id}_motion_signals.json   (metadata sidecar)
          - video_id, fps, pose_model, joint_groups_used,
            smoothing_params,
            coordinate_mode="body_centric_normalized",
            root_mode_resolved=str (which root was actually used),
            root_joint=str ("shoulder_midpoint" or "hip_midpoint"),
            scale_reference_type=str ("shoulder_width" or "torso_length"),
            scale_reference_cv=float,
            num_cuts=int, cut_frames=list,
            nan_fraction=float,
            hip_nan_fraction=float (for auto mode logging)
```

### Processing pipeline (in order)

1. **Load and validate**: Load `{video_id}_keypoints.npz`. Read `pose_model` from its metadata sidecar, then select the matching joint group definitions from config (`joint_groups_dwpose` or `joint_groups_mediapipe`). Verify that the number of joints J matches the expected count for that model (133 for DWPose, 75 for MediaPipe).

2. **Mask low-confidence keypoints**: Set keypoints with `confidence < threshold` to NaN.

3. **Resolve root mode**: If `root_mode` is `"auto"`:
   - Count the fraction of frames where either hip keypoint has NaN or confidence below threshold.
   - If hip NaN fraction > 50%, use `"shoulder"` mode. Log: `"Auto root selection: using shoulders (hips unavailable in {X}% of frames)"`
   - Otherwise, use `"hip"` mode. Log: `"Auto root selection: using hips (available in {X}% of frames)"`

   Store the resolved mode in `root_mode_resolved` in metadata.

4. **Interpolate short gaps**: For each joint independently, if there's a consecutive NaN gap of ≤ `gap_max_interpolate` frames (default 3), linearly interpolate. Longer gaps stay NaN — they likely represent real occlusions or detection failures.

5. **Median filter**: Apply `scipy.signal.medfilt(keypoints[:, j, dim], kernel_size=smoothing_window)` per joint per dimension to remove per-frame jitter. This is critical — raw pose estimation has frame-to-frame noise that creates false motion spikes. Handle NaN values by skipping them (apply filter only on valid segments).

6. **Convert to body-centric coordinates** (camera-invariant):

   The camera may pan, track, or switch angles. Raw pixel-space velocity would reflect camera motion, not performer motion. To fix this, compute a root-relative version of all keypoints.

   If root mode is `"shoulder"`:

   ```
   root[t] = (keypoints[t, SHOULDER_LEFT] + keypoints[t, SHOULDER_RIGHT]) / 2
   ```

   If root mode is `"hip"`:

   ```
   root[t] = (keypoints[t, HIP_LEFT] + keypoints[t, HIP_RIGHT]) / 2
   ```

   Then:

   ```
   keypoints_local[t, j] = keypoints[t, j] - root[t]
   ```

   This cancels out any global translation (camera pan, dolly, or performer walking) and isolates gesture/posture movement relative to the body center.

   If either reference joint is NaN at frame t, root[t] is NaN, and all keypoints_local[t] become NaN (conservative — don't guess the root).

7. **Scale normalization**: Even in body-centric coordinates, the magnitude depends on apparent body size (camera zoom/distance). Normalize by a stable anatomical reference:

   If root mode is `"shoulder"`:

   ```
   scale[t] = ||keypoints[t, SHOULDER_LEFT] - keypoints[t, SHOULDER_RIGHT]||
   ```

   (shoulder width)

   If root mode is `"hip"`:

   ```
   scale[t] = ||keypoints[t, NECK_PROXY] - root[t]||
   ```

   (torso length, neck to hip midpoint)

   Then:

   ```
   keypoints_normalized[t, j] = keypoints_local[t, j] / scale[t]
   ```

   This makes the signal invariant to both camera translation AND zoom. Save `scale` as a diagnostic signal. Compute and store the coefficient of variation (std/mean) as `scale_reference_cv`.

   If `scale[t]` is NaN or near zero (< 5 pixels, likely bad detection), set all normalized keypoints at that frame to NaN.

8. **Compute velocity, respecting shot boundaries**:

   Load `{video_id}_shot_boundaries.json`. Never compute velocity across a cut — the frame pair straddling a cut is meaningless.

   ```
   for each consecutive frame pair (t, t+1):
       if a cut occurs between t and t+1:
           v[t, :] = NaN   # undefined at cut boundary
       else:
           v[t, j] = ||keypoints_normalized[t+1, j] - keypoints_normalized[t, j]|| * fps
   ```

   Also set a ±`nan_margin` frame (default 2) NaN padding on each side of a cut to absorb any pose estimation instability during the transition (some codecs blend frames near cuts).

   Units are "body-proportions per second" — fully camera-invariant.

   Additionally, compute `root_displacement[t] = ||root_pixel[t+1] - root_pixel[t]|| * fps` in raw pixel coordinates (before root subtraction). This is a diagnostic: if the camera is static, it captures performer global motion; if the camera tracks, it stays near zero.

9. **Aggregate by body region**: For each joint group defined in config, compute the mean of per-joint velocities:

   ```
   region_motion[t] = nanmean(v[t, j] for j in group)
   ```

   Use nanmean (not sum) so that missing joints from occlusion don't create artificial dips — the signal represents "average joint speed in this region" regardless of how many joints are tracked in that frame. If ALL joints in a group are NaN at frame t, the output is NaN for that frame.

10. **Gaussian smooth the final signals**: Apply `scipy.ndimage.gaussian_filter1d(signal, sigma=velocity_sigma * fps)` to get clean curves suitable for correlation analysis. Before smoothing, temporarily fill NaN with 0, smooth, then re-apply the NaN mask. (This avoids NaN propagation through the Gaussian kernel while keeping the NaN positions marked.)

11. **Compute summary diagnostics**: Calculate and store in metadata:
    - `nan_fraction`: % of frames that are NaN in `total_motion` (from cuts + failed detections combined). If >15%, log a warning.
    - `scale_reference_cv`: coefficient of variation of the scale reference signal.
    - `hip_nan_fraction`: % of frames with NaN hips (useful context regardless of which root was chosen).
    - `num_cuts` and `cut_frames`: copied from shot_boundaries.json for convenience.

### Camera motion diagnostic summary

As part of QA, the script prints a human-readable summary:

```
[xunmeng] Camera analysis:
  - Root mode: shoulder (auto — hips unavailable in 78% of frames)
  - Detected cuts: 5
  - Scale reference CV: 0.06 (low → camera distance is stable)
  - Mean root displacement: 8.1 px/sec
  - Motion NaN fraction: 3.2%
  Recommendation: body-centric signals are reliable.
```

### CLI interface

```bash
uv run python src/compute_motion.py --config configs/default.yaml --video_id xunmeng
uv run python src/compute_motion.py --config configs/default.yaml --all
```

---

## Step 6: Signal Alignment — `src/align_signals.py`

### What it does

Resample all three modalities to a common time axis and save a single aligned file ready for analysis.

### Implementation details

```
Input:  data/processed/{video_id}_text_features.npz
        data/processed/{video_id}_audio_features.npz
        data/processed/{video_id}_motion_signals.npz
        data/processed/{video_id}_shot_boundaries.json
Output: data/processed/{video_id}_aligned_signals.npz
          - times: (N,) common time axis at common_fps (seconds)
          # Text
          - text_onset: (N,) character onset impulses
          - text_density: (N,) smoothed character density
          - text_char_duration: (N,) per-frame character duration (tempo proxy)
          - text_ioi: (N,) inter-onset interval (pacing proxy)
          - text_breath: (N,) breath mark pulses
          - text_silence_mask: (N,) boolean, True during non-singing frames
          # Audio
          - audio_onset: (N,)
          - audio_rms: (N,)
          - audio_f0: (N,)
          - audio_pitch_delta: (N,)
          # Motion
          - motion_total: (N,)
          - motion_hand: (N,)
          - motion_hand_left: (N,)
          - motion_hand_right: (N,)
          - motion_torso: (N,)
          - motion_upper_body: (N,)
          - motion_head: (N,)
          - motion_root_displacement: (N,)
          # Masks
          - cut_mask: (N,) boolean, True at frames within nan_margin of a cut
        data/processed/{video_id}_aligned_signals.json   (metadata sidecar)
          - merged text + audio + motion metadata, plus:
            video_id, common_fps, total_duration_sec, total_frames,
            nan_fraction_motion, text_coverage_fraction
```

### Key logic

- Audio features have a different frame rate (`sr / hop_length` ≈ 43 fps at default settings) than motion (video fps, ~24–30). Use `np.interp` to resample everything to `common_fps` (30). This is sufficient since the signals are already smooth.
- Text features are already generated at `common_fps` in extract_text.py, so they can be copied directly. Just verify length matches and trim if needed.
- Motion signals are already at video fps. If video fps ≠ common_fps (e.g., video is 24fps but common_fps is 30), resample motion too. If they match, just copy.
- **Verify duration alignment**: Check that all three modalities have approximately equal duration (from their respective metadata). If any differ by more than 0.5 seconds, log a warning — audio/video tracks may have slight length mismatches from codec behavior. **Trim all signals to the duration of the shortest one.**
- **Preserve NaN**: Motion signals contain NaN at cut boundaries and failed detections. These must survive resampling. Approach: resample the valid (non-NaN) portions via `np.interp`, then re-apply NaN at the corresponding positions in the new time grid.
- **Generate `cut_mask`**: A boolean array where `True` marks frames within `nan_margin` of any cut. Downstream analysis can use this to skip cut regions or segment continuous shots.
- **Print summary** on completion:

  ```
  [xunmeng] Aligned signals saved: 54000 frames, 1800.0 sec, 30 fps
  Text signals: 6 channels (onset, density, char_duration, ioi, breath, silence_mask)
  Audio signals: 4 channels (onset, rms, f0, pitch_delta)
  Motion signals: 7 channels (total, hand, hand_left, hand_right, torso, upper_body, head)
  NaN coverage in motion: 2.3% (from 5 cuts + 12 failed detection frames)
  Text coverage: 78.2% of frames have active singing
  ```

### CLI interface

```bash
uv run python src/align_signals.py --config configs/default.yaml --video_id xunmeng
uv run python src/align_signals.py --config configs/default.yaml --all
```

---

## Step 7: Sanity Check Visualization — `src/visualize.py`

### Quick plots to verify everything works

Generate per-video QA plots automatically after the pipeline runs. All plots use matplotlib.

**Plot 1 — Signal overview** (`outputs/figures/{video_id}/01_signals_overview.png`):

A vertically stacked plot with shared x-axis (time in seconds), ~7 rows:

- Row 1: Audio onset envelope
- Row 2: Audio RMS energy
- Row 3: Audio pitch contour (f0), with unvoiced gaps visible
- Row 4: Text character density (smoothed) with character onset ticks along the top
- Row 5: Text breath marks (vertical ticks) + silence regions (grey shading)
- Row 6: Total body motion
- Row 7: Hand motion (blue) vs torso motion (orange), overlaid
- Vertical red dashed lines at every detected camera cut across all rows
- Light red shading over the NaN-margin zones around each cut

This is the most important QA plot. Visually inspect: do motion spikes loosely correspond to audio/text events? Are there weird artifacts? Do the cut boundaries look correct?

**Plot 2 — Pose quality report** (`outputs/figures/{video_id}/02_pose_quality.png`):

- Top: % of valid keypoints per frame over time, split into upper-body joints vs lower-body joints (expect lower-body to be much worse if framing is upper-body)
- Middle: Average confidence score over time (computed over upper-body joints only)
- Bottom: Scale reference signal over time (shoulder width or torso length, depending on resolved root mode)
- Highlight regions where upper-body quality drops below thresholds.

**Plot 3 — Short clip deep-dive** (`outputs/figures/{video_id}/03_clip_detail.png`):
Zoom into a 30-second segment (pick the segment with the most motion activity, or make it configurable). Same stacked layout as Plot 1 but at a scale where you can see individual note onsets, character boundaries, and motion peaks align (or not).

**Plot 4 — Camera diagnostic** (`outputs/figures/{video_id}/04_camera_diagnostic.png`):

- Top: Root joint pixel displacement over time — shows camera/performer global motion
- Bottom: Scale reference signal over time with its mean ± 1 std marked
- Annotate with the computed `scale_reference_cv` value and the resolved root mode

### CLI interface

```bash
uv run python src/visualize.py --config configs/default.yaml --video_id xunmeng
uv run python src/visualize.py --config configs/default.yaml --all
# Optional: specify clip range for Plot 3
uv run python src/visualize.py --config configs/default.yaml --video_id xunmeng --clip_start 120 --clip_end 150
```

---

## Batch Runner — `src/run_all.py`

Convenience script that runs the full pipeline (steps 1–7) for all videos listed in config, with proper error handling.

```bash
uv run python src/run_all.py --config configs/default.yaml
# Or for a single video:
uv run python src/run_all.py --config configs/default.yaml --video_id xunmeng
```

For each video, it runs steps sequentially and stops on that video (but continues to the next) if any step fails. Prints a summary at the end:

```
Pipeline complete.
  xunmeng:  ✓ all steps passed (7/7)
  youyuan:  ✗ failed at extract_poses (MediaPipe returned 0 detections)
```

---

## `pyproject.toml`

We use **[uv](https://docs.astral.sh/uv/)** for dependency management and running scripts. `uv` handles virtual-environment creation, dependency resolution, locking, and Python version management in a single fast tool.

```toml
[project]
name = "kunqu-analysis"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "numpy>=1.24",
    "scipy>=1.10",
    "librosa>=0.10",
    "matplotlib>=3.7",
    "opencv-python>=4.8",
    "tqdm>=4.65",
    "pyyaml>=6.0",
    "scenedetect[opencv]>=0.6",
]

[project.optional-dependencies]
# Pose estimation backends — install the one you need
mediapipe = ["mediapipe>=0.10"]
dwpose = [
    "openmim>=0.3",
    "mmengine>=0.10",
    "mmcv>=2.1",
    "mmdet>=3.2",
    "mmpose>=1.2",
]
# For interactive exploration
notebooks = ["jupyterlab>=4.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### System dependencies

The following must be installed outside of Python (not managed by uv):

- **ffmpeg** — required by `extract_audio.py` for audio extraction. Install via system package manager (`apt install ffmpeg`, `brew install ffmpeg`, etc.)
- **CUDA toolkit** — required only for DWPose (GPU-accelerated pose estimation). Not needed for the MediaPipe fallback.

---

## Execution Order (per video)

```bash
# 0. Setup — uv handles venv creation and dependency installation automatically
uv python install 3.12           # Download Python 3.12 if not already available
uv sync                          # Install core dependencies
uv sync --extra mediapipe        # Also install MediaPipe backend
# Or for DWPose:
# uv sync --extra dwpose

mkdir -p data/{raw,annotations,audio,poses,processed} outputs/figures
# Place video file(s) in data/raw/, annotation JSONs in data/annotations/
# List them in configs/default.yaml

# Run everything (uv run activates the project venv automatically):
uv run python src/run_all.py --config configs/default.yaml

# Or step by step for a single video:
uv run python src/detect_cuts.py --config configs/default.yaml --video_id xunmeng
uv run python src/extract_text.py --config configs/default.yaml --video_id xunmeng
uv run python src/extract_audio.py --config configs/default.yaml --video_id xunmeng
uv run python src/extract_poses.py --config configs/default.yaml --video_id xunmeng
uv run python src/compute_motion.py --config configs/default.yaml --video_id xunmeng
uv run python src/align_signals.py --config configs/default.yaml --video_id xunmeng
uv run python src/visualize.py --config configs/default.yaml --video_id xunmeng
```

Each step is idempotent — re-running overwrites previous output. Each step validates that its input files exist before starting and prints a clear error message (e.g., `"Error: data/poses/xunmeng_keypoints.npz not found. Run extract_poses.py first."`).

Steps 1, 2, and 3 have no dependencies on each other and can run in parallel.

---

## What "done" looks like

The pipeline is complete when, for each video:

1. `data/processed/{video_id}_aligned_signals.npz` exists and contains all signals at the same length and sample rate
2. The signal overview plot looks reasonable — no massive artifacts, signals have plausible dynamics, cut boundaries are correctly placed
3. The pose quality report shows high valid-frame rates for the joints that are expected to be visible (upper-body if upper-body framing, full body if full body)
4. The camera diagnostic confirms that body-centric normalization is working (scale_reference_cv is documented, root mode selection is logged and makes sense)
5. The `nan_fraction` in metadata is documented and acceptable (<15%)
6. A researcher can load the aligned file in one line (`data = np.load('data/processed/xunmeng_aligned_signals.npz')`) and load metadata via `json.load(open('data/processed/xunmeng_aligned_signals.json'))`, then immediately start computing cross-correlations between any pair of signals across all three modalities

**This pipeline is the foundation.** With all three modalities in a single aligned file, the downstream cross-modal analysis (correlation, wavelet coherence, per-section comparison) can proceed directly.
