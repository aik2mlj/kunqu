"""Step 2: Extract audio from video and compute audio features."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np

from utils import (
    base_argparser,
    load_config,
    get_video_entry,
    resolve_video_ids,
    require_file,
    require_ffmpeg,
    save_signals,
)


def extract_wav(video_path: Path, wav_path: Path, sample_rate: int) -> None:
    """Extract audio track from video to WAV using ffmpeg."""
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    if not wav_path.exists() or wav_path.stat().st_size == 0:
        print(f"Error: ffmpeg produced empty or missing output: {wav_path}", file=sys.stderr)
        sys.exit(1)


def compute_audio_features(wav_path: Path, cfg: dict) -> tuple[dict[str, np.ndarray], dict]:
    """Compute onset, RMS, pitch, and pitch-delta from a WAV file."""
    audio_cfg = cfg["audio"]
    sr = audio_cfg["sample_rate"]
    hop = audio_cfg["hop_length"]

    y, _ = librosa.load(wav_path, sr=sr)
    duration_sec = len(y) / sr

    # Onset strength envelope
    onset_env = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=hop, fmax=audio_cfg["onset_env_fmax"]
    )

    # RMS energy
    rms = librosa.feature.rms(
        y=y, frame_length=audio_cfg["rms_frame_length"], hop_length=hop
    )[0]

    # Pitch (f0) via pyin
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=audio_cfg["pyin_fmin"], fmax=audio_cfg["pyin_fmax"],
        sr=sr, hop_length=hop,
    )

    # Pitch delta — absolute frame-to-frame f0 change
    pitch_delta = np.abs(np.diff(f0, prepend=np.nan))
    # Zero out where either adjacent frame is NaN (unvoiced)
    nan_mask = np.isnan(f0) | np.isnan(np.roll(f0, 1))
    nan_mask[0] = True  # first frame has no previous
    pitch_delta[nan_mask] = 0.0

    # Shared time axis
    times = librosa.times_like(onset_env, sr=sr, hop_length=hop)

    # Ensure all arrays have the same length (trim to shortest)
    min_len = min(len(onset_env), len(rms), len(f0), len(pitch_delta), len(times))
    onset_env = onset_env[:min_len]
    rms = rms[:min_len]
    f0 = f0[:min_len]
    pitch_delta = pitch_delta[:min_len]
    times = times[:min_len]

    arrays = {
        "onset_env": onset_env,
        "rms": rms,
        "f0": f0,
        "pitch_delta": pitch_delta,
        "times": times,
    }

    metadata = {
        "video_id": None,  # filled by caller
        "sample_rate": sr,
        "hop_length": hop,
        "duration_sec": round(duration_sec, 3),
        "audio_frame_rate": round(sr / hop, 2),
        "num_frames": min_len,
    }

    return arrays, metadata


def process_video(video_id: str, cfg: dict) -> None:
    require_ffmpeg()
    entry = get_video_entry(cfg, video_id)
    video_path = require_file(entry["path"])

    audio_cfg = cfg["audio"]
    wav_path = Path("data/audio") / f"{video_id}.wav"

    print(f"[{video_id}] Extracting audio...")
    extract_wav(video_path, wav_path, audio_cfg["sample_rate"])
    print(f"[{video_id}] Audio saved to {wav_path}")

    print(f"[{video_id}] Computing audio features...")
    arrays, metadata = compute_audio_features(wav_path, cfg)
    metadata["video_id"] = video_id

    out_path = Path("data/processed") / f"{video_id}_audio_features.npz"
    save_signals(out_path, metadata, **arrays)
    print(
        f"[{video_id}] Audio features saved: {metadata['num_frames']} frames, "
        f"{metadata['duration_sec']}s, {metadata['audio_frame_rate']} fps"
    )


def main():
    parser = base_argparser("Extract audio and compute audio features")
    args = parser.parse_args()
    cfg = load_config(args.config)

    for vid in resolve_video_ids(cfg, args):
        process_video(vid, cfg)


if __name__ == "__main__":
    main()
