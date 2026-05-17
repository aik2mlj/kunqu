#!/usr/bin/env python3
"""Extract vocals from audio files using MelBand Roformer.

Models are downloaded to /storage/external/lejun/ by default.

Usage:
  uv run python extract_vocals.py --input song.wav
  uv run python extract_vocals.py --input song.wav --output-dir ./vocals
  uv run python extract_vocals.py --input folder_of_wavs/ --model melband-roformer-kim-vocals
  uv run python extract_vocals.py --input song.wav --device cpu
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml
from ml_collections import ConfigDict

from mel_band_roformer import MODEL_REGISTRY
from mel_band_roformer.download import download_model_assets
from mel_band_roformer.utils import demix_track, get_model_from_config

MODEL_DIR = Path("/storage/external/lejun")
DEFAULT_MODEL = "melband-roformer-kim-vocals"


def ensure_model(model_slug: str, model_dir: Path) -> Path:
    """Download model if not present; return the model subdirectory."""
    model_subdir = model_dir / model_slug
    config_path = model_subdir / MODEL_REGISTRY.get(model_slug).config
    ckpt_path = model_subdir / MODEL_REGISTRY.get(model_slug).checkpoint

    if not config_path.exists() or not ckpt_path.exists():
        print(f"Downloading model '{model_slug}' to {model_dir} ...")
        entry = MODEL_REGISTRY.get(model_slug)
        ok = download_model_assets([entry], model_dir)
        if not ok:
            print(f"Error: failed to download model '{model_slug}'")
            sys.exit(1)

    return model_subdir


def find_wavs(input_path: Path) -> list[Path]:
    """Return list of WAV files from a file or folder."""
    if input_path.is_file():
        if input_path.suffix.lower() != ".wav":
            print(f"Error: input must be a .wav file, got: {input_path}")
            sys.exit(1)
        return [input_path]
    elif input_path.is_dir():
        wavs = sorted(input_path.glob("*.wav"))
        if not wavs:
            print(f"Error: no .wav files found in {input_path}")
            sys.exit(1)
        return wavs
    else:
        print(f"Error: input not found: {input_path}")
        sys.exit(1)


def process_files(
    wav_paths: list[Path],
    model_dir: Path,
    model_slug: str,
    output_dir: Path,
    device: torch.device,
) -> None:
    """Run vocal extraction on all WAV files."""
    entry = MODEL_REGISTRY.get(model_slug)
    config_path = model_dir / entry.config
    ckpt_path = model_dir / entry.checkpoint

    with open(config_path) as f:
        config = ConfigDict(yaml.safe_load(f))

    model = get_model_from_config("mel_band_roformer", config)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model = model.to(device)
    model.eval()

    instruments = config.training.instruments
    if config.training.target_instrument is not None:
        instruments = [config.training.target_instrument]

    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    first_chunk_time = None

    for i, path in enumerate(wav_paths, 1):
        print(f"\n[{i}/{len(wav_paths)}] Processing: {path.name}")

        original_mix, sr = sf.read(path)          # pristine copy for instrumental
        n_channels = 1 if original_mix.ndim == 1 else original_mix.shape[1]

        # Model always expects stereo; duplicate mono channel if needed
        if n_channels == 1:
            mix = np.stack([original_mix, original_mix], axis=-1)
        else:
            mix = original_mix

        mixture = torch.tensor(mix.T, dtype=torch.float32)
        res, first_chunk_time = demix_track(config, model, mixture, device, first_chunk_time)

        # Save per-instrument stems (preserve original channel count)
        for instr in instruments:
            out = res[instr].T          # (samples, channels_from_model)
            if n_channels == 1:
                out = out[:, 0]         # squeeze back to mono
            sf.write(str(output_dir / f"{path.stem}_{instr}.wav"), out, sr, subtype="FLOAT")
            print(f"  -> {path.stem}_{instr}.wav")

        # Instrumental = original mix minus vocals
        vocal_out = res[instruments[0]].T
        if n_channels == 1:
            vocal_out = vocal_out[:, 0]
        instrumental = original_mix - vocal_out
        sf.write(str(output_dir / f"{path.stem}_instrumental.wav"), instrumental, sr, subtype="FLOAT")
        print(f"  -> {path.stem}_instrumental.wav")

    elapsed = time.time() - start_time
    print(f"\nDone. {len(wav_paths)} file(s) processed in {elapsed:.1f}s. Output in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract vocals from audio using MelBand Roformer")
    parser.add_argument("--input", "-i", default=None, help="Input .wav file or folder of .wav files")
    parser.add_argument("--output-dir", "-o", default="./vocals_output", help="Output directory (default: ./vocals_output)")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help=f"Model slug (default: {DEFAULT_MODEL})")
    parser.add_argument("--model-dir", default=str(MODEL_DIR), help=f"Model storage directory (default: {MODEL_DIR})")
    parser.add_argument("--device", default=None, help="Torch device (cpu, cuda:0, etc.). Auto-detected if not set.")
    parser.add_argument("--list-models", action="store_true", help="List available vocal models and exit")
    args = parser.parse_args()

    if args.list_models:
        print("Available vocal models:")
        for m in MODEL_REGISTRY.list("vocals"):
            print(f"  {m.slug:50s}  {m.name}")
        print(f"\nDefault: {DEFAULT_MODEL}")
        print("\nAll categories:", ", ".join(MODEL_REGISTRY.categories()))
        if not args.input:
            return

    if not args.input:
        parser.error("--input/-i is required")

    # Resolve device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        print("CUDA not available, using CPU (slow).")
        device = torch.device("cpu")

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    wavs = find_wavs(input_path)
    print(f"Found {len(wavs)} .wav file(s)")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")

    ensure_model(args.model, model_dir)
    process_files(wavs, model_dir / args.model, args.model, output_dir, device)


if __name__ == "__main__":
    main()
