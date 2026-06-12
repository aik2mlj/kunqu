"""
Segment a long audio file into phrase-level clips for SOFA forced alignment.

Reads a JSON annotation file (with subtitleLines and characterAnnotations),
splits the source .wav into per-phrase .wav segments, and generates
matching .lab files with pinyin transcriptions.

Usage:
    python data/prepare_segments.py \
        --annotation data/xunmeng/xunmeng_annotation.json \
        --wav data/xunmeng/xunmeng.wav \
        --output segments/xunmeng \
        --dictionary dictionary/opencpop-extension.txt
"""

import argparse
import json
import os
import re

import soundfile as sf
from pypinyin import lazy_pinyin, Style


def load_dictionary(dict_path):
    """Load the SOFA dictionary and return the set of valid pinyin keys."""
    valid = set()
    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                key = line.split("\t")[0].strip()
                valid.add(key)
    return valid


def chinese_to_pinyin(text, valid_pinyin):
    """
    Convert Chinese text to space-separated pinyin compatible with the
    opencpop-extension dictionary.

    Returns (pinyin_string, warnings_list).
    """
    pinyins = lazy_pinyin(text, style=Style.NORMAL)
    warnings = []
    result = []
    for i, py in enumerate(pinyins):
        py = py.strip().lower()
        if not py:
            continue
        # pypinyin may return the original char for punctuation / unknown
        if re.match(r'^[\u4e00-\u9fff]$', py):
            warnings.append(f"Could not convert character: {py}")
            continue
        # Skip pure punctuation
        if re.match(r'^[^\w]+$', py):
            continue
        if py not in valid_pinyin:
            # Try common mappings: pypinyin uses 'lv' for 'lü', dict uses 'lv'
            # Most should already match, but flag if not
            warnings.append(f"Pinyin '{py}' (from '{text[i] if i < len(text) else '?'}') not in dictionary")
        result.append(py)
    return " ".join(result), warnings


def read_wav_params(wav_path):
    """Read audio file metadata without loading all the samples."""
    info = sf.info(wav_path)
    return {
        "nchannels": info.channels,
        "framerate": info.samplerate,
        "nframes": info.frames,
        "subtype": info.subtype,
    }


def extract_wav_segment(src_path, dst_path, start_sec, end_sec):
    """
    Extract a time segment from an audio file and write it as 16-bit PCM mono.

    Uses soundfile (libsndfile), so it transparently handles any source
    encoding (16/24/32-bit PCM or 32-bit float) and any channel count, reading
    only the requested frame range. Multi-channel input is downmixed to mono by
    averaging channels. Writing 16-bit PCM mono yields the format the
    downstream SOFA reader (torchaudio) decodes most reliably.
    """
    sr = sf.info(src_path).samplerate
    start_frame = int(start_sec * sr)
    stop_frame = int(end_sec * sr)

    data, sr = sf.read(
        src_path,
        start=start_frame,
        stop=stop_frame,
        dtype="float32",
        always_2d=True,
    )
    mono = data.mean(axis=1)
    sf.write(dst_path, mono, sr, subtype="PCM_16")


def main():
    parser = argparse.ArgumentParser(description="Segment audio for SOFA alignment")
    parser.add_argument("--annotation", "-a", required=True, help="Path to JSON annotation file")
    parser.add_argument("--wav", "-w", required=True, help="Path to source .wav file")
    parser.add_argument("--output", "-o", required=True, help="Output directory for segments")
    parser.add_argument("--dictionary", "-d", default="dictionary/opencpop-extension.txt",
                        help="Path to SOFA dictionary file")
    parser.add_argument("--max-duration", type=float, default=45.0,
                        help="Warn if a segment exceeds this duration in seconds (default: 45)")
    parser.add_argument("--skip-long", action="store_true",
                        help="Skip subtitle lines whose duration exceeds --max-duration")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip segment if both output .wav and .lab already exist")
    args = parser.parse_args()

    with open(args.annotation, "r", encoding="utf-8") as f:
        data = json.load(f)

    lines = data["project"]["subtitleLines"]
    valid_pinyin = load_dictionary(args.dictionary)
    wav_params = read_wav_params(args.wav)
    total_duration = wav_params["nframes"] / wav_params["framerate"]

    print(f"Source: {args.wav}")
    print(f"  Duration: {total_duration:.1f}s, Sample rate: {wav_params['framerate']}Hz, "
          f"Channels: {wav_params['nchannels']}, Format: {wav_params['subtype']}")
    print(f"Subtitle lines: {len(lines)}")
    print(f"Dictionary entries: {len(valid_pinyin)}")
    print()

    os.makedirs(args.output, exist_ok=True)

    all_warnings = []
    skipped = []
    skipped_existing = 0
    skipped_long = 0
    created = 0

    for i, line in enumerate(lines):
        line_id = line["id"]
        text = line["text"]
        start = line["startTime"]
        end = line["endTime"]
        duration = end - start

        if args.skip_long and duration > args.max_duration:
            print(
                f"  [{i+1:3d}/{len(lines)}] {line_id}: duration {duration:.1f}s "
                f"> {args.max_duration}s, skipping"
            )
            skipped_long += 1
            continue

        # Convert to pinyin
        pinyin_str, warnings = chinese_to_pinyin(text, valid_pinyin)

        if not pinyin_str.strip():
            skipped.append((line_id, text, "no valid pinyin produced"))
            continue

        seg_start = max(0.0, start)
        seg_end = min(total_duration, end)

        safe_name = f"{i+1:03d}_{line_id}_t{int(round(start * 1000))}"
        wav_out = os.path.join(args.output, f"{safe_name}.wav")
        lab_out = os.path.join(args.output, f"{safe_name}.lab")

        if args.skip_existing and os.path.exists(wav_out) and os.path.exists(lab_out):
            print(f"  [{i+1:3d}/{len(lines)}] {safe_name}: already exists, skipping")
            skipped_existing += 1
            continue

        # Extract audio segment
        extract_wav_segment(args.wav, wav_out, seg_start, seg_end)

        # Write lab file
        with open(lab_out, "w", encoding="utf-8") as f:
            f.write(pinyin_str)

        status = ""
        if duration > args.max_duration:
            status = f" [WARNING: {duration:.1f}s > {args.max_duration}s]"

        if warnings:
            for w in warnings:
                all_warnings.append(f"  {line_id} ({text}): {w}")

        print(f"  [{i+1:3d}/{len(lines)}] {safe_name}: \"{text}\" -> \"{pinyin_str}\" "
              f"({duration:.1f}s){status}")
        created += 1

    print(f"\nCreated {created} segments in {args.output}/")
    if skipped_existing:
        print(f"Skipped existing pairs: {skipped_existing}")
    if skipped_long:
        print(f"Skipped long segments: {skipped_long} (>{args.max_duration}s)")

    if skipped:
        print(f"\nSkipped {len(skipped)} lines:")
        for lid, txt, reason in skipped:
            print(f"  {lid}: \"{txt}\" ({reason})")

    if all_warnings:
        print(f"\nWarnings ({len(all_warnings)}):")
        for w in all_warnings:
            print(w)

    print(f"\nTo run SOFA inference:")
    print(f"  python infer.py --ckpt ckpt/pretrained_mandarin_singing/v1.0.0_mandarin_singing.ckpt "
          f"--folder {args.output}")


if __name__ == "__main__":
    main()
