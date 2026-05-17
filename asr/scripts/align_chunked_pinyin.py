"""Force-align by chunking audio to subtitle lines, using pinyin for alignment."""
import json
import sys
import numpy as np
import soundfile as sf
import torch
from pathlib import Path
from pypinyin import lazy_pinyin, Style
from qwen_asr import Qwen3ForcedAligner

PROJECT_ROOT = Path(__file__).resolve().parent.parent

AUDIO = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT.parent / "data" / "xunmeng.wav"
ANNOTATION = Path(sys.argv[2]) if len(sys.argv) > 2 else \
    PROJECT_ROOT.parent / "statistic-multimodal-analysis/data/annotations/xunmeng_annotation.json"
OUTPUT = Path(sys.argv[3]) if len(sys.argv) > 3 else \
    PROJECT_ROOT / "aligned" / f"{AUDIO.stem}.aligned_chunked_pinyin.json"

ALIGNER_MODEL = "/storage/external/lejun/Qwen3-ForcedAligner-0.6B"
BUFFER = 0.5


def load_mono(path: Path):
    data, sr = sf.read(str(path), always_2d=True)
    if data.shape[1] > 1:
        data = data.mean(axis=1)
    else:
        data = data[:, 0]
    return data.astype(np.float32), sr


def crop(waveform, sr, t_start, t_end):
    s0 = int(t_start * sr)
    s1 = int(t_end * sr)
    return waveform[s0:s1]


def to_pinyin(text: str) -> tuple[str, list[str]]:
    syllables = lazy_pinyin(text, style=Style.NORMAL)
    return " ".join(syllables), syllables


def main():
    waveform, sr = load_mono(AUDIO)
    total_dur = len(waveform) / sr
    print(f"Audio: {total_dur:.1f}s at {sr}Hz")

    data = json.loads(ANNOTATION.read_text())
    lines = data["project"]["subtitleLines"]

    char_map: dict[str, list[str]] = {}
    for c in data["project"]["characterAnnotations"]:
        char_map.setdefault(c["lineId"], []).append(c["char"])

    model = Qwen3ForcedAligner.from_pretrained(
        ALIGNER_MODEL,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )

    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"VRAM — allocated: {alloc:.2f} GB, reserved: {reserved:.2f} GB")

    all_segments = []
    n_failed = 0

    for line in lines:
        lid = line["id"]
        text = line["text"]
        chars = char_map.get(lid, list(text))

        pinyin_str, syllables = to_pinyin(text)
        # Guard: character list should match syllable count
        if len(chars) != len(syllables):
            print(f"[WARN] {lid}: {len(chars)} chars vs {len(syllables)} pinyin syllables — falling back to hanzi")
            align_text = text
            align_tokens = chars
        else:
            align_text = pinyin_str
            align_tokens = chars

        t0 = max(0.0, line["startTime"] - BUFFER)
        t1 = min(total_dur, line["endTime"] + BUFFER)
        chunk = crop(waveform, sr, t0, t1)

        try:
            segs = model.align(audio=(chunk, sr), text=align_text, language="Chinese")[0]

            if len(segs) != len(align_tokens):
                print(f"[WARN] {lid}: {len(segs)} aligned segs vs {len(align_tokens)} tokens — using hanzi fallback")
                segs = model.align(audio=(chunk, sr), text=text, language="Chinese")[0]

            for seg, char in zip(segs, chars):
                all_segments.append({
                    "text": char,
                    "start": round(seg.start_time + t0, 4),
                    "end":   round(seg.end_time   + t0, 4),
                    "line_id": lid,
                })
            print(f"[OK]  {lid}: '{text}' → {len(segs)} segs (pinyin)")

        except Exception as e:
            n_failed += 1
            print(f"[ERR] {lid}: '{text}' — {e}")
            dur = (line["endTime"] - line["startTime"]) / max(len(chars), 1)
            for i, char in enumerate(chars):
                all_segments.append({
                    "text": char,
                    "start": round(line["startTime"] + i * dur, 4),
                    "end":   round(line["startTime"] + (i + 1) * dur, 4),
                    "line_id": lid,
                })

    print(f"\n{len(all_segments)} segments, {n_failed} failed lines")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            {"text": "".join(s["text"] for s in all_segments), "segments": all_segments},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved → {OUTPUT}")


if __name__ == "__main__":
    main()
