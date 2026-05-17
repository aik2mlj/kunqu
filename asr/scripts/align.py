"""Force-align ground-truth libretto text to audio using Qwen3ForcedAligner."""
import json
import sys
import torch
from pathlib import Path
from qwen_asr import Qwen3ForcedAligner

PROJECT_ROOT = Path(__file__).resolve().parent.parent

AUDIO = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT.parent / "data" / "xunmeng.wav"
TEXT_FILE = Path(sys.argv[2]) if len(sys.argv) > 2 else PROJECT_ROOT.parent / "data" / "xunmeng_libretto.txt"
OUTPUT = Path(sys.argv[3]) if len(sys.argv) > 3 else \
    PROJECT_ROOT / "aligned" / f"{AUDIO.stem}.aligned.json"

ALIGNER_MODEL = "/storage/external/lejun/Qwen3-ForcedAligner-0.6B"


def main():
    text = TEXT_FILE.read_text(encoding="utf-8").strip()
    print(f"Aligning {len(text)} characters from {TEXT_FILE.name}")

    model = Qwen3ForcedAligner.from_pretrained(
        ALIGNER_MODEL,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )

    # log VRAM after model load
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"VRAM after model load — allocated: {alloc:.2f} GB, reserved: {reserved:.2f} GB")

    results = model.align(
        audio=str(AUDIO),
        text=text,
        language="Chinese",
    )

    segments = [
        {"text": seg.text, "start": seg.start_time, "end": seg.end_time}
        for seg in results[0]
    ]

    for seg in segments:
        print(f"[{seg['start']:.3f}s – {seg['end']:.3f}s]  {seg['text']}")

    OUTPUT.write_text(
        json.dumps({"text": text, "segments": segments}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved {len(segments)} aligned segments → {OUTPUT}")


if __name__ == "__main__":
    main()
