"""Force-align audio using pinyin transliteration of the ground-truth libretto."""
import json
import sys
import torch
from pathlib import Path
from pypinyin import lazy_pinyin, Style
from qwen_asr import Qwen3ForcedAligner

PROJECT_ROOT = Path(__file__).resolve().parent.parent

AUDIO = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT.parent / "data" / "xunmeng.wav"
TEXT_FILE = Path(sys.argv[2]) if len(sys.argv) > 2 else PROJECT_ROOT.parent / "data" / "xunmeng_libretto.txt"
OUTPUT = Path(sys.argv[3]) if len(sys.argv) > 3 else \
    PROJECT_ROOT / "aligned" / f"{AUDIO.stem}.aligned_pinyin.json"

ALIGNER_MODEL = "/storage/external/lejun/Qwen3-ForcedAligner-0.6B"


def to_pinyin(text: str) -> tuple[str, list[str]]:
    syllables = lazy_pinyin(text, style=Style.NORMAL)
    return " ".join(syllables), syllables


def main():
    hanzi = TEXT_FILE.read_text(encoding="utf-8").strip()
    pinyin_str, syllables = to_pinyin(hanzi)
    print(f"Text ({len(hanzi)} chars) → pinyin ({len(syllables)} syllables)")
    print(f"Preview: {pinyin_str[:80]}...")

    model = Qwen3ForcedAligner.from_pretrained(
        ALIGNER_MODEL,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )

    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"VRAM after model load — allocated: {alloc:.2f} GB, reserved: {reserved:.2f} GB")

    results = model.align(
        audio=str(AUDIO),
        text=pinyin_str,
        language="Chinese",
    )

    segments = []
    for seg, hanzi_char, pinyin_syl in zip(results[0], hanzi, syllables):
        entry = {
            "hanzi": hanzi_char,
            "pinyin": pinyin_syl,
            "start": seg.start_time,
            "end": seg.end_time,
        }
        segments.append(entry)
        print(f"[{seg.start_time:.3f}s – {seg.end_time:.3f}s]  {hanzi_char} ({pinyin_syl})")

    OUTPUT.write_text(
        json.dumps({"pinyin": pinyin_str, "segments": segments}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved {len(segments)} aligned segments → {OUTPUT}")


if __name__ == "__main__":
    main()
