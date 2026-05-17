"""Convert character-level timestamp JSON to per-character SRT subtitles."""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    PROJECT_ROOT / "aligned" / "xunmeng.aligned_chunked.json"
OUTPUT = Path(sys.argv[2]) if len(sys.argv) > 2 else INPUT.with_suffix(".srt")

MIN_DURATION = 0.5


def format_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main():
    segments = json.loads(INPUT.read_text())["segments"]
    lines = []
    for i, seg in enumerate(segments, 1):
        start = seg["start"]
        end = max(seg["end"], start + MIN_DURATION)
        # don't overlap with next
        if i < len(segments):
            next_start = segments[i]["start"]
            if end > next_start:
                end = max(start + 0.1, next_start - 0.01)
        lines.append(f"{i}")
        lines.append(f"{format_ts(start)} --> {format_ts(end)}")
        lines.append(seg["text"])
        lines.append("")
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written {len(segments)} characters to {OUTPUT}")


if __name__ == "__main__":
    main()
