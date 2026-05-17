"""Extract ordered libretto text from xunmeng annotation JSON to a plain txt file."""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ANNOTATION = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    PROJECT_ROOT.parent / "statistic-multimodal-analysis/data/annotations/xunmeng_annotation.json"
OUTPUT = Path(sys.argv[2]) if len(sys.argv) > 2 else PROJECT_ROOT.parent / "data" / "xunmeng_libretto.txt"


def main():
    data = json.loads(ANNOTATION.read_text())
    lines = data["project"]["subtitleLines"]
    text = "".join(l["text"] for l in lines)
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"Extracted {len(text)} characters → {OUTPUT}")


if __name__ == "__main__":
    main()
