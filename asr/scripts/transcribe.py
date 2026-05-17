import json
import torch
from pathlib import Path
from qwen_asr import Qwen3ASRModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent

AUDIO = PROJECT_ROOT.parent / "data" / "xunmeng.wav"
ASR_MODEL = "/storage/external/lejun/Qwen3-ASR-1.7B"
ALIGNER_MODEL = "/storage/external/lejun/Qwen3-ForcedAligner-0.6B"


def main():
    model = Qwen3ASRModel.from_pretrained(
        ASR_MODEL,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_new_tokens=4096,
        forced_aligner=ALIGNER_MODEL,
        forced_aligner_kwargs=dict(dtype=torch.bfloat16, device_map="cuda:0"),
    )

    results = model.transcribe(
        audio=[str(AUDIO)],
        language=["Chinese"],
        return_time_stamps=True,
    )

    r = results[0]
    print(f"Language: {r.language}")
    print(f"Full text: {r.text}\n")

    segments = []
    for ts in r.time_stamps:
        seg = {"text": ts.text, "start": ts.start_time, "end": ts.end_time}
        segments.append(seg)
        print(f"[{ts.start_time:.3f}s – {ts.end_time:.3f}s]  {ts.text}")

    out = AUDIO.with_suffix(".json")
    out.write_text(json.dumps({"text": r.text, "segments": segments}, ensure_ascii=False, indent=2))
    print(f"\nSaved timestamped output to {out}")


if __name__ == "__main__":
    main()
