#!/usr/bin/env python
"""Unit tests for srt_to_annotation. Run: python test_srt_to_annotation.py"""

from srt_to_annotation import (
    srt_ts_to_sec, normalize_text, majority_merge, merge_runs, parse_srt,
)


def test_timestamp():
    assert abs(srt_ts_to_sec("00:00:58,223") - 58.223) < 1e-6
    assert abs(srt_ts_to_sec("01:02:03,500") - 3723.5) < 1e-6


def test_normalize():
    assert normalize_text("寻来寻去 都不见了") == "寻来寻去都不见了"
    assert normalize_text("a　b c") == "abc"  # full-width + ascii space


def test_majority_vote_basic():
    # user's example: 又(2)/叉(1)->又 ; 之(2)/乏(1)->之
    variants = ["又素之平生半面", "叉素乏平生半面", "又素之平生半面"]
    assert majority_merge(variants) == "又素之平生半面"


def test_majority_vote_5way():
    # 杏(3)/香(2) -> 杏
    variants = ["杏无人迹", "香无人迹", "杏无人迹", "杏无人迹", "香无人迹"]
    assert majority_merge(variants) == "杏无人迹"


def test_majority_vote_length_variants():
    # modal length wins; vote within it
    variants = ["话到其间脑腆", "话到其间面肿", "话到其间脑腆", "话到其间肿", "话到其间脑膜"]
    out = majority_merge(variants)
    assert len(out) == 6 and out.startswith("话到其间")


def test_merge_runs_collapses():
    cues = [
        {"start": 0.0, "end": 4.0, "text": "一径行来", "multiline": False},
        {"start": 10.0, "end": 21.0, "text": "又素之平生半面", "multiline": False},
        {"start": 21.0, "end": 21.6, "text": "叉素乏平生半面", "multiline": False},
        {"start": 21.6, "end": 27.8, "text": "又素之平生半面", "multiline": False},
        {"start": 40.0, "end": 44.0, "text": "园内风物依然", "multiline": False},
    ]
    merged, records = merge_runs(cues, sim=0.6, gap=2.0)
    assert len(merged) == 3            # the 3-cue run collapsed to 1
    assert len(records) == 1
    assert merged[1]["text"] == "又素之平生半面"
    assert merged[1]["start"] == 10.0 and merged[1]["end"] == 27.8


def test_merge_respects_gap():
    # identical text but a big time gap -> NOT merged (different occurrences)
    cues = [
        {"start": 0.0, "end": 2.0, "text": "秀才秀才", "multiline": False},
        {"start": 100.0, "end": 102.0, "text": "秀才秀才", "multiline": False},
    ]
    merged, records = merge_runs(cues, sim=0.6, gap=2.0)
    assert len(merged) == 2 and len(records) == 0


def test_parse_srt(tmp_path_factory=None):
    import tempfile, os
    srt = "1\n00:00:00,023 --> 00:00:00,223\n春香—\n北方昆曲剧院\n\n2\n00:00:58,223 --> 00:01:01,623\n一径行来\n"
    fd, path = tempfile.mkstemp(suffix=".srt")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(srt)
    cues = parse_srt(path)
    os.remove(path)
    assert len(cues) == 2
    assert cues[0]["multiline"] is True
    assert cues[0]["text"] == "春香—北方昆曲剧院"
    assert cues[1]["text"] == "一径行来"
    assert abs(cues[1]["start"] - 58.223) < 1e-6


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nALL {len(tests)} TESTS PASSED")
