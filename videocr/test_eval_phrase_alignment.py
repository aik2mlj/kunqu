#!/usr/bin/env python
"""Unit tests for eval_phrase_alignment. Run: python test_eval_phrase_alignment.py"""

from eval_phrase_alignment import iou, text_sim, match_hybrid


def L(i, t, s, e):
    return {"id": i, "text": t, "startTime": s, "endTime": e}


def test_iou():
    assert abs(iou(L("a", "x", 0, 10), L("b", "y", 0, 10)) - 1.0) < 1e-9
    assert iou(L("a", "x", 0, 10), L("b", "y", 20, 30)) == 0.0
    # [0,10] vs [5,15]: inter 5, union 15 -> 1/3
    assert abs(iou(L("a", "x", 0, 10), L("b", "y", 5, 15)) - (5 / 15)) < 1e-9


def test_greedy_one_to_one():
    gt = [L("g1", "abc", 0, 10)]
    ocr = [L("o1", "abc", 0, 9), L("o2", "abc", 0, 1)]  # both overlap g1
    pairs, um_gt, um_ocr = match_hybrid(gt, ocr, min_iou=0.05)
    assert len(pairs) == 1
    gi, oi, v, ts = pairs[0]
    assert ocr[oi]["id"] == "o1"          # higher IoU wins
    assert um_ocr == [1]                  # o2 spurious
    assert um_gt == []


def test_text_tiebreak():
    # two GT both perfectly overlap one OCR -> text similarity breaks the tie
    ocr = [L("o", "abc", 0, 10)]
    gt = [L("g1", "xyz", 0, 10), L("g2", "abc", 0, 10)]
    pairs, um_gt, um_ocr = match_hybrid(gt, ocr, min_iou=0.5)
    assert len(pairs) == 1
    gi, oi, v, ts = pairs[0]
    assert gt[gi]["id"] == "g2"           # same IoU, better text -> g2
    assert um_gt == [0]                   # g1 left unmatched


def test_min_iou_floor():
    gt = [L("g", "abc", 0, 10)]
    ocr = [L("o", "abc", 9.5, 12)]        # tiny overlap (0.5s), iou ~0.04
    pairs, um_gt, um_ocr = match_hybrid(gt, ocr, min_iou=0.1)
    assert len(pairs) == 0
    assert um_gt == [0] and um_ocr == [0]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nALL {len(tests)} TESTS PASSED")
