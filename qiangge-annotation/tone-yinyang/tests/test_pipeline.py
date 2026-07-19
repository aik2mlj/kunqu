from __future__ import annotations

import unittest

from yxlz_tone_annotator.pipeline import (
    _build_same_glyph_conflicts,
    build_batches,
    build_line_payloads,
    validate_batch_result,
)


class TonePipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.document = {
            "version": 5,
            "project": {
                "subtitleLines": [
                    {"id": "line-1", "text": "哭？转", "startTime": 0, "endTime": 2},
                ],
                "characterAnnotations": [
                    {
                        "id": "char-1",
                        "lineId": "line-1",
                        "char": "哭？",
                        "startTime": 0,
                        "endTime": 1,
                        "tone": None,
                    },
                    {
                        "id": "char-2",
                        "lineId": "line-1",
                        "char": "转",
                        "startTime": 1,
                        "endTime": 2,
                        "tone": None,
                    },
                ],
            },
        }

    def test_build_payload_preserves_block_text_and_extracts_lookup_character(self) -> None:
        lines, warnings = build_line_payloads(self.document, [], overwrite_existing=False)

        self.assertEqual(len(lines), 1)
        first = lines[0]["characters"][0]
        self.assertEqual(first["char"], "哭？")
        self.assertEqual(first["lookupChar"], "哭")
        self.assertFalse(first["referenceAvailable"])
        self.assertTrue(any("含标点" in warning for warning in warnings))

    def test_no_reference_forces_review_and_caps_confidence(self) -> None:
        lines, _ = build_line_payloads(self.document, [], overwrite_existing=False)
        batch = build_batches(lines, max_lines=3, max_characters=24)[0]
        result = {
            "batchId": batch["batchId"],
            "annotations": [
                {
                    "id": "char-1",
                    "char": "哭？",
                    "lookupChar": "哭",
                    "toneClass": "yin_ru",
                    "yxlzShangSubtype": None,
                    "confidence": 0.98,
                    "needsReview": False,
                    "basis": {"source": "韵学骊珠", "explanation": "模型自行声称"},
                    "alternatives": [],
                },
                {
                    "id": "char-2",
                    "char": "转",
                    "lookupChar": "转",
                    "toneClass": "yang_shang",
                    "yxlzShangSubtype": "yang_shang",
                    "confidence": 0.95,
                    "needsReview": False,
                    "basis": {"source": "韵学骊珠", "explanation": "模型自行声称"},
                    "alternatives": [],
                },
            ],
        }

        validated = validate_batch_result(batch, result, confidence_threshold=0.8)

        self.assertTrue(all(item["needsReview"] for item in validated))
        self.assertTrue(all(item["confidence"] == 0.79 for item in validated))
        self.assertTrue(
            all(item["basis"]["source"] == "context_inference" for item in validated)
        )

    def test_same_glyph_conflicts_are_reported(self) -> None:
        conflicts = _build_same_glyph_conflicts(
            [
                {
                    "id": "a",
                    "char": "转",
                    "toneClass": "yin_shang",
                    "yxlzShangSubtype": "yin_shang",
                    "confidence": 0.7,
                },
                {
                    "id": "b",
                    "char": "转",
                    "toneClass": "yang_shang",
                    "yxlzShangSubtype": "yang_shang",
                    "confidence": 0.7,
                },
            ]
        )

        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["char"], "转")
        self.assertEqual(len(conflicts[0]["toneOptions"]), 2)


if __name__ == "__main__":
    unittest.main()

