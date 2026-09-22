from __future__ import annotations

import unittest

from private_research_assistant.extraction import _normalize_local_evidence_refs, _parse_json_object


class ExtractionRefTests(unittest.TestCase):
    def test_json_parser_accepts_wrappers_without_separate_fence_removal(self) -> None:
        payload = '{"opportunities": []}'
        for response in (payload, f"```json\n{payload}\n```", f"Answer:\n{payload}",
                         f"<think>{{not JSON}}</think>\n```json\n{payload}\n```"):
            with self.subTest(response=response):
                self.assertEqual(_parse_json_object(response), {"opportunities": []})

    def test_normalizes_short_chunk_refs_for_current_document(self) -> None:
        chunks = {
            "doc-abc#chunk-1": {"text": "First chunk"},
            "doc-abc#chunk-2": {"text": "Second chunk"},
        }
        refs = {"title": ["chunk-1"], "deadline": ["#chunk-2"]}
        self.assertEqual(
            _normalize_local_evidence_refs(refs, "doc-abc", chunks),
            {"title": ["doc-abc#chunk-1"], "deadline": ["doc-abc#chunk-2"]},
        )


if __name__ == "__main__":
    unittest.main()
