from __future__ import annotations

import unittest

from private_research_assistant.extraction import _normalize_local_evidence_refs


class ExtractionRefTests(unittest.TestCase):
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

