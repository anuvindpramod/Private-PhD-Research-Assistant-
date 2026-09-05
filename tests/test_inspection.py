from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from private_research_assistant.inspection import inspect_chunks, inspect_raw_documents, inspect_status


class InspectionTests(unittest.TestCase):
    def test_status_reports_missing_pipeline_stages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = inspect_status(directory)
        self.assertIn("Crawl: not found", output)
        self.assertIn("Index: not found", output)
        self.assertIn("Extraction: not found", output)

    def test_can_peek_at_raw_documents_and_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_run = root / "data" / "raw" / "run1"
            processed_run = root / "data" / "processed" / "run1"
            chroma = root / "data" / "chroma"
            raw_run.mkdir(parents=True)
            processed_run.mkdir(parents=True)
            chroma.mkdir(parents=True)

            doc = {
                "doc_id": "doc1",
                "source_name": "Example",
                "title": "Fully funded PhD in AI",
                "url": "https://example.edu/phd",
                "markdown": "This fully funded PhD in AI closes 30 Sep 2026.",
            }
            (raw_run / "documents.jsonl").write_text(json.dumps(doc) + "\n", encoding="utf-8")
            raw_manifest = {
                "run_id": "run1",
                "crawl_date": "2026-09-04",
                "document_count": 1,
                "documents_path": str(raw_run / "documents.jsonl"),
            }
            (raw_run / "manifest.json").write_text(json.dumps(raw_manifest), encoding="utf-8")
            (root / "data" / "raw" / "latest_manifest.json").write_text(json.dumps(raw_manifest), encoding="utf-8")

            chunk = {
                "evidence_id": "doc1#chunk-1",
                "doc_id": "doc1",
                "source_name": "Example",
                "title": "Fully funded PhD in AI",
                "url": "https://example.edu/phd",
                "text": "This fully funded PhD in AI closes 30 Sep 2026.",
            }
            (processed_run / "chunks.jsonl").write_text(json.dumps(chunk) + "\n", encoding="utf-8")
            index_manifest = {
                "run_id": "run1",
                "collection_name": "phd_ads_run1",
                "chunk_count": 1,
                "document_count": 1,
                "chunks_path": str(processed_run / "chunks.jsonl"),
            }
            (chroma / "latest_index.json").write_text(json.dumps(index_manifest), encoding="utf-8")

            self.assertIn("Fully funded PhD in AI", inspect_raw_documents(base_dir=directory))
            self.assertIn("doc1#chunk-1", inspect_chunks(base_dir=directory, query="fully funded"))


if __name__ == "__main__":
    unittest.main()

