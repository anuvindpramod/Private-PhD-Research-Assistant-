import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from private_research_assistant.cli import main
from private_research_assistant.demo import sample_chunks
from private_research_assistant import UserFacingError
from private_research_assistant.extraction import extract_opportunities
from private_research_assistant.crawl import FirecrawlClient
from private_research_assistant.indexing import _assign_evidence_ids
from private_research_assistant.retrieval import _to_evidence


class ExtractionFlowTests(unittest.TestCase):
    def setUp(self):
        self.chunks = {chunk["evidence_id"]: chunk for chunk in sample_chunks()}
        self.manifest = {"run_id": "test", "crawl_date": "2026-09-05"}
        self.llm = Mock()
        for target, value in (
            ("load_index_manifest", self.manifest),
            ("load_chunks", self.chunks),
            ("save_opportunities", {}),
            ("create_llm", self.llm),
        ):
            replacement = patch(f"private_research_assistant.extraction.{target}", return_value=value)
            replacement.start()
            self.addCleanup(replacement.stop)

    def use_unsupported_source(self):
        for chunk in self.chunks.values():
            chunk["source_id"] = "findaphd"

    def test_default_parser_and_fallback_skip(self):
        result = extract_opportunities(llm_fallback=True)
        self.assertEqual(len(result.opportunities), 1)
        self.llm.complete.assert_not_called()

    def test_parser_warning_blocks_fallback(self):
        self.chunks["demo#chunk-2"]["text"] = "| Closes: | 31 February 2026 |"
        result = extract_opportunities(llm_fallback=True)
        self.assertEqual(result.opportunities, [])
        self.assertEqual(result.warnings[0]["kind"], "inactive_or_unparseable_deadline")
        self.llm.complete.assert_not_called()

    def test_unknown_source_skipped_without_llm(self):
        self.use_unsupported_source()
        result = extract_opportunities()
        self.assertEqual(result.warnings[0]["kind"], "skipped_no_structured_parser")
        self.llm.complete.assert_not_called()

    def test_malformed_model_output_produces_warning(self):
        self.use_unsupported_source()
        for response in ("not JSON", "{broken}", "{}", '{"opportunities": null}', '{"opportunities": {}}'):
            with self.subTest(response=response):
                self.llm.complete.return_value = response
                result = extract_opportunities(llm_fallback=True)
                self.assertEqual(result.opportunities, [])
                self.assertEqual(result.warnings[0]["kind"], "invalid_json")

    def test_llm_result_clears_unsupported_fields_and_keeps_citations(self):
        self.use_unsupported_source()
        self.llm.complete.return_value = json.dumps({"opportunities": [{
            "title": "PhD in Machine Learning",
            "funding": "Invented funding",
            "evidence_refs": {"title": ["chunk-1"], "funding": ["chunk-2"]},
        }]})
        result = extract_opportunities(llm_fallback=True)
        row, = result.opportunities
        self.assertEqual(row.title, "PhD in Machine Learning")
        self.assertEqual(row.funding, "unknown")
        self.assertEqual(row.evidence_refs, {"title": ["demo#chunk-1"]})

    def test_model_runtime_failure_is_not_swallowed(self):
        self.use_unsupported_source()
        self.llm.complete.side_effect = RuntimeError("model unavailable")
        with self.assertRaisesRegex(RuntimeError, "model unavailable"):
            extract_opportunities(llm_fallback=True)

    def test_invalid_crawl_date_does_not_fall_back_to_today(self):
        self.manifest["crawl_date"] = "invalid"
        with self.assertRaises(ValueError):
            extract_opportunities()
        del self.manifest["crawl_date"]
        with self.assertRaises(KeyError):
            extract_opportunities()


class LibraryFailureTests(unittest.TestCase):
    def test_text_read_errors_are_not_replaced_with_empty_text(self):
        node = Mock(metadata={"doc_id": "demo", "evidence_id": "demo#chunk-1"})
        node.get_content.side_effect = RuntimeError("text read failed")
        with self.assertRaisesRegex(RuntimeError, "text read failed"):
            _assign_evidence_ids([node])
        with self.assertRaisesRegex(RuntimeError, "text read failed"):
            _to_evidence(SimpleNamespace(node=node, score=0.5))

    def test_cli_does_not_hide_programming_errors(self):
        with patch("private_research_assistant.cli.build_index", side_effect=TypeError("bad argument")):
            with self.assertRaisesRegex(TypeError, "bad argument"):
                main(["index"])


class FirecrawlResponseTests(unittest.TestCase):
    def test_missing_key_fails_without_falling_back_to_another_project(self):
        with self.assertRaises(UserFacingError):
            FirecrawlClient(api_key=None)

    def test_search_returns_web_results_and_requests_markdown(self):
        client = FirecrawlClient(api_key="test-only")
        page = {"url": "https://example.org", "markdown": "Advert"}
        with patch.object(client, "_post", return_value={"data": {"web": [page]}}) as post:
            self.assertEqual(client.search("PhD", limit=1), [page])
            self.assertEqual(post.call_args.args[1]["scrapeOptions"], {"formats": ["markdown"]})

    def test_unexpected_results_are_not_treated_as_empty_search(self):
        client = FirecrawlClient(api_key="test-only")
        for data in (None, "wrong", {"web": None}, {"web": "wrong"}):
            with self.subTest(data=data), patch.object(client, "_post", return_value={"data": data}):
                with self.assertRaises(UserFacingError):
                    client.search("PhD", limit=1)
