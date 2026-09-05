import json
import tempfile
import unittest
from datetime import date

from private_research_assistant.demo import sample_chunks, run_demo
from private_research_assistant.dates import is_active_deadline
from private_research_assistant.evidence import validate_evidence_refs, clear_unsupported_fields
from private_research_assistant.models import Opportunity
from private_research_assistant.structured_extractors import extract_known_source_opportunities, _infer_country
from private_research_assistant.retrieval import citation_errors, RetrievedEvidence
from private_research_assistant.extraction import save_opportunities
from private_research_assistant.paths import latest_output_path, ensure_data_dirs


class ReleaseTests(unittest.TestCase):
    def test_unrelated_evidence_rejected_and_cleared(self):
        row = Opportunity(title="Fully funded PhD", evidence_refs={"title": ["c1"]})
        chunks = {"c1": {"text": "Lunch today"}}
        self.assertFalse(validate_evidence_refs(row, chunks).passing)
        self.assertEqual(clear_unsupported_fields(row, chunks).title, "unknown")

    def test_url_requires_metadata_or_exact_link(self):
        row = Opportunity(application_url="https://example.org/apply", evidence_refs={"application_url": ["c1"]})
        self.assertFalse(validate_evidence_refs(row, {"c1": {"text": "https://example.org/apply-fake"}}).passing)
        self.assertTrue(validate_evidence_refs(row, {"c1": {"text": "[Apply](https://example.org/apply)"}}).passing)

    def test_normalized_excerpt(self):
        row = Opportunity(funding="Full tuition and stipend", evidence_refs={"funding": ["c1"]})
        self.assertTrue(validate_evidence_refs(row, {"c1": {"text": "Full tuition<br> and   stipend"}}).passing)

    def test_split_ad_preserves_field_sources(self):
        chunks = sample_chunks()
        row, = extract_known_source_opportunities("demo", chunks, date(2026, 9, 5))
        self.assertEqual(row.deadline, "30 September 2026")
        self.assertEqual(row.evidence_refs["deadline"], ["demo#chunk-2"])
        self.assertEqual(row.evidence_refs["eligibility"], ["demo#chunk-3"])
        self.assertTrue(validate_evidence_refs(row, {c['evidence_id']: c for c in chunks}).passing)

    def test_country_not_guessed(self):
        self.assertEqual(_infer_country("Newcastle, Australia", "University of Newcastle"), "unknown")

    def test_closed_overrides_rolling(self):
        self.assertFalse(is_active_deadline("Applications closed; previously open until filled"))

    def test_invalid_date_excluded_with_warning(self):
        chunks = sample_chunks()
        chunks[1]['text'] = chunks[1]['text'].replace("30 September 2026", "31 February 2026")
        warnings = []
        self.assertEqual(extract_known_source_opportunities("demo", chunks, date(2026, 9, 5), warnings), [])
        self.assertEqual(warnings[0]['kind'], 'inactive_or_unparseable_deadline')

    def test_missing_and_rolling_still_allowed(self):
        self.assertTrue(is_active_deadline(None))
        self.assertTrue(is_active_deadline("Open until filled"))

    def test_answer_citation_gate(self):
        evidence = [RetrievedEvidence("demo#chunk-1", "title", "https://example.org", "text")]
        self.assertTrue(citation_errors("Unsupported answer", evidence))
        self.assertTrue(citation_errors("Answer [wrong#chunk-1]", evidence))
        self.assertEqual(citation_errors("Answer [demo#chunk-1]", evidence), [])

    def test_every_export_updates_latest_json(self):
        with tempfile.TemporaryDirectory() as directory:
            ensure_data_dirs(directory)
            for fmt in ('json', 'csv', 'markdown'):
                save_opportunities(fmt, [Opportunity(title=fmt)], [], fmt, directory)
                self.assertEqual(json.loads(latest_output_path(directory).read_text())['run_id'], fmt)
