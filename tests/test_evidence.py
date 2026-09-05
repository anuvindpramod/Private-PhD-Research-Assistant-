from __future__ import annotations

import unittest

from private_research_assistant.evidence import collect_cited_refs, validate_evidence_refs
from private_research_assistant.models import Opportunity


class EvidenceValidationTests(unittest.TestCase):
    def test_requires_evidence_for_filled_fields(self) -> None:
        opportunity = Opportunity(title="A PhD", university_or_lab="A Lab", evidence_refs={"title": ["c1"]})
        result = validate_evidence_refs(opportunity, {"c1": {"text": "A PhD"}})
        self.assertFalse(result.passing)
        self.assertIn("university_or_lab is filled but has no evidence_refs entry", result.errors)

    def test_unknown_fields_do_not_need_evidence(self) -> None:
        opportunity = Opportunity(title="A PhD", evidence_refs={"title": ["c1"]})
        result = validate_evidence_refs(opportunity, {"c1": {"text": "A PhD"}})
        self.assertTrue(result.passing)

    def test_collects_unique_refs(self) -> None:
        opportunity = Opportunity(
            title="A PhD",
            deadline="30 Sep 2026",
            evidence_refs={"title": ["c1"], "deadline": ["c1", "c2"]},
        )
        self.assertEqual(collect_cited_refs(opportunity), ["c1", "c2"])


if __name__ == "__main__":
    unittest.main()

