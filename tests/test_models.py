from __future__ import annotations

import unittest

from private_research_assistant.models import Opportunity


class OpportunityModelTests(unittest.TestCase):
    def test_normalizes_missing_values(self) -> None:
        opportunity = Opportunity.from_mapping({"title": "", "country": None})
        self.assertEqual(opportunity.title, "unknown")
        self.assertEqual(opportunity.country, "unknown")

    def test_accepts_list_level_refs_as_fallback(self) -> None:
        opportunity = Opportunity.from_mapping({"title": "A PhD", "evidence_refs": ["chunk-1"]})
        self.assertEqual(opportunity.evidence_refs["title"], ["chunk-1"])


if __name__ == "__main__":
    unittest.main()

