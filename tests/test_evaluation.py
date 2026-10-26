from __future__ import annotations

import json
import unittest
from pathlib import Path

from private_research_assistant.models import Opportunity


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "src/private_research_assistant/fixtures"


class FixtureEvaluationTests(unittest.TestCase):
    def test_fixture_evaluation_passes_all_ten_questions(self) -> None:
        questions = json.loads((FIXTURE_DIR / "eval_questions.json").read_text(encoding="utf-8"))
        rows = json.loads((FIXTURE_DIR / "fixture_opportunities.json").read_text(encoding="utf-8"))
        opportunities = {row["title"]: Opportunity.from_mapping(row) for row in rows}
        chunks = {}
        for line in (FIXTURE_DIR / "fixture_chunks.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                chunk = json.loads(line)
                chunks[chunk["evidence_id"]] = chunk

        self.assertEqual(len(questions), 10)
        for question in questions:
            with self.subTest(id=question["id"], question=question["question"]):
                self.assertIn(question["opportunity_title"], opportunities)
                opportunity = opportunities[question["opportunity_title"]]
                field = question["field"]
                refs = opportunity.evidence_refs.get(field, [])
                evidence_text = " ".join(chunks[ref]["text"] for ref in refs if ref in chunks)
                self.assertEqual(getattr(opportunity, field), question["expected_value"])
                self.assertIn(question["expected_ref"], refs)
                self.assertIn(question["evidence_must_contain"], evidence_text)


if __name__ == "__main__":
    unittest.main()

