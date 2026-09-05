from __future__ import annotations

import unittest

from private_research_assistant.retrieval import RetrievedEvidence, _build_answer_prompt


class RetrievalPromptTests(unittest.TestCase):
    def test_prompt_includes_freshness_rule(self) -> None:
        prompt = _build_answer_prompt(
            "Which opportunities are active?",
            [RetrievedEvidence("chunk-1", "Example", "https://example.edu", "Closes 31 August 2026.")],
            crawl_date="2026-09-04",
        )
        self.assertIn("Crawl date / freshness date: 2026-09-04", prompt)
        self.assertIn("deadline before the crawl date as expired", prompt)


if __name__ == "__main__":
    unittest.main()

