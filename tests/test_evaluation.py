from __future__ import annotations

import unittest

from private_research_assistant.evaluation import run_evaluation


class FixtureEvaluationTests(unittest.TestCase):
    def test_fixture_evaluation_passes_all_ten_questions(self) -> None:
        result = run_evaluation()
        self.assertTrue(result.passing)
        self.assertEqual(len(result.fixture_results), 10)
        self.assertEqual(sum(1 for case in result.fixture_results if case.passing), 10)


if __name__ == "__main__":
    unittest.main()

