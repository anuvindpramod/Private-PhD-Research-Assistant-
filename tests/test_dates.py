from __future__ import annotations

import unittest
from datetime import date

from private_research_assistant.dates import is_active_deadline, normalize_deadline, parse_deadline_text


class DeadlineTests(unittest.TestCase):
    def test_parses_common_deadline_formats(self) -> None:
        reference = date(2026, 9, 3)
        self.assertEqual(parse_deadline_text("30 Sep 2026 - 00:00", reference), date(2026, 9, 30))
        self.assertEqual(parse_deadline_text("Closes 07 Sep", reference), date(2026, 9, 7))
        self.assertEqual(parse_deadline_text("Closing on: 2026-11-30", reference), date(2026, 11, 30))
        self.assertEqual(parse_deadline_text("Apply by 01/10/26", reference), date(2026, 10, 1))

    def test_active_deadline_rule(self) -> None:
        reference = date(2026, 9, 3)
        self.assertTrue(is_active_deadline("30 Sep 2026", reference))
        self.assertTrue(is_active_deadline("Year round applications", reference))
        self.assertTrue(is_active_deadline("unknown", reference))
        self.assertFalse(is_active_deadline("02 Sep 2026", reference))

    def test_normalizes_parseable_deadline(self) -> None:
        self.assertEqual(normalize_deadline("30 Sep 2026", date(2026, 9, 3)), "2026-09-30")


if __name__ == "__main__":
    unittest.main()

