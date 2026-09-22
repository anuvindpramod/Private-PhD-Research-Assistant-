from __future__ import annotations

import unittest

from private_research_assistant.sources import CURATED_SOURCES, get_sources


class SourceConfigTests(unittest.TestCase):
    def test_has_six_curated_sources(self) -> None:
        self.assertEqual(
            {source.source_id for source in CURATED_SOURCES},
            {"jobs_ac_uk", "findaphd", "euraxess", "academic_positions", "ellis_jobs", "phdscanner"},
        )

    def test_source_ids_are_unique(self) -> None:
        ids = [source.source_id for source in CURATED_SOURCES]
        self.assertEqual(len(ids), len(set(ids)))

    def test_can_select_source_by_id(self) -> None:
        sources = get_sources(["euraxess"])
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].source_id, "euraxess")


if __name__ == "__main__":
    unittest.main()
