from __future__ import annotations

import unittest
from datetime import date

from private_research_assistant.structured_extractors import extract_known_source_opportunities


class StructuredExtractorTests(unittest.TestCase):
    def test_extracts_active_jobs_ac_uk_ad(self) -> None:
        chunks = [
            {
                "evidence_id": "job1#chunk-1",
                "doc_id": "job1",
                "source_id": "jobs_ac_uk",
                "url": "https://www.jobs.ac.uk/job/ABC/phd-studentship-ai",
                "text": """
# PhD Studentship in Responsible Artificial Intelligence
### **Example University** - Department of Computer Science
| Qualification Type: | PhD |
| Location: | London |
| Funding for: | UK Students, EU Students, International Students |
| Funding amount: | £21,805<br> tax-free annual living allowance plus 100% home fees covered |
| Closes: | 10th October 2026 |
This project studies responsible artificial intelligence and machine learning.
Applicants should hold a strong degree in computer science or a related subject.
""",
            }
        ]
        opportunities = extract_known_source_opportunities("job1", chunks, date(2026, 9, 4))
        self.assertEqual(len(opportunities), 1)
        opportunity = opportunities[0]
        self.assertEqual(opportunity.title, "PhD Studentship in Responsible Artificial Intelligence")
        self.assertEqual(opportunity.country, "unknown")
        self.assertIn("100% home fees covered", opportunity.funding)
        self.assertEqual(opportunity.evidence_refs["deadline"], ["job1#chunk-1"])

    def test_drops_expired_jobs_ac_uk_ad(self) -> None:
        chunks = [
            {
                "evidence_id": "job1#chunk-1",
                "doc_id": "job1",
                "source_id": "jobs_ac_uk",
                "url": "https://www.jobs.ac.uk/job/ABC/phd-studentship-ai",
                "text": """
# PhD Studentship in Artificial Intelligence
### **Example University**
| Qualification Type: | PhD |
| Location: | London |
| Funding amount: | fully funded |
| Closes: | 31st August 2026 |
""",
            }
        ]
        self.assertEqual(extract_known_source_opportunities("job1", chunks, date(2026, 9, 4)), [])


if __name__ == "__main__":
    unittest.main()
