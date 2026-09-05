"""Small synthetic input, passed through the real extraction and evidence checks."""
from datetime import date
import json

from .structured_extractors import extract_known_source_opportunities
from .evidence import validate_evidence_refs


def sample_chunks():
    common = {"doc_id": "demo", "source_id": "jobs_ac_uk", "url": "https://example.org/phd"}
    return [
        {**common, "evidence_id": "demo#chunk-1", "text": "# PhD in Machine Learning\n### **Example University**\n| Qualification Type: | PhD |"},
        {**common, "evidence_id": "demo#chunk-2", "text": "| Location: | Germany |\n| Closes: | 30 September 2026 |\n| Funding amount: | Full tuition and a stipend |"},
        {**common, "evidence_id": "demo#chunk-3", "text": "Applicants should hold a computer science degree. The project studies machine learning."},
    ]


def run_demo():
    chunks = sample_chunks()
    rows = extract_known_source_opportunities("demo", chunks, date(2026, 9, 5))
    print("Synthetic offline demo, as of 2026-09-05. No real vacancy or model call.")
    if len(rows) != 1 or not validate_evidence_refs(rows[0], {c['evidence_id']: c for c in chunks}).passing:
        print("FAIL: expected one textually supported sample opportunity")
        return 1
    print(json.dumps(rows[0].to_dict(), indent=2))
    print("PASS: actual parser and textual support validator executed.")
    return 0
