"""Shared synthetic advert chunks for extraction tests."""


def sample_chunks():
    common = {"doc_id": "demo", "source_id": "jobs_ac_uk", "url": "https://example.org/phd"}
    return [
        {**common, "evidence_id": "demo#chunk-1", "text": "# PhD in Machine Learning\n### **Example University**\n| Qualification Type: | PhD |"},
        {**common, "evidence_id": "demo#chunk-2", "text": "| Location: | Germany |\n| Closes: | 30 September 2026 |\n| Funding amount: | Full tuition and a stipend |"},
        {**common, "evidence_id": "demo#chunk-3", "text": "Applicants should hold a computer science degree. The project studies machine learning."},
    ]
