from __future__ import annotations

import re
from datetime import date
from typing import Any

from .dates import is_active_deadline
from .models import UNKNOWN, Opportunity
from .evidence import clear_unsupported_fields, normalize_excerpt


AI_TERMS = (
    "agentic",
    "artificial intelligence",
    "medical data science"
    "ai in medicine"
    "machine learning",
    "deep learning",
    "reinforcement learning",
    "generative",
    "computer vision",
    "nlp",
    "robotics",
    "trustworthy ai",
    "explainable ai",
    "xai",
    "health ai"
)

def extract_known_source_opportunities(
    doc_id: str,
    chunks: list[dict[str, Any]],
    crawl_date: date,
    warnings: list[dict[str, Any]] | None = None,
) -> list[Opportunity]:
    if not chunks:
        return []
    source_id = str(chunks[0].get("source_id", ""))
    if source_id == "jobs_ac_uk":
        return _extract_jobs_ac_uk_ad(doc_id, chunks, crawl_date, warnings)
    return []


def _extract_jobs_ac_uk_ad(
    doc_id: str,
    chunks: list[dict[str, Any]],
    crawl_date: date,
    warnings: list[dict[str, Any]] | None = None,
) -> list[Opportunity]:
    first_chunk = chunks[0]
    first_text = "\n".join(str(chunk.get("text", "")) for chunk in chunks)
    first_ref = str(first_chunk.get("evidence_id", f"{doc_id}#chunk-1"))

    if "The job you're looking for is no longer being advertised" in first_text:
        return []
    if "| Qualification Type: | PhD |" not in first_text:
        return []

    title = _first_match(first_text, r"(?m)^#\s+(.+?)\s*$")
    institution = _first_match(first_text, r"(?m)^###\s+\*\*(.+?)\*\*")
    location = _table_value(first_text, "Location")
    funding_amount = _table_value(first_text, "Funding amount")
    deadline = _table_value(first_text, "Closes") or _table_value(first_text, "Expires")
    url = str(first_chunk.get("url") or UNKNOWN)

    if not title or "PhD" not in title:
        return []
    if not _is_ai_related(" ".join(chunk.get("text", "") for chunk in chunks)):
        return []
    if not is_active_deadline(deadline, crawl_date):
        if warnings is not None:
            warnings.append({"doc_id": doc_id, "kind": "inactive_or_unparseable_deadline", "deadline": deadline})
        return []

    funding = funding_amount or UNKNOWN
    eligibility, eligibility_ref = _find_sentence(
        chunks,
        (
            "applicants should",
            "candidate should",
            "candidates should",
            "you should",
            "you must have",
            "eligibility",
            "requirements",
        ),
    )
    topic_fit, topic_ref = _find_sentence(chunks, AI_TERMS)

    refs: dict[str, list[str]] = {
        "title": [first_ref],
        "university_or_lab": [first_ref],
        "country": [first_ref],
        "deadline": [first_ref],
        "funding": [first_ref],
        "application_url": [first_ref],
    }
    if eligibility != UNKNOWN:
        refs["eligibility"] = [eligibility_ref]
    if topic_fit != UNKNOWN:
        refs["topic_fit"] = [topic_ref]

    opportunity = Opportunity(
            title=_clean(title),
            university_or_lab=_clean(institution or UNKNOWN),
            country=_infer_country(location, institution),
            deadline=_clean(deadline or UNKNOWN),
            funding=_clean(funding or UNKNOWN),
            eligibility=_clean(eligibility),
            topic_fit=_clean(topic_fit),
            application_url=url,
            evidence_refs=refs,
        )
    for name in refs:
        value = getattr(opportunity, name)
        refs[name] = [str(chunk["evidence_id"]) for chunk in chunks if
                      (value == chunk.get("url") if name == "application_url" else
                       normalize_excerpt(value) in normalize_excerpt(str(chunk.get("text", ""))))]
    opportunity = clear_unsupported_fields(opportunity, {c["evidence_id"]: c for c in chunks})
    return [opportunity] if opportunity.title != UNKNOWN else []


def _table_value(text: str, label: str) -> str | None:
    pattern = rf"\|\s*{re.escape(label)}:\s*\|\s*(.*?)\s*\|"
    return _first_match(text, pattern)


def _first_match(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return _clean(match.group(1))


def _find_sentence(chunks: list[dict[str, Any]], needles: tuple[str, ...]) -> tuple[str, str]:
    for chunk in chunks:
        text = str(chunk.get("text", ""))
        normalized_text = _clean(text)
        lower_text = normalized_text.lower()
        for needle in needles:
            index = lower_text.find(needle.lower())
            if index == -1:
                continue
            boundary = lower_text.rfind(". ", 0, index)
            start = boundary + 2 if boundary >= 0 else 0
            end = lower_text.find(". ", index)
            if end == -1:
                end = min(len(normalized_text), index + 320)
            if end - start > 400:
                start = index
                end = min(end, index + 319)
            sentence = normalized_text[start : end + 1].strip()
            if sentence:
                return sentence, str(chunk.get("evidence_id", ""))
    return UNKNOWN, ""


def _is_ai_related(text: str) -> bool:
    lower = text.lower()
    return any(term in lower for term in AI_TERMS)


def _infer_country(location: str | None, institution: str | None) -> str:
    value = f"{location or ''} {institution or ''}".lower()
    for country in ("denmark", "germany", "sweden", "switzerland", "france", "spain", "netherlands", "norway","italy"):
        if country in value:
            return country.title()
    if "united kingdom" in value:
        return "United Kingdom"
    return UNKNOWN


def _join_known(values: list[str | None]) -> str:
    return "; ".join(_clean(value) for value in values if value and _clean(value) != UNKNOWN)


def _clean(value: str | None) -> str:
    if value is None:
        return UNKNOWN
    cleaned = re.sub(r"<br\s*/?>", " ", str(value), flags=re.IGNORECASE)
    cleaned = cleaned.replace("\\-", "-")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or UNKNOWN
