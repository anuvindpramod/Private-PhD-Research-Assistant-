from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Any

from .models import OPPORTUNITY_FIELDS, UNKNOWN, Opportunity


@dataclass
class EvidenceValidationResult:
    passing: bool
    errors: list[str]


def is_filled(value: str | None) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() not in {"", UNKNOWN, "n/a", "none", "null"}


def validate_evidence_refs(opportunity: Opportunity, chunk_map: dict[str, Any]) -> EvidenceValidationResult:
    """Check citation IDs and text matches, not factual correctness."""
    errors: list[str] = []
    for field_name in OPPORTUNITY_FIELDS:
        value = getattr(opportunity, field_name)
        refs = opportunity.evidence_refs.get(field_name, [])
        errors.extend(_field_errors(field_name, value, refs, chunk_map))
    return EvidenceValidationResult(passing=not errors, errors=errors)


def _field_errors(name: str, value: str, refs: list[str], chunks: dict[str, Any]) -> list[str]:
    if not is_filled(value):
        return []
    if not refs:
        return [f"{name} is filled but has no evidence_refs entry"]

    errors = []
    supported = False
    for ref in refs:
        if ref not in chunks:
            errors.append(f"{name} cites missing evidence chunk {ref}")
            continue
        chunk = chunks[ref]
        if name == "application_url":
            urls = re.findall(r'https?://[^\s<>\)\]\"\x27]+', chunk.get("text", ""))
            matches = value == chunk.get("url") or value in urls
        else:
            matches = normalize_excerpt(value) in normalize_excerpt(chunk.get("text", ""))
        supported = supported or matches
    if not supported:
        errors.append(f"{name} has no matching source excerpt")
    return errors


def normalize_excerpt(value: str) -> str:
    """Ignore HTML formatting, whitespace and case when matching excerpts."""
    value = html.unescape(re.sub(r"<br\s*/?>", " ", value, flags=re.I))
    return " ".join(value.replace("\\-", "-").split()).casefold()


def clear_unsupported_fields(opportunity: Opportunity, chunks: dict[str, Any]) -> Opportunity:
    """Replace unsupported fields in place with unknown and remove their refs."""
    for name in OPPORTUNITY_FIELDS:
        value = getattr(opportunity, name)
        refs = opportunity.evidence_refs.get(name, [])
        if _field_errors(name, value, refs, chunks):
            setattr(opportunity, name, UNKNOWN)
            opportunity.evidence_refs.pop(name, None)
    return opportunity


def collect_cited_refs(opportunity: Opportunity) -> list[str]:
    refs: list[str] = []
    for field_refs in opportunity.evidence_refs.values():
        for ref in field_refs:
            if ref not in refs:
                refs.append(ref)
    return refs
