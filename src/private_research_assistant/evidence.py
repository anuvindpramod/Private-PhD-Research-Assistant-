from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re
import html

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
    """Check each filled field against its cited chunks; return pass/errors.

    Missing fields are skipped. Filled fields need existing reference IDs and
    at least one matching normalized excerpt (or source/link match for a URL).
    This does not change the row or prove the source is factually correct.
    """
    errors: list[str] = []
    for field_name in OPPORTUNITY_FIELDS:
        value = getattr(opportunity, field_name)
        if not is_filled(value):
            continue
        refs = opportunity.evidence_refs.get(field_name, [])
        if not refs:
            errors.append(f"{field_name} is filled but has no evidence_refs entry")
            continue
        for ref in refs:
            if ref not in chunk_map:
                errors.append(f"{field_name} cites missing evidence chunk {ref}")
        cited = [chunk_map[ref] for ref in refs if ref in chunk_map]
        if field_name == "application_url":
            supported = any(value == chunk.get("url") or value in re.findall(r'https?://[^\s<>\)\]\"\x27]+', chunk.get("text", "")) for chunk in cited)
        else:
            supported = any(normalize_excerpt(value) in normalize_excerpt(chunk.get("text", "")) for chunk in cited)
        if not supported:
            errors.append(f"{field_name} has no matching source excerpt")
    return EvidenceValidationResult(passing=not errors, errors=errors)


def normalize_excerpt(value: str) -> str:
    """Normalize HTML breaks/entities, whitespace, case, and escaped hyphens.

    This enables textual matching; it does not recognize semantic paraphrases.
    """
    value = html.unescape(re.sub(r"<br\s*/?>", " ", value, flags=re.I))
    return " ".join(value.replace("\\-", "-").split()).casefold()


def clear_unsupported_fields(opportunity: Opportunity, chunks: dict[str, Any]) -> Opportunity:
    """Check fields individually, replacing unsupported values with unknown.

    Mutates and returns the SAME Opportunity, removing refs for cleared fields.
    It does not itself drop the row; the caller decides whether to retain it.
    """
    for name in OPPORTUNITY_FIELDS:
        single = Opportunity(**{name: getattr(opportunity, name)}, evidence_refs={name: opportunity.evidence_refs.get(name, [])})
        if not validate_evidence_refs(single, chunks).passing:
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
