from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


UNKNOWN = "unknown"

OPPORTUNITY_FIELDS = (
    "title",
    "university_or_lab",
    "country",
    "deadline",
    "funding",
    "eligibility",
    "topic_fit",
    "application_url",
)


@dataclass
class Opportunity:
    title: str = UNKNOWN
    university_or_lab: str = UNKNOWN
    country: str = UNKNOWN
    deadline: str = UNKNOWN
    funding: str = UNKNOWN
    eligibility: str = UNKNOWN
    topic_fit: str = UNKNOWN
    application_url: str = UNKNOWN
    evidence_refs: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "Opportunity":
        values = {field_name: normalize_value(mapping.get(field_name)) for field_name in OPPORTUNITY_FIELDS}
        raw_refs = (
            mapping.get("evidence_refs")
            or mapping.get("field_evidence")
            or mapping.get("citations")
            or mapping.get("source_refs")
            or {}
        )
        refs = normalize_evidence_refs(raw_refs, values)
        return cls(**values, evidence_refs=refs)

    def to_dict(self) -> dict[str, Any]:
        payload = {field_name: getattr(self, field_name) for field_name in OPPORTUNITY_FIELDS}
        payload["evidence_refs"] = self.evidence_refs
        return payload


def normalize_value(value: Any) -> str:
    if value is None:
        return UNKNOWN
    if isinstance(value, list):
        value = "; ".join(str(item).strip() for item in value if str(item).strip())
    value = str(value).strip()
    if not value:
        return UNKNOWN
    if value.lower() in {"n/a", "na", "none", "null", "not specified", "not available"}:
        return UNKNOWN
    return " ".join(value.split())


def normalize_evidence_refs(raw_refs: Any, values: dict[str, str]) -> dict[str, list[str]]:
    if isinstance(raw_refs, list):
        refs = _normalize_ref_list(raw_refs)
        return {field_name: refs for field_name, value in values.items() if value != UNKNOWN}

    if not isinstance(raw_refs, dict):
        return {}

    normalized: dict[str, list[str]] = {}
    for field_name in OPPORTUNITY_FIELDS:
        field_refs = raw_refs.get(field_name)
        if field_refs is None:
            continue
        normalized[field_name] = _normalize_ref_list(field_refs)
    return normalized


def _normalize_ref_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        candidates = [value]
    else:
        try:
            candidates = list(value)
        except TypeError:
            candidates = [value]
    refs = []
    for candidate in candidates:
        ref = str(candidate).strip()
        if ref and ref not in refs:
            refs.append(ref)
    return refs

