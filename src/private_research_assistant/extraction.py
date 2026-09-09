"""Turn saved advertisement chunks into a table with source references.

Entry: cli._cmd_extract -> extract_opportunities -> save_opportunities.
The default path uses source-specific Python rules, not an LLM or vector search.
An optional LLM fallback handles documents that reach the fallback branch.
Both paths check evidence before keeping rows. Outputs describe the crawl date,
not necessarily today's availability.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from .dates import is_active_deadline
from .evidence import validate_evidence_refs, clear_unsupported_fields
from .exceptions import UserFacingError
from .indexing import load_chunks, load_index_manifest
from .llm import create_llm
from .models import OPPORTUNITY_FIELDS, Opportunity
from .paths import latest_output_path, output_run_dir
from .structured_extractors import extract_known_source_opportunities


@dataclass
class ExtractionResult:
    """Return rows, warnings, and written file paths to the CLI for printing."""
    run_id: str
    opportunities: list[Opportunity]
    warnings: list[dict[str, Any]]
    output_paths: dict[str, str]
    crawl_date: str = "unknown"


def extract_opportunities(
    *,
    run_id: str | None = None,
    max_documents: int | None = None,
    llm_fallback: bool = False,
    output_format: str = "all",
    base_dir: str | Path | None = None,
) -> ExtractionResult:
    """Load saved chunks, extract supported rows, and write the result files.

    Input: an indexed run (latest if omitted) and optional limits/settings.
    Output: ExtractionResult; the CLI prints its counts and file paths.
    Side effects: replaces this run's exports and the latest-output JSON.

    For each document, try the known-source parser first. A parser warning
    skips that document. Returned rows are checked and kept or rejected.
    Only a document with no parser rows/warnings can reach the LLM fallback,
    and only when llm_fallback=True. No new pages are crawled or embedded here.
    """
    index_manifest = load_index_manifest(run_id, base_dir)
    run_id = index_manifest["run_id"]
    chunks = load_chunks(run_id, base_dir)
    grouped = _group_chunks_by_doc(chunks)
    if max_documents:
        grouped = dict(list(grouped.items())[:max_documents])

    llm = create_llm(json_mode=True) if llm_fallback else None
    opportunities: list[Opportunity] = []
    warnings: list[dict[str, Any]] = []
    crawl_date = _crawl_date_from_manifest(index_manifest)

    for doc_id, doc_chunks in grouped.items():
        # The parser can append to this shared list; a new warning blocks fallback.
        warning_count = len(warnings)
        structured_opportunities = extract_known_source_opportunities(doc_id, doc_chunks, crawl_date, warnings)
        if len(warnings) > warning_count:
            continue
        if structured_opportunities:
            for opportunity in structured_opportunities:
                validation = validate_evidence_refs(opportunity, chunks)
                if validation.passing:
                    opportunities.append(opportunity)
                else:
                    warnings.append(
                        {
                            "doc_id": doc_id,
                            "kind": "structured_parser_unsupported_fields",
                            "title": opportunity.title,
                            "errors": validation.errors,
                        }
                    )
            continue

        # No supported parser result: the default command skips instead of guessing.
        if llm is None:
            warnings.append({"doc_id": doc_id, "kind": "skipped_no_structured_parser"})
            continue

        # Everything below this point in the loop belongs to optional LLM extraction.
        prompt = _build_extraction_prompt(doc_chunks, crawl_date)
        raw_response = str(llm.complete(prompt))
        try:
            parsed = _parse_json_object(raw_response)
        except ValueError as exc:
            warnings.append({"doc_id": doc_id, "kind": "invalid_json", "detail": str(exc), "raw": raw_response[:1000]})
            continue
        for raw_opportunity in parsed.get("opportunities", []):
            if not isinstance(raw_opportunity, dict):
                continue
            opportunity = Opportunity.from_mapping(raw_opportunity)
            opportunity.evidence_refs = _normalize_local_evidence_refs(opportunity.evidence_refs, doc_id, chunks)
            if not is_active_deadline(opportunity.deadline, crawl_date):
                warnings.append({"doc_id": doc_id, "kind": "expired", "title": opportunity.title})
                continue
            # This helper changes the row in place: unsupported fields become unknown.
            opportunity = clear_unsupported_fields(opportunity, {c['evidence_id']: c for c in doc_chunks})
            if opportunity.title == "unknown":
                warnings.append({"doc_id": doc_id, "kind": "unsupported_title"})
                continue
            validation = validate_evidence_refs(opportunity, chunks)
            if not validation.passing:
                warnings.append(
                    {
                        "doc_id": doc_id,
                        "kind": "unsupported_fields",
                        "title": opportunity.title,
                        "errors": validation.errors,
                    }
                )
                continue
            opportunities.append(opportunity)

    opportunities = _dedupe_opportunities(opportunities)
    output_paths = save_opportunities(run_id, opportunities, warnings, output_format, base_dir, crawl_date=crawl_date.isoformat())
    return ExtractionResult(run_id=run_id, opportunities=opportunities, warnings=warnings, output_paths=output_paths, crawl_date=crawl_date.isoformat())


def save_opportunities(
    run_id: str,
    opportunities: list[Opportunity],
    warnings: list[dict[str, Any]],
    output_format: str = "all",
    base_dir: str | Path | None = None,
    *,
    crawl_date: str = "unknown",
) -> dict[str, str]:
    """Write rows/warnings to disk and return a label-to-file-path dictionary.

    JSON is always written, even for a CSV-only or Markdown-only request,
    because inspection/live evaluation read the latest JSON. Existing files
    at these paths are overwritten; this is not an append-only history.
    """
    output_directory = output_run_dir(run_id, base_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "crawl_date": crawl_date,
        "opportunity_count": len(opportunities),
        "opportunities": [opportunity.to_dict() for opportunity in opportunities],
        "warnings": warnings,
    }
    paths: dict[str, str] = {}

    # Inspect and eval always read canonical JSON, regardless of export format.
    json_path = output_directory / "opportunities.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    latest_output_path(base_dir).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["json"] = str(json_path)

    if output_format in {"all", "markdown"}:
        markdown_path = output_directory / "opportunities.md"
        markdown_path.write_text(f"As of crawl date: {crawl_date}\n\n" + _to_markdown(opportunities), encoding="utf-8")
        paths["markdown"] = str(markdown_path)

    if output_format in {"all", "csv"}:
        csv_path = output_directory / "opportunities.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[*OPPORTUNITY_FIELDS, "evidence_refs"])
            writer.writeheader()
            for opportunity in opportunities:
                row = {field_name: getattr(opportunity, field_name) for field_name in OPPORTUNITY_FIELDS}
                row["evidence_refs"] = json.dumps(opportunity.evidence_refs, ensure_ascii=False)
                writer.writerow(row)
        paths["csv"] = str(csv_path)

    warnings_path = output_directory / "warnings.json"
    warnings_path.write_text(json.dumps(warnings, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["warnings"] = str(warnings_path)
    return paths


def _group_chunks_by_doc(chunks: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Reorganize {chunk_id: chunk} into {doc_id: [chunk, ...]}.

    Each group holds pieces of one original page, preserving their input order.
    This is grouping by identity, not searching by semantic similarity.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks.values():
        grouped.setdefault(chunk["doc_id"], []).append(chunk)
    return grouped


def _build_extraction_prompt(chunks: list[dict[str, Any]], crawl_date: date) -> str:
    """Return instructions plus evidence text for the optional LLM request.

    This function builds a string; llm.complete later sends it to the model.
    Only the first eight chunks and 1800 text characters per chunk are included.
    These are prompt instructions, not Python guarantees. The dated rule means
    deadline >= crawl_date; unknown applies separately to missing field values.
    Missing/rolling deadlines are allowed by policy, not confirmed to be open.
    """
    context = "\n\n".join(
        f"[{chunk['evidence_id']}]\nTitle: {chunk.get('title', '')}\nURL: {chunk.get('url', '')}\nText: {chunk.get('text', '')[:1800]}"
        for chunk in chunks[:8]
    )
    return f"""/no_think
Extract active AI/ML PhD advertisement opportunities from the evidence chunks.

Crawl date: {crawl_date.isoformat()}

Return valid JSON only with this exact shape:
{{
  "opportunities": [
    {{
      "title": "unknown",
      "university_or_lab": "unknown",
      "country": "unknown",
      "deadline": "unknown",
      "funding": "unknown",
      "eligibility": "unknown",
      "topic_fit": "unknown",
      "application_url": "unknown",
      "evidence_refs": {{
        "title": ["chunk-id"],
        "university_or_lab": ["chunk-id"],
        "country": ["chunk-id"],
        "deadline": ["chunk-id"],
        "funding": ["chunk-id"],
        "eligibility": ["chunk-id"],
        "topic_fit": ["chunk-id"],
        "application_url": ["chunk-id"]
      }}
    }}
  ]
}}

Rules:
- Include only PhD, doctoral, doctorate, or studentship opportunities related to AI, machine learning, deep learning, NLP, computer vision, data science, robotics, or agentic AI.
- Do not include postdoc-only, lecturer, faculty, or industry jobs.
- Include an opportunity only if the deadline is unknown, year-round, rolling, open until filled, or on/after the crawl date.
- Use "unknown" for fields not directly stated in the chunks.
- Every non-unknown field must have one or more evidence_refs entries pointing to the chunk ids below.
- application_url can use the source URL when a separate application link is not stated.

Evidence chunks:
{context}
"""


def _parse_json_object(raw: str) -> dict[str, Any]:
    """Remove common model wrappers and parse the remaining JSON object.

    Parsing checks JSON syntax, not whether fields or claims are correct.
    Missing braces or malformed JSON raise ValueError (including JSONDecodeError).
    """
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response.")
    return json.loads(cleaned[start : end + 1])


def _dedupe_opportunities(opportunities: list[Opportunity]) -> list[Opportunity]:
    """Keep the first row per lowercased (title, institution, deadline) tuple.

    Later matches are discarded, not merged; differing funding is not compared.
    """
    seen: set[tuple[str, str, str]] = set()
    deduped: list[Opportunity] = []
    for opportunity in opportunities:
        key = (
            opportunity.title.lower(),
            opportunity.university_or_lab.lower(),
            opportunity.deadline.lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(opportunity)
    return deduped


def _normalize_local_evidence_refs(
    evidence_refs: dict[str, list[str]],
    doc_id: str,
    chunks: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    """Expand resolvable short citation IDs and remove repeats within each field.

    Returns a new mapping. References that cannot be resolved are omitted;
    later evidence validation decides whether the remaining references suffice.
    """
    normalized: dict[str, list[str]] = {}
    for field_name, refs in evidence_refs.items():
        normalized_refs: list[str] = []
        for ref in refs:
            resolved = _resolve_local_ref(ref, doc_id, chunks)
            if resolved and resolved not in normalized_refs:
                normalized_refs.append(resolved)
        normalized[field_name] = normalized_refs
    return normalized


def _resolve_local_ref(ref: str, doc_id: str, chunks: dict[str, dict[str, Any]]) -> str | None:
    """Find an existing full ID, or return None if resolution fails.

    Try an exact ID, a document-relative ID (such as chunk-2), then a unique
    suffix match. Resolving an ID does not establish that it supports a claim.
    """
    value = str(ref).strip()
    if value in chunks:
        return value

    if value.startswith("#"):
        candidate = f"{doc_id}{value}"
        if candidate in chunks:
            return candidate

    chunk_match = re.fullmatch(r"(?:#?chunk[-_\s]?)(\d+)", value, flags=re.IGNORECASE)
    if chunk_match:
        candidate = f"{doc_id}#chunk-{int(chunk_match.group(1))}"
        if candidate in chunks:
            return candidate

    suffix_matches = [chunk_id for chunk_id in chunks if chunk_id.endswith(value)]
    if len(suffix_matches) == 1:
        return suffix_matches[0]

    return None


def _crawl_date_from_manifest(index_manifest: dict[str, Any]) -> date:
    """Read an ISO crawl date; fall back to today if absent or invalid.

    The fallback is a convenience, not recovered historical crawl information.
    """
    raw = index_manifest.get("crawl_date")
    if isinstance(raw, str):
        try:
            return date.fromisoformat(raw)
        except ValueError:
            pass
    return date.today()


def _to_markdown(opportunities: list[Opportunity]) -> str:
    """Build table text in memory; save_opportunities is responsible for writing it."""
    headers = [*OPPORTUNITY_FIELDS, "evidence_refs"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for opportunity in opportunities:
        row = [getattr(opportunity, field_name) for field_name in OPPORTUNITY_FIELDS]
        row.append(json.dumps(opportunity.evidence_refs, ensure_ascii=False))
        lines.append("| " + " | ".join(_escape_markdown_cell(value) for value in row) + " |")
    return "\n".join(lines) + "\n"


def _escape_markdown_cell(value: Any) -> str:
    """Keep pipes/newlines inside a value from breaking the Markdown table layout."""
    return str(value).replace("|", "\\|").replace("\n", " ")
