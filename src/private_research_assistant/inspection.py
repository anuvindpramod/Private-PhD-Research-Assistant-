from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .crawl import load_raw_documents, load_raw_manifest
from .exceptions import UserFacingError
from .indexing import load_chunks, load_index_manifest
from .paths import latest_output_path


def inspect_status(base_dir: str | Path | None = None) -> str:
    lines = ["Private Research Assistant status"]
    try:
        raw_manifest = load_raw_manifest(base_dir=base_dir)
    except (FileNotFoundError, UserFacingError):
        lines.append("- Crawl: not found. Run `pra crawl`.")
    else:
        lines.append(
            f"- Crawl: run {raw_manifest['run_id']}, "
            f"{raw_manifest['document_count']} documents, crawl date {raw_manifest.get('crawl_date', 'unknown')}."
        )

    try:
        index_manifest = load_index_manifest(base_dir=base_dir)
    except UserFacingError:
        lines.append("- Index: not found. Run `pra index`.")
    else:
        lines.append(
            f"- Index: {index_manifest['chunk_count']} chunks in collection "
            f"{index_manifest['collection_name']}."
        )

    output_path = latest_output_path(base_dir)
    if not output_path.exists():
        lines.append("- Extraction: not found. Run `pra extract`.")
    else:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        lines.append(
            f"- Extraction: {payload.get('opportunity_count', 0)} supported opportunities "
            f"for run {payload.get('run_id', 'unknown')}."
        )
    return "\n".join(lines)


def inspect_raw_documents(
    *,
    run_id: str | None = None,
    limit: int = 5,
    query: str | None = None,
    base_dir: str | Path | None = None,
) -> str:
    documents = load_raw_documents(run_id, base_dir)
    if query:
        documents = [doc for doc in documents if _matches_query(doc, query)]
    lines = [f"Raw documents: showing {min(limit, len(documents))} of {len(documents)}"]
    for document in documents[:limit]:
        lines.extend(
            [
                "",
                f"[{document['doc_id']}] {document.get('source_name', '')}",
                document.get("title", ""),
                document.get("url", ""),
                _snippet(document.get("markdown", "")),
            ]
        )
    return "\n".join(lines)


def inspect_chunks(
    *,
    run_id: str | None = None,
    limit: int = 5,
    query: str | None = None,
    base_dir: str | Path | None = None,
) -> str:
    chunk_map = load_chunks(run_id, base_dir)
    chunks = list(chunk_map.values())
    if query:
        chunks = [chunk for chunk in chunks if _matches_query(chunk, query)]
    lines = [f"Chunks: showing {min(limit, len(chunks))} of {len(chunks)}"]
    for chunk in chunks[:limit]:
        lines.extend(
            [
                "",
                f"[{chunk['evidence_id']}] {chunk.get('source_name', '')}",
                chunk.get("title", ""),
                chunk.get("url", ""),
                _snippet(chunk.get("text", "")),
            ]
        )
    return "\n".join(lines)


def inspect_opportunities(*, limit: int = 5, base_dir: str | Path | None = None) -> str:
    output_path = latest_output_path(base_dir)
    if not output_path.exists():
        raise UserFacingError(f"No extracted opportunity table found at {output_path}. Run `pra extract` first.")
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    opportunities = payload.get("opportunities", [])
    lines = [f"Opportunities: showing {min(limit, len(opportunities))} of {len(opportunities)}"]
    for opportunity in opportunities[:limit]:
        lines.extend(
            [
                "",
                f"Title: {opportunity.get('title', 'unknown')}",
                f"University/lab: {opportunity.get('university_or_lab', 'unknown')}",
                f"Country: {opportunity.get('country', 'unknown')}",
                f"Deadline: {opportunity.get('deadline', 'unknown')}",
                f"Funding: {opportunity.get('funding', 'unknown')}",
                f"Application: {opportunity.get('application_url', 'unknown')}",
                f"Evidence refs: {json.dumps(opportunity.get('evidence_refs', {}), ensure_ascii=False)}",
            ]
        )
    return "\n".join(lines)


def _matches_query(item: dict[str, Any], query: str) -> bool:
    haystack = json.dumps(item, ensure_ascii=False).lower()
    return query.lower() in haystack


def _snippet(text: str, max_chars: int = 700) -> str:
    value = " ".join(str(text).split())
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3] + "..."
