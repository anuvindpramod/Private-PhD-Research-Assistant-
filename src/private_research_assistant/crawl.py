from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from .firecrawl_client import FirecrawlClient
from .paths import ensure_data_dirs, latest_raw_manifest_path, raw_run_dir
from .sources import SourceConfig, get_sources


def make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def crawl_sources(
    *,
    selected_source_ids: list[str] | None = None,
    run_id: str | None = None,
    limit_override: int | None = None,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    ensure_data_dirs(base_dir)
    run_id = run_id or make_run_id()
    sources = get_sources(selected_source_ids)
    client = FirecrawlClient()
    run_directory = raw_run_dir(run_id, base_dir)
    run_directory.mkdir(parents=True, exist_ok=True)

    documents: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []

    for source in sources:
        limit = limit_override or source.limit
        results = client.search(
            source.query,
            limit=limit,
            include_domains=source.include_domains,
            scrape=True,
        )
        normalized_docs = [_normalize_search_result(source, item, index) for index, item in enumerate(results, start=1)]
        normalized_docs = [doc for doc in normalized_docs if doc["url"] and doc["markdown"]]
        documents.extend(normalized_docs)
        source_summaries.append(
            {
                "source_id": source.source_id,
                "name": source.name,
                "query": source.query,
                "include_domains": list(source.include_domains),
                "requested_limit": limit,
                "documents_saved": len(normalized_docs),
            }
        )

    documents_path = run_directory / "documents.jsonl"
    with documents_path.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")

    manifest = {
        "run_id": run_id,
        "crawl_date": date.today().isoformat(),
        "fetched_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "document_count": len(documents),
        "sources": source_summaries,
        "documents_path": str(documents_path),
    }
    manifest_path = run_directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    latest_raw_manifest_path(base_dir).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_raw_manifest(run_id: str | None = None, base_dir: str | Path | None = None) -> dict[str, Any]:
    manifest_path = raw_run_dir(run_id, base_dir) / "manifest.json" if run_id else latest_raw_manifest_path(base_dir)
    if not manifest_path.exists():
        raise FileNotFoundError(f"No crawl manifest found at {manifest_path}. Run pra crawl first.")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_raw_documents(run_id: str | None = None, base_dir: str | Path | None = None) -> list[dict[str, Any]]:
    manifest = load_raw_manifest(run_id, base_dir)
    documents_path = Path(manifest["documents_path"])
    if not documents_path.exists():
        documents_path = raw_run_dir(manifest["run_id"], base_dir) / "documents.jsonl"
    documents: list[dict[str, Any]] = []
    with documents_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                documents.append(json.loads(line))
    return documents


def _normalize_search_result(source: SourceConfig, item: dict[str, Any], index: int) -> dict[str, Any]:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    url = item.get("url") or metadata.get("sourceURL") or metadata.get("source_url") or ""
    title = item.get("title") or metadata.get("title") or url
    markdown = item.get("markdown") or item.get("content") or item.get("description") or ""
    doc_hash = hashlib.sha1(f"{source.source_id}:{url}:{index}".encode("utf-8")).hexdigest()[:10]
    return {
        "doc_id": f"{source.source_id}-{doc_hash}",
        "source_id": source.source_id,
        "source_name": source.name,
        "title": str(title).strip(),
        "url": str(url).strip(),
        "markdown": str(markdown).strip(),
        "fetched_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }

