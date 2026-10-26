from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from . import UserFacingError
from .config import get_firecrawl_api_key
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
    client = FirecrawlClient(api_key=get_firecrawl_api_key(base_dir))
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
    manifest_text = json.dumps(manifest, indent=2)
    manifest_path.write_text(manifest_text, encoding="utf-8")
    latest_raw_manifest_path(base_dir).write_text(manifest_text, encoding="utf-8")
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


@dataclass
class FirecrawlClient:
    api_key: str | None
    api_url: str = "https://api.firecrawl.dev"
    timeout_seconds: int = 90

    def __post_init__(self) -> None:
        if not self.api_key:
            raise UserFacingError("Missing FIRECRAWL_API_KEY. Copy .env.example to .env and add your key (Its free upto a limit, DW ;) ")

    def search(
        self,
        query: str,
        *,
        limit: int,
        include_domains: tuple[str, ...] = (),
    ) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {
            "query": query,
            "limit": limit,
            "sources": ["web"],
            "ignoreInvalidURLs": True,
            "timeout": self.timeout_seconds * 1000,
            "scrapeOptions": {"formats": ["markdown"]},
        }
        if include_domains:
            payload["includeDomains"] = list(include_domains)

        response = self._post("/v2/search", payload)
        data = response.get("data", response)
        if isinstance(data, dict):
            web = data.get("web", [])
        elif isinstance(data, list):
            web = data
        else:
            raise UserFacingError("Firecrawl returned an unexpected search response.")
        if not isinstance(web, list):
            raise UserFacingError("Firecrawl search results must be a list.")
        return [item for item in web if isinstance(item, dict)]

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.api_url.rstrip('/')}{endpoint}",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise UserFacingError(f"Firecrawl HTTP {exc.code}: {error_body[:500]}") from exc
        except urllib.error.URLError as exc:
            raise UserFacingError(f"Could not reach Firecrawl: {exc.reason}") from exc

        parsed = json.loads(raw)
        if isinstance(parsed, dict) and parsed.get("success") is False:
            raise UserFacingError(f"Firecrawl request failed: {parsed}")
        if not isinstance(parsed, dict):
            raise UserFacingError("Firecrawl returned an unexpected response shape.")
        return parsed

