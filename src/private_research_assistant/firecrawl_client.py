from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .env import get_firecrawl_api_key
from .exceptions import UserFacingError


@dataclass
class FirecrawlClient:
    api_key: str | None = None
    api_url: str = "https://api.firecrawl.dev"
    timeout_seconds: int = 90

    def __post_init__(self) -> None:
        self.api_key = self.api_key or get_firecrawl_api_key()
        if not self.api_key:
            raise UserFacingError("Missing FIRECRAWL_API_KEY. Copy .env.example to .env and add your key.")

    def search(
        self,
        query: str,
        *,
        limit: int,
        include_domains: tuple[str, ...] = (),
        scrape: bool = True,
    ) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {
            "query": query,
            "limit": limit,
            "sources": ["web"],
            "ignoreInvalidURLs": True,
            "timeout": self.timeout_seconds * 1000,
        }
        if include_domains:
            payload["includeDomains"] = list(include_domains)
        if scrape:
            payload["scrapeOptions"] = {"formats": ["markdown"]}

        response = self._post("/v2/search", payload)
        data = response.get("data", response)
        if isinstance(data, dict):
            web = data.get("web", [])
        elif isinstance(data, list):
            web = data
        else:
            web = []
        return [item for item in web if isinstance(item, dict)]

    def scrape(self, url: str) -> dict[str, Any]:
        response = self._post("/v2/scrape", {"url": url, "formats": ["markdown"]})
        data = response.get("data", response)
        if not isinstance(data, dict):
            raise UserFacingError(f"Firecrawl returned an unexpected scrape response for {url}")
        return data

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

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise UserFacingError(f"Firecrawl returned non-JSON content: {raw[:200]}") from exc
        if isinstance(parsed, dict) and parsed.get("success") is False:
            raise UserFacingError(f"Firecrawl request failed: {parsed}")
        if not isinstance(parsed, dict):
            raise UserFacingError("Firecrawl returned an unexpected response shape.")
        return parsed

