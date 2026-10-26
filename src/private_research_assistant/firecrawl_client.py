from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .exceptions import UserFacingError


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

