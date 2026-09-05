from __future__ import annotations

import os
from pathlib import Path

from .paths import project_root


def load_env_file(base_dir: str | Path | None = None) -> dict[str, str]:
    env_path = project_root(base_dir) / ".env"
    loaded: dict[str, str] = {}
    if not env_path.exists():
        return loaded

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        loaded[key] = value
        os.environ.setdefault(key, value)
    return loaded


def get_firecrawl_api_key(base_dir: str | Path | None = None) -> str | None:
    load_env_file(base_dir)
    value = os.environ.get("FIRECRAWL_API_KEY")
    if value and value.strip() and value.strip() != "fc-YOUR-API-KEY":
        return value.strip()
    return None

