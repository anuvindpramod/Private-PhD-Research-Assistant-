from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .env import load_env_file
from .exceptions import UserFacingError


CHAT_MODEL = "qwen3:1.7b"
EMBED_MODEL = "qwen3-embedding:0.6b"
OLLAMA_BASE_URL = "http://localhost:11434"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
INSERT_BATCH_SIZE = 64
DEFAULT_TOP_K = 8
REQUEST_TIMEOUT_SECONDS = 180.0
CONTEXT_WINDOW = 8192

MIN_FREE_DISK_GB = 2.5


@dataclass(frozen=True)
class RuntimeConfig:
    """The settings used for one operation; this class does not call a model."""

    chat_model: str = CHAT_MODEL
    embed_model: str = EMBED_MODEL
    ollama_base_url: str = OLLAMA_BASE_URL
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP
    insert_batch_size: int = INSERT_BATCH_SIZE
    top_k: int = DEFAULT_TOP_K
    request_timeout_seconds: float = REQUEST_TIMEOUT_SECONDS
    context_window: int = CONTEXT_WINDOW


def runtime_config_from_env(base_dir: str | Path | None = None) -> RuntimeConfig:
    """Choose settings: process environment > project .env > defaults above.

    Read the file each time, so editing .env affects the next command. Nothing
    is copied into os.environ, and blank settings cause a readable error.
    """
    settings = load_env_file(base_dir)
    settings.update(os.environ)
    defaults = {
        "PRA_CHAT_MODEL": CHAT_MODEL,
        "PRA_EMBED_MODEL": EMBED_MODEL,
        "PRA_OLLAMA_BASE_URL": OLLAMA_BASE_URL,
    }
    for name, default in defaults.items():
        settings[name] = settings.get(name, default).strip()
        if not settings[name]:
            raise UserFacingError(f"{name} cannot be empty. Set it in .env or remove it to use the default.")

    return RuntimeConfig(
        chat_model=settings["PRA_CHAT_MODEL"],
        embed_model=settings["PRA_EMBED_MODEL"],
        ollama_base_url=settings["PRA_OLLAMA_BASE_URL"],
    )


def normalized_model_name(name: str) -> str:
    """Compare Ollama names ignoring outer spaces and an omitted :latest tag.

    Other tags remain distinct. This compares names, not model file contents.
    """
    name = name.strip()
    if name and ":" not in name.rsplit("/", 1)[-1]:
        return name + ":latest"
    return name
