from __future__ import annotations

import os
from dataclasses import dataclass


CHAT_MODEL = "qwen3:1.7b"
EMBED_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
INSERT_BATCH_SIZE = 64
DEFAULT_TOP_K = 8
REQUEST_TIMEOUT_SECONDS = 180.0
CONTEXT_WINDOW = 8192

MIN_FREE_DISK_GB = 2.5
MODEL_DISK_GB = {
    CHAT_MODEL: 1.8,
    EMBED_MODEL: 0.274,
}


@dataclass(frozen=True)
class RuntimeConfig:
    chat_model: str = CHAT_MODEL
    embed_model: str = EMBED_MODEL
    ollama_base_url: str = OLLAMA_BASE_URL
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP
    insert_batch_size: int = INSERT_BATCH_SIZE
    top_k: int = DEFAULT_TOP_K
    request_timeout_seconds: float = REQUEST_TIMEOUT_SECONDS
    context_window: int = CONTEXT_WINDOW


def runtime_config_from_env() -> RuntimeConfig:
    return RuntimeConfig(
        chat_model=os.environ.get("PRA_CHAT_MODEL", CHAT_MODEL),
        embed_model=os.environ.get("PRA_EMBED_MODEL", EMBED_MODEL),
        ollama_base_url=os.environ.get("PRA_OLLAMA_BASE_URL", OLLAMA_BASE_URL),
    )

