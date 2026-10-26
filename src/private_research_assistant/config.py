from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .env import load_env_file
from .exceptions import UserFacingError


DEFAULT_TOP_K = 8
MIN_FREE_DISK_GB = 2.5


@dataclass(frozen=True)
class RuntimeConfig:
    chat_model: str = "qwen3:1.7b"
    embed_model: str = "qwen3-embedding:0.6b"
    ollama_base_url: str = "http://localhost:11434"
    chunk_size: int = 800
    chunk_overlap: int = 120
    insert_batch_size: int = 64
    request_timeout_seconds: float = 180.0
    context_window: int = 8192


def runtime_config_from_env(base_dir: str | Path | None = None) -> RuntimeConfig:
    """Choose settings: process environment > .env > RuntimeConfig defaults."""
    settings = load_env_file(base_dir)
    settings.update(os.environ)
    defaults = RuntimeConfig()
    model_settings = {
        "PRA_CHAT_MODEL": defaults.chat_model,
        "PRA_EMBED_MODEL": defaults.embed_model,
        "PRA_OLLAMA_BASE_URL": defaults.ollama_base_url,
    }
    for name, default in model_settings.items():
        settings[name] = settings.get(name, default).strip()
        if not settings[name]:
            raise UserFacingError(f"{name} cannot be empty. Set it in .env or remove it to use the default.")

    return RuntimeConfig(
        chat_model=settings["PRA_CHAT_MODEL"],
        embed_model=settings["PRA_EMBED_MODEL"],
        ollama_base_url=settings["PRA_OLLAMA_BASE_URL"],
    )


def normalized_model_name(name: str) -> str:
    """Ignore outer spaces and an omitted :latest tag when comparing names."""
    name = name.strip()
    if name and ":" not in name.rsplit("/", 1)[-1]:
        return name + ":latest"
    return name
