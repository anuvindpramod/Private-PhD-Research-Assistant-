from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from . import UserFacingError
from .paths import project_root


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


def get_firecrawl_api_key(base_dir: str | Path | None = None) -> str | None:
    load_dotenv(project_root(base_dir) / ".env", override=False)
    value = os.environ.get("FIRECRAWL_API_KEY")
    if value and value.strip() and value.strip() != "fc-YOUR-API-KEY":
        return value.strip()
    return None


def runtime_config_from_env(base_dir: str | Path | None = None) -> RuntimeConfig:
    """Choose settings: process environment > .env > RuntimeConfig defaults."""
    load_dotenv(project_root(base_dir) / ".env", override=False)
    settings = os.environ.copy()
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


def create_llm(*, json_mode: bool = False, base_dir: str | Path | None = None):
    from llama_index.llms.ollama import Ollama

    config = runtime_config_from_env(base_dir)
    return Ollama(
        model=config.chat_model,
        base_url=config.ollama_base_url,
        request_timeout=config.request_timeout_seconds,
        context_window=config.context_window,
        json_mode=json_mode,
    )


def create_embed_model(config: RuntimeConfig):
    from llama_index.embeddings.ollama import OllamaEmbedding

    return OllamaEmbedding(model_name=config.embed_model, base_url=config.ollama_base_url)
