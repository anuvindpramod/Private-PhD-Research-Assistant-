from __future__ import annotations

from pathlib import Path

from .config import RuntimeConfig, runtime_config_from_env
from .exceptions import UserFacingError


def create_llm(*, json_mode: bool = False, base_dir: str | Path | None = None):
    """Read this project's settings and create the Ollama text-model adapter."""
    try:
        from llama_index.llms.ollama import Ollama
    except ImportError as exc:
        raise UserFacingError("Missing LlamaIndex Ollama integration. Run `uv sync` first.") from exc

    config = runtime_config_from_env(base_dir)
    return Ollama(
        model=config.chat_model,
        base_url=config.ollama_base_url,
        request_timeout=config.request_timeout_seconds,
        context_window=config.context_window,
        json_mode=json_mode,
    )


def create_embed_model(config: RuntimeConfig):
    """Use the caller's settings, so saved index metadata matches the adapter."""
    try:
        from llama_index.embeddings.ollama import OllamaEmbedding
    except ImportError as exc:
        raise UserFacingError("Missing LlamaIndex Ollama embeddings integration. Run `uv sync` first.") from exc

    return OllamaEmbedding(model_name=config.embed_model, base_url=config.ollama_base_url)
