from __future__ import annotations

from pathlib import Path

from .config import RuntimeConfig, runtime_config_from_env


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
