from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import MIN_FREE_DISK_GB, RuntimeConfig, get_firecrawl_api_key, normalized_model_name, runtime_config_from_env
from .paths import project_root


@dataclass
class CheckResult:
    name: str
    passing: bool
    detail: str


def run_setup_checks(base_dir: str | Path | None = None) -> list[CheckResult]:
    config = runtime_config_from_env(base_dir)
    root = project_root(base_dir)
    results = [
        _check_python(),
        _check_disk(root),
        _check_command("uv", "uv is installed."),
        _check_command("ollama", "Ollama CLI is installed."),
        _check_firecrawl_key(base_dir),
    ]
    ollama_result, model_results = _check_ollama_models(config)
    results.append(ollama_result)
    results.extend(model_results)
    results.extend(_check_python_dependencies())
    return results


def _check_python() -> CheckResult:
    version = sys.version_info
    passing = version >= (3, 12)
    return CheckResult(
        "Python 3.12+",
        passing,
        f"Running Python {version.major}.{version.minor}.{version.micro}.",
    )


def _check_disk(root: Path) -> CheckResult:
    usage = shutil.disk_usage(root)
    free_gb = usage.free / (1024**3)
    passing = free_gb >= MIN_FREE_DISK_GB
    return CheckResult(
        "Disk space",
        passing,
        f"{free_gb:.1f} GiB free; minimum working-space check: {MIN_FREE_DISK_GB:.1f} GiB. "
        "This does not estimate space for missing model downloads.",
    )


def _check_command(command: str, success_detail: str) -> CheckResult:
    path = shutil.which(command)
    return CheckResult(command, bool(path), success_detail if path else f"{command} was not found on PATH.")


def _check_firecrawl_key(base_dir: str | Path | None = None) -> CheckResult:
    key = get_firecrawl_api_key(base_dir)
    return CheckResult(
        "Firecrawl API key",
        bool(key),
        "FIRECRAWL_API_KEY is set." if key else "Missing FIRECRAWL_API_KEY in .env or environment.",
    )


def _check_ollama_models(config: RuntimeConfig) -> tuple[CheckResult, list[CheckResult]]:
    command_env = os.environ.copy()
    command_env["OLLAMA_HOST"] = config.ollama_base_url
    try:
        completed = subprocess.run(
            ["ollama", "list"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
            env=command_env,
        )
    except FileNotFoundError:
        return CheckResult("Ollama server", False, "Ollama CLI is missing."), []
    except subprocess.TimeoutExpired:
        return CheckResult("Ollama server", False, "Timed out while asking Ollama for local models."), []

    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "Ollama did not respond."
        return CheckResult("Ollama server", False, detail), [
            CheckResult(model, False, f"Cannot verify. Start Ollama, then run `ollama pull {model}`.")
            for model in (config.chat_model, config.embed_model)
        ]

    installed_models = {
        normalized_model_name(line.split()[0])
        for line in completed.stdout.splitlines()[1:]
        if line.strip()
    }
    model_results = []
    for model in (config.chat_model, config.embed_model):
        available = normalized_model_name(model) in installed_models
        model_results.append(
            CheckResult(
                model,
                available,
                "Model is available locally." if available else f"Missing. Run `ollama pull {model}`.",
            )
        )
    return CheckResult("Ollama server", True, "Ollama responded to `ollama list`."), model_results


def _check_python_dependencies() -> list[CheckResult]:
    modules = (
        ("chromadb", "ChromaDB"),
        ("llama_index.core", "LlamaIndex core"),
        ("llama_index.llms.ollama", "LlamaIndex Ollama LLM"),
        ("llama_index.embeddings.ollama", "LlamaIndex Ollama embeddings"),
        ("llama_index.vector_stores.chroma", "LlamaIndex Chroma vector store"),
    )
    results = []
    for module_name, label in modules:
        try:
            importlib.import_module(module_name)
        except ImportError:
            results.append(CheckResult(label, False, f"Missing Python module {module_name}. Run `uv sync`."))
        else:
            results.append(CheckResult(label, True, "Dependency imports successfully."))
    return results
