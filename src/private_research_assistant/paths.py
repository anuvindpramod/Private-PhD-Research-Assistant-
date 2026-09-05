from __future__ import annotations

import os
import re
from pathlib import Path


def project_root(base_dir: str | Path | None = None) -> Path:
    if base_dir is not None:
        return Path(base_dir).resolve()
    return Path(os.environ.get("PRA_PROJECT_ROOT", Path.cwd())).resolve()


def data_dir(base_dir: str | Path | None = None) -> Path:
    return project_root(base_dir) / "data"


def raw_dir(base_dir: str | Path | None = None) -> Path:
    return data_dir(base_dir) / "raw"


def processed_dir(base_dir: str | Path | None = None) -> Path:
    return data_dir(base_dir) / "processed"


def chroma_dir(base_dir: str | Path | None = None) -> Path:
    return data_dir(base_dir) / "chroma"


def outputs_dir(base_dir: str | Path | None = None) -> Path:
    return data_dir(base_dir) / "outputs"


def raw_run_dir(run_id: str, base_dir: str | Path | None = None) -> Path:
    return raw_dir(base_dir) / safe_run_id(run_id)


def processed_run_dir(run_id: str, base_dir: str | Path | None = None) -> Path:
    return processed_dir(base_dir) / safe_run_id(run_id)


def output_run_dir(run_id: str, base_dir: str | Path | None = None) -> Path:
    return outputs_dir(base_dir) / safe_run_id(run_id)


def latest_raw_manifest_path(base_dir: str | Path | None = None) -> Path:
    return raw_dir(base_dir) / "latest_manifest.json"


def latest_index_manifest_path(base_dir: str | Path | None = None) -> Path:
    return chroma_dir(base_dir) / "latest_index.json"


def latest_output_path(base_dir: str | Path | None = None) -> Path:
    return outputs_dir(base_dir) / "latest_opportunities.json"


def ensure_data_dirs(base_dir: str | Path | None = None) -> None:
    for directory in (raw_dir(base_dir), processed_dir(base_dir), chroma_dir(base_dir), outputs_dir(base_dir)):
        directory.mkdir(parents=True, exist_ok=True)


def safe_run_id(run_id: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]", "_", run_id.strip())
    if not value:
        raise ValueError("run_id cannot be empty")
    return value

