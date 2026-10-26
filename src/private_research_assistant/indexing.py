from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .config import create_embed_model, normalized_model_name, runtime_config_from_env
from .crawl import load_raw_documents, load_raw_manifest
from . import UserFacingError
from .paths import chroma_dir, ensure_data_dirs, latest_index_manifest_path, processed_run_dir


def build_index(run_id: str | None = None, base_dir: str | Path | None = None) -> dict[str, Any]:
    ensure_data_dirs(base_dir)
    manifest = load_raw_manifest(run_id, base_dir)
    run_id = manifest["run_id"]
    documents = load_raw_documents(run_id, base_dir)
    if not documents:
        raise UserFacingError("No raw documents found. Run `pra crawl` and make sure it saves at least one page.")

    from llama_index.core import Document, StorageContext, VectorStoreIndex
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.vector_stores.chroma import ChromaVectorStore
    import chromadb

    config = runtime_config_from_env(base_dir)
    embed_model = create_embed_model(config)
    parser = SentenceSplitter(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    llama_documents = [
        Document(
            text=document["markdown"],
            id_=document["doc_id"],
            metadata={
                "doc_id": document["doc_id"],
                "source_id": document["source_id"],
                "source_name": document["source_name"],
                "title": document["title"],
                "url": document["url"],
                "fetched_at": document["fetched_at"],
            },
        )
        for document in documents
    ]
    nodes = parser.get_nodes_from_documents(llama_documents)
    chunks = _assign_evidence_ids(nodes)

    processed_directory = processed_run_dir(run_id, base_dir)
    processed_directory.mkdir(parents=True, exist_ok=True)
    chunks_path = processed_directory / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    collection_name = _collection_name(run_id)
    chroma_client = chromadb.PersistentClient(path=str(chroma_dir(base_dir)))
    try:
        chroma_client.delete_collection(collection_name)
    except chromadb.errors.NotFoundError:
        pass
    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
        insert_batch_size=config.insert_batch_size,
    )

    index_manifest = {
        "run_id": run_id,
        "crawl_date": manifest.get("crawl_date"),
        "collection_name": collection_name,
        "chunk_count": len(chunks),
        "document_count": len(documents),
        "chunks_path": str(chunks_path),
        "chat_model": config.chat_model,
        "embed_model": config.embed_model,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
    }
    manifest_path = chroma_dir(base_dir) / f"{collection_name}.json"
    manifest_path.write_text(json.dumps(index_manifest, indent=2), encoding="utf-8")
    latest_index_manifest_path(base_dir).write_text(json.dumps(index_manifest, indent=2), encoding="utf-8")
    return index_manifest


def load_index_manifest(run_id: str | None = None, base_dir: str | Path | None = None) -> dict[str, Any]:
    """Read a run's index metadata, or the latest run when no ID is given."""
    if run_id:
        path = chroma_dir(base_dir) / f"{_collection_name(run_id)}.json"
    else:
        path = latest_index_manifest_path(base_dir)
    if not path.exists():
        raise UserFacingError(f"No index manifest found at {path}. Run `pra index` first.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_chunks(run_id: str | None = None, base_dir: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Read saved text as {evidence_id: chunk}, without calling a model."""
    manifest = load_index_manifest(run_id, base_dir)
    chunks_path = Path(manifest["chunks_path"])
    if not chunks_path.exists():
        chunks_path = processed_run_dir(manifest["run_id"], base_dir) / "chunks.jsonl"
    if not chunks_path.exists():
        raise UserFacingError(f"No chunks file found at {chunks_path}. Run `pra index` first.")
    chunks: dict[str, dict[str, Any]] = {}
    with chunks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            chunk = json.loads(line)
            chunks[chunk["evidence_id"]] = chunk
    return chunks


def load_vector_index(run_id: str | None = None, base_dir: str | Path | None = None):
    """Open a saved index only when its embedding model name matches settings."""
    manifest = load_index_manifest(run_id, base_dir)
    config = runtime_config_from_env(base_dir)
    saved_model = manifest.get("embed_model", "")
    if not isinstance(saved_model, str) or not saved_model.strip():
        raise UserFacingError("The saved index has no embedding model name. Rebuild it with `pra index`.")
    if normalized_model_name(saved_model) != normalized_model_name(config.embed_model):
        raise UserFacingError(
            f"Index uses {saved_model.strip()}, but PRA_EMBED_MODEL is {config.embed_model}. "
            "Use the original model in .env or rebuild with `pra index`. No data was changed."
        )

    from llama_index.core import StorageContext, VectorStoreIndex
    from llama_index.vector_stores.chroma import ChromaVectorStore
    import chromadb

    chroma_client = chromadb.PersistentClient(path=str(chroma_dir(base_dir)))
    chroma_collection = chroma_client.get_collection(manifest["collection_name"])
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
        embed_model=create_embed_model(config),
    )


def _assign_evidence_ids(nodes: list[Any]) -> list[dict[str, Any]]:
    counters: dict[str, int] = {}
    chunks: list[dict[str, Any]] = []
    for node in nodes:
        metadata = dict(node.metadata)
        doc_id = metadata["doc_id"]
        counters[doc_id] = counters.get(doc_id, 0) + 1
        evidence_id = f"{doc_id}#chunk-{counters[doc_id]}"
        node.id_ = evidence_id
        metadata["evidence_id"] = evidence_id
        node.metadata = metadata
        chunks.append(
            {
                "evidence_id": evidence_id,
                "doc_id": doc_id,
                "source_id": metadata.get("source_id", ""),
                "source_name": metadata.get("source_name", ""),
                "title": metadata.get("title", ""),
                "url": metadata.get("url", ""),
                "text": node.get_content(metadata_mode="none"),
            }
        )
    return chunks


def _collection_name(run_id: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_]", "_", run_id)
    return f"phd_ads_{clean}"
