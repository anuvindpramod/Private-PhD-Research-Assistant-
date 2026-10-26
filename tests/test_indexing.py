import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from private_research_assistant.config import RuntimeConfig
from private_research_assistant.indexing import build_index, load_chunks, load_vector_index
from private_research_assistant.retrieval import _to_evidence


class IndexIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import chromadb
            from llama_index.core.embeddings import MockEmbedding
            from llama_index.vector_stores.chroma import ChromaVectorStore
        except ModuleNotFoundError as error:
            raise unittest.SkipTest("Install project dependencies to run index integration tests.") from error
        cls.mock_embedding = MockEmbedding

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.document = {
            "doc_id": "demo", "source_id": "jobs_ac_uk", "source_name": "Example",
            "title": "PhD in Machine Learning", "url": "https://example.org/phd",
            "markdown": "A PhD in Machine Learning at Example University.", "fetched_at": "2026-09-05",
        }
        self.manifest = {"run_id": "test", "crawl_date": "2026-09-05"}
        for target, value in (
            ("load_raw_manifest", self.manifest),
            ("load_raw_documents", [self.document]),
            ("runtime_config_from_env", RuntimeConfig()),
            ("create_embed_model", self.mock_embedding(embed_dim=8)),
        ):
            replacement = patch(f"private_research_assistant.indexing.{target}", return_value=value)
            replacement.start()
            self.addCleanup(replacement.stop)
        environment = patch.dict(os.environ, {"ANONYMIZED_TELEMETRY": "False"})
        environment.start()
        self.addCleanup(environment.stop)

    def test_build_load_retrieve_and_rebuild_in_temporary_database(self):
        manifest = build_index(base_dir=self.root)
        self.assertEqual(manifest["chunk_count"], 1)
        self.assertEqual(load_chunks(base_dir=self.root)["demo#chunk-1"]["text"], self.document["markdown"])
        index = load_vector_index(base_dir=self.root)
        retrieved = index.as_retriever(similarity_top_k=1).retrieve("machine learning")
        evidence = _to_evidence(retrieved[0])
        self.assertEqual(evidence.evidence_id, "demo#chunk-1")
        self.assertEqual(evidence.url, self.document["url"])
        self.assertEqual(evidence.text, self.document["markdown"])
        self.document["markdown"] = "Updated machine learning advert."
        build_index(base_dir=self.root)
        self.assertEqual(load_chunks(base_dir=self.root)["demo#chunk-1"]["text"], self.document["markdown"])

    def test_collection_delete_failure_stops_indexing(self):
        client = Mock()
        client.delete_collection.side_effect = RuntimeError("storage failed")
        with patch("chromadb.PersistentClient", return_value=client):
            with self.assertRaisesRegex(RuntimeError, "storage failed"):
                build_index(base_dir=self.root)
        client.get_or_create_collection.assert_not_called()
