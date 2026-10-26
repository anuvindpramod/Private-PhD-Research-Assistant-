"""Settings checks use temporary files and fake Ollama responses, not real models."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from private_research_assistant.config import (
    RuntimeConfig,
    normalized_model_name,
    runtime_config_from_env,
)
from private_research_assistant.env import get_firecrawl_api_key, load_env_file
from private_research_assistant.exceptions import UserFacingError
from private_research_assistant.indexing import load_vector_index
from private_research_assistant.setup_check import _check_ollama_models


class ConfigTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def test_defaults_without_env_file(self):
        self.assertEqual(runtime_config_from_env(self.root), RuntimeConfig())
        self.assertEqual(RuntimeConfig().embed_model, "qwen3-embedding:0.6b")

    def test_file_overrides_defaults_and_strips_spaces(self):
        (self.root / ".env").write_text(
            'PRA_CHAT_MODEL=" custom-chat:1b "\n'
            'PRA_EMBED_MODEL=" custom-embed:small "\n'
            'PRA_OLLAMA_BASE_URL=http://localhost:11435\n'
        )
        config = runtime_config_from_env(self.root)
        self.assertEqual(config.chat_model, "custom-chat:1b")
        self.assertEqual(config.embed_model, "custom-embed:small")
        self.assertEqual(config.ollama_base_url, "http://localhost:11435")

    def test_process_environment_overrides_file(self):
        (self.root / ".env").write_text("PRA_EMBED_MODEL=file-model\n")
        os.environ["PRA_EMBED_MODEL"] = "shell-model"
        self.assertEqual(runtime_config_from_env(self.root).embed_model, "shell-model")

    def test_blank_settings_are_errors_not_silent_fallbacks(self):
        for name in ("PRA_CHAT_MODEL", "PRA_EMBED_MODEL", "PRA_OLLAMA_BASE_URL"):
            with self.subTest(name=name), patch.dict(os.environ, {name: "  "}):
                with self.assertRaisesRegex(UserFacingError, name):
                    runtime_config_from_env(self.root)
        (self.root / ".env").write_text("PRA_EMBED_MODEL=\n")
        with self.assertRaisesRegex(UserFacingError, "PRA_EMBED_MODEL"):
            runtime_config_from_env(self.root)

    def test_file_reads_do_not_change_environment_or_cache_old_values(self):
        path = self.root / ".env"
        path.write_text("PRA_EMBED_MODEL=first\n")
        self.assertEqual(runtime_config_from_env(self.root).embed_model, "first")
        path.write_text("PRA_EMBED_MODEL=second\n")
        self.assertEqual(runtime_config_from_env(self.root).embed_model, "second")
        self.assertNotIn("PRA_EMBED_MODEL", os.environ)

    def test_simple_env_file_syntax(self):
        (self.root / ".env").write_text("# comment\n\ninvalid line\n=value\n KEY = 'a=b'\n")
        self.assertEqual(load_env_file(self.root), {"KEY": "a=b"})
        self.assertNotIn("KEY", os.environ)

    def test_api_key_precedence_without_environment_mutation(self):
        (self.root / ".env").write_text("FIRECRAWL_API_KEY=fc-file-test\n")
        self.assertEqual(get_firecrawl_api_key(self.root), "fc-file-test")
        self.assertNotIn("FIRECRAWL_API_KEY", os.environ)
        os.environ["FIRECRAWL_API_KEY"] = "fc-shell-test"
        self.assertEqual(get_firecrawl_api_key(self.root), "fc-shell-test")

    def test_missing_blank_and_placeholder_keys(self):
        self.assertIsNone(get_firecrawl_api_key(self.root))
        for value in ("", "  ", "fc-YOUR-API-KEY"):
            with self.subTest(value=value):
                (self.root / ".env").write_text(f"FIRECRAWL_API_KEY={value}\n")
                self.assertIsNone(get_firecrawl_api_key(self.root))

    def test_model_name_normalization_preserves_other_tags(self):
        self.assertEqual(normalized_model_name(" model "), "model:latest")
        self.assertEqual(normalized_model_name("model:latest"), "model:latest")
        self.assertEqual(normalized_model_name("model:0.6b "), "model:0.6b")
        self.assertEqual(normalized_model_name("host:5000/team/model"), "host:5000/team/model:latest")
        self.assertEqual(normalized_model_name("  "), "")


class SetupModelTests(unittest.TestCase):
    def test_selected_models_and_server_are_checked(self):
        config = RuntimeConfig(chat_model="my-chat", embed_model="my-embed:small",
                               ollama_base_url="http://localhost:11435")
        response = subprocess.CompletedProcess([], 0,
            "NAME ID SIZE MODIFIED\nmy-chat:latest abc 1GB now\nmy-embed:small def 1GB now\n", "")
        with patch("private_research_assistant.setup_check.subprocess.run", return_value=response) as run:
            server, models = _check_ollama_models(config)
        self.assertTrue(server.passing)
        self.assertTrue(all(model.passing for model in models))
        self.assertEqual([model.name for model in models], ["my-chat", "my-embed:small"])
        self.assertEqual(run.call_args.kwargs["env"]["OLLAMA_HOST"], config.ollama_base_url)

    def test_similar_model_names_are_not_matches(self):
        response = subprocess.CompletedProcess([], 0,
            "NAME ID SIZE MODIFIED\nqwen3:1.7b-extra abc 1GB now\nqwen3-embedding:0.6b-extra def 1GB now\n", "")
        with patch("private_research_assistant.setup_check.subprocess.run", return_value=response):
            server, models = _check_ollama_models(RuntimeConfig())
        self.assertTrue(server.passing)
        self.assertTrue(all(not model.passing for model in models))

    def test_server_failure_reports_selected_models(self):
        config = RuntimeConfig(chat_model="chosen-chat", embed_model="chosen-embed")
        response = subprocess.CompletedProcess([], 1, "", "Server unavailable")
        with patch("private_research_assistant.setup_check.subprocess.run", return_value=response):
            server, models = _check_ollama_models(config)
        self.assertFalse(server.passing)
        self.assertEqual([model.name for model in models], ["chosen-chat", "chosen-embed"])
        self.assertTrue(all(not model.passing for model in models))

    def test_missing_cli_and_timeout_are_readable_failures(self):
        for error in (FileNotFoundError(), subprocess.TimeoutExpired("ollama", 20)):
            with self.subTest(error=type(error).__name__):
                with patch("private_research_assistant.setup_check.subprocess.run", side_effect=error):
                    server, models = _check_ollama_models(RuntimeConfig())
                self.assertFalse(server.passing)
                self.assertTrue(server.detail)
                self.assertEqual(models, [])


class IndexModelGuardTests(unittest.TestCase):
    def setUp(self):
        self.chroma = MagicMock()
        self.core = MagicMock()
        modules = patch.dict(sys.modules, {
            "chromadb": self.chroma,
            "llama_index.core": self.core,
            "llama_index.vector_stores.chroma": MagicMock(),
        })
        modules.start()
        self.addCleanup(modules.stop)

    def test_different_or_missing_embedding_name_stops_before_database_access(self):
        for saved_model in ("nomic-embed-text", "", None):
            with self.subTest(saved_model=saved_model):
                with patch("private_research_assistant.indexing.load_index_manifest",
                           return_value={"embed_model": saved_model}), \
                     patch("private_research_assistant.indexing.runtime_config_from_env", return_value=RuntimeConfig()):
                    with self.assertRaises(UserFacingError):
                        load_vector_index()
                    self.chroma.PersistentClient.assert_not_called()

    def test_spaces_and_latest_alias_pass_the_guard(self):
        for saved, configured in (("qwen3-embedding:0.6b ", "qwen3-embedding:0.6b"),
                                  ("embed:latest", "embed")):
            with self.subTest(saved=saved):
                with patch("private_research_assistant.indexing.load_index_manifest", return_value={
                         "embed_model": saved, "collection_name": "test_collection",
                     }), \
                     patch("private_research_assistant.indexing.runtime_config_from_env",
                           return_value=RuntimeConfig(embed_model=configured)) as settings, \
                     patch("private_research_assistant.indexing.create_embed_model"), \
                     patch("private_research_assistant.indexing.chroma_dir", return_value="unused"):
                    load_vector_index(base_dir="selected-project")
                    settings.assert_called_once_with("selected-project")
                    self.core.VectorStoreIndex.from_vector_store.assert_called()

    def test_database_failure_is_not_relabelled_as_missing_collection(self):
        self.chroma.PersistentClient.return_value.get_collection.side_effect = RuntimeError("database damaged")
        with (
            patch("private_research_assistant.indexing.load_index_manifest", return_value={
                "embed_model": "qwen3-embedding:0.6b", "collection_name": "test_collection",
            }),
            patch("private_research_assistant.indexing.runtime_config_from_env", return_value=RuntimeConfig()),
        ):
            with self.assertRaisesRegex(RuntimeError, "database damaged"):
                load_vector_index()
