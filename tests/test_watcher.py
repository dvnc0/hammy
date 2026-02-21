"""Tests for watch mode: incremental reindex, Qdrant delete-by-file, CLI."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hammy.schema.models import Edge, EdgeMetadata, Location, Node, NodeType, RelationType


def _make_node(name: str, file: str, language: str = "php") -> Node:
    return Node(
        id=Node.make_id(file, name),
        type=NodeType.FUNCTION,
        name=name,
        loc=Location(file=file, lines=(1, 10)),
        language=language,
    )


def _make_edge(source_id: str, target_id: str) -> Edge:
    return Edge(
        source=source_id,
        target=target_id,
        relation=RelationType.CALLS,
        metadata=EdgeMetadata(confidence=0.8, context="call"),
    )


# ---------------------------------------------------------------------------
# QdrantManager.delete_nodes_by_file
# ---------------------------------------------------------------------------

class TestDeleteNodesByFile:
    @pytest.fixture
    def qdrant_available(self):
        try:
            from qdrant_client import QdrantClient
            QdrantClient(host="localhost", port=6333, timeout=2).get_collections()
            return True
        except Exception:
            return False

    def test_delete_returns_zero_for_empty(self, qdrant_available):
        if not qdrant_available:
            pytest.skip("Qdrant not available")

        from hammy.config import QdrantConfig
        from hammy.tools.qdrant_tools import QdrantManager
        import uuid

        cfg = QdrantConfig(collection_prefix=f"test_del_{uuid.uuid4().hex[:8]}")
        mgr = QdrantManager(cfg)
        try:
            mgr.ensure_collections()
            deleted = mgr.delete_nodes_by_file("nonexistent/file.php")
            assert deleted == 0
        finally:
            mgr.delete_collections()

    def test_delete_removes_correct_nodes(self, qdrant_available):
        if not qdrant_available:
            pytest.skip("Qdrant not available")

        from hammy.config import QdrantConfig
        from hammy.tools.qdrant_tools import QdrantManager
        import uuid

        cfg = QdrantConfig(collection_prefix=f"test_del_{uuid.uuid4().hex[:8]}")
        mgr = QdrantManager(cfg)
        try:
            mgr.ensure_collections()
            nodes = [
                _make_node("funcA", "a.php"),
                _make_node("funcB", "a.php"),
                _make_node("funcC", "b.php"),
            ]
            mgr.upsert_nodes(nodes)
            time.sleep(0.2)  # allow Qdrant to flush

            deleted = mgr.delete_nodes_by_file("a.php")
            assert deleted == 2

            # b.php nodes should still be searchable
            results = mgr.search_code("funcC", limit=5)
            assert any(r["name"] == "funcC" for r in results)
        finally:
            mgr.delete_collections()


# ---------------------------------------------------------------------------
# index_files incremental function
# ---------------------------------------------------------------------------

class TestIndexFiles:
    def test_indexes_single_file(self, tmp_path: Path):
        from hammy.config import HammyConfig
        from hammy.indexer.code_indexer import index_files

        php_file = tmp_path / "foo.php"
        php_file.write_text("<?php\nfunction fooBar() {}\n")

        config_file = tmp_path / "hammy.yaml"
        config_file.write_text("project:\n  name: test\nparsing:\n  languages:\n    - php\n")
        config = HammyConfig.load(tmp_path)

        nodes, edges, errors = index_files([php_file], config, tmp_path)
        assert any(n.name == "fooBar" for n in nodes)
        assert errors == []

    def test_skips_nonexistent_files(self, tmp_path: Path):
        from hammy.config import HammyConfig
        from hammy.indexer.code_indexer import index_files

        config_file = tmp_path / "hammy.yaml"
        config_file.write_text("project:\n  name: test\nparsing:\n  languages:\n    - php\n")
        config = HammyConfig.load(tmp_path)

        nodes, edges, errors = index_files(
            [tmp_path / "does_not_exist.php"], config, tmp_path
        )
        assert nodes == []
        assert errors == []

    def test_skips_unsupported_extension(self, tmp_path: Path):
        from hammy.config import HammyConfig
        from hammy.indexer.code_indexer import index_files

        txt_file = tmp_path / "readme.txt"
        txt_file.write_text("just text")

        config_file = tmp_path / "hammy.yaml"
        config_file.write_text("project:\n  name: test\nparsing:\n  languages:\n    - php\n")
        config = HammyConfig.load(tmp_path)

        nodes, edges, errors = index_files([txt_file], config, tmp_path)
        assert nodes == []


# ---------------------------------------------------------------------------
# process_changed_files — core incremental logic (no filesystem events needed)
# ---------------------------------------------------------------------------

class TestProcessChangedFiles:
    def _config(self, tmp_path: Path):
        from hammy.config import HammyConfig
        (tmp_path / "hammy.yaml").write_text(
            "project:\n  name: test\nparsing:\n  languages:\n    - php\n"
        )
        return HammyConfig.load(tmp_path)

    def test_adds_new_file_nodes(self, tmp_path: Path):
        from collections import defaultdict
        from hammy.watcher import process_changed_files

        config = self._config(tmp_path)
        php = tmp_path / "new.php"
        php.write_text("<?php\nfunction newFunc() {}\n")

        all_nodes: list[Node] = []
        all_edges: list[Edge] = []
        file_node_ids: dict = defaultdict(list)
        changes: list[dict] = []

        process_changed_files(
            {php}, tmp_path, config, all_nodes, all_edges, file_node_ids,
            on_change=lambda t, a, r, e: changes.append({"added": a}),
        )

        assert any(n.name == "newFunc" for n in all_nodes)
        assert len(changes) == 1
        assert changes[0]["added"] > 0

    def test_replaces_modified_file_nodes(self, tmp_path: Path):
        from collections import defaultdict
        from hammy.watcher import process_changed_files

        config = self._config(tmp_path)
        php = tmp_path / "mod.php"
        php.write_text("<?php\nfunction original() {}\n")

        # Seed with the old node
        old_node = _make_node("original", "mod.php")
        all_nodes: list[Node] = [old_node]
        all_edges: list[Edge] = []
        file_node_ids: dict = defaultdict(list, {"mod.php": [old_node.id]})

        # Now "modify" the file
        php.write_text("<?php\nfunction replaced() {}\n")

        process_changed_files(
            {php}, tmp_path, config, all_nodes, all_edges, file_node_ids,
        )

        names = [n.name for n in all_nodes]
        assert "original" not in names
        assert "replaced" in names

    def test_removes_deleted_file_nodes(self, tmp_path: Path):
        from collections import defaultdict
        from hammy.watcher import process_changed_files

        config = self._config(tmp_path)
        # Don't create the file — simulate deletion
        deleted_path = tmp_path / "gone.php"

        gone_node = _make_node("goneFunc", "gone.php")
        all_nodes: list[Node] = [gone_node]
        all_edges: list[Edge] = []
        file_node_ids: dict = defaultdict(list, {"gone.php": [gone_node.id]})
        changes: list[dict] = []

        process_changed_files(
            {deleted_path}, tmp_path, config, all_nodes, all_edges, file_node_ids,
            on_change=lambda t, a, r, e: changes.append({"event": t, "removed": r}),
        )

        assert not any(n.name == "goneFunc" for n in all_nodes)
        assert changes[0]["event"] == "deleted"
        assert changes[0]["removed"] == 1

    def test_prunes_edges_for_removed_nodes(self, tmp_path: Path):
        from collections import defaultdict
        from hammy.watcher import process_changed_files

        config = self._config(tmp_path)
        deleted_path = tmp_path / "gone.php"

        gone_node = _make_node("goneFunc", "gone.php")
        other_node = _make_node("otherFunc", "keep.php")
        edge = _make_edge(gone_node.id, other_node.id)

        all_nodes: list[Node] = [gone_node, other_node]
        all_edges: list[Edge] = [edge]
        file_node_ids: dict = defaultdict(list, {
            "gone.php": [gone_node.id],
            "keep.php": [other_node.id],
        })

        process_changed_files(
            {deleted_path}, tmp_path, config, all_nodes, all_edges, file_node_ids,
        )

        assert len(all_edges) == 0  # edge was pruned
        assert any(n.name == "otherFunc" for n in all_nodes)  # keep.php node intact

    def test_on_change_callback(self, tmp_path: Path):
        from collections import defaultdict
        from hammy.watcher import process_changed_files

        config = self._config(tmp_path)
        php = tmp_path / "cb.php"
        php.write_text("<?php\nfunction cbFunc() {}\n")

        events: list[tuple] = []
        process_changed_files(
            {php}, tmp_path, config, [], [], defaultdict(list),
            on_change=lambda t, a, r, e: events.append((t, a, r, e)),
        )
        assert len(events) == 1
        assert events[0][1] > 0  # added count > 0


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------

class TestWatchCLI:
    def test_watch_help(self):
        from typer.testing import CliRunner
        from hammy.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["watch", "--help"])
        assert result.exit_code == 0
        assert "watch" in result.output.lower() or "Watch" in result.output
