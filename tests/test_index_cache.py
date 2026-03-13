"""Tests for the disk-based index cache."""

from __future__ import annotations

from pathlib import Path

import pytest

from hammy.indexer.index_cache import cache_info, cache_path, load_index, save_index
from hammy.schema.models import Edge, EdgeMetadata, Location, Node, NodeMeta, NodeType, RelationType


def _make_node(name: str, file: str = "src/example.py") -> Node:
    return Node(
        id=Node.make_id(file, name),
        type=NodeType.FUNCTION,
        name=name,
        loc=Location(file=file, lines=(1, 10)),
        language="python",
        meta=NodeMeta(visibility="public", parameters=["x", "y"], return_type="bool"),
        summary="Does something useful",
    )


def _make_edge(src: Node, tgt: Node) -> Edge:
    return Edge(
        source=src.id,
        target=tgt.id,
        relation=RelationType.CALLS,
        metadata=EdgeMetadata(confidence=0.9, context="src.foo(tgt)"),
    )


class TestSaveAndLoad:
    def test_roundtrip_nodes_and_edges(self, tmp_path: Path) -> None:
        nodes = [_make_node("foo"), _make_node("bar", "src/other.py")]
        edges = [_make_edge(nodes[0], nodes[1])]

        save_index(tmp_path, nodes, edges)
        result = load_index(tmp_path)

        assert result is not None
        loaded_nodes, loaded_edges = result
        assert len(loaded_nodes) == 2
        assert len(loaded_edges) == 1
        assert loaded_nodes[0].name == "foo"
        assert loaded_nodes[1].name == "bar"
        assert loaded_edges[0].relation == RelationType.CALLS
        assert loaded_edges[0].metadata.context == "src.foo(tgt)"

    def test_preserves_all_node_fields(self, tmp_path: Path) -> None:
        node = _make_node("process_payment")
        save_index(tmp_path, [node], [])
        loaded_nodes, _ = load_index(tmp_path)

        n = loaded_nodes[0]
        assert n.meta.parameters == ["x", "y"]
        assert n.meta.return_type == "bool"
        assert n.meta.visibility == "public"
        assert n.summary == "Does something useful"

    def test_empty_index(self, tmp_path: Path) -> None:
        save_index(tmp_path, [], [])
        result = load_index(tmp_path)
        assert result is not None
        nodes, edges = result
        assert nodes == []
        assert edges == []

    def test_creates_hammy_dir(self, tmp_path: Path) -> None:
        save_index(tmp_path, [], [])
        assert (tmp_path / ".hammy").is_dir()
        assert (tmp_path / ".hammy" / "index.json").exists()

    def test_overwrites_existing_cache(self, tmp_path: Path) -> None:
        save_index(tmp_path, [_make_node("old")], [])
        save_index(tmp_path, [_make_node("new1"), _make_node("new2")], [])
        nodes, _ = load_index(tmp_path)
        assert len(nodes) == 2
        assert {n.name for n in nodes} == {"new1", "new2"}


class TestLoadMissing:
    def test_returns_none_when_no_cache(self, tmp_path: Path) -> None:
        assert load_index(tmp_path) is None

    def test_returns_none_on_corrupt_file(self, tmp_path: Path) -> None:
        path = cache_path(tmp_path)
        path.parent.mkdir()
        path.write_text("this is not json {{{")
        assert load_index(tmp_path) is None

    def test_returns_none_on_empty_file(self, tmp_path: Path) -> None:
        path = cache_path(tmp_path)
        path.parent.mkdir()
        path.write_text("")
        assert load_index(tmp_path) is None


class TestCacheInfo:
    def test_returns_metadata_without_full_load(self, tmp_path: Path) -> None:
        nodes = [_make_node("a"), _make_node("b")]
        edges = [_make_edge(nodes[0], nodes[1])]
        save_index(tmp_path, nodes, edges)

        info = cache_info(tmp_path)
        assert info is not None
        assert info["node_count"] == 2
        assert info["edge_count"] == 1
        assert "indexed_at" in info

    def test_returns_none_when_missing(self, tmp_path: Path) -> None:
        assert cache_info(tmp_path) is None
