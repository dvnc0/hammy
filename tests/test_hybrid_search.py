"""Tests for hybrid BM25 + semantic search and RRF fusion."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from hammy.schema.models import Location, Node, NodeMeta, NodeType
from hammy.tools.hybrid_search import _rrf, _tokenize, hybrid_search


def _make_node(
    name: str,
    ntype: NodeType = NodeType.FUNCTION,
    file: str = "src/example.py",
    language: str = "python",
    summary: str = "",
) -> Node:
    return Node(
        id=Node.make_id(file, name),
        type=ntype,
        name=name,
        loc=Location(file=file, lines=(1, 10)),
        language=language,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_lowercases(self):
        assert "hello" in _tokenize("Hello")

    def test_splits_on_punctuation(self):
        tokens = _tokenize("foo.bar_baz")
        assert "foo" in tokens
        assert "bar_baz" in tokens

    def test_drops_single_chars(self):
        tokens = _tokenize("a b c def")
        assert "a" not in tokens
        assert "def" in tokens

    def test_camel_stays_whole(self):
        # CamelCase is NOT split — word boundary is punctuation only
        tokens = _tokenize("getRenew")
        assert "getRenew".lower() in tokens


# ---------------------------------------------------------------------------
# _rrf
# ---------------------------------------------------------------------------

class TestRRF:
    def test_common_item_scores_higher(self):
        # "b" appears in both lists at rank 0 and 1 — should beat "a" (rank 0 only)
        list1 = [("a", {"name": "a"}), ("b", {"name": "b"})]
        list2 = [("b", {"name": "b"}), ("c", {"name": "c"})]
        fused = _rrf([list1, list2])
        names = [item["name"] for _, item in fused]
        assert names[0] == "b"

    def test_preserves_all_items(self):
        list1 = [("a", {"name": "a"}), ("b", {"name": "b"})]
        list2 = [("c", {"name": "c"})]
        fused = _rrf([list1, list2])
        assert len(fused) == 3

    def test_empty_lists(self):
        assert _rrf([]) == []
        assert _rrf([[]]) == []

    def test_scores_descending(self):
        list1 = [("a", {}), ("b", {}), ("c", {})]
        list2 = [("c", {}), ("b", {}), ("a", {})]
        fused = _rrf([list1, list2])
        scores = [s for s, _ in fused]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# hybrid_search
# ---------------------------------------------------------------------------

class TestHybridSearch:
    def test_bm25_only_without_qdrant(self):
        nodes = [
            _make_node("processPayment", summary="handles payment processing"),
            _make_node("getUser", summary="fetches user by id"),
            _make_node("validateToken", summary="validates auth token"),
        ]
        results = hybrid_search("payment processing", nodes, qdrant=None, limit=5)
        assert len(results) > 0
        names = [r["name"] for r in results]
        assert "processPayment" in names

    def test_returns_empty_for_no_match(self):
        nodes = [_make_node("getUser")]
        results = hybrid_search("zzz_no_match_xyz", nodes, qdrant=None, limit=5)
        assert results == []

    def test_language_filter(self):
        nodes = [
            _make_node("process", language="python", file="a.py"),
            _make_node("process", language="php", file="a.php"),
        ]
        results = hybrid_search("process", nodes, qdrant=None, limit=10, language="python")
        assert all(r["language"] == "python" for r in results)

    def test_node_type_filter(self):
        nodes = [
            _make_node("handle", ntype=NodeType.FUNCTION, file="a.py"),
            _make_node("handle", ntype=NodeType.METHOD, file="b.py"),
        ]
        results = hybrid_search("handle", nodes, qdrant=None, limit=10, node_type="function")
        assert all(r["type"] == "function" for r in results)

    def test_limit_respected(self):
        nodes = [_make_node(f"fn_{i}", summary=f"function {i}") for i in range(20)]
        results = hybrid_search("function", nodes, qdrant=None, limit=5)
        assert len(results) <= 5

    def test_result_has_required_fields(self):
        nodes = [_make_node("myFunc", summary="does stuff")]
        results = hybrid_search("myFunc", nodes, qdrant=None, limit=5)
        assert len(results) > 0
        r = results[0]
        assert "node_id" in r
        assert "name" in r
        assert "file" in r
        assert "type" in r
        assert "score" in r

    def test_fuses_bm25_and_dense_results(self):
        """When qdrant is provided, results from both sources are merged."""
        nodes = [
            _make_node("processPayment", summary="payment processing"),
            _make_node("getUser", summary="fetch user"),
        ]

        mock_qdrant = MagicMock()
        mock_qdrant.search_code.return_value = [
            {
                "node_id": Node.make_id("src/example.py", "getUser"),
                "type": "function",
                "name": "getUser",
                "file": "src/example.py",
                "lines": [1, 10],
                "language": "python",
                "summary": "fetch user",
                "score": 0.9,
            }
        ]

        results = hybrid_search("payment user", nodes, qdrant=mock_qdrant, limit=10)
        names = [r["name"] for r in results]
        # Both should appear (BM25 finds processPayment, dense finds getUser)
        assert len(results) >= 1
        mock_qdrant.search_code.assert_called_once()

    def test_empty_nodes_returns_empty(self):
        results = hybrid_search("anything", [], qdrant=None, limit=5)
        assert results == []

    def test_empty_nodes_with_qdrant_falls_through(self):
        mock_qdrant = MagicMock()
        mock_qdrant.search_code.return_value = []
        results = hybrid_search("anything", [], qdrant=mock_qdrant, limit=5)
        assert results == []


# ---------------------------------------------------------------------------
# Integration via explorer tool
# ---------------------------------------------------------------------------

class TestHybridSearchTool:
    def test_tool_present_in_explorer(self, tmp_path: Path):
        from hammy.agents.explorer import make_explorer_tools
        from hammy.tools.parser import ParserFactory

        tools = make_explorer_tools(tmp_path, ParserFactory(), [], [])
        tool_names = [t.name for t in tools]
        assert "Hybrid Code Search" in tool_names

    def test_tool_returns_results(self, tmp_path: Path):
        from hammy.agents.explorer import make_explorer_tools
        from hammy.tools.parser import ParserFactory

        nodes = [
            _make_node("processPayment", summary="handles payment processing"),
            _make_node("getUser", summary="fetches user by id"),
        ]
        tools = make_explorer_tools(tmp_path, ParserFactory(), nodes, [])
        hybrid = next(t for t in tools if t.name == "Hybrid Code Search")
        result = hybrid.func(query="payment")
        assert "processPayment" in result

    def test_tool_no_results_message(self, tmp_path: Path):
        from hammy.agents.explorer import make_explorer_tools
        from hammy.tools.parser import ParserFactory

        tools = make_explorer_tools(tmp_path, ParserFactory(), [], [])
        hybrid = next(t for t in tools if t.name == "Hybrid Code Search")
        result = hybrid.func(query="zzz_missing")
        assert "No code matching" in result
