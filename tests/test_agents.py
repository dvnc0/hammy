"""Tests for agent tools, bridge resolution, and context pack generation.

These tests cover the non-LLM parts of the agent system:
- Bridge tool (cross-language matching)
- Context pack Markdown generation
- Explorer and Historian tool creation
"""

from pathlib import Path

import pytest

from hammy.schema.models import (
    ContextPack,
    Edge,
    EdgeMetadata,
    Location,
    Node,
    NodeHistory,
    NodeMeta,
    NodeType,
    RelationType,
)
from hammy.tools.bridge import resolve_bridges


@pytest.fixture
def sample_nodes() -> list[Node]:
    """Create a set of nodes that should have bridge matches."""
    return [
        # PHP route endpoint (provider — defined by Route attribute)
        Node(
            id=Node.make_id("UserController.php", "endpoint:/api/v1/users"),
            type=NodeType.ENDPOINT,
            name="/api/v1/users",
            loc=Location(file="UserController.php", lines=(7, 7)),
            language="php",
        ),
        Node(
            id=Node.make_id("UserController.php", "endpoint:/api/v1/users/{id}/pay"),
            type=NodeType.ENDPOINT,
            name="/api/v1/users/{id}/pay",
            loc=Location(file="UserController.php", lines=(16, 16)),
            language="php",
        ),
        # JS fetch endpoint (consumer — from fetch/axios call)
        Node(
            id=Node.make_id("api.js", "endpoint:/api/v1/users"),
            type=NodeType.ENDPOINT,
            name="/api/v1/users",
            loc=Location(file="api.js", lines=(4, 4)),
            language="javascript",
        ),
        Node(
            id=Node.make_id("api.js", "endpoint:/api/v1/users/{id}/pay"),
            type=NodeType.ENDPOINT,
            name="/api/v1/users/{id}/pay",
            loc=Location(file="api.js", lines=(14, 14)),
            language="javascript",
        ),
        # Regular nodes (not endpoints)
        Node(
            id=Node.make_id("UserController.php", "App\\Controllers\\UserController"),
            type=NodeType.CLASS,
            name="App\\Controllers\\UserController",
            loc=Location(file="UserController.php", lines=(8, 27)),
            language="php",
        ),
        # JS function that makes the fetch calls
        Node(
            id=Node.make_id("api.js", "fetchUsers"),
            type=NodeType.FUNCTION,
            name="fetchUsers",
            loc=Location(file="api.js", lines=(3, 6)),
            language="javascript",
        ),
    ]


@pytest.fixture
def sample_edges(sample_nodes) -> list[Edge]:
    """Create edges that establish provider/consumer relationships for bridges."""
    return [
        # PHP class DEFINES its endpoint (provider)
        Edge(
            source=Node.make_id("UserController.php", "App\\Controllers\\UserController"),
            target=Node.make_id("UserController.php", "endpoint:/api/v1/users"),
            relation=RelationType.DEFINES,
        ),
        Edge(
            source=Node.make_id("UserController.php", "App\\Controllers\\UserController"),
            target=Node.make_id("UserController.php", "endpoint:/api/v1/users/{id}/pay"),
            relation=RelationType.DEFINES,
        ),
        # JS function NETWORKS_TO its endpoint (consumer)
        Edge(
            source=Node.make_id("api.js", "fetchUsers"),
            target=Node.make_id("api.js", "endpoint:/api/v1/users"),
            relation=RelationType.NETWORKS_TO,
            metadata=EdgeMetadata(is_bridge=True, context="fetch('/api/v1/users')"),
        ),
        Edge(
            source=Node.make_id("api.js", "fetchUsers"),
            target=Node.make_id("api.js", "endpoint:/api/v1/users/{id}/pay"),
            relation=RelationType.NETWORKS_TO,
            metadata=EdgeMetadata(is_bridge=True, context="fetch('/api/v1/users/{id}/pay')"),
        ),
    ]


class TestBridgeResolution:
    def test_exact_match(self, sample_nodes, sample_edges):
        bridges = resolve_bridges(sample_nodes, sample_edges)
        # /api/v1/users should match exactly
        exact_matches = [
            b for b in bridges
            if b.metadata.confidence == 1.0
        ]
        assert len(exact_matches) >= 1

    def test_wildcard_match(self, sample_nodes, sample_edges):
        bridges = resolve_bridges(sample_nodes, sample_edges)
        # /api/v1/users/{id}/pay should match with wildcard
        assert len(bridges) >= 2

    def test_bridge_metadata(self, sample_nodes, sample_edges):
        bridges = resolve_bridges(sample_nodes, sample_edges)
        for bridge in bridges:
            assert bridge.metadata.is_bridge is True
            assert bridge.metadata.confidence > 0.0
            assert bridge.relation == RelationType.NETWORKS_TO

    def test_no_bridges_same_language(self):
        """Nodes of the same language should not bridge to each other."""
        nodes = [
            Node(
                id="php1",
                type=NodeType.ENDPOINT,
                name="/api/users",
                loc=Location(file="a.php", lines=(1, 1)),
                language="php",
            ),
            Node(
                id="php2",
                type=NodeType.ENDPOINT,
                name="/api/users",
                loc=Location(file="b.php", lines=(1, 1)),
                language="php",
            ),
        ]
        bridges = resolve_bridges(nodes, [])
        assert len(bridges) == 0

    def test_no_bridges_no_endpoints(self):
        nodes = [
            Node(
                id="n1",
                type=NodeType.CLASS,
                name="Foo",
                loc=Location(file="a.php", lines=(1, 10)),
                language="php",
            ),
        ]
        bridges = resolve_bridges(nodes, [])
        assert len(bridges) == 0


class TestContextPackGeneration:
    def test_empty_pack(self):
        from hammy.core.context_pack import generate_context_pack_markdown

        pack = ContextPack(query="test query")
        md = generate_context_pack_markdown(pack)
        assert "test query" in md
        assert "# Hammy Context Pack" in md

    def test_pack_with_summary(self):
        from hammy.core.context_pack import generate_context_pack_markdown

        pack = ContextPack(
            query="Why is profile slow?",
            summary="The slowness is in UserRepository.",
        )
        md = generate_context_pack_markdown(pack)
        assert "UserRepository" in md
        assert "Summary" in md

    def test_pack_with_warnings(self):
        from hammy.core.context_pack import generate_context_pack_markdown

        pack = ContextPack(
            query="test",
            warnings=["High churn: 85% in 90 days", "Legacy code detected"],
        )
        md = generate_context_pack_markdown(pack)
        assert "High churn" in md
        assert "Legacy code" in md

    def test_pack_with_nodes(self):
        from hammy.core.context_pack import generate_context_pack_markdown

        pack = ContextPack(
            query="test",
            nodes=[
                Node(
                    id="n1",
                    type=NodeType.FUNCTION,
                    name="processPayment",
                    loc=Location(file="payment.php", lines=(10, 30)),
                    language="php",
                    meta=NodeMeta(visibility="public", return_type="void"),
                    history=NodeHistory(
                        churn_rate=15,
                        blame_owners=["Dan", "Alice"],
                    ),
                ),
            ],
        )
        md = generate_context_pack_markdown(pack)
        assert "processPayment" in md
        assert "payment.php" in md
        assert "public" in md or "visibility" in md
        assert "15 changes" in md
        assert "Dan" in md

    def test_pack_with_bridges(self):
        from hammy.core.context_pack import generate_context_pack_markdown

        pack = ContextPack(
            query="test",
            edges=[
                Edge(
                    source="js_endpoint",
                    target="php_endpoint",
                    relation=RelationType.NETWORKS_TO,
                    metadata=EdgeMetadata(
                        is_bridge=True,
                        confidence=0.95,
                        context="JS '/api/users' -> PHP '/api/users'",
                    ),
                ),
            ],
        )
        md = generate_context_pack_markdown(pack)
        assert "Cross-Language Bridges" in md
        assert "95%" in md


class TestExplorerTools:
    def test_creates_tools(self, tmp_path: Path):
        from hammy.agents.explorer import make_explorer_tools
        from hammy.tools.parser import ParserFactory

        factory = ParserFactory()
        tools = make_explorer_tools(tmp_path, factory, [], [])
        assert len(tools) == 6
        tool_names = [t.name for t in tools]
        assert "AST Query" in tool_names
        assert "Search Code Symbols" in tool_names
        assert "Lookup Symbol" in tool_names
        assert "Find Usages" in tool_names
        assert "Find Cross-Language Bridges" in tool_names
        assert "List Files" in tool_names


def _make_node(name: str, ntype: NodeType, file: str, language: str = "php") -> Node:
    return Node(
        id=Node.make_id(file, name),
        type=ntype,
        name=name,
        loc=Location(file=file, lines=(1, 10)),
        language=language,
    )


def _make_calls_edge(source_id: str, callee: str) -> Edge:
    return Edge(
        source=source_id,
        target=Node.make_id("", callee),
        relation=RelationType.CALLS,
        metadata=EdgeMetadata(confidence=0.8, context=callee),
    )


class TestSearchSymbolsRanking:
    """Verify ranked result ordering: exact > prefix > substring > summary."""

    def test_exact_floats_above_prefix(self, tmp_path: Path):
        from hammy.agents.explorer import make_explorer_tools
        from hammy.tools.parser import ParserFactory

        nodes = [
            _make_node("getRenewList", NodeType.METHOD, "a.php"),   # prefix match
            _make_node("getRenew", NodeType.METHOD, "b.php"),        # exact match
            _make_node("checkRenewStatus", NodeType.METHOD, "c.php"),  # substring
        ]
        tools = make_explorer_tools(tmp_path, ParserFactory(), nodes, [])
        search = next(t for t in tools if t.name == "Search Code Symbols")
        result = search.func(query="getRenew")

        lines = result.strip().splitlines()
        # First result should be the exact match
        assert "getRenew" in lines[0]
        assert "getRenewList" not in lines[0]

    def test_exact_case_insensitive(self, tmp_path: Path):
        from hammy.agents.explorer import make_explorer_tools
        from hammy.tools.parser import ParserFactory

        nodes = [_make_node("GetRenew", NodeType.METHOD, "a.php")]
        tools = make_explorer_tools(tmp_path, ParserFactory(), nodes, [])
        search = next(t for t in tools if t.name == "Search Code Symbols")
        result = search.func(query="getrenew")
        assert "GetRenew" in result

    def test_file_filter_narrows_results(self, tmp_path: Path):
        from hammy.agents.explorer import make_explorer_tools
        from hammy.tools.parser import ParserFactory

        nodes = [
            _make_node("save", NodeType.METHOD, "controllers/UserController.php"),
            _make_node("save", NodeType.METHOD, "models/Post.php"),
        ]
        tools = make_explorer_tools(tmp_path, ParserFactory(), nodes, [])
        search = next(t for t in tools if t.name == "Search Code Symbols")
        result = search.func(query="save", file_filter="controllers")
        assert "controllers" in result
        assert "models" not in result


class TestLookupSymbol:
    """Verify lookup_symbol exact definition lookup."""

    def test_exact_match_returned(self, tmp_path: Path):
        from hammy.agents.explorer import make_explorer_tools
        from hammy.tools.parser import ParserFactory

        nodes = [
            _make_node("getRenew", NodeType.METHOD, "Subscription.php"),
            _make_node("getRenewToken", NodeType.METHOD, "Token.php"),
        ]
        tools = make_explorer_tools(tmp_path, ParserFactory(), nodes, [])
        lookup = next(t for t in tools if t.name == "Lookup Symbol")
        result = lookup.func(name="getRenew")

        assert "getRenew" in result
        assert "Subscription.php" in result
        # Should not include getRenewToken
        assert "getRenewToken" not in result

    def test_shows_not_found_with_hint(self, tmp_path: Path):
        from hammy.agents.explorer import make_explorer_tools
        from hammy.tools.parser import ParserFactory

        tools = make_explorer_tools(tmp_path, ParserFactory(), [], [])
        lookup = next(t for t in tools if t.name == "Lookup Symbol")
        result = lookup.func(name="totallyMissing")
        assert "not found" in result.lower() or "search_symbols" in result

    def test_node_type_filter(self, tmp_path: Path):
        from hammy.agents.explorer import make_explorer_tools
        from hammy.tools.parser import ParserFactory

        nodes = [
            _make_node("process", NodeType.METHOD, "a.php"),
            _make_node("process", NodeType.FUNCTION, "b.php"),
        ]
        tools = make_explorer_tools(tmp_path, ParserFactory(), nodes, [])
        lookup = next(t for t in tools if t.name == "Lookup Symbol")
        result = lookup.func(name="process", node_type="function")
        assert "b.php" in result
        assert "a.php" not in result


class TestFindUsagesWordBoundary:
    """Verify find_usages uses word-boundary matching, not substring."""

    def test_exact_name_matches(self, tmp_path: Path):
        from hammy.agents.explorer import make_explorer_tools
        from hammy.tools.parser import ParserFactory

        caller = _make_node("myMethod", NodeType.METHOD, "a.php")
        edge = _make_calls_edge(caller.id, "save")
        nodes = [caller]
        tools = make_explorer_tools(tmp_path, ParserFactory(), nodes, [edge])
        find = next(t for t in tools if t.name == "Find Usages")
        result = find.func(symbol_name="save")
        assert "myMethod" in result

    def test_no_mid_word_match(self, tmp_path: Path):
        from hammy.agents.explorer import make_explorer_tools
        from hammy.tools.parser import ParserFactory

        caller = _make_node("someFunc", NodeType.FUNCTION, "a.php")
        # Edge calls "saveAll" — should NOT match a search for "save"
        edge = _make_calls_edge(caller.id, "saveAll")
        nodes = [caller]
        tools = make_explorer_tools(tmp_path, ParserFactory(), nodes, [edge])
        find = next(t for t in tools if t.name == "Find Usages")
        result = find.func(symbol_name="save")
        assert "No call sites" in result

    def test_file_filter(self, tmp_path: Path):
        from hammy.agents.explorer import make_explorer_tools
        from hammy.tools.parser import ParserFactory

        caller_ctrl = _make_node("ctrlMethod", NodeType.METHOD, "controllers/Ctrl.php")
        caller_model = _make_node("modelMethod", NodeType.METHOD, "models/Model.php")
        edges = [
            _make_calls_edge(caller_ctrl.id, "getRenew"),
            _make_calls_edge(caller_model.id, "getRenew"),
        ]
        nodes = [caller_ctrl, caller_model]
        tools = make_explorer_tools(tmp_path, ParserFactory(), nodes, edges)
        find = next(t for t in tools if t.name == "Find Usages")
        result = find.func(symbol_name="getRenew", file_filter="controllers")
        assert "ctrlMethod" in result
        assert "modelMethod" not in result


class TestHistorianTools:
    def test_creates_tools_with_vcs(self, tmp_path: Path):
        import subprocess

        from hammy.agents.historian import make_historian_tools
        from hammy.tools.vcs import VCSWrapper

        # Create a git repo
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
        subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=tmp_path, capture_output=True, check=True)
        subprocess.run(["git", "config", "user.name", "T"], cwd=tmp_path, capture_output=True, check=True)
        (tmp_path / "a.txt").write_text("hello")
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True, check=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True, check=True)

        vcs = VCSWrapper(tmp_path)
        tools = make_historian_tools(vcs)
        assert len(tools) == 4
        tool_names = [t.name for t in tools]
        assert "Git Log" in tool_names
        assert "Git Blame" in tool_names
        assert "File Churn Analysis" in tool_names
        assert "Search Commit History" in tool_names
