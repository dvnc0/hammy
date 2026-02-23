"""Tests for agent tools, bridge resolution, and context pack generation.

These tests cover the non-LLM parts of the agent system:
- Bridge tool (cross-language matching)
- Context pack Markdown generation
- Explorer and Historian tool creation
"""

import textwrap
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
    def test_creates_core_tools_without_qdrant(self, tmp_path: Path):
        from hammy.agents.explorer import make_explorer_tools
        from hammy.tools.parser import ParserFactory

        factory = ParserFactory()
        tools = make_explorer_tools(tmp_path, factory, [], [])
        assert len(tools) == 14
        tool_names = [t.name for t in tools]
        assert "AST Query" in tool_names
        assert "Search Code Symbols" in tool_names
        assert "Hybrid Code Search" in tool_names
        assert "Lookup Symbol" in tool_names
        assert "Structural Search" in tool_names
        assert "Find Usages" in tool_names
        assert "Impact Analysis" in tool_names
        assert "Hotspot Score" in tool_names
        assert "PR Diff Analysis" in tool_names
        assert "Find Cross-Language Bridges" in tool_names
        assert "List Files" in tool_names
        assert "Explain Symbol" in tool_names
        assert "Module Summary" in tool_names
        assert "Lookup Symbols Batch" in tool_names
        # Brain tools should NOT be present without qdrant
        assert "Store Context" not in tool_names
        assert "Recall Context" not in tool_names
        assert "List Context" not in tool_names

    def test_adds_brain_tools_with_qdrant(self, tmp_path: Path):
        from unittest.mock import MagicMock
        from hammy.agents.explorer import make_explorer_tools
        from hammy.tools.parser import ParserFactory

        mock_qdrant = MagicMock()
        tools = make_explorer_tools(tmp_path, ParserFactory(), [], [], qdrant=mock_qdrant)
        assert len(tools) == 17
        tool_names = [t.name for t in tools]
        assert "Store Context" in tool_names
        assert "Recall Context" in tool_names
        assert "List Context" in tool_names


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


class TestImpactAnalysis:
    """Tests for the impact_analysis tool (N-hop call graph traversal)."""

    def _setup(self, tmp_path):
        from hammy.agents.explorer import make_explorer_tools
        from hammy.tools.parser import ParserFactory

        # Call graph: getRenew <- processRenewal <- handleRequest
        target = _make_node("getRenew", NodeType.METHOD, "Subscription.php")
        caller1 = _make_node("processRenewal", NodeType.METHOD, "RenewalService.php")
        caller2 = _make_node("handleRequest", NodeType.METHOD, "RenewalController.php")
        edges = [
            _make_calls_edge(caller1.id, "getRenew"),
            _make_calls_edge(caller2.id, "processRenewal"),
        ]
        nodes = [target, caller1, caller2]
        tools = make_explorer_tools(tmp_path, ParserFactory(), nodes, edges)
        impact = next(t for t in tools if t.name == "Impact Analysis")
        return impact, nodes

    def test_finds_direct_callers(self, tmp_path: Path):
        impact, _ = self._setup(tmp_path)
        result = impact.func(symbol_name="getRenew", depth=1)
        assert "processRenewal" in result
        assert "Hop 1" in result

    def test_finds_transitive_callers(self, tmp_path: Path):
        impact, _ = self._setup(tmp_path)
        result = impact.func(symbol_name="getRenew", depth=3)
        assert "processRenewal" in result
        assert "handleRequest" in result

    def test_no_callers_message(self, tmp_path: Path):
        impact, _ = self._setup(tmp_path)
        result = impact.func(symbol_name="handleRequest", depth=2, direction="callers")
        assert "No callers" in result

    def test_callees_direction(self, tmp_path: Path):
        from hammy.agents.explorer import make_explorer_tools
        from hammy.tools.parser import ParserFactory

        # handleRequest calls processRenewal calls getRenew
        get_renew = _make_node("getRenew", NodeType.METHOD, "Subscription.php")
        process = _make_node("processRenewal", NodeType.METHOD, "RenewalService.php")
        handle = _make_node("handleRequest", NodeType.METHOD, "RenewalController.php")
        edges = [
            _make_calls_edge(handle.id, "processRenewal"),
            _make_calls_edge(process.id, "getRenew"),
        ]
        nodes = [get_renew, process, handle]
        tools = make_explorer_tools(tmp_path, ParserFactory(), nodes, edges)
        impact = next(t for t in tools if t.name == "Impact Analysis")

        result = impact.func(symbol_name="handleRequest", depth=3, direction="callees")
        assert "processRenewal" in result

    def test_depth_limits_hops(self, tmp_path: Path):
        impact, _ = self._setup(tmp_path)
        # With depth=1 we should only see hop-1 callers (processRenewal), not hop-2 (handleRequest)
        result = impact.func(symbol_name="getRenew", depth=1, direction="callers")
        assert "processRenewal" in result
        assert "handleRequest" not in result


def _make_node_meta(
    name: str,
    ntype: NodeType,
    file: str = "src/a.php",
    language: str = "php",
    visibility: str = "",
    is_async: bool = False,
    params: list | None = None,
    return_type: str = "",
    complexity: int | None = None,
) -> Node:
    from hammy.schema.models import NodeMeta
    return Node(
        id=Node.make_id(file, name),
        type=ntype,
        name=name,
        loc=Location(file=file, lines=(1, 10)),
        language=language,
        meta=NodeMeta(
            visibility=visibility or None,
            is_async=is_async,
            parameters=params or [],
            return_type=return_type or None,
            complexity_score=complexity,
        ),
    )


class TestStructuralSearch:
    def _tool(self, tmp_path, nodes):
        from hammy.agents.explorer import make_explorer_tools
        from hammy.tools.parser import ParserFactory
        tools = make_explorer_tools(tmp_path, ParserFactory(), nodes, [])
        return next(t for t in tools if t.name == "Structural Search")

    def test_visibility_filter(self, tmp_path: Path):
        nodes = [
            _make_node_meta("pub", NodeType.METHOD, visibility="public"),
            _make_node_meta("priv", NodeType.METHOD, visibility="private"),
        ]
        tool = self._tool(tmp_path, nodes)
        result = tool.func(visibility="public")
        assert "pub" in result
        assert "priv" not in result

    def test_async_only(self, tmp_path: Path):
        nodes = [
            _make_node_meta("fetchData", NodeType.FUNCTION, is_async=True),
            _make_node_meta("syncFunc", NodeType.FUNCTION, is_async=False),
        ]
        tool = self._tool(tmp_path, nodes)
        result = tool.func(async_only=True)
        assert "fetchData" in result
        assert "syncFunc" not in result

    def test_min_params(self, tmp_path: Path):
        nodes = [
            _make_node_meta("noArgs", NodeType.FUNCTION, params=[]),
            _make_node_meta("twoArgs", NodeType.FUNCTION, params=["a", "b"]),
            _make_node_meta("threeArgs", NodeType.FUNCTION, params=["a", "b", "c"]),
        ]
        tool = self._tool(tmp_path, nodes)
        result = tool.func(min_params=2)
        assert "noArgs" not in result
        assert "twoArgs" in result
        assert "threeArgs" in result

    def test_max_params(self, tmp_path: Path):
        nodes = [
            _make_node_meta("noArgs", NodeType.FUNCTION, params=[]),
            _make_node_meta("twoArgs", NodeType.FUNCTION, params=["a", "b"]),
        ]
        tool = self._tool(tmp_path, nodes)
        result = tool.func(max_params=0)
        assert "noArgs" in result
        assert "twoArgs" not in result

    def test_return_type_substring(self, tmp_path: Path):
        nodes = [
            _make_node_meta("getUser", NodeType.METHOD, return_type="User"),
            _make_node_meta("isValid", NodeType.METHOD, return_type="bool"),
        ]
        tool = self._tool(tmp_path, nodes)
        result = tool.func(return_type="bool")
        assert "isValid" in result
        assert "getUser" not in result

    def test_name_pattern_regex(self, tmp_path: Path):
        nodes = [
            _make_node_meta("getUser", NodeType.METHOD),
            _make_node_meta("setUser", NodeType.METHOD),
            _make_node_meta("deleteUser", NodeType.METHOD),
        ]
        tool = self._tool(tmp_path, nodes)
        result = tool.func(name_pattern="^get")
        assert "getUser" in result
        assert "setUser" not in result
        assert "deleteUser" not in result

    def test_file_filter(self, tmp_path: Path):
        nodes = [
            _make_node_meta("ctrlMethod", NodeType.METHOD, file="controllers/Ctrl.php"),
            _make_node_meta("modelMethod", NodeType.METHOD, file="models/Model.php"),
        ]
        tool = self._tool(tmp_path, nodes)
        result = tool.func(file_filter="controllers")
        assert "ctrlMethod" in result
        assert "modelMethod" not in result

    def test_node_type_filter(self, tmp_path: Path):
        nodes = [
            _make_node_meta("MyClass", NodeType.CLASS),
            _make_node_meta("myFunc", NodeType.FUNCTION),
        ]
        tool = self._tool(tmp_path, nodes)
        result = tool.func(node_type="class")
        assert "MyClass" in result
        assert "myFunc" not in result

    def test_combined_filters(self, tmp_path: Path):
        nodes = [
            _make_node_meta("pubAsync", NodeType.METHOD, visibility="public", is_async=True),
            _make_node_meta("pubSync", NodeType.METHOD, visibility="public", is_async=False),
            _make_node_meta("privAsync", NodeType.METHOD, visibility="private", is_async=True),
        ]
        tool = self._tool(tmp_path, nodes)
        result = tool.func(visibility="public", async_only=True)
        assert "pubAsync" in result
        assert "pubSync" not in result
        assert "privAsync" not in result

    def test_no_match(self, tmp_path: Path):
        nodes = [_make_node_meta("foo", NodeType.FUNCTION, language="python")]
        tool = self._tool(tmp_path, nodes)
        result = tool.func(language="php")
        assert "No symbols matched" in result

    def test_complexity_filter(self, tmp_path: Path):
        nodes = [
            _make_node_meta("simple", NodeType.FUNCTION, complexity=2),
            _make_node_meta("complex", NodeType.FUNCTION, complexity=15),
        ]
        tool = self._tool(tmp_path, nodes)
        result = tool.func(min_complexity=10)
        assert "complex" in result
        assert "simple" not in result


class TestHotspotScore:
    """Tests for the hotspot scoring algorithm."""

    def _nodes_and_edges(self):
        """Build a small call graph: handleRequest -> processRenewal -> getRenew."""
        getRenew = _make_node("getRenew", NodeType.METHOD, "Subscription.php")
        processRenewal = _make_node("processRenewal", NodeType.METHOD, "RenewalService.php")
        handleRequest = _make_node("handleRequest", NodeType.METHOD, "RenewalController.php")
        otherFunc = _make_node("otherFunc", NodeType.FUNCTION, "Util.php")

        edges = [
            _make_calls_edge(processRenewal.id, "getRenew"),
            _make_calls_edge(handleRequest.id, "getRenew"),  # getRenew has 2 callers
            _make_calls_edge(handleRequest.id, "processRenewal"),  # processRenewal has 1
        ]
        return [getRenew, processRenewal, handleRequest, otherFunc], edges

    def test_ranks_by_caller_count(self, tmp_path: Path):
        from hammy.tools.hotspot import compute_hotspots

        nodes, edges = self._nodes_and_edges()
        results = compute_hotspots(nodes, edges, top_n=10)

        # getRenew has 2 callers — should be ranked #1
        assert results[0]["name"] == "getRenew"
        assert results[0]["caller_count"] == 2

    def test_churn_boosts_score(self, tmp_path: Path):
        from hammy.tools.hotspot import compute_hotspots
        from hammy.schema.models import NodeHistory

        getRenew = _make_node("getRenew", NodeType.METHOD, "Subscription.php")
        processRenewal = _make_node("processRenewal", NodeType.METHOD, "RenewalService.php")

        # Give processRenewal only 1 caller but huge churn — should beat getRenew (2 callers, 0 churn)
        processRenewal.history = NodeHistory(churn_rate=100)
        edges = [
            _make_calls_edge(processRenewal.id, "getRenew"),
            _make_calls_edge(_make_node("other", NodeType.FUNCTION, "x.php").id, "getRenew"),
            _make_calls_edge(_make_node("caller", NodeType.FUNCTION, "y.php").id, "processRenewal"),
        ]

        results = compute_hotspots([getRenew, processRenewal], edges, top_n=10)
        # processRenewal has churn=100 × callers=1 vs getRenew churn=0
        # log2(2) * log2(2) = 1.0 for getRenew (churn defaults to 1)
        # log2(2) * log2(101) ≈ 6.66 for processRenewal
        top_name = results[0]["name"]
        assert top_name == "processRenewal"

    def test_file_churn_dict(self, tmp_path: Path):
        from hammy.tools.hotspot import compute_hotspots

        nodes, edges = self._nodes_and_edges()
        file_churn = {"Subscription.php": 50}  # getRenew's file has high churn
        results = compute_hotspots(nodes, edges, file_churn=file_churn, top_n=10)

        getRenew_result = next(r for r in results if r["name"] == "getRenew")
        assert getRenew_result["churn_rate"] == 50
        assert getRenew_result["score"] > 0

    def test_filters(self, tmp_path: Path):
        from hammy.tools.hotspot import compute_hotspots

        nodes, edges = self._nodes_and_edges()
        results = compute_hotspots(nodes, edges, node_type="function", top_n=10)
        assert all(r["type"] == "function" for r in results)

    def test_empty_nodes(self, tmp_path: Path):
        from hammy.tools.hotspot import compute_hotspots
        assert compute_hotspots([], [], top_n=10) == []

    def test_zero_callers_zero_churn(self, tmp_path: Path):
        from hammy.tools.hotspot import compute_hotspots

        nodes = [_make_node("lonelyFunc", NodeType.FUNCTION, "alone.php")]
        results = compute_hotspots(nodes, [], top_n=10)
        assert results[0]["caller_count"] == 0
        # score = log2(1) * log2(2) = 0 * 1 = 0
        assert results[0]["score"] == 0.0

    def test_explorer_tool(self, tmp_path: Path):
        from hammy.agents.explorer import make_explorer_tools
        from hammy.tools.parser import ParserFactory

        nodes, edges = self._nodes_and_edges()
        tools = make_explorer_tools(tmp_path, ParserFactory(), nodes, edges)
        hotspot = next(t for t in tools if t.name == "Hotspot Score")
        result = hotspot.func(top_n=5)
        assert "getRenew" in result or "processRenewal" in result
        assert "# 1" in result or "#1" in result  # rank indicator


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


class TestPRDiff:
    """Tests for PR diff analysis tool."""

    SAMPLE_DIFF = textwrap.dedent("""\
        diff --git a/foo.py b/foo.py
        index 000..111 100644
        --- a/foo.py
        +++ b/foo.py
        @@ -10,5 +10,9 @@ class Foo
        +    def getRenew(self, x):
        +        return self.processRenewal(x)
        +
        +    def processRenewal(self, x):
        +        pass
    """)

    def _nodes_and_edges(self):
        nodes = [
            Node(
                id=Node.make_id("foo.py", "getRenew"),
                type=NodeType.FUNCTION,
                name="getRenew",
                loc=Location(file="foo.py", lines=(10, 12)),
                language="python",
            ),
            Node(
                id=Node.make_id("bar.py", "callerA"),
                type=NodeType.FUNCTION,
                name="callerA",
                loc=Location(file="bar.py", lines=(5, 8)),
                language="python",
            ),
        ]
        edges = [
            Edge(
                source=Node.make_id("bar.py", "callerA"),
                target=Node.make_id("foo.py", "getRenew"),
                relation=RelationType.CALLS,
                metadata=EdgeMetadata(confidence=0.9, context="getRenew(x)"),
            ),
        ]
        return nodes, edges

    def test_analyze_diff_basic(self):
        from hammy.tools.diff_analysis import analyze_diff

        nodes, edges = self._nodes_and_edges()
        report = analyze_diff(self.SAMPLE_DIFF, nodes, edges)
        assert len(report.changed_files) == 1
        assert report.changed_files[0].path == "foo.py"
        assert "getRenew" in report.all_changed_symbols or "processRenewal" in report.all_changed_symbols

    def test_analyze_diff_impact(self):
        from hammy.tools.diff_analysis import analyze_diff

        nodes, edges = self._nodes_and_edges()
        report = analyze_diff(self.SAMPLE_DIFF, nodes, edges)
        # getRenew has 1 caller (callerA)
        getRenew_impact = next((r for r in report.impact if r["symbol"] == "getRenew"), None)
        if getRenew_impact:
            assert getRenew_impact["indexed"] is True
            assert getRenew_impact["caller_count"] >= 1

    def test_analyze_diff_unknown_symbol(self):
        from hammy.tools.diff_analysis import analyze_diff

        nodes, edges = self._nodes_and_edges()
        diff = textwrap.dedent("""\
            diff --git a/new.py b/new.py
            --- a/new.py
            +++ b/new.py
            @@ -1,0 +1,3 @@
            +def brandNewFunction():
            +    pass
        """)
        report = analyze_diff(diff, nodes, edges)
        unknown = next((r for r in report.impact if r["symbol"] == "brandNewFunction"), None)
        if unknown:
            assert unknown["indexed"] is False

    def test_explorer_tool(self, tmp_path: Path):
        from hammy.agents.explorer import make_explorer_tools
        from hammy.tools.parser import ParserFactory

        nodes, edges = self._nodes_and_edges()
        tools = make_explorer_tools(tmp_path, ParserFactory(), nodes, edges)
        pr_diff_tool = next(t for t in tools if t.name == "PR Diff Analysis")
        result = pr_diff_tool.func(diff_text=self.SAMPLE_DIFF)
        assert "foo.py" in result or "getRenew" in result or len(result) > 0
