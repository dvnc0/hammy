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
        # PHP route endpoint
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
        # JS fetch endpoint
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
    ]


class TestBridgeResolution:
    def test_exact_match(self, sample_nodes):
        bridges = resolve_bridges(sample_nodes, [])
        # /api/v1/users should match exactly
        exact_matches = [
            b for b in bridges
            if b.metadata.confidence == 1.0
        ]
        assert len(exact_matches) >= 1

    def test_wildcard_match(self, sample_nodes):
        bridges = resolve_bridges(sample_nodes, [])
        # /api/v1/users/{id}/pay should match with wildcard
        assert len(bridges) >= 2

    def test_bridge_metadata(self, sample_nodes):
        bridges = resolve_bridges(sample_nodes, [])
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
        assert len(tools) == 4
        tool_names = [t.name for t in tools]
        assert "AST Query" in tool_names
        assert "Search Code Symbols" in tool_names
        assert "Find Cross-Language Bridges" in tool_names
        assert "List Files" in tool_names


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
