"""Tests for the Hammy schema models."""

from hammy.schema.models import (
    ContextPack,
    Edge,
    EdgeMetadata,
    Location,
    Node,
    NodeMeta,
    NodeType,
    RelationType,
)


class TestNode:
    def test_make_id_deterministic(self):
        id1 = Node.make_id("src/UserController.php", "processPayment")
        id2 = Node.make_id("src/UserController.php", "processPayment")
        assert id1 == id2

    def test_make_id_unique_for_different_symbols(self):
        id1 = Node.make_id("src/UserController.php", "processPayment")
        id2 = Node.make_id("src/UserController.php", "getUser")
        assert id1 != id2

    def test_create_node(self):
        node = Node(
            id=Node.make_id("src/app.js", "fetchUsers"),
            type=NodeType.FUNCTION,
            name="fetchUsers",
            loc=Location(file="src/app.js", lines=(10, 25)),
            language="javascript",
            meta=NodeMeta(is_async=True),
            summary="Fetches users from the API.",
        )
        assert node.name == "fetchUsers"
        assert node.type == NodeType.FUNCTION
        assert node.language == "javascript"
        assert node.meta.is_async is True
        assert node.loc.lines == (10, 25)

    def test_node_serialization(self):
        node = Node(
            id="abc123",
            type=NodeType.CLASS,
            name="UserController",
            loc=Location(file="src/UserController.php", lines=(1, 100)),
            language="php",
            meta=NodeMeta(visibility="public"),
        )
        data = node.model_dump()
        assert data["type"] == "class"
        assert data["meta"]["visibility"] == "public"

        restored = Node.model_validate(data)
        assert restored.name == node.name


class TestEdge:
    def test_create_edge(self):
        edge = Edge(
            source="node_1",
            target="node_2",
            relation=RelationType.CALLS,
            metadata=EdgeMetadata(is_bridge=True, confidence=0.95, context="fetch('/api/users')"),
        )
        assert edge.relation == RelationType.CALLS
        assert edge.metadata.is_bridge is True
        assert edge.metadata.confidence == 0.95

    def test_edge_defaults(self):
        edge = Edge(source="a", target="b", relation=RelationType.IMPORTS)
        assert edge.metadata.is_bridge is False
        assert edge.metadata.confidence == 1.0


class TestContextPack:
    def test_empty_context_pack(self):
        pack = ContextPack(query="Why is the profile slow?")
        assert pack.query == "Why is the profile slow?"
        assert pack.nodes == []
        assert pack.edges == []
        assert pack.warnings == []

    def test_context_pack_with_data(self):
        node = Node(
            id="n1",
            type=NodeType.METHOD,
            name="getUser",
            loc=Location(file="UserRepo.php", lines=(10, 30)),
            language="php",
        )
        pack = ContextPack(
            query="What does getUser do?",
            nodes=[node],
            warnings=["High churn: 85% in last 90 days"],
            summary="getUser queries the legacy user table.",
        )
        assert len(pack.nodes) == 1
        assert len(pack.warnings) == 1
