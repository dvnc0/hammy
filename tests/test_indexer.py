"""Tests for the indexing pipeline.

Requires Qdrant running: docker compose up -d
"""

import subprocess
from pathlib import Path

import pytest

from hammy.config import HammyConfig, ParsingConfig, QdrantConfig
from hammy.tools.qdrant_tools import QdrantManager

# Use a test-specific prefix to avoid colliding with real data
TEST_QDRANT_CONFIG = QdrantConfig(collection_prefix="hammy_test")


def qdrant_available() -> bool:
    """Check if Qdrant is reachable."""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host="localhost", port=6333, timeout=2)
        client.get_collections()
        return True
    except Exception:
        return False


requires_qdrant = pytest.mark.skipif(
    not qdrant_available(),
    reason="Qdrant not available (run: docker compose up -d)",
)


@pytest.fixture
def qdrant():
    """Create a QdrantManager with test prefix, clean up after test."""
    manager = QdrantManager(TEST_QDRANT_CONFIG)
    manager.delete_collections()
    manager.ensure_collections()
    yield manager
    manager.delete_collections()


@pytest.fixture
def sample_project(tmp_path: Path) -> Path:
    """Create a sample project for indexing."""
    src = tmp_path / "src"
    src.mkdir()

    (src / "UserController.php").write_text(
        '<?php\nnamespace App;\n\n#[Route("/api/users")]\n'
        "class UserController {\n"
        "    public function getUser(int $id): User {\n"
        "        return User::find($id);\n"
        "    }\n"
        "}\n"
    )

    (src / "api.js").write_text(
        'import { config } from "./config";\n\n'
        "export async function fetchUsers() {\n"
        '    const response = await fetch("/api/users");\n'
        "    return response.json();\n"
        "}\n"
    )

    # Vendor should be ignored
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    (vendor / "lib.php").write_text("<?php // vendor")

    return tmp_path


@requires_qdrant
class TestQdrantManager:
    def test_ensure_collections(self, qdrant: QdrantManager):
        stats = qdrant.get_stats()
        assert "code_symbols" in stats
        assert "commits" in stats

    def test_embed(self, qdrant: QdrantManager):
        embeddings = qdrant.embed(["hello world"])
        assert len(embeddings) == 1
        assert len(embeddings[0]) > 0  # Non-empty vector

    def test_upsert_and_search_nodes(self, qdrant: QdrantManager):
        from hammy.schema.models import Location, Node, NodeMeta, NodeType

        nodes = [
            Node(
                id="test1",
                type=NodeType.FUNCTION,
                name="processPayment",
                loc=Location(file="payment.php", lines=(10, 30)),
                language="php",
                meta=NodeMeta(visibility="public", return_type="void"),
                summary="Processes credit card payments via Stripe API",
            ),
            Node(
                id="test2",
                type=NodeType.FUNCTION,
                name="fetchUsers",
                loc=Location(file="api.js", lines=(1, 5)),
                language="javascript",
                meta=NodeMeta(is_async=True),
                summary="Fetches user list from the REST API",
            ),
        ]

        count = qdrant.upsert_nodes(nodes)
        assert count == 2

        # Search for payment-related code
        results = qdrant.search_code("payment processing")
        assert len(results) > 0
        assert results[0]["name"] == "processPayment"

    def test_search_with_language_filter(self, qdrant: QdrantManager):
        from hammy.schema.models import Location, Node, NodeType

        nodes = [
            Node(
                id="php1",
                type=NodeType.FUNCTION,
                name="getUser",
                loc=Location(file="user.php", lines=(1, 10)),
                language="php",
                summary="Gets a user by ID",
            ),
            Node(
                id="js1",
                type=NodeType.FUNCTION,
                name="getUser",
                loc=Location(file="user.js", lines=(1, 10)),
                language="javascript",
                summary="Gets a user by ID from the API",
            ),
        ]
        qdrant.upsert_nodes(nodes)

        # Filter to PHP only
        results = qdrant.search_code("get user", language="php")
        assert all(r["language"] == "php" for r in results)

    def test_upsert_and_search_commits(self, qdrant: QdrantManager):
        commits = [
            {
                "revision": "abc123",
                "author": "Dan",
                "date": "2025-01-01T00:00:00",
                "message": "Fix critical security vulnerability in auth module",
                "files_changed": ["auth.php"],
            },
            {
                "revision": "def456",
                "author": "Dan",
                "date": "2025-01-02T00:00:00",
                "message": "Add new user profile page with avatar upload",
                "files_changed": ["profile.js", "upload.php"],
            },
        ]

        count = qdrant.upsert_commits(commits)
        assert count == 2

        results = qdrant.search_commits("security fix")
        assert len(results) > 0
        assert "security" in results[0]["message"].lower()

    def test_get_stats(self, qdrant: QdrantManager):
        stats = qdrant.get_stats()
        assert stats["code_symbols"] == 0
        assert stats["commits"] == 0


@requires_qdrant
class TestCodeIndexer:
    def test_index_codebase(self, qdrant: QdrantManager, sample_project: Path):
        from hammy.indexer.code_indexer import index_codebase

        config = HammyConfig(
            parsing=ParsingConfig(languages=["php", "javascript"]),
            qdrant=TEST_QDRANT_CONFIG,
        )
        config.project.root = str(sample_project)

        result, nodes, edges = index_codebase(config, qdrant=qdrant)

        assert result.files_processed == 2  # UserController.php + api.js
        assert result.nodes_extracted > 0
        assert result.edges_extracted > 0
        assert result.nodes_indexed > 0
        assert len(result.errors) == 0

        # Verify vendor was ignored
        node_files = [n.loc.file for n in nodes]
        assert not any("vendor" in f for f in node_files)

    def test_index_then_search(self, qdrant: QdrantManager, sample_project: Path):
        from hammy.indexer.code_indexer import index_codebase

        config = HammyConfig(
            parsing=ParsingConfig(languages=["php", "javascript"]),
            qdrant=TEST_QDRANT_CONFIG,
        )
        config.project.root = str(sample_project)

        index_codebase(config, qdrant=qdrant)

        # Should find the PHP controller
        results = qdrant.search_code("user controller")
        assert len(results) > 0

    def test_index_without_qdrant(self, sample_project: Path):
        """Test indexing without storing in Qdrant (parse only)."""
        from hammy.indexer.code_indexer import index_codebase

        config = HammyConfig(
            parsing=ParsingConfig(languages=["php", "javascript"]),
        )
        config.project.root = str(sample_project)

        result, nodes, edges = index_codebase(config, store_in_qdrant=False)

        assert result.files_processed == 2
        assert result.nodes_extracted > 0
        assert result.nodes_indexed == 0  # Nothing stored


@requires_qdrant
class TestCommitIndexer:
    def test_index_commits(self, qdrant: QdrantManager, tmp_path: Path):
        from hammy.indexer.commit_indexer import index_commits

        # Create a git repo with commits
        def run(*args: str) -> None:
            subprocess.run(args, cwd=tmp_path, capture_output=True, check=True)

        run("git", "init")
        run("git", "config", "user.email", "test@example.com")
        run("git", "config", "user.name", "Test User")
        (tmp_path / "app.php").write_text("<?php echo 'v1';")
        run("git", "add", "app.php")
        run("git", "commit", "-m", "Initial version of the app")
        (tmp_path / "app.php").write_text("<?php echo 'v2';")
        run("git", "add", "app.php")
        run("git", "commit", "-m", "Fix bug in payment processing")

        config = HammyConfig(qdrant=TEST_QDRANT_CONFIG)
        config.project.root = str(tmp_path)

        result = index_commits(config, qdrant=qdrant)
        assert result.commits_processed == 2
        assert result.commits_indexed == 2

        # Search for the payment-related commit
        results = qdrant.search_commits("payment bug")
        assert len(results) > 0
        assert "payment" in results[0]["message"].lower()
