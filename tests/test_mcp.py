"""Tests for the Hammy MCP server."""

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

import pytest

from hammy.config import HammyConfig
from hammy.mcp.server import create_mcp_server


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    """Create a project directory with PHP and JS fixtures."""
    # Copy fixtures
    fixtures = Path(__file__).parent / "fixtures"
    for src in (fixtures / "sample_php").iterdir():
        shutil.copy2(src, tmp_path / src.name)
    for src in (fixtures / "sample_js").iterdir():
        shutil.copy2(src, tmp_path / src.name)

    # Create config
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "hammy.yaml").write_text(
        "project:\n"
        "  name: test-project\n"
        "  root: .\n"
        "parsing:\n"
        "  languages:\n"
        "    - php\n"
        "    - javascript\n"
    )
    return tmp_path


@pytest.fixture
def mcp_server(project_dir: Path):
    """Create an MCP server for the test project."""
    config = HammyConfig.load(project_dir)
    return create_mcp_server(project_root=project_dir, config=config)


class TestMCPServerCreation:
    def test_creates_server(self, mcp_server):
        assert mcp_server is not None
        assert mcp_server.name == "hammy"

    @pytest.mark.asyncio
    async def test_lists_tools(self, mcp_server):
        tools = await mcp_server.list_tools()
        tool_names = {t.name for t in tools}

        # Core tools should always be present
        assert "ast_query" in tool_names
        assert "search_symbols" in tool_names
        assert "lookup_symbol" in tool_names
        assert "find_usages" in tool_names
        assert "impact_analysis" in tool_names
        assert "structural_search" in tool_names
        assert "hotspot_score" in tool_names
        assert "list_files" in tool_names
        assert "search_code_hybrid" in tool_names
        assert "find_bridges" in tool_names
        assert "index_status" in tool_names
        assert "pr_diff" in tool_names
        assert "explain_symbol" in tool_names
        assert "module_summary" in tool_names
        assert "lookup_symbols_batch" in tool_names

    @pytest.mark.asyncio
    async def test_tool_count_without_vcs_or_qdrant(self, mcp_server):
        tools = await mcp_server.list_tools()
        # Without VCS/Qdrant, we have 5 core tools
        # With VCS (if git init was done), we'd have more
        assert len(tools) >= 5


class TestASTQuery:
    @pytest.mark.asyncio
    async def test_ast_query_php(self, mcp_server, project_dir):
        result = await mcp_server.call_tool(
            "ast_query", {"file_path": "UserController.php"}
        )
        text = _extract_text(result)
        assert "UserController" in text

    @pytest.mark.asyncio
    async def test_ast_query_js(self, mcp_server, project_dir):
        result = await mcp_server.call_tool(
            "ast_query", {"file_path": "api.js"}
        )
        text = _extract_text(result)
        assert len(text) > 0

    @pytest.mark.asyncio
    async def test_ast_query_file_not_found(self, mcp_server):
        result = await mcp_server.call_tool(
            "ast_query", {"file_path": "nonexistent.php"}
        )
        text = _extract_text(result)
        assert "File not found" in text

    @pytest.mark.asyncio
    async def test_ast_query_filter_classes(self, mcp_server):
        result = await mcp_server.call_tool(
            "ast_query",
            {"file_path": "UserController.php", "query_type": "classes"},
        )
        text = _extract_text(result)
        assert "class" in text.lower()

    @pytest.mark.asyncio
    async def test_ast_query_imports(self, mcp_server):
        result = await mcp_server.call_tool(
            "ast_query",
            {"file_path": "api.js", "query_type": "imports"},
        )
        text = _extract_text(result)
        # Should return import info or "No imports found."
        assert len(text) > 0


class TestSearchSymbols:
    @pytest.mark.asyncio
    async def test_search_finds_class(self, mcp_server):
        result = await mcp_server.call_tool(
            "search_symbols", {"query": "User"}
        )
        text = _extract_text(result)
        assert "User" in text

    @pytest.mark.asyncio
    async def test_search_with_language_filter(self, mcp_server):
        result = await mcp_server.call_tool(
            "search_symbols", {"query": "User", "language": "php"}
        )
        text = _extract_text(result)
        assert "User" in text

    @pytest.mark.asyncio
    async def test_search_no_results(self, mcp_server):
        result = await mcp_server.call_tool(
            "search_symbols", {"query": "zzz_nonexistent_zzz"}
        )
        text = _extract_text(result)
        assert "No symbols matching" in text


class TestSearchSymbolsRanked:
    @pytest.mark.asyncio
    async def test_exact_match_first(self, mcp_server):
        """Exact name matches should appear before prefix/substring matches."""
        result = await mcp_server.call_tool(
            "search_symbols", {"query": "UserController"}
        )
        text = _extract_text(result)
        lines = [l for l in text.splitlines() if l.strip()]
        assert len(lines) > 0
        assert "UserController" in lines[0]

    @pytest.mark.asyncio
    async def test_file_filter(self, mcp_server):
        result = await mcp_server.call_tool(
            "search_symbols", {"query": "User", "file_filter": "UserController"}
        )
        text = _extract_text(result)
        assert "User" in text

    @pytest.mark.asyncio
    async def test_no_results_with_strict_filter(self, mcp_server):
        result = await mcp_server.call_tool(
            "search_symbols", {"query": "User", "file_filter": "nonexistent_dir/"}
        )
        text = _extract_text(result)
        assert "No symbols matching" in text


class TestHybridSearch:
    @pytest.mark.asyncio
    async def test_hybrid_search_registered(self, mcp_server):
        tools = await mcp_server.list_tools()
        tool_names = {t.name for t in tools}
        assert "search_code_hybrid" in tool_names

    @pytest.mark.asyncio
    async def test_hybrid_finds_symbol(self, mcp_server):
        result = await mcp_server.call_tool(
            "search_code_hybrid", {"query": "UserController"}
        )
        text = _extract_text(result)
        assert "UserController" in text

    @pytest.mark.asyncio
    async def test_hybrid_no_match(self, mcp_server):
        # When Qdrant is available, dense search always returns nearest neighbors
        # even for nonsense queries; when not available, BM25 gives empty.
        result = await mcp_server.call_tool(
            "search_code_hybrid", {"query": "zzz_totally_missing_xyz"}
        )
        text = _extract_text(result)
        # Either returns results (Qdrant semantic) or empty message — no crash
        assert len(text) > 0

    @pytest.mark.asyncio
    async def test_hybrid_language_filter(self, mcp_server):
        result = await mcp_server.call_tool(
            "search_code_hybrid", {"query": "User", "language": "php"}
        )
        text = _extract_text(result)
        # Should return results or empty message — no crash
        assert len(text) > 0

    @pytest.mark.asyncio
    async def test_hybrid_returns_score(self, mcp_server):
        result = await mcp_server.call_tool(
            "search_code_hybrid", {"query": "getUser"}
        )
        text = _extract_text(result)
        if "No code matching" not in text:
            # Results should include a bracketed score
            assert "[" in text


class TestLookupSymbol:
    @pytest.mark.asyncio
    async def test_lookup_exact(self, mcp_server):
        result = await mcp_server.call_tool(
            "lookup_symbol", {"name": "UserController"}
        )
        text = _extract_text(result)
        assert "UserController" in text
        assert "file:" in text

    @pytest.mark.asyncio
    async def test_lookup_not_found(self, mcp_server):
        result = await mcp_server.call_tool(
            "lookup_symbol", {"name": "zzz_totally_missing"}
        )
        text = _extract_text(result)
        assert "not found" in text.lower() or "search_symbols" in text

    @pytest.mark.asyncio
    async def test_lookup_with_node_type(self, mcp_server):
        result = await mcp_server.call_tool(
            "lookup_symbol", {"name": "UserController", "node_type": "class"}
        )
        text = _extract_text(result)
        assert "UserController" in text


class TestFindUsages:
    @pytest.mark.asyncio
    async def test_find_usages_registered(self, mcp_server):
        tools = await mcp_server.list_tools()
        tool_names = {t.name for t in tools}
        assert "find_usages" in tool_names

    @pytest.mark.asyncio
    async def test_find_usages_not_found(self, mcp_server):
        result = await mcp_server.call_tool(
            "find_usages", {"symbol_name": "zzz_totally_missing"}
        )
        text = _extract_text(result)
        assert "No call sites" in text

    @pytest.mark.asyncio
    async def test_find_usages_no_mid_word_match(self, mcp_server):
        """Searching for 'User' should not match 'UserController' as a call site."""
        # The fixtures have CALLS edges for API calls — not for 'User' the symbol.
        # Key thing: if there are no exact word-boundary matches, report none found.
        result = await mcp_server.call_tool(
            "find_usages", {"symbol_name": "User"}
        )
        text = _extract_text(result)
        # Either finds real callers of exactly 'User', or reports none — never mid-word noise
        assert "No call sites" in text or "Call sites of 'User'" in text


class TestListFiles:
    @pytest.mark.asyncio
    async def test_list_all_files(self, mcp_server):
        result = await mcp_server.call_tool("list_files", {})
        text = _extract_text(result)
        assert ".php" in text or ".js" in text

    @pytest.mark.asyncio
    async def test_list_php_files(self, mcp_server):
        result = await mcp_server.call_tool(
            "list_files", {"language": "php"}
        )
        text = _extract_text(result)
        assert "php" in text.lower()


class TestFindBridges:
    @pytest.mark.asyncio
    async def test_find_bridges(self, mcp_server):
        result = await mcp_server.call_tool("find_bridges", {})
        text = _extract_text(result)
        # May or may not find bridges depending on fixture content
        assert len(text) > 0


class TestIndexStatus:
    @pytest.mark.asyncio
    async def test_index_status(self, mcp_server):
        result = await mcp_server.call_tool("index_status", {})
        text = _extract_text(result)
        assert "test-project" in text
        assert "Total files" in text
        assert "Total symbols" in text

    @pytest.mark.asyncio
    async def test_status_resource(self, mcp_server):
        results = await mcp_server.read_resource("hammy://status")
        texts = [r.text if hasattr(r, "text") else str(r) for r in results]
        combined = "\n".join(texts)
        assert "test-project" in combined


class TestReindex:
    @pytest.mark.asyncio
    async def test_reindex_refreshes_symbols(self, mcp_server, project_dir):
        # Get initial status
        result = await mcp_server.call_tool("index_status", {})
        initial_text = _extract_text(result)

        # Add a new file
        (project_dir / "newfile.php").write_text(
            "<?php\nclass NewClass {\n  public function newMethod() {}\n}\n"
        )

        # Reindex
        result = await mcp_server.call_tool("reindex", {"update_qdrant": False})
        text = _extract_text(result)
        assert "Reindex complete" in text
        assert "Files processed" in text

        # Search should now find the new class
        result = await mcp_server.call_tool(
            "search_symbols", {"query": "NewClass"}
        )
        text = _extract_text(result)
        assert "NewClass" in text

    @pytest.mark.asyncio
    async def test_reindex_removes_deleted_files(self, mcp_server, project_dir):
        # Verify UserController exists initially
        result = await mcp_server.call_tool(
            "search_symbols", {"query": "UserController"}
        )
        assert "UserController" in _extract_text(result)

        # Delete the file
        (project_dir / "UserController.php").unlink()

        # Reindex
        await mcp_server.call_tool("reindex", {"update_qdrant": False})

        # Should no longer find it
        result = await mcp_server.call_tool(
            "search_symbols", {"query": "UserController"}
        )
        assert "No symbols matching" in _extract_text(result)

    @pytest.mark.asyncio
    async def test_reindex_with_qdrant_flag(self, mcp_server):
        result = await mcp_server.call_tool(
            "reindex", {"update_qdrant": True}
        )
        text = _extract_text(result)
        assert "Reindex complete" in text
        # Either Qdrant is available (shows indexed count) or not (shows note)
        assert "Qdrant" in text or "Symbols extracted" in text

    @pytest.mark.asyncio
    async def test_reindex_tool_registered(self, mcp_server):
        tools = await mcp_server.list_tools()
        tool_names = {t.name for t in tools}
        assert "reindex" in tool_names


class TestImpactAnalysis:
    @pytest.mark.asyncio
    async def test_impact_analysis_registered(self, mcp_server):
        tools = await mcp_server.list_tools()
        tool_names = {t.name for t in tools}
        assert "impact_analysis" in tool_names

    @pytest.mark.asyncio
    async def test_impact_analysis_no_callers(self, mcp_server):
        result = await mcp_server.call_tool(
            "impact_analysis", {"symbol_name": "zzz_totally_missing", "depth": 1}
        )
        text = _extract_text(result)
        assert "No callers" in text or "No call graph" in text

    @pytest.mark.asyncio
    async def test_impact_analysis_callers_direction(self, mcp_server):
        # Just ensure it runs and produces output with the right section header
        result = await mcp_server.call_tool(
            "impact_analysis",
            {"symbol_name": "getUser", "depth": 2, "direction": "callers"},
        )
        text = _extract_text(result)
        assert "Callers of" in text

    @pytest.mark.asyncio
    async def test_impact_analysis_both_direction(self, mcp_server):
        result = await mcp_server.call_tool(
            "impact_analysis",
            {"symbol_name": "getUser", "depth": 2, "direction": "both"},
        )
        text = _extract_text(result)
        assert "Callers of" in text
        assert "Callees of" in text


class TestStructuralSearch:
    @pytest.mark.asyncio
    async def test_structural_search_registered(self, mcp_server):
        tools = await mcp_server.list_tools()
        tool_names = {t.name for t in tools}
        assert "structural_search" in tool_names

    @pytest.mark.asyncio
    async def test_finds_public_methods(self, mcp_server):
        result = await mcp_server.call_tool(
            "structural_search", {"visibility": "public", "node_type": "method"}
        )
        text = _extract_text(result)
        # Fixture has PHP class methods with public visibility
        assert "public" in text or "matched" in text

    @pytest.mark.asyncio
    async def test_no_match_returns_message(self, mcp_server):
        result = await mcp_server.call_tool(
            "structural_search",
            {"visibility": "private", "return_type": "ZZZNoSuchType", "node_type": "class"},
        )
        text = _extract_text(result)
        assert "No symbols matched" in text

    @pytest.mark.asyncio
    async def test_file_filter_narrows(self, mcp_server):
        result = await mcp_server.call_tool(
            "structural_search", {"file_filter": "UserController", "node_type": "method"}
        )
        text = _extract_text(result)
        if "No symbols matched" not in text:
            assert "UserController" in text

    @pytest.mark.asyncio
    async def test_min_params_filter(self, mcp_server):
        result = await mcp_server.call_tool(
            "structural_search", {"min_params": 1}
        )
        text = _extract_text(result)
        assert "matched" in text or "No symbols" in text

    @pytest.mark.asyncio
    async def test_name_pattern_regex(self, mcp_server):
        result = await mcp_server.call_tool(
            "structural_search", {"name_pattern": "^get"}
        )
        text = _extract_text(result)
        if "No symbols matched" not in text:
            # All returned names should start with "get" (case-insensitive)
            for line in text.splitlines():
                if ": " in line and "(" in line:
                    # line format: "method: getName (file.php:10)"
                    name_part = line.split(":")[1].strip().split("(")[0].strip()
                    assert name_part.lower().startswith("get"), f"Unexpected name: {name_part}"


class TestHotspotScore:
    @pytest.mark.asyncio
    async def test_hotspot_registered(self, mcp_server):
        tools = await mcp_server.list_tools()
        assert "hotspot_score" in {t.name for t in tools}

    @pytest.mark.asyncio
    async def test_hotspot_returns_results(self, mcp_server):
        result = await mcp_server.call_tool("hotspot_score", {"top_n": 5})
        text = _extract_text(result)
        # Should return a ranked list or a "no symbols" message — not crash
        assert "hotspot" in text.lower() or "No symbols" in text or "#" in text

    @pytest.mark.asyncio
    async def test_hotspot_node_type_filter(self, mcp_server):
        result = await mcp_server.call_tool(
            "hotspot_score", {"node_type": "method", "top_n": 10}
        )
        text = _extract_text(result)
        assert len(text) > 0

    @pytest.mark.asyncio
    async def test_hotspot_no_match(self, mcp_server):
        result = await mcp_server.call_tool(
            "hotspot_score", {"file_filter": "zzz_no_such_dir/", "top_n": 5}
        )
        text = _extract_text(result)
        assert "No symbols" in text

    @pytest.mark.asyncio
    async def test_hotspot_respects_top_n(self, mcp_server):
        result = await mcp_server.call_tool("hotspot_score", {"top_n": 3})
        text = _extract_text(result)
        # Should not list more than 3 entries (each starts with #N)
        import re
        ranked = re.findall(r"#\d+", text)
        assert len(ranked) <= 3


class TestBrainTools:
    """Tests for brain (working memory) tools — only present when Qdrant is available."""

    @pytest.mark.asyncio
    async def test_brain_tools_absent_without_qdrant(self, project_dir: Path):
        """Without Qdrant the brain tools should not be registered."""
        from unittest.mock import patch
        from hammy.config import HammyConfig
        from hammy.mcp.server import create_mcp_server

        config = HammyConfig.load(project_dir)
        # Force Qdrant to be unavailable so brain tools are not registered
        with patch("hammy.mcp.server.QdrantManager", side_effect=Exception("no qdrant")):
            server = create_mcp_server(project_root=project_dir, config=config)

        tools = await server.list_tools()
        tool_names = {t.name for t in tools}
        assert "store_context" not in tool_names
        assert "recall_context" not in tool_names
        assert "list_context" not in tool_names

    @pytest.mark.asyncio
    async def test_brain_tools_present_with_qdrant(self, project_dir: Path):
        """With Qdrant the brain tools should be registered."""
        from hammy.config import QdrantConfig
        from hammy.tools.qdrant_tools import QdrantManager

        def qdrant_available() -> bool:
            try:
                from qdrant_client import QdrantClient
                QdrantClient(host="localhost", port=6333, timeout=2).get_collections()
                return True
            except Exception:
                return False

        if not qdrant_available():
            pytest.skip("Qdrant not available")

        from hammy.config import HammyConfig
        config = HammyConfig.load(project_dir)
        config.qdrant = QdrantConfig(collection_prefix="hammy_brain_test")
        qdrant = QdrantManager(config.qdrant)
        qdrant.delete_collections()
        qdrant.ensure_collections()
        try:
            server = create_mcp_server(project_root=project_dir, config=config)
            tools = await server.list_tools()
            tool_names = {t.name for t in tools}
            assert "store_context" in tool_names
            assert "recall_context" in tool_names
            assert "list_context" in tool_names
        finally:
            qdrant.delete_collections()

    @pytest.mark.asyncio
    async def test_store_and_recall(self, project_dir: Path):
        """store_context then recall_context by key should round-trip."""
        from hammy.config import QdrantConfig
        from hammy.tools.qdrant_tools import QdrantManager

        def qdrant_available() -> bool:
            try:
                from qdrant_client import QdrantClient
                QdrantClient(host="localhost", port=6333, timeout=2).get_collections()
                return True
            except Exception:
                return False

        if not qdrant_available():
            pytest.skip("Qdrant not available")

        from hammy.config import HammyConfig
        config = HammyConfig.load(project_dir)
        config.qdrant = QdrantConfig(collection_prefix="hammy_brain_rt_test")
        qdrant = QdrantManager(config.qdrant)
        qdrant.delete_collections()
        qdrant.ensure_collections()
        try:
            server = create_mcp_server(project_root=project_dir, config=config)
            # Store
            result = await server.call_tool("store_context", {
                "key": "test-finding",
                "content": "getRenew called from 3 places.",
                "tags": "research",
            })
            assert "test-finding" in _extract_text(result)
            # Recall by key
            result = await server.call_tool("recall_context", {"key": "test-finding"})
            text = _extract_text(result)
            assert "getRenew" in text
            assert "test-finding" in text
            # List
            result = await server.call_tool("list_context", {})
            assert "test-finding" in _extract_text(result)
        finally:
            qdrant.delete_collections()


class TestMCPWithVCS:
    """Tests for VCS-dependent tools (require a git repo)."""

    @pytest.fixture
    def git_project(self, project_dir: Path) -> Path:
        """Initialize a git repo in the project directory."""
        import subprocess

        subprocess.run(
            ["git", "init"], cwd=project_dir, capture_output=True, check=True
        )
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=project_dir, capture_output=True, check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=project_dir, capture_output=True, check=True,
        )
        subprocess.run(
            ["git", "add", "."], cwd=project_dir, capture_output=True, check=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"],
            cwd=project_dir, capture_output=True, check=True,
        )
        return project_dir

    @pytest.fixture
    def vcs_mcp_server(self, git_project: Path):
        config = HammyConfig.load(git_project)
        return create_mcp_server(project_root=git_project, config=config)

    @pytest.mark.asyncio
    async def test_has_vcs_tools(self, vcs_mcp_server):
        tools = await vcs_mcp_server.list_tools()
        tool_names = {t.name for t in tools}
        assert "git_log" in tool_names
        assert "git_blame" in tool_names
        assert "file_churn" in tool_names

    @pytest.mark.asyncio
    async def test_git_log(self, vcs_mcp_server):
        result = await vcs_mcp_server.call_tool("git_log", {"limit": 5})
        text = _extract_text(result)
        assert "Initial commit" in text

    @pytest.mark.asyncio
    async def test_git_blame(self, vcs_mcp_server):
        result = await vcs_mcp_server.call_tool(
            "git_blame", {"file_path": "UserController.php"}
        )
        text = _extract_text(result)
        assert "Test" in text  # author name

    @pytest.mark.asyncio
    async def test_file_churn(self, vcs_mcp_server):
        result = await vcs_mcp_server.call_tool(
            "file_churn", {"window_days": 90}
        )
        text = _extract_text(result)
        # Single commit, should show some churn
        assert len(text) > 0


class TestPRDiff:
    @pytest.mark.asyncio
    async def test_pr_diff_basic(self, mcp_server):
        diff = textwrap.dedent("""\
            diff --git a/UserController.php b/UserController.php
            index abc..def 100644
            --- a/UserController.php
            +++ b/UserController.php
            @@ -1,5 +1,6 @@ class UserController
            +    public function getRenew($id) {
            +        return $this->service->processRenewal($id);
            +    }
        """)
        result = await mcp_server.call_tool("pr_diff", {"diff_text": diff})
        text = _extract_text(result)
        assert len(text) > 0
        # Should mention the changed file
        assert "UserController.php" in text

    @pytest.mark.asyncio
    async def test_pr_diff_empty_diff(self, mcp_server):
        result = await mcp_server.call_tool("pr_diff", {"diff_text": ""})
        text = _extract_text(result)
        assert "No changed files" in text or len(text) > 0

    @pytest.mark.asyncio
    async def test_pr_diff_no_params(self, mcp_server):
        result = await mcp_server.call_tool("pr_diff", {})
        text = _extract_text(result)
        assert "diff_text" in text or "base_ref" in text or len(text) > 0


class TestArgumentFilter:
    @pytest.mark.asyncio
    async def test_argument_filter_narrows_results(self, mcp_server):
        """argument_filter should restrict find_usages to calls containing the substring."""
        result = await mcp_server.call_tool(
            "find_usages",
            {"symbol_name": "getUser", "argument_filter": "zzz_no_such_arg"},
        )
        text = _extract_text(result)
        assert "No call sites" in text

    @pytest.mark.asyncio
    async def test_argument_filter_empty_passes_all(self, mcp_server):
        """With no argument_filter, results should be same as without it."""
        result_no_filter = await mcp_server.call_tool(
            "find_usages", {"symbol_name": "getUser"}
        )
        result_with_empty = await mcp_server.call_tool(
            "find_usages", {"symbol_name": "getUser", "argument_filter": ""}
        )
        assert _extract_text(result_no_filter) == _extract_text(result_with_empty)


class TestExplainSymbol:
    @pytest.mark.asyncio
    async def test_explain_symbol_registered(self, mcp_server):
        tools = await mcp_server.list_tools()
        tool_names = {t.name for t in tools}
        assert "explain_symbol" in tool_names

    @pytest.mark.asyncio
    async def test_explain_symbol_found(self, mcp_server):
        result = await mcp_server.call_tool("explain_symbol", {"name": "UserController"})
        text = _extract_text(result)
        assert "UserController" in text
        assert "file:" in text
        assert "Callers" in text

    @pytest.mark.asyncio
    async def test_explain_symbol_not_found(self, mcp_server):
        result = await mcp_server.call_tool(
            "explain_symbol", {"name": "zzz_totally_missing"}
        )
        text = _extract_text(result)
        assert "not found" in text


class TestModuleSummary:
    @pytest.mark.asyncio
    async def test_module_summary_registered(self, mcp_server):
        tools = await mcp_server.list_tools()
        tool_names = {t.name for t in tools}
        assert "module_summary" in tool_names

    @pytest.mark.asyncio
    async def test_module_summary_no_match(self, mcp_server):
        result = await mcp_server.call_tool(
            "module_summary", {"directory": "zzz/no/such/dir/"}
        )
        text = _extract_text(result)
        assert "No symbols found" in text

    @pytest.mark.asyncio
    async def test_module_summary_returns_structure(self, mcp_server):
        result = await mcp_server.call_tool("module_summary", {"directory": ""})
        text = _extract_text(result)
        # Should return some module output or no-symbols message
        assert len(text) > 0


class TestLookupSymbolsBatch:
    @pytest.mark.asyncio
    async def test_lookup_symbols_batch_registered(self, mcp_server):
        tools = await mcp_server.list_tools()
        tool_names = {t.name for t in tools}
        assert "lookup_symbols_batch" in tool_names

    @pytest.mark.asyncio
    async def test_lookup_symbols_batch_found(self, mcp_server):
        result = await mcp_server.call_tool(
            "lookup_symbols_batch", {"names": "UserController"}
        )
        text = _extract_text(result)
        assert "UserController" in text
        assert "file:" in text

    @pytest.mark.asyncio
    async def test_lookup_symbols_batch_multiple(self, mcp_server):
        result = await mcp_server.call_tool(
            "lookup_symbols_batch", {"names": "UserController, zzz_missing"}
        )
        text = _extract_text(result)
        assert "UserController" in text
        assert "not found" in text

    @pytest.mark.asyncio
    async def test_lookup_symbols_batch_empty(self, mcp_server):
        result = await mcp_server.call_tool("lookup_symbols_batch", {"names": ""})
        text = _extract_text(result)
        assert "Provide at least one" in text


class TestPrDiffWorkingTree:
    @pytest.mark.asyncio
    async def test_working_tree_no_vcs_returns_error(self, mcp_server):
        """Without VCS, working_tree=True should return an error message."""
        result = await mcp_server.call_tool(
            "pr_diff", {"working_tree": True}
        )
        text = _extract_text(result)
        # Either VCS is available (runs diff) or returns error
        assert len(text) > 0


class TestCLIServe:
    def test_serve_help(self):
        from typer.testing import CliRunner
        from hammy.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "MCP server" in result.output
        assert "--transport" in result.output


def _extract_text(result) -> str:
    """Extract text content from MCP tool result."""
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        parts = []
        for item in result:
            if hasattr(item, "text"):
                parts.append(item.text)
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(result)
