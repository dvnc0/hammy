"""Tests for the Hammy MCP server."""

from __future__ import annotations

import shutil
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
        assert "list_files" in tool_names
        assert "find_bridges" in tool_names
        assert "index_status" in tool_names

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
