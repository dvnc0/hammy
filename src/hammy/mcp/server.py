"""Hammy MCP Server — exposes codebase intelligence tools via Model Context Protocol.

Provides tools for code exploration, VCS history, and semantic search
that AI coding agents can call through the MCP interface.
"""

from __future__ import annotations

import re
from pathlib import Path

from mcp.server import FastMCP

from hammy.config import HammyConfig
from hammy.indexer.code_indexer import index_codebase
from hammy.schema.models import Edge, Node, NodeType, RelationType
from hammy.tools.bridge import resolve_bridges
from hammy.tools.parser import ParserFactory
from hammy.tools.qdrant_tools import QdrantManager
from hammy.tools.vcs import VCSWrapper


def create_mcp_server(
    project_root: Path | None = None,
    *,
    config: HammyConfig | None = None,
) -> FastMCP:
    """Create and configure the Hammy MCP server.

    Args:
        project_root: Path to the project to analyze. Defaults to cwd.
        config: Optional pre-loaded config. Loaded from project_root if None.

    Returns:
        Configured FastMCP server instance.
    """
    if project_root is None:
        project_root = Path.cwd()
    project_root = project_root.resolve()

    if config is None:
        config = HammyConfig.load(project_root)

    # Index the codebase
    qdrant: QdrantManager | None = None
    try:
        qdrant = QdrantManager(config.qdrant)
        qdrant.ensure_collections()
    except Exception:
        qdrant = None

    _, initial_nodes, initial_edges = index_codebase(
        config, qdrant=qdrant, store_in_qdrant=qdrant is not None
    )

    # Use mutable lists so the reindex tool can update them in-place
    all_nodes: list[Node] = list(initial_nodes)
    all_edges: list[Edge] = list(initial_edges)

    # Set up parser and VCS
    parser_factory = ParserFactory(config.parsing.languages)

    vcs: VCSWrapper | None = None
    try:
        vcs = VCSWrapper(project_root)
    except ValueError:
        pass

    # Create MCP server
    mcp = FastMCP(
        name="hammy",
        instructions=(
            "Hammy is a codebase intelligence engine. Use its tools to explore "
            "code structure, search for symbols, analyze VCS history, and find "
            "cross-language connections in the codebase."
        ),
    )

    # --- Code Exploration Tools ---

    @mcp.tool(
        name="ast_query",
        description=(
            "Query the AST of a specific file. Returns structured information "
            "about classes, functions, methods, endpoints, and imports."
        ),
    )
    def ast_query(file_path: str, query_type: str = "all") -> str:
        """Query AST of a file.

        Args:
            file_path: Path to the file (relative to project root).
            query_type: What to extract - 'all', 'classes', 'functions',
                        'methods', 'endpoints', or 'imports'.
        """
        full_path = project_root / file_path
        if not full_path.exists():
            return f"File not found: {file_path}"

        result = parser_factory.parse_file(full_path)
        if result is None:
            return f"Unsupported file type: {file_path}"

        tree, lang = result
        from hammy.tools.ast_tools import extract_symbols

        nodes, edges = extract_symbols(tree, lang, file_path)

        type_filter = {
            "classes": NodeType.CLASS,
            "functions": NodeType.FUNCTION,
            "methods": NodeType.METHOD,
            "endpoints": NodeType.ENDPOINT,
        }.get(query_type)

        if type_filter:
            nodes = [n for n in nodes if n.type == type_filter]

        if query_type == "imports":
            import_edges = [e for e in edges if e.relation == RelationType.IMPORTS]
            return "\n".join(
                f"import: {e.metadata.context}" for e in import_edges
            ) or "No imports found."

        lines = []
        for n in nodes:
            line = f"{n.type.value}: {n.name} ({n.loc.file}:{n.loc.lines[0]}-{n.loc.lines[1]})"
            if n.meta.visibility:
                line += f" [{n.meta.visibility}]"
            if n.meta.is_async:
                line += " [async]"
            if n.meta.return_type:
                line += f" -> {n.meta.return_type}"
            if n.summary:
                line += f" | {n.summary}"
            lines.append(line)

        return "\n".join(lines) or "No symbols found."

    @mcp.tool(
        name="search_symbols",
        description=(
            "Search for code symbols (classes, functions, methods) by name or keyword. "
            "Results are ranked: exact name matches appear first, then prefix matches, "
            "then substring matches, then summary matches. "
            "Use node_type to narrow to a specific kind (class/function/method/endpoint). "
            "Use file_filter to restrict to a directory or filename substring. "
            "For exact definition lookup of a known symbol, prefer lookup_symbol instead."
        ),
    )
    def search_symbols(
        query: str,
        language: str = "",
        node_type: str = "",
        file_filter: str = "",
    ) -> str:
        """Search indexed code symbols with ranked results.

        Args:
            query: Search term (symbol name or keyword).
            language: Optional language filter ('php', 'javascript', 'python', etc.).
            node_type: Optional type filter ('class', 'function', 'method', 'endpoint').
            file_filter: Optional path substring to restrict results (e.g. 'controllers/').
        """
        query_lower = query.lower()
        scored: list[tuple[int, Node]] = []

        for node in all_nodes:
            if language and node.language != language:
                continue
            if node_type and node.type.value != node_type:
                continue
            if file_filter and file_filter.lower() not in node.loc.file.lower():
                continue

            name_lower = node.name.lower()
            if name_lower == query_lower:
                scored.append((4, node))
            elif name_lower.startswith(query_lower):
                scored.append((3, node))
            elif query_lower in name_lower:
                scored.append((2, node))
            elif query_lower in node.summary.lower():
                scored.append((1, node))

        if not scored:
            return f"No symbols matching '{query}' found."

        scored.sort(key=lambda x: (-x[0], len(x[1].name)))
        results = [n for _, n in scored]

        lines = []
        for n in results[:25]:
            line = f"{n.type.value}: {n.name} ({n.loc.file}:{n.loc.lines[0]}-{n.loc.lines[1]})"
            if n.meta.visibility:
                line += f" [{n.meta.visibility}]"
            if n.summary:
                line += f" | {n.summary}"
            lines.append(line)

        if len(results) > 25:
            lines.append(f"\n... and {len(results) - 25} more. Use file_filter or node_type to narrow.")

        return "\n".join(lines)

    @mcp.tool(
        name="find_usages",
        description=(
            "Find all call sites of a specific function or method by exact name. "
            "Uses word-boundary matching so 'save' won't match 'saveAll' or 'isSaved'. "
            "Use file_filter to restrict results to a directory or filename substring. "
            "Returns the containing function/method and file location for each call site."
        ),
    )
    def find_usages(symbol_name: str, file_filter: str = "") -> str:
        """Find all callers of a function or method by exact name.

        Args:
            symbol_name: Exact name of the function/method to find call sites for.
            file_filter: Optional path substring to restrict results (e.g. 'controllers/').
        """
        pattern = re.compile(r"\b" + re.escape(symbol_name) + r"\b", re.IGNORECASE)
        node_index = {n.id: n for n in all_nodes}

        callers = []
        for edge in all_edges:
            if edge.relation != RelationType.CALLS:
                continue
            context = edge.metadata.context or ""
            if not pattern.search(context):
                continue
            source_node = node_index.get(edge.source)
            if source_node is None:
                continue
            if file_filter and file_filter.lower() not in source_node.loc.file.lower():
                continue
            callers.append((source_node, context))

        if not callers:
            return (
                f"No call sites of '{symbol_name}' found. "
                "Check spelling (search is exact/word-boundary). "
                "Use search_symbols to find the definition first."
            )

        lines = [f"Call sites of '{symbol_name}' ({len(callers)} found):"]
        for node, context in callers[:30]:
            lines.append(
                f"  {node.type.value}: {node.name} "
                f"({node.loc.file}:{node.loc.lines[0]}) "
                f"→ calls: {context}"
            )
        if len(callers) > 30:
            lines.append(f"\n... and {len(callers) - 30} more. Use file_filter to narrow.")
        return "\n".join(lines)

    @mcp.tool(
        name="lookup_symbol",
        description=(
            "Look up the exact definition of a known symbol by its precise name. "
            "Returns the file, line numbers, parameters, return type, and visibility. "
            "Use this when you know the exact name (e.g. 'getRenew', 'UserController'). "
            "For fuzzy/keyword search, use search_symbols instead."
        ),
    )
    def lookup_symbol(name: str, node_type: str = "") -> str:
        """Look up a symbol by exact name.

        Args:
            name: Exact symbol name to look up (case-insensitive).
            node_type: Optional type filter ('class', 'function', 'method', 'endpoint').
        """
        name_lower = name.lower()
        matches = [
            n for n in all_nodes
            if n.name.lower() == name_lower
            and (not node_type or n.type.value == node_type)
        ]

        if not matches:
            # Fall back to word-boundary partial match
            pattern = re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE)
            matches = [
                n for n in all_nodes
                if pattern.search(n.name)
                and (not node_type or n.type.value == node_type)
            ]
            if not matches:
                return (
                    f"Symbol '{name}' not found. "
                    "Try search_symbols for fuzzy matching."
                )
            prefix = f"No exact match for '{name}', showing word-boundary matches:\n"
        else:
            prefix = ""

        lines = [prefix] if prefix else []
        for n in matches[:20]:
            line = f"{n.type.value}: {n.name}"
            line += f"\n  file: {n.loc.file}:{n.loc.lines[0]}-{n.loc.lines[1]}"
            line += f"\n  language: {n.language}"
            if n.meta.visibility:
                line += f"\n  visibility: {n.meta.visibility}"
            if n.meta.parameters:
                line += f"\n  params: {', '.join(n.meta.parameters)}"
            if n.meta.return_type:
                line += f"\n  returns: {n.meta.return_type}"
            if n.meta.is_async:
                line += "\n  async: true"
            if n.summary:
                line += f"\n  summary: {n.summary}"
            lines.append(line)

        return "\n\n".join(lines)

    @mcp.tool(
        name="list_files",
        description="List all indexed files, optionally filtered by language.",
    )
    def list_files(language: str = "") -> str:
        """List indexed files.

        Args:
            language: Optional language filter ('php' or 'javascript').
        """
        files: dict[str, set[str]] = {}
        for node in all_nodes:
            if language and node.language != language:
                continue
            files.setdefault(node.loc.file, set()).add(node.language)

        if not files:
            return "No files found."

        lines = []
        for f in sorted(files.keys()):
            langs = ", ".join(sorted(files[f]))
            lines.append(f"{f} [{langs}]")

        return "\n".join(lines)

    @mcp.tool(
        name="find_bridges",
        description=(
            "Find cross-language connections (e.g., JS fetch calls matching PHP routes). "
            "Returns bridge relationships between frontend and backend code."
        ),
    )
    def find_bridges() -> str:
        """Find cross-language bridges."""
        bridges = resolve_bridges(all_nodes, all_edges)

        if not bridges:
            return "No cross-language bridges found."

        lines = []
        for bridge in bridges:
            lines.append(
                f"BRIDGE: {bridge.metadata.context} "
                f"(confidence: {bridge.metadata.confidence:.0%})"
            )

        return "\n".join(lines)

    @mcp.tool(
        name="index_status",
        description="Show the current index statistics — files, symbols, and languages.",
    )
    def index_status() -> str:
        """Show index stats."""
        by_lang: dict[str, int] = {}
        by_type: dict[str, int] = {}
        files: set[str] = set()

        for node in all_nodes:
            by_lang[node.language] = by_lang.get(node.language, 0) + 1
            by_type[node.type.value] = by_type.get(node.type.value, 0) + 1
            files.add(node.loc.file)

        lines = [
            f"Project: {config.project.name}",
            f"Root: {config.project.root}",
            f"Total files: {len(files)}",
            f"Total symbols: {len(all_nodes)}",
            f"Total edges: {len(all_edges)}",
            "",
            "By language:",
        ]
        for lang, count in sorted(by_lang.items()):
            lines.append(f"  {lang}: {count} symbols")

        lines.append("\nBy type:")
        for ntype, count in sorted(by_type.items()):
            lines.append(f"  {ntype}: {count}")

        bridges = resolve_bridges(all_nodes, all_edges)
        if bridges:
            lines.append(f"\nCross-language bridges: {len(bridges)}")

        return "\n".join(lines)

    @mcp.tool(
        name="reindex",
        description=(
            "Re-index the codebase to pick up changes made since the server started. "
            "Use this after modifying files to refresh search results. "
            "Set update_qdrant=true to also update semantic search embeddings (slower)."
        ),
    )
    def reindex(update_qdrant: bool = False) -> str:
        """Re-index the codebase.

        Args:
            update_qdrant: If true, also update Qdrant embeddings (slower).
                          If false, only refreshes the in-memory symbol index.
        """
        store = update_qdrant and qdrant is not None

        if update_qdrant and qdrant is None:
            qdrant_note = " (Qdrant not available — skipping embedding update)"
        else:
            qdrant_note = ""

        result, new_nodes, new_edges = index_codebase(
            config, qdrant=qdrant, store_in_qdrant=store
        )

        # Update in-place so all tools see the new data
        all_nodes.clear()
        all_nodes.extend(new_nodes)
        all_edges.clear()
        all_edges.extend(new_edges)

        lines = [
            f"Reindex complete{qdrant_note}",
            f"  Files processed: {result.files_processed}",
            f"  Files skipped: {result.files_skipped}",
            f"  Symbols extracted: {result.nodes_extracted}",
            f"  Edges extracted: {result.edges_extracted}",
        ]

        if store:
            lines.append(f"  Symbols indexed in Qdrant: {result.nodes_indexed}")

        if result.errors:
            lines.append(f"  Errors: {len(result.errors)}")
            for err in result.errors[:5]:
                lines.append(f"    - {err}")

        return "\n".join(lines)

    # --- VCS History Tools ---

    if vcs is not None:

        @mcp.tool(
            name="git_log",
            description=(
                "Get commit history for a file or the entire repository. "
                "Shows revision, date, author, message, and files changed."
            ),
        )
        def git_log(file_path: str = "", limit: int = 20) -> str:
            """Get VCS commit log.

            Args:
                file_path: Optional path to filter commits (empty for all).
                limit: Maximum number of commits to return.
            """
            path = file_path if file_path else None
            commits = vcs.log(path=path, limit=limit)

            if not commits:
                return "No commits found."

            lines = []
            for c in commits:
                date = c.date.strftime("%Y-%m-%d")
                files = ", ".join(c.files_changed[:5])
                if len(c.files_changed) > 5:
                    files += f" (+{len(c.files_changed) - 5} more)"
                lines.append(f"[{c.revision[:8]}] {date} by {c.author}: {c.message}")
                if files:
                    lines.append(f"  files: {files}")

            return "\n".join(lines)

        @mcp.tool(
            name="git_blame",
            description=(
                "Get line-by-line authorship for a file. "
                "Shows who last modified each line."
            ),
        )
        def git_blame(file_path: str) -> str:
            """Get blame data for a file.

            Args:
                file_path: Path to the file to blame.
            """
            try:
                blame_lines = vcs.blame(file_path)
            except RuntimeError as e:
                return f"Error: {e}"

            if not blame_lines:
                return f"No blame data for {file_path}."

            lines = []
            for bl in blame_lines:
                lines.append(
                    f"L{bl.line_number:4d} | {bl.revision} | {bl.author:15s} | {bl.content}"
                )

            return "\n".join(lines)

        @mcp.tool(
            name="file_churn",
            description=(
                "Analyze which files change most frequently. "
                "High churn files are potential hotspots or areas of instability."
            ),
        )
        def file_churn(window_days: int = 90) -> str:
            """Analyze file change frequency.

            Args:
                window_days: How many days back to analyze (default: 90).
            """
            churn = vcs.churn(window_days=window_days)

            if not churn:
                return "No changes found in the specified window."

            lines = [f"File churn in last {window_days} days:\n"]
            for file_path, count in list(churn.items())[:30]:
                bar = "█" * min(count, 20)
                lines.append(f"  {count:4d} changes | {bar} | {file_path}")

            return "\n".join(lines)

    # --- Semantic Search Tools (require Qdrant) ---

    if qdrant is not None:

        @mcp.tool(
            name="search_code",
            description=(
                "Semantic search through code symbols using natural language. "
                "Best for conceptual queries like 'authentication logic' or 'database connection'. "
                "Uses MMR (Maximal Marginal Relevance) to return diverse results from "
                "different files/classes rather than many similar hits. "
                "For exact symbol names, use lookup_symbol or search_symbols instead."
            ),
        )
        def search_code(
            query: str,
            limit: int = 10,
            language: str = "",
            node_type: str = "",
        ) -> str:
            """Semantic code search with MMR diversity via Qdrant.

            Args:
                query: Natural language description of what you're looking for.
                limit: Maximum results to return (capped at 20).
                language: Optional language filter ('php', 'javascript', etc.).
                node_type: Optional type filter ('class', 'function', 'method').
            """
            limit = min(limit, 20)
            results = qdrant.search_code_mmr(
                query,
                limit=limit,
                language=language or None,
                node_type=node_type or None,
            )

            if not results:
                return f"No code matching '{query}' found."

            lines = []
            for r in results:
                score = r.get("score", 0)
                lines.append(
                    f"[{score:.2f}] {r.get('type', '?')}: {r.get('name', '?')} "
                    f"({r.get('file', '?')}:{r.get('lines', '?')})"
                )
                if r.get("summary"):
                    lines.append(f"  {r['summary']}")

            return "\n".join(lines)

        @mcp.tool(
            name="search_commits",
            description=(
                "Semantic search through commit messages. "
                "Finds commits related to a natural language topic."
            ),
        )
        def search_commits(query: str, limit: int = 10) -> str:
            """Semantic commit search via Qdrant.

            Args:
                query: Natural language description of what you're looking for.
                limit: Maximum results to return.
            """
            results = qdrant.search_commits(query, limit=limit)

            if not results:
                return f"No commits matching '{query}' found."

            lines = []
            for r in results:
                score = r.get("score", 0)
                lines.append(
                    f"[{r['revision'][:8]}] (relevance: {score:.2f}) "
                    f"by {r['author']}: {r['message']}"
                )
                files = r.get("files_changed", [])
                if files:
                    lines.append(f"  files: {', '.join(files[:5])}")

            return "\n".join(lines)

    # --- Resources ---

    @mcp.resource(
        "hammy://status",
        name="index_status",
        description="Current Hammy index status and statistics.",
        mime_type="text/plain",
    )
    def resource_status() -> str:
        """Return index status as a resource."""
        return index_status()

    return mcp
