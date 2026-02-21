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
        name="impact_analysis",
        description=(
            "Analyse the blast radius of changing a function or method. "
            "Traverses the call graph to show what code depends on a symbol (callers) "
            "or what the symbol depends on (callees), up to N hops deep. "
            "Use direction='callers' (default) to answer 'if I change X, what breaks?', "
            "direction='callees' to see what X depends on, or direction='both' for full neighbourhood."
        ),
    )
    def impact_analysis(
        symbol_name: str,
        depth: int = 3,
        direction: str = "callers",
    ) -> str:
        """Analyse the call-graph blast radius of a symbol.

        Args:
            symbol_name: Exact name of the function/method to analyse.
            depth: How many hops to traverse (1=direct only, default 3, max 6).
            direction: 'callers', 'callees', or 'both'.
        """
        depth = max(1, min(depth, 6))
        pattern = re.compile(r"\b" + re.escape(symbol_name) + r"\b", re.IGNORECASE)
        node_index = {n.id: n for n in all_nodes}
        name_index: dict[str, list[Node]] = {}
        for n in all_nodes:
            name_index.setdefault(n.name.lower(), []).append(n)

        call_edges = [e for e in all_edges if e.relation == RelationType.CALLS]

        def _find_callers(names: set[str], visited: set[str]) -> list[tuple[Node, str]]:
            found = []
            pats = {n: re.compile(r"\b" + re.escape(n) + r"\b", re.IGNORECASE) for n in names}
            for edge in call_edges:
                ctx = edge.metadata.context or ""
                for callee_name, p in pats.items():
                    if p.search(ctx):
                        caller = node_index.get(edge.source)
                        if caller and caller.id not in visited:
                            found.append((caller, callee_name))
                            break
            return found

        def _find_callees(node_ids: set[str], visited: set[str]) -> list[tuple[Node, str]]:
            found = []
            for edge in call_edges:
                if edge.source not in node_ids:
                    continue
                ctx = edge.metadata.context or ""
                callee_name = re.split(r"[:\.\s]", ctx)[-1].strip() if ctx else ""
                if not callee_name:
                    continue
                for n in name_index.get(callee_name.lower(), []):
                    if n.id not in visited:
                        found.append((n, ctx))
                        break
            return found

        lines: list[str] = []
        hop = 0

        if direction in ("callers", "both"):
            lines.append(f"=== Callers of '{symbol_name}' (what breaks if it changes) ===")
            visited: set[str] = set()
            current_names = {symbol_name}
            total_found = 0
            for hop in range(1, depth + 1):
                results = _find_callers(current_names, visited)
                if not results:
                    break
                lines.append(f"\nHop {hop}:")
                next_names: set[str] = set()
                for caller, callee in sorted(results, key=lambda x: x[0].loc.file):
                    visited.add(caller.id)
                    lines.append(
                        f"  {'  ' * (hop - 1)}{caller.type.value}: {caller.name} "
                        f"({caller.loc.file}:{caller.loc.lines[0]}) calls {callee}"
                    )
                    next_names.add(caller.name)
                    total_found += 1
                current_names = next_names
            if total_found == 0:
                lines.append(f"  No callers found for '{symbol_name}'.")
            else:
                lines.append(f"\nTotal callers found: {total_found} across {hop} hop(s).")

        if direction in ("callees", "both"):
            lines.append(f"\n=== Callees of '{symbol_name}' (what it depends on) ===")
            start_nodes = name_index.get(symbol_name.lower(), [])
            if not start_nodes:
                lines.append(f"  Definition of '{symbol_name}' not found in index.")
            else:
                visited_c: set[str] = {n.id for n in start_nodes}
                current_ids = visited_c.copy()
                total_c = 0
                for hop in range(1, depth + 1):
                    results_c = _find_callees(current_ids, visited_c)
                    if not results_c:
                        break
                    lines.append(f"\nHop {hop}:")
                    next_ids: set[str] = set()
                    for callee, ctx in sorted(results_c, key=lambda x: x[0].loc.file):
                        visited_c.add(callee.id)
                        next_ids.add(callee.id)
                        lines.append(
                            f"  {'  ' * (hop - 1)}{callee.type.value}: {callee.name} "
                            f"({callee.loc.file}:{callee.loc.lines[0]})"
                        )
                        total_c += 1
                    current_ids = next_ids
                if total_c == 0:
                    lines.append(f"  No known callees found for '{symbol_name}'.")
                else:
                    lines.append(f"\nTotal callees found: {total_c} across {hop} hop(s).")

        return "\n".join(lines) if lines else f"No call graph data found for '{symbol_name}'."

    @mcp.tool(
        name="structural_search",
        description=(
            "Filter code symbols by structural attributes: visibility, async, parameter count, "
            "return type, name regex, file path, or complexity. "
            "Examples: all public methods with 3+ params; async functions in controllers/; "
            "methods returning bool; classes with complexity > 10. "
            "All filters are optional and combine with AND. Leave blank to skip a filter."
        ),
    )
    def structural_search(
        node_type: str = "",
        language: str = "",
        visibility: str = "",
        async_only: bool = False,
        min_params: int = 0,
        max_params: int = -1,
        return_type: str = "",
        name_pattern: str = "",
        file_filter: str = "",
        min_complexity: int = 0,
        limit: int = 50,
    ) -> str:
        """Filter symbols by structural metadata.

        Args:
            node_type: 'class', 'function', 'method', or 'endpoint'.
            language: Language filter ('php', 'javascript', 'python', etc.).
            visibility: 'public', 'private', or 'protected'.
            async_only: If true, return only async functions/methods.
            min_params: Minimum number of parameters (0 = no minimum).
            max_params: Maximum number of parameters (-1 = no limit).
            return_type: Substring match on return type (e.g. 'bool', 'void', 'User').
            name_pattern: Regex pattern to match symbol names.
            file_filter: Path substring to restrict results (e.g. 'controllers/').
            min_complexity: Minimum complexity score (0 = no minimum).
            limit: Maximum results (capped at 200).
        """
        limit = min(limit, 200)
        name_re = re.compile(name_pattern, re.IGNORECASE) if name_pattern else None
        results: list[Node] = []

        for node in all_nodes:
            if node_type and node.type.value != node_type:
                continue
            if language and node.language != language:
                continue
            if visibility and (node.meta.visibility or "").lower() != visibility.lower():
                continue
            if async_only and not node.meta.is_async:
                continue
            param_count = len(node.meta.parameters)
            if param_count < min_params:
                continue
            if max_params >= 0 and param_count > max_params:
                continue
            if return_type and return_type.lower() not in (node.meta.return_type or "").lower():
                continue
            if name_re and not name_re.search(node.name):
                continue
            if file_filter and file_filter.lower() not in node.loc.file.lower():
                continue
            if min_complexity > 0 and (node.meta.complexity_score or 0) < min_complexity:
                continue
            results.append(node)

        if not results:
            return "No symbols matched the given filters."

        lines = [f"{len(results)} symbol(s) matched:\n"]
        for n in results[:limit]:
            parts = [f"{n.type.value}: {n.name} ({n.loc.file}:{n.loc.lines[0]})"]
            attrs: list[str] = []
            if n.meta.visibility:
                attrs.append(n.meta.visibility)
            if n.meta.is_async:
                attrs.append("async")
            if n.meta.parameters:
                attrs.append(f"{len(n.meta.parameters)} params")
            if n.meta.return_type:
                attrs.append(f"-> {n.meta.return_type}")
            if n.meta.complexity_score is not None:
                attrs.append(f"complexity={n.meta.complexity_score}")
            if attrs:
                parts.append(f"  [{', '.join(attrs)}]")
            if n.summary:
                parts.append(f"  {n.summary}")
            lines.append("\n".join(parts))

        if len(results) > limit:
            lines.append(f"\n... and {len(results) - limit} more. Narrow with additional filters.")

        return "\n\n".join(lines)

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
        name="hotspot_score",
        description=(
            "Rank code symbols by composite risk: caller_count × file_churn. "
            "High-scoring symbols are both heavily depended upon AND frequently changed — "
            "the highest-risk places to touch. Uses VCS history for churn when available. "
            "Filter by node_type, language, or file_filter to focus on a subsystem."
        ),
    )
    def hotspot_score(
        top_n: int = 20,
        node_type: str = "",
        language: str = "",
        file_filter: str = "",
        window_days: int = 90,
    ) -> str:
        """Compute composite hotspot scores for code symbols.

        Args:
            top_n: Number of top hotspots to return (capped at 50).
            node_type: Optional type filter ('function', 'method', 'class').
            language: Optional language filter.
            file_filter: Optional path substring to restrict results.
            window_days: Churn lookback window in days (requires VCS).
        """
        from hammy.tools.hotspot import compute_hotspots

        top_n = min(top_n, 50)

        # Get file-level churn from VCS if available
        file_churn: dict[str, int] | None = None
        if vcs is not None:
            try:
                file_churn = dict(vcs.churn(window_days=window_days))
            except Exception:
                pass

        results = compute_hotspots(
            all_nodes,
            all_edges,
            file_churn=file_churn,
            node_type=node_type,
            language=language,
            file_filter=file_filter,
            top_n=top_n,
        )

        if not results:
            return "No symbols found matching the given filters."

        churn_note = f" (churn window: {window_days}d)" if file_churn else " (no VCS churn data — scoring by callers only)"
        lines = [f"Top {len(results)} hotspots{churn_note}:\n"]

        for rank, r in enumerate(results, 1):
            attrs = []
            if r["visibility"]:
                attrs.append(r["visibility"])
            if r["is_async"]:
                attrs.append("async")
            attr_str = f" [{', '.join(attrs)}]" if attrs else ""
            lines.append(
                f"#{rank:2d} [score={r['score']:.1f}] "
                f"{r['type']}: {r['name']}{attr_str}\n"
                f"     {r['file']}:{r['lines'][0]}\n"
                f"     callers: {r['caller_count']}  |  churn: {r['churn_rate']}"
            )
            if r["summary"]:
                lines.append(f"     {r['summary']}")

        return "\n\n".join(lines)

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
            "Set update_qdrant=true to also update semantic search embeddings (slower). "
            "Set enrich=true to generate LLM summaries for newly indexed symbols "
            "(requires ANTHROPIC_API_KEY and update_qdrant=true)."
        ),
    )
    def reindex(update_qdrant: bool = False, enrich: bool = False) -> str:
        """Re-index the codebase.

        Args:
            update_qdrant: If true, also update Qdrant embeddings (slower).
                          If false, only refreshes the in-memory symbol index.
            enrich: If true, generate LLM summaries for symbols after indexing.
                   Requires update_qdrant=true and a configured API key.
        """
        store = update_qdrant and qdrant is not None

        if update_qdrant and qdrant is None:
            qdrant_note = " (Qdrant not available — skipping embedding update)"
        else:
            qdrant_note = ""

        run_enrich = enrich and store
        if enrich and not store:
            qdrant_note += " (enrich requires update_qdrant=true)"

        result, new_nodes, new_edges = index_codebase(
            config, qdrant=qdrant, store_in_qdrant=store, enrich=run_enrich
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

        if run_enrich:
            lines.append(f"  Symbols enriched with LLM summaries: {result.nodes_enriched}")

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

    # --- PR / Diff Analysis ---

    @mcp.tool(
        name="pr_diff",
        description=(
            "Analyse a pull request or diff to show what symbols changed and their blast radius. "
            "Pass a raw unified diff via diff_text (paste from 'git diff' or GitHub), OR "
            "provide base_ref (e.g. 'main', 'HEAD~3') to auto-compute the diff from VCS. "
            "Returns: changed files, modified symbols, and their callers (who is affected)."
        ),
    )
    def pr_diff(
        diff_text: str = "",
        base_ref: str = "",
        head_ref: str = "",
        depth: int = 2,
    ) -> str:
        """Analyse a diff and return symbol-level impact.

        Args:
            diff_text: Raw unified diff text (paste from git diff / GitHub).
            base_ref: Base git ref to diff from (e.g. 'main', 'HEAD~1').
                      Used when diff_text is empty and VCS is available.
            head_ref: Head ref to diff to (default: working tree / HEAD).
            depth: Caller traversal depth for impact analysis (default 2).
        """
        from hammy.tools.diff_analysis import analyze_diff

        raw_diff = diff_text.strip()

        # If no diff_text, try to fetch from VCS
        if not raw_diff:
            if vcs is None:
                return (
                    "No diff_text provided and VCS is not available. "
                    "Paste a unified diff using the diff_text parameter."
                )
            if not base_ref:
                return (
                    "Provide either diff_text (raw unified diff) or base_ref "
                    "(e.g. 'main', 'HEAD~1') to compute the diff automatically."
                )
            try:
                head = head_ref if head_ref else "HEAD"
                raw_diff = vcs.diff(base_ref, head)
            except Exception as e:
                return f"Failed to compute diff from VCS: {e}"

        if not raw_diff:
            return "Diff is empty — no changes to analyse."

        report = analyze_diff(raw_diff, all_nodes, all_edges, depth=depth)

        if not report.changed_files:
            return "Could not parse any changed files from the diff."

        lines: list[str] = []

        # --- Summary header ---
        total_symbols = len(report.all_changed_symbols)
        total_files = len(report.changed_files)
        lines.append(f"PR Diff Analysis  ({total_files} file(s) changed, {total_symbols} symbol(s) detected)\n")

        # --- Changed files ---
        lines.append("Changed files:")
        for cf in report.changed_files:
            sym_note = f"  [{', '.join(cf.changed_symbols[:5])}{'…' if len(cf.changed_symbols) > 5 else ''}]" if cf.changed_symbols else ""
            lines.append(f"  [{cf.change_type:8s}] {cf.path}{sym_note}")

        # --- Impact per symbol ---
        indexed = [r for r in report.impact if r["indexed"]]
        unindexed = [r for r in report.impact if not r["indexed"]]

        if indexed:
            lines.append(f"\nImpact analysis (depth={depth}):\n")
            for r in indexed:
                caller_count = r["caller_count"]
                risk = "HIGH" if caller_count >= 5 else "MED" if caller_count >= 2 else "LOW"
                attrs = []
                if r.get("visibility"):
                    attrs.append(r["visibility"])
                attr_str = f" [{', '.join(attrs)}]" if attrs else ""
                lines.append(
                    f"  [{risk}] {r['type']}: {r['symbol']}{attr_str}  "
                    f"({r['file']}:{r.get('line', '?')})  callers={caller_count}"
                )
                if r.get("summary"):
                    lines.append(f"         {r['summary']}")
                for caller in r["callers"][:5]:
                    lines.append(
                        f"         ← {caller['type']}: {caller['name']} "
                        f"({caller['file']}:{caller['line']})"
                    )
                if caller_count > 5:
                    lines.append(f"         … and {caller_count - 5} more callers")

        if unindexed:
            lines.append(f"\nNew/unindexed symbols (not yet in graph):")
            for r in unindexed:
                lines.append(f"  + {r['symbol']}")

        # Overall risk summary
        high_risk = [r for r in indexed if r["caller_count"] >= 5]
        if high_risk:
            lines.append(f"\n⚠  {len(high_risk)} HIGH-RISK symbol(s) changed (5+ callers):")
            for r in high_risk:
                lines.append(f"   • {r['symbol']} — {r['caller_count']} callers")

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
            name="search_code_hybrid",
            description=(
                "Hybrid search combining BM25 keyword matching with semantic embeddings. "
                "Use when you want both exact keyword precision (e.g. variable names, method "
                "names) AND conceptual similarity. Results are merged via Reciprocal Rank "
                "Fusion so highly-ranked in either list floats to the top. "
                "Prefer this over search_code when the query mixes exact terms and concepts."
            ),
        )
        def search_code_hybrid(
            query: str,
            limit: int = 10,
            language: str = "",
            node_type: str = "",
        ) -> str:
            """Hybrid BM25 + semantic code search with RRF fusion.

            Args:
                query: Keywords or natural language description.
                limit: Maximum results to return (capped at 20).
                language: Optional language filter.
                node_type: Optional type filter ('class', 'function', 'method').
            """
            from hammy.tools.hybrid_search import hybrid_search

            limit = min(limit, 20)
            results = hybrid_search(
                query,
                all_nodes,
                qdrant=qdrant,
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
                    f"[{score:.3f}] {r.get('type', '?')}: {r.get('name', '?')} "
                    f"({r.get('file', '?')}:{r.get('lines', '?')})"
                )
                if r.get("summary"):
                    lines.append(f"  {r['summary']}")

            return "\n".join(lines)

        @mcp.tool(
            name="store_context",
            description=(
                "Store a research finding or discovered context in the brain (persistent memory). "
                "Each entry has a key for direct retrieval and is semantically indexed so related "
                "findings can be discovered by concept. Upserting the same key overwrites the "
                "previous entry. Use tags to group related findings (e.g. task name, sprint). "
                "Sub-agents can retrieve this by key using recall_context."
            ),
        )
        def store_context(
            key: str,
            content: str,
            tags: str = "",
            source_files: str = "",
        ) -> str:
            """Store a finding in the brain.

            Args:
                key: Unique identifier for this entry (e.g. 'payment-flow-research').
                content: The discovered information to store.
                tags: Comma-separated labels for grouping (e.g. 'payment,sprint-42').
                source_files: Comma-separated file paths this entry relates to.
            """
            tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
            file_list = [f.strip() for f in source_files.split(",") if f.strip()] if source_files else []

            qdrant.upsert_brain_entry(key, content, tags=tag_list, source_files=file_list)

            tag_note = f" [tags: {', '.join(tag_list)}]" if tag_list else ""
            return f"Stored '{key}'{tag_note}. Retrieve with: recall_context(key='{key}')"

        @mcp.tool(
            name="recall_context",
            description=(
                "Retrieve stored research from the brain. "
                "Fetch by exact key for direct lookup, or use a natural language query "
                "to find semantically related findings. Optionally filter by tag. "
                "Use after store_context to hand off context to sub-agents or resume work."
            ),
        )
        def recall_context(
            query: str = "",
            key: str = "",
            tag: str = "",
            limit: int = 5,
        ) -> str:
            """Retrieve brain entries by key or semantic query.

            Args:
                query: Natural language query to find related findings.
                key: Exact key for direct lookup (takes priority over query).
                tag: Optional tag to restrict results.
                limit: Max results for semantic search.
            """
            if not query and not key:
                return "Provide either a key (exact lookup) or a query (semantic search)."

            results = qdrant.search_brain(query, key=key, tag=tag, limit=min(limit, 10))

            if not results:
                if key:
                    return f"No brain entry found for key '{key}'."
                return f"No brain entries found matching '{query}'."

            lines = []
            for r in results:
                score = r.get("score")
                header = f"[{r['key']}]"
                if score is not None:
                    header += f" (relevance: {score:.2f})"
                if r.get("tags"):
                    header += f" tags: {', '.join(r['tags'])}"
                lines.append(header)
                lines.append(r["content"])
                if r.get("source_files"):
                    lines.append(f"  files: {', '.join(r['source_files'])}")
                lines.append(f"  stored: {r.get('created_at', '?')[:19]}")
                lines.append("")

            return "\n".join(lines).strip()

        @mcp.tool(
            name="list_context",
            description=(
                "List all keys and summaries stored in the brain. "
                "Use to see what research has been accumulated, optionally filtered by tag. "
                "Then use recall_context(key=...) to fetch the full content of any entry."
            ),
        )
        def list_context(tag: str = "") -> str:
            """List stored brain entries.

            Args:
                tag: Optional tag to restrict results.
            """
            entries = qdrant.list_brain_entries(tag=tag)

            if not entries:
                note = f" with tag '{tag}'" if tag else ""
                return f"No brain entries{note}. Use store_context to save findings."

            lines = [f"{len(entries)} brain {'entry' if len(entries) == 1 else 'entries'}:\n"]
            for e in entries:
                created = e.get("created_at", "")[:10]
                tag_note = f" [{', '.join(e['tags'])}]" if e.get("tags") else ""
                # First line of content as summary
                summary = e["content"].splitlines()[0][:80]
                if len(e["content"]) > 80:
                    summary += "…"
                lines.append(f"  {e['key']}{tag_note}  ({created})")
                lines.append(f"    {summary}")

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
