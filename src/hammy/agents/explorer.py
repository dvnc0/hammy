"""Explorer agent tools — AST queries, dependency mapping, bridge resolution.

These are CrewAI @tool functions that wrap the underlying parser and
bridge logic for use by the Explorer agent.
"""

from __future__ import annotations

import re
from pathlib import Path

from crewai.tools import tool

from hammy.schema.models import Node, NodeType
from hammy.tools.ast_tools import extract_symbols
from hammy.tools.parser import ParserFactory


def make_explorer_tools(
    project_root: Path,
    parser_factory: ParserFactory,
    all_nodes: list[Node],
    all_edges: list,
    qdrant=None,
) -> list:
    """Create Explorer agent tools bound to the current project context."""

    @tool("AST Query")
    def ast_query(file_path: str, query_type: str = "all") -> str:
        """Query the AST of a specific file. Returns structured information about code elements.

        Args:
            file_path: Path to the file (relative to project root).
            query_type: What to extract - 'all', 'classes', 'functions', 'methods', 'endpoints', or 'imports'.
        """
        full_path = project_root / file_path
        if not full_path.exists():
            return f"File not found: {file_path}"

        result = parser_factory.parse_file(full_path)
        if result is None:
            return f"Unsupported file type: {file_path}"

        tree, lang = result
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
            from hammy.schema.models import RelationType
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

    @tool("Search Code Symbols")
    def search_symbols(
        query: str,
        language: str = "",
        node_type: str = "",
        file_filter: str = "",
    ) -> str:
        """Search for code symbols by name or keyword with ranked results.

        Results are ranked: exact name matches appear first, then prefix matches,
        then substring matches, then summary matches. Use node_type to narrow to
        a specific kind (class/function/method/endpoint). Use file_filter to restrict
        to a directory or filename substring. For exact definition lookup of a known
        symbol name, prefer lookup_symbol instead.

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
            line = f"{n.type.value}: {n.name} ({n.loc.file}:{n.loc.lines[0]})"
            if n.summary:
                line += f" | {n.summary}"
            lines.append(line)

        if len(results) > 25:
            lines.append(f"\n... and {len(results) - 25} more. Use file_filter or node_type to narrow.")

        return "\n".join(lines)

    @tool("Lookup Symbol")
    def lookup_symbol(name: str, node_type: str = "") -> str:
        """Look up the exact definition of a known symbol by its precise name.

        Returns file, line numbers, parameters, return type, and visibility.
        Use this when you know the exact name (e.g. 'getRenew', 'UserController').
        For fuzzy/keyword search, use search_symbols instead.

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

    @tool("Find Usages")
    def find_usages(symbol_name: str, file_filter: str = "") -> str:
        """Find all call sites of a specific function or method by exact name.

        Uses word-boundary matching so 'save' won't match 'saveAll' or 'isSaved'.
        Use file_filter to restrict results to a directory or filename substring.
        Returns the containing function/method and file location for each call site.

        Args:
            symbol_name: Exact name of the function/method to find call sites for.
            file_filter: Optional path substring to restrict results (e.g. 'controllers/').
        """
        from hammy.schema.models import RelationType

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

    @tool("Impact Analysis")
    def impact_analysis(
        symbol_name: str,
        depth: int = 3,
        direction: str = "callers",
    ) -> str:
        """Analyse the blast radius of changing a function or method.

        Traverses the call graph to show what code depends on a symbol (callers)
        or what the symbol depends on (callees), up to N hops deep. Use this to
        answer "if I change getRenew, what else is affected?"

        Args:
            symbol_name: Exact name of the function/method to analyse.
            depth: How many hops to traverse (1=direct only, default 3).
            direction: 'callers' (what depends on X), 'callees' (what X depends on),
                       or 'both'.
        """
        from hammy.schema.models import RelationType

        depth = max(1, min(depth, 6))
        pattern = re.compile(r"\b" + re.escape(symbol_name) + r"\b", re.IGNORECASE)
        node_index = {n.id: n for n in all_nodes}
        name_index: dict[str, list[Node]] = {}
        for n in all_nodes:
            name_index.setdefault(n.name.lower(), []).append(n)

        call_edges = [e for e in all_edges if e.relation == RelationType.CALLS]

        def _find_callers(names: set[str], visited: set[str]) -> list[tuple[Node, str]]:
            """Return (caller_node, callee_name) pairs for a set of callee names."""
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
            """Return (callee_node, context) pairs for a set of source node IDs."""
            found = []
            for edge in call_edges:
                if edge.source not in node_ids:
                    continue
                ctx = edge.metadata.context or ""
                # Extract callee name: last identifier in context (foo, obj.foo, Class::foo)
                callee_name = re.split(r"[:\.\s]", ctx)[-1].strip() if ctx else ""
                if not callee_name:
                    continue
                for n in name_index.get(callee_name.lower(), []):
                    if n.id not in visited:
                        found.append((n, ctx))
                        break
            return found

        lines: list[str] = []

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
            # Find the starting node(s) by name
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

    @tool("Find Cross-Language Bridges")
    def find_bridges() -> str:
        """Find all cross-language connections (e.g., JS fetch calls matching PHP routes).

        Returns bridge relationships between frontend and backend code.
        """
        from hammy.tools.bridge import resolve_bridges

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

    @tool("List Files")
    def list_files(language: str = "") -> str:
        """List all indexed files, optionally filtered by language.

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

    core_tools = [ast_query, search_symbols, lookup_symbol, find_usages, impact_analysis, find_bridges, list_files]

    if qdrant is None:
        return core_tools

    # --- Brain tools (require Qdrant) ---

    @tool("Store Context")
    def store_context(
        key: str,
        content: str,
        tags: str = "",
        source_files: str = "",
    ) -> str:
        """Store a research finding in the brain (persistent memory).

        Each entry has a key for direct retrieval and is semantically indexed.
        Upserting the same key overwrites the previous entry. Sub-agents can
        retrieve this by key using recall_context.

        Args:
            key: Unique identifier (e.g. 'payment-flow-research').
            content: The discovered information to store.
            tags: Comma-separated labels for grouping (e.g. 'payment,sprint-42').
            source_files: Comma-separated file paths this entry relates to.
        """
        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
        file_list = [f.strip() for f in source_files.split(",") if f.strip()] if source_files else []
        qdrant.upsert_brain_entry(key, content, tags=tag_list, source_files=file_list)
        tag_note = f" [tags: {', '.join(tag_list)}]" if tag_list else ""
        return f"Stored '{key}'{tag_note}. Retrieve with: recall_context(key='{key}')"

    @tool("Recall Context")
    def recall_context(
        query: str = "",
        key: str = "",
        tag: str = "",
        limit: int = 5,
    ) -> str:
        """Retrieve stored research from the brain.

        Fetch by exact key for direct lookup, or use a natural language query
        to find semantically related findings. Optionally filter by tag.

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

    @tool("List Context")
    def list_context(tag: str = "") -> str:
        """List all keys and summaries stored in the brain.

        Use to see what research has been accumulated, optionally filtered
        by tag. Then use recall_context(key=...) to fetch the full content.

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
            summary = e["content"].splitlines()[0][:80]
            if len(e["content"]) > 80:
                summary += "…"
            lines.append(f"  {e['key']}{tag_note}  ({created})")
            lines.append(f"    {summary}")

        return "\n".join(lines)

    return [*core_tools, store_context, recall_context, list_context]
