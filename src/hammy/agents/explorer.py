"""Explorer agent tools — AST queries, dependency mapping, bridge resolution.

These are CrewAI @tool functions that wrap the underlying parser and
bridge logic for use by the Explorer agent.
"""

from __future__ import annotations

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
    def search_symbols(query: str, language: str = "", node_type: str = "") -> str:
        """Search for code symbols by name or description in the indexed codebase.

        Args:
            query: Search term (symbol name, description, or keyword).
            language: Optional language filter ('php' or 'javascript').
            node_type: Optional type filter ('class', 'function', 'method', 'endpoint').
        """
        results = []
        query_lower = query.lower()

        for node in all_nodes:
            if language and node.language != language:
                continue
            if node_type and node.type.value != node_type:
                continue
            if query_lower in node.name.lower() or query_lower in node.summary.lower():
                results.append(node)

        if not results:
            return f"No symbols matching '{query}' found."

        lines = []
        for n in results[:20]:  # Limit results
            line = f"{n.type.value}: {n.name} ({n.loc.file}:{n.loc.lines[0]})"
            if n.summary:
                line += f" | {n.summary}"
            lines.append(line)

        return "\n".join(lines)

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

    return [ast_query, search_symbols, find_bridges, list_files]
