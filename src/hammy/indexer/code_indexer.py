"""Code indexing pipeline — walks files, parses ASTs, stores in Qdrant.

This is the main indexing entry point that ties together the file walker,
tree-sitter parser, and Qdrant storage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from hammy.config import HammyConfig
from hammy.ignore import IgnoreManager
from hammy.indexer.file_walker import walk_project
from hammy.schema.models import Edge, Node
from hammy.tools.ast_tools import extract_symbols
from hammy.tools.parser import ParserFactory
from hammy.tools.qdrant_tools import QdrantManager


@dataclass
class IndexResult:
    """Results from an indexing run."""

    files_processed: int = 0
    files_skipped: int = 0
    nodes_extracted: int = 0
    edges_extracted: int = 0
    nodes_indexed: int = 0
    errors: list[str] = field(default_factory=list)


def index_codebase(
    config: HammyConfig,
    *,
    qdrant: QdrantManager | None = None,
    store_in_qdrant: bool = True,
) -> tuple[IndexResult, list[Node], list[Edge]]:
    """Run the full code indexing pipeline.

    Args:
        config: Hammy configuration.
        qdrant: Optional QdrantManager instance (created from config if None).
        store_in_qdrant: Whether to store results in Qdrant.

    Returns:
        Tuple of (result stats, all nodes, all edges).
    """
    project_root = Path(config.project.root).resolve()
    ignore_manager = IgnoreManager(project_root, config.ignore)
    parser_factory = ParserFactory(config.parsing.languages)

    result = IndexResult()
    all_nodes: list[Node] = []
    all_edges: list[Edge] = []

    for file_entry in walk_project(
        project_root,
        ignore_manager,
        max_file_size_kb=config.parsing.max_file_size_kb,
        languages=config.parsing.languages,
    ):
        parsed = parser_factory.parse_file(file_entry.path)
        if parsed is None:
            result.files_skipped += 1
            continue

        tree, language = parsed
        rel_path = str(file_entry.path.relative_to(project_root))

        try:
            nodes, edges = extract_symbols(tree, language, rel_path)
            all_nodes.extend(nodes)
            all_edges.extend(edges)
            result.files_processed += 1
            result.nodes_extracted += len(nodes)
            result.edges_extracted += len(edges)
        except Exception as e:
            result.errors.append(f"{rel_path}: {e}")
            result.files_skipped += 1

    if store_in_qdrant and all_nodes:
        if qdrant is None:
            qdrant = QdrantManager(config.qdrant)
        qdrant.ensure_collections()
        result.nodes_indexed = qdrant.upsert_nodes(all_nodes)

    return result, all_nodes, all_edges
