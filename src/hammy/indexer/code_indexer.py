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
    nodes_enriched: int = 0
    errors: list[str] = field(default_factory=list)


def index_codebase(
    config: HammyConfig,
    *,
    qdrant: QdrantManager | None = None,
    store_in_qdrant: bool = True,
    enrich: bool = False,
    progress_callback=None,
) -> tuple[IndexResult, list[Node], list[Edge]]:
    """Run the full code indexing pipeline.

    Args:
        config: Hammy configuration.
        qdrant: Optional QdrantManager instance (created from config if None).
        store_in_qdrant: Whether to store results in Qdrant.
        enrich: Whether to run LLM enrichment after indexing.
        progress_callback: Optional fn(completed, total) for enrichment progress.

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
            qdrant = QdrantManager(config.qdrant, project_name=config.project.name)
        qdrant.ensure_collections()
        result.nodes_indexed = qdrant.upsert_nodes(all_nodes)

    if enrich and all_nodes:
        from hammy.indexer.enricher import enrich_nodes

        enriched_count, enrich_errors = enrich_nodes(
            all_nodes,
            project_root,
            config.enrichment,
            progress_callback=progress_callback,
        )
        result.nodes_enriched = enriched_count
        result.errors.extend(enrich_errors)

        # Re-upsert with enriched summaries so embeddings reflect the new text
        if store_in_qdrant and enriched_count > 0:
            if qdrant is None:
                qdrant = QdrantManager(config.qdrant, project_name=config.project.name)
            qdrant.upsert_nodes(all_nodes)

    return result, all_nodes, all_edges


def index_files(
    file_paths: list[Path],
    config: HammyConfig,
    project_root: Path,
) -> tuple[list[Node], list[Edge], list[str]]:
    """Parse and index a specific set of files (for incremental / watch-mode updates).

    Does NOT write to Qdrant — caller handles that.

    Args:
        file_paths: Absolute paths to the files to index.
        config: Hammy configuration (used for language settings).
        project_root: Project root for computing relative paths.

    Returns:
        Tuple of (nodes, edges, errors).
    """
    parser_factory = ParserFactory(config.parsing.languages)
    nodes: list[Node] = []
    edges: list[Edge] = []
    errors: list[str] = []

    for path in file_paths:
        if not path.exists():
            continue
        parsed = parser_factory.parse_file(path)
        if parsed is None:
            continue
        tree, language = parsed
        try:
            rel_path = str(path.relative_to(project_root))
            file_nodes, file_edges = extract_symbols(tree, language, rel_path)
            nodes.extend(file_nodes)
            edges.extend(file_edges)
        except Exception as e:
            errors.append(f"{path}: {e}")

    return nodes, edges, errors
