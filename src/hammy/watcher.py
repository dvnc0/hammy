"""File system watcher for incremental codebase reindexing.

Watches the project directory for file changes and incrementally updates
the in-memory node/edge index and Qdrant embeddings without a full reindex.
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from hammy.config import HammyConfig
    from hammy.schema.models import Edge, Node
    from hammy.tools.qdrant_tools import QdrantManager


def _is_indexed_extension(path: Path, languages: list[str]) -> bool:
    """Return True if the file extension is handled by configured languages."""
    from hammy.tools.parser import ParserFactory

    factory = ParserFactory(languages)
    return factory.parse_file(path) is not None or path.suffix in {
        ".php", ".js", ".jsx", ".ts", ".tsx", ".py", ".go",
    }


def process_changed_files(
    changed_paths: set[Path],
    project_root: Path,
    config: "HammyConfig",
    all_nodes: list["Node"],
    all_edges: list["Edge"],
    file_node_ids: dict[str, list[str]],
    *,
    qdrant: "QdrantManager | None" = None,
    on_change: Callable[[str, int, int, int], None] | None = None,
) -> None:
    """Process a batch of changed/deleted file paths and update the index in place.

    This is the core incremental reindex logic, separated from the filesystem
    watch loop so it can be tested independently.

    Args:
        changed_paths: Absolute paths that changed or were deleted.
        project_root: Project root for computing relative paths.
        config: Hammy configuration.
        all_nodes: Mutable node list — updated in place.
        all_edges: Mutable edge list — updated in place.
        file_node_ids: Mutable mapping of rel_path -> [node_id, ...] — updated in place.
        qdrant: Optional QdrantManager for embedding updates.
        on_change: Callback(event_type, nodes_added, nodes_removed, errors) for UI.
    """
    from hammy.indexer.code_indexer import index_files

    files_to_reindex: list[Path] = []
    files_deleted: list[str] = []

    for path in changed_paths:
        rel = str(path.relative_to(project_root))
        if path.exists():
            files_to_reindex.append(path)
        else:
            files_deleted.append(rel)

    # Collect all relative paths affected (both changed and deleted)
    affected_rel = {
        str(p.relative_to(project_root)) for p in files_to_reindex
    } | set(files_deleted)

    # Remove old nodes/edges for affected files
    old_ids: set[str] = set()
    for rel in affected_rel:
        old_ids.update(file_node_ids.pop(rel, []))

    removed = len([n for n in all_nodes if n.id in old_ids])
    all_nodes[:] = [n for n in all_nodes if n.id not in old_ids]
    all_edges[:] = [
        e for e in all_edges
        if e.source not in old_ids and e.target not in old_ids
    ]

    # Remove from Qdrant
    if qdrant is not None:
        for rel in affected_rel:
            try:
                qdrant.delete_nodes_by_file(rel)
            except Exception:
                pass

    # Reindex surviving files
    new_nodes, new_edges, errors = index_files(files_to_reindex, config, project_root)

    all_nodes.extend(new_nodes)
    all_edges.extend(new_edges)
    for node in new_nodes:
        file_node_ids[node.loc.file].append(node.id)

    # Upsert new nodes to Qdrant
    if qdrant is not None and new_nodes:
        try:
            qdrant.upsert_nodes(new_nodes)
        except Exception:
            pass

    if on_change is not None:
        event_type = "deleted" if files_deleted and not files_to_reindex else "changed"
        on_change(event_type, len(new_nodes), removed, len(errors))


def watch_project(
    project_root: Path,
    config: "HammyConfig",
    all_nodes: list["Node"],
    all_edges: list["Edge"],
    *,
    qdrant: "QdrantManager | None" = None,
    debounce_seconds: float = 1.5,
    on_change: Callable[[str, int, int, int], None] | None = None,
    stop_event=None,
) -> None:
    """Watch project_root and incrementally reindex changed files.

    Mutates ``all_nodes`` and ``all_edges`` in place, and updates Qdrant
    when available. Blocks until ``stop_event`` is set or KeyboardInterrupt.

    Args:
        project_root: Directory to watch.
        config: Hammy configuration.
        all_nodes: Mutable node list from the initial index — updated in place.
        all_edges: Mutable edge list from the initial index — updated in place.
        qdrant: Optional QdrantManager for embedding updates.
        debounce_seconds: Wait this long after the last change before processing.
        on_change: Callback(event_type, nodes_added, nodes_removed, errors) for UI.
        stop_event: threading.Event that stops the loop when set.
    """
    from watchfiles import watch, Change

    from hammy.ignore import IgnoreManager

    ignore = IgnoreManager(project_root, config.ignore)

    # Track which nodes belong to each file for incremental removal
    file_node_ids: dict[str, list[str]] = defaultdict(list)
    for node in all_nodes:
        file_node_ids[node.loc.file].append(node.id)

    def _filter(change: Change, path: str) -> bool:
        p = Path(path)
        try:
            p.relative_to(project_root)
        except ValueError:
            return False
        if ignore.is_ignored(p):
            return False
        return p.suffix in {
            ".php", ".js", ".jsx", ".ts", ".tsx", ".py", ".go", ".mjs", ".cjs",
        }

    pending: dict[Path, float] = {}

    for changes in watch(project_root, watch_filter=_filter, stop_event=stop_event):
        now = time.monotonic()
        for _change_type, path_str in changes:
            pending[Path(path_str)] = now

        # Process paths whose debounce window has passed
        ready = {p for p, t in pending.items() if now - t >= debounce_seconds}
        if ready:
            for p in ready:
                pending.pop(p, None)
            process_changed_files(
                ready,
                project_root,
                config,
                all_nodes,
                all_edges,
                file_node_ids,
                qdrant=qdrant,
                on_change=on_change,
            )

        if stop_event is not None and stop_event.is_set():
            break
