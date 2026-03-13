"""Disk cache for the parsed node/edge index.

Saves the full node+edge graph to .hammy/index.json in the project root so
that hammy serve and hammy query can skip re-parsing on startup.

Usage:
    from hammy.indexer.index_cache import save_index, load_index

    # After indexing:
    save_index(project_root, nodes, edges)

    # At startup:
    cached = load_index(project_root)
    if cached:
        nodes, edges = cached
    else:
        # fall back to full parse
        ...
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from hammy.schema.models import Edge, Node

logger = logging.getLogger(__name__)

_CACHE_DIR = ".hammy"
_CACHE_FILE = "index.json"


def cache_path(project_root: Path) -> Path:
    return project_root / _CACHE_DIR / _CACHE_FILE


def save_index(project_root: Path, nodes: list[Node], edges: list[Edge]) -> Path:
    """Serialize nodes and edges to .hammy/index.json.

    Creates .hammy/ if it doesn't exist. Returns the path written.
    """
    path = cache_path(project_root)
    path.parent.mkdir(exist_ok=True)

    data = {
        "indexed_at": datetime.now(timezone.utc).isoformat(),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": [n.model_dump() for n in nodes],
        "edges": [e.model_dump() for e in edges],
    }

    path.write_text(json.dumps(data, separators=(",", ":")))
    logger.debug("Saved index cache: %d nodes, %d edges → %s", len(nodes), len(edges), path)
    return path


def load_index(project_root: Path) -> tuple[list[Node], list[Edge]] | None:
    """Load nodes and edges from .hammy/index.json.

    Returns None if the cache doesn't exist or is corrupt (caller should
    fall back to a full re-parse).
    """
    path = cache_path(project_root)
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text())
        nodes = [Node.model_validate(n) for n in data["nodes"]]
        edges = [Edge.model_validate(e) for e in data["edges"]]
        logger.debug("Loaded index cache: %d nodes, %d edges from %s", len(nodes), len(edges), path)
        return nodes, edges
    except Exception as exc:
        logger.warning("Index cache corrupt or unreadable (%s) — will re-parse", exc)
        return None


def cache_info(project_root: Path) -> dict | None:
    """Return metadata about the cache without loading all nodes/edges.

    Returns a dict with indexed_at, node_count, edge_count, or None if missing.
    """
    path = cache_path(project_root)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return {
            "indexed_at": data.get("indexed_at"),
            "node_count": data.get("node_count", 0),
            "edge_count": data.get("edge_count", 0),
            "path": str(path),
        }
    except Exception:
        return None
