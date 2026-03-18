"""FastAPI visualization server for Hammy's call graph index."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from hammy.indexer.index_cache import load_index
from hammy.schema.models import Edge, EdgeMetadata, NodeType, RelationType
from hammy.tools.hotspot import compute_hotspots


def _resolve_edges(nodes: list, edges: list) -> list:
    """Replace synthetic CALLS/EXTENDS/IMPORTS targets with real node IDs.

    Parsers store CALLS edge targets as Node.make_id("", callee_name) which
    never matches any real node.  This resolves them by extracting the bare
    callee name from edge.metadata.context and matching against known nodes,
    mirroring the logic used by _find_callers in the MCP server.
    """
    # bare_name_lower → [node_id]
    name_to_ids: dict[str, list[str]] = defaultdict(list)
    for n in nodes:
        if n.type == NodeType.COMMENT:
            continue
        bare = re.split(r"::|\\|\.", n.name)[-1].lower()
        name_to_ids[bare].append(n.id)

    resolved: list = []
    for e in edges:
        if e.relation not in (RelationType.CALLS, RelationType.EXTENDS, RelationType.IMPLEMENTS):
            resolved.append(e)
            continue

        ctx = e.metadata.context or ""
        # Extract bare callee name: last word before "("
        m = re.findall(r"\b(\w+)\s*\(", ctx)
        bare = (m[-1] if m else re.split(r"[:\\.\\s]", ctx)[-1].strip()).lower()
        targets = [t for t in name_to_ids.get(bare, []) if t != e.source]

        if not targets:
            # Nothing resolved — keep original (won't traverse but won't break)
            resolved.append(e)
            continue

        for t_id in targets:
            resolved.append(Edge(
                source=e.source,
                target=t_id,
                relation=e.relation,
                metadata=EdgeMetadata(
                    confidence=e.metadata.confidence,
                    context=e.metadata.context,
                    is_bridge=e.metadata.is_bridge,
                ),
            ))

    return resolved


def create_viz_app(project_root: Path) -> FastAPI:
    """Create and configure the FastAPI visualization app.

    Loads the index from .hammy/index.json at startup — raises RuntimeError
    if no cache exists.
    """
    cached = load_index(project_root)
    if not cached:
        raise RuntimeError(
            f"No index found at {project_root}/.hammy/index.json\n"
            "Run `hammy index` first to build the index."
        )
    nodes, raw_edges = cached

    # Resolve synthetic CALLS targets → real node IDs
    edges = _resolve_edges(nodes, raw_edges)

    # ── Lookup structures ──────────────────────────────────────────────────
    node_by_id: dict[str, Any] = {n.id: n for n in nodes}

    outgoing: dict[str, list] = defaultdict(list)
    incoming: dict[str, list] = defaultdict(list)
    for e in edges:
        outgoing[e.source].append(e)
        incoming[e.target].append(e)

    degree: dict[str, int] = defaultdict(int)
    for e in edges:
        degree[e.source] += 1
        degree[e.target] += 1

    # Pre-compute hotspot scores — use raw_edges (hotspot uses name matching internally)
    hotspots = compute_hotspots(nodes, raw_edges, top_n=50)
    hotspot_score: dict[str, float] = {h["node_id"]: h["score"] for h in hotspots}

    # Comments indexed by parent_symbol name for O(1) lookup
    comments_by_symbol: dict[str, list] = defaultdict(list)
    for n in nodes:
        if n.type == NodeType.COMMENT and n.meta.parent_symbol:
            comments_by_symbol[n.meta.parent_symbol].append(n)

    # ── App ────────────────────────────────────────────────────────────────
    app = FastAPI(title="Hammy Viz", docs_url=None, redoc_url=None)

    @app.get("/api/stats")
    def get_stats() -> dict[str, Any]:
        symbol_nodes = [n for n in nodes if n.type != NodeType.COMMENT]
        files = len({n.loc.file for n in symbol_nodes})
        return {
            "symbols": len(symbol_nodes),
            "comments": sum(1 for n in nodes if n.type == NodeType.COMMENT),
            "edges": len(edges),
            "files": files,
            "project": project_root.name,
        }

    @app.get("/api/search")
    def search(
        q: str = Query(""),
        limit: int = Query(20, ge=1, le=100),
    ) -> list[dict[str, Any]]:
        if not q or len(q) < 2:
            return []
        q_lower = q.lower()
        results = []
        for n in nodes:
            if n.type == NodeType.COMMENT:
                continue
            if q_lower in n.name.lower():
                results.append({
                    "id": n.id,
                    "name": n.name,
                    "type": n.type.value,
                    "file": n.loc.file,
                    "lines": list(n.loc.lines),
                    "language": n.language,
                    "degree": degree[n.id],
                })
        # Sort: exact prefix matches first, then by degree descending
        results.sort(key=lambda r: (
            0 if r["name"].lower().startswith(q_lower) else 1,
            -r["degree"],
            r["name"].lower(),
        ))
        return results[:limit]

    @app.get("/api/graph")
    def get_subgraph(
        name: str = Query(...),
        hops: int = Query(2, ge=1, le=4),
        limit: int = Query(50, ge=1, le=200),
    ) -> dict[str, Any]:
        """Return a BFS subgraph centered on all symbols matching `name`."""
        name_lower = name.lower()
        # Exact match first, then substring
        seeds = [n for n in nodes if n.type != NodeType.COMMENT and n.name.lower() == name_lower]
        if not seeds:
            seeds = [n for n in nodes if n.type != NodeType.COMMENT and name_lower in n.name.lower()]
        if not seeds:
            raise HTTPException(status_code=404, detail=f"Symbol '{name}' not found")

        seed_ids = {s.id for s in seeds}
        visited_ids: set[str] = set(seed_ids)
        frontier = set(seed_ids)
        included_edge_keys: set[tuple[str, str]] = set()
        raw_edges: list = []

        for _ in range(hops):
            next_frontier: set[str] = set()
            for nid in frontier:
                for e in outgoing[nid] + incoming[nid]:
                    key = (e.source, e.target)
                    if key not in included_edge_keys:
                        included_edge_keys.add(key)
                        raw_edges.append(e)
                    other = e.target if e.source == nid else e.source
                    other_node = node_by_id.get(other)
                    if other_node and other_node.type != NodeType.COMMENT and other not in visited_ids:
                        next_frontier.add(other)
            visited_ids.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break

        # Cap: keep seeds + highest-degree remainder
        non_seed = [nid for nid in visited_ids if nid not in seed_ids]
        non_seed.sort(key=lambda nid: -degree[nid])
        all_ids = list(seed_ids) + non_seed[: max(0, limit - len(seed_ids))]
        all_ids_set = set(all_ids)

        result_nodes = []
        for nid in all_ids:
            n = node_by_id.get(nid)
            if not n:
                continue
            coms = comments_by_symbol.get(n.name, [])
            result_nodes.append({
                "id": n.id,
                "name": n.name,
                "type": n.type.value,
                "file": n.loc.file,
                "lines": list(n.loc.lines),
                "language": n.language,
                "summary": n.summary,
                "degree": degree[n.id],
                "hotspot": round(hotspot_score.get(n.id, 0.0), 2),
                "is_seed": n.id in seed_ids,
                "comment_count": len(coms),
                "comments": [{"line": c.loc.lines[0], "text": c.name} for c in coms],
            })

        result_edges = [
            {
                "source": e.source,
                "target": e.target,
                "relation": e.relation.value,
            }
            for e in raw_edges
            if e.source in all_ids_set and e.target in all_ids_set
        ]

        return {"nodes": result_nodes, "edges": result_edges}

    @app.get("/api/hotspots")
    def get_hotspots(top_n: int = Query(20, ge=1, le=50)) -> list[dict[str, Any]]:
        results = []
        for h in hotspots[:top_n]:
            n = node_by_id.get(h["node_id"])
            if not n or n.type == NodeType.COMMENT:
                continue
            results.append({
                "id": n.id,
                "name": n.name,
                "type": n.type.value,
                "file": n.loc.file,
                "score": round(h["score"], 2),
                "caller_count": h["caller_count"],
                "churn_rate": h["churn_rate"],
                "degree": degree[n.id],
                "comment_count": len(comments_by_symbol.get(n.name, [])),
            })
        return results

    @app.get("/", response_class=HTMLResponse)
    def index_html() -> HTMLResponse:
        html_path = Path(__file__).parent / "static" / "index.html"
        return HTMLResponse(html_path.read_text())

    return app
