"""Hotspot scoring — composite risk metric for code symbols.

A hotspot is a symbol that is both heavily depended upon (many callers) and
frequently changed (high churn). These are the highest-risk locations to
modify: they have the widest blast radius and the most turbulent history.

Score formula:  log2(1 + caller_count) × log2(1 + churn_rate)

Using log-scale dampens outliers so a function with 100 callers and 10 churn
doesn't completely dominate the list over one with 20 callers and 8 churn.
When churn data is unavailable, the score degrades to log2(1 + caller_count)
alone, which still surfaces highly-depended-upon symbols.
"""

from __future__ import annotations

import math
from typing import Any

from hammy.schema.models import Edge, Node, RelationType


def _caller_counts(nodes: list[Node], edges: list[Edge]) -> dict[str, int]:
    """Return a dict of node_id -> unique caller count from CALLS edges.

    Uses word-boundary name matching (same as find_usages) to attribute each
    call edge to the matching node(s).
    """
    import re

    call_edges = [e for e in edges if e.relation == RelationType.CALLS]

    # Map lowercase name -> list of node_ids (may have duplicates across files)
    name_to_ids: dict[str, list[str]] = {}
    for node in nodes:
        name_to_ids.setdefault(node.name.lower(), []).append(node.id)

    # callee_name_lower -> set of caller_ids
    callee_callers: dict[str, set[str]] = {}

    for edge in call_edges:
        ctx = edge.metadata.context or ""
        # Extract the bare callee name from the context (last segment after . or ::)
        bare = re.split(r"[:\.\s]", ctx)[-1].strip().lower()
        if not bare:
            continue
        callee_callers.setdefault(bare, set()).add(edge.source)

    # Map node_id -> caller count
    result: dict[str, int] = {n.id: 0 for n in nodes}
    for name_lower, node_ids in name_to_ids.items():
        callers = len(callee_callers.get(name_lower, set()))
        for nid in node_ids:
            result[nid] = callers

    return result


def compute_hotspots(
    nodes: list[Node],
    edges: list[Edge],
    *,
    file_churn: dict[str, int] | None = None,
    node_type: str = "",
    language: str = "",
    file_filter: str = "",
    top_n: int = 20,
) -> list[dict[str, Any]]:
    """Compute hotspot scores for all symbols and return top_n results.

    Args:
        nodes: All indexed nodes.
        edges: All indexed edges.
        file_churn: Optional mapping of relative file path -> churn count (commits
            touching the file). When provided this is used as the churn component.
            Falls back to node.history.churn_rate, then 0 if neither is available.
        node_type: Optional filter by node type value ('function', 'method', etc.).
        language: Optional filter by language.
        file_filter: Optional path substring filter.
        top_n: Maximum results to return.

    Returns:
        List of dicts sorted by score descending, each containing:
        node_id, name, type, file, lines, language, caller_count, churn_rate, score.
    """
    # Apply filters
    candidates = [
        n for n in nodes
        if (not node_type or n.type.value == node_type)
        and (not language or n.language == language)
        and (not file_filter or file_filter.lower() in n.loc.file.lower())
    ]

    if not candidates:
        return []

    counts = _caller_counts(candidates, edges)

    results: list[dict[str, Any]] = []
    for node in candidates:
        caller_count = counts.get(node.id, 0)

        # Churn: prefer file_churn dict, then node.history, then 0
        if file_churn is not None:
            churn_rate = file_churn.get(node.loc.file, 0)
        elif node.history is not None:
            churn_rate = node.history.churn_rate
        else:
            churn_rate = 0

        # Log-scale composite score
        score = math.log2(1 + caller_count) * math.log2(1 + max(churn_rate, 1))

        results.append({
            "node_id": node.id,
            "name": node.name,
            "type": node.type.value,
            "file": node.loc.file,
            "lines": list(node.loc.lines),
            "language": node.language,
            "caller_count": caller_count,
            "churn_rate": churn_rate,
            "score": score,
            "visibility": node.meta.visibility,
            "is_async": node.meta.is_async,
            "summary": node.summary,
        })

    results.sort(key=lambda r: (-r["score"], -r["caller_count"]))
    return results[:top_n]
