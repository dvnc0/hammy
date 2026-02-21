"""Hybrid search: BM25 sparse + semantic dense via Reciprocal Rank Fusion.

BM25 runs in-memory on the indexed node list (fast for typical codebases).
When Qdrant is available, dense embeddings are fetched from Qdrant and
the two result lists are merged with RRF for a diverse, high-precision result.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from hammy.schema.models import Node

if TYPE_CHECKING:
    from hammy.tools.qdrant_tools import QdrantManager

# RRF constant — standard value, dampens rank differences
_RRF_K = 60


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-word chars, drop single-char tokens."""
    return [t for t in re.split(r"[^a-zA-Z0-9_]+", text.lower()) if len(t) > 1]


def _node_text(node: Node) -> str:
    """Build the full-text representation used for BM25 indexing."""
    parts = [node.type.value, node.name]
    if node.summary:
        parts.append(node.summary)
    if node.meta.parameters:
        parts.extend(node.meta.parameters)
    if node.meta.return_type:
        parts.append(node.meta.return_type)
    return " ".join(parts)


def _rrf(
    ranked_lists: list[list[tuple[str, dict[str, Any]]]],
    k: int = _RRF_K,
) -> list[tuple[float, dict[str, Any]]]:
    """Reciprocal Rank Fusion over multiple ranked result lists.

    Each list contains ``(id, payload)`` pairs in ranked order.
    Returns ``(rrf_score, payload)`` sorted descending by score.
    """
    scores: dict[str, float] = {}
    payloads: dict[str, dict[str, Any]] = {}

    for ranked in ranked_lists:
        for rank, (item_id, payload) in enumerate(ranked):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
            payloads[item_id] = payload

    return [
        (scores[item_id], payloads[item_id])
        for item_id in sorted(scores, key=lambda x: -scores[x])
    ]


def hybrid_search(
    query: str,
    nodes: list[Node],
    *,
    qdrant: QdrantManager | None = None,
    limit: int = 10,
    language: str | None = None,
    node_type: str | None = None,
) -> list[dict[str, Any]]:
    """Hybrid BM25 + semantic search with Reciprocal Rank Fusion.

    BM25 is always computed in-memory on ``nodes``. When ``qdrant`` is
    provided, a dense semantic search is also performed and the two result
    sets are merged via RRF. Without Qdrant only BM25 results are returned.

    Args:
        query: Natural language or keyword search query.
        nodes: The full indexed node list to BM25-search over.
        qdrant: Optional QdrantManager for semantic search.
        limit: Number of results to return.
        language: Optional language filter.
        node_type: Optional node type filter.

    Returns:
        List of result dicts with at minimum: node_id, type, name, file,
        lines, language, summary, score.
    """
    from rank_bm25 import BM25Plus

    fetch_k = limit * 4

    # Apply filters for BM25 candidates
    candidates = [
        n for n in nodes
        if (not language or n.language == language)
        and (not node_type or n.type.value == node_type)
    ]

    bm25_list: list[tuple[str, dict[str, Any]]] = []

    if candidates:
        texts = [_node_text(n) for n in candidates]
        tokenized = [_tokenize(t) for t in texts]
        # BM25Plus avoids zero/negative IDF on small corpora (BM25Okapi can
        # produce negative scores when df == N, e.g. single-document corpora)
        bm25 = BM25Plus(tokenized)
        scores = bm25.get_scores(_tokenize(query))

        # Rank by BM25 score, keep top fetch_k with score > 0
        ranked_indices = sorted(
            (i for i, s in enumerate(scores) if s > 0),
            key=lambda i: -scores[i],
        )[:fetch_k]

        for i in ranked_indices:
            n = candidates[i]
            bm25_list.append((
                n.id,
                {
                    "node_id": n.id,
                    "type": n.type.value,
                    "name": n.name,
                    "file": n.loc.file,
                    "lines": list(n.loc.lines),
                    "language": n.language,
                    "summary": n.summary,
                    "score": float(scores[i]),
                },
            ))

    if qdrant is None:
        return [payload for _, payload in bm25_list[:limit]]

    # Dense semantic search via Qdrant
    dense_results = qdrant.search_code(
        query, limit=fetch_k, language=language, node_type=node_type
    )
    dense_list: list[tuple[str, dict[str, Any]]] = [
        (r["node_id"], r) for r in dense_results
    ]

    if not bm25_list and not dense_list:
        return []

    # Merge with RRF
    fused = _rrf([bm25_list, dense_list])
    return [payload for _, payload in fused[:limit]]
