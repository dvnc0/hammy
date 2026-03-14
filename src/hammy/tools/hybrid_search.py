"""Hybrid search: BM25 sparse + semantic dense via Reciprocal Rank Fusion.

BM25 runs in-memory on the indexed node list. When Qdrant is available,
dense embeddings are fetched and merged with BM25 via RRF.

For large codebases, build a BM25Index once at startup with build_bm25_index()
and pass it to hybrid_search() to avoid re-tokenizing on every query.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from hammy.schema.models import Node, NodeType

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


@dataclass
class BM25Index:
    """Pre-built BM25 index for fast repeated queries on large codebases.

    Build once at startup with build_bm25_index() and pass to hybrid_search().
    Invalidate by calling build_bm25_index() again after reindex.
    """

    node_ids: list[str] = field(default_factory=list)
    payloads: list[dict[str, Any]] = field(default_factory=list)
    tokenized: list[list[str]] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    node_types: list[str] = field(default_factory=list)


def build_bm25_index(nodes: list[Node]) -> BM25Index:
    """Build a BM25Index from the current node list.

    Tokenizes every node's text representation once and stores the result.
    Subsequent queries skip tokenization entirely, only constructing BM25Plus
    from the (already-tokenized) filtered subset — fast even on 50k+ symbols.
    """
    idx = BM25Index()
    for n in nodes:
        if n.type == NodeType.COMMENT:
            continue
        idx.node_ids.append(n.id)
        idx.payloads.append({
            "node_id": n.id,
            "type": n.type.value,
            "name": n.name,
            "file": n.loc.file,
            "lines": list(n.loc.lines),
            "language": n.language,
            "summary": n.summary,
        })
        idx.tokenized.append(_tokenize(_node_text(n)))
        idx.languages.append(n.language)
        idx.node_types.append(n.type.value)
    return idx


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
    bm25_index: BM25Index | None = None,
    qdrant: QdrantManager | None = None,
    limit: int = 10,
    language: str | None = None,
    node_type: str | None = None,
) -> list[dict[str, Any]]:
    """Hybrid BM25 + semantic search with Reciprocal Rank Fusion.

    BM25 is always computed on ``nodes``. When ``bm25_index`` is provided,
    pre-tokenized data is used so only BM25Plus construction runs per query
    (tokenization is skipped). When Qdrant is provided, dense semantic results
    are merged via RRF.

    Args:
        query: Natural language or keyword search query.
        nodes: The full indexed node list (used when bm25_index is None).
        bm25_index: Optional pre-built index from build_bm25_index().
                    Pass this on large codebases to avoid per-query tokenization.
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
    bm25_list: list[tuple[str, dict[str, Any]]] = []

    if bm25_index is not None:
        # Fast path: use pre-tokenized index, just filter and construct BM25Plus
        indices = [
            i for i, (lang, ntype) in enumerate(zip(bm25_index.languages, bm25_index.node_types))
            if (not language or lang == language)
            and (not node_type or ntype == node_type)
        ]

        if indices:
            filtered_tokenized = [bm25_index.tokenized[i] for i in indices]
            bm25 = BM25Plus(filtered_tokenized)
            scores = bm25.get_scores(_tokenize(query))

            ranked = sorted(
                (i for i, s in enumerate(scores) if s > 0),
                key=lambda i: -scores[i],
            )[:fetch_k]

            for rank_i in ranked:
                orig_i = indices[rank_i]
                payload = dict(bm25_index.payloads[orig_i])
                payload["score"] = float(scores[rank_i])
                bm25_list.append((bm25_index.node_ids[orig_i], payload))

    else:
        # Slow path: build from scratch (no pre-built index)
        candidates = [
            n for n in nodes
            if n.type != NodeType.COMMENT
            and (not language or n.language == language)
            and (not node_type or n.type.value == node_type)
        ]

        if candidates:
            texts = [_node_text(n) for n in candidates]
            tokenized = [_tokenize(t) for t in texts]
            # BM25Plus avoids zero/negative IDF on small corpora
            bm25 = BM25Plus(tokenized)
            scores = bm25.get_scores(_tokenize(query))

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
