"""Qdrant vector database tools for Hammy.

Manages collections, upserts embeddings, and performs semantic search
for both code symbols and commit messages.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

from hammy.config import QdrantConfig
from hammy.schema.models import Node


class QdrantManager:
    """Manages Qdrant collections and operations for Hammy."""

    CODES_COLLECTION = "code_symbols"
    COMMITS_COLLECTION = "commits"
    BRAIN_COLLECTION = "brain"
    BATCH_SIZE = 500

    def __init__(self, config: QdrantConfig | None = None, project_name: str = ""):
        import re as _re
        if config is None:
            config = QdrantConfig()

        self._client = QdrantClient(host=config.host, port=config.port)

        # If no explicit prefix was set (still the default "hammy"), derive from project name
        # so multiple projects don't share the same Qdrant collections.
        if config.collection_prefix == "hammy" and project_name:
            slug = _re.sub(r"[^a-z0-9]+", "_", project_name.lower()).strip("_")
            self._prefix = f"hammy_{slug}"
        else:
            self._prefix = config.collection_prefix

        self._model = SentenceTransformer(config.embedding_model)
        self._embedding_dim = self._model.get_sentence_embedding_dimension()

    def _collection_name(self, base: str) -> str:
        return f"{self._prefix}_{base}"

    def ensure_collections(self) -> None:
        """Create collections if they don't exist."""
        for base in (self.CODES_COLLECTION, self.COMMITS_COLLECTION, self.BRAIN_COLLECTION):
            name = self._collection_name(base)
            if not self._client.collection_exists(name):
                self._client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(
                        size=self._embedding_dim,
                        distance=Distance.COSINE,
                    ),
                )

    def delete_collections(self) -> None:
        """Delete all Hammy collections."""
        for base in (self.CODES_COLLECTION, self.COMMITS_COLLECTION, self.BRAIN_COLLECTION):
            name = self._collection_name(base)
            if self._client.collection_exists(name):
                self._client.delete_collection(name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts, batched to avoid OOM."""
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i : i + self.BATCH_SIZE]
            embeddings = self._model.encode(batch)
            all_embeddings.extend(embeddings.tolist())
        return all_embeddings

    def upsert_nodes(self, nodes: list[Node]) -> int:
        """Upsert code symbol nodes into the code collection.

        Returns the number of points upserted.
        """
        if not nodes:
            return 0

        # Build text representations for embedding
        texts = []
        for node in nodes:
            text = f"{node.type.value} {node.name}"
            if node.summary:
                text += f" - {node.summary}"
            if node.meta.parameters:
                text += f" params: {', '.join(node.meta.parameters)}"
            if node.meta.return_type:
                text += f" returns: {node.meta.return_type}"
            texts.append(text)

        embeddings = self.embed(texts)

        points = []
        for node, embedding in zip(nodes, embeddings):
            # Stable ID derived from node.id so partial re-upserts (e.g. after
            # enrichment) overwrite the correct point rather than a positional one.
            point_id = int(hashlib.md5(node.id.encode()).hexdigest(), 16) % (2**53)
            points.append(PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "node_id": node.id,
                    "type": node.type.value,
                    "name": node.name,
                    "file": node.loc.file,
                    "lines": list(node.loc.lines),
                    "language": node.language,
                    "summary": node.summary,
                    "visibility": node.meta.visibility,
                    "is_async": node.meta.is_async,
                },
            ))

        collection = self._collection_name(self.CODES_COLLECTION)
        for i in range(0, len(points), self.BATCH_SIZE):
            self._client.upsert(collection_name=collection, points=points[i : i + self.BATCH_SIZE])
        return len(points)

    def delete_nodes_by_file(self, file_path: str) -> int:
        """Delete all code symbol nodes belonging to a specific file.

        Returns the number of points deleted (approximate — Qdrant count before delete).
        """
        collection = self._collection_name(self.CODES_COLLECTION)
        file_filter = Filter(
            must=[FieldCondition(key="file", match=MatchValue(value=file_path))]
        )
        # Count first so we can report how many were removed
        count_result = self._client.count(collection_name=collection, count_filter=file_filter)
        deleted = count_result.count
        if deleted > 0:
            self._client.delete(
                collection_name=collection,
                points_selector=file_filter,
            )
        return deleted

    def upsert_commits(
        self,
        commits: list[dict[str, Any]],
    ) -> int:
        """Upsert commit data into the commits collection.

        Each commit dict should have: revision, author, date, message, files_changed.
        Returns the number of points upserted.
        """
        if not commits:
            return 0

        texts = [c["message"] for c in commits]
        embeddings = self.embed(texts)

        points = []
        for i, (commit, embedding) in enumerate(zip(commits, embeddings)):
            points.append(PointStruct(
                id=i,
                vector=embedding,
                payload={
                    "revision": commit["revision"],
                    "author": commit["author"],
                    "date": commit["date"],
                    "message": commit["message"],
                    "files_changed": commit.get("files_changed", []),
                },
            ))

        collection = self._collection_name(self.COMMITS_COLLECTION)
        for i in range(0, len(points), self.BATCH_SIZE):
            self._client.upsert(collection_name=collection, points=points[i : i + self.BATCH_SIZE])
        return len(points)

    def search_code(
        self,
        query: str,
        *,
        limit: int = 10,
        language: str | None = None,
        node_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search over code symbols.

        Args:
            query: Natural language search query.
            limit: Max results to return.
            language: Optional language filter (e.g., "php", "javascript").
            node_type: Optional node type filter (e.g., "function", "class").

        Returns:
            List of result dicts with score and payload.
        """
        query_embedding = self.embed([query])[0]

        conditions = []
        if language:
            conditions.append(
                FieldCondition(key="language", match=MatchValue(value=language))
            )
        if node_type:
            conditions.append(
                FieldCondition(key="type", match=MatchValue(value=node_type))
            )

        search_filter = Filter(must=conditions) if conditions else None

        collection = self._collection_name(self.CODES_COLLECTION)
        results = self._client.query_points(
            collection_name=collection,
            query=query_embedding,
            query_filter=search_filter,
            limit=limit,
        )

        return [
            {"score": r.score, **r.payload}
            for r in results.points
        ]

    def search_code_mmr(
        self,
        query: str,
        *,
        limit: int = 10,
        fetch_k: int | None = None,
        lambda_: float = 0.6,
        language: str | None = None,
        node_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search with Maximal Marginal Relevance for diverse results.

        Fetches a larger candidate pool then iteratively selects items that
        balance relevance to the query against redundancy with already-selected
        results. Prevents returning many near-duplicate results from the same
        file or class.

        Args:
            query: Natural language search query.
            limit: Final number of results to return.
            fetch_k: Candidate pool size (defaults to limit * 4, min 40).
            lambda_: Trade-off between relevance (1.0) and diversity (0.0).
            language: Optional language filter.
            node_type: Optional node type filter.
        """
        if fetch_k is None:
            fetch_k = max(limit * 4, 40)

        query_embedding = self.embed([query])[0]
        query_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(query_vec)
        if q_norm > 0:
            query_vec = query_vec / q_norm

        conditions = []
        if language:
            conditions.append(
                FieldCondition(key="language", match=MatchValue(value=language))
            )
        if node_type:
            conditions.append(
                FieldCondition(key="type", match=MatchValue(value=node_type))
            )

        search_filter = Filter(must=conditions) if conditions else None
        collection = self._collection_name(self.CODES_COLLECTION)

        results = self._client.query_points(
            collection_name=collection,
            query=query_embedding,
            query_filter=search_filter,
            limit=fetch_k,
            with_vectors=True,
        )

        if not results.points:
            return []

        # Build normalized candidate vectors
        candidates: list[dict[str, Any]] = []
        for r in results.points:
            raw = r.vector
            if isinstance(raw, dict):
                raw = next(iter(raw.values()))
            vec = np.array(raw, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            rel = float(np.dot(query_vec, vec))
            candidates.append({"payload": r.payload, "score": r.score, "vec": vec, "rel": rel})

        # MMR selection loop
        selected: list[dict[str, Any]] = []
        selected_vecs: list[np.ndarray] = []

        while candidates and len(selected) < limit:
            if not selected_vecs:
                best_idx = max(range(len(candidates)), key=lambda i: candidates[i]["rel"])
            else:
                sel_matrix = np.stack(selected_vecs)  # (k, dim)
                best_score = float("-inf")
                best_idx = 0
                for i, c in enumerate(candidates):
                    redundancy = float(np.max(sel_matrix @ c["vec"]))
                    mmr = lambda_ * c["rel"] - (1 - lambda_) * redundancy
                    if mmr > best_score:
                        best_score = mmr
                        best_idx = i

            chosen = candidates.pop(best_idx)
            selected.append({"score": chosen["score"], **chosen["payload"]})
            selected_vecs.append(chosen["vec"])

        return selected

    def search_commits(
        self,
        query: str,
        *,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Semantic search over commit messages."""
        query_embedding = self.embed([query])[0]

        collection = self._collection_name(self.COMMITS_COLLECTION)
        results = self._client.query_points(
            collection_name=collection,
            query=query_embedding,
            limit=limit,
        )

        return [
            {"score": r.score, **r.payload}
            for r in results.points
        ]

    # --- Brain (working memory) ---

    def _brain_point_id(self, key: str) -> int:
        """Deterministic point ID from a key string."""
        return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**53)

    def upsert_brain_entry(
        self,
        key: str,
        content: str,
        tags: list[str] | None = None,
        source_files: list[str] | None = None,
    ) -> None:
        """Store or overwrite a brain entry.

        The key acts as a stable identifier — upserting with the same key
        replaces the previous entry. Content is embedded for semantic recall.

        Args:
            key: Unique identifier for this entry (e.g. 'payment-flow-research').
            content: The discovered information to store.
            tags: Optional labels for grouping/filtering (e.g. ['payment', 'sprint-42']).
            source_files: Optional file paths this entry relates to.
        """
        tags = tags or []
        source_files = source_files or []
        text_to_embed = f"{key}: {content}"
        embedding = self.embed([text_to_embed])[0]

        collection = self._collection_name(self.BRAIN_COLLECTION)
        self._client.upsert(
            collection_name=collection,
            points=[
                PointStruct(
                    id=self._brain_point_id(key),
                    vector=embedding,
                    payload={
                        "key": key,
                        "content": content,
                        "tags": tags,
                        "source_files": source_files,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
            ],
        )

    def search_brain(
        self,
        query: str = "",
        *,
        key: str = "",
        tag: str = "",
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve brain entries by exact key, tag filter, or semantic query.

        Args:
            query: Semantic search text (used when key is not provided).
            key: Exact key for direct lookup (takes priority over query).
            tag: Optional tag to restrict results.
            limit: Max results for semantic search.
        """
        collection = self._collection_name(self.BRAIN_COLLECTION)

        if key:
            # Exact key lookup via payload filter
            results, _ = self._client.scroll(
                collection_name=collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="key", match=MatchValue(value=key))]
                ),
                limit=1,
                with_payload=True,
            )
            return [r.payload for r in results]

        # Semantic search with optional tag filter
        if not query:
            return []

        conditions = []
        if tag:
            conditions.append(
                FieldCondition(key="tags", match=MatchValue(value=tag))
            )

        query_embedding = self.embed([query])[0]
        results = self._client.query_points(
            collection_name=collection,
            query=query_embedding,
            query_filter=Filter(must=conditions) if conditions else None,
            limit=limit,
        )
        return [{"score": r.score, **r.payload} for r in results.points]

    def list_brain_entries(self, tag: str = "") -> list[dict[str, Any]]:
        """List all brain entries, optionally filtered by tag.

        Args:
            tag: Optional tag to restrict results.
        """
        collection = self._collection_name(self.BRAIN_COLLECTION)

        conditions = []
        if tag:
            conditions.append(
                FieldCondition(key="tags", match=MatchValue(value=tag))
            )

        results, _ = self._client.scroll(
            collection_name=collection,
            scroll_filter=Filter(must=conditions) if conditions else None,
            limit=200,
            with_payload=True,
        )
        # Sort newest first
        entries = [r.payload for r in results]
        entries.sort(key=lambda e: e.get("created_at", ""), reverse=True)
        return entries

    def delete_brain_entry(self, key: str) -> None:
        """Delete a brain entry by key.

        Args:
            key: The exact key to delete.
        """
        from qdrant_client.models import PointIdsList

        collection = self._collection_name(self.BRAIN_COLLECTION)
        self._client.delete(
            collection_name=collection,
            points_selector=PointIdsList(points=[self._brain_point_id(key)]),
        )

    def get_stats(self) -> dict[str, int]:
        """Get collection statistics."""
        stats = {}
        for base in (self.CODES_COLLECTION, self.COMMITS_COLLECTION, self.BRAIN_COLLECTION):
            name = self._collection_name(base)
            if self._client.collection_exists(name):
                info = self._client.get_collection(name)
                stats[base] = info.points_count or 0
            else:
                stats[base] = 0
        return stats
