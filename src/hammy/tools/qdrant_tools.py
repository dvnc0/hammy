"""Qdrant vector database tools for Hammy.

Manages collections, upserts embeddings, and performs semantic search
for both code symbols and commit messages.
"""

from __future__ import annotations

from typing import Any

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
    BATCH_SIZE = 500

    def __init__(self, config: QdrantConfig | None = None):
        if config is None:
            config = QdrantConfig()

        self._client = QdrantClient(host=config.host, port=config.port)
        self._prefix = config.collection_prefix
        self._model = SentenceTransformer(config.embedding_model)
        self._embedding_dim = self._model.get_sentence_embedding_dimension()

    def _collection_name(self, base: str) -> str:
        return f"{self._prefix}_{base}"

    def ensure_collections(self) -> None:
        """Create collections if they don't exist."""
        for base in (self.CODES_COLLECTION, self.COMMITS_COLLECTION):
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
        for base in (self.CODES_COLLECTION, self.COMMITS_COLLECTION):
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
        for i, (node, embedding) in enumerate(zip(nodes, embeddings)):
            points.append(PointStruct(
                id=i,
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

    def get_stats(self) -> dict[str, int]:
        """Get collection statistics."""
        stats = {}
        for base in (self.CODES_COLLECTION, self.COMMITS_COLLECTION):
            name = self._collection_name(base)
            if self._client.collection_exists(name):
                info = self._client.get_collection(name)
                stats[base] = info.points_count or 0
            else:
                stats[base] = 0
        return stats
