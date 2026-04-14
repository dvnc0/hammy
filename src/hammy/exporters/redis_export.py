"""Export indexed function/method data to Redis as per-function JSON blobs.

Designed for point-in-time export from the .hammy/index.json cache so that
external knowledge-base tools can query structured code intelligence data
from Redis without needing Qdrant or the MCP server.

Usage:
    hammy export redis --host redis.internal --db 2 --prefix myapp
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import datetime, timezone

from hammy.schema.models import Node, NodeType

logger = logging.getLogger(__name__)

try:
    import redis as redis_lib
except ImportError:
    redis_lib = None  # type: ignore[assignment]

_EXPORTABLE_TYPES = frozenset({NodeType.FUNCTION, NodeType.METHOD})

# Max lines above a function to look for docblock/XML-doc comments
_DOCBLOCK_PROXIMITY = 5


def _collect_function_comments(
    node: Node,
    comments: list[Node],
) -> list[str]:
    """Gather comments within a function body or docblocks just above it.

    Matches by file path and line proximity rather than parent_symbol to
    correctly attribute XML doc comments (/// <summary>) that sit above
    the function declaration.
    """
    result: list[str] = []
    for c in comments:
        if c.loc.file != node.loc.file:
            continue
        cstart = c.loc.lines[0]
        cend = c.loc.lines[1]
        # Inline: comment starts within the function body
        if node.loc.lines[0] <= cstart <= node.loc.lines[1]:
            result.append(c.name)
        # Docblock: comment ends within _DOCBLOCK_PROXIMITY lines before function start
        elif cend < node.loc.lines[0] and node.loc.lines[0] - cend <= _DOCBLOCK_PROXIMITY:
            result.append(c.name)
    return result


def build_function_payload(
    node: Node,
    comments: list[Node],
    commit_depth: int,
    indexed_at: str,
) -> dict:
    """Build the JSON-serializable payload for a single function/method."""
    func_comments = _collect_function_comments(node, comments)

    payload: dict = {
        "id": node.id,
        "name": node.name,
        "type": node.type.value,
        "file": node.loc.file,
        "start_line": node.loc.lines[0],
        "end_line": node.loc.lines[1],
        "language": node.language,
        "parent": node.meta.parent_symbol or None,
        "visibility": node.meta.visibility,
        "parameters": node.meta.parameters,
        "return_type": node.meta.return_type,
        "is_async": node.meta.is_async,
        "summary": node.summary or None,
        "comments": func_comments,
        "commits": [],
        "churn_rate": 0,
        "blame_owners": [],
        "indexed_at": indexed_at,
    }

    if node.history:
        payload["churn_rate"] = node.history.churn_rate
        payload["blame_owners"] = node.history.blame_owners
        payload["commits"] = [
            {
                "rev": entry.revision,
                "author": entry.author,
                "date": entry.date,
                "message": entry.message,
            }
            for entry in node.history.intent_logs[:commit_depth]
        ]

    return payload


def export_to_redis(
    nodes: list[Node],
    *,
    host: str = "localhost",
    port: int = 6379,
    db: int = 0,
    password: str | None = None,
    key_prefix: str = "hammy",
    batch_size: int = 200,
    commit_depth: int = 10,
    flush: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[int, list[str]]:
    """Export function/method nodes to Redis as JSON key-value pairs.

    Each function is stored as ``{key_prefix}:func:{node_id}`` → JSON blob.
    Uses Redis pipelines for efficient bulk writes.

    Args:
        nodes: All indexed nodes (functions, methods, comments, etc.).
        host: Redis server hostname.
        port: Redis server port.
        db: Redis database number.
        password: Redis authentication password.
        key_prefix: Prefix for all Redis keys.
        batch_size: Number of SET commands per pipeline flush.
        commit_depth: Maximum number of commits to include per function.
        flush: If True, delete existing keys matching the prefix before export.
        progress_callback: Called with (exported_so_far, total) after each batch.

    Returns:
        Tuple of (exported_count, errors).
    """
    if redis_lib is None:
        return 0, [
            "redis package not installed. Install with: pip install 'hammy[redis]'"
        ]

    functions = [n for n in nodes if n.type in _EXPORTABLE_TYPES]
    comments = [n for n in nodes if n.type == NodeType.COMMENT]

    if not functions:
        return 0, ["No functions or methods found in index."]

    errors: list[str] = []
    exported = 0
    indexed_at = datetime.now(timezone.utc).isoformat()

    client = redis_lib.Redis(
        host=host,
        port=port,
        db=db,
        password=password,
        decode_responses=True,
    )

    try:
        client.ping()
    except Exception as e:
        return 0, [f"Cannot connect to Redis at {host}:{port} — {e}"]

    try:
        if flush:
            _flush_keys(client, key_prefix)

        for i in range(0, len(functions), batch_size):
            batch = functions[i : i + batch_size]
            pipe = client.pipeline(transaction=False)
            batch_count = 0

            for node in batch:
                try:
                    payload = build_function_payload(
                        node, comments, commit_depth, indexed_at,
                    )
                    key = f"{key_prefix}:func:{node.id}"
                    pipe.set(key, json.dumps(payload, separators=(",", ":")))
                    batch_count += 1
                except Exception as e:
                    errors.append(f"{node.name}: {e}")

            try:
                pipe.execute()
            except Exception as e:
                errors.append(f"Pipeline execute failed (batch {i // batch_size + 1}): {e}")
                continue
            exported += batch_count

            if progress_callback:
                progress_callback(exported, len(functions))
    finally:
        client.close()

    return exported, errors


def _flush_keys(client: object, key_prefix: str) -> int:
    """Delete all existing keys matching ``{key_prefix}:func:*``."""
    pattern = f"{key_prefix}:func:*"
    deleted = 0
    cursor = 0
    while True:
        cursor, keys = client.scan(cursor, match=pattern, count=500)  # type: ignore[union-attr]
        if keys:
            client.delete(*keys)  # type: ignore[union-attr]
            deleted += len(keys)
        if cursor == 0:
            break
    logger.info("Flushed %d existing keys matching %s", deleted, pattern)
    return deleted
