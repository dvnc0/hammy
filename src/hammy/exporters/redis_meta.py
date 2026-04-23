"""Redis meta enrichment client for MCP tools.

Fetches the `meta` field from Redis function payloads written by external tools,
and formats it for appending to MCP tool output as LLM context.
"""

from __future__ import annotations

import json
import logging

from hammy.config import RedisExportConfig

logger = logging.getLogger(__name__)

try:
    import redis as redis_lib
except ImportError:
    redis_lib = None  # type: ignore[assignment]


class RedisMetaClient:
    """Read-only Redis client for fetching external meta annotations on functions."""

    def __init__(self, config: RedisExportConfig) -> None:
        self._config = config
        self._client = None

    def connect(self) -> None:
        if redis_lib is None:
            raise RuntimeError("redis package not installed")
        self._client = redis_lib.Redis(
            host=self._config.host,
            port=self._config.port,
            db=self._config.db,
            password=self._config.password,
            decode_responses=True,
            socket_connect_timeout=2,
        )
        self._client.ping()

    def get_meta(self, node_id: str) -> list | None:
        """Return the meta list for a node, or None on miss/error."""
        if self._client is None:
            return None
        try:
            raw = self._client.get(f"{self._config.key_prefix}:func:{node_id}")
            if raw is None:
                return None
            payload = json.loads(raw)
            meta = payload.get("meta")
            if not meta:
                return None
            return meta
        except Exception:
            logger.debug("Redis meta fetch failed for node %s", node_id, exc_info=True)
            return None

    def format_meta(self, node_id: str) -> str:
        """Return a formatted meta line for tool output, or '' if none."""
        meta = self.get_meta(node_id)
        if not meta:
            return ""
        return f"\n  meta: {json.dumps(meta)}"

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
