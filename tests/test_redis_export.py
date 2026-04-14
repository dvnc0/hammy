"""Tests for Redis export functionality."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from hammy.exporters.redis_export import (
    _collect_function_comments,
    build_function_payload,
    export_to_redis,
)
from hammy.schema.models import (
    HistoryEntry,
    Location,
    Node,
    NodeHistory,
    NodeMeta,
    NodeType,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_function(
    name: str = "process_payment",
    file: str = "src/Payments/PaymentService.cs",
    lines: tuple[int, int] = (10, 30),
    parent: str = "PaymentService",
    **kwargs,
) -> Node:
    return Node(
        id=Node.make_id(file, name),
        type=NodeType.METHOD,
        name=name,
        loc=Location(file=file, lines=lines),
        language="csharp",
        meta=NodeMeta(
            visibility="public",
            parameters=["Order order", "CancellationToken ct"],
            return_type="Task<PaymentResult>",
            is_async=True,
            parent_symbol=parent,
        ),
        **kwargs,
    )


def _make_comment(
    text: str,
    file: str = "src/Payments/PaymentService.cs",
    line: int = 15,
    parent: str = "process_payment",
) -> Node:
    return Node(
        id=Node.make_id(file, f"comment:{line}"),
        type=NodeType.COMMENT,
        name=text,
        loc=Location(file=file, lines=(line, line)),
        language="csharp",
        meta=NodeMeta(parent_symbol=parent),
    )


# ── Comment collection ───────────────────────────────────────────────


class TestCollectFunctionComments:

    def test_inline_comments_collected(self):
        func = _make_function(lines=(10, 30))
        comments = [
            _make_comment("validate input", line=12),
            _make_comment("process charge", line=20),
        ]
        result = _collect_function_comments(func, comments)
        assert result == ["validate input", "process charge"]

    def test_docblock_above_collected(self):
        func = _make_function(lines=(10, 30))
        comment = _make_comment("Process a payment transaction", line=8)
        result = _collect_function_comments(func, [comment])
        assert result == ["Process a payment transaction"]

    def test_docblock_at_boundary_collected(self):
        """Comment ending exactly 5 lines before function start should be included."""
        func = _make_function(lines=(10, 30))
        comment = _make_comment("summary", line=5)
        result = _collect_function_comments(func, [comment])
        assert result == ["summary"]

    def test_distant_comment_excluded(self):
        """Comment more than 5 lines above should be excluded."""
        func = _make_function(lines=(20, 40))
        comment = _make_comment("unrelated comment", line=5)
        result = _collect_function_comments(func, [comment])
        assert result == []

    def test_comment_after_function_excluded(self):
        func = _make_function(lines=(10, 30))
        comment = _make_comment("trailing remark", line=35)
        result = _collect_function_comments(func, [comment])
        assert result == []

    def test_different_file_excluded(self):
        func = _make_function(lines=(10, 30))
        comment = _make_comment("overlapping line", file="other.cs", line=15)
        result = _collect_function_comments(func, [comment])
        assert result == []

    def test_multiline_docblock(self):
        """Multi-line XML doc comment ending just above the function."""
        func = _make_function(lines=(10, 30))
        comment = Node(
            id=Node.make_id(func.loc.file, "comment:7"),
            type=NodeType.COMMENT,
            name="Handles payment processing with retry logic",
            loc=Location(file=func.loc.file, lines=(7, 9)),
            language="csharp",
            meta=NodeMeta(parent_symbol=""),
        )
        result = _collect_function_comments(func, [comment])
        assert result == ["Handles payment processing with retry logic"]


# ── Payload building ─────────────────────────────────────────────────


class TestBuildFunctionPayload:

    def test_basic_payload_shape(self):
        func = _make_function()
        payload = build_function_payload(func, [], commit_depth=10, indexed_at="2026-04-12T00:00:00Z")

        assert payload["id"] == func.id
        assert payload["name"] == "process_payment"
        assert payload["type"] == "method"
        assert payload["file"] == "src/Payments/PaymentService.cs"
        assert payload["start_line"] == 10
        assert payload["end_line"] == 30
        assert payload["language"] == "csharp"
        assert payload["parent"] == "PaymentService"
        assert payload["visibility"] == "public"
        assert payload["parameters"] == ["Order order", "CancellationToken ct"]
        assert payload["return_type"] == "Task<PaymentResult>"
        assert payload["is_async"] is True
        assert payload["summary"] is None
        assert payload["comments"] == []
        assert payload["commits"] == []
        assert payload["churn_rate"] == 0
        assert payload["blame_owners"] == []
        assert payload["indexed_at"] == "2026-04-12T00:00:00Z"

    def test_with_history(self):
        func = _make_function(
            history=NodeHistory(
                churn_rate=5,
                blame_owners=["alice", "bob"],
                intent_logs=[
                    HistoryEntry(revision="abc123", author="alice", date="2026-04-01", message="Fix retry"),
                    HistoryEntry(revision="def456", author="bob", date="2026-03-15", message="Add timeout"),
                ],
            ),
        )
        payload = build_function_payload(func, [], commit_depth=10, indexed_at="2026-04-12T00:00:00Z")

        assert payload["churn_rate"] == 5
        assert payload["blame_owners"] == ["alice", "bob"]
        assert len(payload["commits"]) == 2
        assert payload["commits"][0] == {
            "rev": "abc123", "author": "alice", "date": "2026-04-01", "message": "Fix retry",
        }

    def test_commit_depth_truncates(self):
        entries = [
            HistoryEntry(revision=f"rev{i}", author="dev", date="2026-01-01", message=f"commit {i}")
            for i in range(20)
        ]
        func = _make_function(history=NodeHistory(intent_logs=entries))
        payload = build_function_payload(func, [], commit_depth=5, indexed_at="2026-04-12T00:00:00Z")

        assert len(payload["commits"]) == 5
        assert payload["commits"][-1]["message"] == "commit 4"

    def test_with_summary_and_comments(self):
        func = _make_function(summary="Processes payment for an order")
        comments = [_make_comment("validate card", line=12)]
        payload = build_function_payload(func, comments, commit_depth=10, indexed_at="2026-04-12T00:00:00Z")

        assert payload["summary"] == "Processes payment for an order"
        assert payload["comments"] == ["validate card"]

    def test_no_parent_becomes_none(self):
        func = _make_function(parent="")
        payload = build_function_payload(func, [], commit_depth=10, indexed_at="2026-04-12T00:00:00Z")
        assert payload["parent"] is None


# ── Redis export integration ─────────────────────────────────────────


class TestExportToRedis:

    @patch("hammy.exporters.redis_export.redis_lib", MagicMock())
    def test_no_functions_returns_error(self):
        comment = _make_comment("just a comment")
        count, errors = export_to_redis([comment])
        assert count == 0
        assert "No functions or methods" in errors[0]

    @patch("hammy.exporters.redis_export.redis_lib", MagicMock())
    def test_empty_nodes_returns_error(self):
        count, errors = export_to_redis([])
        assert count == 0
        assert "No functions or methods" in errors[0]

    @patch("hammy.exporters.redis_export.redis_lib")
    def test_connection_failure(self, mock_redis_mod):
        mock_redis_mod.Redis.return_value.ping.side_effect = ConnectionError("refused")

        func = _make_function()
        count, errors = export_to_redis([func], host="badhost")
        assert count == 0
        assert any("Cannot connect" in e for e in errors)

    @patch("hammy.exporters.redis_export.redis_lib")
    def test_successful_export(self, mock_redis_mod):
        mock_client = MagicMock()
        mock_pipe = MagicMock()
        mock_client.pipeline.return_value = mock_pipe
        mock_redis_mod.Redis.return_value = mock_client

        func = _make_function()
        count, errors = export_to_redis([func], key_prefix="test")

        assert count == 1
        assert errors == []
        mock_pipe.set.assert_called_once()
        key_used = mock_pipe.set.call_args[0][0]
        assert key_used == f"test:func:{func.id}"
        mock_pipe.execute.assert_called_once()

    @patch("hammy.exporters.redis_export.redis_lib")
    def test_batching(self, mock_redis_mod):
        """Verify that functions are split into batches of the given size."""
        mock_client = MagicMock()
        mock_pipe = MagicMock()
        mock_client.pipeline.return_value = mock_pipe
        mock_redis_mod.Redis.return_value = mock_client

        funcs = [
            _make_function(name=f"func_{i}", lines=(i * 10, i * 10 + 5))
            for i in range(5)
        ]
        count, errors = export_to_redis(funcs, batch_size=2, key_prefix="t")

        assert count == 5
        assert errors == []
        # 5 funcs / batch_size 2 → 3 pipeline flushes (2 + 2 + 1)
        assert mock_pipe.execute.call_count == 3

    @patch("hammy.exporters.redis_export.redis_lib")
    def test_pipeline_execute_failure_returns_error(self, mock_redis_mod):
        """A Redis execute failure should append to errors, not raise."""
        mock_client = MagicMock()
        mock_pipe = MagicMock()
        mock_pipe.execute.side_effect = ConnectionError("connection lost")
        mock_client.pipeline.return_value = mock_pipe
        mock_redis_mod.Redis.return_value = mock_client

        func = _make_function()
        count, errors = export_to_redis([func])

        assert count == 0
        assert any("Pipeline execute failed" in e for e in errors)

    @patch("hammy.exporters.redis_export.redis_lib")
    def test_flush_before_export(self, mock_redis_mod):
        mock_client = MagicMock()
        mock_pipe = MagicMock()
        mock_client.pipeline.return_value = mock_pipe
        # scan returns no keys (nothing to flush)
        mock_client.scan.return_value = (0, [])
        mock_redis_mod.Redis.return_value = mock_client

        func = _make_function()
        count, errors = export_to_redis([func], flush=True, key_prefix="myapp")

        assert count == 1
        mock_client.scan.assert_called_once_with(0, match="myapp:func:*", count=500)

    @patch("hammy.exporters.redis_export.redis_lib")
    def test_progress_callback(self, mock_redis_mod):
        mock_client = MagicMock()
        mock_pipe = MagicMock()
        mock_client.pipeline.return_value = mock_pipe
        mock_redis_mod.Redis.return_value = mock_client

        funcs = [
            _make_function(name=f"func_{i}", lines=(i * 10, i * 10 + 5))
            for i in range(3)
        ]
        progress_calls: list[tuple[int, int]] = []

        def _track(completed: int, total: int) -> None:
            progress_calls.append((completed, total))

        export_to_redis(funcs, batch_size=2, progress_callback=_track)

        # 2 batches: first with 2, second with 1
        assert progress_calls == [(2, 3), (3, 3)]

    @patch("hammy.exporters.redis_export.redis_lib", None)
    def test_missing_redis_package(self):
        func = _make_function()
        count, errors = export_to_redis([func])
        assert count == 0
        assert "redis package not installed" in errors[0]

    @patch("hammy.exporters.redis_export.redis_lib")
    def test_filters_only_functions_and_methods(self, mock_redis_mod):
        """Classes, variables, etc. should be excluded from export."""
        mock_client = MagicMock()
        mock_pipe = MagicMock()
        mock_client.pipeline.return_value = mock_pipe
        mock_redis_mod.Redis.return_value = mock_client

        nodes = [
            _make_function(name="process"),
            Node(
                id=Node.make_id("file.cs", "PaymentService"),
                type=NodeType.CLASS,
                name="PaymentService",
                loc=Location(file="file.cs", lines=(1, 100)),
                language="csharp",
            ),
            _make_comment("a comment"),
        ]
        count, errors = export_to_redis(nodes)

        assert count == 1
        assert mock_pipe.set.call_count == 1
