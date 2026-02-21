"""Tests for the LLM enrichment pipeline.

All Anthropic API calls are mocked — no real calls made.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hammy.config import EnrichmentConfig
from hammy.indexer.enricher import (
    _ENRICHABLE_TYPES,
    _parse_summaries,
    enrich_nodes,
    get_code_snippet,
)
from hammy.schema.models import Location, Node, NodeMeta, NodeType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_node(
    name: str,
    ntype: NodeType = NodeType.FUNCTION,
    file: str = "src/example.py",
    lines: tuple[int, int] = (1, 5),
    summary: str = "",
    language: str = "python",
) -> Node:
    return Node(
        id=Node.make_id(file, name),
        type=ntype,
        name=name,
        loc=Location(file=file, lines=lines),
        language=language,
        summary=summary,
    )


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    """Project with a real source file for snippet extraction."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "example.py").write_text(
        "def process_payment(amount, currency):\n"
        "    \"\"\"Old docstring.\"\"\"\n"
        "    charge = amount * 1.02\n"
        "    return {'charged': charge, 'currency': currency}\n"
        "\n"
        "class UserService:\n"
        "    def get_user(self, user_id):\n"
        "        return db.query(User).filter_by(id=user_id).first()\n"
    )
    return tmp_path


# ---------------------------------------------------------------------------
# get_code_snippet
# ---------------------------------------------------------------------------

class TestGetCodeSnippet:
    def test_extracts_correct_lines(self, project_dir):
        node = _make_node("process_payment", lines=(1, 4), file="src/example.py")
        snippet = get_code_snippet(node, project_dir)
        assert "process_payment" in snippet
        assert "amount" in snippet

    def test_missing_file_returns_empty(self, tmp_path):
        node = _make_node("foo", file="nonexistent.py", lines=(1, 5))
        assert get_code_snippet(node, tmp_path) == ""

    def test_truncates_long_functions(self, tmp_path):
        src = tmp_path / "big.py"
        src.write_text("\n".join(f"    line_{i} = {i}" for i in range(200)))
        node = _make_node("big_func", file="big.py", lines=(1, 200))
        snippet = get_code_snippet(node, tmp_path, max_lines=10)
        assert "more lines" in snippet
        assert snippet.count("\n") < 15

    def test_1indexed_lines(self, project_dir):
        # Line 6 is "class UserService:"
        node = _make_node("UserService", ntype=NodeType.CLASS, file="src/example.py", lines=(6, 8))
        snippet = get_code_snippet(node, project_dir)
        assert "UserService" in snippet


# ---------------------------------------------------------------------------
# _parse_summaries
# ---------------------------------------------------------------------------

class TestParseSummaries:
    def test_clean_json_array(self):
        raw = '["Does X.", "Does Y.", "Does Z."]'
        result = _parse_summaries(raw, 3)
        assert result == ["Does X.", "Does Y.", "Does Z."]

    def test_strips_markdown_fences(self):
        raw = "```json\n[\"Does X.\"]\n```"
        result = _parse_summaries(raw, 1)
        assert result == ["Does X."]

    def test_extracts_array_from_noisy_response(self):
        raw = 'Here are the summaries: ["Handles auth.", "Processes data."] Hope that helps!'
        result = _parse_summaries(raw, 2)
        assert result == ["Handles auth.", "Processes data."]

    def test_wrong_count_pads_with_none(self):
        raw = '["Only one."]'
        result = _parse_summaries(raw, 3)
        assert result[0] == "Only one."
        assert result[1] is None
        assert result[2] is None

    def test_invalid_json_returns_nones(self):
        result = _parse_summaries("totally not json", 2)
        assert result == [None, None]


# ---------------------------------------------------------------------------
# enrich_nodes
# ---------------------------------------------------------------------------

def _mock_anthropic_response(summaries: list[str]) -> MagicMock:
    """Build a mock Anthropic message response."""
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock()]
    mock_msg.content[0].text = json.dumps(summaries)
    return mock_msg


class TestEnrichNodes:
    def test_updates_summary_in_place(self, project_dir):
        nodes = [_make_node("process_payment", file="src/example.py", lines=(1, 4))]
        config = EnrichmentConfig(batch_size=10, skip_if_summary=False)

        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_anthropic_response(
            ["Charges a payment with currency conversion."]
        )

        with patch("anthropic.Anthropic", return_value=mock_client):
            count, errors = enrich_nodes(nodes, project_dir, config)

        assert count == 1
        assert errors == []
        assert nodes[0].summary == "Charges a payment with currency conversion."

    def test_skips_existing_summary_when_flag_set(self, project_dir):
        nodes = [_make_node("process_payment", file="src/example.py", lines=(1, 4),
                            summary="Already has one.")]
        config = EnrichmentConfig(skip_if_summary=True)

        with patch("anthropic.Anthropic") as mock_cls:
            count, errors = enrich_nodes(nodes, project_dir, config)
            mock_cls.assert_not_called()

        assert count == 0
        assert nodes[0].summary == "Already has one."

    def test_does_not_skip_when_flag_false(self, project_dir):
        nodes = [_make_node("process_payment", file="src/example.py", lines=(1, 4),
                            summary="Old summary.")]
        config = EnrichmentConfig(skip_if_summary=False)

        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_anthropic_response(["New summary."])

        with patch("anthropic.Anthropic", return_value=mock_client):
            count, errors = enrich_nodes(nodes, project_dir, config)

        assert count == 1
        assert nodes[0].summary == "New summary."

    def test_skips_endpoint_nodes(self, project_dir):
        nodes = [
            _make_node("process_payment", file="src/example.py", lines=(1, 4)),
            _make_node("/api/pay", ntype=NodeType.ENDPOINT, file="src/example.py", lines=(1, 1)),
        ]
        config = EnrichmentConfig(skip_if_summary=False)

        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_anthropic_response(["Does payment."])

        with patch("anthropic.Anthropic", return_value=mock_client):
            count, errors = enrich_nodes(nodes, project_dir, config)

        # Only the FUNCTION node enriched, ENDPOINT skipped
        assert count == 1
        # Endpoint summary untouched
        assert nodes[1].summary == ""

    def test_batches_correctly(self, project_dir):
        # 3 nodes, batch_size=2 → 2 API calls
        nodes = [
            _make_node("fn_a", file="src/example.py", lines=(1, 2)),
            _make_node("fn_b", file="src/example.py", lines=(1, 2)),
            _make_node("fn_c", file="src/example.py", lines=(1, 2)),
        ]
        config = EnrichmentConfig(batch_size=2, skip_if_summary=False)

        call_count = 0
        def fake_create(**kwargs):
            nonlocal call_count
            n = len([m for m in kwargs["messages"][0]["content"].split("\n")
                     if m.strip().startswith(("1.", "2.", "3."))])
            call_count += 1
            # Return as many summaries as nodes in this batch
            msgs = kwargs["messages"]
            # Count numbered items in the prompt to determine batch size
            import re
            prompt = msgs[0]["content"]
            items = re.findall(r"^\d+\.", prompt, re.MULTILINE)
            return _mock_anthropic_response([f"Summary {i}." for i in range(len(items))])

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = fake_create

        with patch("anthropic.Anthropic", return_value=mock_client):
            count, errors = enrich_nodes(nodes, project_dir, config)

        assert mock_client.messages.create.call_count == 2

    def test_max_symbols_limits_candidates(self, project_dir):
        nodes = [_make_node(f"fn_{i}", file="src/example.py", lines=(1, 2))
                 for i in range(10)]
        config = EnrichmentConfig(max_symbols=3, skip_if_summary=False)

        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_anthropic_response(
            ["S1.", "S2.", "S3."]
        )

        with patch("anthropic.Anthropic", return_value=mock_client):
            count, errors = enrich_nodes(nodes, project_dir, config)

        assert count == 3

    def test_api_error_recorded_and_continues(self, project_dir):
        nodes = [
            _make_node("fn_a", file="src/example.py", lines=(1, 2)),
            _make_node("fn_b", file="src/example.py", lines=(1, 2)),
        ]
        config = EnrichmentConfig(batch_size=1, skip_if_summary=False)

        responses = [Exception("API timeout"), _mock_anthropic_response(["Works fine."])]
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = responses

        with patch("anthropic.Anthropic", return_value=mock_client):
            count, errors = enrich_nodes(nodes, project_dir, config)

        assert len(errors) == 1
        assert "API timeout" in errors[0]
        # Second batch still enriched
        assert count == 1

    def test_progress_callback_called(self, project_dir):
        nodes = [_make_node("fn_a", file="src/example.py", lines=(1, 2))]
        config = EnrichmentConfig(batch_size=10, skip_if_summary=False)

        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_anthropic_response(["Summary."])

        calls = []
        def on_progress(done, total):
            calls.append((done, total))

        with patch("anthropic.Anthropic", return_value=mock_client):
            enrich_nodes(nodes, project_dir, config, progress_callback=on_progress)

        assert len(calls) == 1
        assert calls[0] == (1, 1)

    def test_no_candidates_returns_zero(self, tmp_path):
        # All nodes are ENDPOINT type → no candidates
        nodes = [_make_node("/api/users", ntype=NodeType.ENDPOINT, file="x.php")]
        config = EnrichmentConfig()
        count, errors = enrich_nodes(nodes, tmp_path, config)
        assert count == 0
        assert errors == []

    def test_unsupported_provider_error(self, project_dir):
        nodes = [_make_node("fn_a", file="src/example.py", lines=(1, 2))]
        config = EnrichmentConfig(provider="openai", skip_if_summary=False)
        count, errors = enrich_nodes(nodes, project_dir, config)
        assert count == 0
        assert any("openai" in e.lower() or "anthropic" in e.lower() for e in errors)
