"""Tests for RedisMetaClient."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from hammy.config import RedisExportConfig
from hammy.exporters.redis_meta import RedisMetaClient


def _config(**kwargs) -> RedisExportConfig:
    return RedisExportConfig(key_prefix="hammy", **kwargs)


def _client_with_mock(mock_redis_mod, data: dict | None = None) -> tuple[RedisMetaClient, MagicMock]:
    mock_r = MagicMock()
    mock_redis_mod.Redis.return_value = mock_r
    if data is not None:
        mock_r.get.return_value = json.dumps(data)
    else:
        mock_r.get.return_value = None
    client = RedisMetaClient(_config())
    client.connect()
    return client, mock_r


class TestRedisMetaClientConnect:

    @patch("hammy.exporters.redis_meta.redis_lib", None)
    def test_raises_when_redis_not_installed(self):
        client = RedisMetaClient(_config())
        with pytest.raises(RuntimeError, match="redis package not installed"):
            client.connect()

    @patch("hammy.exporters.redis_meta.redis_lib")
    def test_raises_on_connection_failure(self, mock_redis_mod):
        mock_redis_mod.Redis.return_value.ping.side_effect = ConnectionError("refused")
        client = RedisMetaClient(_config())
        with pytest.raises(ConnectionError):
            client.connect()

    @patch("hammy.exporters.redis_meta.redis_lib")
    def test_successful_connect(self, mock_redis_mod):
        client = RedisMetaClient(_config())
        client.connect()
        mock_redis_mod.Redis.return_value.ping.assert_called_once()


class TestGetMeta:

    @patch("hammy.exporters.redis_meta.redis_lib")
    def test_returns_none_when_not_connected(self, _mock):
        client = RedisMetaClient(_config())
        assert client.get_meta("some-id") is None

    @patch("hammy.exporters.redis_meta.redis_lib")
    def test_returns_none_on_key_miss(self, mock_redis_mod):
        client, _ = _client_with_mock(mock_redis_mod, data=None)
        assert client.get_meta("missing-id") is None

    @patch("hammy.exporters.redis_meta.redis_lib")
    def test_returns_none_on_empty_meta(self, mock_redis_mod):
        client, _ = _client_with_mock(mock_redis_mod, data={"name": "fn", "meta": []})
        assert client.get_meta("node-1") is None

    @patch("hammy.exporters.redis_meta.redis_lib")
    def test_returns_meta_list(self, mock_redis_mod):
        meta = [{"ticket": "PROJ-123"}, {"owner": "alice"}]
        client, _ = _client_with_mock(mock_redis_mod, data={"name": "fn", "meta": meta})
        assert client.get_meta("node-1") == meta

    @patch("hammy.exporters.redis_meta.redis_lib")
    def test_uses_correct_redis_key(self, mock_redis_mod):
        client, mock_r = _client_with_mock(mock_redis_mod, data={"meta": []})
        client.get_meta("abc123")
        mock_r.get.assert_called_once_with("hammy:func:abc123")

    @patch("hammy.exporters.redis_meta.redis_lib")
    def test_returns_none_on_redis_error(self, mock_redis_mod):
        mock_r = MagicMock()
        mock_r.get.side_effect = ConnectionError("lost")
        mock_redis_mod.Redis.return_value = mock_r
        client = RedisMetaClient(_config())
        client.connect()
        assert client.get_meta("node-1") is None

    @patch("hammy.exporters.redis_meta.redis_lib")
    def test_returns_none_on_bad_json(self, mock_redis_mod):
        mock_r = MagicMock()
        mock_r.get.return_value = "not-json{"
        mock_redis_mod.Redis.return_value = mock_r
        client = RedisMetaClient(_config())
        client.connect()
        assert client.get_meta("node-1") is None


class TestFormatMeta:

    @patch("hammy.exporters.redis_meta.redis_lib")
    def test_returns_empty_string_on_miss(self, mock_redis_mod):
        client, _ = _client_with_mock(mock_redis_mod, data=None)
        assert client.format_meta("missing") == ""

    @patch("hammy.exporters.redis_meta.redis_lib")
    def test_returns_empty_string_on_empty_meta(self, mock_redis_mod):
        client, _ = _client_with_mock(mock_redis_mod, data={"meta": []})
        assert client.format_meta("node-1") == ""

    @patch("hammy.exporters.redis_meta.redis_lib")
    def test_formats_meta_as_json_line(self, mock_redis_mod):
        meta = [{"ticket": "PROJ-42", "priority": "high"}]
        client, _ = _client_with_mock(mock_redis_mod, data={"meta": meta})
        result = client.format_meta("node-1")
        assert result.startswith("\n  meta: ")
        assert json.loads(result.split("meta: ", 1)[1]) == meta

    @patch("hammy.exporters.redis_meta.redis_lib")
    def test_format_meta_with_mixed_types(self, mock_redis_mod):
        meta = [{"key": "val"}, "plain string note", 42]
        client, _ = _client_with_mock(mock_redis_mod, data={"meta": meta})
        result = client.format_meta("node-1")
        assert "plain string note" in result
