"""Tests for the Hammy CLI."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from hammy.cli import app

runner = CliRunner()


class TestInit:
    def test_init_creates_config(self, tmp_path: Path):
        result = runner.invoke(app, ["init", str(tmp_path)])
        assert result.exit_code == 0
        assert (tmp_path / "config" / "hammy.yaml").exists()
        assert (tmp_path / ".hammyignore").exists()

    def test_init_doesnt_overwrite(self, tmp_path: Path):
        # First init
        runner.invoke(app, ["init", str(tmp_path)])
        # Write custom content
        (tmp_path / "config" / "hammy.yaml").write_text("custom: true")
        # Second init should not overwrite
        result = runner.invoke(app, ["init", str(tmp_path)])
        assert result.exit_code == 0
        assert "custom: true" in (tmp_path / "config" / "hammy.yaml").read_text()


class TestIndex:
    def test_index_no_qdrant(self, tmp_path: Path):
        # Create some source files
        (tmp_path / "app.php").write_text("<?php class App {}")
        (tmp_path / "main.js").write_text("function main() {}")

        # Create minimal config
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "hammy.yaml").write_text(
            "project:\n  root: .\nparsing:\n  languages:\n    - php\n    - javascript\n"
        )

        result = runner.invoke(app, ["index", str(tmp_path), "--no-qdrant", "--no-commits"])
        assert result.exit_code == 0
        assert "Files processed" in result.output
        assert "2" in result.output  # 2 files


class TestStatus:
    def test_status_basic(self, tmp_path: Path):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "hammy.yaml").write_text(
            "project:\n  root: .\nparsing:\n  languages:\n    - php\n"
        )

        result = runner.invoke(app, ["status", str(tmp_path)])
        assert result.exit_code == 0
        assert "Project root" in result.output
