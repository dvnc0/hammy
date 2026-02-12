"""Tests for the ignore system."""

from pathlib import Path

import pytest

from hammy.config import IgnoreConfig
from hammy.ignore import IgnoreManager


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    """Create a mock project directory with various files and ignore files."""
    # Create directories
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.php").write_text("<?php // app")
    (tmp_path / "src" / "utils.js").write_text("// utils")
    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor" / "lib.php").write_text("<?php // vendor lib")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "pkg.js").write_text("// pkg")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "mod.pyc").write_bytes(b"\x00")
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "bundle.min.js").write_text("// minified")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "output.js").write_text("// built")
    (tmp_path / "src" / "app.min.js").write_text("// minified")
    (tmp_path / "README.md").write_text("# Project")
    return tmp_path


class TestDefaultIgnores:
    def test_vendor_ignored(self, project_dir: Path):
        manager = IgnoreManager(project_dir)
        assert manager.is_ignored(project_dir / "vendor" / "lib.php")

    def test_node_modules_ignored(self, project_dir: Path):
        manager = IgnoreManager(project_dir)
        assert manager.is_ignored(project_dir / "node_modules" / "pkg.js")

    def test_pycache_ignored(self, project_dir: Path):
        manager = IgnoreManager(project_dir)
        assert manager.is_ignored(project_dir / "__pycache__" / "mod.pyc")

    def test_min_js_ignored(self, project_dir: Path):
        manager = IgnoreManager(project_dir)
        assert manager.is_ignored(project_dir / "src" / "app.min.js")

    def test_source_files_not_ignored(self, project_dir: Path):
        manager = IgnoreManager(project_dir)
        assert not manager.is_ignored(project_dir / "src" / "app.php")
        assert not manager.is_ignored(project_dir / "src" / "utils.js")

    def test_readme_not_ignored(self, project_dir: Path):
        manager = IgnoreManager(project_dir)
        assert not manager.is_ignored(project_dir / "README.md")


class TestGitignore:
    def test_reads_gitignore(self, project_dir: Path):
        (project_dir / ".gitignore").write_text("*.log\ntmp/\n")
        (project_dir / "debug.log").write_text("log data")

        manager = IgnoreManager(project_dir)
        assert manager.is_ignored(project_dir / "debug.log")

    def test_gitignore_disabled(self, project_dir: Path):
        (project_dir / ".gitignore").write_text("*.log\n")
        (project_dir / "debug.log").write_text("log data")

        config = IgnoreConfig(use_gitignore=False)
        manager = IgnoreManager(project_dir, config)
        assert not manager.is_ignored(project_dir / "debug.log")


class TestHgignore:
    def test_reads_glob_patterns(self, project_dir: Path):
        (project_dir / ".hgignore").write_text("syntax: glob\n*.tmp\nbackup/\n")
        manager = IgnoreManager(project_dir)
        assert manager.is_ignored(project_dir / "data.tmp")

    def test_skips_regexp_patterns(self, project_dir: Path):
        # Default hgignore mode is regexp; those patterns should be skipped
        (project_dir / ".hgignore").write_text("^temp/.*\\.bak$\n")
        manager = IgnoreManager(project_dir)
        # regexp pattern should NOT be applied (silently skipped)
        assert not manager.is_ignored(project_dir / "temp" / "file.bak")

    def test_mixed_syntax(self, project_dir: Path):
        content = "syntax: regexp\n^temp/\nsyntax: glob\n*.bak\n"
        (project_dir / ".hgignore").write_text(content)
        manager = IgnoreManager(project_dir)
        # Only the glob pattern (*.bak) should work
        assert manager.is_ignored(project_dir / "file.bak")

    def test_hgignore_disabled(self, project_dir: Path):
        (project_dir / ".hgignore").write_text("syntax: glob\n*.bak\n")
        config = IgnoreConfig(use_hgignore=False)
        manager = IgnoreManager(project_dir, config)
        assert not manager.is_ignored(project_dir / "file.bak")


class TestHammyignore:
    def test_reads_hammyignore(self, project_dir: Path):
        (project_dir / ".hammyignore").write_text("*.generated.php\nsecrets/\n")
        manager = IgnoreManager(project_dir)
        assert manager.is_ignored(project_dir / "models.generated.php")

    def test_hammyignore_disabled(self, project_dir: Path):
        (project_dir / ".hammyignore").write_text("*.generated.php\n")
        config = IgnoreConfig(use_hammyignore=False)
        manager = IgnoreManager(project_dir, config)
        assert not manager.is_ignored(project_dir / "models.generated.php")


class TestExtraPatterns:
    def test_extra_patterns_from_config(self, project_dir: Path):
        config = IgnoreConfig(extra_patterns=["*.csv", "data/"])
        manager = IgnoreManager(project_dir, config)
        assert manager.is_ignored(project_dir / "export.csv")


class TestFilterPaths:
    def test_filter_paths(self, project_dir: Path):
        manager = IgnoreManager(project_dir)
        paths = [
            project_dir / "src" / "app.php",
            project_dir / "vendor" / "lib.php",
            project_dir / "src" / "utils.js",
            project_dir / "node_modules" / "pkg.js",
        ]
        filtered = manager.filter_paths(paths)
        assert len(filtered) == 2
        assert all("vendor" not in str(p) and "node_modules" not in str(p) for p in filtered)

    def test_relative_paths(self, project_dir: Path):
        manager = IgnoreManager(project_dir)
        assert manager.is_ignored(Path("vendor/lib.php"))
        assert not manager.is_ignored(Path("src/app.php"))
