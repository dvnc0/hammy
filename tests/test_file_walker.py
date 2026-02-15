"""Tests for the file walker."""

from pathlib import Path

import pytest

from hammy.config import IgnoreConfig
from hammy.ignore import IgnoreManager
from hammy.indexer.file_walker import FileEntry, detect_language, walk_project


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    """Create a realistic project directory structure."""
    # Source files
    src = tmp_path / "src"
    src.mkdir()
    (src / "UserController.php").write_text("<?php class UserController {}")
    (src / "api.js").write_text("export function fetchUsers() {}")
    (src / "styles.css").write_text("body { color: red; }")
    (src / "README.md").write_text("# Source")

    # Nested directory
    models = src / "models"
    models.mkdir()
    (models / "User.php").write_text("<?php class User {}")

    # Vendor (should be ignored)
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    (vendor / "autoload.php").write_text("<?php // autoload")

    # Node modules (should be ignored)
    nm = tmp_path / "node_modules"
    nm.mkdir()
    (nm / "lodash.js").write_text("// lodash")

    # Large file
    (src / "huge.js").write_text("x" * (600 * 1024))  # 600KB

    return tmp_path


class TestDetectLanguage:
    def test_php(self):
        assert detect_language(Path("file.php")) == "php"

    def test_javascript(self):
        assert detect_language(Path("file.js")) == "javascript"

    def test_jsx(self):
        assert detect_language(Path("component.jsx")) == "javascript"

    def test_mjs(self):
        assert detect_language(Path("module.mjs")) == "javascript"

    def test_unknown(self):
        assert detect_language(Path("file.rb")) is None

    def test_css(self):
        assert detect_language(Path("styles.css")) is None


class TestWalkProject:
    def test_skips_vendor_and_node_modules(self, project_dir: Path):
        manager = IgnoreManager(project_dir)
        files = list(walk_project(project_dir, manager))
        paths = [str(f.path) for f in files]
        # Use path separator to avoid matching temp dir names containing these strings
        assert not any("/vendor/" in p for p in paths)
        assert not any("/node_modules/" in p for p in paths)

    def test_finds_source_files(self, project_dir: Path):
        manager = IgnoreManager(project_dir)
        files = list(walk_project(project_dir, manager))
        names = [f.path.name for f in files]
        assert "UserController.php" in names
        assert "api.js" in names
        assert "User.php" in names

    def test_skips_large_files(self, project_dir: Path):
        manager = IgnoreManager(project_dir)
        files = list(walk_project(project_dir, manager, max_file_size_kb=500))
        names = [f.path.name for f in files]
        assert "huge.js" not in names

    def test_language_filter(self, project_dir: Path):
        manager = IgnoreManager(project_dir)
        files = list(walk_project(project_dir, manager, languages=["php"]))
        assert all(f.language == "php" for f in files)
        names = [f.path.name for f in files]
        assert "UserController.php" in names
        assert "api.js" not in names

    def test_detects_languages(self, project_dir: Path):
        manager = IgnoreManager(project_dir)
        files = list(walk_project(project_dir, manager))
        php_files = [f for f in files if f.language == "php"]
        js_files = [f for f in files if f.language == "javascript"]
        assert len(php_files) >= 2  # UserController.php, User.php
        assert len(js_files) >= 1  # api.js (huge.js filtered by size)

    def test_deterministic_order(self, project_dir: Path):
        manager = IgnoreManager(project_dir)
        files1 = [f.path.name for f in walk_project(project_dir, manager)]
        files2 = [f.path.name for f in walk_project(project_dir, manager)]
        assert files1 == files2

    def test_file_entry_has_size(self, project_dir: Path):
        manager = IgnoreManager(project_dir)
        files = list(walk_project(project_dir, manager))
        for f in files:
            assert f.size_bytes > 0

    def test_yields_all_files_when_no_language_filter(self, project_dir: Path):
        manager = IgnoreManager(project_dir)
        files = list(walk_project(project_dir, manager))
        names = [f.path.name for f in files]
        # Should include non-code files like CSS and MD
        assert "styles.css" in names
        assert "README.md" in names
