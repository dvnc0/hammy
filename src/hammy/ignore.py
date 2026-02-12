"""Centralized ignore-pattern manager for Hammy.

Single source of truth for which paths should be skipped across all subsystems.
Composes patterns from hardcoded defaults, .gitignore, .hgignore, .hammyignore,
and config-level extra patterns.
"""

from pathlib import Path

from pathspec import PathSpec

from hammy.config import IgnoreConfig

# Always ignored, regardless of config
DEFAULT_IGNORE_PATTERNS = [
    ".git/",
    ".hg/",
    "__pycache__/",
    "*.pyc",
    "node_modules/",
    "vendor/",
    ".vendor/",
    "bower_components/",
    "dist/",
    "build/",
    ".cache/",
    ".tox/",
    ".venv/",
    "venv/",
    ".env/",
    "*.min.js",
    "*.min.css",
    "*.map",
    "*.lock",
    "package-lock.json",
    "composer.lock",
    ".DS_Store",
    "Thumbs.db",
]


class IgnoreManager:
    """Manages file ignore patterns from multiple sources.

    Usage:
        manager = IgnoreManager(project_root, config.ignore)
        if manager.is_ignored(some_path):
            skip...
    """

    def __init__(self, project_root: Path, config: IgnoreConfig | None = None):
        if config is None:
            config = IgnoreConfig()

        self.project_root = project_root.resolve()
        patterns: list[str] = list(DEFAULT_IGNORE_PATTERNS)

        if config.use_gitignore:
            gitignore = self.project_root / ".gitignore"
            if gitignore.exists():
                patterns.extend(self._read_ignore_file(gitignore))

        if config.use_hgignore:
            hgignore = self.project_root / ".hgignore"
            if hgignore.exists():
                patterns.extend(self._parse_hgignore(hgignore))

        if config.use_hammyignore:
            hammyignore = self.project_root / ".hammyignore"
            if hammyignore.exists():
                patterns.extend(self._read_ignore_file(hammyignore))

        patterns.extend(config.extra_patterns)

        self._spec = PathSpec.from_lines("gitignore", patterns)

    def is_ignored(self, path: Path, *, is_dir: bool = False) -> bool:
        """Check if a path should be ignored.

        Args:
            path: Absolute or relative path to check.
            is_dir: If True, treat this path as a directory (appends / for matching).
                    If False and the path is absolute, will check the filesystem.
        """
        path = Path(path)
        if path.is_absolute():
            try:
                rel = path.relative_to(self.project_root)
            except ValueError:
                return False
            if not is_dir:
                is_dir = path.is_dir()
        else:
            rel = path

        rel_str = str(rel)

        # Check the path as-is
        if self._spec.match_file(rel_str):
            return True

        # For directories, also check with trailing slash (gitignore convention)
        if is_dir:
            return self._spec.match_file(rel_str + "/")

        return False

    def filter_paths(self, paths: list[Path]) -> list[Path]:
        """Return only non-ignored paths."""
        return [p for p in paths if not self.is_ignored(p)]

    @staticmethod
    def _read_ignore_file(path: Path) -> list[str]:
        """Read a gitignore-style file, skipping comments and blank lines."""
        lines = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                lines.append(line)
        return lines

    @staticmethod
    def _parse_hgignore(hgignore: Path) -> list[str]:
        """Parse .hgignore, extracting only glob-syntax lines.

        Mercurial's ignore file supports 'syntax: glob' and 'syntax: regexp'
        directives. We only use glob patterns since pathspec handles gitignore-style
        globs. Regex patterns are silently skipped.
        """
        lines = hgignore.read_text().splitlines()
        mode = "regexp"  # hgignore default
        result: list[str] = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("syntax:"):
                mode = line.split(":", 1)[1].strip()
                continue
            if mode == "glob":
                result.append(line)
        return result
