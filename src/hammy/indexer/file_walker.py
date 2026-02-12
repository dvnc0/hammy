"""Directory traversal that respects Hammy's ignore system.

Walks a project directory tree, yielding files that should be analyzed.
Skips ignored paths and files exceeding the configured size limit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from hammy.ignore import IgnoreManager

# Extension-to-language mapping
EXTENSION_MAP: dict[str, str] = {
    ".php": "php",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
}


@dataclass
class FileEntry:
    """A file discovered during directory walking."""

    path: Path
    language: str | None
    size_bytes: int


def detect_language(filepath: Path) -> str | None:
    """Detect the programming language from a file extension."""
    return EXTENSION_MAP.get(filepath.suffix.lower())


def walk_project(
    root: Path,
    ignore_manager: IgnoreManager,
    *,
    max_file_size_kb: int = 500,
    languages: list[str] | None = None,
) -> Iterator[FileEntry]:
    """Walk a project directory, yielding non-ignored files.

    Args:
        root: The project root directory to walk.
        ignore_manager: The IgnoreManager instance for filtering.
        max_file_size_kb: Skip files larger than this (in KB).
        languages: If provided, only yield files matching these languages.
                   If None, yield all non-ignored files.

    Yields:
        FileEntry for each file that passes all filters.
    """
    root = root.resolve()
    max_size_bytes = max_file_size_kb * 1024

    for dirpath, dirnames, filenames in os.walk(root):
        current_dir = Path(dirpath)

        # Prune ignored directories in-place so os.walk doesn't descend into them
        dirnames[:] = [
            d
            for d in dirnames
            if not ignore_manager.is_ignored(current_dir / d, is_dir=True)
        ]
        # Sort for deterministic order
        dirnames.sort()

        for filename in sorted(filenames):
            filepath = current_dir / filename

            if ignore_manager.is_ignored(filepath):
                continue

            try:
                size = filepath.stat().st_size
            except OSError:
                continue

            if size > max_size_bytes:
                continue

            language = detect_language(filepath)

            if languages is not None and language not in languages:
                continue

            yield FileEntry(path=filepath, language=language, size_bytes=size)
