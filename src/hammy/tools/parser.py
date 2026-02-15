"""Tree-sitter parser factory for Hammy.

Creates and caches parsers for supported languages. Adding a new language
requires installing the grammar package and adding entries to LANGUAGE_REGISTRY.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import tree_sitter
import tree_sitter_go as ts_go
import tree_sitter_javascript as ts_js
import tree_sitter_php as ts_php
import tree_sitter_python as ts_py
import tree_sitter_typescript as ts_ts

# Registry: language name -> (grammar function, file extensions)
LANGUAGE_REGISTRY: dict[str, tuple[Callable[[], object], list[str]]] = {
    "php": (ts_php.language_php, [".php"]),
    "javascript": (ts_js.language, [".js", ".jsx", ".mjs"]),
    "python": (ts_py.language, [".py", ".pyi"]),
    "typescript": (ts_ts.language_typescript, [".ts", ".tsx"]),
    "go": (ts_go.language, [".go"]),
}

# Build reverse mapping: extension -> language name
EXTENSION_MAP: dict[str, str] = {}
for lang_name, (_, extensions) in LANGUAGE_REGISTRY.items():
    for ext in extensions:
        EXTENSION_MAP[ext] = lang_name


class ParserFactory:
    """Creates and caches tree-sitter parsers per language."""

    def __init__(self, enabled_languages: list[str] | None = None):
        if enabled_languages is None:
            enabled_languages = list(LANGUAGE_REGISTRY.keys())

        self._parsers: dict[str, tree_sitter.Parser] = {}
        self._languages: dict[str, tree_sitter.Language] = {}

        for lang in enabled_languages:
            if lang not in LANGUAGE_REGISTRY:
                raise ValueError(
                    f"Unsupported language: {lang}. "
                    f"Available: {list(LANGUAGE_REGISTRY.keys())}"
                )
            grammar_fn, _ = LANGUAGE_REGISTRY[lang]
            language = tree_sitter.Language(grammar_fn())
            self._languages[lang] = language
            self._parsers[lang] = tree_sitter.Parser(language)

    @property
    def enabled_languages(self) -> list[str]:
        return list(self._parsers.keys())

    def detect_language(self, filepath: Path) -> str | None:
        """Detect language from file extension, only if that language is enabled."""
        lang = EXTENSION_MAP.get(filepath.suffix.lower())
        if lang and lang in self._parsers:
            return lang
        return None

    def get_parser(self, language: str) -> tree_sitter.Parser:
        """Get the cached parser for a language."""
        if language not in self._parsers:
            raise ValueError(f"Language not enabled: {language}")
        return self._parsers[language]

    def parse_file(self, filepath: Path) -> tuple[tree_sitter.Tree, str] | None:
        """Parse a file, returning the tree and detected language.

        Returns None if the file's language is not supported or not enabled.
        """
        lang = self.detect_language(filepath)
        if lang is None:
            return None
        source = filepath.read_bytes()
        tree = self._parsers[lang].parse(source)
        return tree, lang

    def parse_bytes(self, source: bytes, language: str) -> tree_sitter.Tree:
        """Parse raw bytes with a specified language."""
        return self._parsers[language].parse(source)
