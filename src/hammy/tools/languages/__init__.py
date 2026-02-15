"""Language extractor registry and shared protocol.

Each language has its own extraction module (e.g., php.py, javascript.py)
that registers an extract function. The registry is used by ast_tools.py
to dispatch extraction to the correct language module.
"""

from __future__ import annotations

from typing import Protocol

import tree_sitter

from hammy.schema.models import Edge, Node


class LanguageExtractor(Protocol):
    """Contract for a language extraction function.

    Each language module must expose a function matching this signature.
    """

    def __call__(
        self,
        tree: tree_sitter.Tree,
        file_path: str,
    ) -> tuple[list[Node], list[Edge]]: ...


# Registry populated by register_extractor() calls at module load time.
EXTRACTOR_REGISTRY: dict[str, LanguageExtractor] = {}


def register_extractor(language: str, extractor: LanguageExtractor) -> None:
    """Register a language extractor."""
    EXTRACTOR_REGISTRY[language] = extractor


def get_extractor(language: str) -> LanguageExtractor | None:
    """Look up the extractor for a language."""
    return EXTRACTOR_REGISTRY.get(language)
