"""AST extraction tools — dispatches to per-language extraction modules.

Each language has its own module in hammy.tools.languages/ that registers
an extractor function. This module provides the single public entry point.
"""

from __future__ import annotations

import tree_sitter

from hammy.schema.models import Edge, Node
from hammy.tools.languages import get_extractor

# Load built-in language extractors so they register themselves.
import hammy.tools.languages.php  # noqa: F401
import hammy.tools.languages.javascript  # noqa: F401
import hammy.tools.languages.python  # noqa: F401
import hammy.tools.languages.typescript  # noqa: F401
import hammy.tools.languages.go  # noqa: F401


def extract_symbols(
    tree: tree_sitter.Tree,
    language: str,
    file_path: str,
) -> tuple[list[Node], list[Edge]]:
    """Extract all symbols from a parsed tree.

    Args:
        tree: The parsed tree-sitter tree.
        language: The language name (e.g., "php", "javascript").
        file_path: The file path for location metadata.

    Returns:
        Tuple of (nodes, edges) extracted from the tree.
    """
    extractor = get_extractor(language)
    if extractor is None:
        return [], []
    return extractor(tree, file_path)
