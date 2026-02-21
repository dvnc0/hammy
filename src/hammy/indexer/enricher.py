"""LLM-powered symbol summarization for Hammy.

Reads source code for each indexed symbol and calls an LLM to generate
a concise one-sentence summary. Summaries are stored back on the node
and re-upserted to Qdrant so embeddings reflect the richer descriptions.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

from hammy.config import EnrichmentConfig
from hammy.schema.models import Node, NodeType


# Node types worth summarizing — endpoints are just URL strings,
# imports are just paths; neither benefit from LLM description.
_ENRICHABLE_TYPES = {NodeType.FUNCTION, NodeType.METHOD, NodeType.CLASS}

_SYSTEM_PROMPT = (
    "You are a code documentation assistant. "
    "Return ONLY a valid JSON array of strings with no other text, "
    "markdown, or code fences. Each string must be one sentence "
    "(max 20 words) describing what the corresponding code symbol does."
)

_USER_TEMPLATE = """\
For each symbol below write one sentence describing what it does.
Return a JSON array with exactly {n} strings, one per symbol.

{symbols}"""


def get_code_snippet(node: Node, project_root: Path, max_lines: int = 40) -> str:
    """Extract the source lines for a node from disk.

    Args:
        node: The symbol whose source to read.
        project_root: Absolute path to the project root.
        max_lines: Maximum lines to include before truncating.

    Returns:
        Source code string, or empty string if the file can't be read.
    """
    file_path = project_root / node.loc.file
    if not file_path.exists():
        return ""

    try:
        source = file_path.read_text(errors="replace")
    except OSError:
        return ""

    lines = source.splitlines()
    start = max(0, node.loc.lines[0] - 1)  # 1-indexed → 0-indexed
    end = min(len(lines), node.loc.lines[1])
    snippet = lines[start:end]

    if len(snippet) > max_lines:
        snippet = snippet[:max_lines] + [f"... ({len(lines[start:end]) - max_lines} more lines)"]

    return "\n".join(snippet)


def _build_prompt(items: list[tuple[Node, str]]) -> str:
    """Build the user prompt for a batch of (node, snippet) pairs."""
    parts = []
    for i, (node, snippet) in enumerate(items, 1):
        lang = node.language or "code"
        sig = node.name
        if node.meta.parameters:
            sig += f"({', '.join(node.meta.parameters)})"
        if node.meta.return_type:
            sig += f" -> {node.meta.return_type}"
        header = f"{i}. [{lang} {node.type.value}] {sig}"
        if snippet:
            parts.append(f"{header}\n```{lang}\n{snippet}\n```")
        else:
            parts.append(header)

    return _USER_TEMPLATE.format(n=len(items), symbols="\n\n".join(parts))


def _parse_summaries(text: str, expected: int) -> list[str | None]:
    """Extract the JSON array from LLM response, tolerating minor formatting noise."""
    # Strip markdown fences if present
    text = re.sub(r"```[a-z]*\n?", "", text).strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list) and len(parsed) == expected:
            return [str(s) if s else None for s in parsed]
    except json.JSONDecodeError:
        pass

    # Try to extract just the array portion
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                result = [str(s) if s else None for s in parsed]
                # Pad or trim to expected length
                result = result[:expected]
                result += [None] * (expected - len(result))
                return result
        except json.JSONDecodeError:
            pass

    return [None] * expected


def _summarize_batch_anthropic(
    items: list[tuple[Node, str]],
    model: str,
) -> list[str | None]:
    """Call the Anthropic API to summarize a batch of symbols."""
    try:
        import anthropic
    except ImportError:
        raise RuntimeError(
            "anthropic package not installed. Run: uv add anthropic"
        )

    client = anthropic.Anthropic()
    prompt = _build_prompt(items)

    message = client.messages.create(
        model=model,
        max_tokens=512,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    return _parse_summaries(message.content[0].text, len(items))


def enrich_nodes(
    nodes: list[Node],
    project_root: Path,
    config: EnrichmentConfig,
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[int, list[str]]:
    """Generate LLM summaries for enrichable nodes, updating them in place.

    Filters to FUNCTION/METHOD/CLASS nodes. Skips nodes that already have
    summaries when config.skip_if_summary is True. Calls the configured LLM
    in batches and writes the result back to node.summary.

    Args:
        nodes: All indexed nodes (modified in place).
        project_root: Absolute path to project root for reading source files.
        config: Enrichment configuration.
        progress_callback: Optional fn(completed, total) called after each batch.

    Returns:
        Tuple of (number enriched, list of error strings).
    """
    # Select candidates
    candidates = [
        n for n in nodes
        if n.type in _ENRICHABLE_TYPES
        and not (config.skip_if_summary and n.summary)
    ]

    if config.max_symbols > 0:
        candidates = candidates[: config.max_symbols]

    if not candidates:
        return 0, []

    # Build (node, snippet) pairs, skipping nodes with no readable code
    enrichable: list[tuple[Node, str]] = []
    for node in candidates:
        snippet = get_code_snippet(node, project_root)
        if snippet:
            enrichable.append((node, snippet))

    if not enrichable:
        return 0, []

    errors: list[str] = []
    enriched = 0
    total = len(enrichable)

    if config.provider != "anthropic":
        errors.append(f"Unsupported provider '{config.provider}'. Only 'anthropic' is supported.")
        return 0, errors

    for batch_start in range(0, total, config.batch_size):
        batch = enrichable[batch_start : batch_start + config.batch_size]
        try:
            summaries = _summarize_batch_anthropic(batch, config.model)
        except Exception as e:
            errors.append(f"Batch {batch_start // config.batch_size + 1}: {e}")
            if progress_callback:
                progress_callback(batch_start + len(batch), total)
            continue

        for (node, _), summary in zip(batch, summaries):
            if summary:
                node.summary = summary
                enriched += 1

        if progress_callback:
            progress_callback(batch_start + len(batch), total)

    return enriched, errors
