"""PR / diff analysis — extract changed symbols and compute their blast radius.

Given a unified diff (from `git diff` or pasted directly), this module:
1. Parses which files changed and which symbol definitions appear in the hunks
2. Matches those symbol names against the indexed node list
3. Runs N-hop caller traversal to show the blast radius of each change

Diff input can be:
- Raw unified diff text (paste from GitHub PR, `git diff`, etc.)
- A base git ref — the caller computes the diff and passes it here
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from hammy.schema.models import Edge, Node, RelationType

# Regex patterns that match function/method/class definition lines in diffs.
# Each pattern captures the symbol name in group 1.
_DEF_PATTERNS = [
    # Python: def foo / async def foo
    re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w+)"),
    # PHP: public/private/protected function foo
    re.compile(r"^\s*(?:(?:public|private|protected|static|abstract|final)\s+)*function\s+([A-Za-z_]\w+)"),
    # JS/TS: function foo / export function foo / export async function foo
    re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_]\w+)"),
    # JS/TS: const foo = (async) (function|() =>
    re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_]\w+)\s*=\s*(?:async\s+)?(?:function|\()"),
    # Go: func FooName / func (r *Recv) FooName
    re.compile(r"^\s*func\s+(?:\([^)]+\)\s+)?([A-Za-z_]\w+)"),
    # Class definitions: class Foo / abstract class Foo
    re.compile(r"^\s*(?:abstract\s+)?class\s+([A-Za-z_]\w+)"),
    # TypeScript/Java-style methods: public async fooBar(
    re.compile(r"^\s*(?:(?:public|private|protected|override|static|abstract|async|readonly)\s+)+([A-Za-z_]\w+)\s*[\(<]"),
]

# Extract context hint after the last @@ on a hunk header
_HUNK_CONTEXT = re.compile(r"^@@[^@]+@@\s*(.+)")

# Match file path lines in unified diff
_FILE_PATH = re.compile(r"^\+\+\+ b/(.+)")
_FILE_PATH_OLD = re.compile(r"^--- a/(.+)")
_NEW_FILE = re.compile(r"^\+\+\+ /dev/null")
_DEL_FILE = re.compile(r"^--- /dev/null")


@dataclass
class ChangedFile:
    """A file touched by the diff."""

    path: str
    change_type: str  # "modified", "added", "deleted"
    changed_symbols: list[str] = field(default_factory=list)


@dataclass
class DiffReport:
    """Parsed result from a unified diff."""

    changed_files: list[ChangedFile]
    all_changed_symbols: list[str]  # Deduplicated symbol names from all files
    impact: list[dict[str, Any]]  # impact_analysis results per symbol


def _extract_symbols_from_diff(diff_text: str) -> list[ChangedFile]:
    """Parse unified diff and extract changed files + symbol names from hunks."""
    files: list[ChangedFile] = []
    current_file: ChangedFile | None = None
    is_new_file = False
    is_del_file = False
    seen_symbols: set[str] = set()

    for line in diff_text.splitlines():
        # New file header
        if line.startswith("--- "):
            is_del_file = bool(_DEL_FILE.match(line))
            is_new_file = False

        elif line.startswith("+++ "):
            if _NEW_FILE.match(line):
                is_new_file = True
                if current_file:
                    files.append(current_file)
                # Will be set when we see the --- line was /dev/null
            else:
                m = _FILE_PATH.match(line)
                if m:
                    path = m.group(1)
                    change_type = "added" if is_del_file else "modified"
                    if current_file:
                        files.append(current_file)
                    current_file = ChangedFile(path=path, change_type=change_type)
                    seen_symbols = set()

        elif line.startswith("diff --git"):
            # Reset for new file section
            is_new_file = False
            is_del_file = False

        elif line.startswith("deleted file"):
            if current_file:
                current_file.change_type = "deleted"

        elif line.startswith("new file"):
            if current_file:
                current_file.change_type = "added"

        elif line.startswith("@@"):
            # Extract context hint (e.g., function name after the last @@)
            m = _HUNK_CONTEXT.match(line)
            if m and current_file is not None:
                ctx = m.group(1).strip()
                # The context often contains "class Foo::method" or just "methodName("
                for part in re.split(r"[:\s]", ctx):
                    part = part.strip().rstrip("(")
                    if part and re.match(r"^[A-Za-z_]\w+$", part) and part not in seen_symbols:
                        seen_symbols.add(part)
                        current_file.changed_symbols.append(part)

        elif (line.startswith("+") or line.startswith("-")) and not line.startswith("+++") and not line.startswith("---"):
            # Changed line — scan for symbol definitions
            content = line[1:]
            if current_file is not None:
                for pattern in _DEF_PATTERNS:
                    m = pattern.match(content)
                    if m:
                        name = m.group(1)
                        if name not in seen_symbols:
                            seen_symbols.add(name)
                            current_file.changed_symbols.append(name)
                        break

    if current_file:
        files.append(current_file)

    return files


def _compute_impact_for_symbols(
    symbol_names: list[str],
    nodes: list[Node],
    edges: list[Edge],
    depth: int = 2,
) -> list[dict[str, Any]]:
    """For each symbol name, find matching nodes and compute caller counts."""
    call_edges = [e for e in edges if e.relation == RelationType.CALLS]
    node_index = {n.id: n for n in nodes}
    name_index: dict[str, list[Node]] = {}
    for n in nodes:
        name_index.setdefault(n.name.lower(), []).append(n)

    results: list[dict[str, Any]] = []
    seen_node_ids: set[str] = set()

    for sym_name in symbol_names:
        matching_nodes = name_index.get(sym_name.lower(), [])
        if not matching_nodes:
            # Symbol not in index — may be new/deleted
            results.append({
                "symbol": sym_name,
                "indexed": False,
                "caller_count": 0,
                "callers": [],
                "type": "unknown",
                "file": "?",
            })
            continue

        for node in matching_nodes:
            if node.id in seen_node_ids:
                continue
            seen_node_ids.add(node.id)

            # BFS to find callers up to `depth` hops
            pattern = re.compile(r"\b" + re.escape(node.name) + r"\b", re.IGNORECASE)
            direct_callers: list[dict[str, Any]] = []
            visited: set[str] = {node.id}
            current_names = {node.name}

            for _hop in range(1, depth + 1):
                hop_pats = {
                    n: re.compile(r"\b" + re.escape(n) + r"\b", re.IGNORECASE)
                    for n in current_names
                }
                next_names: set[str] = set()
                for edge in call_edges:
                    ctx = edge.metadata.context or ""
                    for name, p in hop_pats.items():
                        if p.search(ctx):
                            caller = node_index.get(edge.source)
                            if caller and caller.id not in visited:
                                visited.add(caller.id)
                                direct_callers.append({
                                    "name": caller.name,
                                    "type": caller.type.value,
                                    "file": caller.loc.file,
                                    "line": caller.loc.lines[0],
                                })
                                next_names.add(caller.name)
                            break
                current_names = next_names
                if not current_names:
                    break

            results.append({
                "symbol": node.name,
                "indexed": True,
                "type": node.type.value,
                "file": node.loc.file,
                "line": node.loc.lines[0],
                "caller_count": len(direct_callers),
                "callers": direct_callers,
                "summary": node.summary,
                "visibility": node.meta.visibility,
            })

    # Sort by caller count descending
    results.sort(key=lambda r: -r["caller_count"])
    return results


def analyze_diff(
    diff_text: str,
    nodes: list[Node],
    edges: list[Edge],
    *,
    depth: int = 2,
) -> DiffReport:
    """Analyse a unified diff and return a structured impact report.

    Args:
        diff_text: Raw unified diff string.
        nodes: Indexed nodes to look up symbols against.
        edges: Indexed edges for call graph traversal.
        depth: Caller traversal depth (hops).

    Returns:
        DiffReport with changed files, symbols, and blast radius.
    """
    changed_files = _extract_symbols_from_diff(diff_text)

    # Deduplicate all changed symbol names across files
    seen: set[str] = set()
    all_symbols: list[str] = []
    for cf in changed_files:
        for sym in cf.changed_symbols:
            if sym not in seen:
                seen.add(sym)
                all_symbols.append(sym)

    impact = _compute_impact_for_symbols(all_symbols, nodes, edges, depth=depth)

    return DiffReport(
        changed_files=changed_files,
        all_changed_symbols=all_symbols,
        impact=impact,
    )
