"""Cross-language bridge tool — links frontend endpoints to backend routes.

Discovers connections between languages by matching:
- JS fetch/axios URLs to PHP Route attributes
- Symbol name similarity across languages
"""

from __future__ import annotations

from hammy.schema.models import Edge, EdgeMetadata, Node, NodeType, RelationType


def resolve_bridges(nodes: list[Node], edges: list[Edge]) -> list[Edge]:
    """Find cross-language connections between endpoint nodes.

    Matches JavaScript outbound API calls (fetch/axios) to PHP route definitions
    based on URL/path matching.

    Args:
        nodes: All extracted nodes from the codebase.
        edges: All extracted edges from the codebase.

    Returns:
        New bridge edges connecting matched endpoints.
    """
    bridge_edges: list[Edge] = []

    # Collect PHP endpoints (defined by Route attributes)
    php_endpoints: dict[str, Node] = {}
    for node in nodes:
        if node.type == NodeType.ENDPOINT and node.language == "php":
            php_endpoints[_normalize_path(node.name)] = node

    # Collect JS endpoints (from fetch/axios calls)
    js_endpoints: dict[str, Node] = {}
    for node in nodes:
        if node.type == NodeType.ENDPOINT and node.language == "javascript":
            js_endpoints[_normalize_path(node.name)] = node

    # Match JS endpoints to PHP endpoints
    for js_path, js_node in js_endpoints.items():
        for php_path, php_node in php_endpoints.items():
            confidence = _match_paths(js_path, php_path)
            if confidence > 0.0:
                bridge_edges.append(Edge(
                    source=js_node.id,
                    target=php_node.id,
                    relation=RelationType.NETWORKS_TO,
                    metadata=EdgeMetadata(
                        is_bridge=True,
                        confidence=confidence,
                        context=f"JS '{js_node.name}' -> PHP '{php_node.name}'",
                    ),
                ))

    return bridge_edges


def _normalize_path(path: str) -> str:
    """Normalize an API path for comparison.

    Strips leading/trailing slashes and replaces path parameters
    like {id} or :id with a wildcard placeholder.
    """
    import re
    path = path.strip("/")
    # Replace {param} style
    path = re.sub(r"\{[^}]+\}", "*", path)
    # Replace :param style
    path = re.sub(r":(\w+)", "*", path)
    # Replace template literal ${...} style
    path = re.sub(r"\$\{[^}]+\}", "*", path)
    return path.lower()


def _match_paths(path_a: str, path_b: str) -> float:
    """Compare two normalized paths and return a confidence score.

    Returns:
        1.0 for exact match
        0.8 for match with wildcard substitution
        0.0 for no match
    """
    if path_a == path_b:
        return 1.0

    # Split into segments and compare
    segments_a = path_a.split("/")
    segments_b = path_b.split("/")

    if len(segments_a) != len(segments_b):
        return 0.0

    mismatches = 0
    for seg_a, seg_b in zip(segments_a, segments_b):
        if seg_a == seg_b:
            continue
        if seg_a == "*" or seg_b == "*":
            mismatches += 1
        else:
            return 0.0

    if mismatches == 0:
        return 1.0
    return 0.8
