"""Cross-language bridge tool — links frontend endpoints to backend routes.

Discovers connections between languages by matching endpoint nodes:
- Providers: endpoints defined via route attributes/decorators (DEFINES edge)
- Consumers: endpoints from API calls like fetch/axios (NETWORKS_TO edge)

Language-agnostic — any language that emits ENDPOINT nodes participates.
"""

from __future__ import annotations

import re

from hammy.schema.models import Edge, EdgeMetadata, Node, NodeType, RelationType


def resolve_bridges(nodes: list[Node], edges: list[Edge]) -> list[Edge]:
    """Find cross-language connections between endpoint nodes.

    Matches consumer endpoints (API calls) to provider endpoints (route
    definitions) across different languages based on URL/path matching.

    Args:
        nodes: All extracted nodes from the codebase.
        edges: All extracted edges from the codebase.

    Returns:
        New bridge edges connecting matched endpoints.
    """
    bridge_edges: list[Edge] = []

    # Identify provider and consumer endpoint IDs from edge relationships
    provider_ids: set[str] = set()
    consumer_source_map: dict[str, str] = {}  # endpoint_id -> source_id

    for edge in edges:
        if edge.relation == RelationType.DEFINES:
            provider_ids.add(edge.target)
        elif edge.relation == RelationType.NETWORKS_TO and edge.metadata.is_bridge:
            consumer_source_map[edge.target] = edge.source

    # Build lookup dicts for endpoint nodes
    providers: dict[str, Node] = {}
    consumers: dict[str, Node] = {}

    for node in nodes:
        if node.type != NodeType.ENDPOINT:
            continue
        if node.id in provider_ids:
            providers[_normalize_path(node.name)] = node
        elif node.id in consumer_source_map:
            consumers[_normalize_path(node.name)] = node

    # Match consumers to providers across languages
    for consumer_path, consumer_node in consumers.items():
        for provider_path, provider_node in providers.items():
            if consumer_node.language == provider_node.language:
                continue  # Only bridge across different languages
            confidence = _match_paths(consumer_path, provider_path)
            if confidence > 0.0:
                bridge_edges.append(Edge(
                    source=consumer_node.id,
                    target=provider_node.id,
                    relation=RelationType.NETWORKS_TO,
                    metadata=EdgeMetadata(
                        is_bridge=True,
                        confidence=confidence,
                        context=(
                            f"{consumer_node.language} '{consumer_node.name}' -> "
                            f"{provider_node.language} '{provider_node.name}'"
                        ),
                    ),
                ))

    return bridge_edges


def _normalize_path(path: str) -> str:
    """Normalize an API path for comparison.

    Strips leading/trailing slashes and replaces path parameters
    like {id}, :id, or ${...} with a wildcard placeholder.
    """
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
