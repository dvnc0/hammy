"""Context Pack generator — formats Hammy's analysis output for LLM consumption.

Produces structured Markdown documents containing dependency maps,
historical context, and warnings.
"""

from __future__ import annotations

from hammy.schema.models import ContextPack, Edge, Node, NodeType


def generate_context_pack_markdown(pack: ContextPack) -> str:
    """Convert a ContextPack into a Markdown document optimized for LLM consumption."""
    sections = []

    # Header
    sections.append(f"# Hammy Context Pack\n\n**Query:** {pack.query}\n")

    # Summary
    if pack.summary:
        sections.append(f"## Summary\n\n{pack.summary}\n")

    # Warnings
    if pack.warnings:
        sections.append("## Warnings\n")
        for warning in pack.warnings:
            sections.append(f"- {warning}")
        sections.append("")

    # Nodes by type
    if pack.nodes:
        sections.append("## Code Entities\n")

        # Group by type
        by_type: dict[NodeType, list[Node]] = {}
        for node in pack.nodes:
            by_type.setdefault(node.type, []).append(node)

        for node_type in NodeType:
            type_nodes = by_type.get(node_type, [])
            if not type_nodes:
                continue

            sections.append(f"### {node_type.value.title()}s\n")
            for node in type_nodes:
                loc = f"{node.loc.file}:{node.loc.lines[0]}-{node.loc.lines[1]}"
                sections.append(f"- **{node.name}** (`{loc}`, {node.language})")

                details = []
                if node.meta.visibility:
                    details.append(f"visibility: {node.meta.visibility}")
                if node.meta.is_async:
                    details.append("async")
                if node.meta.return_type:
                    details.append(f"returns: {node.meta.return_type}")
                if node.meta.parameters:
                    details.append(f"params: {', '.join(node.meta.parameters)}")
                if details:
                    sections.append(f"  - {', '.join(details)}")

                if node.summary:
                    sections.append(f"  - {node.summary}")

                if node.history:
                    h = node.history
                    if h.churn_rate > 0:
                        sections.append(f"  - Churn: {h.churn_rate} changes")
                    if h.blame_owners:
                        sections.append(f"  - Owners: {', '.join(h.blame_owners)}")

            sections.append("")

    # Edges / Relationships
    if pack.edges:
        sections.append("## Relationships\n")

        # Separate bridges from regular edges
        bridges = [e for e in pack.edges if e.metadata.is_bridge]
        regular = [e for e in pack.edges if not e.metadata.is_bridge]

        if bridges:
            sections.append("### Cross-Language Bridges\n")
            for edge in bridges:
                conf = f"{edge.metadata.confidence:.0%}"
                sections.append(
                    f"- {edge.metadata.context} "
                    f"(confidence: {conf})"
                )
            sections.append("")

        if regular:
            sections.append("### Dependencies\n")
            for edge in regular:
                ctx = f" — {edge.metadata.context}" if edge.metadata.context else ""
                sections.append(f"- `{edge.source}` —[{edge.relation.value}]→ `{edge.target}`{ctx}")
            sections.append("")

    return "\n".join(sections)
