"""Pydantic models implementing Hammy's Unified JSON Schema (UJS).

These models represent the property graph of a codebase: Nodes (code entities)
and Edges (relationships between them).
"""

from __future__ import annotations

import hashlib
from enum import Enum

from pydantic import BaseModel, Field


class NodeType(str, Enum):
    """Types of code entities."""

    FILE = "file"
    CLASS = "class"
    METHOD = "method"
    FUNCTION = "function"
    VARIABLE = "variable"
    ENDPOINT = "endpoint"
    TABLE = "table"
    INTERFACE = "interface"


class RelationType(str, Enum):
    """Types of relationships between nodes."""

    CALLS = "calls"
    IMPORTS = "imports"
    IMPLEMENTS = "implements"
    DEFINES = "defines"
    NETWORKS_TO = "networks_to"
    EXTENDS = "extends"


class Location(BaseModel):
    """Source location of a code entity."""

    file: str
    lines: tuple[int, int]  # (start_line, end_line)


class NodeMeta(BaseModel):
    """Language-specific metadata for a node."""

    visibility: str | None = None
    is_async: bool = False
    complexity_score: int | None = None
    parameters: list[str] = Field(default_factory=list)
    return_type: str | None = None


class HistoryEntry(BaseModel):
    """A single historical event for a node."""

    revision: str
    author: str
    date: str
    message: str


class NodeHistory(BaseModel):
    """Temporal metadata for a node, provided by the Historian."""

    churn_rate: int = 0  # Changes in the configured window
    blame_owners: list[str] = Field(default_factory=list)
    intent_logs: list[HistoryEntry] = Field(default_factory=list)


class Node(BaseModel):
    """A code entity in the property graph."""

    id: str
    type: NodeType
    name: str
    loc: Location
    language: str
    meta: NodeMeta = Field(default_factory=NodeMeta)
    summary: str = ""
    history: NodeHistory | None = None

    @staticmethod
    def make_id(file_path: str, symbol_name: str) -> str:
        """Generate a deterministic ID from file path and symbol name."""
        raw = f"{file_path}::{symbol_name}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


class EdgeMetadata(BaseModel):
    """Metadata for a relationship edge."""

    is_bridge: bool = False
    confidence: float = 1.0
    context: str = ""


class Edge(BaseModel):
    """A relationship between two nodes in the property graph."""

    source: str  # Node ID
    target: str  # Node ID
    relation: RelationType
    metadata: EdgeMetadata = Field(default_factory=EdgeMetadata)


class ContextPack(BaseModel):
    """The output format for a Hammy query — a structured context document."""

    query: str
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    summary: str = ""
