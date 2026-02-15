"""PHP AST extraction — classes, methods, functions, routes, imports."""

from __future__ import annotations

import tree_sitter

from hammy.schema.models import (
    Edge,
    EdgeMetadata,
    Location,
    Node,
    NodeMeta,
    NodeType,
    RelationType,
)
from hammy.tools.languages.helpers import (
    extract_parameters,
    find_child,
    get_child_text,
    node_lines,
    node_text,
    resolve_callee_name,
)


def extract(tree: tree_sitter.Tree, file_path: str) -> tuple[list[Node], list[Edge]]:
    """Extract all symbols from a PHP file."""
    nodes: list[Node] = []
    edges: list[Edge] = []
    namespace = ""

    for child in tree.root_node.children:
        if child.type == "namespace_definition":
            namespace = get_child_text(child, "namespace_name")

        elif child.type == "namespace_use_declaration":
            _extract_use(child, file_path, edges)

        elif child.type == "class_declaration":
            _extract_class(child, file_path, namespace, nodes, edges)

        elif child.type == "function_definition":
            _extract_function(child, file_path, namespace, nodes, edges)

    return nodes, edges


def _extract_class(
    node: tree_sitter.Node,
    file_path: str,
    namespace: str,
    nodes: list[Node],
    edges: list[Edge],
) -> None:
    class_name = get_child_text(node, "name")
    if not class_name:
        return

    full_name = f"{namespace}\\{class_name}" if namespace else class_name

    route = _extract_route_attribute(node)

    class_node = Node(
        id=Node.make_id(file_path, full_name),
        type=NodeType.CLASS,
        name=full_name,
        loc=Location(file=file_path, lines=node_lines(node)),
        language="php",
        meta=NodeMeta(),
        summary=f"Route: {route}" if route else "",
    )
    nodes.append(class_node)

    if route:
        endpoint_node = Node(
            id=Node.make_id(file_path, f"endpoint:{route}"),
            type=NodeType.ENDPOINT,
            name=route,
            loc=Location(file=file_path, lines=node_lines(node)),
            language="php",
        )
        nodes.append(endpoint_node)
        edges.append(Edge(
            source=class_node.id,
            target=endpoint_node.id,
            relation=RelationType.DEFINES,
        ))

    decl_list = find_child(node, "declaration_list")
    if decl_list:
        for member in decl_list.children:
            if member.type == "method_declaration":
                _extract_method(member, file_path, full_name, class_node.id, nodes, edges)


def _extract_method(
    node: tree_sitter.Node,
    file_path: str,
    class_name: str,
    class_id: str,
    nodes: list[Node],
    edges: list[Edge],
) -> None:
    method_name = get_child_text(node, "name")
    if not method_name:
        return

    full_name = f"{class_name}::{method_name}"
    visibility = _get_visibility(node)
    params = extract_parameters(node)
    return_type = _get_return_type(node)
    route = _extract_route_attribute(node)

    method_node = Node(
        id=Node.make_id(file_path, full_name),
        type=NodeType.METHOD,
        name=full_name,
        loc=Location(file=file_path, lines=node_lines(node)),
        language="php",
        meta=NodeMeta(
            visibility=visibility,
            parameters=params,
            return_type=return_type,
        ),
        summary=f"Route: {route}" if route else "",
    )
    nodes.append(method_node)

    edges.append(Edge(
        source=class_id,
        target=method_node.id,
        relation=RelationType.DEFINES,
    ))

    if route:
        endpoint_node = Node(
            id=Node.make_id(file_path, f"endpoint:{route}"),
            type=NodeType.ENDPOINT,
            name=route,
            loc=Location(file=file_path, lines=node_lines(node)),
            language="php",
        )
        nodes.append(endpoint_node)
        edges.append(Edge(
            source=method_node.id,
            target=endpoint_node.id,
            relation=RelationType.DEFINES,
        ))

    # Walk method body for function calls
    body = find_child(node, "compound_statement")
    if body:
        _extract_calls(body, file_path, method_node.id, edges)


def _extract_function(
    node: tree_sitter.Node,
    file_path: str,
    namespace: str,
    nodes: list[Node],
    edges: list[Edge],
) -> None:
    func_name = get_child_text(node, "name")
    if not func_name:
        return

    full_name = f"{namespace}\\{func_name}" if namespace else func_name
    params = extract_parameters(node)
    return_type = _get_return_type(node)

    func_node = Node(
        id=Node.make_id(file_path, full_name),
        type=NodeType.FUNCTION,
        name=full_name,
        loc=Location(file=file_path, lines=node_lines(node)),
        language="php",
        meta=NodeMeta(parameters=params, return_type=return_type),
    )
    nodes.append(func_node)

    body = find_child(node, "compound_statement")
    if body:
        _extract_calls(body, file_path, func_node.id, edges)


def _extract_use(
    node: tree_sitter.Node,
    file_path: str,
    edges: list[Edge],
) -> None:
    clause = find_child(node, "namespace_use_clause")
    if clause:
        imported = clause.text.decode("utf-8") if clause.text else ""
        if imported:
            edges.append(Edge(
                source=Node.make_id(file_path, "__file__"),
                target=Node.make_id("", imported),
                relation=RelationType.IMPORTS,
                metadata=EdgeMetadata(context=f"use {imported}"),
            ))


def _extract_calls(
    node: tree_sitter.Node,
    file_path: str,
    source_id: str,
    edges: list[Edge],
) -> None:
    """Recursively find function/method calls in PHP."""
    if node.type in ("function_call_expression", "member_call_expression", "scoped_call_expression"):
        callee_text = ""
        if node.type == "function_call_expression":
            callee = node.children[0] if node.children else None
            callee_text = node_text(callee) if callee else ""
        elif node.type == "member_call_expression":
            name_node = find_child(node, "name")
            callee_text = node_text(name_node) if name_node else ""
        elif node.type == "scoped_call_expression":
            # Class::method() — get both parts
            parts = [node_text(c) for c in node.children if c.type in ("name", "qualified_name")]
            callee_text = "::".join(parts) if parts else ""

        callee_name = resolve_callee_name(callee_text)
        if callee_name:
            edges.append(Edge(
                source=source_id,
                target=Node.make_id("", callee_name),
                relation=RelationType.CALLS,
                metadata=EdgeMetadata(confidence=0.8, context=callee_name),
            ))

    for child in node.children:
        _extract_calls(child, file_path, source_id, edges)


def _extract_route_attribute(node: tree_sitter.Node) -> str | None:
    """Extract route path from PHP 8 attributes like #[Route('/api/users')]."""
    attr_list = find_child(node, "attribute_list")
    if not attr_list:
        return None
    for group in attr_list.children:
        if group.type == "attribute_group":
            for attr in group.children:
                if attr.type == "attribute":
                    attr_text = attr.text.decode("utf-8") if attr.text else ""
                    if "Route" in attr_text:
                        args = find_child(attr, "arguments")
                        if args:
                            for arg in args.children:
                                if arg.type == "argument":
                                    val = find_child(arg, "string")
                                    if val and val.text:
                                        return val.text.decode("utf-8").strip("'\"")
                                elif arg.type == "string":
                                    if arg.text:
                                        return arg.text.decode("utf-8").strip("'\"")
    return None


def _get_visibility(node: tree_sitter.Node) -> str:
    for child in node.children:
        if child.type == "visibility_modifier":
            return child.text.decode("utf-8") if child.text else "public"
    return "public"


def _get_return_type(node: tree_sitter.Node) -> str | None:
    found_colon = False
    for child in node.children:
        if child.type == ":":
            found_colon = True
        elif found_colon and child.type in ("named_type", "primitive_type", "optional_type"):
            return child.text.decode("utf-8") if child.text else None
    return None


# Register this extractor
from hammy.tools.languages import register_extractor  # noqa: E402

register_extractor("php", extract)
