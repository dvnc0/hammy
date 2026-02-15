"""Go AST extraction — packages, functions, methods, interfaces, structs, imports."""

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
    """Extract all symbols from a Go file."""
    nodes: list[Node] = []
    edges: list[Edge] = []

    for child in tree.root_node.children:
        if child.type == "import_declaration":
            _extract_imports(child, file_path, edges)

        elif child.type == "function_declaration":
            _extract_function(child, file_path, nodes, edges)

        elif child.type == "method_declaration":
            _extract_method(child, file_path, nodes, edges)

        elif child.type == "type_declaration":
            _extract_type_declaration(child, file_path, nodes)

    return nodes, edges


def _extract_imports(
    node: tree_sitter.Node,
    file_path: str,
    edges: list[Edge],
) -> None:
    """Extract import declarations (single or grouped)."""
    for child in node.children:
        if child.type == "import_spec":
            _add_import_edge(child, file_path, edges)
        elif child.type == "import_spec_list":
            for spec in child.children:
                if spec.type == "import_spec":
                    _add_import_edge(spec, file_path, edges)


def _add_import_edge(
    spec: tree_sitter.Node,
    file_path: str,
    edges: list[Edge],
) -> None:
    path_node = find_child(spec, "interpreted_string_literal")
    if path_node and path_node.text:
        module_path = path_node.text.decode("utf-8").strip('"')
        # Check for alias
        alias_node = find_child(spec, "package_identifier")
        alias = node_text(alias_node) if alias_node else ""
        context = f'import {alias} "{module_path}"' if alias else f'import "{module_path}"'

        edges.append(Edge(
            source=Node.make_id(file_path, "__file__"),
            target=Node.make_id(module_path, "__file__"),
            relation=RelationType.IMPORTS,
            metadata=EdgeMetadata(context=context),
        ))


def _extract_function(
    node: tree_sitter.Node,
    file_path: str,
    nodes: list[Node],
    edges: list[Edge],
) -> None:
    name = get_child_text(node, "identifier")
    if not name:
        return

    params = extract_parameters(node)
    return_type = _get_return_type(node)

    func_node = Node(
        id=Node.make_id(file_path, name),
        type=NodeType.FUNCTION,
        name=name,
        loc=Location(file=file_path, lines=node_lines(node)),
        language="go",
        meta=NodeMeta(parameters=params, return_type=return_type),
    )
    nodes.append(func_node)

    body = find_child(node, "block")
    if body:
        _extract_http_calls(body, file_path, func_node.id, nodes, edges)


def _extract_method(
    node: tree_sitter.Node,
    file_path: str,
    nodes: list[Node],
    edges: list[Edge],
) -> None:
    """Extract a method declaration (func (receiver) Name(...) ...)."""
    # Get method name
    method_name = get_child_text(node, "field_identifier")
    if not method_name:
        return

    # Get receiver type
    receiver_type = ""
    param_list = find_child(node, "parameter_list")
    if param_list:
        for child in param_list.children:
            if child.type == "parameter_declaration":
                # The type is the last named child (could be pointer_type or type_identifier)
                for sub in child.children:
                    if sub.type == "type_identifier":
                        receiver_type = node_text(sub)
                    elif sub.type == "pointer_type":
                        inner = find_child(sub, "type_identifier")
                        if inner:
                            receiver_type = node_text(inner)

    full_name = f"{receiver_type}.{method_name}" if receiver_type else method_name
    params = _get_method_params(node)
    return_type = _get_return_type(node)

    method_node = Node(
        id=Node.make_id(file_path, full_name),
        type=NodeType.METHOD,
        name=full_name,
        loc=Location(file=file_path, lines=node_lines(node)),
        language="go",
        meta=NodeMeta(parameters=params, return_type=return_type),
    )
    nodes.append(method_node)

    # Link method to its receiver type if we have it
    if receiver_type:
        edges.append(Edge(
            source=Node.make_id(file_path, receiver_type),
            target=method_node.id,
            relation=RelationType.DEFINES,
        ))

    body = find_child(node, "block")
    if body:
        _extract_http_calls(body, file_path, method_node.id, nodes, edges)


def _get_method_params(node: tree_sitter.Node) -> list[str]:
    """Get parameters from a method, skipping the receiver (first parameter_list)."""
    params: list[str] = []
    found_first = False
    for child in node.children:
        if child.type == "parameter_list":
            if not found_first:
                found_first = True  # Skip receiver
                continue
            # This is the actual parameter list
            for param in child.children:
                if param.type == "parameter_declaration":
                    ident = find_child(param, "identifier")
                    if ident and ident.text:
                        params.append(ident.text.decode("utf-8"))
            break
    return params


def _extract_type_declaration(
    node: tree_sitter.Node,
    file_path: str,
    nodes: list[Node],
) -> None:
    """Extract type declarations: structs, interfaces, type aliases."""
    for child in node.children:
        if child.type == "type_spec":
            name = get_child_text(child, "type_identifier")
            if not name:
                continue

            # Determine the kind of type
            struct_type = find_child(child, "struct_type")
            interface_type = find_child(child, "interface_type")

            if interface_type:
                nodes.append(Node(
                    id=Node.make_id(file_path, name),
                    type=NodeType.INTERFACE,
                    name=name,
                    loc=Location(file=file_path, lines=node_lines(child)),
                    language="go",
                ))
            elif struct_type:
                nodes.append(Node(
                    id=Node.make_id(file_path, name),
                    type=NodeType.CLASS,  # Structs as CLASS type
                    name=name,
                    loc=Location(file=file_path, lines=node_lines(child)),
                    language="go",
                    summary="struct",
                ))
            else:
                # Type alias
                nodes.append(Node(
                    id=Node.make_id(file_path, name),
                    type=NodeType.CLASS,
                    name=name,
                    loc=Location(file=file_path, lines=node_lines(child)),
                    language="go",
                    summary="type alias",
                ))


def _extract_http_calls(
    node: tree_sitter.Node,
    file_path: str,
    source_id: str,
    nodes: list[Node],
    edges: list[Edge],
) -> None:
    """Recursively find function calls, creating CALLS edges and HTTP endpoint nodes."""
    if node.type == "call_expression":
        callee = node.children[0] if node.children else None
        if callee:
            callee_text = callee.text.decode("utf-8") if callee.text else ""

            callee_name = resolve_callee_name(callee_text)
            if callee_name:
                edges.append(Edge(
                    source=source_id,
                    target=Node.make_id("", callee_name),
                    relation=RelationType.CALLS,
                    metadata=EdgeMetadata(confidence=0.8, context=callee_name),
                ))

            if callee_text.startswith("http.") and callee_text in (
                "http.Get", "http.Post", "http.Head",
                "http.NewRequest",
            ):
                args = find_child(node, "argument_list")
                if args:
                    for arg in args.children:
                        if arg.type == "interpreted_string_literal":
                            url = arg.text.decode("utf-8").strip('"') if arg.text else ""
                            if url:
                                endpoint_id = Node.make_id("", f"endpoint:{url}")
                                nodes.append(Node(
                                    id=endpoint_id,
                                    type=NodeType.ENDPOINT,
                                    name=url,
                                    loc=Location(file=file_path, lines=node_lines(node)),
                                    language="go",
                                ))
                                edges.append(Edge(
                                    source=source_id,
                                    target=endpoint_id,
                                    relation=RelationType.NETWORKS_TO,
                                    metadata=EdgeMetadata(
                                        is_bridge=True,
                                        context=f"{callee_text}(\"{url}\")",
                                    ),
                                ))
                            break

    for child in node.children:
        _extract_http_calls(child, file_path, source_id, nodes, edges)


def _get_return_type(node: tree_sitter.Node) -> str | None:
    """Extract return type from a Go function/method."""
    for child in node.children:
        if child.type in ("type_identifier", "pointer_type", "slice_type",
                          "map_type", "qualified_type", "array_type"):
            return node_text(child)
        elif child.type == "parameter_list":
            # Could be result parameters — check if it comes after the func params
            pass
    return None


# Register this extractor
from hammy.tools.languages import register_extractor  # noqa: E402

register_extractor("go", extract)
