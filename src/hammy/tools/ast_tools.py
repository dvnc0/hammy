"""AST extraction tools — extract structured Node and Edge objects from parsed trees.

Each language has its own extraction logic since AST node types differ.
All extractors produce the same output: lists of Node and Edge objects.
"""

from __future__ import annotations

from pathlib import Path

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
    if language == "php":
        return _extract_php(tree, file_path)
    elif language == "javascript":
        return _extract_javascript(tree, file_path)
    else:
        return [], []


# --- PHP Extraction ---


def _extract_php(tree: tree_sitter.Tree, file_path: str) -> tuple[list[Node], list[Edge]]:
    nodes: list[Node] = []
    edges: list[Edge] = []
    namespace = ""

    for child in tree.root_node.children:
        if child.type == "namespace_definition":
            namespace = _get_child_text(child, "namespace_name")

        elif child.type == "namespace_use_declaration":
            _extract_php_use(child, file_path, edges)

        elif child.type == "class_declaration":
            _extract_php_class(child, file_path, namespace, nodes, edges)

        elif child.type == "function_definition":
            _extract_php_function(child, file_path, namespace, nodes)

    return nodes, edges


def _extract_php_class(
    node: tree_sitter.Node,
    file_path: str,
    namespace: str,
    nodes: list[Node],
    edges: list[Edge],
) -> None:
    class_name = _get_child_text(node, "name")
    if not class_name:
        return

    full_name = f"{namespace}\\{class_name}" if namespace else class_name

    # Check for route attributes
    route = _extract_php_route_attribute(node)

    class_node = Node(
        id=Node.make_id(file_path, full_name),
        type=NodeType.CLASS,
        name=full_name,
        loc=Location(file=file_path, lines=_node_lines(node)),
        language="php",
        meta=NodeMeta(),
        summary=f"Route: {route}" if route else "",
    )
    nodes.append(class_node)

    # If it has a route, also create an endpoint node
    if route:
        endpoint_node = Node(
            id=Node.make_id(file_path, f"endpoint:{route}"),
            type=NodeType.ENDPOINT,
            name=route,
            loc=Location(file=file_path, lines=_node_lines(node)),
            language="php",
        )
        nodes.append(endpoint_node)
        edges.append(Edge(
            source=class_node.id,
            target=endpoint_node.id,
            relation=RelationType.DEFINES,
        ))

    # Extract methods
    decl_list = _find_child(node, "declaration_list")
    if decl_list:
        for member in decl_list.children:
            if member.type == "method_declaration":
                _extract_php_method(member, file_path, full_name, class_node.id, nodes, edges)


def _extract_php_method(
    node: tree_sitter.Node,
    file_path: str,
    class_name: str,
    class_id: str,
    nodes: list[Node],
    edges: list[Edge],
) -> None:
    method_name = _get_child_text(node, "name")
    if not method_name:
        return

    full_name = f"{class_name}::{method_name}"
    visibility = _get_php_visibility(node)
    is_static = any(c.type == "static_modifier" for c in node.children)

    # Extract parameters
    params = _extract_parameters(node)

    # Extract return type
    return_type = _get_php_return_type(node)

    # Check for route attributes on the method
    route = _extract_php_route_attribute(node)

    method_node = Node(
        id=Node.make_id(file_path, full_name),
        type=NodeType.METHOD,
        name=full_name,
        loc=Location(file=file_path, lines=_node_lines(node)),
        language="php",
        meta=NodeMeta(
            visibility=visibility,
            parameters=params,
            return_type=return_type,
        ),
        summary=f"Route: {route}" if route else "",
    )
    nodes.append(method_node)

    # Edge from class to method
    edges.append(Edge(
        source=class_id,
        target=method_node.id,
        relation=RelationType.DEFINES,
    ))

    # If method has a route, create endpoint
    if route:
        endpoint_node = Node(
            id=Node.make_id(file_path, f"endpoint:{route}"),
            type=NodeType.ENDPOINT,
            name=route,
            loc=Location(file=file_path, lines=_node_lines(node)),
            language="php",
        )
        nodes.append(endpoint_node)
        edges.append(Edge(
            source=method_node.id,
            target=endpoint_node.id,
            relation=RelationType.DEFINES,
        ))


def _extract_php_function(
    node: tree_sitter.Node,
    file_path: str,
    namespace: str,
    nodes: list[Node],
) -> None:
    func_name = _get_child_text(node, "name")
    if not func_name:
        return

    full_name = f"{namespace}\\{func_name}" if namespace else func_name
    params = _extract_parameters(node)
    return_type = _get_php_return_type(node)

    nodes.append(Node(
        id=Node.make_id(file_path, full_name),
        type=NodeType.FUNCTION,
        name=full_name,
        loc=Location(file=file_path, lines=_node_lines(node)),
        language="php",
        meta=NodeMeta(parameters=params, return_type=return_type),
    ))


def _extract_php_use(
    node: tree_sitter.Node,
    file_path: str,
    edges: list[Edge],
) -> None:
    clause = _find_child(node, "namespace_use_clause")
    if clause:
        imported = clause.text.decode("utf-8") if clause.text else ""
        if imported:
            edges.append(Edge(
                source=Node.make_id(file_path, "__file__"),
                target=Node.make_id("", imported),
                relation=RelationType.IMPORTS,
                metadata=EdgeMetadata(context=f"use {imported}"),
            ))


def _extract_php_route_attribute(node: tree_sitter.Node) -> str | None:
    """Extract route path from PHP 8 attributes like #[Route('/api/users')]."""
    attr_list = _find_child(node, "attribute_list")
    if not attr_list:
        return None
    for group in attr_list.children:
        if group.type == "attribute_group":
            for attr in group.children:
                if attr.type == "attribute":
                    attr_text = attr.text.decode("utf-8") if attr.text else ""
                    if "Route" in attr_text:
                        # Extract the string argument
                        args = _find_child(attr, "arguments")
                        if args:
                            for arg in args.children:
                                if arg.type == "argument":
                                    val = _find_child(arg, "string")
                                    if val and val.text:
                                        return val.text.decode("utf-8").strip("'\"")
                                # Also check direct string children
                                elif arg.type == "string":
                                    if arg.text:
                                        return arg.text.decode("utf-8").strip("'\"")
    return None


def _get_php_visibility(node: tree_sitter.Node) -> str:
    for child in node.children:
        if child.type == "visibility_modifier":
            return child.text.decode("utf-8") if child.text else "public"
    return "public"


def _get_php_return_type(node: tree_sitter.Node) -> str | None:
    # Look for return type after the ':'
    found_colon = False
    for child in node.children:
        if child.type == ":":
            found_colon = True
        elif found_colon and child.type in ("named_type", "primitive_type", "optional_type"):
            return child.text.decode("utf-8") if child.text else None
    return None


# --- JavaScript Extraction ---


def _extract_javascript(
    tree: tree_sitter.Tree, file_path: str
) -> tuple[list[Node], list[Edge]]:
    nodes: list[Node] = []
    edges: list[Edge] = []

    for child in tree.root_node.children:
        if child.type == "import_statement":
            _extract_js_import(child, file_path, edges)

        elif child.type == "export_statement":
            _extract_js_export(child, file_path, nodes, edges)

        elif child.type == "function_declaration":
            _extract_js_function(child, file_path, nodes, edges)

        elif child.type == "class_declaration":
            _extract_js_class(child, file_path, nodes, edges)

        elif child.type == "lexical_declaration":
            _extract_js_lexical(child, file_path, nodes, edges)

    return nodes, edges


def _extract_js_import(
    node: tree_sitter.Node,
    file_path: str,
    edges: list[Edge],
) -> None:
    # Find the source string (the module path)
    source = _find_child(node, "string")
    if source and source.text:
        module_path = source.text.decode("utf-8").strip("'\"")

        # Find imported names
        import_clause = _find_child(node, "import_clause")
        imported_names: list[str] = []
        if import_clause:
            named = _find_child(import_clause, "named_imports")
            if named:
                for spec in named.children:
                    if spec.type == "import_specifier":
                        name_node = _find_child(spec, "identifier")
                        if name_node and name_node.text:
                            imported_names.append(name_node.text.decode("utf-8"))

        context = f"import {{{', '.join(imported_names)}}} from '{module_path}'"
        edges.append(Edge(
            source=Node.make_id(file_path, "__file__"),
            target=Node.make_id(module_path, "__file__"),
            relation=RelationType.IMPORTS,
            metadata=EdgeMetadata(context=context),
        ))


def _extract_js_export(
    node: tree_sitter.Node,
    file_path: str,
    nodes: list[Node],
    edges: list[Edge],
) -> None:
    # An export wraps a declaration
    for child in node.children:
        if child.type == "function_declaration":
            _extract_js_function(child, file_path, nodes, edges)
        elif child.type == "class_declaration":
            _extract_js_class(child, file_path, nodes, edges)
        elif child.type == "lexical_declaration":
            _extract_js_lexical(child, file_path, nodes, edges)


def _extract_js_function(
    node: tree_sitter.Node,
    file_path: str,
    nodes: list[Node],
    edges: list[Edge],
) -> None:
    name = _get_child_text(node, "identifier")
    if not name:
        return

    params = _extract_parameters(node)
    is_async = any(c.type == "async" for c in node.children)

    func_node = Node(
        id=Node.make_id(file_path, name),
        type=NodeType.FUNCTION,
        name=name,
        loc=Location(file=file_path, lines=_node_lines(node)),
        language="javascript",
        meta=NodeMeta(is_async=is_async, parameters=params),
    )
    nodes.append(func_node)

    # Look for fetch/axios calls in the function body
    body = _find_child(node, "statement_block")
    if body:
        _extract_js_api_calls(body, file_path, func_node.id, nodes, edges)


def _extract_js_class(
    node: tree_sitter.Node,
    file_path: str,
    nodes: list[Node],
    edges: list[Edge],
) -> None:
    class_name = _get_child_text(node, "identifier")
    if not class_name:
        return

    class_node = Node(
        id=Node.make_id(file_path, class_name),
        type=NodeType.CLASS,
        name=class_name,
        loc=Location(file=file_path, lines=_node_lines(node)),
        language="javascript",
    )
    nodes.append(class_node)

    # Extract methods from class body
    body = _find_child(node, "class_body")
    if body:
        for member in body.children:
            if member.type == "method_definition":
                _extract_js_method(member, file_path, class_name, class_node.id, nodes, edges)


def _extract_js_method(
    node: tree_sitter.Node,
    file_path: str,
    class_name: str,
    class_id: str,
    nodes: list[Node],
    edges: list[Edge],
) -> None:
    method_name = _get_child_text(node, "property_identifier")
    if not method_name:
        return

    full_name = f"{class_name}.{method_name}"
    is_async = any(c.type == "async" for c in node.children)
    params = _extract_parameters(node)

    method_node = Node(
        id=Node.make_id(file_path, full_name),
        type=NodeType.METHOD,
        name=full_name,
        loc=Location(file=file_path, lines=_node_lines(node)),
        language="javascript",
        meta=NodeMeta(is_async=is_async, parameters=params),
    )
    nodes.append(method_node)

    edges.append(Edge(
        source=class_id,
        target=method_node.id,
        relation=RelationType.DEFINES,
    ))

    # Look for fetch/axios calls inside the method body
    body = _find_child(node, "statement_block")
    if body:
        _extract_js_api_calls(body, file_path, method_node.id, nodes, edges)


def _extract_js_lexical(
    node: tree_sitter.Node,
    file_path: str,
    nodes: list[Node],
    edges: list[Edge],
) -> None:
    """Extract arrow functions assigned to const/let variables."""
    for child in node.children:
        if child.type == "variable_declarator":
            name_node = _find_child(child, "identifier")
            arrow = _find_child(child, "arrow_function")
            if name_node and arrow and name_node.text:
                name = name_node.text.decode("utf-8")
                is_async = any(c.type == "async" for c in arrow.children)
                params = _extract_parameters(arrow)

                func_node = Node(
                    id=Node.make_id(file_path, name),
                    type=NodeType.FUNCTION,
                    name=name,
                    loc=Location(file=file_path, lines=_node_lines(node)),
                    language="javascript",
                    meta=NodeMeta(is_async=is_async, parameters=params),
                )
                nodes.append(func_node)

                # Check for API calls in the arrow function body
                body = _find_child(arrow, "statement_block")
                if body:
                    _extract_js_api_calls(body, file_path, func_node.id, nodes, edges)


def _extract_js_api_calls(
    node: tree_sitter.Node,
    file_path: str,
    source_id: str,
    nodes: list[Node],
    edges: list[Edge],
) -> None:
    """Recursively find fetch() and axios calls, creating endpoint nodes and bridge edges."""
    if node.type == "call_expression":
        callee = node.children[0] if node.children else None
        if callee:
            callee_text = callee.text.decode("utf-8") if callee.text else ""
            if callee_text in ("fetch",) or callee_text.startswith("axios."):
                # Get the URL argument
                args = _find_child(node, "arguments")
                if args:
                    for arg in args.children:
                        if arg.type == "string":
                            url = arg.text.decode("utf-8").strip("'\"") if arg.text else ""
                            if url:
                                endpoint_id = Node.make_id("", f"endpoint:{url}")
                                endpoint_node = Node(
                                    id=endpoint_id,
                                    type=NodeType.ENDPOINT,
                                    name=url,
                                    loc=Location(file=file_path, lines=_node_lines(node)),
                                    language="javascript",
                                )
                                nodes.append(endpoint_node)
                                edges.append(Edge(
                                    source=source_id,
                                    target=endpoint_id,
                                    relation=RelationType.NETWORKS_TO,
                                    metadata=EdgeMetadata(
                                        is_bridge=True,
                                        context=f"{callee_text}('{url}')",
                                    ),
                                ))
                            break

    for child in node.children:
        _extract_js_api_calls(child, file_path, source_id, nodes, edges)


# --- Shared Helpers ---


def _find_child(node: tree_sitter.Node, child_type: str) -> tree_sitter.Node | None:
    """Find the first child of a given type."""
    for child in node.children:
        if child.type == child_type:
            return child
    return None


def _get_child_text(node: tree_sitter.Node, child_type: str) -> str:
    """Get the text of the first child of a given type."""
    child = _find_child(node, child_type)
    if child and child.text:
        return child.text.decode("utf-8")
    return ""


def _node_lines(node: tree_sitter.Node) -> tuple[int, int]:
    """Get 1-indexed line range for a node."""
    return (node.start_point[0] + 1, node.end_point[0] + 1)


def _extract_parameters(node: tree_sitter.Node) -> list[str]:
    """Extract parameter names from a function/method node."""
    params_node = _find_child(node, "formal_parameters")
    if not params_node:
        return []

    params: list[str] = []
    for child in params_node.children:
        if child.type in (
            "simple_parameter",
            "required_parameter",
            "optional_parameter",
        ):
            # PHP: look for variable_name child
            var = _find_child(child, "variable_name")
            if var and var.text:
                params.append(var.text.decode("utf-8"))
            else:
                # JS: look for identifier
                ident = _find_child(child, "identifier")
                if ident and ident.text:
                    params.append(ident.text.decode("utf-8"))
        elif child.type == "identifier":
            # JS simple parameter
            if child.text:
                params.append(child.text.decode("utf-8"))
    return params
