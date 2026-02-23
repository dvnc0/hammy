"""Python AST extraction — classes, methods, functions, imports, route decorators."""

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

# Decorator patterns that indicate route endpoints (Flask, FastAPI, etc.)
_ROUTE_METHODS = {"route", "get", "post", "put", "patch", "delete", "head", "options"}
_ROUTE_OBJECTS = {"app", "router", "blueprint", "bp", "api"}


def extract(tree: tree_sitter.Tree, file_path: str) -> tuple[list[Node], list[Edge]]:
    """Extract all symbols from a Python file."""
    nodes: list[Node] = []
    edges: list[Edge] = []

    for child in tree.root_node.children:
        if child.type == "import_statement":
            _extract_import(child, file_path, edges)

        elif child.type == "import_from_statement":
            _extract_from_import(child, file_path, edges)

        elif child.type == "class_definition":
            _extract_class(child, file_path, nodes, edges)

        elif child.type == "function_definition":
            _extract_function(child, file_path, nodes, edges)

        elif child.type == "decorated_definition":
            _extract_decorated(child, file_path, nodes, edges)

    return nodes, edges


def _extract_import(
    node: tree_sitter.Node,
    file_path: str,
    edges: list[Edge],
) -> None:
    """Extract `import foo` statements."""
    for child in node.children:
        if child.type == "dotted_name":
            module = node_text(child)
            if module:
                edges.append(Edge(
                    source=Node.make_id(file_path, "__file__"),
                    target=Node.make_id(module, "__file__"),
                    relation=RelationType.IMPORTS,
                    metadata=EdgeMetadata(context=f"import {module}"),
                ))


def _extract_from_import(
    node: tree_sitter.Node,
    file_path: str,
    edges: list[Edge],
) -> None:
    """Extract `from foo import bar` statements."""
    module = ""
    imported_names: list[str] = []

    for child in node.children:
        if child.type == "dotted_name":
            if not module:
                module = node_text(child)
            else:
                imported_names.append(node_text(child))
        elif child.type == "relative_import":
            module = node_text(child)

    if module:
        context = f"from {module} import {', '.join(imported_names)}" if imported_names else f"import {module}"
        edges.append(Edge(
            source=Node.make_id(file_path, "__file__"),
            target=Node.make_id(module, "__file__"),
            relation=RelationType.IMPORTS,
            metadata=EdgeMetadata(context=context),
        ))


def _extract_class(
    node: tree_sitter.Node,
    file_path: str,
    nodes: list[Node],
    edges: list[Edge],
) -> None:
    class_name = get_child_text(node, "identifier")
    if not class_name:
        return

    class_node = Node(
        id=Node.make_id(file_path, class_name),
        type=NodeType.CLASS,
        name=class_name,
        loc=Location(file=file_path, lines=node_lines(node)),
        language="python",
    )
    nodes.append(class_node)

    # Extract methods from the class body
    body = find_child(node, "block")
    if body:
        for member in body.children:
            if member.type == "function_definition":
                _extract_method(member, file_path, class_name, class_node.id, nodes, edges)
            elif member.type == "decorated_definition":
                func = find_child(member, "function_definition")
                if func:
                    _extract_method(func, file_path, class_name, class_node.id, nodes, edges)


def _extract_method(
    node: tree_sitter.Node,
    file_path: str,
    class_name: str,
    class_id: str,
    nodes: list[Node],
    edges: list[Edge],
) -> None:
    method_name = get_child_text(node, "identifier")
    if not method_name:
        return

    full_name = f"{class_name}.{method_name}"
    is_async = any(c.type == "async" for c in node.children)
    params = extract_parameters(node)
    return_type = _get_return_type(node)

    # Filter out 'self' and 'cls' from params
    params = [p for p in params if p not in ("self", "cls")]

    method_node = Node(
        id=Node.make_id(file_path, full_name),
        type=NodeType.METHOD,
        name=full_name,
        loc=Location(file=file_path, lines=node_lines(node)),
        language="python",
        meta=NodeMeta(is_async=is_async, parameters=params, return_type=return_type),
    )
    nodes.append(method_node)

    edges.append(Edge(
        source=class_id,
        target=method_node.id,
        relation=RelationType.DEFINES,
    ))

    body = find_child(node, "block")
    if body:
        _extract_calls(body, file_path, method_node.id, edges)


def _extract_function(
    node: tree_sitter.Node,
    file_path: str,
    nodes: list[Node],
    edges: list[Edge],
    route: str | None = None,
) -> None:
    func_name = get_child_text(node, "identifier")
    if not func_name:
        return

    is_async = any(c.type == "async" for c in node.children)
    params = extract_parameters(node)
    return_type = _get_return_type(node)

    func_node = Node(
        id=Node.make_id(file_path, func_name),
        type=NodeType.FUNCTION,
        name=func_name,
        loc=Location(file=file_path, lines=node_lines(node)),
        language="python",
        meta=NodeMeta(is_async=is_async, parameters=params, return_type=return_type),
        summary=f"Route: {route}" if route else "",
    )
    nodes.append(func_node)

    # Walk function body for calls
    body = find_child(node, "block")
    if body:
        _extract_calls(body, file_path, func_node.id, edges)

    # If this function has a route decorator, create an endpoint node
    if route:
        endpoint_node = Node(
            id=Node.make_id(file_path, f"endpoint:{route}"),
            type=NodeType.ENDPOINT,
            name=route,
            loc=Location(file=file_path, lines=node_lines(node)),
            language="python",
        )
        nodes.append(endpoint_node)
        edges.append(Edge(
            source=func_node.id,
            target=endpoint_node.id,
            relation=RelationType.DEFINES,
        ))


def _extract_decorated(
    node: tree_sitter.Node,
    file_path: str,
    nodes: list[Node],
    edges: list[Edge],
) -> None:
    """Extract a decorated definition (function or class with decorators)."""
    # Check for route decorators
    route = None
    for child in node.children:
        if child.type == "decorator":
            route = _extract_route_from_decorator(child)

    func = find_child(node, "function_definition")
    if func:
        _extract_function(func, file_path, nodes, edges, route=route)
        return

    cls = find_child(node, "class_definition")
    if cls:
        _extract_class(cls, file_path, nodes, edges)


def _extract_calls(
    node: tree_sitter.Node,
    file_path: str,
    source_id: str,
    edges: list[Edge],
) -> None:
    """Recursively find function/method calls in Python."""
    if node.type == "call":
        callee = node.children[0] if node.children else None
        if callee:
            callee_text = node_text(callee)
            callee_name = resolve_callee_name(callee_text)
            if callee_name:
                full_expr = node.text.decode("utf-8") if node.text else callee_name
                context_text = full_expr[:200]
                edges.append(Edge(
                    source=source_id,
                    target=Node.make_id("", callee_name),
                    relation=RelationType.CALLS,
                    metadata=EdgeMetadata(confidence=0.8, context=context_text),
                ))

    for child in node.children:
        _extract_calls(child, file_path, source_id, edges)


def _extract_route_from_decorator(node: tree_sitter.Node) -> str | None:
    """Extract route path from decorators like @app.route('/api/users')."""
    call = find_child(node, "call")
    if not call:
        return None

    # Check if the callee matches a route pattern (e.g., app.route, router.get)
    callee = call.children[0] if call.children else None
    if not callee:
        return None

    if callee.type == "attribute":
        obj = find_child(callee, "identifier")
        method_ident = callee.children[-1] if callee.children else None
        if obj and method_ident and method_ident.type == "identifier":
            obj_name = node_text(obj).lower()
            method_name = node_text(method_ident).lower()
            if obj_name in _ROUTE_OBJECTS and method_name in _ROUTE_METHODS:
                return _extract_first_string_arg(call)

    return None


def _extract_first_string_arg(call_node: tree_sitter.Node) -> str | None:
    """Extract the first string argument from a function call."""
    args = find_child(call_node, "argument_list")
    if not args:
        return None
    for child in args.children:
        if child.type == "string":
            return node_text(child).strip("'\"")
    return None


def _get_return_type(node: tree_sitter.Node) -> str | None:
    """Extract return type annotation from a function definition."""
    # In Python's tree-sitter, return type is a 'type' node after '->'
    found_arrow = False
    for child in node.children:
        if child.type == "->":
            found_arrow = True
        elif found_arrow and child.type == "type":
            return node_text(child)
    return None


# Register this extractor
from hammy.tools.languages import register_extractor  # noqa: E402

register_extractor("python", extract)
