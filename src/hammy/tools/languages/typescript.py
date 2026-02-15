"""TypeScript AST extraction — classes, interfaces, enums, functions, imports, API calls.

Extends the JavaScript extraction patterns with TypeScript-specific constructs:
interfaces, enums, type annotations, and type identifiers.
"""

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
    """Extract all symbols from a TypeScript file."""
    nodes: list[Node] = []
    edges: list[Edge] = []

    for child in tree.root_node.children:
        if child.type == "import_statement":
            _extract_import(child, file_path, edges)

        elif child.type == "export_statement":
            _extract_export(child, file_path, nodes, edges)

        elif child.type == "function_declaration":
            _extract_function(child, file_path, nodes, edges)

        elif child.type == "class_declaration":
            _extract_class(child, file_path, nodes, edges)

        elif child.type == "lexical_declaration":
            _extract_lexical(child, file_path, nodes, edges)

        elif child.type == "variable_declaration":
            _extract_lexical(child, file_path, nodes, edges)

        elif child.type == "expression_statement":
            _extract_expression_statement(child, file_path, nodes, edges)

        elif child.type == "interface_declaration":
            _extract_interface(child, file_path, nodes)

        elif child.type == "enum_declaration":
            _extract_enum(child, file_path, nodes)

    return nodes, edges


def _extract_import(
    node: tree_sitter.Node,
    file_path: str,
    edges: list[Edge],
) -> None:
    source = find_child(node, "string")
    if source and source.text:
        raw = source.text.decode("utf-8")
        module_path = raw.strip("'\"")

        import_clause = find_child(node, "import_clause")
        imported_names: list[str] = []
        if import_clause:
            named = find_child(import_clause, "named_imports")
            if named:
                for spec in named.children:
                    if spec.type == "import_specifier":
                        name_node = find_child(spec, "identifier")
                        if name_node and name_node.text:
                            imported_names.append(name_node.text.decode("utf-8"))
            # Default import
            ident = find_child(import_clause, "identifier")
            if ident and ident.text:
                imported_names.append(ident.text.decode("utf-8"))

        context = f"import {{{', '.join(imported_names)}}} from '{module_path}'"
        edges.append(Edge(
            source=Node.make_id(file_path, "__file__"),
            target=Node.make_id(module_path, "__file__"),
            relation=RelationType.IMPORTS,
            metadata=EdgeMetadata(context=context),
        ))


def _extract_export(
    node: tree_sitter.Node,
    file_path: str,
    nodes: list[Node],
    edges: list[Edge],
) -> None:
    for child in node.children:
        if child.type == "function_declaration":
            _extract_function(child, file_path, nodes, edges)
        elif child.type == "class_declaration":
            _extract_class(child, file_path, nodes, edges)
        elif child.type == "lexical_declaration":
            _extract_lexical(child, file_path, nodes, edges)
        elif child.type == "variable_declaration":
            _extract_lexical(child, file_path, nodes, edges)
        elif child.type == "interface_declaration":
            _extract_interface(child, file_path, nodes)
        elif child.type == "enum_declaration":
            _extract_enum(child, file_path, nodes)


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
    is_async = any(c.type == "async" for c in node.children)
    return_type = _get_type_annotation(node)

    func_node = Node(
        id=Node.make_id(file_path, name),
        type=NodeType.FUNCTION,
        name=name,
        loc=Location(file=file_path, lines=node_lines(node)),
        language="typescript",
        meta=NodeMeta(is_async=is_async, parameters=params, return_type=return_type),
    )
    nodes.append(func_node)

    body = find_child(node, "statement_block")
    if body:
        _extract_api_calls(body, file_path, func_node.id, nodes, edges)


def _extract_class(
    node: tree_sitter.Node,
    file_path: str,
    nodes: list[Node],
    edges: list[Edge],
) -> None:
    # TypeScript uses type_identifier for class names
    class_name = get_child_text(node, "type_identifier")
    if not class_name:
        class_name = get_child_text(node, "identifier")
    if not class_name:
        return

    class_node = Node(
        id=Node.make_id(file_path, class_name),
        type=NodeType.CLASS,
        name=class_name,
        loc=Location(file=file_path, lines=node_lines(node)),
        language="typescript",
    )
    nodes.append(class_node)

    body = find_child(node, "class_body")
    if body:
        for member in body.children:
            if member.type == "method_definition":
                _extract_method(member, file_path, class_name, class_node.id, nodes, edges)


def _extract_method(
    node: tree_sitter.Node,
    file_path: str,
    class_name: str,
    class_id: str,
    nodes: list[Node],
    edges: list[Edge],
) -> None:
    method_name = get_child_text(node, "property_identifier")
    if not method_name:
        return

    full_name = f"{class_name}.{method_name}"
    is_async = any(c.type == "async" for c in node.children)
    params = extract_parameters(node)
    return_type = _get_type_annotation(node)

    # Extract visibility from accessibility_modifier
    visibility = None
    for child in node.children:
        if child.type == "accessibility_modifier":
            visibility = node_text(child)

    method_node = Node(
        id=Node.make_id(file_path, full_name),
        type=NodeType.METHOD,
        name=full_name,
        loc=Location(file=file_path, lines=node_lines(node)),
        language="typescript",
        meta=NodeMeta(
            is_async=is_async,
            parameters=params,
            return_type=return_type,
            visibility=visibility,
        ),
    )
    nodes.append(method_node)

    edges.append(Edge(
        source=class_id,
        target=method_node.id,
        relation=RelationType.DEFINES,
    ))

    body = find_child(node, "statement_block")
    if body:
        _extract_api_calls(body, file_path, method_node.id, nodes, edges)


def _extract_lexical(
    node: tree_sitter.Node,
    file_path: str,
    nodes: list[Node],
    edges: list[Edge],
) -> None:
    """Extract functions assigned to const/let/var variables."""
    for child in node.children:
        if child.type == "variable_declarator":
            name_node = find_child(child, "identifier")
            arrow = find_child(child, "arrow_function")
            if not arrow:
                arrow = find_child(child, "function_expression")
            if name_node and arrow and name_node.text:
                name = name_node.text.decode("utf-8")
                is_async = any(c.type == "async" for c in arrow.children)
                params = extract_parameters(arrow)
                return_type = _get_type_annotation(arrow)

                func_node = Node(
                    id=Node.make_id(file_path, name),
                    type=NodeType.FUNCTION,
                    name=name,
                    loc=Location(file=file_path, lines=node_lines(node)),
                    language="typescript",
                    meta=NodeMeta(is_async=is_async, parameters=params, return_type=return_type),
                )
                nodes.append(func_node)

                body = find_child(arrow, "statement_block")
                if body:
                    _extract_api_calls(body, file_path, func_node.id, nodes, edges)


def _extract_interface(
    node: tree_sitter.Node,
    file_path: str,
    nodes: list[Node],
) -> None:
    name = get_child_text(node, "type_identifier")
    if not name:
        return

    nodes.append(Node(
        id=Node.make_id(file_path, name),
        type=NodeType.INTERFACE,
        name=name,
        loc=Location(file=file_path, lines=node_lines(node)),
        language="typescript",
    ))


def _extract_enum(
    node: tree_sitter.Node,
    file_path: str,
    nodes: list[Node],
) -> None:
    name = get_child_text(node, "identifier")
    if not name:
        return

    nodes.append(Node(
        id=Node.make_id(file_path, name),
        type=NodeType.CLASS,  # Enums represented as class type
        name=name,
        loc=Location(file=file_path, lines=node_lines(node)),
        language="typescript",
        summary="enum",
    ))


def _extract_expression_statement(
    node: tree_sitter.Node,
    file_path: str,
    nodes: list[Node],
    edges: list[Edge],
) -> None:
    """Extract CommonJS module.exports and exports.foo assignments."""
    assign = find_child(node, "assignment_expression")
    if not assign:
        return

    left = assign.children[0] if assign.children else None
    if not left:
        return

    left_text = node_text(left)

    name = ""
    if left_text == "module.exports":
        name = "__default__"
    elif left_text.startswith("exports."):
        name = left_text[len("exports."):]
    else:
        return

    right = None
    for child in assign.children:
        if child.type in ("arrow_function", "function_expression", "function"):
            right = child
            break

    if not right:
        return

    func_name_node = find_child(right, "identifier")
    if func_name_node and func_name_node.text:
        name = func_name_node.text.decode("utf-8")

    is_async = any(c.type == "async" for c in right.children)
    params = extract_parameters(right)
    return_type = _get_type_annotation(right)

    func_node = Node(
        id=Node.make_id(file_path, name),
        type=NodeType.FUNCTION,
        name=name,
        loc=Location(file=file_path, lines=node_lines(node)),
        language="typescript",
        meta=NodeMeta(is_async=is_async, parameters=params, return_type=return_type),
    )
    nodes.append(func_node)

    body = find_child(right, "statement_block")
    if body:
        _extract_api_calls(body, file_path, func_node.id, nodes, edges)


def _extract_api_calls(
    node: tree_sitter.Node,
    file_path: str,
    source_id: str,
    nodes: list[Node],
    edges: list[Edge],
) -> None:
    """Recursively find function calls, creating CALLS edges and endpoint nodes."""
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

            if callee_text in ("fetch",) or callee_text.startswith("axios."):
                args = find_child(node, "arguments")
                if args:
                    for arg in args.children:
                        if arg.type == "string":
                            raw = arg.text.decode("utf-8") if arg.text else ""
                            url = raw.strip("'\"")
                            if url:
                                endpoint_id = Node.make_id("", f"endpoint:{url}")
                                nodes.append(Node(
                                    id=endpoint_id,
                                    type=NodeType.ENDPOINT,
                                    name=url,
                                    loc=Location(file=file_path, lines=node_lines(node)),
                                    language="typescript",
                                ))
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
        _extract_api_calls(child, file_path, source_id, nodes, edges)


def _get_type_annotation(node: tree_sitter.Node) -> str | None:
    """Extract return type annotation from a TypeScript function."""
    for child in node.children:
        if child.type == "type_annotation":
            # Type annotation has a ':' child and a type child
            for sub in child.children:
                if sub.type != ":":
                    return node_text(sub)
    return None


# Register this extractor
from hammy.tools.languages import register_extractor  # noqa: E402

register_extractor("typescript", extract)
