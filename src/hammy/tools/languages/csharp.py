"""C# AST extraction — classes, interfaces, methods, constructors, imports, ASP.NET endpoints."""

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
    extract_comments,
    find_child,
    node_lines,
    node_text,
    resolve_callee_name,
)

# ASP.NET HTTP verb attributes → endpoint detection
_HTTP_VERBS = {"HttpGet", "HttpPost", "HttpPut", "HttpDelete", "HttpPatch", "HttpHead", "HttpOptions"}


def extract(tree: tree_sitter.Tree, file_path: str) -> tuple[list[Node], list[Edge]]:
    """Extract all symbols from a C# file."""
    nodes: list[Node] = []
    edges: list[Edge] = []

    for child in tree.root_node.children:
        if child.type == "using_directive":
            _extract_using(child, file_path, edges)
        elif child.type == "namespace_declaration":
            _extract_namespace(child, file_path, nodes, edges)
        elif child.type == "class_declaration":
            _extract_class(child, file_path, nodes, edges, namespace="")
        elif child.type == "interface_declaration":
            _extract_interface(child, file_path, nodes, edges, namespace="")

    comment_nodes = extract_comments(
        tree, file_path, "csharp",
        [n for n in nodes if n.type != NodeType.COMMENT],
        frozenset({"line_comment", "block_comment"}),
    )
    nodes.extend(comment_nodes)
    return nodes, edges


def _extract_using(
    node: tree_sitter.Node,
    file_path: str,
    edges: list[Edge],
) -> None:
    """Extract `using Foo.Bar;` import edges."""
    # Children: 'using' keyword, then qualified_name or identifier, then ';'
    for child in node.children:
        if child.type in ("qualified_name", "identifier"):
            module = node_text(child)
            if module:
                edges.append(Edge(
                    source=Node.make_id(file_path, "__file__"),
                    target=Node.make_id(module, "__file__"),
                    relation=RelationType.IMPORTS,
                    metadata=EdgeMetadata(context=f"using {module}"),
                ))
            break


def _extract_namespace(
    node: tree_sitter.Node,
    file_path: str,
    nodes: list[Node],
    edges: list[Edge],
) -> None:
    """Walk a namespace_declaration and extract its type declarations."""
    ns_name = ""
    for child in node.children:
        if child.type in ("qualified_name", "identifier"):
            ns_name = node_text(child)
            break

    decl_list = find_child(node, "declaration_list")
    if decl_list:
        for member in decl_list.children:
            if member.type == "class_declaration":
                _extract_class(member, file_path, nodes, edges, namespace=ns_name)
            elif member.type == "interface_declaration":
                _extract_interface(member, file_path, nodes, edges, namespace=ns_name)
            elif member.type == "namespace_declaration":
                _extract_namespace(member, file_path, nodes, edges)


def _extract_class(
    node: tree_sitter.Node,
    file_path: str,
    nodes: list[Node],
    edges: list[Edge],
    namespace: str,
) -> None:
    class_name = _get_identifier(node)
    if not class_name:
        return

    full_name = f"{namespace}.{class_name}" if namespace else class_name
    visibility = _get_visibility(node)

    class_node = Node(
        id=Node.make_id(file_path, full_name),
        type=NodeType.CLASS,
        name=full_name,
        loc=Location(file=file_path, lines=node_lines(node)),
        language="csharp",
        meta=NodeMeta(visibility=visibility),
    )
    nodes.append(class_node)

    decl_list = find_child(node, "declaration_list")
    if decl_list:
        for member in decl_list.children:
            if member.type == "method_declaration":
                _extract_method(member, file_path, full_name, class_node.id, nodes, edges)
            elif member.type == "constructor_declaration":
                _extract_constructor(member, file_path, full_name, class_node.id, nodes, edges)
            elif member.type == "class_declaration":
                # Nested class
                _extract_class(member, file_path, nodes, edges, namespace=full_name)


def _extract_interface(
    node: tree_sitter.Node,
    file_path: str,
    nodes: list[Node],
    edges: list[Edge],
    namespace: str,
) -> None:
    iface_name = _get_identifier(node)
    if not iface_name:
        return

    full_name = f"{namespace}.{iface_name}" if namespace else iface_name

    iface_node = Node(
        id=Node.make_id(file_path, full_name),
        type=NodeType.INTERFACE,
        name=full_name,
        loc=Location(file=file_path, lines=node_lines(node)),
        language="csharp",
        meta=NodeMeta(visibility="public"),
    )
    nodes.append(iface_node)


def _extract_method(
    node: tree_sitter.Node,
    file_path: str,
    class_name: str,
    class_id: str,
    nodes: list[Node],
    edges: list[Edge],
) -> None:
    method_name = _get_identifier(node)
    if not method_name:
        return

    full_name = f"{class_name}.{method_name}"
    visibility = _get_visibility(node)
    is_async = _has_modifier(node, "async")
    params = _extract_parameters(node)
    return_type = _get_return_type(node)

    # Check for ASP.NET route attributes
    route = _extract_route(node)

    method_node = Node(
        id=Node.make_id(file_path, full_name),
        type=NodeType.METHOD,
        name=full_name,
        loc=Location(file=file_path, lines=node_lines(node)),
        language="csharp",
        meta=NodeMeta(
            visibility=visibility,
            is_async=is_async,
            parameters=params,
            return_type=return_type,
        ),
    )
    nodes.append(method_node)

    edges.append(Edge(
        source=class_id,
        target=method_node.id,
        relation=RelationType.DEFINES,
    ))

    # Extract endpoint node for ASP.NET action methods
    if route:
        endpoint_node = Node(
            id=Node.make_id(file_path, f"endpoint:{route}"),
            type=NodeType.ENDPOINT,
            name=route,
            loc=Location(file=file_path, lines=node_lines(node)),
            language="csharp",
        )
        nodes.append(endpoint_node)
        edges.append(Edge(
            source=method_node.id,
            target=endpoint_node.id,
            relation=RelationType.DEFINES,
        ))

    # Extract call edges from method body
    body = find_child(node, "block")
    if body:
        _extract_calls(body, file_path, method_node.id, edges)

    # Expression-bodied methods: => expression
    arrow = find_child(node, "arrow_expression_clause")
    if arrow:
        _extract_calls(arrow, file_path, method_node.id, edges)


def _extract_constructor(
    node: tree_sitter.Node,
    file_path: str,
    class_name: str,
    class_id: str,
    nodes: list[Node],
    edges: list[Edge],
) -> None:
    ctor_name = _get_identifier(node)
    if not ctor_name:
        return

    full_name = f"{class_name}.{ctor_name}.__ctor__"
    visibility = _get_visibility(node)
    params = _extract_parameters(node)

    ctor_node = Node(
        id=Node.make_id(file_path, full_name),
        type=NodeType.METHOD,
        name=f"{class_name}.{ctor_name}",
        loc=Location(file=file_path, lines=node_lines(node)),
        language="csharp",
        meta=NodeMeta(visibility=visibility, parameters=params),
    )
    nodes.append(ctor_node)

    edges.append(Edge(
        source=class_id,
        target=ctor_node.id,
        relation=RelationType.DEFINES,
    ))

    body = find_child(node, "block")
    if body:
        _extract_calls(body, file_path, ctor_node.id, edges)


def _extract_calls(
    node: tree_sitter.Node,
    file_path: str,
    source_id: str,
    edges: list[Edge],
) -> None:
    """Recursively find invocation_expression nodes and emit CALLS edges."""
    if node.type == "invocation_expression":
        # Children: callee_expression, argument_list
        callee = node.children[0] if node.children else None
        if callee:
            callee_text = node_text(callee)
            callee_name = resolve_callee_name(callee_text)
            if callee_name:
                full_expr = node.text.decode("utf-8") if node.text else callee_name
                edges.append(Edge(
                    source=source_id,
                    target=Node.make_id("", callee_name),
                    relation=RelationType.CALLS,
                    metadata=EdgeMetadata(confidence=0.8, context=full_expr[:200]),
                ))

    for child in node.children:
        _extract_calls(child, file_path, source_id, edges)


# --- Helpers ---

def _get_identifier(node: tree_sitter.Node) -> str:
    """Get the name identifier from a declaration node."""
    for child in node.children:
        if child.type == "identifier" and child.text:
            return child.text.decode("utf-8")
    return ""


def _get_visibility(node: tree_sitter.Node) -> str | None:
    """Extract the first visibility modifier (public/private/protected/internal)."""
    for child in node.children:
        if child.type == "modifier" and child.text:
            text = child.text.decode("utf-8")
            if text in ("public", "private", "protected", "internal"):
                return text
    return None


def _has_modifier(node: tree_sitter.Node, modifier: str) -> bool:
    """Check if a declaration has a specific modifier."""
    for child in node.children:
        if child.type == "modifier" and child.text and child.text.decode("utf-8") == modifier:
            return True
    return False


def _extract_parameters(node: tree_sitter.Node) -> list[str]:
    """Extract parameter names from a method or constructor parameter_list."""
    param_list = find_child(node, "parameter_list")
    if not param_list:
        return []

    params: list[str] = []
    for child in param_list.children:
        if child.type == "parameter":
            # parameter: [attributes] [modifier] type identifier [= default]
            # The identifier is the last identifier child
            for pc in reversed(child.children):
                if pc.type == "identifier" and pc.text:
                    params.append(pc.text.decode("utf-8"))
                    break
    return params


def _get_return_type(node: tree_sitter.Node) -> str | None:
    """Extract the return type from a method_declaration.

    In C# tree-sitter, the return type appears as a type node before the
    method name identifier. We find the last non-modifier, non-attribute child
    before the identifier.
    """
    _SKIP_TYPES = {"modifier", "attribute_list", "class", "interface", "identifier", "parameter_list", "block", "arrow_expression_clause", ";"}
    type_node = None
    for child in node.children:
        if child.type == "identifier":
            break
        if child.type not in _SKIP_TYPES:
            type_node = child
    return node_text(type_node) if type_node else None


def _extract_route(node: tree_sitter.Node) -> str | None:
    """Extract route path from ASP.NET action attributes.

    Looks for [HttpGet], [HttpPost], etc. and [Route("...")] attributes.
    """
    http_verb: str | None = None
    explicit_route: str | None = None

    for child in node.children:
        if child.type != "attribute_list":
            continue
        for attr in child.children:
            if attr.type != "attribute":
                continue
            attr_name = _get_identifier(attr)
            if attr_name in _HTTP_VERBS:
                http_verb = attr_name
                # [HttpGet("path")] — route embedded in verb attribute
                arg_list = find_child(attr, "attribute_argument_list")
                if arg_list:
                    explicit_route = _extract_string_from_arg_list(arg_list)
            elif attr_name == "Route":
                arg_list = find_child(attr, "attribute_argument_list")
                if arg_list:
                    explicit_route = _extract_string_from_arg_list(arg_list)

    if http_verb is None:
        return None

    return explicit_route or f"/{http_verb}"


def _extract_string_from_arg_list(node: tree_sitter.Node) -> str | None:
    """Extract the first string literal from an attribute argument list."""
    for child in node.children:
        if child.type == "attribute_argument":
            for gc in child.children:
                if gc.type in ("string_literal", "verbatim_string_literal") and gc.text:
                    raw = gc.text.decode("utf-8")
                    return raw.strip('"@$').strip('"')
                # interpolated_string_expression or literal_expression
                elif gc.type == "literal_expression" and gc.text:
                    raw = gc.text.decode("utf-8")
                    return raw.strip('"\'')
        elif child.type in ("string_literal", "literal_expression") and child.text:
            raw = child.text.decode("utf-8")
            return raw.strip('"\'')
    return None


# Register this extractor
from hammy.tools.languages import register_extractor  # noqa: E402

register_extractor("csharp", extract)
