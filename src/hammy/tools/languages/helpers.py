"""Shared tree-sitter AST helpers used across all language extractors."""

from __future__ import annotations

import tree_sitter


def find_child(node: tree_sitter.Node, child_type: str) -> tree_sitter.Node | None:
    """Find the first child of a given type."""
    for child in node.children:
        if child.type == child_type:
            return child
    return None


def get_child_text(node: tree_sitter.Node, child_type: str) -> str:
    """Get the text of the first child of a given type."""
    child = find_child(node, child_type)
    if child and child.text:
        return child.text.decode("utf-8")
    return ""


def node_lines(node: tree_sitter.Node) -> tuple[int, int]:
    """Get 1-indexed line range for a node."""
    return (node.start_point[0] + 1, node.end_point[0] + 1)


def node_text(node: tree_sitter.Node) -> str:
    """Get the text content of a node."""
    return node.text.decode("utf-8") if node.text else ""


def extract_parameters(node: tree_sitter.Node) -> list[str]:
    """Extract parameter names from a function/method node.

    Handles PHP (variable_name), JS/TS (identifier), Python (identifier),
    and Go (parameter_declaration) parameter styles.
    """
    params_node = find_child(node, "formal_parameters")
    if params_node is None:
        # Go uses "parameter_list"
        params_node = find_child(node, "parameter_list")
    if params_node is None:
        # Python uses "parameters"
        params_node = find_child(node, "parameters")
    if params_node is None:
        return []

    params: list[str] = []
    for child in params_node.children:
        if child.type in (
            "simple_parameter",
            "required_parameter",
            "optional_parameter",
        ):
            # PHP: look for variable_name child
            var = find_child(child, "variable_name")
            if var and var.text:
                params.append(var.text.decode("utf-8"))
            else:
                # JS/TS: look for identifier
                ident = find_child(child, "identifier")
                if ident and ident.text:
                    params.append(ident.text.decode("utf-8"))
        elif child.type == "identifier":
            # JS simple parameter / Python parameter
            if child.text:
                params.append(child.text.decode("utf-8"))
        elif child.type == "parameter_declaration":
            # Go: parameter_declaration contains identifier(s) and type
            ident = find_child(child, "identifier")
            if ident and ident.text:
                params.append(ident.text.decode("utf-8"))
        elif child.type in ("typed_parameter", "typed_default_parameter", "default_parameter"):
            # Python typed/default parameters
            ident = find_child(child, "identifier")
            if ident and ident.text:
                params.append(ident.text.decode("utf-8"))
    return params


# Built-in/noise functions to skip when tracking call edges.
CALL_NOISE = frozenset({
    # JavaScript / TypeScript
    "console.log", "console.error", "console.warn", "console.info", "console.debug",
    "require", "setTimeout", "setInterval", "clearTimeout", "clearInterval",
    "parseInt", "parseFloat", "isNaN", "isFinite", "JSON.parse", "JSON.stringify",
    "Promise.resolve", "Promise.reject", "Promise.all", "Object.keys", "Object.values",
    "Object.assign", "Object.entries", "Array.isArray", "Array.from",
    "Math.floor", "Math.ceil", "Math.round", "Math.random", "Math.max", "Math.min",
    "String", "Number", "Boolean", "Date.now", "Error",
    # Python
    "print", "len", "range", "str", "int", "float", "bool", "list", "dict", "set",
    "tuple", "type", "isinstance", "issubclass", "hasattr", "getattr", "setattr",
    "super", "enumerate", "zip", "map", "filter", "sorted", "reversed",
    "open", "repr", "abs", "min", "max", "sum", "any", "all", "next", "iter",
    # Go
    "fmt.Println", "fmt.Printf", "fmt.Sprintf", "fmt.Fprintf", "fmt.Errorf",
    "log.Println", "log.Printf", "log.Fatal", "log.Fatalf",
    "make", "append", "len", "cap", "close", "delete", "copy", "new", "panic", "recover",
    # PHP
    "var_dump", "print_r", "echo", "isset", "unset", "empty", "is_null",
    "is_array", "is_string", "is_int", "array_map", "array_filter", "array_merge",
    "count", "strlen", "substr", "strpos", "sprintf", "implode", "explode",
    "json_encode", "json_decode", "intval", "floatval",
})


def resolve_callee_name(callee_text: str) -> str | None:
    """Resolve a callee expression to a clean function name for CALLS edges.

    Returns None if the callee should be skipped (noise/built-in).
    """
    if not callee_text or callee_text in CALL_NOISE:
        return None

    # Skip new expressions
    if callee_text.startswith("new "):
        return None

    # For method calls like this.foo() or obj.foo(), use the last segment
    # but keep the full text as context
    name = callee_text

    # Skip if it's just a property access without a call name
    if name.endswith("."):
        return None

    return name
