"""Tests for the tree-sitter parser factory and AST extraction."""

from pathlib import Path

import pytest

from hammy.schema.models import NodeType, RelationType
from hammy.tools.ast_tools import extract_symbols
from hammy.tools.parser import ParserFactory

FIXTURES = Path(__file__).parent / "fixtures"


class TestParserFactory:
    def test_creates_parsers(self):
        factory = ParserFactory(["php", "javascript"])
        assert "php" in factory.enabled_languages
        assert "javascript" in factory.enabled_languages

    def test_default_enables_all(self):
        factory = ParserFactory()
        assert "php" in factory.enabled_languages
        assert "javascript" in factory.enabled_languages

    def test_unsupported_language_raises(self):
        with pytest.raises(ValueError, match="Unsupported language"):
            ParserFactory(["python"])

    def test_detect_language(self):
        factory = ParserFactory()
        assert factory.detect_language(Path("file.php")) == "php"
        assert factory.detect_language(Path("file.js")) == "javascript"
        assert factory.detect_language(Path("file.jsx")) == "javascript"
        assert factory.detect_language(Path("file.mjs")) == "javascript"
        assert factory.detect_language(Path("file.py")) is None

    def test_parse_php_file(self):
        factory = ParserFactory()
        result = factory.parse_file(FIXTURES / "sample_php" / "UserController.php")
        assert result is not None
        tree, lang = result
        assert lang == "php"
        assert tree.root_node.type == "program"

    def test_parse_js_file(self):
        factory = ParserFactory()
        result = factory.parse_file(FIXTURES / "sample_js" / "api.js")
        assert result is not None
        tree, lang = result
        assert lang == "javascript"
        assert tree.root_node.type == "program"

    def test_parse_unsupported_file_returns_none(self):
        factory = ParserFactory()
        result = factory.parse_file(Path("file.py"))
        assert result is None

    def test_parse_bytes(self):
        factory = ParserFactory()
        tree = factory.parse_bytes(b"function hello() {}", "javascript")
        assert tree.root_node.type == "program"


class TestPHPExtraction:
    @pytest.fixture
    def php_symbols(self):
        factory = ParserFactory()
        tree, lang = factory.parse_file(FIXTURES / "sample_php" / "UserController.php")
        return extract_symbols(tree, lang, "UserController.php")

    def test_extracts_class(self, php_symbols):
        nodes, _ = php_symbols
        classes = [n for n in nodes if n.type == NodeType.CLASS]
        assert len(classes) == 1
        assert classes[0].name == "App\\Controllers\\UserController"
        assert classes[0].language == "php"

    def test_extracts_methods(self, php_symbols):
        nodes, _ = php_symbols
        methods = [n for n in nodes if n.type == NodeType.METHOD]
        method_names = [m.name for m in methods]
        assert "App\\Controllers\\UserController::getUser" in method_names
        assert "App\\Controllers\\UserController::processPayment" in method_names
        assert "App\\Controllers\\UserController::validateInput" in method_names

    def test_extracts_visibility(self, php_symbols):
        nodes, _ = php_symbols
        methods = {n.name.split("::")[-1]: n for n in nodes if n.type == NodeType.METHOD}
        assert methods["getUser"].meta.visibility == "public"
        assert methods["validateInput"].meta.visibility == "private"

    def test_extracts_return_types(self, php_symbols):
        nodes, _ = php_symbols
        methods = {n.name.split("::")[-1]: n for n in nodes if n.type == NodeType.METHOD}
        assert methods["getUser"].meta.return_type == "User"
        assert methods["processPayment"].meta.return_type == "void"
        assert methods["validateInput"].meta.return_type == "bool"

    def test_extracts_route_endpoints(self, php_symbols):
        nodes, _ = php_symbols
        endpoints = [n for n in nodes if n.type == NodeType.ENDPOINT]
        endpoint_names = [e.name for e in endpoints]
        assert "/api/v1/users" in endpoint_names
        assert "/api/v1/users/{id}/pay" in endpoint_names

    def test_extracts_imports(self, php_symbols):
        _, edges = php_symbols
        imports = [e for e in edges if e.relation == RelationType.IMPORTS]
        contexts = [e.metadata.context for e in imports]
        assert any("App\\Models\\User" in c for c in contexts)
        assert any("App\\Services\\PaymentService" in c for c in contexts)

    def test_class_defines_methods(self, php_symbols):
        _, edges = php_symbols
        defines = [e for e in edges if e.relation == RelationType.DEFINES]
        assert len(defines) >= 3  # 3 methods + endpoint definitions

    def test_extracts_standalone_function(self):
        factory = ParserFactory()
        tree, lang = factory.parse_file(FIXTURES / "sample_php" / "UserRepository.php")
        nodes, _ = extract_symbols(tree, lang, "UserRepository.php")
        functions = [n for n in nodes if n.type == NodeType.FUNCTION]
        assert len(functions) == 1
        assert "helperFunction" in functions[0].name


class TestJavaScriptExtraction:
    @pytest.fixture
    def js_api_symbols(self):
        factory = ParserFactory()
        tree, lang = factory.parse_file(FIXTURES / "sample_js" / "api.js")
        return extract_symbols(tree, lang, "api.js")

    @pytest.fixture
    def js_component_symbols(self):
        factory = ParserFactory()
        tree, lang = factory.parse_file(FIXTURES / "sample_js" / "UserComponent.jsx")
        return extract_symbols(tree, lang, "UserComponent.jsx")

    def test_extracts_functions(self, js_api_symbols):
        nodes, _ = js_api_symbols
        functions = [n for n in nodes if n.type == NodeType.FUNCTION]
        names = [f.name for f in functions]
        assert "fetchUsers" in names
        assert "fetchUserProfile" in names
        assert "submitPayment" in names

    def test_extracts_async(self, js_api_symbols):
        nodes, _ = js_api_symbols
        funcs = {n.name: n for n in nodes if n.type == NodeType.FUNCTION}
        assert funcs["fetchUsers"].meta.is_async is True
        assert funcs["submitPayment"].meta.is_async is True

    def test_extracts_imports(self, js_api_symbols):
        _, edges = js_api_symbols
        imports = [e for e in edges if e.relation == RelationType.IMPORTS]
        assert len(imports) >= 1
        assert any("config" in e.metadata.context for e in imports)

    def test_extracts_fetch_calls(self, js_api_symbols):
        nodes, edges = js_api_symbols
        endpoints = [n for n in nodes if n.type == NodeType.ENDPOINT]
        endpoint_names = [e.name for e in endpoints]
        assert "/api/v1/users" in endpoint_names

    def test_extracts_axios_calls(self, js_api_symbols):
        nodes, edges = js_api_symbols
        bridge_edges = [e for e in edges if e.metadata.is_bridge]
        assert len(bridge_edges) >= 1
        assert any("axios.post" in e.metadata.context for e in bridge_edges)

    def test_extracts_class(self, js_component_symbols):
        nodes, _ = js_component_symbols
        classes = [n for n in nodes if n.type == NodeType.CLASS]
        assert len(classes) == 1
        assert classes[0].name == "UserComponent"

    def test_extracts_class_methods(self, js_component_symbols):
        nodes, _ = js_component_symbols
        methods = [n for n in nodes if n.type == NodeType.METHOD]
        names = [m.name for m in methods]
        assert "UserComponent.loadUsers" in names
        assert "UserComponent.render" in names

    def test_extracts_exported_function(self, js_component_symbols):
        nodes, _ = js_component_symbols
        funcs = [n for n in nodes if n.type == NodeType.FUNCTION]
        assert any(f.name == "formatUser" for f in funcs)

    def test_method_async(self, js_component_symbols):
        nodes, _ = js_component_symbols
        methods = {n.name: n for n in nodes if n.type == NodeType.METHOD}
        assert methods["UserComponent.loadUsers"].meta.is_async is True
        assert methods["UserComponent.render"].meta.is_async is False
