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
            ParserFactory(["cobol"])

    def test_detect_language(self):
        factory = ParserFactory()
        assert factory.detect_language(Path("file.php")) == "php"
        assert factory.detect_language(Path("file.js")) == "javascript"
        assert factory.detect_language(Path("file.jsx")) == "javascript"
        assert factory.detect_language(Path("file.mjs")) == "javascript"
        assert factory.detect_language(Path("file.py")) == "python"
        assert factory.detect_language(Path("file.ts")) == "typescript"
        assert factory.detect_language(Path("file.go")) == "go"
        assert factory.detect_language(Path("file.rb")) is None

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
        result = factory.parse_file(Path("file.rb"))
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


class TestPythonExtraction:
    @pytest.fixture
    def py_app_symbols(self):
        factory = ParserFactory()
        tree, lang = factory.parse_file(FIXTURES / "sample_python" / "app.py")
        return extract_symbols(tree, lang, "app.py")

    @pytest.fixture
    def py_model_symbols(self):
        factory = ParserFactory()
        tree, lang = factory.parse_file(FIXTURES / "sample_python" / "models.py")
        return extract_symbols(tree, lang, "models.py")

    def test_extracts_functions(self, py_app_symbols):
        nodes, _ = py_app_symbols
        functions = [n for n in nodes if n.type == NodeType.FUNCTION]
        names = [f.name for f in functions]
        assert "get_users" in names
        assert "get_user" in names
        assert "create_user" in names
        assert "helper_function" in names

    def test_extracts_async(self, py_app_symbols):
        nodes, _ = py_app_symbols
        funcs = {n.name: n for n in nodes if n.type == NodeType.FUNCTION}
        assert funcs["get_user"].meta.is_async is True
        assert funcs["get_users"].meta.is_async is False

    def test_extracts_route_endpoints(self, py_app_symbols):
        nodes, _ = py_app_symbols
        endpoints = [n for n in nodes if n.type == NodeType.ENDPOINT]
        endpoint_names = [e.name for e in endpoints]
        assert "/api/users" in endpoint_names
        assert "/api/users/<int:user_id>" in endpoint_names

    def test_extracts_imports(self, py_app_symbols):
        _, edges = py_app_symbols
        imports = [e for e in edges if e.relation == RelationType.IMPORTS]
        contexts = [e.metadata.context for e in imports]
        assert any("flask" in c for c in contexts)
        assert any("models" in c for c in contexts)

    def test_extracts_classes(self, py_model_symbols):
        nodes, _ = py_model_symbols
        classes = [n for n in nodes if n.type == NodeType.CLASS]
        names = [c.name for c in classes]
        assert "User" in names
        assert "UserService" in names

    def test_extracts_methods(self, py_model_symbols):
        nodes, _ = py_model_symbols
        methods = [n for n in nodes if n.type == NodeType.METHOD]
        names = [m.name for m in methods]
        assert "UserService.get_user" in names
        assert "UserService.list_users" in names
        assert "UserService.create_user" in names

    def test_method_filters_self(self, py_model_symbols):
        nodes, _ = py_model_symbols
        methods = {n.name: n for n in nodes if n.type == NodeType.METHOD}
        # 'self' should be filtered from params
        assert "self" not in methods["UserService.get_user"].meta.parameters

    def test_extracts_standalone_function(self, py_model_symbols):
        nodes, _ = py_model_symbols
        funcs = [n for n in nodes if n.type == NodeType.FUNCTION]
        assert any(f.name == "format_user" for f in funcs)

    def test_extracts_return_types(self, py_model_symbols):
        nodes, _ = py_model_symbols
        methods = {n.name: n for n in nodes if n.type == NodeType.METHOD}
        assert methods["UserService.get_user"].meta.return_type == "dict"
        assert methods["UserService.list_users"].meta.return_type == "list"


class TestTypeScriptExtraction:
    @pytest.fixture
    def ts_symbols(self):
        factory = ParserFactory()
        tree, lang = factory.parse_file(FIXTURES / "sample_typescript" / "app.ts")
        return extract_symbols(tree, lang, "app.ts")

    def test_extracts_interface(self, ts_symbols):
        nodes, _ = ts_symbols
        interfaces = [n for n in nodes if n.type == NodeType.INTERFACE]
        assert len(interfaces) == 1
        assert interfaces[0].name == "UserDTO"

    def test_extracts_enum(self, ts_symbols):
        nodes, _ = ts_symbols
        enums = [n for n in nodes if n.summary == "enum"]
        assert len(enums) == 1
        assert enums[0].name == "UserRole"

    def test_extracts_class(self, ts_symbols):
        nodes, _ = ts_symbols
        classes = [n for n in nodes if n.type == NodeType.CLASS and n.summary != "enum"]
        assert len(classes) == 1
        assert classes[0].name == "UserController"

    def test_extracts_methods(self, ts_symbols):
        nodes, _ = ts_symbols
        methods = [n for n in nodes if n.type == NodeType.METHOD]
        names = [m.name for m in methods]
        assert "UserController.getUser" in names
        assert "UserController.createUser" in names
        assert "UserController.validateEmail" in names

    def test_method_async(self, ts_symbols):
        nodes, _ = ts_symbols
        methods = {n.name: n for n in nodes if n.type == NodeType.METHOD}
        assert methods["UserController.getUser"].meta.is_async is True
        assert methods["UserController.validateEmail"].meta.is_async is False

    def test_method_visibility(self, ts_symbols):
        nodes, _ = ts_symbols
        methods = {n.name: n for n in nodes if n.type == NodeType.METHOD}
        assert methods["UserController.getUser"].meta.visibility == "public"
        assert methods["UserController.validateEmail"].meta.visibility == "private"

    def test_extracts_imports(self, ts_symbols):
        _, edges = ts_symbols
        imports = [e for e in edges if e.relation == RelationType.IMPORTS]
        contexts = [e.metadata.context for e in imports]
        assert any("express" in c for c in contexts)
        assert any("services" in c for c in contexts)

    def test_extracts_arrow_function(self, ts_symbols):
        nodes, _ = ts_symbols
        funcs = [n for n in nodes if n.type == NodeType.FUNCTION]
        names = [f.name for f in funcs]
        assert "fetchUsers" in names

    def test_extracts_exported_function(self, ts_symbols):
        nodes, _ = ts_symbols
        funcs = [n for n in nodes if n.type == NodeType.FUNCTION]
        assert any(f.name == "formatUser" for f in funcs)

    def test_extracts_fetch_calls(self, ts_symbols):
        nodes, _ = ts_symbols
        endpoints = [n for n in nodes if n.type == NodeType.ENDPOINT]
        assert any(e.name == "/api/v1/users" for e in endpoints)

    def test_class_defines_methods(self, ts_symbols):
        _, edges = ts_symbols
        defines = [e for e in edges if e.relation == RelationType.DEFINES]
        assert len(defines) >= 3


class TestGoExtraction:
    @pytest.fixture
    def go_symbols(self):
        factory = ParserFactory()
        tree, lang = factory.parse_file(FIXTURES / "sample_go" / "main.go")
        return extract_symbols(tree, lang, "main.go")

    def test_extracts_functions(self, go_symbols):
        nodes, _ = go_symbols
        functions = [n for n in nodes if n.type == NodeType.FUNCTION]
        names = [f.name for f in functions]
        assert "NewUser" in names
        assert "fetchData" in names

    def test_extracts_struct(self, go_symbols):
        nodes, _ = go_symbols
        structs = [n for n in nodes if n.summary == "struct"]
        names = [s.name for s in structs]
        assert "User" in names

    def test_extracts_interface(self, go_symbols):
        nodes, _ = go_symbols
        interfaces = [n for n in nodes if n.type == NodeType.INTERFACE]
        assert len(interfaces) == 1
        assert interfaces[0].name == "UserService"

    def test_extracts_methods(self, go_symbols):
        nodes, _ = go_symbols
        methods = [n for n in nodes if n.type == NodeType.METHOD]
        names = [m.name for m in methods]
        assert "User.FullName" in names
        assert "User.Validate" in names

    def test_method_defines_edge(self, go_symbols):
        _, edges = go_symbols
        defines = [e for e in edges if e.relation == RelationType.DEFINES]
        assert len(defines) >= 2  # FullName and Validate

    def test_extracts_imports(self, go_symbols):
        _, edges = go_symbols
        imports = [e for e in edges if e.relation == RelationType.IMPORTS]
        contexts = [e.metadata.context for e in imports]
        assert any("fmt" in c for c in contexts)
        assert any("net/http" in c for c in contexts)

    def test_extracts_http_calls(self, go_symbols):
        nodes, edges = go_symbols
        endpoints = [n for n in nodes if n.type == NodeType.ENDPOINT]
        assert any("http://api.example.com/users" in e.name for e in endpoints)
        bridge_edges = [e for e in edges if e.metadata and e.metadata.is_bridge]
        assert len(bridge_edges) >= 1

    def test_function_params(self, go_symbols):
        nodes, _ = go_symbols
        funcs = {n.name: n for n in nodes if n.type == NodeType.FUNCTION}
        assert "name" in funcs["NewUser"].meta.parameters
        assert "email" in funcs["NewUser"].meta.parameters

    def test_extracts_calls(self, go_symbols):
        _, edges = go_symbols
        calls = [e for e in edges if e.relation == RelationType.CALLS]
        contexts = [c.metadata.context for c in calls]
        assert any("http.Get" in c for c in contexts)


class TestCommonJSExtraction:
    @pytest.fixture
    def commonjs_symbols(self):
        factory = ParserFactory()
        tree, lang = factory.parse_file(FIXTURES / "sample_js" / "commonjs.js")
        return extract_symbols(tree, lang, "commonjs.js")

    def test_var_function_expression(self, commonjs_symbols):
        nodes, _ = commonjs_symbols
        funcs = [n for n in nodes if n.type == NodeType.FUNCTION]
        names = [f.name for f in funcs]
        assert "legacyHelper" in names

    def test_const_function_expression(self, commonjs_symbols):
        nodes, _ = commonjs_symbols
        funcs = [n for n in nodes if n.type == NodeType.FUNCTION]
        # function name takes priority over variable name
        names = [f.name for f in funcs]
        assert "compute" in names or "withFuncExpr" in names

    def test_plain_function(self, commonjs_symbols):
        nodes, _ = commonjs_symbols
        funcs = [n for n in nodes if n.type == NodeType.FUNCTION]
        names = [f.name for f in funcs]
        assert "plainFunction" in names

    def test_module_exports_function(self, commonjs_symbols):
        nodes, _ = commonjs_symbols
        funcs = [n for n in nodes if n.type == NodeType.FUNCTION]
        names = [f.name for f in funcs]
        assert "handler" in names

    def test_exports_named(self, commonjs_symbols):
        nodes, _ = commonjs_symbols
        funcs = [n for n in nodes if n.type == NodeType.FUNCTION]
        names = [f.name for f in funcs]
        assert "namedExport" in names

    def test_extracts_params(self, commonjs_symbols):
        nodes, _ = commonjs_symbols
        funcs = {n.name: n for n in nodes if n.type == NodeType.FUNCTION}
        assert funcs["legacyHelper"].meta.parameters == ["x"]
        assert funcs["plainFunction"].meta.parameters == ["a", "b"]


class TestCallTracking:
    """Test that CALLS edges are created when functions invoke other functions."""

    def test_php_calls(self):
        factory = ParserFactory()
        tree, lang = factory.parse_file(FIXTURES / "sample_php" / "UserController.php")
        _, edges = extract_symbols(tree, lang, "UserController.php")
        calls = [e for e in edges if e.relation == RelationType.CALLS]
        contexts = [c.metadata.context for c in calls]
        assert len(calls) >= 1
        assert any("charge" in c for c in contexts)

    def test_js_calls(self):
        factory = ParserFactory()
        tree, lang = factory.parse_file(FIXTURES / "sample_js" / "api.js")
        _, edges = extract_symbols(tree, lang, "api.js")
        calls = [e for e in edges if e.relation == RelationType.CALLS]
        contexts = [c.metadata.context for c in calls]
        assert len(calls) >= 1
        assert any("fetch" in c for c in contexts)

    def test_python_calls(self):
        factory = ParserFactory()
        tree, lang = factory.parse_file(FIXTURES / "sample_python" / "models.py")
        _, edges = extract_symbols(tree, lang, "models.py")
        calls = [e for e in edges if e.relation == RelationType.CALLS]
        contexts = [c.metadata.context for c in calls]
        assert len(calls) >= 1
        assert any("self.db" in c for c in contexts)

    def test_ts_calls(self):
        factory = ParserFactory()
        tree, lang = factory.parse_file(FIXTURES / "sample_typescript" / "app.ts")
        _, edges = extract_symbols(tree, lang, "app.ts")
        calls = [e for e in edges if e.relation == RelationType.CALLS]
        contexts = [c.metadata.context for c in calls]
        assert len(calls) >= 1
        assert any("this.service" in c for c in contexts)

    def test_go_calls(self):
        factory = ParserFactory()
        tree, lang = factory.parse_file(FIXTURES / "sample_go" / "main.go")
        _, edges = extract_symbols(tree, lang, "main.go")
        calls = [e for e in edges if e.relation == RelationType.CALLS]
        contexts = [c.metadata.context for c in calls]
        assert len(calls) >= 1
        assert any("http.Get" in c for c in contexts)

    def test_calls_have_confidence(self):
        factory = ParserFactory()
        tree, lang = factory.parse_file(FIXTURES / "sample_js" / "api.js")
        _, edges = extract_symbols(tree, lang, "api.js")
        calls = [e for e in edges if e.relation == RelationType.CALLS]
        assert all(c.metadata.confidence == 0.8 for c in calls)
