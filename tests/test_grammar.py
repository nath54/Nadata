"""
Tests keeping the VS Code grammar in sync with the language.

The grammar must highlight every builtin type name (extracted from the
live TypeResolver registry plus the spatial grids and Tensor), every
language keyword, and must be structurally valid (JSON parses, scope
names match package.json).
"""

# Import Modules
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any

from src.core.types.types import SPATIAL_GRID_NAMES
from src.parser.nastruct_parser.resolver import TypeResolver

EXTENSION_DIR: Path = Path(__file__).resolve().parent.parent / "editors" / "vscode-nadata"

# Language keywords that must be highlighted
KEYWORDS: list[str] = [
    "class",
    "fn",
    "const",
    "if",
    "elif",
    "else",
    "for",
    "in",
    "while",
    "return",
    "break",
    "continue",
    "pass",
    "and",
    "or",
    "not",
    "true",
    "false",
    "self",
]

# Generic containers and home types outside the builtin scalar registry
EXTRA_TYPES: list[str] = [
    "Array",
    "Dict",
    "Set",
    "Queue",
    "PriorityQueue",
    "Tensor",
    *SPATIAL_GRID_NAMES.keys(),
]


def _load_json(relative_path: str) -> Any:
    """
    Loads a JSON file from the extension directory.

    Args:
        relative_path (str): Path relative to the extension root.

    Returns:
        Any: The parsed JSON document.
    """

    with open(EXTENSION_DIR / relative_path, encoding="utf-8") as handle:
        return json.load(handle)


class TestGrammarSync:
    """
    Tests the extension structure and language coverage of the grammar.
    """

    def test_json_files_parse(self) -> None:
        """
        Tests that all extension JSON files parse.
        """

        for relative_path in (
            "package.json",
            "language-configuration.json",
            "syntaxes/nadata.tmLanguage.json",
            "syntaxes/nastruct.tmLanguage.json",
        ):
            document: Any = _load_json(relative_path=relative_path)

            assert isinstance(document, dict)

    def test_scope_names_registered(self) -> None:
        """
        Tests that package.json wires both scope names to both languages.
        """

        package: Any = _load_json(relative_path="package.json")

        grammars: list[Any] = package["contributes"]["grammars"]
        languages: list[Any] = package["contributes"]["languages"]

        scope_to_language: dict[str, str] = {
            grammar["scopeName"]: grammar["language"] for grammar in grammars
        }

        assert scope_to_language == {
            "source.nadata": "nadata",
            "source.nastruct": "nastruct",
        }

        extensions: set[str] = {
            extension
            for language in languages
            for extension in language["extensions"]
        }

        assert extensions == {".nadata", ".nastruct"}

    def test_nastruct_aliases_nadata(self) -> None:
        """
        Tests that the nastruct grammar includes the nadata grammar.
        """

        grammar: Any = _load_json(relative_path="syntaxes/nastruct.tmLanguage.json")

        assert grammar["scopeName"] == "source.nastruct"
        assert grammar["patterns"] == [{"include": "source.nadata#"}]

    def test_builtin_types_highlighted(self) -> None:
        """
        Tests that every registered builtin type name is matched by the
        grammar's type rule (so new scalar families fail the test until
        the pattern is extended).
        """

        grammar: Any = _load_json(relative_path="syntaxes/nadata.tmLanguage.json")

        type_pattern: str = grammar["repository"]["builtin-types"]["match"]

        type_names: list[str] = [
            *TypeResolver().builtin_types.keys(),
            *EXTRA_TYPES,
        ]

        unmatched: list[str] = [
            name
            for name in type_names
            if re.search(type_pattern, name) is None
        ]

        assert unmatched == []

    def test_keywords_highlighted(self) -> None:
        """
        Tests that every language keyword is covered.
        """

        grammar_text: str = json.dumps(
            _load_json(relative_path="syntaxes/nadata.tmLanguage.json")
        )

        missing: list[str] = [
            keyword
            for keyword in KEYWORDS
            if not re.search(rf"\b{re.escape(keyword)}\b", grammar_text)
        ]

        assert missing == []

    def test_grammar_scopes_valid(self) -> None:
        """
        Tests that every scope name is a compound scope ending with the
        nadata suffix (the last token may be a canonical fallback so
        themes match either form).
        """

        grammar: Any = _load_json(relative_path="syntaxes/nadata.tmLanguage.json")

        scopes: list[str] = [
            rule["name"]
            for rule in grammar["repository"].values()
            if "name" in rule
        ]

        assert scopes
        assert all(
            any(token.endswith(".nadata") for token in scope.split(" "))
            for scope in scopes
        )

    def test_user_types_highlighted(self) -> None:
        """
        Tests that the grammar colors user-defined type names (UpperCamel)
        but leaves lowercase identifiers alone.
        """

        grammar: Any = _load_json(relative_path="syntaxes/nadata.tmLanguage.json")

        pattern: str = grammar["repository"]["user-types"]["match"]

        assert re.search(pattern, "Animal") is not None
        assert re.search(pattern, "GridHex2D") is not None
        assert re.search(pattern, "animal_attr") is None

    def test_naming_rule_scopes_present(self) -> None:
        """
        Tests that declarations, members, self, and constants all have
        dedicated highlight scopes (class keyword, class name, member
        access, storage modifier).
        """

        grammar: Any = _load_json(relative_path="syntaxes/nadata.tmLanguage.json")

        repository: dict[str, Any] = grammar["repository"]

        assert "class-declaration" in repository
        assert "method-declaration" in repository
        assert "member-access" in repository
        assert "const-keyword" in repository
        assert "self" in repository

        assert "storage.type.class" in repository["class-declaration"]["captures"]["1"]["name"]
        assert "entity.name.class" in repository["class-declaration"]["captures"]["2"]["name"]
        assert "entity.name.function" in repository["method-declaration"]["captures"]["2"]["name"]
        assert "variable.other.member" in repository["member-access"]["name"]
        assert "storage.modifier" in repository["const-keyword"]["name"]

    def test_class_keyword_without_parens(self) -> None:
        """
        Tests that class declarations without inheritance still color the
        keyword and the class name.
        """

        grammar: Any = _load_json(relative_path="syntaxes/nadata.tmLanguage.json")

        pattern: str = grammar["repository"]["class-declaration"]["match"]

        plain: re.Match[str] | None = re.search(pattern, "class Cat {")
        inherited: re.Match[str] | None = re.search(pattern, "class Dog(Animal) {")

        assert plain is not None and plain.group(1) == "class"
        assert plain.group(2) == "Cat"
        assert inherited is not None and inherited.group(3) == "Animal"

    def test_call_sites_colored(self) -> None:
        """
        Tests that callable names get a call scope at call sites, both
        bare (constructors, functions) and after a dot (methods).
        """

        grammar: Any = _load_json(relative_path="syntaxes/nadata.tmLanguage.json")

        repository: dict[str, Any] = grammar["repository"]

        assert "entity.name.function.call" in repository["call"]["name"]
        assert "entity.name.function.call" in repository["member-call"]["name"]

        call_pattern: str = repository["call"]["match"]

        assert re.search(call_pattern, "Hero(health = 10)") is not None
        assert re.search(call_pattern, "normalize(values)") is not None

    def test_parameters_and_kwargs_colored(self) -> None:
        """
        Tests that parameter names in definitions and keyword arguments in
        calls get the parameter scope.
        """

        grammar: Any = _load_json(relative_path="syntaxes/nadata.tmLanguage.json")

        rule: Any = grammar["repository"]["parameter"]
        pattern: str = rule["match"]

        assert "variable.parameter" in rule["captures"]["1"]["name"]

        definition: re.Match[str] | None = re.search(pattern, "@attack(target: Entity, amount: Int32 = 5)")
        call: re.Match[str] | None = re.search(pattern, "M(dim = 4, w = [1.0])")

        assert definition is not None and definition.group(1) == "target"
        assert call is not None and call.group(1) == "dim"

    def test_builtins_colored(self) -> None:
        """
        Tests that builtin callables (print and typed-container methods)
        get the builtin function scope.
        """

        grammar: Any = _load_json(relative_path="syntaxes/nadata.tmLanguage.json")

        repository: dict[str, Any] = grammar["repository"]

        assert "support.function.builtin" in repository["builtin-function-call"]["name"]
        assert "support.function.builtin" in repository["builtin-method-call"]["name"]

        function_pattern: str = repository["builtin-function-call"]["match"]
        method_pattern: str = repository["builtin-method-call"]["match"]

        assert re.search(function_pattern, "print(values)") is not None
        assert re.search(method_pattern, "q.push(x)") is not None
        assert re.search(method_pattern, "tensor.size()") is not None
        assert re.search(method_pattern, "s.length()") is not None

    def test_statement_separator_scoped(self) -> None:
        """
        Tests that the C-style for header separator :: has its own scope
        (so themes can color it distinctly from other operators).
        """

        grammar: Any = _load_json(relative_path="syntaxes/nadata.tmLanguage.json")

        rule: Any = grammar["repository"]["statement-separator"]
        operators: str = grammar["repository"]["operators"]["match"]

        assert rule["name"] == "punctuation.separator.statement.nadata"
        assert rule["match"] == "::"
        assert "::" not in operators
