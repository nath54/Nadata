"""
Tests for indexing (arrays and dictionaries), string concatenation, and
elif chains.
"""

# Import Modules
from __future__ import annotations
from typing import Any

import pytest

from nadata.api import Nadata
from nadata.core.expressions.expression import IdentifierExpr, IndexExpr
from nadata.core.expressions.scope import ExecutionScope
from nadata.core.values.values import ValueStr
from nadata.parser.expressions.dependencies import collect_dependencies
from nadata.parser.expressions.expression_parser import ExpressionParser


def _parse_expression(text: str) -> Any:
    """
    Parses a single expression text into an AST node.

    Args:
        text (str): Expression source.

    Returns:
        Any: The parsed expression node.
    """

    return ExpressionParser.parse_text(text=text)


def _eval_expression(text: str) -> Any:
    """
    Parses and evaluates an expression in an empty scope.

    Args:
        text (str): Expression source.

    Returns:
        Any: The raw result value.
    """

    expression: Any = _parse_expression(text=text)

    return expression.evaluate(scope=ExecutionScope()).get()


class TestIndexing:
    """
    Tests parsing and evaluating index expressions.
    """

    def test_parse_index_expression(self) -> None:
        """
        Tests that 'target[index]' parses into an IndexExpr.
        """

        expression: Any = _parse_expression(text="tags[0]")

        assert isinstance(expression, IndexExpr)
        assert isinstance(expression.target, IdentifierExpr)

    def test_parse_chained_index(self) -> None:
        """
        Tests that 'a[0][1]' parses into nested IndexExpr nodes.
        """

        expression: Any = _parse_expression(text="grid[0][1]")

        assert isinstance(expression, IndexExpr)
        assert isinstance(expression.target, IndexExpr)

    def test_index_dependencies_include_index(self) -> None:
        """
        Tests that the dependency collector walks the index expression.
        """

        dependencies: set[str] = collect_dependencies(
            expression=_parse_expression(text="grid[row + 1]")
        )

        assert dependencies == {"grid", "row"}

    def test_index_array_literal(self) -> None:
        """
        Tests indexing into an array literal.
        """

        assert _eval_expression(text="[10, 20, 30][1]") == 20

    def test_index_negative(self) -> None:
        """
        Tests negative array indexing (Python semantics).
        """

        assert _eval_expression(text="[10, 20, 30][-1]") == 30

    def test_index_out_of_range_raises(self) -> None:
        """
        Tests that out-of-range indices raise IndexError.
        """

        with pytest.raises(IndexError, match="out of range"):
            _eval_expression(text="[10, 20][5]")

    def test_index_non_integer_raises(self) -> None:
        """
        Tests that indexing an array with a string raises TypeError.
        """

        with pytest.raises(TypeError, match="integer"):
            _eval_expression(text='[10, 20]["a"]')

    def test_index_dict_literal(self) -> None:
        """
        Tests indexing into a dictionary literal.
        """

        assert _eval_expression(text='{"floor": "1"}["floor"]') == "1"

    def test_index_dict_missing_key_raises(self) -> None:
        """
        Tests that missing dictionary keys raise KeyError.
        """

        with pytest.raises(KeyError, match="nope"):
            _eval_expression(text='{"floor": "1"}["nope"]')

    def test_index_through_member_access(self, capsys: Any) -> None:
        """
        Tests indexing the result of a member access on a live database.
        """

        app: Nadata = Nadata.load(path="./examples/ex0_simple")

        app.execute_data(data_text="print(tom.parents[0].name);")

        assert "Jean" in capsys.readouterr().out

    def test_index_nested_arrays(self, capsys: Any) -> None:
        """
        Tests chained indexing into nested array literals.
        """

        app: Nadata = Nadata.load(path="./examples/ex0_simple")

        app.execute_data(data_text="print([[1, 2], [3, 4]][1][0]);")

        assert "3" in capsys.readouterr().out

    def test_index_dict_on_instance(self, capsys: Any) -> None:
        """
        Tests dict indexing through an instance attribute.
        """

        app: Nadata = Nadata.load(path="./examples/ex0_simple")

        app.execute_data(data_text='print({"floor": "3"}["floor"]);')

        assert "3" in capsys.readouterr().out


class TestStringConcat:
    """
    Tests the '+' operator on Str values.
    """

    def test_concat_two_literals(self) -> None:
        """
        Tests concatenating two string literals.
        """

        assert _eval_expression(text='"Hello, " + "world"') == "Hello, world"

    def test_concat_chain(self) -> None:
        """
        Tests that concatenation chains left-associated.
        """

        assert _eval_expression(text='"a" + "b" + "c"') == "abc"

    def test_concat_keeps_str_type(self) -> None:
        """
        Tests that the result is a typed Str value.
        """

        expression: Any = _parse_expression(text='"a" + "b"')
        result: Any = expression.evaluate(scope=ExecutionScope())

        assert isinstance(result, ValueStr)

    def test_concat_with_non_str_raises(self) -> None:
        """
        Tests that Str + Int is rejected (strict typing).
        """

        with pytest.raises(ValueError, match="requires two Str operands"):
            _eval_expression(text='"a" + 1')

    def test_concat_with_non_str_left_raises(self) -> None:
        """
        Tests that Int + Str is rejected in both directions.
        """

        with pytest.raises(ValueError, match="requires two Str operands"):
            _eval_expression(text='1 + "a"')

    def test_concat_in_pipeline(self, capsys: Any) -> None:
        """
        Tests that concat composes with attributes on a live database.
        """

        app: Nadata = Nadata.load(path="./examples/ex0_simple")

        app.execute_data(
            data_text='full = tom.name + " of " + tom.birthplace; print(full);'
        )

        assert "Tom of Paris" in capsys.readouterr().out


class TestElif:
    """
    Tests elif chain parsing and execution.
    """

    def setup_method(self) -> None:
        """
        Loads the ex0 example for the elif tests.
        """

        self.app: Nadata = Nadata.load(path="./examples/ex0_simple")

    def _classify(self, value: int) -> None:
        """
        Executes an elif chain printing a classification for the value.

        Args:
            value (int): The value to classify.
        """

        self.app.execute_data(
            data_text=(
                "if ({v} < 0) {{\n"
                '    print("NEGATIVE");\n'
                "}} elif ({v} == 0) {{\n"
                '    print("ZERO");\n'
                "}} elif ({v} < 100) {{\n"
                '    print("SMALL");\n'
                "}} else {{\n"
                '    print("LARGE");\n'
                "}}"
            ).format(v=value)
        )

    def test_first_branch(self, capsys: Any) -> None:
        """
        Tests that the first matching branch executes.
        """

        self._classify(value=-5)

        assert "NEGATIVE" in capsys.readouterr().out

    def test_middle_branch(self, capsys: Any) -> None:
        """
        Tests that an elif branch executes.
        """

        self._classify(value=0)

        assert "ZERO" in capsys.readouterr().out

    def test_later_branch(self, capsys: Any) -> None:
        """
        Tests that a later elif branch executes.
        """

        self._classify(value=42)

        assert "SMALL" in capsys.readouterr().out

    def test_else_branch(self, capsys: Any) -> None:
        """
        Tests that the else branch executes when no elif matches.
        """

        self._classify(value=1000)

        assert "LARGE" in capsys.readouterr().out

    def test_elif_in_method_body(self, capsys: Any) -> None:
        """
        Tests that elif chains work inside schema method bodies.
        """

        self.app.execute_schema(
            schema_text=(
                'class Classifier {\n'
                '    @classify(value: Int32) -> Str {\n'
                '        if (value < 0) {\n'
                '            return "NEG";\n'
                '        } elif (value == 0) {\n'
                '            return "ZERO";\n'
                '        }\n'
                '        return "POS";\n'
                '    }\n'
                '}'
            )
        )

        self.app.execute_data(
            data_text=(
                'c = Classifier();\n'
                'print(c.classify(0));\n'
                'print(c.classify(-1));\n'
                'print(c.classify(7));'
            )
        )

        out: str = capsys.readouterr().out

        assert "ZERO" in out
        assert "NEG" in out
        assert "POS" in out
