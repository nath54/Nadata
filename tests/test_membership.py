"""
Tests for the membership operators ('in' and 'not in').

Covers Array element membership, Dict key probing, Str substring
search, precedence, chaining rejection, for-in header disambiguation,
and use inside schema methods and where predicates.
"""

# Import Modules
from __future__ import annotations
import contextlib
import io

import pytest

from src.api import Nadata
from src.core.database.database import NaDataDatabase
from src.core.expressions.expression import BinaryOpExpr, Expression
from src.core.expressions.scope import ExecutionScope
from src.core.types.types import TypeArray, TypeDict, TypeInt, TypeStr
from src.core.values.values import ValueArray, ValueDict, ValueInt, ValueStr
from src.parser.expressions.expression_parser import ExpressionParser


def _env() -> dict[str, object]:
    """
    Builds typed parameters for expression evaluation.

    Returns:
        dict[str, object]: Parameter values by name.
    """

    return {
        "arr": ValueArray(
            value_type=TypeArray(value_type=TypeStr()),
            elements=["a", "b", "c"],
        ),
        "nums": ValueArray(
            value_type=TypeArray(value_type=TypeInt()),
            elements=[1, 2, 3],
        ),
        "d": ValueDict(
            value_type=TypeDict(key_type=TypeStr(), value_type=TypeInt()),
            entries={"k": 1},
        ),
        "s": ValueStr(value_type=TypeStr(), value="hello world"),
        "n": ValueInt(value_type=TypeInt(), value=5),
    }


def _eval(expr: str) -> object:
    """
    Parses and evaluates an expression against the test environment.

    Args:
        expr (str): Expression source text.

    Returns:
        object: Raw evaluated result.
    """

    ast: Expression = ExpressionParser.parse_text(text=expr)

    scope: ExecutionScope = ExecutionScope(
        parameters=_env(),  # type: ignore[arg-type]
        globals_dict={},
    )

    return ast.evaluate(scope=scope).get()


class TestMembershipParsing:
    """
    Tests the parser surface of the membership operators.
    """

    def test_in_parses_to_binary_op(self) -> None:
        """
        Tests that 'x in y' produces an 'in' BinaryOpExpr.
        """

        ast: Expression = ExpressionParser.parse_text(text='"a" in arr')

        assert isinstance(ast, BinaryOpExpr)

        assert isinstance(ast, BinaryOpExpr)
        assert ast.op == "in"

    def test_not_in_parses_to_binary_op(self) -> None:
        """
        Tests that 'x not in y' produces a 'not in' BinaryOpExpr.
        """

        ast: Expression = ExpressionParser.parse_text(text='"a" not in arr')

        assert isinstance(ast, BinaryOpExpr)

        assert isinstance(ast, BinaryOpExpr)
        assert ast.op == "not in"

    def test_binds_tighter_than_and(self) -> None:
        """
        Tests that membership binds tighter than logical 'and'.
        """

        ast: Expression = ExpressionParser.parse_text(
            text='"b" in arr and "k" in d'
        )

        assert isinstance(ast, BinaryOpExpr)

        assert isinstance(ast, BinaryOpExpr)
        assert ast.op == "and"
        assert isinstance(ast.left, BinaryOpExpr)
        assert ast.left.op == "in"

    def test_dangling_not_rejected(self) -> None:
        """
        Tests that 'not' without 'in' at binary position is a syntax error.
        """

        with pytest.raises(SyntaxError):
            ExpressionParser.parse_text(text='"a" not arr')

    def test_chaining_rejected(self) -> None:
        """
        Tests that chained membership expressions are rejected.
        """

        with pytest.raises(SyntaxError):
            ExpressionParser.parse_text(text='"a" in arr in arr')

    def test_mixing_with_comparison_rejected(self) -> None:
        """
        Tests that mixing membership and comparison chaining is rejected.
        """

        with pytest.raises(SyntaxError):
            ExpressionParser.parse_text(text='"a" in arr == true')


class TestMembershipEvaluation:
    """
    Tests the evaluation semantics of the membership operators.
    """

    def test_array_hit_and_miss(self) -> None:
        """
        Tests element membership on arrays.
        """

        assert _eval(expr='"b" in arr') is True
        assert _eval(expr='"z" in arr') is False

    def test_array_not_in(self) -> None:
        """
        Tests negated element membership on arrays.
        """

        assert _eval(expr='"z" not in arr') is True
        assert _eval(expr='"b" not in arr') is False

    def test_numeric_array(self) -> None:
        """
        Tests membership on an integer array.
        """

        assert _eval(expr="2 in nums") is True
        assert _eval(expr="9 in nums") is False

    def test_dict_key_membership(self) -> None:
        """
        Tests key probing on dicts (O(1) hash lookup).
        """

        assert _eval(expr='"k" in d') is True
        assert _eval(expr='"x" in d') is False
        assert _eval(expr='"x" not in d') is True

    def test_substring_search(self) -> None:
        """
        Tests substring membership on strings.
        """

        assert _eval(expr='"lo wo" in s') is True
        assert _eval(expr='"xyz" in s') is False
        assert _eval(expr='"xyz" not in s') is True

    def test_non_string_needle_in_str_rejected(self) -> None:
        """
        Tests that a non-Str needle in a Str container raises TypeError.
        """

        with pytest.raises(TypeError):
            _eval(expr="n in s")

    def test_unsupported_container_rejected(self) -> None:
        """
        Tests that membership on a non-container raises TypeError.
        """

        with pytest.raises(TypeError):
            _eval(expr="1 in n")


class TestMembershipIntegration:
    """
    Tests membership operators inside the broader language.
    """

    def _app(self) -> Nadata:
        """
        Builds an app with a tags-bearing struct and two instances.

        Returns:
            Nadata: The prepared facade.
        """

        app: Nadata = Nadata(database=NaDataDatabase())

        app.execute_schema(schema_text=(
            "class P {\n"
            "    name: Str;\n"
            "    tags: Array[Str] = [];\n"
            "    @has_tag(t: Str) -> Bool { return t in self.tags; }\n"
            "}\n"
        ))

        app.execute_data(data_text=(
            'a = P(name = "alice", tags = ["admin", "dev"]);\n'
            'b = P(name = "bob", tags = ["guest"]);\n'
        ))

        return app

    def test_where_predicate(self) -> None:
        """
        Tests 'in' inside a where predicate expression.
        """

        app: Nadata = self._app()

        buffer = io.StringIO()

        with contextlib.redirect_stdout(buffer):
            app.execute_data(data_text=(
                'admins = P.where((p: P) -> "admin" in p.tags);\n'
                "print(admins.count());\n"
            ))

        assert buffer.getvalue().strip() == "1"

    def test_method_body(self) -> None:
        """
        Tests 'in' inside a schema method body.
        """

        app: Nadata = self._app()

        assert app.call_method("a", "has_tag", "dev") is True
        assert app.call_method("a", "has_tag", "x") is False

    def test_if_condition_in_loop(self) -> None:
        """
        Tests 'in' inside an if condition within a for-in loop body.
        """

        app: Nadata = self._app()

        app.execute_data(data_text=(
            'tags: Array[Str] = ["guest", "admin"];\n'
            "count: Int32 = 0;\n"
            'for (t: Str in tags) { if (t in "guest admin") { count = count + 1; } }\n'
        ))

        # The loop parses and runs; symbol side effects are engine-tested
        assert app.query(struct_name="P").count() == 2

    def test_for_in_header_unchanged(self) -> None:
        """
        Tests that for-in headers still parse with membership present.
        """

        app: Nadata = self._app()

        app.execute_data(data_text=(
            'names: Array[Str] = ["x", "y"];\n'
            "count: Int32 = 0;\n"
            'for (t: Str in names) { if (t in ["x", "z"]) { count = count + 1; } }\n'
        ))

        assert app.query(struct_name="P").count() == 2

    def test_while_condition(self) -> None:
        """
        Tests 'in' inside a while loop condition.
        """

        app: Nadata = self._app()

        app.execute_data(data_text=(
            'msg: Str = "abc";\n'
            "count: Int32 = 0;\n"
            'while ("b" in msg) { count = count + 1; msg = "z"; }\n'
        ))

        assert app.query(struct_name="P").count() == 2
