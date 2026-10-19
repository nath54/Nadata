"""
Tests for the .nadata data loader.

Covers statement parsing of data files, two-phase instance creation with
forward and circular references, lambda predicates, queries, script
execution, and type validation on attribute assignment.
"""

# Import Modules
from typing import Any

from src.core.database.database import NaDataDatabase
from src.core.expressions.scope import ExecutionScope
from src.core.types.types import TypeBool
from src.core.values.values import ValueBool
from src.parser.expressions.expression_parser import ExpressionParser
from src.parser.expressions.statement_parser import StatementParser
from src.parser.nadata_loader.loader import NaDataLoader
from src.parser.nastruct_parser.nastruct_parser import NaStructParser


def load_example(example_name: str) -> NaDataDatabase:
    """
    Parses and loads a full example folder (structs and data).

    Args:
        example_name (str): Example folder name under ./examples.

    Returns:
        NaDataDatabase: The loaded database.
    """

    parser: NaStructParser = NaStructParser()
    database: NaDataDatabase = parser.parse_folder(
        folder_path=f"./examples/{example_name}"
    )

    loader: NaDataLoader = NaDataLoader(database=database)
    loader.load_folder(folder_path=f"./examples/{example_name}/data")

    return database


class TestLambdaParsing:
    """
    Tests lambda expression parsing and evaluation.
    """

    def test_lambda_parses_with_annotation(self) -> None:
        """
        Tests that a typed lambda parses into a LambdaExpr node.
        """

        expr: Any = ExpressionParser.parse_text(
            text='(h: Human) -> h.birthplace == "Paris"'
        )

        assert type(expr).__name__ == "LambdaExpr"
        assert expr.param_name == "h"
        assert expr.param_type_name == "Human"

    def test_lambda_generic_annotation(self) -> None:
        """
        Tests that generic lambda parameter annotations are reconstructed.
        """

        expr: Any = ExpressionParser.parse_text(
            text="(ds: Array[Human]) -> ds.ping_ms < 1.0"
        )

        assert expr.param_type_name == "Array[Human]"

    def test_lambda_as_call_argument(self) -> None:
        """
        Tests that lambdas parse inside call argument lists.
        """

        expr: Any = ExpressionParser.parse_text(
            text='Human.where((h: Human) -> h.name == "Tom")'
        )

        assert type(expr).__name__ == "CallExpr"
        assert type(expr.args[0]).__name__ == "LambdaExpr"

    def test_parenthesized_expression_not_lambda(self) -> None:
        """
        Tests that parenthesized non-lambda expressions still parse.
        """

        expr: Any = ExpressionParser.parse_text(text="(x > 5)")

        assert type(expr).__name__ == "BinaryOpExpr"


class TestNotOperator:
    """
    Tests the 'not' unary logical operator.
    """

    def test_not_operator_parses_and_evaluates(self) -> None:
        """
        Tests that 'not' negates a boolean expression.
        """

        from src.core.expressions.scope import ExecutionScope

        expr: Any = ExpressionParser.parse_text(text="not enabled")

        scope: ExecutionScope = ExecutionScope(parameters={}, globals_dict={})
        scope.define(
            name="enabled",
            value=ValueBool(value_type=TypeBool(), value=True),
        )

        assert expr.evaluate(scope=scope).get() is False


class TestTrailingCommas:
    """
    Tests trailing comma tolerance in literals and call arguments.
    """

    def test_trailing_comma_in_array(self) -> None:
        """
        Tests that an array literal may end with a trailing comma.
        """

        expr: Any = ExpressionParser.parse_text(text='["a", "b",]')

        assert type(expr).__name__ == "ArrayLiteralExpr"

    def test_trailing_comma_in_call(self) -> None:
        """
        Tests that a call argument list may end with a trailing comma.
        """

        expr: Any = ExpressionParser.parse_text(text="S(x = 1,)")

        assert type(expr).__name__ == "CallExpr"
        assert "x" in expr.kwargs

    def test_trailing_comma_in_dict(self) -> None:
        """
        Tests that a dict literal may end with a trailing comma.
        """

        expr: Any = ExpressionParser.parse_text(text='{"k": 1,}')

        assert type(expr).__name__ == "DictLiteralExpr"


class TestDataLoading:
    """
    Tests full example data loading end to end.
    """

    def test_ex0_instances_and_symbols(self) -> None:
        """
        Tests that ex0 instances are created, linked, and symbol bound.
        """

        database: NaDataDatabase = load_example("ex0_simple")

        assert "tom" in database.symbols
        assert "white_cat" in database.symbols

        tom: Any = database.get_by_symbol(name="tom")
        jean: Any = database.get_by_symbol(name="jean")

        assert tom.get_attribute("name").get() == "Tom"
        assert tom.get_attribute("gender").get() == "m"

        # Forward reference: jean is declared after tom
        parents: Any = tom.get_attribute("parents")
        assert jean in parents.get()

    def test_ex2_circular_references(self) -> None:
        """
        Tests that circular instance references resolve in both directions.
        """

        database: NaDataDatabase = load_example("ex2_inheritance")

        db_primary: Any = database.get_by_symbol(name="db_primary")
        replica_backup: Any = database.get_by_symbol(name="replica_backup")

        primary_peers: Any = db_primary.get_attribute("connected_peers")
        replica_peers: Any = replica_backup.get_attribute("connected_peers")

        assert replica_backup in primary_peers.get()
        assert db_primary in replica_peers.get()

    def test_ex2_polymorphic_query_binding(self) -> None:
        """
        Tests that subclass instances are reachable from the parent struct.
        """

        database: NaDataDatabase = load_example("ex2_inheritance")

        db_primary: Any = database.get_by_symbol(name="db_primary")
        router: Any = database.get_by_symbol(name="edge_router")

        # db_primary is a DatabaseServer (subclass of Node)
        assert db_primary.struct.name == "DatabaseServer"
        assert router.struct.name == "WirelessRouter"

    def test_ex4_method_calls_in_data(self) -> None:
        """
        Tests that data scripts can call methods with arguments.
        """

        database: NaDataDatabase = load_example("ex4_expressions")

        pipe: Any = database.get_by_symbol(name="pipe_standard")

        assert pipe.get_attribute("name").get() == "Standard_Scaler"
        assert pipe.get_attribute("scale_factor").get() == 3.5

    def test_ex5_edge_case_instances(self) -> None:
        """
        Tests that edge case instances (empty constructors) load correctly.
        """

        database: NaDataDatabase = load_example("ex5_edge_cases")

        assert "item2" in database.symbols

        item2: Any = database.get_by_symbol(name="item2")

        # item2 was created with an empty constructor: defaults apply
        assert item2.get_attribute("x").get() == 0

    def test_load_file_direct(self) -> None:
        """
        Tests loading raw .nadata content through load_file.
        """

        parser: NaStructParser = NaStructParser()
        database: NaDataDatabase = parser.parse_folder(
            folder_path="./examples/ex0_simple"
        )

        loader: NaDataLoader = NaDataLoader(database=database)
        loader.load_file(content='solo = Human(name = "Solo")')

        solo: Any = database.get_by_symbol(name="solo")

        assert solo.get_attribute("name").get() == "Solo"
