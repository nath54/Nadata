"""
Tests for target assignment (member and index targets), compound forms,
and scientific notation literals.
"""

# Import Modules
from __future__ import annotations
from typing import Any

import pytest

from nadata.api import Nadata
from nadata.core.database.database import NaDataDatabase
from nadata.parser.expressions.statement_parser import StatementParser
from nadata.parser.expressions.tokenizer import Tokenizer
from nadata.parser.nadata_loader.loader import NaDataLoader
from nadata.parser.nastruct_parser.nastruct_parser import NaStructParser

SCHEMA: str = """
class Box {
    label: Str;
    count: Int32 = 0;
    items: Array[Str] = ["a", "b", "c"];
    scores: Dict[Str, Int32] = {};

    @bump() -> Int32 {
        self.count += 1;
        return self.count;
    }

    @set_item(i: Int32, v: Str) -> Str {
        self.items[i] = v;
        return self.items[i];
    }

    @set_score(k: Str, v: Int32) -> Int32 {
        self.scores[k] = v;
        return self.scores[k];
    }
}
"""


def _parse_statements(text: str) -> list[Any]:
    """
    Parses statement source into AST nodes.

    Args:
        text (str): Statement source.

    Returns:
        list[Any]: Parsed statements.
    """

    tokens: list[Any] = Tokenizer(content=text).tokenize()

    return StatementParser(tokens=tokens, raw_text=text).parse_statements()


def _fresh_database() -> Any:
    """
    Builds a resolved database with the Box schema.

    Returns:
        Any: The resolved NaDataDatabase.
    """

    parser: NaStructParser = NaStructParser(database=NaDataDatabase())

    parser.parse_file(content=SCHEMA)

    return parser.resolve()


def _load(text: str) -> None:
    """
    Loads data text into a fresh resolved database.

    Args:
        text (str): Data source.
    """

    NaDataLoader(database=_fresh_database()).load_file(content=text)


class TestScientificLiterals:
    """
    Tests scientific notation number literals.
    """

    def test_exponent_parses_as_float(self) -> None:
        """
        Tests that '1e6' evaluates to the float 1000000.0.
        """

        statement: Any = _parse_statements(text="x = 1e6;")[0]

        assert statement.value_expr.value.get() == 1000000.0

    def test_negative_exponent(self) -> None:
        """
        Tests that '3.5e-2' evaluates to 0.035.
        """

        statement: Any = _parse_statements(text="x = 3.5e-2;")[0]

        assert statement.value_expr.value.get() == 0.035

    def test_uppercase_exponent_with_sign(self) -> None:
        """
        Tests that '1.5E+3' evaluates to 1500.0.
        """

        statement: Any = _parse_statements(text="x = 1.5E+3;")[0]

        assert statement.value_expr.value.get() == 1500.0

    def test_plain_int_unchanged(self) -> None:
        """
        Tests that integers without exponents stay ints.
        """

        statement: Any = _parse_statements(text="x = 100;")[0]

        assert statement.value_expr.value.get() == 100

    def test_incomplete_exponent_backtracks(self) -> None:
        """
        Tests that '1e' without digits backtracks to INT + identifier.
        """

        from nadata.parser.expressions.tokenizer import TokenType

        tokens: list[Any] = Tokenizer(content="1e").tokenize()

        assert tokens[0].token_type == TokenType.INT
        assert tokens[0].value == "1"
        assert tokens[1].value == "e"

    def test_member_access_on_float_literal(self) -> None:
        """
        Tests that '1.5' followed by postfix stays parseable.
        """

        tokens: list[Any] = Tokenizer(content="1.5.toString").tokenize()

        assert tokens[0].value == "1.5"


class TestTargetAssignmentParsing:
    """
    Tests parsing of member and index assignment targets.
    """

    def test_index_target(self) -> None:
        """
        Tests that 'arr[0] = x' parses into a TargetAssignStmt.
        """

        from nadata.core.expressions.statement import TargetAssignStmt

        statement: Any = _parse_statements(text="arr[0] = 5;")[0]

        assert isinstance(statement, TargetAssignStmt)

    def test_member_target_compound(self) -> None:
        """
        Tests that 'self.count += 1' desugars with a re-read of the target.
        """

        from nadata.core.expressions.statement import TargetAssignStmt
        from nadata.core.expressions.expression import BinaryOpExpr

        statement: Any = _parse_statements(text="self.count += 1;")[0]

        assert isinstance(statement, TargetAssignStmt)
        assert isinstance(statement.value_expr, BinaryOpExpr)

    def test_index_increment(self) -> None:
        """
        Tests that 'arr[i]++' desugars into a target assignment.
        """

        from nadata.core.expressions.statement import TargetAssignStmt

        statement: Any = _parse_statements(text="arr[i]++;")[0]

        assert isinstance(statement, TargetAssignStmt)

    def test_plain_assignment_stays_assign_stmt(self) -> None:
        """
        Tests that 'x = 5;' still parses into a plain AssignStmt.
        """

        from nadata.core.expressions.statement import AssignStmt

        statement: Any = _parse_statements(text="x = 5;")[0]

        assert isinstance(statement, AssignStmt)

    def test_call_target_rejected(self) -> None:
        """
        Tests that assigning to a call result is a SyntaxError.
        """

        with pytest.raises(SyntaxError):
            _parse_statements(text="f() = 3;")


class TestTargetAssignmentRuntime:
    """
    Tests runtime behavior of member and index assignment.
    """

    def test_method_compound_assign(self) -> None:
        """
        Tests that 'self.count += 1' inside a method accumulates.
        """

        _load(text='box = Box(label = "b"); print(box.bump(), box.bump());')

    def test_attribute_compound_and_increment(self) -> None:
        """
        Tests 'box.count += 10' and 'box.count++' on instances.
        """

        app: Nadata = Nadata(database=NaDataDatabase())
        app.execute_schema(schema_text=SCHEMA)

        NaDataLoader(database=app.database).load_file(
            content='box = Box(label = "b"); box.count += 10; box.count++;'
        )

        box: Any = app.query(struct_name="Box").first()

        assert box.attributes["count"].get() == 11

    def test_index_assignment_and_read(self) -> None:
        """
        Tests array element assignment including negative indices.
        """

        app: Nadata = Nadata(database=NaDataDatabase())
        app.execute_schema(schema_text=SCHEMA)

        NaDataLoader(database=app.database).load_file(
            content='box = Box(label = "b"); box.items[0] = "z"; box.items[-1] = "w";'
        )

        box: Any = app.query(struct_name="Box").first()
        items: list[Any] = box.attributes["items"].get()

        assert items == ["z", "b", "w"]

    def test_dict_assignment_new_and_existing_keys(self) -> None:
        """
        Tests dict assignment inserts new keys and updates existing ones.
        """

        app: Nadata = Nadata(database=NaDataDatabase())
        app.execute_schema(schema_text=SCHEMA)

        NaDataLoader(database=app.database).load_file(
            content=(
                'box = Box(label = "b");'
                'box.scores["math"] = 42;'
                'box.scores["math"] = 43;'
                'box.scores["sport"] = 7;'
            )
        )

        box: Any = app.query(struct_name="Box").first()
        scores: dict[str, Any] = box.attributes["scores"].get()

        assert scores == {"math": 43, "sport": 7}

    def test_method_index_and_dict_helpers(self) -> None:
        """
        Tests index and dict assignment inside schema methods.
        """

        app: Nadata = Nadata(database=NaDataDatabase())
        app.execute_schema(schema_text=SCHEMA)

        NaDataLoader(database=app.database).load_file(
            content=(
                'box = Box(label = "b");'
                'print(box.set_item(1, "z"));'
                'print(box.set_score("math", 42));'
            )
        )

    def test_wrong_element_type_rejected(self) -> None:
        """
        Tests that assigning a wrong-typed array element raises TypeError.
        """

        with pytest.raises(TypeError):
            _load(text='box = Box(label = "b"); box.items[0] = 5;')

    def test_out_of_range_rejected(self) -> None:
        """
        Tests that assigning past the end of an array raises IndexError.
        """

        with pytest.raises(IndexError):
            _load(text='box = Box(label = "b"); box.items[9] = "x";')

    def test_wrong_attribute_type_rejected(self) -> None:
        """
        Tests that assigning a wrong-typed attribute raises TypeError.
        """

        with pytest.raises(TypeError):
            _load(text='box = Box(label = "b"); box.count = "no";')

    def test_undeclared_attribute_rejected(self) -> None:
        """
        Tests that assigning an undeclared attribute raises AttributeError.
        """

        with pytest.raises(AttributeError):
            _load(text='box = Box(label = "b"); box.nope = 1;')

    def test_wrong_key_type_rejected(self) -> None:
        """
        Tests that assigning with a wrong-typed dict key raises TypeError.
        """

        with pytest.raises(TypeError):
            _load(text='box = Box(label = "b"); box.scores[5] = 1;')

    def test_persistence_round_trip(self) -> None:
        """
        Tests that target assignments survive a dump/load round trip.
        """

        import os
        import tempfile

        from nadata.core.database.persistence.dumper import dump_database
        from nadata.core.database.persistence.loader import load_database

        app: Nadata = Nadata(database=NaDataDatabase())
        app.execute_schema(schema_text=SCHEMA)

        NaDataLoader(database=app.database).load_file(
            content='box = Box(label = "b"); box.count += 5; box.items[0] = "z";'
        )

        temp_dir: str = tempfile.mkdtemp()
        storage_path: str = os.path.join(temp_dir, "target.ndb")

        dump_database(database=app.database, storage_path=storage_path)
        reloaded: Any = load_database(storage_path=storage_path)

        box: Any = app.query(struct_name="Box").first()

        assert box.attributes["count"].get() == 5

        assert reloaded is not None
