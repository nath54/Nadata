"""
Tests for loops: for-in, C-style for, while, break/continue, fuel limits,
loop variable freshness, dict iteration helpers, and loop persistence.
"""

# Import Modules
from typing import Any, Optional

import pytest

from nadata.api import Nadata
from nadata.core.expressions.scope import LoopLimitError
from nadata.parser.expressions.statement_parser import StatementParser
from nadata.parser.nastruct_parser.nastruct_parser import NaStructParser


class TestLoopParsing:
    """
    Tests the statement parser on loop syntax.
    """

    def test_parse_for_in(self) -> None:
        """
        Tests that a for-in loop parses with its typed variable.
        """

        statements: list[Any] = StatementParser.parse_body(
            "for(x: Int32 in items){ total = total + x; }"
        )

        loop: Any = statements[0]

        assert loop.var_name == "x"
        assert loop.var_type_name == "Int32"

    def test_parse_c_style_for(self) -> None:
        """
        Tests that a C-style loop parses inits, condition, and updates.
        """

        statements: list[Any] = StatementParser.parse_body(
            "for(i: Int32 = 0, j: Int32 = 10 :: i < j :: i++, j--)"
            "{ total = i; }"
        )

        loop: Any = statements[0]

        assert len(loop.init_statements) == 2
        assert len(loop.update_statements) == 2

    def test_parse_while(self) -> None:
        """
        Tests that a while loop parses.
        """

        statements: list[Any] = StatementParser.parse_body(
            "while(count < 10){ count++; }"
        )

        assert statements[0].condition is not None

    def test_break_continue_outside_loop_rejected(self) -> None:
        """
        Tests that break / continue outside loops are syntax errors.
        """

        with pytest.raises(SyntaxError, match="outside of a loop"):
            StatementParser.parse_body("break;")

        with pytest.raises(SyntaxError, match="outside of a loop"):
            StatementParser.parse_body("if(a){ continue; }")

    def test_c_style_init_requires_initializer(self) -> None:
        """
        Tests that a C-style init declaration without '=' is rejected.
        """

        with pytest.raises(SyntaxError, match="requires an initializer"):
            StatementParser.parse_body("for(i: Int32 :: i < 3 :: i++){ }")


class TestLoopExecution:
    """
    Tests loop execution through the data loader.
    """

    def setup_method(self) -> None:
        """
        Loads the example project for each test.
        """

        self.app: Nadata = Nadata.load(path="./examples/ex0_simple")

    def test_for_in_over_array_literal(self) -> None:
        """
        Tests iterating a typed variable over an array literal.
        """

        self.app.execute_data(
            'total: Int32 = 0;'
            'for(x: Int32 in [1, 2, 3]){ total = total + x; }'
            'print("total:", total);'
        )

    def test_for_in_over_instances(self) -> None:
        """
        Tests iterating instance references and calling methods on them.
        """

        self.app.execute_data(
            'names: Str = "";'
            "for(h: Human in [tom, jean]){ names = names + h.name; }"
            'print("names:", names);'
        )

    def test_for_in_rejects_wrong_element_type(self) -> None:
        """
        Tests that a mismatching element violates the declared loop type.
        """

        with pytest.raises(TypeError):
            self.app.execute_data("for(x: Int32 in [\"a\"]){ }")

    def test_for_in_rejects_non_array(self) -> None:
        """
        Tests that iterating a non-array raises TypeError.
        """

        with pytest.raises(TypeError, match="Cannot iterate"):
            self.app.execute_data("for(x: Int32 in 5){ }")

    def test_c_style_loop(self) -> None:
        """
        Tests the C-style loop with compound update operators.
        """

        self.app.execute_data(
            "total: Int32 = 0;"
            "for(i: Int32 = 0 :: i < 10 :: i++){ total += i; }"
            'print("total:", total);'
        )

    def test_while_loop_with_break_continue(self) -> None:
        """
        Tests while loops with break and continue control flow.
        """

        self.app.execute_data(
            "n: Int32 = 0;"
            "hits: Int32 = 0;"
            "while(true){"
            "  n++;"
            "  if(n > 5){ break; }"
            "  if(n == 2){ continue; }"
            "  hits++;"
            "}"
            'print("hits:", hits);'
        )

    def test_break_exits_innermost_loop_only(self) -> None:
        """
        Tests that break inside a nested loop leaves the outer one running.
        """

        self.app.execute_data(
            "outer: Int32 = 0;"
            "for(i: Int32 = 0 :: i < 3 :: i++){"
            "  for(j: Int32 = 0 :: j < 10 :: j++){ break; }"
            "  outer++;"
            "}"
            'print("outer:", outer);'
        )

    def test_loop_variable_scoped_to_body(self) -> None:
        """
        Tests that the loop variable does not leak into the outer scope.
        """

        with pytest.raises(NameError):
            self.app.execute_data("for(v: Int32 in [1]){ } print(\"v:\", v);")

    def test_sequential_loops_may_reuse_names(self) -> None:
        """
        Tests that two sequential loops can use the same variable name.
        """

        self.app.execute_data(
            "for(i: Int32 = 0 :: i < 2 :: i++){ }"
            "for(i: Int32 = 0 :: i < 3 :: i++){ }"
        )

    def test_compound_assignment_statement(self) -> None:
        """
        Tests compound assignments and increments as standalone statements.
        """

        self.app.execute_data(
            "x: Int32 = 10;"
            "x += 5;"
            "x *= 2;"
            "x -= 5;"
            's: Str = "a";'
            's += "b";'
            'print("x:", x, "s:", s);'
        )

    def test_dict_keys_and_values(self) -> None:
        """
        Tests iterating dict keys and values with typed loop variables.
        """

        self.app.execute_data(
            'counts: Dict[Str, Int32] = {"a": 1, "b": 2};'
            'ksum: Str = "";'
            "for(k: Str in counts.keys()){ ksum = ksum + k; }"
            "vsum: Int32 = 0;"
            "for(v: Int32 in counts.values()){ vsum = vsum + v; }"
            'print("keys:", ksum, "values:", vsum);'
        )


class TestFuelLimits:
    """
    Tests the loop fuel counter and set_loop_limit.
    """

    def setup_method(self) -> None:
        """
        Loads the example project for each test.
        """

        self.app: Nadata = Nadata.load(path="./examples/ex0_simple")

    def test_default_limit_raises(self) -> None:
        """
        Tests that a loop exceeding 10000 iterations raises LoopLimitError.
        """

        with pytest.raises(LoopLimitError, match="fuel limit"):
            self.app.execute_data("for(i: Int32 = 0 :: true :: i++){ }")

    def test_loop_under_limit_runs(self) -> None:
        """
        Tests that loops under the fuel limit complete normally.
        """

        self.app.execute_data("for(i: Int32 = 0 :: i < 100 :: i++){ }")

    def test_native_set_loop_limit(self) -> None:
        """
        Tests the set_loop_limit native called from script code.
        """

        self.app.execute_data(
            "set_loop_limit(5);"
            "for(i: Int32 = 0 :: i < 3 :: i++){ }"
        )

        with pytest.raises(LoopLimitError):
            self.app.execute_data("for(i: Int32 = 0 :: i < 10 :: i++){ }")

    def test_facade_set_loop_limit(self) -> None:
        """
        Tests the Python facade set_loop_limit and its validation.
        """

        self.app.set_loop_limit(100)

        with pytest.raises(LoopLimitError):
            self.app.execute_data("for(i: Int32 = 0 :: i < 200 :: i++){ }")

        with pytest.raises(ValueError):
            self.app.set_loop_limit(0)

        with pytest.raises(TypeError):
            self.app.set_loop_limit("many")  # type: ignore[arg-type]

    def test_limit_persists_through_save(self, tmp_path: Any) -> None:
        """
        Tests that the loop limit is persisted in the database settings.
        """

        storage_path: str = str(tmp_path / "fuel.ndb")

        self.app.set_loop_limit(50)
        self.app.dump(storage_path=storage_path)

        reloaded: Nadata = Nadata.open(storage_path=storage_path)

        assert reloaded.database.settings["loop_limit"] == 50

        with pytest.raises(LoopLimitError):
            reloaded.execute_data("for(i: Int32 = 0 :: i < 100 :: i++){ }")

        reloaded.set_loop_limit(500)
        reloaded.save_changes(storage_path=storage_path)

        reloaded2: Nadata = Nadata.open(storage_path=storage_path)

        assert reloaded2.database.settings["loop_limit"] == 500

        reloaded2.execute_data("for(i: Int32 = 0 :: i < 200 :: i++){ }")


class TestLoopResolution:
    """
    Tests static checks on loop variables during schema resolution.
    """

    def _resolve(self, schema_text: str) -> None:
        """
        Parses and resolves a schema text.

        Args:
            schema_text (str): The .nastruct source to resolve.
        """

        parser: NaStructParser = NaStructParser()

        parser.parse_file(content=schema_text)
        parser.resolve()

    def test_loop_variable_shadowing_param_rejected(self) -> None:
        """
        Tests that a loop variable shadowing a parameter is a syntax error.
        """

        with pytest.raises(SyntaxError, match="shadows an existing variable"):
            self._resolve(
                "class A {\n"
                "    @m(n: Int32) -> Int32 {\n"
                "        for(n: Int32 = 0 :: n < 3 :: n++){ }\n"
                "        return 0;\n"
                "    }\n"
                "}\n"
            )

    def test_loop_variable_shadowing_attribute_rejected(self) -> None:
        """
        Tests that a loop variable shadowing an attribute is a syntax error.
        """

        with pytest.raises(SyntaxError, match="shadows an attribute"):
            self._resolve(
                "class A {\n"
                "    count: Int32;\n"
                "\n"
                "    @m() -> Int32 {\n"
                "        for(count: Int32 = 0 :: count < 3 :: count++){ }\n"
                "        return 0;\n"
                "    }\n"
                "}\n"
            )

    def test_loop_variable_shadowing_local_rejected(self) -> None:
        """
        Tests that a loop variable shadowing an earlier local is rejected.
        """

        with pytest.raises(SyntaxError, match="shadows an existing variable"):
            self._resolve(
                "class A {\n"
                "    @m() -> Int32 {\n"
                "        total: Int32 = 0;\n"
                "        for(total: Int32 = 0 :: total < 3 :: total++){ }\n"
                "        return total;\n"
                "    }\n"
                "}\n"
            )

    def test_loop_variable_shadowing_in_for_in_rejected(self) -> None:
        """
        Tests that for-in variables obey the same freshness rule.
        """

        with pytest.raises(SyntaxError, match="shadows an existing variable"):
            self._resolve(
                "class A {\n"
                "    @m(items: Array[Int32]) -> Int32 {\n"
                "        item: Int32 = 0;\n"
                "        for(item: Int32 in items){ }\n"
                "        return item;\n"
                "    }\n"
                "}\n"
            )

    def test_loop_variable_in_body_scope(self) -> None:
        """
        Tests that the loop variable is visible inside its own body.
        """

        self._resolve(
            "class A {\n"
            "    @m(items: Array[Int32]) -> Int32 {\n"
            "        total: Int32 = 0;\n"
            "        for(item: Int32 in items){ total = total + item; }\n"
            "        return total;\n"
            "    }\n"
            "}\n"
        )


class TestLoopInMethods:
    """
    Tests loops inside method bodies, including persistence.
    """

    def test_method_loop_executes(self) -> None:
        """
        Tests calling a method whose body contains a loop.
        """

        app: Nadata = Nadata.load(path="./examples/ex0_simple")

        app.execute_schema(
            "class Summer {\n"
            "    @sum_upto(n: Int32) -> Int32 {\n"
            "        total: Int32 = 0;\n"
            "        for(i: Int32 = 0 :: i < n :: i++){ total += i; }\n"
            "        return total;\n"
            "    }\n"
            "}\n"
        )

        app.execute_data("s = Summer()")

        assert app.call_method("s", "sum_upto", 10) == 45

    def test_method_loop_survives_persistence(self, tmp_path: Any) -> None:
        """
        Tests that loop ASTs round-trip through dump and load.
        """

        storage_path: str = str(tmp_path / "loops.ndb")

        app: Nadata = Nadata.load(path="./examples/ex0_simple")

        app.execute_schema(
            "class Summer {\n"
            "    @sum_upto(n: Int32) -> Int32 {\n"
            "        total: Int32 = 0;\n"
            "        for(i: Int32 = 0 :: i < n :: i++){ total += i; }\n"
            "        return total;\n"
            "    }\n"
            "}\n"
        )

        app.execute_data("s = Summer()")
        app.dump(storage_path=storage_path)

        reloaded: Nadata = Nadata.open(storage_path=storage_path)

        assert reloaded.call_method("s", "sum_upto", 5) == 10


class TestConsoleMultiLine:
    """
    Tests the console buffering of multi-line loop requests.
    """

    def test_execute_line_multi_line_loop(self, capsys: Any) -> None:
        """
        Tests that a multi-line request with a loop body executes.
        """

        from nadata.console import NadataConsole

        app: Nadata = Nadata.load(path="./examples/ex0_simple")
        console: NadataConsole = NadataConsole(app=app)

        output: Optional[str] = console.execute_line(
            "total: Int32 = 0;\n"
            "for(i: Int32 = 0 :: i < 4 :: i++){ total += i; }\n"
            'print("total:", total);\n'
        )

        assert output is None
        assert "total: 6" in capsys.readouterr().out

    def test_run_buffer_until_blank_line(self, capsys: Any, monkeypatch: Any) -> None:
        """
        Tests that run() executes the buffered request on a blank line.
        """

        from nadata.console import NadataConsole

        app: Nadata = Nadata.load(path="./examples/ex0_simple")
        console: NadataConsole = NadataConsole(app=app)

        lines: list[str] = [
            "total: Int32 = 0;",
            "for(i: Int32 = 0 :: i < 3 :: i++){ total += i; }",
            'print("total:", total);',
            "",
            ":quit",
        ]

        monkeypatch.setattr(
            "builtins.input",
            lambda prompt: lines.pop(0) if lines else ":quit",
        )

        exit_code: int = console.run()

        assert exit_code == 0
        assert "total: 3" in capsys.readouterr().out
