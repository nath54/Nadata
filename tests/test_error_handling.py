"""
Tests for in-language error handling: raise, try / catch, bare catch,
control flow passing through protected blocks, and runtime errors
being catchable.
"""

# Import Modules
from typing import Any

import pytest

from nadata.core.expressions.scope import RaiseSignal
from nadata.parser.expressions.statement_parser import StatementParser


class TestRaiseParsing:
    """
    Parsing of raise statements and try / catch blocks.
    """

    def test_parse_raise(self) -> None:
        """
        A raise statement parses to a RaiseStmt.
        """

        from nadata.core.expressions.statement import RaiseStmt

        statements = StatementParser.parse_body('raise "boom";')

        assert len(statements) == 1
        assert isinstance(statements[0], RaiseStmt)

    def test_parse_raise_requires_message(self) -> None:
        """
        A bare raise is a syntax error.
        """

        with pytest.raises(SyntaxError):
            StatementParser.parse_body("raise;")

    def test_parse_try_catch(self) -> None:
        """
        A try / catch with a binding parses to a TryCatchStmt.
        """

        from nadata.core.expressions.statement import TryCatchStmt

        statements = StatementParser.parse_body(
            'try { raise "x"; } catch (err) { print(err); }'
        )

        assert len(statements) == 1
        assert isinstance(statements[0], TryCatchStmt)
        assert statements[0].error_name == "err"

    def test_parse_bare_catch(self) -> None:
        """
        A catch without a binding parses with error_name None.
        """

        from nadata.core.expressions.statement import TryCatchStmt

        statements = StatementParser.parse_body(
            'try { print(1); } catch { print(2); }'
        )

        assert len(statements) == 1
        assert isinstance(statements[0], TryCatchStmt)
        assert statements[0].error_name is None

    def test_parse_try_requires_catch(self) -> None:
        """
        A try without a catch handler is a syntax error.
        """

        with pytest.raises(SyntaxError):
            StatementParser.parse_body('try { print(1); }')


class TestRaiseExecution:
    """
    Execution of raise and its catchability.
    """

    def _app(self, schema: str = "") -> Any:
        """
        Creates a fresh Nadata facade with an optional schema.

        Args:
            schema (str): .nastruct source to execute first.

        Returns:
            Any: The Nadata instance.
        """

        from nadata.api import Nadata
        from nadata.core.database.database import NaDataDatabase

        app = Nadata(database=NaDataDatabase())

        if schema:
            app.execute_schema(schema)

        return app

    def test_raise_produces_raise_signal(self) -> None:
        """
        An uncaught raise surfaces as a RaiseSignal.
        """

        app = self._app()

        with pytest.raises(RaiseSignal) as excinfo:
            app.execute_data('raise "boom";')

        assert excinfo.value.message == "boom"

    def test_raise_requires_str_message(self) -> None:
        """
        Raising a non-Str is a TypeError.
        """

        app = self._app()

        with pytest.raises(TypeError):
            app.execute_data("raise 42;")

    def test_catch_user_raise(self) -> None:
        """
        A caught raise binds its message and runs the handler.
        """

        app = self._app()

        # No exception escapes; the handler ran (proven by the raise
        # being swallowed instead of surfacing)
        app.execute_data(
            'flag: Str = "unset";'
            'try { raise "custom failure"; } catch (err) { flag = err; }'
            'print("flag:", flag);'
        )

    def test_catch_runtime_error(self) -> None:
        """
        Runtime errors (index out of range) are catchable.
        """

        app = self._app()

        app.execute_data(
            'try { arr: Array[Str] = ["a"]; x: Str = arr[5]; }'
            " catch (e) { print(\"caught:\", e); }"
        )

    def test_bare_catch(self) -> None:
        """
        A bare catch handles errors without binding them.
        """

        app = self._app()

        app.execute_data(
            'try { y: Int32 = 1 / 0; } catch { print("handled"); }'
        )

    def test_continue_passes_through_try(self) -> None:
        """
        continue inside a try reaches the loop, not the catch handler.
        """

        app = self._app()

        app.execute_data(
            'seen: Str = "";'
            "for (i: Int32 in [1, 2, 3]) {"
            "  try { if (i == 2) { continue; } seen = seen + i; }"
            '  catch (e) { seen = seen + "E"; }'
            "}"
            'print("seen:", seen);'
        )

    def test_return_passes_through_try(self) -> None:
        """
        A return inside a try unwinds to the method caller.
        """

        app = self._app(
            "class Box {\n"
            "    value: Int32 = 0;\n"
            "    @pick() -> Int32 {\n"
            "        try { return self.value; } catch (e) { return 99; }\n"
            "    }\n"
            "}\n"
        )

        app.execute_data('b = Box(); print("pick:", b.pick());')

    def test_error_from_method_call_catchable(self) -> None:
        """
        Errors raised inside method bodies are catchable at call site.
        """

        app = self._app(
            "class Validator {\n"
            "    @check(x: Int32) -> Int32 {\n"
            "        if (x < 0) { raise \"negative\"; }\n"
            "        return x;\n"
            "    }\n"
            "}\n"
        )

        app.execute_data(
            "v = Validator();"
            "ok: Int32 = 0;"
            "try { ok = v.check(-1); } catch (e) { ok = -7; }"
            'print("ok:", ok);'
        )

    def test_uncaught_inside_catch_propagates(self) -> None:
        """
        A raise inside a catch handler propagates.
        """

        app = self._app()

        with pytest.raises(RaiseSignal) as excinfo:
            app.execute_data(
                'try { raise "first"; } catch (e) { raise "second"; }'
            )

        assert excinfo.value.message == "second"
