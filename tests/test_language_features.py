"""
Tests for language features: array/string length(), multi-index tensor
access, derived constants, and method-parameter dim validation.
"""

# Import Modules
from __future__ import annotations
import os
import tempfile
from typing import Any

import pytest

from src.api import Nadata
from src.core.database.database import NaDataDatabase
from src.core.database.persistence.dumper import dump_database
from src.core.database.persistence.loader import load_database


class TestLength:
    """
    Tests the length() member on arrays and strings.
    """

    def _app(self) -> Nadata:
        """
        Builds an app with a probe struct.

        Returns:
            Nadata: The prepared facade.
        """

        app: Nadata = Nadata(database=NaDataDatabase())

        app.execute_schema(schema_text=(
            "class P { n: Int32 = 0; s: Int32 = 0; items: Array[Str] = []; }"
        ))

        return app

    def test_array_length(self) -> None:
        """
        Tests length() on array attributes.
        """

        app: Nadata = self._app()

        app.execute_data(data_text=(
            'p = P(items = ["a", "b", "c"]);\n'
            "p.n = p.items.length();\n"
        ))

        probe: Any = app.query(struct_name="P").first()

        assert probe.attributes["n"].get() == 3

    def test_string_length(self) -> None:
        """
        Tests length() on string literals and attributes.
        """

        app: Nadata = self._app()

        app.execute_data(data_text=(
            'p = P();\n'
            "p.s = \"hello\".length();\n"
        ))

        probe: Any = app.query(struct_name="P").first()

        assert probe.attributes["s"].get() == 5

    def test_empty_array_length(self) -> None:
        """
        Tests length() on an empty array default.
        """

        app: Nadata = self._app()

        app.execute_data(data_text=(
            "p = P();\n"
            "p.n = p.items.length();\n"
        ))

        probe: Any = app.query(struct_name="P").first()

        assert probe.attributes["n"].get() == 0

    def test_length_in_method(self) -> None:
        """
        Tests length() inside a method body.
        """

        app: Nadata = self._app()

        app.execute_schema(schema_text=(
            'class Q { names: Array[Str] = [];\n'
            "  @count() -> Int32 { return self.names.length(); }\n"
            "}"
        ))

        app.execute_data(data_text='q = Q(names = ["x", "y"]);\n')

        assert app.call_method("q", "count") == 2


class TestMultiIndex:
    """
    Tests multi-index tensor reads and writes.
    """

    def _app(self) -> Nadata:
        """
        Builds an app with a 2x3 tensor struct and a probe.

        Returns:
            Nadata: The prepared facade.
        """

        app: Nadata = Nadata(database=NaDataDatabase())

        app.execute_schema(schema_text=(
            "class T { m: Tensor[Float32, 2, 3]; }\n"
            "class P { v: Float32 = 0.0; }\n"
        ))

        app.execute_data(data_text=(
            "t = T(m = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);\n"
            "p = P();\n"
        ))

        return app

    def test_multi_index_read(self) -> None:
        """
        Tests reading an element through a multi-index.
        """

        app: Nadata = self._app()

        app.execute_data(data_text="p.v = t.m[1, 2];\n")

        probe: Any = app.query(struct_name="P").first()

        assert probe.attributes["v"].get() == 6.0

    def test_multi_index_write(self) -> None:
        """
        Tests assigning through a multi-index.
        """

        app: Nadata = self._app()

        app.execute_data(data_text="t.m[0, 1] = 9.0;\n")

        tensor: Any = app.query(struct_name="T").first()

        assert tensor is not None

        value: Any = tensor.attributes["m"]

        assert value.get() == [1.0, 9.0, 3.0, 4.0, 5.0, 6.0]

    def test_multi_index_out_of_bounds(self) -> None:
        """
        Tests that out-of-range multi-indices raise IndexError.
        """

        app: Nadata = self._app()

        with pytest.raises(IndexError):
            app.execute_data(data_text="p.v = t.m[2, 0];\n")

    def test_single_index_still_works(self) -> None:
        """
        Tests that flat single-index access still works on tensors.
        """

        app: Nadata = self._app()

        app.execute_data(data_text="p.v = t.m[4];\n")

        probe: Any = app.query(struct_name="P").first()

        assert probe.attributes["v"].get() == 5.0

    def test_multi_index_requires_tensor(self) -> None:
        """
        Tests that multi-indexing a non-tensor is a type error.
        """

        app: Nadata = self._app()

        with pytest.raises(TypeError):
            app.execute_data(data_text='p.v = t.m["a", "b"];\n')


class TestDerivedConsts:
    """
    Tests const-to-const derived defaults.
    """

    SCHEMA: str = """
    class M {
        const dim: Int32;
        const half: Int32 = self.dim / 2;
        const fourdim: Int32 = self.dim * 4;
        w: Tensor[Float32, half];
    }
    """

    def _app(self) -> Nadata:
        """
        Builds an app with the derived-const schema.

        Returns:
            Nadata: The prepared facade.
        """

        app: Nadata = Nadata(database=NaDataDatabase())

        app.execute_schema(schema_text=self.SCHEMA)

        return app

    def test_derivation(self) -> None:
        """
        Tests that derived consts compute from the bound base const.
        """

        app: Nadata = self._app()

        app.execute_data(data_text="m = M(dim = 4, w = [1.0, 2.0]);\n")

        instance: Any = app.query(struct_name="M").first()

        assert instance.attributes["dim"].get() == 4
        assert instance.attributes["half"].get() == 2
        assert instance.attributes["fourdim"].get() == 16
        assert instance.attributes["w"].concrete_dims == [2]

    def test_per_instance(self) -> None:
        """
        Tests that different instances derive different values.
        """

        app: Nadata = self._app()

        app.execute_data(data_text=(
            "a = M(dim = 4, w = [1.0, 2.0]);\n"
            "b = M(dim = 6, w = [1.0, 2.0, 3.0]);\n"
        ))

        instances: Any = app.query(struct_name="M").to_list()

        assert instances[0].attributes["half"].get() == 2
        assert instances[1].attributes["half"].get() == 3

    def test_direct_set_rejected(self) -> None:
        """
        Tests that derived consts cannot be set explicitly.
        """

        app: Nadata = self._app()

        with pytest.raises(TypeError, match="derived"):
            app.create_instance(struct_name="M", attributes={
                "dim": 4, "half": 3, "w": [1.0, 2.0],
            })

    def test_non_integral_division_rejected(self) -> None:
        """
        Tests that a division with a remainder fails the int type.
        """

        app: Nadata = self._app()

        with pytest.raises(TypeError, match="Expected int"):
            app.create_instance(struct_name="M", attributes={
                "dim": 5, "w": [1.0],
            })

    def test_mutation_rejected(self) -> None:
        """
        Tests that derived consts are immutable like other consts.
        """

        app: Nadata = self._app()

        app.execute_data(data_text="m = M(dim = 4, w = [1.0, 2.0]);\n")

        with pytest.raises(TypeError, match="immutable"):
            app.set_attribute(symbol="m", name="half", value=3)

    def test_round_trip(self) -> None:
        """
        Tests derived const values and expressions across persistence.
        """

        app: Nadata = self._app()

        app.execute_data(data_text="m = M(dim = 4, w = [1.0, 2.0]);\n")

        temp_dir: str = tempfile.mkdtemp()
        storage_path: str = os.path.join(temp_dir, "d.ndb")

        dump_database(database=app.database, storage_path=storage_path)

        reloaded: Any = load_database(storage_path=storage_path)

        table: Any = reloaded.tables_by_struct_name["M"]

        instance: Any = table.instances[0]

        assert instance.attributes["half"].get() == 2
        assert instance.attributes["fourdim"].get() == 16

        struct_attr: Any = table.struct.get_attribute(name="half")

        assert struct_attr.is_const is True
        assert struct_attr.default_expr is not None

    def test_unknown_reference_rejected(self) -> None:
        """
        Tests that derived consts referencing unknown attrs are rejected.
        """

        with pytest.raises(TypeError, match="unknown attribute"):
            Nadata(database=NaDataDatabase()).execute_schema(schema_text=(
                "class X { const bad: Int32 = self.missing + 1; }"
            ))

    def test_non_const_reference_rejected(self) -> None:
        """
        Tests that derived consts referencing mutable attrs are rejected.
        """

        with pytest.raises(TypeError, match="non-const"):
            Nadata(database=NaDataDatabase()).execute_schema(schema_text=(
                "class X { v: Int32 = 0; const bad: Int32 = self.v + 1; }"
            ))


class TestNaming:
    """
    Tests the lower_snake_case rule for attributes, params, and variables.
    """

    def test_uppercase_attribute_rejected(self) -> None:
        """
        Tests that struct attributes cannot start uppercase.
        """

        with pytest.raises(SyntaxError, match="must not start"):
            Nadata(database=NaDataDatabase()).execute_schema(schema_text="class X { Name: Str; }")

    def test_uppercase_const_rejected(self) -> None:
        """
        Tests that consts follow the same rule.
        """

        with pytest.raises(SyntaxError, match="must not start"):
            Nadata(database=NaDataDatabase()).execute_schema(schema_text="class X { const Batch: Int32; }")

    def test_lowercase_underscore_ok(self) -> None:
        """
        Tests that underscore-prefixed names are allowed.
        """

        app: Nadata = Nadata(database=NaDataDatabase())

        app.execute_schema(schema_text="class X { _count: Int32 = 0; }")

        app.execute_data(data_text="x = X(_count = 1);")

        assert app.query(struct_name="X").first() is not None

    def test_uppercase_parameter_rejected(self) -> None:
        """
        Tests that method parameters cannot start uppercase.
        """

        with pytest.raises(SyntaxError, match="must not start"):
            Nadata(database=NaDataDatabase()).execute_schema(schema_text=(
                "class X {\n"
                "  @f(Value: Int32) -> Int32 { return Value; }\n"
                "}"
            ))

    def test_uppercase_local_rejected(self) -> None:
        """
        Tests that typed local declarations cannot start uppercase.
        """

        app: Nadata = Nadata(database=NaDataDatabase())

        app.execute_schema(schema_text="class X { v: Int32 = 0; }")

        with pytest.raises(SyntaxError, match="must not start"):
            app.execute_data(data_text=(
                "x = X();\n"
                "Total: Int32 = 0;\n"
            ))

    def test_uppercase_script_var_rejected(self) -> None:
        """
        Tests that plain script assignments cannot create uppercase names.
        """

        app: Nadata = Nadata(database=NaDataDatabase())

        app.execute_schema(schema_text="class X { v: Int32 = 0; }")

        with pytest.raises(SyntaxError, match="must not start"):
            app.execute_data(data_text=(
                "x = X();\n"
                "Total = 5;\n"
            ))

    def test_class_names_still_uppercase(self) -> None:
        """
        Tests that struct names keep their UpperCamelCase convention.
        """

        app: Nadata = Nadata(database=NaDataDatabase())

        app.execute_schema(schema_text="class MyStruct { v: Int32 = 0; }")

        app.execute_data(data_text="s = MyStruct();\n")

        assert app.query(struct_name="MyStruct").first() is not None


class TestMethodDims:
    """
    Tests symbolic tensor dimensions on method parameters.
    """

    def _app(self) -> Nadata:
        """
        Builds an app with a scaling method over [batch, dim] tensors.

        Returns:
            Nadata: The prepared facade.
        """

        app: Nadata = Nadata(database=NaDataDatabase())

        app.execute_schema(schema_text="""
        class n {
            const batch: Int32;
            const dim: Int32 = 2;
            data: Tensor[Float32, batch, dim];

            @total(x: Tensor[Float32, batch, dim]) -> Float32 {
                t: Float32 = 0.0;
                for (v: Float32 in x) { t = t + v; }
                return t;
            }
        }
        """)

        return app

    def test_matching_shape_accepted(self) -> None:
        """
        Tests that an argument matching the receiver's dims is accepted.
        """

        app: Nadata = self._app()

        app.execute_data(data_text=(
            "n = n(batch = 2, data = [[1.0, 2.0], [3.0, 4.0]]);\n"
        ))

        result: Any = app.call_method(
            "n", "total", [[5.0, 5.0], [5.0, 5.0]]
        )

        assert result == 20.0

    def test_mismatched_shape_rejected(self) -> None:
        """
        Tests that a wrong-length argument is rejected.
        """

        app: Nadata = self._app()

        app.execute_data(data_text=(
            "n = n(batch = 2, data = [[1.0, 2.0], [3.0, 4.0]]);\n"
        ))

        with pytest.raises(TypeError, match="must have 4 elements"):
            app.call_method("n", "total", [[1.0, 2.0]])

    def test_unknown_dim_name_rejected(self) -> None:
        """
        Tests that param dims must name struct consts at resolution.
        """

        with pytest.raises(TypeError, match="not a basic integer const"):
            Nadata(database=NaDataDatabase()).execute_schema(schema_text=(
                "class B { const dim: Int32 = 2;\n"
                "  @f(x: Tensor[Float32, Nope]) -> Float32 { return 0.0; }\n"
                "}"
            ))

    def test_standalone_function_rejected(self) -> None:
        """
        Tests that standalone functions cannot take symbolic tensors.
        """

        with pytest.raises(TypeError, match="only struct methods"):
            Nadata(database=NaDataDatabase()).execute_schema(schema_text=(
                "fn g(x: Tensor[Float32, batch]) -> Float32 { return 0.0; }"
            ))

    def test_symbolic_return_rejected(self) -> None:
        """
        Tests that symbolic tensor return types are rejected.
        """

        with pytest.raises(TypeError, match="symbolic tensor return type"):
            Nadata(database=NaDataDatabase()).execute_schema(schema_text=(
                "class B { const dim: Int32 = 2;\n"
                "  @f() -> Tensor[Float32, dim] { return [1.0, 2.0]; }\n"
                "}"
            ))
