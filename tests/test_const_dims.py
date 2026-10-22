"""
Tests for const attributes and constant-bound symbolic tensor dimensions.

Covers the `const` declaration modifier (constructor-only binding,
immutable afterwards), symbolic tensor dims referencing const integer
attributes, constructor-time shape validation, default-bound dims,
schema resolution errors, and persistence round trips.
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
from src.core.types.types import TypeTensor
from src.parser.expressions.type_parser import TypeParser
from src.parser.nastruct_parser.resolver import TypeResolver

SCHEMA: str = """
class Attention {
    const batch: Int32;
    const dim: Int32 = 8;
    name: Str;
    q: Tensor[Float32, batch, dim];
    k: Tensor[Float32, batch, dim];
}
"""

Q_PAYLOAD: str = "[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]]"


class TestResolution:
    """
    Tests schema-level resolution of const attributes and dim names.
    """

    def test_const_and_symbolic_resolve(self) -> None:
        """
        Tests that a valid const/dim schema resolves cleanly.
        """

        app: Nadata = Nadata(database=NaDataDatabase())

        app.execute_schema(schema_text=SCHEMA)

        struct: Any = app.database.tables_by_struct_name["Attention"].struct

        assert struct.get_attribute(name="batch").is_const is True
        assert struct.get_attribute(name="name").is_const is False

        q_type: Any = struct.get_attribute(name="q").attr_type

        assert isinstance(q_type, TypeTensor)
        assert q_type.dims == ["batch", "dim"]
        assert q_type.is_symbolic is True
        assert q_type.symbolic_dims() == ["batch", "dim"]
        assert q_type.length() is None

    def test_dim_must_exist(self) -> None:
        """
        Tests that an unknown dim name is a resolution error.
        """

        with pytest.raises(TypeError, match="does not match any attribute"):
            Nadata(database=NaDataDatabase()).execute_schema(schema_text=(
                "class X { q: Tensor[Float32, missing]; }"
            ))

    def test_dim_must_be_const(self) -> None:
        """
        Tests that dims referencing non-const attributes are rejected.
        """

        with pytest.raises(TypeError, match="must reference a 'const'"):
            Nadata(database=NaDataDatabase()).execute_schema(schema_text=(
                "class X { batch: Int32; q: Tensor[Float32, batch]; }"
            ))

    def test_dim_must_be_integer_const(self) -> None:
        """
        Tests that dims referencing non-integer consts are rejected.
        """

        with pytest.raises(TypeError, match="basic integer"):
            Nadata(database=NaDataDatabase()).execute_schema(schema_text=(
                "class X { const name: Str; q: Tensor[Float32, name]; }"
            ))

    def test_dim_must_not_be_generic(self) -> None:
        """
        Tests that qualified type children are rejected as dims.
        """

        with pytest.raises(TypeError, match="constant names"):
            TypeResolver().resolve(
                tmp_type=TypeParser.parse_text(text="Tensor[Float32, Array[Int32]]")
            )

    def test_default_length_validated(self) -> None:
        """
        Tests that a symbolic tensor default is length-checked against the
        const defaults at resolution time.
        """

        with pytest.raises(TypeError, match="requires 4"):
            Nadata(database=NaDataDatabase()).execute_schema(schema_text=(
                "class X { const n: Int32 = 4; q: Tensor[Float32, n] = [1.0]; }"
            ))


class TestConstruction:
    """
    Tests instance construction: binding, validation, and ordering.
    """

    def _app(self) -> Nadata:
        """
        Builds an app with the Attention schema.

        Returns:
            Nadata: The prepared facade.
        """

        app: Nadata = Nadata(database=NaDataDatabase())

        app.execute_schema(schema_text=SCHEMA)

        return app

    def test_binding_and_default(self) -> None:
        """
        Tests that explicit and default consts both bind and validate.
        """

        app: Nadata = self._app()

        app.execute_data(data_text=(
            f"a = Attention(name = \"first\", batch = 2, q = {Q_PAYLOAD});\n"
        ))

        instance: Any = app.query(struct_name="Attention").first()

        assert instance.attributes["batch"].get() == 2
        assert instance.attributes["dim"].get() == 8
        assert instance.attributes["q"].concrete_dims == [2, 8]
        assert len(instance.attributes["q"].get()) == 16

    def test_const_order_independent(self) -> None:
        """
        Tests that a tensor listed before its consts still validates.
        """

        app: Nadata = self._app()

        app.execute_data(data_text=(
            f"a = Attention(q = {Q_PAYLOAD}, batch = 2, name = \"x\");\n"
        ))

        instance: Any = app.query(struct_name="Attention").first()

        assert instance.attributes["q"].concrete_dims == [2, 8]

    def test_shape_mismatch_rejected(self) -> None:
        """
        Tests that a payload not matching the bound dims is rejected.
        """

        app: Nadata = self._app()

        with pytest.raises(TypeError, match="requires 16"):
            app.create_instance(struct_name="Attention", attributes={
                "name": "bad", "batch": 2, "q": [[1.0]],
            })

    def test_unbound_dim_rejected(self) -> None:
        """
        Tests that writing a tensor with an unbound dim is rejected.
        """

        app: Nadata = self._app()

        with pytest.raises(TypeError, match="'batch' is not bound"):
            app.create_instance(struct_name="Attention", attributes={
                "name": "bad", "q": [[1.0, 2.0]],
            })

    def test_literal_and_symbolic_mixed(self) -> None:
        """
        Tests that mixed literal/symbolic shapes enforce both parts.
        """

        app: Nadata = self._app()

        app.execute_schema(schema_text=(
            "class M { const n: Int32; w: Tensor[Float32, 3, n]; }"
        ))

        app.create_instance(struct_name="M", attributes={"n": 2, "w": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})

        with pytest.raises(TypeError, match="requires 6"):
            app.create_instance(struct_name="M", attributes={"n": 2, "w": [1.0]})

    def test_instances_differ_in_dims(self) -> None:
        """
        Tests that two instances may bind different extents.
        """

        app: Nadata = self._app()

        app.execute_data(data_text=(
            f"a = Attention(name = \"a\", batch = 2, q = {Q_PAYLOAD});\n"
            "b = Attention(name = \"b\", batch = 3, dim = 2, "
            "q = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);\n"
        ))

        instances: Any = app.query(struct_name="Attention").to_list()

        assert instances[0].attributes["batch"].get() == 2
        assert instances[1].attributes["batch"].get() == 3
        assert instances[1].attributes["q"].concrete_dims == [3, 2]


class TestImmutability:
    """
    Tests that consts reject mutation after construction.
    """

    def _bound(self) -> Nadata:
        """
        Builds an app with one bound Attention instance.

        Returns:
            Nadata: The prepared facade.
        """

        app: Nadata = Nadata(database=NaDataDatabase())

        app.execute_schema(schema_text=SCHEMA)

        app.execute_data(data_text=(
            f"a = Attention(name = \"first\", batch = 2, q = {Q_PAYLOAD});\n"
        ))

        return app

    def test_api_set_attribute_rejected(self) -> None:
        """
        Tests that the facade set_attribute path rejects consts.
        """

        app: Nadata = self._bound()

        with pytest.raises(TypeError, match="immutable"):
            app.set_attribute(symbol="a", name="batch", value=5)

    def test_language_assignment_rejected(self) -> None:
        """
        Tests that target assignment in .nadata scripts is rejected.
        """

        app: Nadata = self._bound()

        with pytest.raises(TypeError, match="immutable"):
            app.execute_data(data_text="a.batch = 7;\n")

    def test_method_body_assignment_rejected(self) -> None:
        """
        Tests that self.const assignment inside a method is rejected.
        """

        app: Nadata = self._bound()

        app.execute_schema(schema_text=(
            "class M { const n: Int32; v: Int32 = 0;\n"
            "  @touch() -> Int32 { self.n = 5; return self.n; }\n"
            "}"
        ))

        app.execute_data(data_text="m = M(n = 1);\n")

        with pytest.raises(TypeError, match="immutable"):
            app.call_method(symbol="m", name="touch")

    def test_non_const_still_mutable(self) -> None:
        """
        Tests that regular attributes remain mutable.
        """

        app: Nadata = self._bound()

        app.set_attribute(symbol="a", name="name", value="renamed")

        assert app.get(symbol="a").attributes["name"].get() == "renamed"

    def test_set_at_unaffected(self) -> None:
        """
        Tests that single-cell tensor writes need no dim logic.
        """

        app: Nadata = self._bound()

        app.execute_data(data_text="a.q.set_at(1, 3, 42.0);\n")

        instance: Any = app.query(struct_name="Attention").first()

        flat: Any = instance.attributes["q"].get()

        assert flat[11] == 42.0


class TestPersistence:
    """
    Tests consts and symbolic tensors across the binary format.
    """

    def test_round_trip(self) -> None:
        """
        Tests values, tensors, and const-ness across dump/load.
        """

        app: Nadata = Nadata(database=NaDataDatabase())

        app.execute_schema(schema_text=SCHEMA)

        app.execute_data(data_text=(
            f"a = Attention(name = \"first\", batch = 2, q = {Q_PAYLOAD});\n"
            "a.k = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],"
            " [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]];\n"
        ))

        temp_dir: str = tempfile.mkdtemp()
        storage_path: str = os.path.join(temp_dir, "c.ndb")

        dump_database(database=app.database, storage_path=storage_path)

        reloaded: Any = load_database(storage_path=storage_path)

        table: Any = reloaded.tables_by_struct_name["Attention"]

        instance: Any = table.instances[0]

        assert instance.attributes["batch"].get() == 2
        assert instance.attributes["dim"].get() == 8
        assert instance.attributes["q"].get() == [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
            8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
        ]
        assert instance.attributes["q"].concrete_dims == [2, 8]
        assert table.struct.get_attribute(name="batch").is_const is True

        # Post-reload immutability still enforced
        with pytest.raises(TypeError, match="immutable"):
            instance.set_attribute(name="batch", value=9)

    def test_default_tensor_stamped(self) -> None:
        """
        Tests that a symbolic tensor default materializes with extents.
        """

        app: Nadata = Nadata(database=NaDataDatabase())

        app.execute_schema(schema_text=(
            "class W { const n: Int32 = 2; w: Tensor[Float32, n] = [1.0, 2.0]; }"
        ))

        app.execute_data(data_text="w = W(n = 2);\n")

        instance: Any = app.query(struct_name="W").first()

        # Member access materializes (and stamps) the lazy default
        instance.ensure_attribute(name="w")

        value: Any = instance.attributes["w"]

        assert value.get() == [1.0, 2.0]
        assert value.concrete_dims == [2]
