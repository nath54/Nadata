"""
Tests for sparse spatial grids (GridSquare2D / GridHex2D / GridCube3D) and
dense tensors (Tensor[T, dims...]).

Covers topology resolution, coordinate validation, typed members,
neighborhoods, shape enforcement, language integration, and persistence
round trips.
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
from src.core.expressions.scope import ExecutionScope
from src.core.types.types import (
    TypeFloat,
    TypeInt,
    TypeSpatialGrid,
    TypeStr,
    TypeTensor,
)
from src.core.values.values import (
    ValueSpatialGrid,
    ValueTensor,
    make_value,
)
from src.parser.expressions.type_parser import TypeParser
from src.parser.nastruct_parser.resolver import TypeResolver

SCHEMA: str = """
class World {
    name: Str;
    terrain: GridSquare2D[Str] = {};
    hexes: GridHex2D[Str] = {};
    volumes: GridCube3D[Str] = {};
    kernel: Tensor[Float32, 3, 3];
    weights: Tensor[Float32, 2, 2, 2] = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
}
"""


class TestTypeResolution:
    """
    Tests resolution of the grid topologies and tensor shapes.
    """

    def test_grid_topologies(self) -> None:
        """
        Tests that each grid name resolves to its topology and dims.
        """

        resolver: TypeResolver = TypeResolver()

        square: Any = resolver.resolve(
            tmp_type=TypeParser.parse_text(text="GridSquare2D[Str]")
        )
        hexagonal: Any = resolver.resolve(
            tmp_type=TypeParser.parse_text(text="GridHex2D[Str]")
        )
        cube: Any = resolver.resolve(
            tmp_type=TypeParser.parse_text(text="GridCube3D[Str]")
        )

        assert isinstance(square, TypeSpatialGrid) and square.dims == 2
        assert square.topology == "square2d"
        assert len(square.neighbor_offsets()) == 4

        assert isinstance(hexagonal, TypeSpatialGrid) and hexagonal.topology == "hex2d"
        assert len(hexagonal.neighbor_offsets()) == 6

        assert isinstance(cube, TypeSpatialGrid) and cube.topology == "cube3d"
        assert cube.dims == 3

    def test_tensor_shape_resolution(self) -> None:
        """
        Tests that tensor dimensions come from the type arguments.
        """

        resolver: TypeResolver = TypeResolver()

        tensor: Any = resolver.resolve(
            tmp_type=TypeParser.parse_text(text="Tensor[Float32, 3, 4]")
        )

        assert isinstance(tensor, TypeTensor)
        assert tensor.dims == [3, 4]
        assert tensor.length() == 12
        assert repr(tensor) == "Tensor[Float32(exp=8, mant=23), 3, 4]" or "3, 4" in repr(tensor)

    def test_tensor_rejects_bad_dims(self) -> None:
        """
        Tests that non-numeric or non-positive dimensions are rejected.
        """

        resolver: TypeResolver = TypeResolver()

        with pytest.raises(TypeError):
            resolver.resolve(tmp_type=TypeParser.parse_text(text="Tensor[Str]"))

        with pytest.raises(TypeError):
            resolver.resolve(tmp_type=TypeParser.parse_text(text="Tensor[Str, Str]"))

    def test_grid_reprs(self) -> None:
        """
        Tests the topology reprs.
        """

        assert repr(TypeSpatialGrid(value_type=TypeStr(), topology="square2d")) == (
            "GridSquare2D[Str]"
        )
        assert repr(TypeSpatialGrid(value_type=TypeStr(), topology="hex2d")) == (
            "GridHex2D[Str]"
        )


class TestValueLayer:
    """
    Tests the ValueSpatialGrid and ValueTensor payloads.
    """

    def test_grid_validates_coordinates(self) -> None:
        """
        Tests that wrong-arity or non-integer coordinates are rejected.
        """

        with pytest.raises(TypeError):
            ValueSpatialGrid(
                value_type=TypeSpatialGrid(value_type=TypeStr(), topology="square2d"),
                cells={(1, 2, 3): "x"},
            )

        with pytest.raises(TypeError):
            ValueSpatialGrid(
                value_type=TypeSpatialGrid(value_type=TypeStr(), topology="square2d"),
                cells={(1.5, 2): "x"},
            )

    def test_tensor_shape_enforced(self) -> None:
        """
        Tests that the flat payload must match the declared shape.
        """

        with pytest.raises(TypeError):
            ValueTensor(
                value_type=TypeTensor(value_type=TypeStr(), dims=[2, 2]),
                cells=["a", "b", "c"],
            )

        tensor: ValueTensor = ValueTensor(
            value_type=TypeTensor(value_type=TypeStr(), dims=[2, 2]),
            cells=["a", "b", "c", "d"],
        )

        assert tensor.flat_index(indices=(1, 0)) == 2

        with pytest.raises(IndexError):
            tensor.flat_index(indices=(2, 0))

    def test_make_value_dispatch(self) -> None:
        """
        Tests that make_value wraps both payloads.
        """

        grid: Any = make_value(
            value_type=TypeSpatialGrid(value_type=TypeStr(), topology="square2d"),
            raw={(0, 0): "x"},
        )

        assert isinstance(grid, ValueSpatialGrid)

        tensor: Any = make_value(
            value_type=TypeTensor(value_type=TypeStr(), dims=[2]),
            raw=["a", "b"],
        )

        assert isinstance(tensor, ValueTensor)


class TestLanguage:
    """
    Tests grids and tensors inside the language.
    """

    def _app(self) -> Nadata:
        """
        Builds an app with the World schema and a populated instance.

        Returns:
            Nadata: The prepared facade.
        """

        app: Nadata = Nadata(database=NaDataDatabase())

        app.execute_schema(schema_text=SCHEMA)

        app.execute_data(data_text=(
            'w = World(name = "map", kernel = [[1.0, 0.0, -1.0], [0.5, 0.0, -0.5],'
            " [0.25, 0.0, -0.25]]);\n"
            'w.terrain.set_at(0, 0, "grass");\n'
            'w.terrain.set_at(1, 0, "water");\n'
            'w.terrain.set_at(-3, 7, "lava");\n'
            'w.hexes.set_at(1, 0, "forest");\n'
        ))

        return app

    def test_sparse_cells_and_negative_coords(self) -> None:
        """
        Tests that sparse cells (including negative coordinates) persist.
        """

        app: Nadata = self._app()

        world: Any = app.query(struct_name="World").first()

        terrain: Any = world.get_attribute(name="terrain")

        assert terrain.cells == {(-3, 7): "lava", (0, 0): "grass", (1, 0): "water"}

    def test_tensor_literal_coercion(self) -> None:
        """
        Tests that nested array literals flatten into the tensor payload.
        """

        app: Nadata = self._app()

        world: Any = app.query(struct_name="World").first()

        assert world.get_attribute(name="kernel").get() == [
            1.0, 0.0, -1.0, 0.5, 0.0, -0.5, 0.25, 0.0, -0.25,
        ]

    def test_tensor_schema_default(self) -> None:
        """
        Tests that a tensor attribute default flattens correctly.
        """

        app: Nadata = self._app()

        world: Any = app.query(struct_name="World").to_list()[0]

        weights: Any = world.get_attribute(name="weights")

        assert isinstance(weights, ValueTensor)
        assert weights.get() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

    def test_neighborhoods_by_topology(self) -> None:
        """
        Tests neighbor offsets per topology via a probe struct.
        """

        app: Nadata = self._app()

        app.execute_schema(schema_text="class Probe { n: Array[Array[Int64]] = []; }")

        app.execute_data(data_text=(
            "p = Probe(n = w.terrain.neighbors(0, 0));\n"
        ))

        probe: Any = app.query(struct_name="Probe").first()

        assert probe.get_attribute(name="n").get() == [[1, 0], [-1, 0], [0, 1], [0, -1]]

    def test_occupied_neighbors(self) -> None:
        """
        Tests that occupied_neighbors filters to existing cells.
        """

        app: Nadata = self._app()

        app.execute_schema(schema_text="class Probe { n: Array[Array[Int64]] = []; }")

        app.execute_data(data_text=(
            "p = Probe(n = w.terrain.occupied_neighbors(0, 0));\n"
        ))

        probe: Any = app.query(struct_name="Probe").first()

        assert probe.get_attribute(name="n").get() == [[1, 0]]

    def test_shape_member(self) -> None:
        """
        Tests the tensor shape() member.
        """

        app: Nadata = self._app()

        app.execute_schema(schema_text="class Probe { s: Array[Int32] = []; }")

        app.execute_data(data_text="p = Probe(s = w.kernel.shape());\n")

        probe: Any = app.query(struct_name="Probe").first()

        assert probe.get_attribute(name="s").get() == [3, 3]

    def test_bounds_and_has(self) -> None:
        """
        Tests tensor index bounds and grid has() via the language.
        """

        app: Nadata = self._app()

        world: Any = app.query(struct_name="World").first()

        with pytest.raises(IndexError):
            world.get_attribute(name="kernel").flat_index(indices=(3, 0))

        app.execute_schema(schema_text="class Probe2 { h: Bool = false; }")

        app.execute_data(data_text="p2 = Probe2(h = w.terrain.has(0, 0));\n")

        probe: Any = app.query(struct_name="Probe2").first()

        assert probe.get_attribute(name="h").get() is True

    def test_for_in_iteration_orders(self) -> None:
        """
        Tests for-in over grids (coordinate sorted) and tensors (row-major).
        """

        app: Nadata = self._app()

        app.execute_schema(schema_text=(
            "class Probe { n: Int32 = 0; f: Float32 = 0.0; }\n"
        ))

        app.execute_data(data_text=(
            "w = World(kernel = [[1.0, 0.0, -1.0], [0.5, 0.0, -0.5], [0.25, 0.0, 2.0]]);\n"
            "w.terrain.set_at(0, 0, \"grass\");\n"
            "w.terrain.set_at(1, 0, \"water\");\n"
            "w.terrain.set_at(-3, 7, \"lava\");\n"
            "p = Probe();\n"
            "for (c: Str in w.terrain) { p.n = p.n + 1; }\n"
            "for (x: Float32 in w.kernel) { p.f = p.f + x; }\n"
        ))

        probe: Any = app.query(struct_name="Probe").first()

        assert probe.get_attribute(name="n").get() == 3
        assert probe.get_attribute(name="f").get() == 2.25


class TestPersistence:
    """
    Tests grids and tensors across the v2 binary format.
    """

    def _round_trip(self) -> tuple[Any, Any]:
        """
        Builds the World database and round trips it twice.

        Returns:
            tuple[Any, Any]: First and second round trip instance mappings.
        """

        app: Nadata = Nadata(database=NaDataDatabase())

        app.execute_schema(schema_text=SCHEMA)

        app.execute_data(data_text=(
            'w = World(name = "map", kernel = [[1.0, 0.0, -1.0], [0.5, 0.0, -0.5],'
            " [0.25, 0.0, -0.25]]);\n"
            'w.terrain.set_at(0, 0, "grass");\n'
            'w.terrain.set_at(-3, 7, "lava");\n'
            'w.hexes.set_at(1, 0, "forest");\n'
            'w.volumes.set_at(0, 0, 1, "core");\n'
        ))

        temp_dir: str = tempfile.mkdtemp()
        storage_path: str = os.path.join(temp_dir, "g.ndb")

        dump_database(database=app.database, storage_path=storage_path)

        reloaded: Any = load_database(storage_path=storage_path)

        storage_path2: str = os.path.join(temp_dir, "g2.ndb")

        dump_database(database=reloaded, storage_path=storage_path2)

        reloaded2: Any = load_database(storage_path=storage_path2)

        return (
            reloaded.tables_by_struct_name["World"].instances,
            reloaded2.tables_by_struct_name["World"].instances,
        )

    def test_round_trip(self) -> None:
        """
        Tests grids and tensors across two dump/load cycles.
        """

        lazy: Any
        lazy2: Any

        lazy, lazy2 = self._round_trip()

        first: Any = lazy[0]

        assert first.get_attribute(name="terrain").get() == {
            (-3, 7): "lava", (0, 0): "grass",
        }
        assert first.get_attribute(name="hexes").get() == {(1, 0): "forest"}
        assert first.get_attribute(name="volumes").get() == {(0, 0, 1): "core"}
        assert first.get_attribute(name="kernel").get() == [
            1.0, 0.0, -1.0, 0.5, 0.0, -0.5, 0.25, 0.0, -0.25,
        ]

        second: Any = lazy2[0]

        assert second.get_attribute(name="terrain").get() == {
            (-3, 7): "lava", (0, 0): "grass",
        }

    def test_lazy_open_undecoded(self) -> None:
        """
        Tests that grid and tensor columns stay pending on lazy open.
        """

        from src.core.database.persistence.lazy_instances import LazyInstanceDict

        app: Nadata = Nadata(database=NaDataDatabase())

        app.execute_schema(schema_text=SCHEMA)

        app.execute_data(data_text='w = World(name = "map");\n')

        temp_dir: str = tempfile.mkdtemp()
        storage_path: str = os.path.join(temp_dir, "lazy.ndb")

        dump_database(database=app.database, storage_path=storage_path)

        reloaded: Any = load_database(storage_path=storage_path)

        lazy: Any = reloaded.tables_by_struct_name["World"].instances

        assert isinstance(lazy, LazyInstanceDict)
        assert dict.keys(lazy) == dict.keys({})
