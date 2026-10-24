"""
Tests for Phase D field layers: the GeoGrid type, cell quantization,
bilinear sampling, sea-level registration, altitude integration, and
persistence.
"""

# Import Modules
import math
from typing import Any

import pytest

from src.api import Nadata
from src.core.database.database import NaDataDatabase
from src.core.types.home_types.coordinate import Coordinate, altitude
from src.core.types.home_types.geogrid import bilinear_sample
from src.core.types.home_types.planet import DEFAULT_PLANET, set_current_planet
from src.core.types.home_types.sea_level import (
    GeoSeaLevelAdapter,
    set_sea_level_surface,
)
from src.core.types.types import TypeGeoGrid
from src.core.values.values import ValueGeoGrid
from src.parser.expressions.type_parser import TypeParser
from src.parser.nastruct_parser.resolver import TypeResolver

R: float = 6371008.8



def _teardown() -> None:
    """
    Restores the default session state.

    Returns:
        None: Nothing.
    """

    set_sea_level_surface(surface=None)
    set_current_planet(planet=DEFAULT_PLANET)



class TestTypeResolution:
    """
    Resolution of GeoGrid[T] and GeoGrid[T, arcsec] declarations.
    """

    def test_default_cell_size(self) -> None:
        """
        GeoGrid[Float64] defaults to 30 arc-second cells.
        """

        resolver: TypeResolver = TypeResolver()

        resolved: Any = resolver.resolve(
            tmp_type=TypeParser.parse_text(text="GeoGrid[Float64]")
        )

        assert isinstance(resolved, TypeGeoGrid)
        assert resolved.cell_arcsec == 30
        assert resolved.value_type is not None

    def test_explicit_cell_size(self) -> None:
        """
        GeoGrid[Str, 3600] carries a one-degree cell size.
        """

        resolver: TypeResolver = TypeResolver()

        resolved: Any = resolver.resolve(
            tmp_type=TypeParser.parse_text(text="GeoGrid[Str, 3600]")
        )

        assert isinstance(resolved, TypeGeoGrid)
        assert resolved.cell_arcsec == 3600
        assert resolved.n_x == 360
        assert resolved.n_y == 180

    def test_rejects_bad_cell_size(self) -> None:
        """
        Non-integer or out-of-range cell sizes are rejected.
        """

        resolver: TypeResolver = TypeResolver()

        with pytest.raises(TypeError):
            resolver.resolve(
                tmp_type=TypeParser.parse_text(text="GeoGrid[Float64, Str]")
            )

        with pytest.raises(ValueError):
            resolver.resolve(
                tmp_type=TypeParser.parse_text(text="GeoGrid[Float64, 0]")
            )

        with pytest.raises(ValueError):
            resolver.resolve(
                tmp_type=TypeParser.parse_text(text="GeoGrid[Float64, 700000]")
            )

    def test_repr(self) -> None:
        """
        The repr shows the cell type and nominal cell size.
        """

        grid_type: TypeGeoGrid = TypeGeoGrid(
            value_type=__import__("src.core.types.types", fromlist=["Float64"]).Float64,
            cell_arcsec=60,
        )

        assert "GeoGrid" in repr(grid_type)
        assert "60" in repr(grid_type)


class TestCellMath:
    """
    Cell quantization and wrapping.
    """

    def test_cell_of_and_center(self) -> None:
        """
        cell_of and cell_center are mutual inverses up to half a cell.
        """

        grid_type: TypeGeoGrid = TypeGeoGrid(
            value_type=__import__("src.core.types.types", fromlist=["Float64"]).Float64,
            cell_arcsec=3600,
        )

        key: tuple[int, int] = grid_type.cell_of(lat_deg=43.6, lon_deg=1.4)

        assert key == (181, 133)

        lat: float
        lon: float
        lat, lon = grid_type.cell_center(cell=key)

        assert lat == 43.5
        assert lon == 1.5

    def test_antimeridian_wrapping(self) -> None:
        """
        Longitudes near plus/minus 180 map to wrapping columns.
        """

        grid_type: TypeGeoGrid = TypeGeoGrid(
            value_type=__import__("src.core.types.types", fromlist=["Float64"]).Float64,
            cell_arcsec=3600,
        )

        east: tuple[int, int] = grid_type.cell_of(lat_deg=0.0, lon_deg=179.5)
        west: tuple[int, int] = grid_type.cell_of(lat_deg=0.0, lon_deg=-179.5)

        # Adjacent columns across the antimeridian: 179.5 is in
        # column 359, -179.5 wraps to column 0
        assert east[0] == 359
        assert west[0] == 0

        # The sample path must treat them as neighbors
        cells: dict[tuple[int, int], Any] = {(359, 90): 10.0}

        value: Any = bilinear_sample(
            cells=cells,
            n_x=grid_type.n_x,
            n_y=grid_type.n_y,
            cell_deg=grid_type.cell_deg,
            lat_deg=0.0,
            lon_deg=179.9,
        )

        assert value == 10.0

    def test_pole_clamping(self) -> None:
        """
        Latitudes clamp at the poles instead of escaping the grid.
        """

        grid_type: TypeGeoGrid = TypeGeoGrid(
            value_type=__import__("src.core.types.types", fromlist=["Float64"]).Float64,
            cell_arcsec=3600,
        )

        assert grid_type.cell_of(lat_deg=95.0, lon_deg=0.0)[1] == grid_type.n_y - 1
        assert grid_type.cell_of(lat_deg=-95.0, lon_deg=0.0)[1] == 0

    def test_check_value_rejects_bad_keys(self) -> None:
        """
        Malformed cell keys are rejected.
        """

        grid_type: TypeGeoGrid = TypeGeoGrid(
            value_type=__import__("src.core.types.types", fromlist=["Float64"]).Float64,
            cell_arcsec=3600,
        )

        with pytest.raises(TypeError):
            grid_type.check_value(value={(0,): 1.0})

        with pytest.raises(TypeError):
            grid_type.check_value(value={(0, 0, 0): 1.0})

        with pytest.raises(TypeError):
            grid_type.check_value(value={(grid_type.n_x, 0): 1.0})

        with pytest.raises(TypeError):
            grid_type.check_value(value={(0, 0): "str"})


class TestBilinear:
    """
    Bilinear sampling over sparse cells.
    """

    def _grid_type(self) -> TypeGeoGrid:
        """
        Builds a one-degree grid type.

        Returns:
            TypeGeoGrid: The grid type.
        """

        return TypeGeoGrid(
            value_type=__import__("src.core.types.types", fromlist=["Float64"]).Float64,
            cell_arcsec=3600,
        )

    def test_exact_at_centers(self) -> None:
        """
        Sampling at a cell center returns that cell's value exactly.
        """

        grid_type: TypeGeoGrid = self._grid_type()

        cells: dict[tuple[int, int], Any] = {(181, 133): 42.0}

        lat: float
        lon: float
        lat, lon = grid_type.cell_center(cell=(181, 133))

        value: Any = bilinear_sample(
            cells=cells,
            n_x=grid_type.n_x,
            n_y=grid_type.n_y,
            cell_deg=grid_type.cell_deg,
            lat_deg=lat,
            lon_deg=lon,
        )

        assert value == 42.0

    def test_four_cell_average(self) -> None:
        """
        Sampling at the shared corner of four cell centers averages
        them with equal weights.
        """

        grid_type: TypeGeoGrid = self._grid_type()

        cells: dict[tuple[int, int], Any] = {
            (181, 133): 150.0,
            (182, 133): 250.0,
            (181, 134): 50.0,
            (182, 134): 350.0,
        }

        value: Any = bilinear_sample(
            cells=cells,
            n_x=grid_type.n_x,
            n_y=grid_type.n_y,
            cell_deg=grid_type.cell_deg,
            lat_deg=44.0,
            lon_deg=2.0,
        )

        assert value == pytest.approx(200.0)

    def test_missing_cells_renormalize(self) -> None:
        """
        Absent cells contribute nothing; weights renormalize.
        """

        grid_type: TypeGeoGrid = self._grid_type()

        # Only two of the four surrounding centers have data
        cells: dict[tuple[int, int], Any] = {
            (181, 133): 100.0,
            (182, 133): 300.0,
        }

        value: Any = bilinear_sample(
            cells=cells,
            n_x=grid_type.n_x,
            n_y=grid_type.n_y,
            cell_deg=grid_type.cell_deg,
            lat_deg=43.5,
            lon_deg=2.0,
        )

        assert value == pytest.approx(200.0)

    def test_no_data_returns_none(self) -> None:
        """
        Sampling far from any data returns None.
        """

        grid_type: TypeGeoGrid = self._grid_type()

        assert (
            bilinear_sample(
                cells={(181, 133): 1.0},
                n_x=grid_type.n_x,
                n_y=grid_type.n_y,
                cell_deg=grid_type.cell_deg,
                lat_deg=0.0,
                lon_deg=0.0,
            )
            is None
        )

    def test_non_numeric_rejected(self) -> None:
        """
        Sampling a non-numeric layer raises TypeError.
        """

        grid_type: TypeGeoGrid = self._grid_type()

        with pytest.raises(TypeError):
            bilinear_sample(
                cells={(181, 133): "str"},
                n_x=grid_type.n_x,
                n_y=grid_type.n_y,
                cell_deg=grid_type.cell_deg,
                lat_deg=43.5,
                lon_deg=1.5,
            )


class TestLanguageIntegration:
    """
    GeoGrid members from .nadata scripts.
    """

    def _app(self) -> Any:
        """
        Creates a facade with a World struct carrying a GeoGrid.

        Returns:
            Any: The Nadata facade.
        """

        app: Any = Nadata(database=NaDataDatabase())

        app.execute_schema(
            "class World { name: Str; elevation: GeoGrid[Float64, 3600] = {}; }"
        )

        return app

    def test_members(self) -> None:
        """
        set_at / size / has / at / remove / cell_of work from scripts.
        """

        app: Any = self._app()

        app.execute_data(
            "w = World(name=\"earth\");"
            "w.elevation.set_at(Coordinate(43, 1, 6371008.8), 150.0);"
            "w.elevation.set_at(Coordinate(43, 2, 6371008.8), 250.0);"
            "print(\"size:\", w.elevation.size());"
            "print(\"has:\", w.elevation.has(Coordinate(43, 1, 6371008.8)));"
            "print(\"at:\", w.elevation.at(Coordinate(43, 2, 6371008.8)));"
            "w.elevation.remove(Coordinate(43, 2, 6371008.8));"
            "print(\"after remove:\", w.elevation.size());"
            "print(\"cell_of col:\", w.elevation.cell_of(Coordinate(43, 1, 6371008.8))[0]);"
        )

        assert app.get_attribute(symbol="w", name="name") == "earth"

    def test_sample_with_default(self) -> None:
        """
        sample interpolates where data exists and falls back to the
        grid default elsewhere.
        """

        app: Any = self._app()

        app.execute_data(
            "w = World(name=\"earth\");"
            "w.elevation.set_at(Coordinate(43, 1, 6371008.8), 150.0);"
            "w.elevation.set_at(Coordinate(43, 2, 6371008.8), 250.0);"
            "w.elevation.set_at(Coordinate(44, 1, 6371008.8), 50.0);"
            "w.elevation.set_at(Coordinate(44, 2, 6371008.8), 350.0);"
            "w.elevation.set_default(0.0);"
            "s1 = w.elevation.sample(Coordinate(44, 2, 6371008.8));"
            "print(\"corner:\", s1);"
            "s2 = w.elevation.sample(Coordinate(80, 80, 6371008.8));"
            "print(\"empty:\", s2);"
        )

        table: Any = app.database.tables_by_struct_name["World"]

        grid: Any = table.get(local_id=0).attributes["elevation"]

        assert grid.sample(lat_deg=44.0, lon_deg=2.0) == pytest.approx(200.0)
        assert grid.sample(lat_deg=80.0, lon_deg=80.0) == 0.0

    def test_sample_without_data_raises(self) -> None:
        """
        sample with no data and no default raises KeyError.
        """

        app: Any = self._app()

        app.execute_data("w = World(name=\"earth\");")

        table: Any = app.database.tables_by_struct_name["World"]

        instance: Any = table.get(local_id=0)

        instance.ensure_attribute(name="elevation")

        grid: Any = instance.attributes["elevation"]

        # Python API: None without a default
        assert grid.sample(lat_deg=0.0, lon_deg=0.0) is None

        # Language member: KeyError without a default
        with pytest.raises(KeyError):
            app.execute_data(
                "w.sample_check = w.elevation.sample("
                "Coordinate(0, 0, 6371008.8));"
            )


class TestSeaLevel:
    """
    register_sea_level and altitude integration.
    """

    def _app_with_sea(self) -> Any:
        """
        Creates a facade, registers a sea grid, and returns both.

        Returns:
            Any: The Nadata facade.
        """

        app: Any = Nadata(database=NaDataDatabase())

        app.execute_schema(
            "class World { name: Str; sea: GeoGrid[Float64, 3600] = {}; }"
        )

        # A one-degree cell at (43.5, 1.5) center with a +10 m sea
        # surface radius bump
        app.execute_data(
            "w = World(name=\"earth\");"
            "w.sea.set_at(Coordinate(43.5, 1.5, 6371008.8), 6371018.8);"
            "register_sea_level(w.sea);"
        )

        return app

    def test_altitude_uses_sea_level(self) -> None:
        """
        altitude references the interpolated sea surface where the
        layer has data, and the geometric surface elsewhere.
        """

        app: Any = self._app_with_sea()

        try:
            # Above the +10 m bump: 20 m radius above the mean sphere
            above: float = altitude(Coordinate(43.5, 1.5, R + 20.0))

            assert above == pytest.approx(10.0, abs=0.5)

            # Far away: geometric surface radius applies
            far: float = altitude(Coordinate(10.0, 10.0, R + 100.0))

            assert far == pytest.approx(100.0)
        finally:
            _teardown()

    def test_member_altitude_consistent(self) -> None:
        """
        The Coordinate member altitude() matches the free function.
        """

        app: Any = self._app_with_sea()

        try:
            member: float = Coordinate(
                lat=43.5, lon=1.5, radius=R + 20.0
            ).altitude()

            assert member == pytest.approx(10.0, abs=0.5)
        finally:
            _teardown()

    def test_clear_sea_level(self) -> None:
        """
        clear_sea_level restores geometric altitudes.
        """

        app: Any = self._app_with_sea()

        try:
            app.execute_data("clear_sea_level();")

            geometric: float = altitude(Coordinate(43.5, 1.5, R + 20.0))

            assert geometric == pytest.approx(20.0)
        finally:
            _teardown()

    def test_register_rejects_non_grid(self) -> None:
        """
        register_sea_level rejects non-GeoGrid arguments.
        """

        from src.core.expressions.builtins import RegisterSeaLevelFunction

        with pytest.raises(TypeError):
            RegisterSeaLevelFunction()()

    def test_adapter_sampling(self) -> None:
        """
        The adapter exposes bilinear radius sampling with None gaps.
        """

        grid: ValueGeoGrid = ValueGeoGrid(
            value_type=TypeGeoGrid(
                value_type=__import__("src.core.types.types", fromlist=["Float64"]).Float64,
                cell_arcsec=3600,
            ),
            cells={(181, 133): R + 10.0},
        )

        adapter: GeoSeaLevelAdapter = GeoSeaLevelAdapter(grid=grid)

        assert adapter.radius_at(lat_deg=43.5, lon_deg=1.5) == pytest.approx(
            R + 10.0
        )

        assert adapter.radius_at(lat_deg=0.0, lon_deg=0.0) is None


class TestPersistence:
    """
    Binary cell encoding round-trips.
    """

    def test_dump_load_roundtrip(self, tmp_path: Any) -> None:
        """
        A dumped GeoGrid reloads with its cells and type intact.
        """

        app: Any = Nadata(database=NaDataDatabase())

        app.execute_schema(
            "class World { name: Str; elevation: GeoGrid[Float64, 3600] = {}; }"
        )

        app.execute_data(
            "w = World(name=\"earth\");"
            "w.elevation.set_at(Coordinate(43, 1, 6371008.8), 150.0);"
            "w.elevation.set_at(Coordinate(43, 2, 6371008.8), 250.0);"
            "w.elevation.set_at(Coordinate(44, 1, 6371008.8), 50.0);"
            "w.elevation.set_at(Coordinate(44, 2, 6371008.8), 350.0);"
        )

        storage: str = str(tmp_path / "geogrid.ndb")

        app.dump(storage_path=storage)

        reloaded: Any = Nadata.load(path=storage)

        table: Any = reloaded.database.tables_by_struct_name["World"]

        grid: Any = table.get(local_id=0).attributes["elevation"]

        assert isinstance(grid, ValueGeoGrid)
        assert len(grid.cells) == 4
        assert grid.value_type.cell_arcsec == 3600

        assert grid.sample(lat_deg=44.0, lon_deg=2.0) == pytest.approx(200.0)

    def teardown_method(self) -> None:
        """
        Restores the default session state.

        Returns:
            None: Nothing.
        """

        _teardown()
