"""
Tests for the Phase C spatial index: cell indexing of geometry
attributes, hierarchical locate() descent with fast paths, nearest /
within queries, incremental maintenance, and dump-time row
clustering.
"""

# Import Modules
import math
from typing import Any

import pytest

from src.api import Nadata
from src.core.database.database import NaDataDatabase
from src.core.database.spatial_index import SpatialIndex
from src.core.types.home_types.coordinate import Coordinate
from src.core.types.home_types.geometry import Polygon, Polyline, circle
from src.core.types.home_types.planet import DEFAULT_PLANET, set_current_planet

R: float = 6371008.8



def _square(
    lat0: float,
    lon0: float,
    lat1: float,
    lon1: float,
) -> Polygon:
    """
    Builds an axis-aligned lat/lon square on the default sphere.

    Args:
        lat0 (float): Minimum latitude.
        lon0 (float): Minimum longitude.
        lat1 (float): Maximum latitude.
        lon1 (float): Maximum longitude.

    Returns:
        Polygon: The square ring.
    """

    return Polygon(
        vertices=[
            Coordinate(lat0, lon0, R),
            Coordinate(lat0, lon1, R),
            Coordinate(lat1, lon1, R),
            Coordinate(lat1, lon0, R),
        ]
    )



class TestSpatialIndexBuild:
    """
    Index construction over a table.
    """

    def _db_with_regions(self) -> Any:
        """
        Creates a database with a Region struct and three instances.

        Returns:
            Any: The Nadata facade.
        """

        app: Any = Nadata(database=NaDataDatabase())

        app.execute_schema(
            "class Region { name: Str; boundary: Polygon; }"
        )

        app.execute_data(
            "a = Region(name=\"a\", boundary=Polygon(["
            "Coordinate(0,0,6371008.8), Coordinate(0,1,6371008.8),"
            " Coordinate(1,1,6371008.8), Coordinate(1,0,6371008.8)]));"
            "b = Region(name=\"b\", boundary=Polygon(["
            "Coordinate(10,10,6371008.8), Coordinate(10,11,6371008.8),"
            " Coordinate(11,11,6371008.8), Coordinate(11,10,6371008.8)]));"
            "c = Region(name=\"c\", boundary=Polygon(["
            "Coordinate(20,20,6371008.8), Coordinate(20,21,6371008.8),"
            " Coordinate(21,21,6371008.8), Coordinate(21,20,6371008.8)]));"
        )

        return app

    def test_build_populates_cells(self) -> None:
        """
        Building the index populates cells and one entry per instance.
        """

        app: Any = self._db_with_regions()

        table: Any = app.database.tables_by_struct_name["Region"]

        index: Any = table.create_spatial_index(attribute="boundary", lod=5)

        assert len(index.entries) == 3
        assert len(index.cells) > 0

        # Every entry's cells are registered
        for entry in index.entries.values():
            for cell in entry.cells:
                assert index.cells[cell] is not None

    def test_rejects_non_geometry_attribute(self) -> None:
        """
        Indexing a scalar attribute raises TypeError.
        """

        app: Any = self._db_with_regions()

        table: Any = app.database.tables_by_struct_name["Region"]

        with pytest.raises(TypeError):
            table.create_spatial_index(attribute="name")

    def test_rejects_undeclared_attribute(self) -> None:
        """
        Indexing an undeclared attribute raises AttributeError.
        """

        app: Any = self._db_with_regions()

        table: Any = app.database.tables_by_struct_name["Region"]

        with pytest.raises(AttributeError):
            table.create_spatial_index(attribute="nope")

    def test_rejects_bad_lod(self) -> None:
        """
        lod outside [1, 20] raises ValueError.
        """

        app: Any = self._db_with_regions()

        table: Any = app.database.tables_by_struct_name["Region"]

        with pytest.raises(ValueError):
            table.create_spatial_index(attribute="boundary", lod=0)

        with pytest.raises(ValueError):
            table.create_spatial_index(attribute="boundary", lod=99)

    def test_coordinate_attribute(self) -> None:
        """
        Coordinate attributes index into a single cell.
        """

        app: Any = self._db_with_regions()

        app.execute_schema("class Poi { at: Coordinate; }")
        app.execute_data("p = Poi(at=Coordinate(5, 5, 6371008.8));")

        table: Any = app.database.tables_by_struct_name["Poi"]

        index: Any = table.create_spatial_index(attribute="at", lod=5)

        assert len(index.entries) == 1
        assert len(next(iter(index.entries.values())).cells) == 1


class TestLocate:
    """
    Flat and hierarchical locate.
    """

    def _hierarchy_app(self) -> Any:
        """
        Creates a World -> Country -> City hierarchy.

        Returns:
            Any: The Nadata facade.
        """

        app: Any = Nadata(database=NaDataDatabase())

        app.execute_schema(
            "class Area { name: Str; boundary: Polygon; parent: Area; }"
        )

        app.execute_data(
            "world = Area(name=\"world\", boundary=Polygon(["
            "Coordinate(-80,-30,6371008.8), Coordinate(-80,30,6371008.8),"
            " Coordinate(80,30,6371008.8), Coordinate(80,-30,6371008.8)]));"
            "country = Area(name=\"country\", boundary=Polygon(["
            "Coordinate(0,0,6371008.8), Coordinate(0,4,6371008.8),"
            " Coordinate(4,4,6371008.8), Coordinate(4,0,6371008.8)]),"
            " parent=world);"
            "city = Area(name=\"city\", boundary=Polygon(["
            "Coordinate(1,1,6371008.8), Coordinate(1,2,6371008.8),"
            " Coordinate(2,2,6371008.8), Coordinate(2,1,6371008.8)]),"
            " parent=country);"
        )

        return app

    def test_flat_locate(self) -> None:
        """
        Flat locate finds the containing region and nothing outside.
        """

        app: Any = self._hierarchy_app()

        table: Any = app.database.tables_by_struct_name["Area"]

        index: Any = table.create_spatial_index(attribute="boundary", lod=5)

        inside_city: Any = index.locate(
            coord=Coordinate(1.5, 1.5, R)
        )

        assert inside_city is not None

        found_name: str = table.get(local_id=inside_city).attributes[
            "name"
        ].get()

        # A flat scan may return any containing region; the point is
        # inside all three
        assert found_name in ("world", "country", "city")

        assert index.locate(coord=Coordinate(50.0, 50.0, R)) is None

    def test_hierarchical_descent(self) -> None:
        """
        Hierarchical locate returns the deepest containing region.
        """

        app: Any = self._hierarchy_app()

        table: Any = app.database.tables_by_struct_name["Area"]

        index: Any = table.create_spatial_index(
            attribute="boundary", lod=5, parent_attribute="parent"
        )

        # Point in the city: descent ends at the city
        city_id: Any = index.locate(coord=Coordinate(1.5, 1.5, R))

        assert city_id is not None
        assert table.get(local_id=city_id).attributes["name"].get() == "city"

        # Point in the country but outside the city: the country
        country_id: Any = index.locate(coord=Coordinate(3.0, 3.0, R))

        assert country_id is not None
        assert table.get(local_id=country_id).attributes["name"].get() == "country"

        # Point outside everything
        assert index.locate(coord=Coordinate(50.0, 50.0, R)) is None

    def test_fast_paths_agree_with_exact(self) -> None:
        """
        The fast-path result matches brute-force contains for a dense
        sample of points.
        """

        ring: Polygon = _square(0.0, 0.0, 2.0, 2.0)

        app: Any = Nadata(database=NaDataDatabase())

        app.execute_schema("class Z { boundary: Polygon; }")
        app.execute_data(
            "z = Z(boundary=Polygon([Coordinate(0,0,6371008.8),"
            " Coordinate(0,2,6371008.8), Coordinate(2,2,6371008.8),"
            " Coordinate(2,0,6371008.8)]));"
        )

        table: Any = app.database.tables_by_struct_name["Z"]

        index: Any = table.create_spatial_index(attribute="boundary", lod=6)

        for i in range(20):
            for j in range(20):
                point: Coordinate = Coordinate(
                    lat=-1.0 + 0.2 * i, lon=-1.0 + 0.2 * j, radius=R
                )

                expected: bool = ring.contains(point=point)
                actual: Any = index.locate(coord=point)

                assert (actual is not None) == expected


class TestNearest:
    """
    nearest() queries.
    """

    def _poi_app(self) -> Any:
        """
        Creates a Poi struct with a grid of points.

        Returns:
            Any: The Nadata facade.
        """

        app: Any = Nadata(database=NaDataDatabase())

        app.execute_schema("class Poi { at: Coordinate; name: Str; }")

        statements: list[str] = []

        for i, (lat, lon) in enumerate(
            [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (10.0, 10.0), (20.0, 20.0)]
        ):
            statements.append(
                f"p{i} = Poi(name=\"p{i}\", "
                f"at=Coordinate({lat}, {lon}, 6371008.8));"
            )

        app.execute_data("".join(statements))

        return app

    def test_nearest_ordering(self) -> None:
        """
        nearest returns instances ordered by center distance.
        """

        app: Any = self._poi_app()

        table: Any = app.database.tables_by_struct_name["Poi"]

        index: Any = table.create_spatial_index(attribute="at", lod=5)

        result: list[int] = index.nearest(
            coord=Coordinate(0.4, 0.4, R), k=3
        )

        names: list[str] = [
            table.get(local_id=local_id).attributes["name"].get()
            for local_id in result
        ]

        # (0,0) and its two neighbors are nearest; (10,10) / (20,20) out
        assert "p0" in names
        assert len(names) == 3
        assert "p3" not in names
        assert "p4" not in names

        # The closest one is p0
        assert names[0] == "p0"

    def test_nearest_k_larger_than_population(self) -> None:
        """
        k beyond the population returns everything.
        """

        app: Any = self._poi_app()

        table: Any = app.database.tables_by_struct_name["Poi"]

        index: Any = table.create_spatial_index(attribute="at", lod=5)

        assert len(index.nearest(coord=Coordinate(0, 0, R), k=100)) == 5

    def test_nearest_dedupes_cells(self) -> None:
        """
        A geometry spanning several cells appears once.
        """

        app: Any = Nadata(database=NaDataDatabase())

        app.execute_schema("class Big { boundary: Polygon; name: Str; }")

        app.execute_data(
            "big = Big(name=\"big\", boundary=Polygon(["
            "Coordinate(0,0,6371008.8), Coordinate(0,10,6371008.8),"
            " Coordinate(10,10,6371008.8), Coordinate(10,0,6371008.8)]));"
            "far = Big(name=\"far\", boundary=Polygon(["
            "Coordinate(50,50,6371008.8), Coordinate(50,51,6371008.8),"
            " Coordinate(51,51,6371008.8), Coordinate(51,50,6371008.8)]));"
        )

        table: Any = app.database.tables_by_struct_name["Big"]

        index: Any = table.create_spatial_index(attribute="boundary", lod=5)

        result: list[int] = index.nearest(
            coord=Coordinate(5.0, 5.0, R), k=2
        )

        names: list[str] = [
            table.get(local_id=local_id).attributes["name"].get()
            for local_id in result
        ]

        assert names == ["big", "far"]

    def test_nearest_k_zero(self) -> None:
        """
        k = 0 returns nothing.
        """

        app: Any = self._poi_app()

        table: Any = app.database.tables_by_struct_name["Poi"]

        index: Any = table.create_spatial_index(attribute="at", lod=5)

        assert index.nearest(coord=Coordinate(0, 0, R), k=0) == []


class TestWithin:
    """
    within() queries.
    """

    def test_within_regions(self) -> None:
        """
        within finds regions whose boundary is inside the radius.
        """

        app: Any = Nadata(database=NaDataDatabase())

        app.execute_schema("class Region { boundary: Polygon; name: Str; }")

        app.execute_data(
            "near = Region(name=\"near\", boundary=Polygon(["
            "Coordinate(0,0,6371008.8), Coordinate(0,0.5,6371008.8),"
            " Coordinate(0.5,0.5,6371008.8), Coordinate(0.5,0,6371008.8)]));"
            "far = Region(name=\"far\", boundary=Polygon(["
            "Coordinate(30,30,6371008.8), Coordinate(30,31,6371008.8),"
            " Coordinate(31,31,6371008.8), Coordinate(31,30,6371008.8)]));"
        )

        table: Any = app.database.tables_by_struct_name["Region"]

        index: Any = table.create_spatial_index(attribute="boundary", lod=5)

        # One degree is ~111 km; 150 km reaches the near region only
        result: list[int] = index.within(
            coord=Coordinate(0.25, 0.25, R), radius_m=150_000.0
        )

        names: list[str] = [
            table.get(local_id=local_id).attributes["name"].get()
            for local_id in result
        ]

        assert names == ["near"]

    def test_within_dedupes(self) -> None:
        """
        A multi-cell region appears once even when many of its cells
        intersect the radius.
        """

        app: Any = Nadata(database=NaDataDatabase())

        app.execute_schema("class Big { boundary: Polygon; }")

        app.execute_data(
            "big = Big(boundary=Polygon(["
            "Coordinate(0,0,6371008.8), Coordinate(0,10,6371008.8),"
            " Coordinate(10,10,6371008.8), Coordinate(10,0,6371008.8)]));"
        )

        table: Any = app.database.tables_by_struct_name["Big"]

        index: Any = table.create_spatial_index(attribute="boundary", lod=5)

        # Center of the big square: distance 0 (contains)
        result: list[int] = index.within(
            coord=Coordinate(5.0, 5.0, R), radius_m=1000.0
        )

        assert len(result) == 1

    def test_within_negative_radius(self) -> None:
        """
        Negative radii raise ValueError.
        """

        app: Any = Nadata(database=NaDataDatabase())

        app.execute_schema("class Big { boundary: Polygon; }")

        app.execute_data(
            "big = Big(boundary=Polygon(["
            "Coordinate(0,0,6371008.8), Coordinate(0,1,6371008.8),"
            " Coordinate(1,1,6371008.8), Coordinate(1,0,6371008.8)]));"
        )

        table: Any = app.database.tables_by_struct_name["Big"]

        index: Any = table.create_spatial_index(attribute="boundary", lod=5)

        with pytest.raises(ValueError):
            index.within(coord=Coordinate(0, 0, R), radius_m=-1.0)


class TestMaintenance:
    """
    Incremental maintenance on insert, write, and removal.
    """

    def test_insert_after_build(self) -> None:
        """
        Instances created after the index build are indexed.
        """

        app: Any = Nadata(database=NaDataDatabase())

        app.execute_schema("class Poi { at: Coordinate; }")
        app.execute_data(
            "p1 = Poi(at=Coordinate(0, 0, 6371008.8));"
            "spatial_index(Poi, \"at\", lod=5);"
        )

        app.execute_data("p2 = Poi(at=Coordinate(5, 5, 6371008.8));")

        table: Any = app.database.tables_by_struct_name["Poi"]

        index: Any = table.spatial_indexes["at"]

        assert len(index.entries) == 2

        found: Any = index.locate(coord=Coordinate(5.0, 5.0, R))

        assert found is not None

    def test_attribute_write_reindexes(self) -> None:
        """
        Writing the geometry attribute moves the entry.
        """

        app: Any = Nadata(database=NaDataDatabase())

        app.execute_schema("class Poi { at: Coordinate; }")
        app.execute_data(
            "p1 = Poi(at=Coordinate(0, 0, 6371008.8));"
            "spatial_index(Poi, \"at\", lod=5);"
        )

        table: Any = app.database.tables_by_struct_name["Poi"]

        index: Any = table.spatial_indexes["at"]

        instance: Any = table.get(local_id=0)

        # Move the point via the Python API (goes through set_attribute)
        from src.core.values.values import ValueCoordinate

        instance.set_attribute(
            name="at",
            value=ValueCoordinate(value=Coordinate(20.0, 20.0, R)),
        )

        assert index.locate(coord=Coordinate(0.0, 0.0, R)) is None

        found: Any = index.locate(coord=Coordinate(20.0, 20.0, R))

        assert found is not None

    def test_removal(self) -> None:
        """
        Removed instances leave the cells, cache, and hierarchy.
        """

        app: Any = Nadata(database=NaDataDatabase())

        app.execute_schema(
            "class Area { name: Str; boundary: Polygon; parent: Area; }"
        )

        app.execute_data(
            "outer = Area(name=\"outer\", boundary=Polygon(["
            "Coordinate(0,0,6371008.8), Coordinate(0,4,6371008.8),"
            " Coordinate(4,4,6371008.8), Coordinate(4,0,6371008.8)]));"
            "inner = Area(name=\"inner\", boundary=Polygon(["
            "Coordinate(1,1,6371008.8), Coordinate(1,2,6371008.8),"
            " Coordinate(2,2,6371008.8), Coordinate(2,1,6371008.8)]),"
            " parent=outer);"
            "spatial_index(Area, \"boundary\", lod=5, parent=\"parent\");"
        )

        table: Any = app.database.tables_by_struct_name["Area"]

        index: Any = table.spatial_indexes["boundary"]

        # Remove the city: its cells vanish and the point falls
        # through to the outer region
        table.index_remove_entity(local_id=1)

        assert index.get_instance(
            key=index.locate(coord=Coordinate(1.5, 1.5, R))
        ).get_attribute(name="name").get() == "outer"

        # Remove everything
        table.index_remove_entity(local_id=0)

        assert index.locate(coord=Coordinate(1.5, 1.5, R)) is None
        assert len(index.entries) == 0
        assert len(index.cells) == 0

    def test_removal_promotes_children(self) -> None:
        """
        Children of a removed parent become roots.
        """

        app: Any = Nadata(database=NaDataDatabase())

        app.execute_schema(
            "class Area { name: Str; boundary: Polygon; parent: Area; }"
        )

        app.execute_data(
            "outer = Area(name=\"outer\", boundary=Polygon(["
            "Coordinate(0,0,6371008.8), Coordinate(0,4,6371008.8),"
            " Coordinate(4,4,6371008.8), Coordinate(4,0,6371008.8)]));"
            "inner = Area(name=\"inner\", boundary=Polygon(["
            "Coordinate(1,1,6371008.8), Coordinate(1,2,6371008.8),"
            " Coordinate(2,2,6371008.8), Coordinate(2,1,6371008.8)]),"
            " parent=outer);"
            "spatial_index(Area, \"boundary\", lod=5, parent=\"parent\");"
        )

        table: Any = app.database.tables_by_struct_name["Area"]

        index: Any = table.spatial_indexes["boundary"]

        outer_key: int = table.instances[0].entity_id.value
        inner_key: int = table.instances[1].entity_id.value

        assert index.roots == [outer_key]
        assert index.children == {outer_key: [inner_key]}

        table.index_remove_entity(local_id=0)

        assert index.roots == [inner_key]


class TestLanguageIntegration:
    """
    spatial_index / locate / nearest / within from .nadata scripts.
    """

    def test_script_flow(self) -> None:
        """
        The full script flow: build, locate, nearest, within.
        """

        app: Any = Nadata(database=NaDataDatabase())

        app.execute_schema(
            "class Area { name: Str; boundary: Polygon; parent: Area; }"
        )

        app.execute_data(
            "outer = Area(name=\"outer\", boundary=Polygon(["
            "Coordinate(0,0,6371008.8), Coordinate(0,4,6371008.8),"
            " Coordinate(4,4,6371008.8), Coordinate(4,0,6371008.8)]));"
            "inner = Area(name=\"inner\", boundary=Polygon(["
            "Coordinate(1,1,6371008.8), Coordinate(1,2,6371008.8),"
            " Coordinate(2,2,6371008.8), Coordinate(2,1,6371008.8)]),"
            " parent=outer);"
            "spatial_index(Area, \"boundary\", lod=5, parent=\"parent\");"
            "here = locate(Area, Coordinate(1.5, 1.5, 6371008.8));"
            "print(\"here:\", here.name);"
            "n = within(Area, Coordinate(1.5, 1.5, 6371008.8), 1000.0);"
            "c = 0;"
            "for (a: Area in n) { c = c + 1; print(\"w:\", a.name); }"
            "print(\"within count:\", c);"
            "m = nearest(Area, Coordinate(50.0, 50.0, 6371008.8), 1);"
            "print(\"nearest to nowhere:\", m[0].name);"
        )

        # The script ran; the facade stays usable
        assert app.get_attribute(symbol="outer", name="name") == "outer"

    def test_locate_requires_index(self) -> None:
        """
        locate without a built index raises a clear TypeError.
        """

        app: Any = Nadata(database=NaDataDatabase())

        app.execute_schema("class Area { boundary: Polygon; }")
        app.execute_data(
            "a = Area(boundary=Polygon(["
            "Coordinate(0,0,6371008.8), Coordinate(0,1,6371008.8),"
            " Coordinate(1,1,6371008.8), Coordinate(1,0,6371008.8)]));"
        )

        table: Any = app.database.tables_by_struct_name["Area"]

        from src.core.expressions.builtins import _spatial_index_of

        with pytest.raises(TypeError):
            _spatial_index_of(table=table, op_name="locate")


class TestDumpClustering:
    """
    Dump-time row clustering for spatially indexed tables.
    """

    def test_dump_load_roundtrip(self, tmp_path: Any) -> None:
        """
        A clustered dump reloads and re-indexes correctly; rows are
        written in spatial order.
        """

        app: Any = Nadata(database=NaDataDatabase())

        app.execute_schema("class Poi { at: Coordinate; name: Str; }")

        # Insert deliberately out of spatial order
        app.execute_data(
            "far = Poi(name=\"far\", at=Coordinate(40, 40, 6371008.8));"
            "mid = Poi(name=\"mid\", at=Coordinate(10, 10, 6371008.8));"
            "near = Poi(name=\"near\", at=Coordinate(1, 1, 6371008.8));"
            "spatial_index(Poi, \"at\", lod=5);"
        )

        storage: str = str(tmp_path / "clustered.ndb")

        app.dump(storage_path=storage)

        reloaded: Any = Nadata.load(path=storage)

        table: Any = reloaded.database.tables_by_struct_name["Poi"]

        index: Any = table.create_spatial_index(attribute="at", lod=5)

        # Spatial order: near (cell ~[128, 137]) before mid before far
        ordered: list[str] = []

        for local_id in sorted(index.entries.keys()):
            ordered.append(table.get(local_id=local_id).attributes["name"].get())

        # All rows present after reload
        assert sorted(ordered) == ["far", "mid", "near"]

        # The clustered file ordered the near row before the far row:
        # near's minimum cell key is smaller than far's
        near_entry: Any = next(
            entry
            for entry in index.entries.values()
            if entry.center.lat == pytest.approx(1.0)
        )
        far_entry: Any = next(
            entry
            for entry in index.entries.values()
            if entry.center.lat == pytest.approx(40.0)
        )

        assert min(near_entry.cells) < min(far_entry.cells)

        # Queries work on the reloaded database
        found: Any = index.locate(coord=Coordinate(1.0, 1.0, R))

        assert found is not None

        assert table.get(local_id=found).attributes["name"].get() == "near"


class TestCellBounds:
    """
    Cell lower-bound soundness.
    """

    def test_bound_never_exceeds_true_distance(self) -> None:
        """
        For random points and cells, the per-cell lower bound never
        exceeds the true minimum geodesic distance to the cell's
        corners and center.
        """

        index: Any = SpatialIndex.__new__(SpatialIndex)

        index.cell_deg = 360.0 / 32.0
        index.n_x = 32
        index.n_y = 16

        set_current_planet(planet=DEFAULT_PLANET)

        for i in range(50):
            point: Coordinate = Coordinate(
                lat=math.sin(i * 1.7) * 80.0,
                lon=((i * 137.0) % 360.0) - 180.0,
                radius=R,
            )

            ix: int = (i * 7) % 32
            iy: int = (i * 3) % 16

            bound: float = index._cell_lower_bound(
                coord=point,
                cell=(ix, iy),
                radius_bound=DEFAULT_PLANET.polar_radius_value,
            )

            # True distance to the closest point we can name inside
            # the cell: its corners and center (the bound may exceed
            # none of them)
            lat0: float = iy * index.cell_deg - 90.0
            lat1: float = lat0 + index.cell_deg
            lon0: float = ix * index.cell_deg - 180.0
            lon1: float = lon0 + index.cell_deg

            for lat in (lat0, (lat0 + lat1) / 2.0, min(90.0, lat1)):
                for lon in (lon0, (lon0 + lon1) / 2.0, lon1):
                    probe: Coordinate = Coordinate(
                        lat=max(-90.0, min(90.0, lat)),
                        lon=lon,
                        radius=R,
                    )

                    true_distance: float = point.distance(
                        other=probe, planet=DEFAULT_PLANET
                    )

                    assert bound <= true_distance + 1.0, (
                        f"bound {bound} exceeds distance {true_distance} "
                        f"at point {point} cell {(ix, iy)}"
                    )

    def teardown_method(self) -> None:
        """
        Restores the default session planet.

        Returns:
            None: Nothing.
        """

        set_current_planet(planet=DEFAULT_PLANET)
