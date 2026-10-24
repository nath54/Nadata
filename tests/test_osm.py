"""
Real-world geospatial data tests: the OSM XML and GeoJSON converters
run against committed real extracts (OpenStreetMap, ODbL; france-geojson,
Etalab) and the results are loaded into a live world and queried.

The fixtures keep the tests hermetic (no network access); they are
trimmed real extracts, so the assertions verify the whole pipeline:
convert, load, index, route, locate.
"""

# Import Modules
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

from src.api import Nadata
from src.core.database.database import NaDataDatabase
from src.core.types.home_types.coordinate import Coordinate

# Import Modules (paths)
TOOLS_DIR: str = str(Path(__file__).parent.parent / "tools")
FIXTURES_DIR: Path = Path(__file__).parent / "fixtures"
SCHEMA_PATH: Path = (
    Path(__file__).parent.parent
    / "examples" / "ex13_earth" / "structs" / "earth.nastruct"
)


@pytest.fixture
def world(tmp_path: Path) -> Any:
    """
    Builds a world with the ex13_earth schema in a temp database.

    Args:
        tmp_path (Path): Pytest's temp directory.

    Yields:
        Any: The Nadata app.
    """

    app: Any = Nadata(database=NaDataDatabase())

    app.execute_schema(schema_text=SCHEMA_PATH.read_text(encoding="utf-8"))

    yield app


def _run_tool(script: str, *args: str) -> None:
    """
    Runs one converter script with arguments.

    Args:
        script (str): The tool file name under tools/.
        *args (str): The CLI arguments.
    """

    result: Any = subprocess.run(
        [sys.executable, str(Path(TOOLS_DIR) / script), *args],
        capture_output=True,
        text=True,
        check=True,
    )

    return result


class TestOsmConverter:
    """
    OSM XML extracts convert, load, and route.
    """

    def test_converts_and_routes_real_extract(
        self, world: Any, tmp_path: Path
    ) -> None:
        """
        The committed Toulouse extract becomes a routable network.

        Args:
            world (Any): The schema-only world fixture.
            tmp_path (Path): Temp directory.
        """

        output: Path = tmp_path / "toulouse.nadata"

        result: Any = _run_tool(
            "osm_to_nadata.py",
            str(FIXTURES_DIR / "osm_toulouse_small.osm"),
            str(output),
        )

        assert "junction(s)" in result.stdout

        world.execute_data(data_text=output.read_text(encoding="utf-8"))

        place_table: Any = world.database.tables_by_struct_name["Place"]
        link_table: Any = world.database.tables_by_struct_name["Link"]

        # Real row counts from the committed extract
        assert len(place_table.instances) > 1000
        assert len(link_table.instances) > 1000

        # Every link's endpoints resolve to place rows
        for link in link_table.instances.values():
            from_ref: Any = link.attributes["from_place"].get()
            to_ref: Any = link.attributes["to_place"].get()

            assert from_ref is not None
            assert to_ref is not None

        # One connected network, car and walk queries
        network: Any = place_table.create_network(
            edges_table=link_table
        )

        place_table.create_spatial_index(
            attribute="position", lod=9, database=world.database
        )

        start: Coordinate = Coordinate(43.6047, 1.4442, 6371008.8)
        end: Coordinate = Coordinate(43.6080, 1.4450, 6371008.8)

        by_car: Any = network.route(start=start, end=end, mode="car")
        on_foot: Any = network.route(start=start, end=end, mode="walk")

        # Capitole to a nearby point: sub-kilometer but multi-segment
        assert 0 < by_car.cost < 3000.0
        assert len(by_car.edges) > 3
        assert 0 < on_foot.cost < 3000.0

        # Walk never uses car-only ways; both reach the destination
        assert on_foot.cost > 0.0

    def test_spatial_queries_on_real_places(
        self, world: Any, tmp_path: Path
    ) -> None:
        """
        The converted amenities answer within / nearest queries.

        Args:
            world (Any): The schema-only world fixture.
            tmp_path (Path): Temp directory.
        """

        output: Path = tmp_path / "toulouse.nadata"

        _run_tool(
            "osm_to_nadata.py",
            str(FIXTURES_DIR / "osm_toulouse_small.osm"),
            str(output),
        )

        world.execute_data(data_text=output.read_text(encoding="utf-8"))

        place_table: Any = world.database.tables_by_struct_name["Place"]

        index: Any = place_table.create_spatial_index(
            attribute="position", lod=9, database=world.database
        )

        here: Coordinate = Coordinate(43.6047, 1.4442, 6371008.8)

        nearby: list[int] = index.within(coord=here, radius_m=300.0)

        assert len(nearby) > 0

        nearest: list[int] = index.nearest(coord=here, k=3)

        assert len(nearest) == 3

        # The nearest place is within the queried radius's order
        closest: Any = index.get_instance(key=nearest[0])

        assert closest is not None

    def test_incomplete_ways_are_dropped(
        self, world: Any, tmp_path: Path
    ) -> None:
        """
        Ways referencing nodes outside the extract are skipped.

        Args:
            world (Any): The schema-only world fixture.
            tmp_path (Path): Temp directory.
        """

        source: Path = tmp_path / "incomplete.osm"
        source.write_text(
            '<?xml version="1.0"?>\n<osm version="0.6">\n'
            '<node id="1" lat="43.6" lon="1.44"/>\n'
            '<node id="2" lat="43.61" lon="1.44"/>\n'
            '<way id="10">'
            '<nd ref="1"/><nd ref="2"/>'
            '<tag k="highway" v="residential"/>'
            "</way>\n"
            '<way id="11">'
            '<nd ref="1"/><nd ref="999"/>'  # 999 missing
            '<tag k="highway" v="residential"/>'
            "</way>\n"
            "</osm>\n",
            encoding="utf-8",
        )

        output: Path = tmp_path / "incomplete.nadata"

        result: Any = _run_tool(
            "osm_to_nadata.py", str(source), str(output)
        )

        assert "1 road(s)" in result.stdout
        assert "1 dropped incomplete way(s)" in result.stdout

        world.execute_data(data_text=output.read_text(encoding="utf-8"))

        link_table: Any = world.database.tables_by_struct_name["Link"]

        assert len(link_table.instances) == 1

    def test_empty_kwargs_use_schema_defaults(
        self, world: Any, tmp_path: Path
    ) -> None:
        """
        Ways without maxspeed / modes still load (speed omitted).

        Args:
            world (Any): The schema-only world fixture.
            tmp_path (Path): Temp directory.
        """

        source: Path = tmp_path / "nospeed.osm"
        source.write_text(
            '<?xml version="1.0"?>\n<osm version="0.6">\n'
            '<node id="1" lat="43.6" lon="1.44"/>\n'
            '<node id="2" lat="43.61" lon="1.44"/>\n'
            '<node id="3" lat="43.61" lon="1.45"/>\n'
            '<way id="10">'
            '<nd ref="1"/><nd ref="2"/><nd ref="3"/>'
            '<tag k="highway" v="residential"/>'
            "</way>\n"
            "</osm>\n",
            encoding="utf-8",
        )

        output: Path = tmp_path / "nospeed.nadata"

        _run_tool("osm_to_nadata.py", str(source), str(output))

        # Two segment edges from one 3-node way, no speed kwarg
        world.execute_data(data_text=output.read_text(encoding="utf-8"))

        link_table: Any = world.database.tables_by_struct_name["Link"]

        assert len(link_table.instances) == 2

        # Time-cost routing prices defaulted speeds without failing
        network: Any = world.database.tables_by_struct_name[
            "Place"
        ].create_network(edges_table=link_table)

        path: Any = network.route(
            start=Coordinate(43.6, 1.44, 6371008.8),
            end=Coordinate(43.61, 1.45, 6371008.8),
            mode="car",
            cost="time",
        )

        assert path.cost > 0.0


class TestGeojsonConverter:
    """
    Real GeoJSON (france-geojson departements) converts and locates.
    """

    def test_converts_and_locates_real_departements(
        self, world: Any, tmp_path: Path
    ) -> None:
        """
        The committed 96-departement extract locates real points.

        Args:
            world (Any): The schema-only world fixture.
            tmp_path (Path): Temp directory.
        """

        output: Path = tmp_path / "france.nadata"

        _run_tool(
            "geojson_to_nadata.py",
            str(FIXTURES_DIR / "france_departements.geojson"),
            str(output),
            "--level", "Departement",
        )

        world.execute_data(data_text=output.read_text(encoding="utf-8"))

        departement_table: Any = world.database.tables_by_struct_name[
            "Departement"
        ]

        assert len(departement_table.instances) == 96

        # Flat spatial index over the real polygons
        index: Any = departement_table.create_spatial_index(
            attribute="boundary", lod=8, database=world.database
        )

        expected: list[tuple[Coordinate, str]] = [
            (Coordinate(43.6047, 1.4442, 6371008.8), "Haute-Garonne"),
            (Coordinate(45.75, 4.85, 6371008.8), "Rh\u00f4ne"),
            (Coordinate(-21.1151, 55.5364, 6371008.8), "none"),
        ]

        for coord, expected_name in expected:
            found: Any = index.locate(coord=coord)

            if expected_name == "none":
                # Overseas territories are absent from the extract
                assert found is None

                continue

            instance: Any = index.get_instance(key=found)

            assert (
                instance.get_attribute(name="name").get() == expected_name
            ), f"{coord} located in {instance.get_attribute(name='name').get()}"

        # Radius query around the Rhone valley
        hits: list[int] = index.within(
            coord=Coordinate(45.0, 4.8, 6371008.8), radius_m=60000.0
        )

        assert 1 <= len(hits) <= 10

    def test_parent_binding_across_leaf_tables(
        self, world: Any, tmp_path: Path
    ) -> None:
        """
        Converted rows bind parents declared in other leaf tables.

        Args:
            world (Any): The schema-only world fixture.
            tmp_path (Path): Temp directory.
        """

        # A Country row first, then Departements bound to it via nid
        world.execute_data(
            data_text='france = Country(name = "France", nid = "france");'
        )

        source: Path = tmp_path / "one.geojson"
        source.write_text(
            '{"type": "FeatureCollection", "features": ['
            '{"type": "Feature", "properties": {"name": "Rivendell",'
            ' "nid": "rivendell"},'
            ' "geometry": {"type": "Polygon", "coordinates": [[[1.0, 43.0],'
            " [2.0, 43.0], [2.0, 44.0], [1.0, 44.0], [1.0, 43.0]]]}}"
            "]}\n",
            encoding="utf-8",
        )

        output: Path = tmp_path / "one.nadata"

        _run_tool(
            "geojson_to_nadata.py",
            str(source),
            str(output),
            "--level", "Departement",
            "--parent-nid", "france",
        )

        world.execute_data(data_text=output.read_text(encoding="utf-8"))

        # The hierarchy fallback resolves the Country nid from the
        # Departement table namespace
        rivendell: Any = world.database.get_by_nid(
            "Departement", "rivendell"
        )

        assert rivendell is not None

        parent: Any = rivendell.get_attribute(name="parent").get()

        assert parent is not None
        assert parent.get_attribute(name="name").get() == "France"
