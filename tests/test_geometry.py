"""
Tests for the geometry layer: the Polyline and Polygon value types,
their measurement and predicate members, shape constructor sugar,
antimeridian handling, language integration, and persistence.
"""

# Import Modules
import math
from typing import Any

import pytest

from src.api import Nadata
from src.core.database.database import NaDataDatabase
from src.core.database.persistence.binary_cells import decode_value, encode_value
from src.core.database.persistence.serialization import DecodeContext, ObjectCodec
from src.core.types.home_types.coordinate import Coordinate, destination
from src.core.types.home_types.geometry import (
    Polygon,
    Polyline,
    area,
    circle,
    contains,
    ellipse,
    perimeter,
    regular_polygon,
)
from src.core.types.home_types.planet import (
    DEFAULT_PLANET,
    EARTH_MEAN_RADIUS,
    Planet,
    set_current_planet,
)
from src.core.types.types import TypePolygon, TypePolyline
from src.core.values.values import make_value

R: float = EARTH_MEAN_RADIUS

# One degree of arc on the default sphere
DEG: float = math.pi * R / 180.0

SQUARE: Polygon = Polygon(
    vertices=[
        Coordinate(0.0, 0.0, R),
        Coordinate(0.0, 1.0, R),
        Coordinate(1.0, 1.0, R),
        Coordinate(1.0, 0.0, R),
    ]
)

MERIDIAN_PATH: Polyline = Polyline(
    vertices=[Coordinate(0.0, 0.0, R), Coordinate(1.0, 0.0, R), Coordinate(2.0, 0.0, R)]
)



class TestPolylineConstruction:
    """
    Validation on construction.
    """

    def test_vertices_stored(self) -> None:
        """
        Vertices are stored as given.
        """

        pl: Polyline = Polyline(
            vertices=[Coordinate(0, 0, 1), Coordinate(1, 1, 1)]
        )

        assert pl.num_vertices() == 2

    def test_empty_allowed(self) -> None:
        """
        The empty polyline is valid.
        """

        assert Polyline(vertices=[]).num_vertices() == 0

    def test_tuple_accepted(self) -> None:
        """
        Tuples of coordinates are accepted.
        """

        assert Polyline(
            vertices=(Coordinate(0, 0, 1), Coordinate(1, 1, 1))
        ).num_vertices() == 2

    def test_non_list_rejected(self) -> None:
        """
        Non-list vertex arguments raise TypeError.
        """

        with pytest.raises(TypeError):
            Polyline(vertices="path")  # type: ignore[arg-type]

    def test_non_coordinate_elements_rejected(self) -> None:
        """
        Non-Coordinate elements raise TypeError.
        """

        with pytest.raises(TypeError):
            Polyline(vertices=[Coordinate(0, 0, 1), "x"])  # type: ignore[list-item]

    def test_equality(self) -> None:
        """
        Equality compares vertex lists.
        """

        a: Polyline = Polyline(vertices=[Coordinate(0, 0, 1)])
        b: Polyline = Polyline(vertices=[Coordinate(0, 0, 1)])

        assert a == b
        assert a != Polyline(vertices=[])
        assert a != "Polyline"


class TestPolylineLength:
    """
    Geodesic length.
    """

    def test_empty_and_single(self) -> None:
        """
        Empty and single-vertex paths have zero length.
        """

        assert Polyline(vertices=[]).length() == 0.0
        assert Polyline(vertices=[Coordinate(0, 0, R)]).length() == 0.0

    def test_meridian_length(self) -> None:
        """
        A 2-degree meridian path measures 2 degrees of arc.
        """

        assert MERIDIAN_PATH.length() == pytest.approx(2.0 * DEG)

    def test_length_ignores_radius_differences(self) -> None:
        """
        Path length is surface distance; radii of the vertices do not
        stretch the path.
        """

        pl: Polyline = Polyline(
            vertices=[Coordinate(0, 0, R), Coordinate(0, 1, 2 * R)]
        )

        assert pl.length() == pytest.approx(DEG)


class TestPolylinePoints:
    """
    Point lookup along the path.
    """

    def test_point_at_endpoints(self) -> None:
        """
        t = 0 and t = 1 land on the first and last vertices.
        """

        start: Coordinate = MERIDIAN_PATH.point_at(t=0.0)
        end: Coordinate = MERIDIAN_PATH.point_at(t=1.0)

        assert start.lat == pytest.approx(0.0, abs=1e-9)
        assert start.lon == pytest.approx(0.0, abs=1e-9)
        assert end.lat == pytest.approx(2.0, abs=1e-9)
        assert end.lon == pytest.approx(0.0, abs=1e-9)

    def test_point_at_middle(self) -> None:
        """
        t = 0.5 on a 2-degree path is 1 degree north.
        """

        mid: Coordinate = MERIDIAN_PATH.point_at(t=0.5)

        assert mid.lat == pytest.approx(1.0, abs=1e-6)

    def test_point_at_distance(self) -> None:
        """
        Distance-based lookup matches the fraction-based one.
        """

        by_fraction: Coordinate = MERIDIAN_PATH.point_at(t=0.25)
        by_distance: Coordinate = MERIDIAN_PATH.point_at_distance(
            distance_m=0.5 * DEG
        )

        assert by_distance.lat == pytest.approx(by_fraction.lat, abs=1e-9)
        assert by_distance.lon == pytest.approx(by_fraction.lon, abs=1e-9)
        assert by_distance.radius == by_fraction.radius

    def test_point_at_rejects_out_of_range(self) -> None:
        """
        t outside [0, 1] raises ValueError.
        """

        with pytest.raises(ValueError):
            MERIDIAN_PATH.point_at(t=1.5)

    def test_point_at_distance_beyond_length(self) -> None:
        """
        Distances beyond the path length raise ValueError.
        """

        with pytest.raises(ValueError):
            MERIDIAN_PATH.point_at_distance(distance_m=3.0 * DEG)

    def test_point_at_negative_distance(self) -> None:
        """
        Negative distances raise ValueError.
        """

        with pytest.raises(ValueError):
            MERIDIAN_PATH.point_at_distance(distance_m=-1.0)

    def test_empty_path_rejects_points(self) -> None:
        """
        Point lookups on the empty path raise ValueError.
        """

        with pytest.raises(ValueError):
            Polyline(vertices=[]).point_at(t=0.5)


class TestPolylineDistanceTo:
    """
    Distance from a coordinate to the path.
    """

    def test_projection_within_segment(self) -> None:
        """
        A point beside the middle of the path measures to the
        projected point.
        """

        # 1 degree east of the path midpoint (lat 1, lon 0)
        point: Coordinate = destination(
            Coordinate(1.0, 0.0, R), 90.0, DEG
        )

        assert MERIDIAN_PATH.distance_to(point=point) == pytest.approx(DEG)

    def test_projection_beyond_end(self) -> None:
        """
        Points beyond the path end measure to the last vertex.
        """

        # 1 degree east of the path end (lat 2, lon 0)
        point: Coordinate = destination(
            Coordinate(2.0, 0.0, R), 90.0, DEG
        )

        expected: float = Coordinate(2.0, 0.0, R).distance(
            other=point, planet=DEFAULT_PLANET
        )

        assert MERIDIAN_PATH.distance_to(point=point) == pytest.approx(
            expected, rel=1e-6
        )

    def test_on_path_is_zero(self) -> None:
        """
        A vertex of the path measures zero.
        """

        assert MERIDIAN_PATH.distance_to(
            point=Coordinate(1.0, 0.0, R)
        ) == pytest.approx(0.0)

    def test_pole_to_meridian_segment(self) -> None:
        """
        The pole to a meridian segment measures to the nearest segment
        endpoint (the pole's projection falls outside the short arc):
        88 degrees from the pole to latitude 2.
        """

        point: Coordinate = Coordinate(90.0, 0.5, R)

        assert MERIDIAN_PATH.distance_to(point=point) == pytest.approx(
            math.radians(88.0) * R, rel=1e-6
        )

    def test_empty_path_rejects(self) -> None:
        """
        Distance to the empty path raises ValueError.
        """

        with pytest.raises(ValueError):
            Polyline(vertices=[]).distance_to(point=Coordinate(0, 0, R))


class TestPolygonConstruction:
    """
    Ring validation.
    """

    def test_square_stored(self) -> None:
        """
        Vertices are stored as given.
        """

        assert SQUARE.num_vertices() == 4

    def test_closing_vertex_stripped(self) -> None:
        """
        An explicit closing vertex is removed.
        """

        closed: Polygon = Polygon(
            vertices=[
                Coordinate(0, 0, R),
                Coordinate(0, 1, R),
                Coordinate(1, 1, R),
                Coordinate(1, 0, R),
                Coordinate(0, 0, R),
            ]
        )

        assert closed.num_vertices() == 4
        assert closed == SQUARE

    def test_degenerate_counts_rejected(self) -> None:
        """
        Rings of 1 or 2 vertices raise ValueError.
        """

        with pytest.raises(ValueError):
            Polygon(vertices=[Coordinate(0, 0, R)])

        with pytest.raises(ValueError):
            Polygon(vertices=[Coordinate(0, 0, R), Coordinate(1, 1, R)])

    def test_empty_allowed(self) -> None:
        """
        The empty polygon is valid.
        """

        assert Polygon(vertices=[]).num_vertices() == 0


class TestPolygonArea:
    """
    Spherical-excess areas.
    """

    def test_empty_is_zero(self) -> None:
        """
        The empty polygon has zero area.
        """

        assert Polygon(vertices=[]).area() == 0.0

    def test_one_degree_square(self) -> None:
        """
        A 1-degree square at the equator is close to the planar
        approximation (111.19 km per side).
        """

        assert SQUARE.area() == pytest.approx(DEG * DEG, rel=1e-3)

    def test_circle_area(self) -> None:
        """
        A geodesic circle of radius r approaches pi r^2 as segments
        grow.
        """

        disk: Polygon = circle(
            center=Coordinate(0.0, 0.0, R), radius_m=100_000.0, segments=360
        )

        assert disk.area() == pytest.approx(math.pi * 100_000.0**2, rel=1e-3)

    def test_area_scales_with_radius_squared(self) -> None:
        """
        Doubling the planet radius quadruples the area of the same
        angular ring.
        """

        small: Planet = Planet(1000.0)

        ring_a: Polygon = Polygon(
            vertices=[
                Coordinate(0, 0, 1000.0),
                Coordinate(0, 1, 1000.0),
                Coordinate(1, 1, 1000.0),
                Coordinate(1, 0, 1000.0),
            ]
        )

        ring_b: Polygon = Polygon(
            vertices=[
                Coordinate(0, 0, 2000.0),
                Coordinate(0, 1, 2000.0),
                Coordinate(1, 1, 2000.0),
                Coordinate(1, 0, 2000.0),
            ]
        )

        assert ring_a.area(planet=small) * 4.0 == pytest.approx(
            ring_b.area(planet=Planet(2000.0)), rel=1e-9
        )

    def test_hemisphere_cap(self) -> None:
        """
        A circle of quarter-circumference radius covers a quarter of
        the sphere (2 pi R^2), within polygonal approximation error.
        """

        disk: Polygon = circle(
            center=Coordinate(0.0, 0.0, R),
            radius_m=math.pi * R / 2.0,
            segments=512,
        )

        assert disk.area() == pytest.approx(2.0 * math.pi * R**2, rel=1e-2)


class TestPolygonPerimeter:
    """
    Boundary lengths.
    """

    def test_square_perimeter(self) -> None:
        """
        The 1-degree square has a 4-degree perimeter at the equator
        (the polar edges are 1 degree; the equatorial edges are
        cos-corrected... actually at the equator all edges are 1
        degree, but the northern edge sits at latitude 1).
        """

        # Equatorial and southern edges: exactly 1 degree each.
        # The northern edge at latitude 1 is slightly shorter.
        assert SQUARE.perimeter() == pytest.approx(4.0 * DEG, rel=1e-4)

    def test_circle_perimeter(self) -> None:
        """
        A geodesic circle approaches 2 pi r as segments grow.
        """

        disk: Polygon = circle(
            center=Coordinate(0.0, 0.0, R), radius_m=100_000.0, segments=360
        )

        assert disk.perimeter() == pytest.approx(
            2.0 * math.pi * 100_000.0, rel=1e-3
        )

    def test_empty_is_zero(self) -> None:
        """
        The empty polygon has zero perimeter.
        """

        assert Polygon(vertices=[]).perimeter() == 0.0


class TestPolygonContains:
    """
    Point-in-polygon.
    """

    def test_inside(self) -> None:
        """
        The center of the square is inside.
        """

        assert contains(SQUARE, Coordinate(0.5, 0.5, R)) is True

    def test_outside_sides(self) -> None:
        """
        Points beyond each side are outside.
        """

        assert contains(SQUARE, Coordinate(-1.0, 0.5, R)) is False
        assert contains(SQUARE, Coordinate(2.0, 0.5, R)) is False
        assert contains(SQUARE, Coordinate(0.5, -1.0, R)) is False
        assert contains(SQUARE, Coordinate(0.5, 2.0, R)) is False

    def test_antimeridian_ring(self) -> None:
        """
        A ring crossing the antimeridian contains points on the other
        side of the ±180 line and not points far away.
        """

        ring: Polygon = Polygon(
            vertices=[
                Coordinate(0.0, 179.0, R),
                Coordinate(0.0, -179.0, R),
                Coordinate(2.0, -179.0, R),
                Coordinate(2.0, 179.0, R),
            ]
        )

        # 180.5 degrees east is stored as -179.5
        assert ring.contains(point=Coordinate(1.0, -179.5, R)) is True
        assert ring.contains(point=Coordinate(1.0, 179.5, R)) is True
        assert ring.contains(point=Coordinate(1.0, 178.0, R)) is False
        assert ring.contains(point=Coordinate(5.0, -179.5, R)) is False

    def test_concave_ring(self) -> None:
        """
        Concave shapes exclude points inside their bounding box but
        outside the ring.
        """

        # U shape (all on a small patch near the equator)
        ring: Polygon = Polygon(
            vertices=[
                Coordinate(0.0, 0.0, R),
                Coordinate(0.0, 3.0, R),
                Coordinate(0.5, 3.0, R),
                Coordinate(0.5, 1.0, R),
                Coordinate(1.0, 1.0, R),
                Coordinate(1.0, 3.0, R),
                Coordinate(1.5, 3.0, R),
                Coordinate(1.5, 0.0, R),
            ]
        )

        # Inside the left arm, the base, the right arm
        assert ring.contains(point=Coordinate(0.25, 0.5, R)) is True
        assert ring.contains(point=Coordinate(0.75, 0.5, R)) is True
        assert ring.contains(point=Coordinate(1.25, 0.5, R)) is True
        # Inside the notch of the U
        assert ring.contains(point=Coordinate(0.75, 2.0, R)) is False

    def test_empty_rejects(self) -> None:
        """
        The empty polygon contains nothing and says so.
        """

        with pytest.raises(ValueError):
            Polygon(vertices=[]).contains(point=Coordinate(0, 0, R))


class TestPolygonCentroidAndRadii:
    """
    Centroid and fast-path radii.
    """

    def test_centroid_of_square(self) -> None:
        """
        The square's vertex centroid is its middle.
        """

        center: Coordinate = SQUARE.centroid()

        # Vertex-direction mean: within a few meters of the planar
        # middle at this scale
        assert center.lat == pytest.approx(0.5, abs=1e-3)
        assert center.lon == pytest.approx(0.5, abs=1e-3)
        assert center.radius == pytest.approx(R)

    def test_circumradius_of_geodesic_circle(self) -> None:
        """
        A geodesic circle's circumradius equals its construction
        radius.
        """

        disk: Polygon = circle(
            center=Coordinate(0.0, 0.0, R), radius_m=100_000.0, segments=64
        )

        assert disk.circumradius() == pytest.approx(100_000.0, rel=1e-4)

    def test_inradius_of_geodesic_circle(self) -> None:
        """
        A geodesic circle's inradius equals its construction radius.
        """

        disk: Polygon = circle(
            center=Coordinate(0.0, 0.0, R), radius_m=100_000.0, segments=64
        )

        # Inscribed polygon: inradius is r * cos(pi / segments)
        assert disk.inradius() == pytest.approx(
            100_000.0 * math.cos(math.pi / 64.0), rel=1e-4
        )

    def test_inradius_of_square(self) -> None:
        """
        The square's inradius is about half a degree (distance from
        centroid to the nearest edge).
        """

        assert SQUARE.inradius() == pytest.approx(DEG / 2.0, rel=1e-3)

    def test_empty_rejects(self) -> None:
        """
        Radii and centroid on the empty polygon raise ValueError.
        """

        empty: Polygon = Polygon(vertices=[])

        with pytest.raises(ValueError):
            empty.centroid()

        with pytest.raises(ValueError):
            empty.circumradius()

        with pytest.raises(ValueError):
            empty.inradius()


class TestPolygonDistanceTo:
    """
    Distance from a coordinate to the polygon.
    """

    def test_inside_is_zero(self) -> None:
        """
        Points inside measure zero.
        """

        assert SQUARE.distance_to(point=Coordinate(0.5, 0.5, R)) == 0.0

    def test_beside_edge(self) -> None:
        """
        A point east of the square measures to the eastern edge.
        """

        # 1 degree east of the eastern edge midpoint
        point: Coordinate = destination(
            Coordinate(0.5, 1.0, R), 90.0, DEG
        )

        assert SQUARE.distance_to(point=point) == pytest.approx(DEG, rel=1e-6)

    def test_empty_rejects(self) -> None:
        """
        Distance to the empty polygon raises ValueError.
        """

        with pytest.raises(ValueError):
            Polygon(vertices=[]).distance_to(point=Coordinate(0, 0, R))


class TestShapeSugar:
    """
    circle / ellipse / regular_polygon constructors.
    """

    def test_circle_vertex_count(self) -> None:
        """
        The circle has the requested vertex count.
        """

        disk: Polygon = circle(
            center=Coordinate(0, 0, R), radius_m=1000.0, segments=8
        )

        assert disk.num_vertices() == 8

    def test_circle_vertices_on_radius(self) -> None:
        """
        Every circle vertex sits at the requested geodesic radius from
        the center.
        """

        center: Coordinate = Coordinate(10.0, 10.0, R)
        disk: Polygon = circle(center=center, radius_m=50_000.0, segments=16)

        for i in range(disk.num_vertices()):
            assert center.distance(other=disk.vertex(i), planet=DEFAULT_PLANET) == (
                pytest.approx(50_000.0, rel=1e-6)
            )

    def test_ellipse_axis_vertices(self) -> None:
        """
        The ellipse's vertices along and across the bearing sit at the
        two semi-axes.
        """

        center: Coordinate = Coordinate(10.0, 10.0, R)
        shape: Polygon = ellipse(
            center=center, radius_a_m=80_000.0, radius_b_m=40_000.0, segments=64
        )

        # Vertex 0 is along the bearing (0 degrees = north)
        assert center.distance(other=shape.vertex(0), planet=DEFAULT_PLANET) == (
            pytest.approx(80_000.0, rel=1e-4)
        )

        # Vertex at a quarter turn is across
        assert center.distance(other=shape.vertex(16), planet=DEFAULT_PLANET) == (
            pytest.approx(40_000.0, rel=1e-4)
        )

    def test_regular_polygon(self) -> None:
        """
        A regular polygon has the requested sides and radius.
        """

        shape: Polygon = regular_polygon(
            center=Coordinate(0, 0, R), radius_m=10_000.0, sides=6
        )

        assert shape.num_vertices() == 6

        center: Coordinate = Coordinate(0.0, 0.0, R)

        for i in range(6):
            assert center.distance(other=shape.vertex(i), planet=DEFAULT_PLANET) == (
                pytest.approx(10_000.0, rel=1e-6)
            )

    def test_validation(self) -> None:
        """
        Bad segment / side counts and radii are rejected.
        """

        with pytest.raises(ValueError):
            circle(center=Coordinate(0, 0, R), radius_m=1.0, segments=2)

        with pytest.raises(ValueError):
            ellipse(
                center=Coordinate(0, 0, R),
                radius_a_m=-1.0,
                radius_b_m=1.0,
            )

        with pytest.raises(ValueError):
            regular_polygon(center=Coordinate(0, 0, R), radius_m=1.0, sides=2)


class TestLanguageIntegration:
    """
    Polyline / Polygon usable from .nadata scripts.
    """

    def _app(self) -> Any:
        """
        Creates a fresh Nadata facade.

        Returns:
            Any: The Nadata instance.
        """

        return Nadata(database=NaDataDatabase())

    def test_schema_attributes(self) -> None:
        """
        Polyline and Polygon work as schema attribute types.
        """

        app: Any = self._app()

        app.execute_schema(
            "class District { boundary: Polygon; main_street: Polyline; }"
        )
        app.execute_data(
            "d = District("
            "boundary=Polygon([Coordinate(0,0,6371008.8),"
            " Coordinate(0,1,6371008.8), Coordinate(1,1,6371008.8),"
            " Coordinate(1,0,6371008.8)]),"
            " main_street=Polyline([Coordinate(0,0,6371008.8),"
            " Coordinate(0,1,6371008.8)]));"
        )

        boundary: Any = app.get_attribute(symbol="d", name="boundary")

        assert isinstance(boundary, Polygon)
        assert boundary.num_vertices() == 4

        street: Any = app.get_attribute(symbol="d", name="main_street")

        assert isinstance(street, Polyline)
        assert street.num_vertices() == 2

    def test_script_calls(self) -> None:
        """
        Members and free functions are callable from scripts.
        """

        app: Any = self._app()

        app.execute_schema("class Spot { ring: Polygon; }")
        app.execute_data(
            "s = Spot(ring=Polygon([Coordinate(0,0,6371008.8),"
            " Coordinate(0,1,6371008.8), Coordinate(1,1,6371008.8),"
            " Coordinate(1,0,6371008.8)]));"
            "a = s.ring.area();"
            "print(\"area:\", a);"
            "c = s.ring.contains(Coordinate(0.5, 0.5, 6371008.8));"
            "print(\"contains:\", c);"
            "p = regular_polygon(Coordinate(0, 0, 6371008.8), 10000.0, 6);"
            "print(\"sides:\", p.num_vertices());"
            "for (v: Coordinate in s.ring.vertices) { print(\"v:\", v); }"
        )

        # No exception and the script ran: the facade stays usable
        assert app.get_attribute(symbol="s", name="ring") is not None


class TestPersistence:
    """
    Binary cells, JSON codec, and full database round-trips.
    """

    def test_binary_roundtrip(self) -> None:
        """
        Polylines and polygons survive binary cell encoding exactly.
        """

        for original in (MERIDIAN_PATH, SQUARE, Polygon(vertices=[])):
            value_type: Any = (
                TypePolyline()
                if isinstance(original, Polyline)
                else TypePolygon()
            )

            payload: bytes = encode_value(raw=original, value_type=value_type)
            decoded: Any
            offset: int
            decoded, offset = decode_value(
                payload=payload, offset=0, value_type=value_type, resolver=lambda _: None
            )

            assert decoded == original
            assert offset == len(payload)

    def test_json_codec_roundtrip(self) -> None:
        """
        Polylines and polygons survive the generic JSON codec.
        """

        codec: ObjectCodec = ObjectCodec()
        context: DecodeContext = DecodeContext()

        for original in (MERIDIAN_PATH, SQUARE):
            assert codec.decode(
                data=codec.encode(obj=original), context=context
            ) == original

    def test_database_dump_load(self, tmp_path: Any) -> None:
        """
        A dumped and reloaded database preserves geometry values.
        """

        app: Any = Nadata(database=NaDataDatabase())

        app.execute_schema("class District { boundary: Polygon; }")
        app.execute_data(
            "d = District(boundary=Polygon([Coordinate(0,0,6371008.8),"
            " Coordinate(0,1,6371008.8), Coordinate(1,1,6371008.8),"
            " Coordinate(1,0,6371008.8)]));"
        )

        storage: str = str(tmp_path / "geom.ndb")

        app.dump(storage_path=storage)
        reloaded: Any = Nadata.load(path=storage)

        assert reloaded.get_attribute(symbol="d", name="boundary") == SQUARE


class TestValueWrapping:
    """
    Schema type / value wrapper plumbing.
    """

    def test_make_value(self) -> None:
        """
        make_value wraps raw polylines and polygons.
        """

        line_value: Any = make_value(value_type=TypePolyline(), raw=MERIDIAN_PATH)

        assert line_value.get() == MERIDIAN_PATH

        ring_value: Any = make_value(value_type=TypePolygon(), raw=SQUARE)

        assert ring_value.get() == SQUARE

    def test_default_values(self) -> None:
        """
        The schema types default to empty geometries.
        """

        assert TypePolyline().default_value.num_vertices() == 0
        assert TypePolygon().default_value.num_vertices() == 0


class TestSetCurrentPlanetRestored:
    """
    Tests mutate the session planet; restore the default afterwards.
    """

    def teardown_method(self) -> None:
        """
        Restores the default session planet.

        Returns:
            None: Nothing.
        """

        set_current_planet(planet=DEFAULT_PLANET)
