"""
Tests for the geodesy layer: the Planet and Coordinate home types,
the free geodesy builtins, planet settings wiring, and Coordinate /
Planet persistence (binary cells, JSON codec, full database dumps).
"""

# Import Modules
import math
from typing import Any

import pytest

from src.api import Nadata
from src.core.database.database import NaDataDatabase
from src.core.database.persistence.binary_cells import decode_value, encode_value
from src.core.database.persistence.serialization import DecodeContext, ObjectCodec
from src.core.types.home_types.coordinate import (
    Coordinate,
    altitude,
    bearing,
    destination,
    distance,
    geocentric_latitude,
    interpolate,
    normalize_lon,
    surface_radius,
)
from src.core.types.home_types.planet import (
    DEFAULT_PLANET,
    EARTH_EQUATORIAL_RADIUS,
    EARTH_MEAN_RADIUS,
    EARTH_POLAR_RADIUS,
    Planet,
    get_current_planet,
    planet_from_settings,
    set_current_planet,
)
from src.core.types.types import TypeCoordinate, TypePlanet
from src.core.values.values import make_value

# WGS84 for ellipsoid tests
WGS84: Planet = Planet(
    equatorial_radius=EARTH_EQUATORIAL_RADIUS,
    polar_radius=EARTH_POLAR_RADIUS,
)

PARIS: tuple[float, float] = (48.8566, 2.3522)
NYC: tuple[float, float] = (40.7128, -74.0060)



def _meridian_arc(lat1_deg: float, lat2_deg: float, planet: Planet) -> float:
    """
    Computes the exact meridian arc distance between two latitudes
    (independent series formula, used as the reference for Vincenty).

    Args:
        lat1_deg (float): Start latitude in degrees.
        lat2_deg (float): End latitude in degrees.
        planet (Planet): Spheroid planet.

    Returns:
        float: Distance in meters.
    """

    a: float = planet.equatorial_radius
    e2: float = planet.eccentricity_squared

    def arc(phi_rad: float) -> float:
        """
        Meridian arc length from the equator to a latitude.

        Args:
            phi_rad (float): Latitude in radians.

        Returns:
            float: Arc length in meters.
        """

        return a * (
            (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256) * phi_rad
            - (3 * e2 / 8 + 3 * e2**2 / 32 + 45 * e2**3 / 1024)
            * math.sin(2 * phi_rad)
            + (15 * e2**2 / 256 + 45 * e2**3 / 1024) * math.sin(4 * phi_rad)
            - (35 * e2**3 / 3072) * math.sin(6 * phi_rad)
        )

    return arc(math.radians(lat2_deg)) - arc(math.radians(lat1_deg))



class TestCoordinateConstruction:
    """
    Validation and normalization on construction.
    """

    def test_components_stored(self) -> None:
        """
        Components are stored as floats.
        """

        c: Coordinate = Coordinate(lat=10.0, lon=20.0, radius=1000.0)

        assert c.lat == 10.0
        assert c.lon == 20.0
        assert c.radius == 1000.0

    def test_lon_normalized(self) -> None:
        """
        Out-of-range longitudes wrap into [-180, 180).
        """

        assert Coordinate(0.0, 190.0, 1000.0).lon == -170.0
        assert Coordinate(0.0, -190.0, 1000.0).lon == 170.0
        assert Coordinate(0.0, 370.0, 1000.0).lon == 10.0
        assert Coordinate(0.0, -180.0, 1000.0).lon == -180.0

    def test_lon_in_range_untouched(self) -> None:
        """
        In-range longitudes keep bit-exact precision.
        """

        c: Coordinate = Coordinate(0.0, 2.3522, 1000.0)

        assert c.lon == 2.3522

    def test_lat_out_of_range_rejected(self) -> None:
        """
        Latitudes beyond the poles raise ValueError.
        """

        with pytest.raises(ValueError):
            Coordinate(90.5, 0.0, 1000.0)

        with pytest.raises(ValueError):
            Coordinate(-90.5, 0.0, 1000.0)

    def test_radius_must_be_positive(self) -> None:
        """
        Non-positive radii raise ValueError.
        """

        with pytest.raises(ValueError):
            Coordinate(0.0, 0.0, 0.0)

        with pytest.raises(ValueError):
            Coordinate(0.0, 0.0, -1.0)

    def test_non_numeric_rejected(self) -> None:
        """
        Non-numeric components raise TypeError.
        """

        with pytest.raises(TypeError):
            Coordinate("0", 0.0, 1000.0)  # type: ignore[arg-type]

        with pytest.raises(TypeError):
            Coordinate(True, 0.0, 1000.0)  # type: ignore[arg-type]

    def test_equality_and_repr(self) -> None:
        """
        Equality compares components; repr round-trips the values.
        """

        a: Coordinate = Coordinate(1.0, 2.0, 3.0)

        assert a == Coordinate(1.0, 2.0, 3.0)
        assert a != Coordinate(1.0, 2.0, 4.0)
        assert a != "Coordinate(1, 2, 3)"
        assert repr(a) == "Coordinate(1.0, 2.0, 3.0)"


class TestSphereDistance:
    """
    Great-circle distances on spheres.
    """

    def test_self_distance_zero(self) -> None:
        """
        Distance to self is zero.
        """

        a: Coordinate = Coordinate(12.0, 34.0, EARTH_MEAN_RADIUS)

        assert distance(a, a) == 0.0

    def test_symmetry(self) -> None:
        """
        Distance is symmetric.
        """

        a: Coordinate = Coordinate(*PARIS, EARTH_MEAN_RADIUS)
        b: Coordinate = Coordinate(*NYC, EARTH_MEAN_RADIUS)

        assert distance(a, b) == pytest.approx(distance(b, a))

    def test_antipodal_half_circumference(self) -> None:
        """
        Antipodal equatorial points span half the circumference.
        """

        a: Coordinate = Coordinate(0.0, 0.0, EARTH_MEAN_RADIUS)
        b: Coordinate = Coordinate(0.0, 180.0, EARTH_MEAN_RADIUS)

        assert distance(a, b) == pytest.approx(math.pi * EARTH_MEAN_RADIUS)

    def test_pole_equator_quarter(self) -> None:
        """
        Pole to equator spans a quarter circumference.
        """

        a: Coordinate = Coordinate(0.0, 0.0, EARTH_MEAN_RADIUS)
        b: Coordinate = Coordinate(90.0, 0.0, EARTH_MEAN_RADIUS)

        assert distance(a, b) == pytest.approx(math.pi * EARTH_MEAN_RADIUS / 2.0)

    def test_one_degree_equator(self) -> None:
        """
        One degree of longitude at the equator is circumference / 360.
        """

        a: Coordinate = Coordinate(0.0, 0.0, EARTH_MEAN_RADIUS)
        b: Coordinate = Coordinate(0.0, 1.0, EARTH_MEAN_RADIUS)

        assert distance(a, b) == pytest.approx(
            2.0 * math.pi * EARTH_MEAN_RADIUS / 360.0
        )

    def test_antimeridian_wrapping(self) -> None:
        """
        Points across the antimeridian measure the short way.
        """

        a: Coordinate = Coordinate(0.0, 179.5, EARTH_MEAN_RADIUS)
        b: Coordinate = Coordinate(0.0, -179.5, EARTH_MEAN_RADIUS)

        assert distance(a, b) == pytest.approx(
            2.0 * math.pi * EARTH_MEAN_RADIUS / 360.0
        )


class TestSpheroidDistance:
    """
    Vincenty distances on spheroids, checked against exact references.
    """

    def test_meridian_arc_exact(self) -> None:
        """
        Meridian geodesics match the exact meridian arc integral to
        sub-millimeter.
        """

        for lat in (10.0, 45.0, 80.0):
            exact: float = _meridian_arc(0.0, lat, WGS84)
            computed: float = distance(
                Coordinate(0.0, 0.0, EARTH_MEAN_RADIUS),
                Coordinate(lat, 0.0, EARTH_MEAN_RADIUS),
                planet=WGS84,
            )

            assert computed == pytest.approx(exact, abs=1e-3)

    def test_equator_arc_exact(self) -> None:
        """
        The equator is a geodesic of radius a; distances along it are
        exact arc lengths.
        """

        a: Coordinate = Coordinate(0.0, 0.0, EARTH_MEAN_RADIUS)
        b: Coordinate = Coordinate(0.0, 90.0, EARTH_MEAN_RADIUS)

        assert distance(a, b, planet=WGS84) == pytest.approx(
            math.pi / 2.0 * WGS84.equatorial_radius, abs=1e-3
        )

    def test_sphere_and_spheroid_agree_on_sphere(self) -> None:
        """
        A spheroid with equal radii behaves exactly like a sphere.
        """

        sphere: Planet = Planet(1000000.0)
        flat: Planet = Planet(1000000.0, 1000000.0)

        a: Coordinate = Coordinate(23.0, 12.0, 1000000.0)
        b: Coordinate = Coordinate(-5.0, -60.0, 1000000.0)

        assert distance(a, b, planet=flat) == pytest.approx(
            distance(a, b, planet=sphere)
        )

    def test_round_trip_paris_nyc(self) -> None:
        """
        Direct then inverse Vincenty round-trips to sub-millimeter.
        """

        a: Coordinate = Coordinate(*PARIS, EARTH_MEAN_RADIUS)
        b: Coordinate = Coordinate(*NYC, EARTH_MEAN_RADIUS)

        leg: float = distance(a, b, planet=WGS84)
        back: Coordinate = destination(a, bearing(a, b, planet=WGS84), leg, planet=WGS84)

        assert distance(back, b, planet=WGS84) == pytest.approx(0.0, abs=1e-3)

    def test_paris_nyc_order_of_magnitude(self) -> None:
        """
        Paris-NYC lands in the expected ~5850 km band (spherical
        great-circle is ~5837 km; the spheroid geodesic is longer).
        """

        a: Coordinate = Coordinate(*PARIS, EARTH_MEAN_RADIUS)
        b: Coordinate = Coordinate(*NYC, EARTH_MEAN_RADIUS)

        assert 5.80e6 < distance(a, b, planet=WGS84) < 5.90e6

    def test_near_antipodal_fallback(self) -> None:
        """
        Near-antipodal pairs still return a sane distance (sphere
        fallback) instead of raising.
        """

        a: Coordinate = Coordinate(0.0, 10.0, EARTH_MEAN_RADIUS)
        b: Coordinate = Coordinate(0.0, 170.0, EARTH_MEAN_RADIUS)

        assert 0.0 < distance(a, b, planet=WGS84) < math.pi * EARTH_MEAN_RADIUS


class TestBearing:
    """
    Initial bearings.
    """

    def test_north_along_meridian(self) -> None:
        """
        Northward along a meridian bears 0.
        """

        a: Coordinate = Coordinate(0.0, 0.0, EARTH_MEAN_RADIUS)
        b: Coordinate = Coordinate(10.0, 0.0, EARTH_MEAN_RADIUS)

        assert bearing(a, b) == pytest.approx(0.0)

    def test_east_along_equator(self) -> None:
        """
        Eastward along the equator bears 90.
        """

        a: Coordinate = Coordinate(0.0, 0.0, EARTH_MEAN_RADIUS)
        b: Coordinate = Coordinate(0.0, 10.0, EARTH_MEAN_RADIUS)

        assert bearing(a, b) == pytest.approx(90.0)

    def test_west_across_antimeridian(self) -> None:
        """
        Crossing the antimeridian eastward still bears 90.
        """

        a: Coordinate = Coordinate(0.0, 179.0, EARTH_MEAN_RADIUS)
        b: Coordinate = Coordinate(0.0, -179.0, EARTH_MEAN_RADIUS)

        assert bearing(a, b) == pytest.approx(90.0)

    def test_south(self) -> None:
        """
        Southward along a meridian bears 180.
        """

        a: Coordinate = Coordinate(10.0, 5.0, EARTH_MEAN_RADIUS)
        b: Coordinate = Coordinate(0.0, 5.0, EARTH_MEAN_RADIUS)

        assert bearing(a, b) == pytest.approx(180.0)

    def test_bearing_range(self) -> None:
        """
        Bearings always land in [0, 360).
        """

        a: Coordinate = Coordinate(*PARIS, EARTH_MEAN_RADIUS)
        b: Coordinate = Coordinate(*NYC, EARTH_MEAN_RADIUS)

        assert 0.0 <= bearing(a, b, planet=WGS84) < 360.0


class TestDestination:
    """
    Direct problems.
    """

    def test_quarter_circumference_reaches_pole(self) -> None:
        """
        Traveling a quarter circumference north from the equator lands
        at the pole.
        """

        a: Coordinate = Coordinate(0.0, 0.0, EARTH_MEAN_RADIUS)

        result: Coordinate = destination(
            a, 0.0, math.pi * EARTH_MEAN_RADIUS / 2.0
        )

        assert result.lat == pytest.approx(90.0, abs=1e-6)

    def test_one_degree_east(self) -> None:
        """
        One degree east along the equator lands one degree east.
        """

        a: Coordinate = Coordinate(0.0, 0.0, EARTH_MEAN_RADIUS)

        result: Coordinate = destination(
            a, 90.0, 2.0 * math.pi * EARTH_MEAN_RADIUS / 360.0
        )

        assert result.lat == pytest.approx(0.0, abs=1e-9)
        assert result.lon == pytest.approx(1.0, abs=1e-9)

    def test_sphere_and_spheroid_consistency(self) -> None:
        """
        Destination on a spheroid round-trips through the inverse.
        """

        a: Coordinate = Coordinate(*PARIS, EARTH_MEAN_RADIUS)

        leg: float = 2_500_000.0
        result: Coordinate = destination(a, 45.0, leg, planet=WGS84)

        assert distance(a, result, planet=WGS84) == pytest.approx(leg, abs=1e-3)

    def test_negative_distance_rejected(self) -> None:
        """
        Negative distances raise ValueError.
        """

        with pytest.raises(ValueError):
            destination(Coordinate(0.0, 0.0, EARTH_MEAN_RADIUS), 0.0, -1.0)

    def test_zero_distance_returns_origin(self) -> None:
        """
        Zero-distance destinations keep the origin direction.
        """

        a: Coordinate = Coordinate(10.0, 20.0, EARTH_MEAN_RADIUS)

        result: Coordinate = destination(a, 30.0, 0.0)

        assert result.lat == pytest.approx(a.lat)
        assert result.lon == pytest.approx(a.lon)


class TestInterpolate:
    """
    Great-circle interpolation.
    """

    def test_endpoints(self) -> None:
        """
        t = 0 and t = 1 return the endpoints (as coordinates).
        """

        a: Coordinate = Coordinate(0.0, 0.0, 1000.0)
        b: Coordinate = Coordinate(0.0, 2.0, 2000.0)

        assert interpolate(a, b, 0.0) == a
        assert interpolate(a, b, 1.0) == b

    def test_equator_midpoint(self) -> None:
        """
        Midpoint along the equator is the angular midpoint; the radius
        interpolates linearly.
        """

        a: Coordinate = Coordinate(0.0, 0.0, 1000.0)
        b: Coordinate = Coordinate(0.0, 2.0, 2000.0)

        mid: Coordinate = interpolate(a, b, 0.5)

        assert mid.lat == pytest.approx(0.0, abs=1e-9)
        assert mid.lon == pytest.approx(1.0, abs=1e-9)
        assert mid.radius == pytest.approx(1500.0)

    def test_same_point(self) -> None:
        """
        Interpolating between identical points returns the point.
        """

        a: Coordinate = Coordinate(45.0, 45.0, 1000.0)

        assert interpolate(a, a, 0.7) == a

    def test_clamping(self) -> None:
        """
        t outside [0, 1] clamps to the endpoints.
        """

        a: Coordinate = Coordinate(0.0, 0.0, 1000.0)
        b: Coordinate = Coordinate(0.0, 2.0, 2000.0)

        assert interpolate(a, b, -1.0) == a
        assert interpolate(a, b, 2.0) == b


class TestSurfaceAndAltitude:
    """
    Surface radius and altitude.
    """

    def test_sphere_constant(self) -> None:
        """
        Spheres have a constant surface radius.
        """

        for lat in (0.0, 45.0, -80.0):
            assert surface_radius(Coordinate(lat, 0.0, EARTH_MEAN_RADIUS)) == (
                pytest.approx(EARTH_MEAN_RADIUS)
            )

    def test_wgs84_extremes(self) -> None:
        """
        WGS84 surface radius equals a at the equator and b at the
        poles, with mid-latitudes in between.
        """

        equator: float = surface_radius(Coordinate(0.0, 0.0, EARTH_MEAN_RADIUS), planet=WGS84)
        pole: float = surface_radius(Coordinate(90.0, 0.0, EARTH_MEAN_RADIUS), planet=WGS84)
        mid: float = surface_radius(Coordinate(45.0, 0.0, EARTH_MEAN_RADIUS), planet=WGS84)

        assert equator == pytest.approx(WGS84.equatorial_radius)
        assert pole == pytest.approx(WGS84.polar_radius_value)
        assert WGS84.polar_radius_value < mid < WGS84.equatorial_radius

    def test_altitude(self) -> None:
        """
        Altitude is the stored radius minus the surface radius; a
        350 m tower sits at +350 m.
        """

        surface: float = surface_radius(Coordinate(0.0, 0.0, EARTH_MEAN_RADIUS))

        assert altitude(Coordinate(0.0, 0.0, surface)) == pytest.approx(0.0)
        assert altitude(Coordinate(0.0, 0.0, surface + 350.0)) == pytest.approx(350.0)
        assert altitude(Coordinate(0.0, 0.0, surface - 100.0)) == pytest.approx(-100.0)


class TestGeocentricLatitude:
    """
    Geodetic to geocentric conversion.
    """

    def test_sphere_identity(self) -> None:
        """
        Spheres keep geodetic and geocentric latitudes equal.
        """

        assert geocentric_latitude(Coordinate(45.0, 0.0, EARTH_MEAN_RADIUS)) == (
            pytest.approx(45.0)
        )

    def test_wgs84_45_degrees(self) -> None:
        """
        WGS84 45 degrees geodetic is about 44.807 degrees geocentric.
        """

        result: float = geocentric_latitude(
            Coordinate(45.0, 0.0, EARTH_MEAN_RADIUS), planet=WGS84
        )

        assert result == pytest.approx(math.degrees(math.atan(
            (WGS84.polar_radius_value / WGS84.equatorial_radius) ** 2
        )), abs=1e-9)

    def test_extremes_unchanged(self) -> None:
        """
        Equator and poles are unchanged by the conversion.
        """

        assert geocentric_latitude(Coordinate(0.0, 0.0, 1.0), planet=WGS84) == (
            pytest.approx(0.0)
        )
        assert geocentric_latitude(Coordinate(90.0, 0.0, 1.0), planet=WGS84) == (
            pytest.approx(90.0)
        )


class TestChordDistance:
    """
    Chord distances (fast path primitive).
    """

    def test_planet_independent(self) -> None:
        """
        Chord distance does not consult any planet.
        """

        a: Coordinate = Coordinate(10.0, 20.0, 1000.0)
        b: Coordinate = Coordinate(30.0, 40.0, 2000.0)

        assert a.chord_distance(other=b) == pytest.approx(b.chord_distance(other=a))

    def test_self_zero(self) -> None:
        """
        Chord distance to self is zero.
        """

        a: Coordinate = Coordinate(10.0, 20.0, 1000.0)

        assert a.chord_distance(other=a) == 0.0

    def test_monotonic_in_geodesic(self) -> None:
        """
        Chord distance is monotonic in the geodesic distance for
        surface points.
        """

        origin: Coordinate = Coordinate(0.0, 0.0, EARTH_MEAN_RADIUS)
        near: Coordinate = destination(origin, 0.0, 100_000.0)
        far: Coordinate = destination(origin, 0.0, 500_000.0)

        assert origin.chord_distance(other=near) < origin.chord_distance(other=far)


class TestNormalizeLon:
    """
    Longitude normalization helper.
    """

    def test_in_range_identity(self) -> None:
        """
        In-range values pass through unchanged.
        """

        assert normalize_lon(0.0) == 0.0
        assert normalize_lon(2.3522) == 2.3522
        assert normalize_lon(-179.9) == -179.9

    def test_wrapping(self) -> None:
        """
        Out-of-range values wrap.
        """

        assert normalize_lon(180.0) == pytest.approx(-180.0)
        assert normalize_lon(190.0) == pytest.approx(-170.0)
        assert normalize_lon(-190.0) == pytest.approx(170.0)
        assert normalize_lon(540.0) == pytest.approx(180.0) or normalize_lon(
            540.0
        ) == pytest.approx(-180.0)


class TestPlanet:
    """
    Planet construction and derived quantities.
    """

    def test_sphere_default(self) -> None:
        """
        A planet without polar radius is a sphere.
        """

        p: Planet = Planet(1000.0)

        assert p.is_sphere
        assert p.polar_radius is None
        assert p.equatorial_radius == 1000.0
        assert p.flattening == 0.0
        assert p.eccentricity_squared == 0.0
        assert repr(p) == "Planet(1000.0)"

    def test_spheroid(self) -> None:
        """
        Spheroids expose flattening and eccentricity.
        """

        assert WGS84.flattening == pytest.approx(
            (EARTH_EQUATORIAL_RADIUS - EARTH_POLAR_RADIUS) / EARTH_EQUATORIAL_RADIUS
        )
        assert not WGS84.is_sphere
        assert repr(WGS84) == f"Planet({EARTH_EQUATORIAL_RADIUS}, {EARTH_POLAR_RADIUS})"

    def test_validation(self) -> None:
        """
        Non-positive or non-numeric radii are rejected.
        """

        with pytest.raises(ValueError):
            Planet(0.0)

        with pytest.raises(ValueError):
            Planet(1000.0, 0.0)

        with pytest.raises(TypeError):
            Planet("big")  # type: ignore[arg-type]

    def test_equality(self) -> None:
        """
        Equality compares radii.
        """

        assert Planet(1000.0) == Planet(1000.0)
        assert Planet(1000.0) != Planet(2000.0)
        assert Planet(1000.0, 900.0) != Planet(1000.0)
        assert Planet(1000.0) != "Planet"

    def test_quadratic_mean_radius(self) -> None:
        """
        The quadratic mean radius sits between the two axes.
        """

        q: float = WGS84.quadratic_mean_radius

        assert WGS84.polar_radius_value < q < WGS84.equatorial_radius
        assert q == pytest.approx(
            math.sqrt(
                (WGS84.equatorial_radius**2 + WGS84.polar_radius_value**2) / 2.0
            )
        )


class TestCurrentPlanet:
    """
    Session current planet plumbing.
    """

    def test_default_earth_sphere(self) -> None:
        """
        With no session planet, the default Earth sphere applies.
        """

        set_current_planet(planet=DEFAULT_PLANET)

        assert get_current_planet() == DEFAULT_PLANET

    def test_set_and_get(self) -> None:
        """
        set_current_planet stores the session planet.
        """

        tiny: Planet = Planet(1000000.0)

        set_current_planet(planet=tiny)
        assert get_current_planet() == tiny

        set_current_planet(planet=DEFAULT_PLANET)

    def test_set_rejects_non_planet(self) -> None:
        """
        Non-Planet overrides are rejected.
        """

        with pytest.raises(TypeError):
            set_current_planet(planet=1000.0)  # type: ignore[arg-type]

        set_current_planet(planet=DEFAULT_PLANET)

    def test_current_planet_used_by_default(self) -> None:
        """
        Geodesy without an explicit override uses the session planet.
        """

        set_current_planet(planet=Planet(1000000.0))

        a: Coordinate = Coordinate(0.0, 0.0, 1000000.0)
        b: Coordinate = Coordinate(0.0, 1.0, 1000000.0)

        assert distance(a, b) == pytest.approx(1000000.0 * math.pi / 180.0)

        set_current_planet(planet=DEFAULT_PLANET)

    def test_planet_from_settings_sphere(self) -> None:
        """
        Sphere settings resolve to a sphere planet.
        """

        assert planet_from_settings({}) == DEFAULT_PLANET
        assert planet_from_settings({"planet.radius": 2000.0}) == Planet(2000.0)

    def test_planet_from_settings_ellipsoid(self) -> None:
        """
        Ellipsoid settings resolve to a spheroid with WGS84 defaults.
        """

        resolved: Planet = planet_from_settings({"planet.kind": "ellipsoid"})

        assert resolved == WGS84

        custom: Planet = planet_from_settings(
            {
                "planet.kind": "ellipsoid",
                "planet.equatorial_radius": 100.0,
                "planet.polar_radius": 90.0,
            }
        )

        assert custom == Planet(100.0, 90.0)

    def test_planet_from_settings_unknown_kind(self) -> None:
        """
        Unknown kinds raise ValueError.
        """

        with pytest.raises(ValueError):
            planet_from_settings({"planet.kind": "torus"})


class TestLanguageIntegration:
    """
    Coordinate / Planet usable from .nadata scripts.
    """

    def _app(self) -> Any:
        """
        Creates a fresh Nadata facade.

        Returns:
            Any: The Nadata instance.
        """

        return Nadata(database=NaDataDatabase())

    def test_schema_attribute_and_instance(self) -> None:
        """
        Coordinate works as a schema attribute type and stores values.
        """

        app: Any = self._app()

        app.execute_schema("class Place { origin: Coordinate; }")
        app.execute_data(
            "paris = Place(origin=Coordinate(lat=48.8566, lon=2.3522,"
            " radius=6371008.8));"
        )

        origin: Any = app.get_attribute(symbol="paris", name="origin")

        assert isinstance(origin, Coordinate)
        assert origin.lat == 48.8566
        assert origin.lon == 2.3522

    def test_planet_schema_attribute(self) -> None:
        """
        Planet works as a schema attribute type.
        """

        app: Any = self._app()

        app.execute_schema("class World { shape: Planet; }")
        app.execute_data("earth = World(shape=Planet(6371008.8));")

        shape: Any = app.get_attribute(symbol="earth", name="shape")

        assert shape == Planet(6371008.8)

    def test_method_style_and_free_function_agree(self) -> None:
        """
        Method-style members and free functions return identical results.
        """

        a: Coordinate = Coordinate(0.0, 0.0, 6371008.8)
        b: Coordinate = Coordinate(0.0, 1.0, 6371008.8)

        assert distance(a, b) == a.distance(other=b)
        assert bearing(a, b) == a.bearing(other=b)
        assert destination(a, 30.0, 1000.0) == a.destination(
            bearing_deg=30.0, distance_m=1000.0
        )
        assert a.chord_distance(other=b) == b.chord_distance(other=a)

    def test_script_distance(self) -> None:
        """
        A script computes a distance through the free function.
        """

        app: Any = self._app()

        app.execute_schema("class Spot { at: Coordinate; }")
        app.execute_data(
            "origin = Spot(at=Coordinate(lat=0, lon=0, radius=1000000.0));"
            "target = Spot(at=Coordinate(lat=0, lon=1, radius=1000000.0));"
            "set_setting(\"planet.radius\", 1000000.0);"
            "d = distance(origin.at, target.at);"
            "print(\"d:\", d);"
        )

        assert app.get_attribute(symbol="origin", name="at") == Coordinate(
            0.0, 0.0, 1000000.0
        )


class TestTypeAndValueWrapping:
    """
    Schema type / value wrapper plumbing.
    """

    def test_make_value(self) -> None:
        """
        make_value wraps raw Coordinates and Planets.
        """

        coord_value: Any = make_value(value_type=TypeCoordinate(), raw=Coordinate(1, 2, 3))

        assert coord_value.get() == Coordinate(1, 2, 3)

        planet_value: Any = make_value(value_type=TypePlanet(), raw=Planet(5.0))

        assert planet_value.get() == Planet(5.0)

    def test_type_builtin(self) -> None:
        """
        type() resolves Coordinate attributes to their schema type.
        """

        app: Any = Nadata(database=NaDataDatabase())

        app.execute_schema("class Place { origin: Coordinate; }")
        app.execute_data("p = Place(origin=Coordinate(0, 0, 1000.0));")

        instance: Any = app.get(symbol="p")

        origin_type: Any = instance.attributes["origin"].value_type

        assert isinstance(origin_type, TypeCoordinate)


class TestPersistence:
    """
    Binary cells, JSON codec, and full database round-trips.
    """

    def test_binary_coordinate_roundtrip(self) -> None:
        """
        Coordinates survive binary cell encoding exactly.
        """

        coord_type: Any = TypeCoordinate()
        original: Coordinate = Coordinate(48.8566, -179.9, 6371008.8)

        payload: bytes = encode_value(raw=original, value_type=coord_type)
        decoded: Any
        offset: int
        decoded, offset = decode_value(
            payload=payload, offset=0, value_type=coord_type, resolver=lambda _: None
        )

        assert decoded == original
        assert offset == len(payload)

    def test_binary_planet_roundtrip(self) -> None:
        """
        Spheres and spheroids survive binary cell encoding exactly.
        """

        sphere_type: Any = TypePlanet()
        sphere: Planet = Planet(1000.0)

        payload: bytes = encode_value(raw=sphere, value_type=sphere_type)
        decoded: Any
        offset: int
        decoded, offset = decode_value(
            payload=payload, offset=0, value_type=sphere_type, resolver=lambda _: None
        )

        assert decoded == sphere
        assert offset == len(payload)

        spheroid: Planet = WGS84

        payload = encode_value(raw=spheroid, value_type=sphere_type)
        decoded, offset = decode_value(
            payload=payload, offset=0, value_type=sphere_type, resolver=lambda _: None
        )

        assert decoded == spheroid
        assert offset == len(payload)

    def test_json_codec_roundtrip(self) -> None:
        """
        Coordinates and planets survive the generic JSON codec.
        """

        codec: ObjectCodec = ObjectCodec()
        context: DecodeContext = DecodeContext()

        coord: Coordinate = Coordinate(48.8566, 2.3522, 6371008.8)

        assert codec.decode(data=codec.encode(obj=coord), context=context) == coord

        planet: Planet = Planet(6378137.0, 6356752.314245)

        assert codec.decode(data=codec.encode(obj=planet), context=context) == planet

    def test_database_dump_load(self, tmp_path: Any) -> None:
        """
        A dumped and reloaded database preserves coordinates, planets,
        and the session planet setting.
        """

        app: Any = Nadata(database=NaDataDatabase())

        app.execute_schema("class Place { origin: Coordinate; shape: Planet; }")
        app.execute_data(
            "paris = Place("
            "origin=Coordinate(lat=48.8566, lon=2.3522, radius=6371008.8),"
            " shape=Planet(6378137.0, 6356752.314245));"
            "set_setting(\"planet.radius\", 2000000.0);"
        )

        storage: str = str(tmp_path / "geo.ndb")

        app.dump(storage_path=storage)
        reloaded: Any = Nadata.load(path=storage)

        assert reloaded.get_attribute(symbol="paris", name="origin") == Coordinate(
            48.8566, 2.3522, 6371008.8
        )
        assert reloaded.get_attribute(symbol="paris", name="shape") == Planet(
            6378137.0, 6356752.314245
        )

        # The persisted planet setting re-seeds the session planet
        reloaded.execute_data(
            "d = distance(Coordinate(0, 0, 2000000.0),"
            " Coordinate(0, 1, 2000000.0));"
            "print(\"d:\", d);"
        )

        assert get_current_planet() == Planet(2000000.0)

        set_current_planet(planet=DEFAULT_PLANET)
