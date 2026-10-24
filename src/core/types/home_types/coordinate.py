"""
Coordinate and geodesy primitives for Nadata.

Provides the Coordinate value type -- a point expressed as geodetic
latitude, longitude, and radius from the planet center -- plus the
geodesic math (distance, bearing, destination, interpolation) computed
against a Planet (sphere: closed forms; spheroid: Vincenty's solutions
with a sphere fallback near antipodes).

Longitude wraps at the antimeridian inside every computation; stored
longitudes are normalized to [-180, 180).
"""

# Import Modules
from __future__ import annotations

from typing import Any, Optional
import math

from src.core.types.home_types.planet import (
    Planet,
    get_current_planet,
)
from src.core.types.home_types.sea_level import get_sea_level_surface



# Iteration / convergence bounds for Vincenty's solutions
_VINCENTY_MAX_ITERATIONS: int = 200
_VINCENTY_CONVERGENCE: float = 1e-12

# Latitude bounds in degrees
_LAT_MIN: float = -90.0
_LAT_MAX: float = 90.0



def _require_angle(name: str, value: Any) -> float:
    """
    Validates an angular argument.

    Args:
        name (str): Argument name for error messages.
        value (Any): Candidate angle in degrees.

    Returns:
        float: The validated angle.

    Raises:
        TypeError: If the value is not a real number.
        ValueError: If the value is not finite.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Coordinate {name} must be a number, got '{type(value).__name__}'")

    if not math.isfinite(float(value)):
        raise ValueError(f"Coordinate {name} must be finite, got {value}")

    return float(value)


def normalize_lon(lon_deg: float) -> float:
    """
    Normalizes a longitude into [-180, 180).

    Args:
        lon_deg (float): Longitude in degrees.

    Returns:
        float: Normalized longitude (unchanged when already in range,
        so in-range values keep bit-exact float precision).
    """

    if -180.0 <= lon_deg < 180.0:
        return lon_deg

    return (lon_deg + 180.0) % 360.0 - 180.0


def wrap_lon_delta(lon_delta_deg: float) -> float:
    """
    Wraps a longitude difference into [-180, 180).

    Args:
        lon_delta_deg (float): Raw longitude difference in degrees.

    Returns:
        float: Wrapped difference.
    """

    return (lon_delta_deg + 540.0) % 360.0 - 180.0


def _haversine(
    lat1_rad: float,
    lon1_rad: float,
    lat2_rad: float,
    lon2_rad: float,
    radius: float,
) -> float:
    """
    Computes the great-circle distance on a sphere.

    Args:
        lat1_rad (float): First latitude in radians.
        lon1_rad (float): First longitude in radians (already wrapped).
        lat2_rad (float): Second latitude in radians.
        lon2_rad (float): Second longitude in radians (already wrapped).
        radius (float): Sphere radius in meters.

    Returns:
        float: Distance in meters.
    """

    sin_dlat: float = math.sin((lat2_rad - lat1_rad) / 2.0)
    sin_dlon: float = math.sin((lon2_rad - lon1_rad) / 2.0)

    a: float = (
        sin_dlat * sin_dlat
        + math.cos(lat1_rad) * math.cos(lat2_rad) * sin_dlon * sin_dlon
    )

    return 2.0 * radius * math.asin(min(1.0, math.sqrt(a)))


def _sphere_bearing(
    lat1_rad: float,
    lon1_rad: float,
    lat2_rad: float,
    lon2_rad: float,
) -> float:
    """
    Computes the initial great-circle bearing on a sphere.

    Args:
        lat1_rad (float): Origin latitude in radians.
        lon1_rad (float): Origin longitude in radians (already wrapped).
        lat2_rad (float): Target latitude in radians.
        lon2_rad (float): Target longitude in radians (already wrapped).

    Returns:
        float: Initial bearing in degrees, in [0, 360).
    """

    dlon: float = lon2_rad - lon1_rad

    y: float = math.sin(dlon) * math.cos(lat2_rad)
    x: float = (
        math.cos(lat1_rad) * math.sin(lat2_rad)
        - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
    )

    return math.degrees(math.atan2(y, x)) % 360.0


def _sphere_destination(
    lat1_rad: float,
    lon1_rad: float,
    bearing_deg: float,
    distance_m: float,
    radius: float,
) -> tuple[float, float]:
    """
    Computes the great-circle destination on a sphere.

    Args:
        lat1_rad (float): Origin latitude in radians.
        lon1_rad (float): Origin longitude in radians (already wrapped).
        bearing_deg (float): Initial bearing in degrees.
        distance_m (float): Distance along the surface in meters.
        radius (float): Sphere radius in meters.

    Returns:
        tuple[float, float]: Destination (lat_rad, lon_rad).
    """

    angular: float = distance_m / radius
    sin_d: float = math.sin(angular)
    cos_d: float = math.cos(angular)
    theta: float = math.radians(bearing_deg)

    lat2: float = math.asin(
        math.sin(lat1_rad) * cos_d
        + math.cos(lat1_rad) * sin_d * math.cos(theta)
    )

    lon2: float = lon1_rad + math.atan2(
        math.sin(theta) * sin_d * math.cos(lat1_rad),
        cos_d - math.sin(lat1_rad) * math.sin(lat2),
    )

    return lat2, lon2


def _vincenty_inverse(
    planet: Planet,
    lat1_deg: float,
    lon1_deg: float,
    lat2_deg: float,
    lon2_deg: float,
) -> Optional[tuple[float, float, float]]:
    """
    Solves the inverse geodesic problem on a spheroid (Vincenty 1975).

    Args:
        planet (Planet): Spheroid planet.
        lat1_deg (float): First geodetic latitude in degrees.
        lon1_deg (float): First longitude in degrees (already wrapped).
        lat2_deg (float): Second geodetic latitude in degrees.
        lon2_deg (float): Second longitude in degrees (already wrapped).

    Returns:
        Optional[tuple[float, float, float]]: (distance_m, azimuth1_deg,
        azimuth2_deg), or None when the iteration fails to converge
        (near-antipodal points).
    """

    a: float = planet.equatorial_radius
    b: float = planet.polar_radius_value
    f: float = planet.flattening

    lat1: float = math.radians(lat1_deg)
    lat2: float = math.radians(lat2_deg)
    # Wrapped longitude difference keeps near-antipodal arcs short
    lam_l: float = math.radians(wrap_lon_delta(lon2_deg - lon1_deg))

    tan_u1: float = (1.0 - f) * math.tan(lat1)
    tan_u2: float = (1.0 - f) * math.tan(lat2)

    u1: float = math.atan(tan_u1)
    u2: float = math.atan(tan_u2)

    sin_u1: float = math.sin(u1)
    cos_u1: float = math.cos(u1)
    sin_u2: float = math.sin(u2)
    cos_u2: float = math.cos(u2)

    lam: float = lam_l
    sin_sigma: float = 0.0
    cos_sigma: float = 1.0
    sigma: float = 0.0
    cos_sq_alpha: float = 1.0
    cos_2sigma_m: float = 1.0

    for _ in range(_VINCENTY_MAX_ITERATIONS):
        sin_lam: float = math.sin(lam)
        cos_lam: float = math.cos(lam)

        sin_sigma = math.sqrt(
            (cos_u2 * sin_lam) ** 2
            + (cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lam) ** 2
        )

        if sin_sigma == 0.0:
            # Coincident points
            return 0.0, 0.0, 0.0

        cos_sigma = (
            sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lam
        )
        sigma = math.atan2(sin_sigma, cos_sigma)

        sin_alpha: float = (
            cos_u1 * cos_u2 * sin_lam / sin_sigma
        )
        cos_sq_alpha = 1.0 - sin_alpha * sin_alpha

        if cos_sq_alpha == 0.0:
            # Equatorial line: cos2sigma_m is undefined, use 0
            cos_2sigma_m = 0.0
        else:
            cos_2sigma_m = cos_sigma - 2.0 * sin_u1 * sin_u2 / cos_sq_alpha

        c: float = f / 16.0 * cos_sq_alpha * (
            4.0 + f * (4.0 - 3.0 * cos_sq_alpha)
        )

        lam_prev: float = lam

        lam = lam_l + (1.0 - c) * f * sin_alpha * (
            sigma
            + c * sin_sigma * (
                cos_2sigma_m
                + c * cos_sigma * (-1.0 + 2.0 * cos_2sigma_m * cos_2sigma_m)
            )
        )

        if abs(lam - lam_prev) < _VINCENTY_CONVERGENCE:
            break
    else:
        return None

    u_sq: float = cos_sq_alpha * (a * a - b * b) / (b * b)

    big_a: float = 1.0 + u_sq / 16384.0 * (
        4096.0 + u_sq * (-768.0 + u_sq * (320.0 - 175.0 * u_sq))
    )
    big_b: float = u_sq / 1024.0 * (
        256.0 + u_sq * (-128.0 + u_sq * (74.0 - 47.0 * u_sq))
    )

    delta_sigma: float = big_b * sin_sigma * (
        cos_2sigma_m
        + big_b / 4.0 * (
            cos_sigma * (-1.0 + 2.0 * cos_2sigma_m * cos_2sigma_m)
            - big_b / 6.0 * cos_2sigma_m * (
                -3.0 + 4.0 * sin_sigma * sin_sigma
            ) * (-3.0 + 4.0 * cos_2sigma_m * cos_2sigma_m)
        )
    )

    result_distance: float = b * big_a * (sigma - delta_sigma)

    azimuth1: float = math.atan2(
        cos_u2 * math.sin(lam),
        cos_u1 * sin_u2 - sin_u1 * cos_u2 * math.cos(lam),
    )

    azimuth2: float = math.atan2(
        cos_u1 * math.sin(lam),
        -sin_u1 * cos_u2 + cos_u1 * sin_u2 * math.cos(lam),
    )

    return result_distance, math.degrees(azimuth1) % 360.0, math.degrees(azimuth2) % 360.0


def _vincenty_direct(
    planet: Planet,
    lat1_deg: float,
    lon1_deg: float,
    bearing_deg: float,
    distance_m: float,
) -> Optional[tuple[float, float]]:
    """
    Solves the direct geodesic problem on a spheroid (Vincenty 1975).

    Args:
        planet (Planet): Spheroid planet.
        lat1_deg (float): Origin geodetic latitude in degrees.
        lon1_deg (float): Origin longitude in degrees (already wrapped).
        bearing_deg (float): Initial bearing in degrees.
        distance_m (float): Distance along the geodesic in meters.

    Returns:
        Optional[tuple[float, float]]: Destination (lat_rad, lon_rad),
        or None when the iteration fails to converge.
    """

    a: float = planet.equatorial_radius
    b: float = planet.polar_radius_value
    f: float = planet.flattening

    lat1: float = math.radians(lat1_deg)
    lon1: float = math.radians(lon1_deg)
    alpha1: float = math.radians(bearing_deg)
    s: float = distance_m

    sin_alpha1: float = math.sin(alpha1)
    cos_alpha1: float = math.cos(alpha1)

    tan_u1: float = (1.0 - f) * math.tan(lat1)
    cos_u1: float = 1.0 / math.sqrt(1.0 + tan_u1 * tan_u1)
    sin_u1: float = tan_u1 * cos_u1

    sigma1: float = math.atan2(sin_u1, cos_alpha1 * cos_u1)
    sin_alpha: float = cos_u1 * sin_alpha1
    cos_sq_alpha: float = 1.0 - sin_alpha * sin_alpha

    u_sq: float = cos_sq_alpha * (a * a - b * b) / (b * b)

    big_a: float = 1.0 + u_sq / 16384.0 * (
        4096.0 + u_sq * (-768.0 + u_sq * (320.0 - 175.0 * u_sq))
    )
    big_b: float = u_sq / 1024.0 * (
        256.0 + u_sq * (-128.0 + u_sq * (74.0 - 47.0 * u_sq))
    )

    sigma: float = s / (b * big_a)
    sigma_prev: float = 2.0 * sigma
    sin_sigma: float = math.sin(sigma)
    cos_sigma: float = math.cos(sigma)
    cos_2sigma_m: float = 1.0

    for _ in range(_VINCENTY_MAX_ITERATIONS):
        cos_2sigma_m = math.cos(2.0 * sigma1 + sigma)

        delta_sigma: float = big_b * sin_sigma * (
            cos_2sigma_m
            + big_b / 4.0 * (
                cos_sigma * (-1.0 + 2.0 * cos_2sigma_m * cos_2sigma_m)
                - big_b / 6.0 * cos_2sigma_m * (
                    -3.0 + 4.0 * sin_sigma * sin_sigma
                ) * (-3.0 + 4.0 * cos_2sigma_m * cos_2sigma_m)
            )
        )

        sigma_prev = sigma
        sigma = delta_sigma + s / (b * big_a)

        if abs(sigma - sigma_prev) < _VINCENTY_CONVERGENCE:
            break

        # Refresh trig values for the next iteration
        sin_sigma = math.sin(sigma)
        cos_sigma = math.cos(sigma)
    else:
        return None

    lat2: float = math.atan2(
        sin_u1 * cos_sigma + cos_u1 * sin_sigma * cos_alpha1,
        (1.0 - f)
        * math.sqrt(
            sin_alpha * sin_alpha
            + (
                sin_u1 * sin_sigma - cos_u1 * cos_sigma * cos_alpha1
            ) ** 2
        ),
    )

    lam: float = math.atan2(
        sin_sigma * sin_alpha1,
        cos_u1 * cos_sigma - sin_u1 * sin_sigma * cos_alpha1,
    )

    c: float = f / 16.0 * cos_sq_alpha * (
        4.0 + f * (4.0 - 3.0 * cos_sq_alpha)
    )

    delta_lon: float = lam - (1.0 - c) * f * sin_alpha * (
        sigma
        + c * sin_sigma * (
            cos_2sigma_m
            + c * cos_sigma * (-1.0 + 2.0 * cos_2sigma_m * cos_2sigma_m)
        )
    )

    return lat2, lon1 + delta_lon


def _to_unit_vector(lat_deg: float, lon_deg: float) -> tuple[float, float, float]:
    """
    Converts a geodetic direction to a unit Cartesian vector (the
    geodetic lat/lon pair is treated as a direction on the unit
    sphere, which keeps interpolation exact on spheres and consistent
    on spheroids).

    Args:
        lat_deg (float): Latitude in degrees.
        lon_deg (float): Longitude in degrees.

    Returns:
        tuple[float, float, float]: The (x, y, z) unit vector.
    """

    lat: float = math.radians(lat_deg)
    lon: float = math.radians(lon_deg)

    return (
        math.cos(lat) * math.cos(lon),
        math.cos(lat) * math.sin(lon),
        math.sin(lat),
    )


def _from_unit_vector(vector: tuple[float, float, float]) -> tuple[float, float]:
    """
    Converts a unit Cartesian vector back to a direction.

    Args:
        vector (tuple[float, float, float]): The (x, y, z) unit vector.

    Returns:
        tuple[float, float]: (lat_deg, lon_deg).
    """

    x: float
    y: float
    z: float
    x, y, z = vector

    return math.degrees(math.atan2(z, math.hypot(x, y))), normalize_lon(
        math.degrees(math.atan2(y, x))
    )



class Coordinate:
    """
    A point on or near a planet: geodetic latitude, longitude, and
    radius from the planet center.

    Attributes:
        lat (float): Geodetic latitude in degrees, in [-90, 90].
        lon (float): Longitude in degrees, normalized to [-180, 180).
        radius (float): Distance from the planet center in meters.
    """

    def __init__(self, lat: float, lon: float, radius: float) -> None:
        """
        Initializes the Coordinate instance.

        Args:
            lat (float): Geodetic latitude in degrees.
            lon (float): Longitude in degrees (wrapped to [-180, 180)).
            radius (float): Radius from the planet center in meters.

        Raises:
            TypeError: If a component is not a real number.
            ValueError: If the latitude is out of range or the radius
                is not strictly positive.
        """

        self.lat: float = _require_angle(name="lat", value=lat)
        self.lon: float = normalize_lon(_require_angle(name="lon", value=lon))
        self.radius: float = _require_angle(name="radius", value=radius)

        if not _LAT_MIN <= self.lat <= _LAT_MAX:
            raise ValueError(
                f"Coordinate lat must be in [{_LAT_MIN}, {_LAT_MAX}], got {self.lat}"
            )

        if self.radius <= 0.0:
            raise ValueError(f"Coordinate radius must be positive, got {self.radius}")

    def _planet(self, planet: Optional[Planet]) -> Planet:
        """
        Resolves the effective planet for a geodesic computation.

        Args:
            planet (Optional[Planet]): Explicit override, or None to
                use the session current planet.

        Returns:
            Planet: The planet to compute against.
        """

        return planet if planet is not None else get_current_planet()

    def distance(self, other: "Coordinate", planet: Optional[Planet] = None) -> float:
        """
        Computes the surface distance to another coordinate.

        Uses closed-form great-circle math on spheres and Vincenty's
        inverse solution on spheroids (with a sphere fallback for
        near-antipodal pairs where the iteration does not converge).

        Args:
            other (Coordinate): Target coordinate.
            planet (Optional[Planet], optional): Explicit planet
                override. Defaults to None.

        Returns:
            float: Distance in meters.
        """

        resolved: Planet = self._planet(planet=planet)

        if resolved.is_sphere:
            return _haversine(
                lat1_rad=math.radians(self.lat),
                lon1_rad=math.radians(self.lon),
                lat2_rad=math.radians(other.lat),
                lon2_rad=math.radians(other.lon),
                radius=resolved.equatorial_radius,
            )

        solved: Optional[tuple[float, float, float]] = _vincenty_inverse(
            planet=resolved,
            lat1_deg=self.lat,
            lon1_deg=self.lon,
            lat2_deg=other.lat,
            lon2_deg=other.lon,
        )

        if solved is not None:
            return solved[0]

        # Near-antipodal fallback: sphere of the quadratic mean radius
        return _haversine(
            lat1_rad=math.radians(self.lat),
            lon1_rad=math.radians(self.lon),
            lat2_rad=math.radians(other.lat),
            lon2_rad=math.radians(other.lon),
            radius=resolved.quadratic_mean_radius,
        )

    def bearing(self, other: "Coordinate", planet: Optional[Planet] = None) -> float:
        """
        Computes the initial bearing toward another coordinate.

        Args:
            other (Coordinate): Target coordinate.
            planet (Optional[Planet], optional): Explicit planet
                override. Defaults to None.

        Returns:
            float: Initial bearing in degrees, in [0, 360).
        """

        resolved: Planet = self._planet(planet=planet)

        if resolved.is_sphere:
            return _sphere_bearing(
                lat1_rad=math.radians(self.lat),
                lon1_rad=math.radians(self.lon),
                lat2_rad=math.radians(other.lat),
                lon2_rad=math.radians(other.lon),
            )

        solved: Optional[tuple[float, float, float]] = _vincenty_inverse(
            planet=resolved,
            lat1_deg=self.lat,
            lon1_deg=self.lon,
            lat2_deg=other.lat,
            lon2_deg=other.lon,
        )

        if solved is not None:
            return solved[1]

        # Near-antipodal fallback: sphere bearing
        return _sphere_bearing(
            lat1_rad=math.radians(self.lat),
            lon1_rad=math.radians(self.lon),
            lat2_rad=math.radians(other.lat),
            lon2_rad=math.radians(other.lon),
        )

    def destination(
        self,
        bearing_deg: float,
        distance_m: float,
        planet: Optional[Planet] = None,
    ) -> "Coordinate":
        """
        Computes the coordinate reached by traveling a surface distance
        along an initial bearing.

        Args:
            bearing_deg (float): Initial bearing in degrees.
            distance_m (float): Distance along the surface in meters.
            planet (Optional[Planet], optional): Explicit planet
                override. Defaults to None.

        Returns:
            Coordinate: The destination coordinate (radius set to the
            surface radius at the destination latitude).

        Raises:
            TypeError: If bearing or distance is not a real number.
            ValueError: If the distance is negative.
        """

        resolved: Planet = self._planet(planet=planet)

        bearing_value: float = _require_angle(name="bearing", value=bearing_deg)
        distance_value: float = _require_angle(name="distance", value=distance_m)

        if distance_value < 0.0:
            raise ValueError(f"Distance must be non-negative, got {distance_value}")

        if resolved.is_sphere:
            lat2, lon2 = _sphere_destination(
                lat1_rad=math.radians(self.lat),
                lon1_rad=math.radians(self.lon),
                bearing_deg=bearing_value,
                distance_m=distance_value,
                radius=resolved.equatorial_radius,
            )
        else:
            solved: Optional[tuple[float, float]] = _vincenty_direct(
                planet=resolved,
                lat1_deg=self.lat,
                lon1_deg=self.lon,
                bearing_deg=bearing_value,
                distance_m=distance_value,
            )

            if solved is None:
                # Near-antipodal fallback: sphere destination
                lat2, lon2 = _sphere_destination(
                    lat1_rad=math.radians(self.lat),
                    lon1_rad=math.radians(self.lon),
                    bearing_deg=bearing_value,
                    distance_m=distance_value,
                    radius=resolved.quadratic_mean_radius,
                )
            else:
                lat2, lon2 = solved

        destination_lat: float = math.degrees(lat2)

        return Coordinate(
            lat=destination_lat,
            lon=math.degrees(lon2),
            radius=resolved.surface_radius(lat_deg=destination_lat),
        )

    def interpolate(
        self,
        other: "Coordinate",
        t: float,
        planet: Optional[Planet] = None,
    ) -> "Coordinate":
        """
        Interpolates along the great-circle path toward another
        coordinate. t = 0 yields self, t = 1 yields other; radii
        interpolate linearly.

        Args:
            other (Coordinate): Target coordinate.
            t (float): Interpolation parameter.
            planet (Optional[Planet], optional): Explicit planet
                override. Defaults to None.

        Returns:
            Coordinate: The interpolated coordinate.

        Raises:
            TypeError: If t is not a real number.
        """

        if isinstance(t, bool) or not isinstance(t, (int, float)):
            raise TypeError(f"Interpolation t must be a number, got '{type(t).__name__}'")

        clamped_t: float = min(1.0, max(0.0, float(t)))

        self._planet(planet=planet)  # Resolves / validates the session planet

        v1: tuple[float, float, float] = _to_unit_vector(lat_deg=self.lat, lon_deg=self.lon)
        v2: tuple[float, float, float] = _to_unit_vector(lat_deg=other.lat, lon_deg=other.lon)

        dot: float = max(-1.0, min(1.0, sum(a * b for a, b in zip(v1, v2))))
        omega: float = math.acos(dot)

        if omega < 1e-12:
            direction: tuple[float, float, float] = v1
        else:
            sin_omega: float = math.sin(omega)

            mixed: list[float] = [
                (
                    math.sin((1.0 - clamped_t) * omega) * a
                    + math.sin(clamped_t * omega) * b
                ) / sin_omega
                for a, b in zip(v1, v2)
            ]

            direction = (mixed[0], mixed[1], mixed[2])

        lat_mid: float
        lon_mid: float
        lat_mid, lon_mid = _from_unit_vector(vector=direction)

        return Coordinate(
            lat=lat_mid,
            lon=lon_mid,
            radius=self.radius + (other.radius - self.radius) * clamped_t,
        )

    def surface_radius(self, planet: Optional[Planet] = None) -> float:
        """
        Returns the planet surface radius at this coordinate's latitude.

        Args:
            planet (Optional[Planet], optional): Explicit planet
                override. Defaults to None.

        Returns:
            float: Surface radius in meters.
        """

        return self._planet(planet=planet).surface_radius(lat_deg=self.lat)

    def altitude(self, planet: Optional[Planet] = None) -> float:
        """
        Returns the height above sea level at this coordinate.

        When a sea-level layer is registered (register_sea_level) and
        has data around this coordinate, the interpolated sea-surface
        radius is the reference; otherwise the geometric surface
        radius at this latitude applies. Negative values are
        underground / underwater.

        Args:
            planet (Optional[Planet], optional): Explicit planet
                override. Defaults to None.

        Returns:
            float: Altitude in meters.
        """

        surface: Optional[Any] = get_sea_level_surface()

        if surface is not None:
            sea_radius: Optional[float] = surface.radius_at(
                lat_deg=self.lat, lon_deg=self.lon
            )

            if sea_radius is not None:
                return self.radius - sea_radius

        return self.radius - self.surface_radius(planet=planet)

    def chord_distance(self, other: "Coordinate") -> float:
        """
        Computes the straight-line (chord) distance through the planet
        to another coordinate. Planet-independent; monotonic in the
        great-circle distance, so it serves fast path comparisons.

        Args:
            other (Coordinate): Target coordinate.

        Returns:
            float: Chord distance in meters.
        """

        x1: float
        y1: float
        z1: float
        x2: float
        y2: float
        z2: float

        x1, y1, z1 = _to_unit_vector(lat_deg=self.lat, lon_deg=self.lon)
        x2, y2, z2 = _to_unit_vector(lat_deg=other.lat, lon_deg=other.lon)

        dx: float = self.radius * x1 - other.radius * x2
        dy: float = self.radius * y1 - other.radius * y2
        dz: float = self.radius * z1 - other.radius * z2

        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def __eq__(self, other: Any) -> bool:
        """
        Checks equality with another Coordinate.

        Args:
            other (Any): Object to compare with.

        Returns:
            bool: True when all three components match exactly.
        """

        if not isinstance(other, Coordinate):
            return False

        return (
            self.lat == other.lat
            and self.lon == other.lon
            and self.radius == other.radius
        )

    def __repr__(self) -> str:
        """
        Returns string representation of the Coordinate.

        Returns:
            str: Coordinate constructor string.
        """

        return f"Coordinate({self.lat}, {self.lon}, {self.radius})"


# Free-function geodesy builtins (raw-argument callables living in a
# home_types module, so the evaluator passes unwrapped payloads)


def distance(a: Coordinate, b: Coordinate, planet: Optional[Planet] = None) -> float:
    """
    Surface distance between two coordinates, in meters.

    Args:
        a (Coordinate): Origin coordinate.
        b (Coordinate): Target coordinate.
        planet (Optional[Planet], optional): Explicit planet override.

    Returns:
        float: Distance in meters.
    """

    return a.distance(other=b, planet=planet)


def bearing(a: Coordinate, b: Coordinate, planet: Optional[Planet] = None) -> float:
    """
    Initial bearing from one coordinate toward another, in degrees.

    Args:
        a (Coordinate): Origin coordinate.
        b (Coordinate): Target coordinate.
        planet (Optional[Planet], optional): Explicit planet override.

    Returns:
        float: Bearing in degrees, in [0, 360).
    """

    return a.bearing(other=b, planet=planet)


def destination(
    a: Coordinate,
    bearing_deg: float,
    distance_m: float,
    planet: Optional[Planet] = None,
) -> Coordinate:
    """
    Destination coordinate reached from an origin traveling a surface
    distance along a bearing.

    Args:
        a (Coordinate): Origin coordinate.
        bearing_deg (float): Initial bearing in degrees.
        distance_m (float): Distance along the surface in meters.
        planet (Optional[Planet], optional): Explicit planet override.

    Returns:
        Coordinate: The destination coordinate.
    """

    return a.destination(bearing_deg=bearing_deg, distance_m=distance_m, planet=planet)


def interpolate(
    a: Coordinate,
    b: Coordinate,
    t: float,
    planet: Optional[Planet] = None,
) -> Coordinate:
    """
    Interpolates along the great-circle path between two coordinates.

    Args:
        a (Coordinate): Start coordinate (t = 0).
        b (Coordinate): End coordinate (t = 1).
        t (float): Interpolation parameter.
        planet (Optional[Planet], optional): Explicit planet override.

    Returns:
        Coordinate: The interpolated coordinate.
    """

    return a.interpolate(other=b, t=t, planet=planet)


def surface_radius(c: Coordinate, planet: Optional[Planet] = None) -> float:
    """
    Planet surface radius at a coordinate's latitude, in meters.

    Args:
        c (Coordinate): The coordinate.
        planet (Optional[Planet], optional): Explicit planet override.

    Returns:
        float: Surface radius in meters.
    """

    return c.surface_radius(planet=planet)


def altitude(c: Coordinate, planet: Optional[Planet] = None) -> float:
    """
    Height of a coordinate above the planet surface, in meters.

    Args:
        c (Coordinate): The coordinate.
        planet (Optional[Planet], optional): Explicit planet override.

    Returns:
        float: Altitude in meters (negative underground).
    """

    return c.altitude(planet=planet)


def geocentric_latitude(c: Coordinate, planet: Optional[Planet] = None) -> float:
    """
    Geocentric latitude of the surface point below a coordinate.

    Args:
        c (Coordinate): The coordinate.
        planet (Optional[Planet], optional): Explicit planet override.

    Returns:
        float: Geocentric latitude in degrees.
    """

    resolved: Planet = planet if planet is not None else get_current_planet()

    return resolved.geocentric_latitude(lat_deg=c.lat)
