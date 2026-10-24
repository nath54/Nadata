"""
Planet representations for Nadata.

Provides the Planet class describing the shape of the world coordinates
live on: a perfect sphere (single radius) or an oblate/prolate spheroid
(equatorial and polar radii). Also hosts the session-level current
planet, resolved from project settings (planet.kind, planet.radius,
planet.equatorial_radius, planet.polar_radius) and consulted by
geodesy builtins when no explicit planet override is passed.
"""

# Import Modules
from __future__ import annotations

from typing import Any, Optional
import math



# Reference radii (meters)
EARTH_EQUATORIAL_RADIUS: float = 6378137.0
EARTH_POLAR_RADIUS: float = 6356752.314245
EARTH_MEAN_RADIUS: float = 6371008.8

# Kind names for planet settings
KIND_SPHERE: str = "sphere"
KIND_ELLIPSOID: str = "ellipsoid"

# Session-level current planet (seeded by the loader from project
# settings; single-process assumption, see the map plan open points)
_current_planet: Optional["Planet"] = None



def _require_positive_length(name: str, value: Any) -> float:
    """
    Validates a length argument.

    Args:
        name (str): Argument name for error messages.
        value (Any): Candidate length.

    Returns:
        float: The validated length.

    Raises:
        TypeError: If the value is not a real number.
        ValueError: If the value is not strictly positive.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Planet {name} must be a number, got '{type(value).__name__}'")

    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"Planet {name} must be a finite positive length, got {value}")

    return float(value)



class Planet:
    """
    Describes the shape of a world: sphere or spheroid.

    Attributes:
        equatorial_radius (float): Equatorial radius in meters.
        polar_radius (Optional[float]): Polar radius in meters, or None
            for a perfect sphere.
    """

    def __init__(
        self,
        equatorial_radius: float = EARTH_MEAN_RADIUS,
        polar_radius: Optional[float] = None,
    ) -> None:
        """
        Initializes the Planet instance.

        Args:
            equatorial_radius (float, optional): Equatorial radius in
                meters. Defaults to the Earth mean radius.
            polar_radius (Optional[float], optional): Polar radius in
                meters. None yields a perfect sphere. Defaults to None.

        Raises:
            TypeError: If a radius is not a real number.
            ValueError: If a radius is not a finite positive length.
        """

        self.equatorial_radius: float = _require_positive_length(
            name="equatorial_radius", value=equatorial_radius
        )

        self.polar_radius: Optional[float] = (
            None
            if polar_radius is None
            else _require_positive_length(name="polar_radius", value=polar_radius)
        )

    @classmethod
    def sphere(cls, radius: float) -> "Planet":
        """
        Creates a perfect sphere planet.

        Args:
            radius (float): Sphere radius in meters.

        Returns:
            Planet: The sphere planet.
        """

        return cls(equatorial_radius=radius, polar_radius=None)

    @property
    def is_sphere(self) -> bool:
        """
        Returns whether the planet is a perfect sphere.

        Returns:
            bool: True when no distinct polar radius is set.
        """

        return self.polar_radius is None

    @property
    def polar_radius_value(self) -> float:
        """
        Returns the effective polar radius (equals the equatorial
        radius for spheres).

        Returns:
            float: Polar radius in meters.
        """

        return (
            self.equatorial_radius
            if self.polar_radius is None
            else self.polar_radius
        )

    @property
    def flattening(self) -> float:
        """
        Returns the flattening f = (a - b) / a.

        Returns:
            float: Flattening (0.0 for spheres).
        """

        return (self.equatorial_radius - self.polar_radius_value) / self.equatorial_radius

    @property
    def eccentricity_squared(self) -> float:
        """
        Returns the first eccentricity squared e^2 = 2f - f^2.

        Returns:
            float: Eccentricity squared (0.0 for spheres).
        """

        flattening: float = self.flattening

        return 2.0 * flattening - flattening * flattening

    @property
    def quadratic_mean_radius(self) -> float:
        """
        Returns the quadratic mean radius sqrt((a^2 + b^2) / 2), used as
        the sphere fallback for near-antipodal spheroid distances.

        Returns:
            float: Quadratic mean radius in meters.
        """

        squares: float = (
            self.equatorial_radius ** 2 + self.polar_radius_value ** 2
        )

        return math.sqrt(squares / 2.0)

    @property
    def authalic_radius(self) -> float:
        """
        Returns the authalic (equal-area) radius: the radius of the
        sphere with the same surface area as the spheroid. Used to
        scale spherical polygon areas to spheroids.

        Returns:
            float: Authalic radius in meters.
        """

        if self.is_sphere:
            return self.equatorial_radius

        e: float = math.sqrt(self.eccentricity_squared)

        if e == 0.0:
            return self.equatorial_radius

        surface_area: float = (
            2.0
            * math.pi
            * self.equatorial_radius ** 2
            * (1.0 + (1.0 - e * e) / e * math.atanh(e))
        )

        return math.sqrt(surface_area / (4.0 * math.pi))

    def surface_radius(self, lat_deg: float) -> float:
        """
        Returns the distance from the planet center to the surface at a
        geodetic latitude.

        Args:
            lat_deg (float): Geodetic latitude in degrees.

        Returns:
            float: Surface radius in meters.
        """

        if self.is_sphere:
            return self.equatorial_radius

        lat_rad: float = math.radians(lat_deg)
        sin_lat: float = math.sin(lat_rad)

        prime_vertical: float = self.equatorial_radius / math.sqrt(
            1.0 - self.eccentricity_squared * sin_lat * sin_lat
        )

        x: float = prime_vertical * math.cos(lat_rad)
        z: float = prime_vertical * (1.0 - self.eccentricity_squared) * sin_lat

        return math.hypot(x, z)

    def geocentric_latitude(self, lat_deg: float) -> float:
        """
        Converts a geodetic latitude to the geocentric latitude of the
        surface point below it.

        Args:
            lat_deg (float): Geodetic latitude in degrees.

        Returns:
            float: Geocentric latitude in degrees.
        """

        if self.is_sphere:
            return lat_deg

        ratio: float = (self.polar_radius_value ** 2) / (self.equatorial_radius ** 2)

        return math.degrees(math.atan(ratio * math.tan(math.radians(lat_deg))))

    def __eq__(self, other: Any) -> bool:
        """
        Checks equality with another Planet.

        Args:
            other (Any): Object to compare with.

        Returns:
            bool: True when both radii match.
        """

        if not isinstance(other, Planet):
            return False

        return (
            self.equatorial_radius == other.equatorial_radius
            and self.polar_radius == other.polar_radius
        )

    def __repr__(self) -> str:
        """
        Returns string representation of the Planet.

        Returns:
            str: Planet constructor string.
        """

        if self.is_sphere:
            return f"Planet({self.equatorial_radius})"

        return f"Planet({self.equatorial_radius}, {self.polar_radius})"


# Default planet: a perfect sphere of the Earth mean radius
DEFAULT_PLANET: Planet = Planet(equatorial_radius=EARTH_MEAN_RADIUS)


def get_current_planet() -> Planet:
    """
    Returns the session current planet, falling back to the default
    Earth sphere when no project has configured one.

    Returns:
        Planet: The current planet.
    """

    return _current_planet if _current_planet is not None else DEFAULT_PLANET


def set_current_planet(planet: Planet) -> None:
    """
    Replaces the session current planet.

    Args:
        planet (Planet): The new current planet.
    """

    global _current_planet

    if not isinstance(planet, Planet):
        raise TypeError(f"Expected Planet, got '{type(planet).__name__}'")

    _current_planet = planet


def planet_from_settings(settings: dict[str, Any]) -> Planet:
    """
    Builds a Planet from project settings.

    Recognized keys:
    - "planet.kind": "sphere" (default) or "ellipsoid".
    - "planet.radius": sphere radius (sphere kind).
    - "planet.equatorial_radius" / "planet.polar_radius": spheroid
      radii (ellipsoid kind; defaults to the WGS84 values).

    Args:
        settings (dict[str, Any]): The project settings mapping.

    Returns:
        Planet: The configured planet.

    Raises:
        ValueError: If the kind is unknown.
    """

    kind: str = settings.get("planet.kind", KIND_SPHERE)

    if kind == KIND_ELLIPSOID:
        return Planet(
            equatorial_radius=settings.get(
                "planet.equatorial_radius", EARTH_EQUATORIAL_RADIUS
            ),
            polar_radius=settings.get(
                "planet.polar_radius", EARTH_POLAR_RADIUS
            ),
        )

    if kind == KIND_SPHERE:
        return Planet(
            equatorial_radius=settings.get("planet.radius", EARTH_MEAN_RADIUS)
        )

    raise ValueError(
        f"Unknown planet kind '{kind}' (expected 'sphere' or 'ellipsoid')"
    )
