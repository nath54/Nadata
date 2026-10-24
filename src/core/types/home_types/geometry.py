"""
Geometry primitives for Nadata.

Provides the Polyline and Polygon value types over Coordinates, with
the measurement and predicate layer of the map plan (Phase B):

- Polyline: geodesic length, point lookup by distance or fraction,
  distance to the path.
- Polygon: spherical-excess area, geodesic perimeter, vertex centroid,
  antimeridian-safe point-in-polygon (planar even-odd test on an
  unwrapped equirectangular projection), distance to the boundary, and
  the inradius / circumradius fast-path radii used by the Phase C
  spatial index.

Sphere math is closed-form; spheroid math uses Vincenty per edge and
sphere formulas on authalic / quadratic-mean radii (documented
approximations). Shape constructor sugar (circle, ellipse, regular
polygon) produces Polygons.

Free functions in this module receive raw payloads from the evaluator
(home_types module convention); method-style members on the types are
the primary API and delegate to the same logic.
"""

# Import Modules
from __future__ import annotations

from typing import Any, Optional
import math

from src.core.types.home_types.coordinate import (
    Coordinate,
    _from_unit_vector,
    _to_unit_vector,
    normalize_lon,
    wrap_lon_delta,
)
from src.core.types.home_types.planet import Planet, get_current_planet



def _require_coordinate_list(name: str, vertices: Any) -> list[Coordinate]:
    """
    Validates a vertex list argument.

    Args:
        name (str): Argument name for error messages.
        vertices (Any): Candidate vertex list.

    Returns:
        list[Coordinate]: The validated vertex list.

    Raises:
        TypeError: If the argument is not a list of Coordinates.
    """

    if isinstance(vertices, tuple):
        vertices = list(vertices)

    if not isinstance(vertices, list):
        raise TypeError(
            f"{name} must be a list of Coordinate, got '{type(vertices).__name__}'"
        )

    for index, vertex in enumerate(vertices):
        if not isinstance(vertex, Coordinate):
            raise TypeError(
                f"{name}[{index}] must be a Coordinate, "
                f"got '{type(vertex).__name__}'"
            )

    return vertices


def _unit_cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Computes the cross product of two unit vectors.

    Args:
        a (tuple[float, float, float]): First unit vector.
        b (tuple[float, float, float]): Second unit vector.

    Returns:
        tuple[float, float, float]: The cross product.
    """

    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _unit_dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    """
    Computes the dot product of two vectors.

    Args:
        a (tuple[float, float, float]): First vector.
        b (tuple[float, float, float]): Second vector.

    Returns:
        float: The dot product.
    """

    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _central_angle(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    """
    Computes the angle between two unit vectors.

    Args:
        a (tuple[float, float, float]): First unit vector.
        b (tuple[float, float, float]): Second unit vector.

    Returns:
        float: Angle in radians, in [0, pi].
    """

    cross: tuple[float, float, float] = _unit_cross(a=a, b=b)

    return math.atan2(
        math.sqrt(cross[0] ** 2 + cross[1] ** 2 + cross[2] ** 2),
        _unit_dot(a=a, b=b),
    )


def _resolve_planet(planet: Optional[Planet]) -> Planet:
    """
    Resolves the effective planet for a computation.

    Args:
        planet (Optional[Planet]): Explicit override, or None to use
            the session current planet.

    Returns:
        Planet: The planet to compute against.
    """

    return planet if planet is not None else get_current_planet()


def _unwrapped_lons(vertices: list[Coordinate]) -> list[float]:
    """
    Unwraps vertex longitudes into a contiguous sequence (consecutive
    deltas within 180 degrees), so polygons crossing the antimeridian
    keep their true shape.

    Args:
        vertices (list[Coordinate]): The polygon or polyline vertices.

    Returns:
        list[float]: Contiguous longitudes in degrees.
    """

    if not vertices:
        return []

    lons: list[float] = [vertices[0].lon]

    for vertex in vertices[1:]:
        lons.append(lons[-1] + wrap_lon_delta(lon_delta_deg=vertex.lon - lons[-1]))

    return lons


def _edge_angle(a: Coordinate, b: Coordinate) -> float:
    """
    Computes the angular separation of two coordinates on the unit
    sphere (their geodetic directions treated as unit vectors).

    Args:
        a (Coordinate): First coordinate.
        b (Coordinate): Second coordinate.

    Returns:
        float: Angle in radians.
    """

    return _central_angle(
        a=_to_unit_vector(lat_deg=a.lat, lon_deg=a.lon),
        b=_to_unit_vector(lat_deg=b.lat, lon_deg=b.lon),
    )


def _edge_length(a: Coordinate, b: Coordinate, planet: Planet) -> float:
    """
    Computes the geodesic length of one edge.

    Args:
        a (Coordinate): First endpoint.
        b (Coordinate): Second endpoint.
        planet (Planet): Effective planet.

    Returns:
        float: Edge length in meters.
    """

    if planet.is_sphere:
        return _edge_angle(a=a, b=b) * planet.equatorial_radius

    return a.distance(other=b, planet=planet)


def _segment_distance(
    point: Coordinate,
    a: Coordinate,
    b: Coordinate,
) -> float:
    """
    Computes the angular distance from a point to the minor great-
    circle arc between two coordinates, on the unit sphere.

    Args:
        point (Coordinate): The query point.
        a (Coordinate): Segment start.
        b (Coordinate): Segment end.

    Returns:
        float: Angular distance in radians.
    """

    p: tuple[float, float, float] = _to_unit_vector(lat_deg=point.lat, lon_deg=point.lon)
    u: tuple[float, float, float] = _to_unit_vector(lat_deg=a.lat, lon_deg=a.lon)
    v: tuple[float, float, float] = _to_unit_vector(lat_deg=b.lat, lon_deg=b.lon)

    # Fall back to endpoints for degenerate or antipodal segments
    angle_ab: float = _central_angle(a=u, b=v)

    if angle_ab < 1e-15 or abs(angle_ab - math.pi) < 1e-9:
        return min(
            _central_angle(a=p, b=u),
            _central_angle(a=p, b=v),
        )

    normal: tuple[float, float, float] = _unit_cross(a=u, b=v)
    normal_norm: float = math.sqrt(_unit_dot(a=normal, b=normal))
    normal = (normal[0] / normal_norm, normal[1] / normal_norm, normal[2] / normal_norm)

    # Project the point onto the great-circle plane
    dot_pn: float = _unit_dot(a=p, b=normal)
    projected: tuple[float, float, float] = (
        p[0] - dot_pn * normal[0],
        p[1] - dot_pn * normal[1],
        p[2] - dot_pn * normal[2],
    )

    projected_norm: float = math.sqrt(_unit_dot(a=projected, b=projected))

    if projected_norm < 1e-15:
        # The point sits on a pole of the great circle: nearest endpoint
        return min(
            _central_angle(a=p, b=u),
            _central_angle(a=p, b=v),
        )

    projected = (
        projected[0] / projected_norm,
        projected[1] / projected_norm,
        projected[2] / projected_norm,
    )

    # Inside the minor arc when closer to both endpoints than they are
    # to each other
    angle_ap: float = _central_angle(a=u, b=projected)
    angle_bp: float = _central_angle(a=v, b=projected)

    if angle_ap <= angle_ab + 1e-12 and angle_bp <= angle_ab + 1e-12:
        return _central_angle(a=p, b=projected)

    return min(
        _central_angle(a=p, b=u),
        _central_angle(a=p, b=v),
    )


def _segment_distance_meters(
    point: Coordinate,
    a: Coordinate,
    b: Coordinate,
    planet: Planet,
) -> float:
    """
    Computes the distance from a point to a boundary segment.

    Args:
        point (Coordinate): The query point.
        a (Coordinate): Segment start.
        b (Coordinate): Segment end.
        planet (Planet): Effective planet.

    Returns:
        float: Distance in meters (spheroid: sphere approximation on
        the quadratic mean radius).
    """

    radius: float = (
        planet.equatorial_radius
        if planet.is_sphere
        else planet.quadratic_mean_radius
    )

    return _segment_distance(point=point, a=a, b=b) * radius


def _point_in_ring(lon: float, lat: float, lons: list[float], lats: list[float]) -> bool:
    """
    Planar even-odd point-in-ring test on the equirectangular plane.

    Args:
        lon (float): Query longitude (already translated).
        lat (float): Query latitude.
        lons (list[float]): Ring longitudes (contiguous unwrap).
        lats (list[float]): Ring latitudes.

    Returns:
        bool: True when the point is inside the ring.
    """

    inside: bool = False
    count: int = len(lons)
    j: int = count - 1

    for i in range(count):
        yi: float = lats[i]
        yj: float = lats[j]

        if (yi > lat) != (yj > lat):
            x_intersect: float = (
                (lons[j] - lons[i]) * (lat - yi) / (yj - yi) + lons[i]
            )

            if lon < x_intersect:
                inside = not inside

        j = i

    return inside



class Polyline:
    """
    A path over the planet: an ordered sequence of Coordinates.

    Attributes:
        vertices (list[Coordinate]): The path vertices.
    """

    def __init__(self, vertices: Any) -> None:
        """
        Initializes the Polyline instance.

        Args:
            vertices (Any): The path vertices (list of Coordinate).

        Raises:
            TypeError: If vertices is not a list of Coordinates.
        """

        self.vertices: list[Coordinate] = _require_coordinate_list(
            name="Polyline vertices", vertices=vertices
        )

    def _planet(self, planet: Optional[Planet]) -> Planet:
        """
        Resolves the effective planet.

        Args:
            planet (Optional[Planet]): Explicit override.

        Returns:
            Planet: The planet to compute against.
        """

        return _resolve_planet(planet=planet)

    def num_vertices(self) -> int:
        """
        Returns the vertex count.

        Returns:
            int: Number of vertices.
        """

        return len(self.vertices)

    def vertex(self, index: int) -> Coordinate:
        """
        Returns the vertex at an index (negative indices count from
        the end).

        Args:
            index (int): Vertex index.

        Returns:
            Coordinate: The vertex.

        Raises:
            IndexError: If the index is out of range.
        """

        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError(f"Vertex index must be an integer, got '{type(index).__name__}'")

        return self.vertices[index]

    def length(self, planet: Optional[Planet] = None) -> float:
        """
        Computes the geodesic length along the path.

        Args:
            planet (Optional[Planet], optional): Explicit planet
                override. Defaults to None.
            center (Optional[Coordinate], optional): Explicit anchor
                override. Defaults to the centroid.

        Returns:
            float: Length in meters.
        """

        resolved: Planet = self._planet(planet=planet)

        total: float = 0.0

        for i in range(len(self.vertices) - 1):
            total += _edge_length(
                a=self.vertices[i],
                b=self.vertices[i + 1],
                planet=resolved,
            )

        return total

    def point_at_distance(
        self,
        distance_m: float,
        planet: Optional[Planet] = None,
    ) -> Coordinate:
        """
        Returns the point at a along-path distance (walking the
        geodesic edges).

        Args:
            distance_m (float): Distance along the path in meters.
            planet (Optional[Planet], optional): Explicit planet
                override. Defaults to None.

        Returns:
            Coordinate: The point at the distance.

        Raises:
            ValueError: If the distance is negative or beyond the path
                length, or the path is empty.
        """

        resolved: Planet = self._planet(planet=planet)

        if not self.vertices:
            raise ValueError("Cannot locate a point on an empty Polyline")

        if not isinstance(distance_m, (int, float)) or isinstance(distance_m, bool):
            raise TypeError(
                f"Distance must be a number, got '{type(distance_m).__name__}'"
            )

        remaining: float = float(distance_m)

        if remaining < 0.0:
            raise ValueError(f"Distance must be non-negative, got {remaining}")

        if remaining > self.length(planet=planet):
            raise ValueError(
                f"Distance {distance_m} exceeds the path length "
                f"{self.length(planet=planet)}"
            )

        last_index: int = len(self.vertices) - 2

        for i in range(len(self.vertices) - 1):
            edge: float = _edge_length(
                a=self.vertices[i], b=self.vertices[i + 1], planet=resolved
            )

            if remaining <= edge or i == last_index:
                # Clamp absorbs float drift at the path end
                fraction: float = 0.0 if edge == 0.0 else min(
                    1.0, remaining / edge
                )

                return self.vertices[i].interpolate(
                    other=self.vertices[i + 1], t=fraction, planet=planet
                )

            remaining -= edge

        raise ValueError("Cannot locate a point on an empty Polyline")

    def point_at(self, t: float, planet: Optional[Planet] = None) -> Coordinate:
        """
        Returns the point at a fraction of the total path length.

        Args:
            t (float): Fraction in [0, 1].
            planet (Optional[Planet], optional): Explicit planet
                override. Defaults to None.

        Returns:
            Coordinate: The point at the fraction.

        Raises:
            ValueError: If t is outside [0, 1] or the path is empty.
        """

        if not isinstance(t, (int, float)) or isinstance(t, bool):
            raise TypeError(f"t must be a number, got '{type(t).__name__}'")

        if not 0.0 <= float(t) <= 1.0:
            raise ValueError(f"t must be in [0, 1], got {t}")

        return self.point_at_distance(
            distance_m=float(t) * self.length(planet=planet), planet=planet
        )

    def distance_to(self, point: Coordinate, planet: Optional[Planet] = None) -> float:
        """
        Computes the distance from a coordinate to the nearest point
        of the path.

        Args:
            point (Coordinate): The query coordinate.
            planet (Optional[Planet], optional): Explicit planet
                override. Defaults to None.

        Returns:
            float: Distance in meters.

        Raises:
            ValueError: If the path is empty.
        """

        resolved: Planet = self._planet(planet=planet)

        if not self.vertices:
            raise ValueError("Cannot measure distance to an empty Polyline")

        best: Optional[float] = None

        for i in range(len(self.vertices) - 1):
            candidate: float = _segment_distance_meters(
                point=point,
                a=self.vertices[i],
                b=self.vertices[i + 1],
                planet=resolved,
            )

            if best is None or candidate < best:
                best = candidate

        # A single-vertex path measures the distance to the vertex
        if best is None:
            best = _segment_distance_meters(
                point=point,
                a=self.vertices[0],
                b=self.vertices[0],
                planet=resolved,
            )

        return best

    def __eq__(self, other: Any) -> bool:
        """
        Checks equality with another Polyline.

        Args:
            other (Any): Object to compare with.

        Returns:
            bool: True when the vertex lists match.
        """

        if not isinstance(other, Polyline):
            return False

        return self.vertices == other.vertices

    def __repr__(self) -> str:
        """
        Returns string representation of the Polyline.

        Returns:
            str: Polyline constructor string.
        """

        return f"Polyline({self.vertices!r})"


class Polygon:
    """
    A closed ring over the planet: an ordered sequence of Coordinates
    (the closing edge is implicit). A duplicate closing vertex is
    stripped on construction.

    Attributes:
        vertices (list[Coordinate]): The ring vertices.
    """

    def __init__(self, vertices: Any) -> None:
        """
        Initializes the Polygon instance.

        Args:
            vertices (Any): The ring vertices (list of Coordinate).

        Raises:
            TypeError: If vertices is not a list of Coordinates.
            ValueError: If the vertex count is 1 or 2.
        """

        ring: list[Coordinate] = _require_coordinate_list(
            name="Polygon vertices", vertices=vertices
        )

        # Strip an explicitly closed ring
        if len(ring) >= 2 and ring[0] == ring[-1]:
            ring = ring[:-1]

        if 0 < len(ring) < 3:
            raise ValueError(
                f"A polygon needs 0 (empty) or at least 3 vertices, got {len(ring)}"
            )

        self.vertices: list[Coordinate] = ring

    def _planet(self, planet: Optional[Planet]) -> Planet:
        """
        Resolves the effective planet.

        Args:
            planet (Optional[Planet]): Explicit override.

        Returns:
            Planet: The planet to compute against.
        """

        return _resolve_planet(planet=planet)

    def num_vertices(self) -> int:
        """
        Returns the vertex count.

        Returns:
            int: Number of vertices.
        """

        return len(self.vertices)

    def vertex(self, index: int) -> Coordinate:
        """
        Returns the vertex at an index (negative indices count from
        the end).

        Args:
            index (int): Vertex index.

        Returns:
            Coordinate: The vertex.

        Raises:
            IndexError: If the index is out of range.
        """

        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError(f"Vertex index must be an integer, got '{type(index).__name__}'")

        return self.vertices[index]

    def area(self, planet: Optional[Planet] = None) -> float:
        """
        Computes the enclosed area.

        Sphere: exact spherical excess (Chamberlain & Duquette). 
        Spheroid: the same formula on the authalic (equal-area)
        radius -- an approximation (see the map plan, open points).

        Args:
            planet (Optional[Planet], optional): Explicit planet
                override. Defaults to None.

        Returns:
            float: Area in square meters (0 for the empty polygon).
        """

        resolved: Planet = self._planet(planet=planet)

        if len(self.vertices) < 3:
            return 0.0

        lons: list[float] = _unwrapped_lons(vertices=self.vertices)

        total: float = 0.0

        for i in range(len(self.vertices)):
            j: int = (i + 1) % len(self.vertices)

            lat_i: float = math.radians(self.vertices[i].lat)
            lat_j: float = math.radians(self.vertices[j].lat)

            total += math.radians(lons[j] - lons[i]) * (
                2.0 + math.sin(lat_i) + math.sin(lat_j)
            )

        return abs(total) * resolved.authalic_radius**2 / 2.0

    def perimeter(self, planet: Optional[Planet] = None) -> float:
        """
        Computes the geodesic length of the closed boundary.

        Args:
            planet (Optional[Planet], optional): Explicit planet
                override. Defaults to None.

        Returns:
            float: Perimeter in meters.
        """

        resolved: Planet = self._planet(planet=planet)

        if len(self.vertices) < 3:
            return 0.0

        total: float = 0.0

        for i in range(len(self.vertices)):
            total += _edge_length(
                a=self.vertices[i],
                b=self.vertices[(i + 1) % len(self.vertices)],
                planet=resolved,
            )

        return total

    def centroid(self, planet: Optional[Planet] = None) -> Coordinate:
        """
        Returns the vertex centroid: the normalized mean of the vertex
        direction vectors, at surface radius (a fast-path anchor, not
        an area centroid).

        Args:
            planet (Optional[Planet], optional): Explicit planet
                override. Defaults to None.

        Returns:
            Coordinate: The centroid coordinate.

        Raises:
            ValueError: If the polygon is empty or the vertices cancel
                out (e.g. antipodal halves).
        """

        resolved: Planet = self._planet(planet=planet)

        if not self.vertices:
            raise ValueError("Cannot compute the centroid of an empty Polygon")

        x: float = 0.0
        y: float = 0.0
        z: float = 0.0

        for vertex in self.vertices:
            vx, vy, vz = _to_unit_vector(lat_deg=vertex.lat, lon_deg=vertex.lon)

            x += vx
            y += vy
            z += vz

        norm: float = math.sqrt(x * x + y * y + z * z)

        if norm < 1e-15:
            raise ValueError("Polygon vertices cancel out; centroid is undefined")

        lat: float
        lon: float
        lat, lon = _from_unit_vector(vector=(x / norm, y / norm, z / norm))

        return Coordinate(
            lat=lat, lon=lon, radius=resolved.surface_radius(lat_deg=lat)
        )

    def contains(self, point: Coordinate, planet: Optional[Planet] = None) -> bool:
        """
        Tests whether a coordinate lies inside the ring.

        Uses a planar even-odd test on an unwrapped equirectangular
        projection, probing the point at longitudes shifted by -360,
        0, and +360 so antimeridian-crossing rings test correctly.
        Region-sized polygons are exact for practical purposes; polar
        cap shapes are an approximation (map plan, open points).

        Args:
            point (Coordinate): The query coordinate.
            planet (Optional[Planet], optional): Unused; accepted for
                API symmetry. Defaults to None.

        Returns:
            bool: True when the point is inside (boundary is inside).

        Raises:
            ValueError: If the polygon is empty.
        """

        if not self.vertices:
            raise ValueError("An empty Polygon contains nothing")

        lons: list[float] = _unwrapped_lons(vertices=self.vertices)
        lats: list[float] = [vertex.lat for vertex in self.vertices]

        for shift in (-360.0, 0.0, 360.0):
            if _point_in_ring(
                lon=point.lon + shift, lat=point.lat, lons=lons, lats=lats
            ):
                return True

        return False

    def distance_to(self, point: Coordinate, planet: Optional[Planet] = None) -> float:
        """
        Computes the distance from a coordinate to the polygon (0 when
        the point is inside).

        Args:
            point (Coordinate): The query coordinate.
            planet (Optional[Planet], optional): Explicit planet
                override. Defaults to None.

        Returns:
            float: Distance in meters.

        Raises:
            ValueError: If the polygon is empty.
        """

        resolved: Planet = self._planet(planet=planet)

        if not self.vertices:
            raise ValueError("Cannot measure distance to an empty Polygon")

        if self.contains(point=point):
            return 0.0

        best: Optional[float] = None
        count: int = len(self.vertices)

        for i in range(count):
            candidate: float = _segment_distance_meters(
                point=point,
                a=self.vertices[i],
                b=self.vertices[(i + 1) % count],
                planet=resolved,
            )

            if best is None or candidate < best:
                best = candidate

        assert best is not None

        return best

    def circumradius(
        self,
        planet: Optional[Planet] = None,
        center: Optional[Coordinate] = None,
    ) -> float:
        """
        Returns the geodesic distance from the centroid (or an
        explicit anchor) to the farthest vertex. Points farther than
        this from the centroid are definitely outside (Phase C fast
        path).

        Args:
            planet (Optional[Planet], optional): Explicit planet
                override. Defaults to None.
            center (Optional[Coordinate], optional): Explicit anchor
                override. Defaults to the centroid.

        Returns:
            float: Circumradius in meters.

        Raises:
            ValueError: If the polygon is empty.
        """

        resolved: Planet = self._planet(planet=planet)

        if not self.vertices:
            raise ValueError("An empty Polygon has no circumradius")

        anchor: Coordinate = (
            center if center is not None else self.centroid(planet=planet)
        )

        return max(
            anchor.distance(other=vertex, planet=resolved)
            for vertex in self.vertices
        )

    def inradius(
        self,
        planet: Optional[Planet] = None,
        center: Optional[Coordinate] = None,
    ) -> float:
        """
        Returns the distance from the centroid to the nearest boundary
        point. Points closer than this to the centroid are definitely
        inside (Phase C fast path). 0 when the centroid falls outside
        the ring (pathological shapes).

        Args:
            planet (Optional[Planet], optional): Explicit planet
                override. Defaults to None.

        Returns:
            float: Inradius in meters.

        Raises:
            ValueError: If the polygon is empty.
        """

        resolved: Planet = self._planet(planet=planet)

        if not self.vertices:
            raise ValueError("An empty Polygon has no inradius")

        anchor: Coordinate = (
            center if center is not None else self.centroid(planet=planet)
        )

        if not self.contains(point=anchor):
            return 0.0

        best: Optional[float] = None
        count: int = len(self.vertices)

        for i in range(count):
            candidate: float = _segment_distance_meters(
                point=anchor,
                a=self.vertices[i],
                b=self.vertices[(i + 1) % count],
                planet=resolved,
            )

            if best is None or candidate < best:
                best = candidate

        assert best is not None

        return best

    def __eq__(self, other: Any) -> bool:
        """
        Checks equality with another Polygon.

        Args:
            other (Any): Object to compare with.

        Returns:
            bool: True when the vertex lists match.
        """

        if not isinstance(other, Polygon):
            return False

        return self.vertices == other.vertices

    def __repr__(self) -> str:
        """
        Returns string representation of the Polygon.

        Returns:
            str: Polygon constructor string.
        """

        return f"Polygon({self.vertices!r})"


# Shape constructor sugar


def circle(
    center: Coordinate,
    radius_m: float,
    segments: int = 64,
    planet: Optional[Planet] = None,
) -> Polygon:
    """
    Builds a geodesic circle: all vertices at a surface distance from
    the center.

    Args:
        center (Coordinate): The circle center.
        radius_m (float): Radius in meters.
        segments (int, optional): Vertex count. Defaults to 64.
        planet (Optional[Planet], optional): Explicit planet override.

    Returns:
        Polygon: The circle polygon.

    Raises:
        TypeError: If segments is not an integer.
        ValueError: If radius is not positive or segments < 3.
    """

    if isinstance(segments, bool) or not isinstance(segments, int):
        raise TypeError(f"Segments must be an integer, got '{type(segments).__name__}'")

    if segments < 3:
        raise ValueError(f"A circle needs at least 3 segments, got {segments}")

    vertices: list[Coordinate] = [
        center.destination(
            bearing_deg=360.0 * i / segments,
            distance_m=radius_m,
            planet=planet,
        )
        for i in range(segments)
    ]

    return Polygon(vertices=vertices)


def ellipse(
    center: Coordinate,
    radius_a_m: float,
    radius_b_m: float,
    bearing_deg: float = 0.0,
    segments: int = 64,
    planet: Optional[Planet] = None,
) -> Polygon:
    """
    Builds a parametric ellipse: the vertex at angle t sits at polar
    radius r(t) = (ra * rb) / sqrt((rb cos t)^2 + (ra sin t)^2) from
    the center, oriented along the bearing.

    Args:
        center (Coordinate): The ellipse center.
        radius_a_m (float): Semi-axis along the bearing, in meters.
        radius_b_m (float): Semi-axis perpendicular, in meters.
        bearing_deg (float, optional): Orientation of the a axis.
            Defaults to 0.0.
        segments (int, optional): Vertex count. Defaults to 64.
        planet (Optional[Planet], optional): Explicit planet override.

    Returns:
        Polygon: The ellipse polygon.

    Raises:
        TypeError: If segments is not an integer.
        ValueError: If a radius is not positive or segments < 3.
    """

    if isinstance(segments, bool) or not isinstance(segments, int):
        raise TypeError(f"Segments must be an integer, got '{type(segments).__name__}'")

    if segments < 3:
        raise ValueError(f"An ellipse needs at least 3 segments, got {segments}")

    if radius_a_m <= 0.0 or radius_b_m <= 0.0:
        raise ValueError(
            f"Semi-axes must be positive, got {radius_a_m} and {radius_b_m}"
        )

    vertices: list[Coordinate] = []

    for i in range(segments):
        theta: float = 2.0 * math.pi * i / segments
        cos_t: float = math.cos(theta)
        sin_t: float = math.sin(theta)

        polar_radius: float = (radius_a_m * radius_b_m) / math.sqrt(
            (radius_b_m * cos_t) ** 2 + (radius_a_m * sin_t) ** 2
        )

        vertices.append(
            center.destination(
                bearing_deg=bearing_deg + math.degrees(theta),
                distance_m=polar_radius,
                planet=planet,
            )
        )

    return Polygon(vertices=vertices)


def regular_polygon(
    center: Coordinate,
    radius_m: float,
    sides: int,
    rotation_deg: float = 0.0,
    planet: Optional[Planet] = None,
) -> Polygon:
    """
    Builds a regular polygon: sides vertices on a geodesic circle,
    rotated by an initial bearing.

    Args:
        center (Coordinate): The polygon center.
        radius_m (float): Circumradius in meters.
        sides (int): Vertex count (at least 3).
        rotation_deg (float, optional): Bearing of the first vertex.
            Defaults to 0.0.
        planet (Optional[Planet], optional): Explicit planet override.

    Returns:
        Polygon: The regular polygon.

    Raises:
        TypeError: If sides is not an integer.
        ValueError: If sides < 3 or radius is not positive.
    """

    if isinstance(sides, bool) or not isinstance(sides, int):
        raise TypeError(f"Sides must be an integer, got '{type(sides).__name__}'")

    if sides < 3:
        raise ValueError(f"A regular polygon needs at least 3 sides, got {sides}")

    vertices: list[Coordinate] = [
        center.destination(
            bearing_deg=rotation_deg + 360.0 * i / sides,
            distance_m=radius_m,
            planet=planet,
        )
        for i in range(sides)
    ]

    return Polygon(vertices=vertices)


# Free-function twins of the member API


def area(polygon: Polygon, planet: Optional[Planet] = None) -> float:
    """
    Enclosed area of a polygon, in square meters.

    Args:
        polygon (Polygon): The polygon.
        planet (Optional[Planet], optional): Explicit planet override.

    Returns:
        float: Area in square meters.
    """

    return polygon.area(planet=planet)


def perimeter(polygon: Polygon, planet: Optional[Planet] = None) -> float:
    """
    Geodesic boundary length of a polygon, in meters.

    Args:
        polygon (Polygon): The polygon.
        planet (Optional[Planet], optional): Explicit planet override.

    Returns:
        float: Perimeter in meters.
    """

    return polygon.perimeter(planet=planet)


def contains(
    polygon: Polygon,
    point: Coordinate,
    planet: Optional[Planet] = None,
) -> bool:
    """
    Tests whether a coordinate lies inside a polygon.

    Args:
        polygon (Polygon): The polygon.
        point (Coordinate): The query coordinate.
        planet (Optional[Planet], optional): Unused; API symmetry.

    Returns:
        bool: True when the point is inside.
    """

    return polygon.contains(point=point, planet=planet)


def polyline_length(
    polyline: Polyline,
    planet: Optional[Planet] = None,
) -> float:
    """
    Geodesic length of a polyline, in meters.

    Args:
        polyline (Polyline): The polyline.
        planet (Optional[Planet], optional): Explicit planet override.

    Returns:
        float: Length in meters.
    """

    return polyline.length(planet=planet)


def normalize_ring(vertices: Any) -> "Polygon":
    """
    Builds a Polygon from a vertex ring, stripping any explicit
    closing vertex.

    Args:
        vertices (Any): The ring vertices (list of Coordinate).

    Returns:
        Polygon: The normalized polygon.
    """

    return Polygon(vertices=vertices)
