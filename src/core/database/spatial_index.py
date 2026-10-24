"""
Session-scoped spatial index for Nadata struct tables.

Indexes one geometry attribute (Polygon, Polyline, or Coordinate) of
one StructTable into quantized lat/lon cells, caches per-instance
fast-path geometry (center, inradius, circumradius as chord
thresholds), and maintains a parent-to-children map for hierarchical
locate() descent.

Lifecycle mirrors the hash indexes: built explicitly from scripts via
spatial_index(...), rebuilt on open, maintained incrementally on
insert / attribute write / removal through the StructTable hooks.

Cell resolution: cell edge = 360 / 2^lod degrees, so a lod always
divides the latitude and longitude ranges evenly. All distances in
the fast paths are 3D chords (planet-independent, monotonic in
surface distance), with safety margins on the accept / reject
thresholds; the ambiguous band falls back to exact geometry tests.
"""

# Import Modules
from __future__ import annotations

from typing import Any, Optional
import math

from src.core.types.home_types.coordinate import Coordinate, _from_unit_vector, _to_unit_vector
from src.core.types.home_types.geometry import Polygon, Polyline, _unwrapped_lons
from src.core.types.home_types.planet import Planet, get_current_planet
from src.core.types.types import TypeCoordinate, TypePolygon, TypePolyline

# Safety margins for the chord fast paths (absorbs spheroid radius
# wobble and normalization effects)
_ACCEPT_MARGIN: float = 0.99
_REJECT_MARGIN: float = 1.01

# Fast-path soundness gate: a polygon whose vertices all lie within a
# hemisphere of its anchor has its whole lat/lon interior within that
# hemisphere, so "chord distance > circumradius implies outside" is
# sound. Beyond that, the outside fast path is disabled.
_HEMISPHERE_GATE_RAD: float = 0.95 * math.pi / 2.0

# Geometry kinds supported by the index
_GEOMETRY_TYPES = (TypeCoordinate, TypePolyline, TypePolygon)



def _mean_center(vertices: list[Coordinate], planet: Planet) -> Coordinate:
    """
    Computes the vertex-direction mean of a geometry, at surface
    radius (the fast-path anchor).

    Args:
        vertices (list[Coordinate]): The geometry vertices.
        planet (Planet): Effective planet.

    Returns:
        Coordinate: The center coordinate.

    Raises:
        ValueError: If the vertices cancel out.
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    for vertex in vertices:
        vx: float
        vy: float
        vz: float
        vx, vy, vz = _to_unit_vector(lat_deg=vertex.lat, lon_deg=vertex.lon)

        x += vx
        y += vy
        z += vz

    norm: float = math.sqrt(x * x + y * y + z * z)

    if norm < 1e-15:
        raise ValueError("Geometry vertices cancel out; center is undefined")

    lat: float
    lon: float
    lat, lon = _from_unit_vector(vector=(x / norm, y / norm, z / norm))

    return Coordinate(
        lat=lat, lon=lon, radius=planet.surface_radius(lat_deg=lat)
    )


def _lat_lon_center(
    vertices: list[Coordinate],
    lons: list[float],
    planet: Planet,
) -> Coordinate:
    """
    Computes a latitude/longitude-space center for rings whose
    unwrapped longitude span exceeds 180 degrees (antimeridian
    straddlers), where the 3D direction mean lands outside the
    lat/lon interior.

    Args:
        vertices (list[Coordinate]): The polygon vertices.
        lons (list[float]): The unwrapped longitudes.
        planet (Planet): Effective planet.

    Returns:
        Coordinate: The center coordinate.
    """

    lat: float = sum(vertex.lat for vertex in vertices) / len(vertices)
    lon: float = (sum(lons) / len(lons) + 180.0) % 360.0 - 180.0

    return Coordinate(
        lat=lat, lon=lon, radius=planet.surface_radius(lat_deg=lat)
    )


def _geodesic_to_chord(geodesic_m: float, radius: float) -> float:
    """
    Converts a surface distance on a sphere of the given radius to
    the corresponding chord length.

    Args:
        geodesic_m (float): Surface distance in meters.
        radius (float): Sphere radius in meters.

    Returns:
        float: Chord length in meters.
    """

    angle: float = geodesic_m / radius

    return 2.0 * radius * math.sin(min(math.pi, angle) / 2.0)



class GeometryEntry:
    """
    Cached fast-path geometry for one indexed instance.

    Attributes:
        center (Coordinate): Fast-path anchor point.
        inradius_chord (float): Chord threshold under which the point
            is definitely inside (0 when unknown).
        circumradius_chord (float): Chord threshold over which the
            point is definitely outside.
        cells (frozenset[tuple[int, int]]): Cells covered by the
            geometry bounding box.
    """

    def __init__(
        self,
        center: Coordinate,
        inradius_chord: float,
        circumradius_chord: float,
        cells: "frozenset[tuple[int, int]]",
    ) -> None:
        """
        Initializes the GeometryEntry instance.

        Args:
            center (Coordinate): Fast-path anchor point.
            inradius_chord (float): Definite-inside chord threshold.
            circumradius_chord (float): Definite-outside chord threshold.
            cells (frozenset): Covered cell keys.
        """

        self.center: Coordinate = center
        self.inradius_chord: float = inradius_chord
        self.circumradius_chord: float = circumradius_chord
        self.cells: "frozenset[tuple[int, int]]" = cells



class SpatialIndex:
    """
    Cell index over one geometry attribute of one struct table.

    Attributes:
        table (Any): The owning StructTable.
        attribute (str): The indexed geometry attribute name.
        parent_attribute (Optional[str]): Self-reference attribute
            enabling hierarchical locate() descent.
        lod (int): Level of detail; cell edge = 360 / 2^lod degrees.
        cell_deg (float): Cell edge in degrees.
        n_x (int): Longitude cell count.
        n_y (int): Latitude cell count.
        cells (dict[tuple[int, int], set[int]]): Cell key to local ids.
        entries (dict[int, GeometryEntry]): Local id to cached geometry.
        children (dict[int, list[int]]): Parent local id to child ids.
        roots (list[int]): Local ids without a parent.
    """

    def __init__(
        self,
        table: Any,
        attribute: str,
        lod: int = 8,
        parent_attribute: Optional[str] = None,
    ) -> None:
        """
        Builds the index over every current instance of the table.

        Args:
            table (Any): The owning StructTable.
            attribute (str): Geometry attribute name.
            lod (int, optional): Level of detail. Defaults to 8.
            parent_attribute (Optional[str], optional): Parent
                attribute for hierarchical descent. Defaults to None.

        Raises:
            AttributeError: If the attribute is not declared.
            TypeError: If the attribute is not a geometry type.
        """

        attr_def: Any = table.struct.get_attribute(name=attribute)

        if attr_def is None:
            raise AttributeError(
                f"Cannot spatially index undeclared attribute '{attribute}' "
                f"on Struct '{table.struct.name}'"
            )

        if not isinstance(attr_def.attr_type, _GEOMETRY_TYPES):
            raise TypeError(
                f"Only geometry attributes (Coordinate / Polyline / "
                f"Polygon) are spatially indexable, but "
                f"'{table.struct.name}.{attribute}' is "
                f"'{repr(attr_def.attr_type)}'"
            )

        if not isinstance(lod, int) or isinstance(lod, bool) or not 1 <= lod <= 20:
            raise ValueError(f"lod must be an integer in [1, 20], got {lod}")

        self.table: Any = table
        self.attribute: str = attribute
        self.parent_attribute: Optional[str] = parent_attribute
        self.lod: int = lod
        self.cell_deg: float = 360.0 / (2**lod)
        self.n_x: int = 2**lod
        self.n_y: int = 2 ** (lod - 1)

        self.cells: dict[tuple[int, int], set[int]] = {}
        self.entries: dict[int, GeometryEntry] = {}
        self.children: dict[int, list[int]] = {}
        self.roots: list[int] = []

        # Lazy rows decode once for the build (hash indexes do the same)
        if hasattr(table.instances, "materialize_all"):
            table.instances.materialize_all()

        for local_id, instance in dict.items(table.instances):
            self._add_instance(instance=instance, local_id=local_id)

    # --- Maintenance ---

    def _geometry_of(self, instance: Any) -> Optional[Any]:
        """
        Reads the indexed geometry raw value of an instance.

        Args:
            instance (Any): The instance.

        Returns:
            Optional[Any]: The raw geometry (Coordinate, Polyline, or
            Polygon), or None when absent.
        """

        if self.attribute in instance.attributes:
            return instance.attributes[self.attribute].get()

        attr_def: Any = self.table.struct.get_attribute(name=self.attribute)

        if attr_def is not None and attr_def.has_default:
            return instance.get_attribute(name=self.attribute).get()

        return None

    def _add_instance(self, instance: Any, local_id: int) -> None:
        """
        Adds one instance to the cells, cache, and hierarchy maps.

        Args:
            instance (Any): The instance.
            local_id (int): Its local id.
        """

        # Drop any previous placement (attribute rewrite)
        self._remove_instance(local_id=local_id)

        geometry: Optional[Any] = self._geometry_of(instance=instance)

        if geometry is None:
            return

        planet: Planet = get_current_planet()
        entry: Optional[GeometryEntry] = self._build_entry(
            geometry=geometry, planet=planet
        )

        if entry is None:
            return

        self.entries[local_id] = entry

        for cell in entry.cells:
            self.cells.setdefault(cell, set()).add(local_id)

        # Hierarchy placement
        if self.parent_attribute is not None:
            parent: Optional[Any] = None

            if self.parent_attribute in instance.attributes:
                parent = instance.attributes[self.parent_attribute].get()

            if parent is not None and hasattr(parent, "entity_id"):
                parent_local: int = parent.entity_id.local_id

                self.children.setdefault(parent_local, []).append(local_id)
            else:
                self.roots.append(local_id)
        else:
            self.roots.append(local_id)

    def _circumradius_chord(
        self,
        geometry: Any,
        center: Coordinate,
        planet: Planet,
    ) -> float:
        """
        Computes the cached circumradius chord with the hemisphere
        soundness gate applied.

        Args:
            geometry (Any): The Polygon or Polyline.
            center (Coordinate): The anchor point.
            planet (Planet): Effective planet.

        Returns:
            float: The chord circumradius in meters, or math.inf when
                the geometry is too large for the outside fast path.
        """

        max_geodesic: float = max(
            center.distance(other=vertex, planet=planet)
            for vertex in geometry.vertices
        )

        if (
            max_geodesic
            >= _HEMISPHERE_GATE_RAD * planet.quadratic_mean_radius
        ):
            return math.inf

        return _geodesic_to_chord(
            geodesic_m=max_geodesic,
            radius=planet.quadratic_mean_radius,
        )

    def _build_entry(self, geometry: Any, planet: Planet) -> Optional[GeometryEntry]:
        """
        Computes the cached geometry for one raw value.

        Args:
            geometry (Any): Raw Coordinate, Polyline, or Polygon.
            planet (Planet): Effective planet.

        Returns:
            Optional[GeometryEntry]: The entry, or None for empty
            geometries.
        """

        if isinstance(geometry, Coordinate):
            return GeometryEntry(
                center=geometry,
                inradius_chord=0.0,
                circumradius_chord=0.0,
                cells=frozenset(
                    self._cells_for_bbox(
                        lat_min=geometry.lat,
                        lat_max=geometry.lat,
                        lon_min=geometry.lon,
                        lon_max=geometry.lon,
                    )
                ),
            )

        if isinstance(geometry, (Polygon, Polyline)):
            vertices: list[Coordinate] = geometry.vertices

            if not vertices:
                return None

            lons: list[float] = _unwrapped_lons(vertices=vertices)

            # Rings straddling the antimeridian the "long way" have a
            # 3D direction mean outside their lat/lon interior; use a
            # lat/lon-space anchor instead
            if isinstance(geometry, Polygon) and (
                max(lons) - min(lons) > 180.0
            ):
                center: Coordinate = _lat_lon_center(
                    vertices=vertices, lons=lons, planet=planet
                )
            else:
                center = _mean_center(vertices=vertices, planet=planet)

            if isinstance(geometry, Polygon) and len(vertices) >= 3:
                inradius_chord: float = _geodesic_to_chord(
                    geodesic_m=geometry.inradius(planet=planet, center=center),
                    radius=planet.quadratic_mean_radius,
                )

                circumradius_chord: float = self._circumradius_chord(
                    geometry=geometry, center=center, planet=planet
                )
            else:
                inradius_chord = 0.0

                circumradius_chord = self._circumradius_chord(
                    geometry=geometry, center=center, planet=planet
                )

            lats: list[float] = [vertex.lat for vertex in vertices]

            return GeometryEntry(
                center=center,
                inradius_chord=inradius_chord,
                circumradius_chord=circumradius_chord,
                cells=frozenset(
                    self._cells_for_bbox(
                        lat_min=min(lats),
                        lat_max=max(lats),
                        lon_min=min(lons),
                        lon_max=max(lons),
                    )
                ),
            )

        return None

    def _remove_instance(self, local_id: int) -> None:
        """
        Removes one local id from the cells, cache, and hierarchy maps.

        Args:
            local_id (int): The local id to drop.
        """

        entry: Optional[GeometryEntry] = self.entries.pop(local_id, None)

        if entry is not None:
            for cell in entry.cells:
                bucket: Optional[set[int]] = self.cells.get(cell)

                if bucket is not None:
                    bucket.discard(local_id)

                    if not bucket:
                        self.cells.pop(cell, None)

        if self.parent_attribute is not None:
            if local_id in self.roots:
                self.roots.remove(local_id)

            child_lists: list[int] = self.children.get(local_id, [])

            if local_id in self.children:
                self.children.pop(local_id, None)

            # Children of a removed parent become roots
            for child_id in child_lists:
                if child_id in self.entries and child_id not in self.roots:
                    self.roots.append(child_id)
        else:
            if local_id in self.roots:
                self.roots.remove(local_id)

    def remove_instance(self, local_id: int) -> None:
        """
        Public removal hook (StructTable.index_remove_entity).

        Args:
            local_id (int): The local id to drop.
        """

        self._remove_instance(local_id=local_id)

    def update_instance(self, instance: Any, local_id: int) -> None:
        """
        Re-indexes one instance after a geometry attribute write.

        Args:
            instance (Any): The instance.
            local_id (int): Its local id.
        """

        self._add_instance(instance=instance, local_id=local_id)

    # --- Cell helpers ---

    def _cells_for_bbox(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
    ) -> list[tuple[int, int]]:
        """
        Enumerates the cells covered by a bounding box (unwrapped
        longitudes wrap through the modulo).

        Args:
            lat_min (float): Minimum latitude.
            lat_max (float): Maximum latitude.
            lon_min (float): Minimum unwrapped longitude.
            lon_max (float): Maximum unwrapped longitude.

        Returns:
            list[tuple[int, int]]: Covered cell keys.
        """

        iy0: int = self._lat_cell(lat=lat_min)
        iy1: int = self._lat_cell(lat=lat_max)
        ix0: int = self._lon_cell(lon=lon_min)
        ix1: int = self._lon_cell(lon=lon_max)

        out: list[tuple[int, int]] = []

        for iy in range(iy0, iy1 + 1):
            clamped_iy: int = min(iy, self.n_y - 1)

            for ix in range(ix0, ix1 + 1):
                out.append((ix % self.n_x, clamped_iy))

        return out

    def _lat_cell(self, lat: float) -> int:
        """
        Quantizes a latitude into its cell row index.

        Args:
            lat (float): Latitude in degrees.

        Returns:
            int: Cell row index.
        """

        return int(math.floor((lat + 90.0) / self.cell_deg))

    def _lon_cell(self, lon: float) -> int:
        """
        Quantizes an (unwrapped) longitude into its cell column index.

        Args:
            lon (float): Longitude in degrees.

        Returns:
            int: Cell column index (unwrapped; wrap through modulo).
        """

        return int(math.floor((lon + 180.0) / self.cell_deg))

    def cell_of(self, coord: Coordinate) -> tuple[int, int]:
        """
        Returns the cell key containing a coordinate.

        Args:
            coord (Coordinate): The coordinate.

        Returns:
            tuple[int, int]: The cell key.
        """

        return (
            self._lon_cell(lon=coord.lon) % self.n_x,
            min(self._lat_cell(lat=coord.lat), self.n_y - 1),
        )

    def _cell_lower_bound(
        self,
        coord: Coordinate,
        cell: tuple[int, int],
        radius_bound: float,
    ) -> float:
        """
        Computes a lower bound on the distance from a coordinate to
        anything inside a cell.

        Latitude separation bounds the distance by its meridian arc.
        Otherwise a spherical sine-rule bound (using the query
        point's latitude) bounds it by the longitude separation. Both
        are safe lower bounds.

        Args:
            coord (Coordinate): The query coordinate.
            cell (tuple[int, int]): The cell key.
            radius_bound (float): Conservative radius for arc
                conversion.

        Returns:
            float: Lower bound in meters.
        """

        lat0: float = cell[1] * self.cell_deg - 90.0
        lat1: float = lat0 + self.cell_deg

        dlat: float = max(0.0, lat0 - coord.lat, coord.lat - lat1)

        if dlat > 0.0:
            return math.radians(dlat) * radius_bound

        # Longitude distance from the point to the cell's interval,
        # considering wrapped duplicates of the interval
        ix: int = cell[0]

        best_dlon: Optional[float] = None

        for shift in (-self.n_x, 0, self.n_x):
            lo: float = (ix + shift) * self.cell_deg - 180.0
            hi: float = lo + self.cell_deg

            if coord.lon >= lo and coord.lon <= hi:
                candidate: float = 0.0
            else:
                candidate = min(
                    abs(coord.lon - lo),
                    abs(coord.lon - hi),
                )

                # Great-circle longitude separation caps at 180 degrees
                candidate = min(candidate, 180.0)

            if best_dlon is None or candidate < best_dlon:
                best_dlon = candidate

        assert best_dlon is not None

        if best_dlon <= 0.0:
            return 0.0

        # The sine rule uses the query point's latitude: for any two
        # points with longitude separation >= dlon, sin(angular
        # distance) >= cos(lat_point) * sin(dlon)
        ratio: float = math.cos(math.radians(coord.lat)) * math.sin(
            math.radians(best_dlon)
        )

        return math.asin(min(1.0, ratio)) * radius_bound

    @staticmethod
    def _bound_radius(planet: Planet) -> float:
        """
        Returns a conservative radius for lower-bound arc conversion.

        On spheres this is the radius itself. On spheroids the maximum
        Gaussian curvature radius (equatorial squared over polar)
        keeps the bound a safe underestimate of surface distances.

        Args:
            planet (Planet): Effective planet.

        Returns:
            float: Conservative radius in meters.
        """

        if planet.is_sphere:
            return planet.equatorial_radius

        return (
            planet.equatorial_radius
            * planet.equatorial_radius
            / planet.polar_radius_value
        )

    def sort_key(self, local_id: int) -> "tuple[int, tuple[int, int], int]":
        """
        Returns a dump-ordering key for one local id (minimum cell
        key; uncached rows sort last).

        Args:
            local_id (int): The local id.

        Returns:
            tuple[int, tuple[int, int], int]: The ordering key.
        """

        entry: Optional[GeometryEntry] = self.entries.get(local_id)

        if entry is None or not entry.cells:
            return (1, (0, 0), local_id)

        return (0, min(entry.cells), local_id)

    # --- Queries ---

    def _is_definitely_inside(self, coord: Coordinate, entry: GeometryEntry) -> bool:
        """
        Fast path: chord-close points to the center are inside.

        Args:
            coord (Coordinate): The query coordinate.
            entry (GeometryEntry): The cached geometry.

        Returns:
            bool: True when containment is certain.
        """

        if entry.inradius_chord <= 0.0:
            return False

        return coord.chord_distance(other=entry.center) < (
            _ACCEPT_MARGIN * entry.inradius_chord
        )

    def _is_definitely_outside(self, coord: Coordinate, entry: GeometryEntry) -> bool:
        """
        Fast path: chord-far points from the center are outside.

        Args:
            coord (Coordinate): The query coordinate.
            entry (GeometryEntry): The cached geometry.

        Returns:
            bool: True when non-containment is certain.
        """

        return coord.chord_distance(other=entry.center) > (
            _REJECT_MARGIN * entry.circumradius_chord
        )

    def _contains(self, coord: Coordinate, instance: Any, entry: GeometryEntry) -> bool:
        """
        Tests containment with fast paths and an exact fallback.

        Args:
            coord (Coordinate): The query coordinate.
            instance (Any): The instance.
            entry (GeometryEntry): The cached geometry.

        Returns:
            bool: True when the geometry contains the coordinate.
        """

        if self._is_definitely_inside(coord=coord, entry=entry):
            return True

        if self._is_definitely_outside(coord=coord, entry=entry):
            return False

        geometry: Any = self._geometry_of(instance=instance)

        if isinstance(geometry, Polygon):
            return geometry.contains(point=coord)

        if isinstance(geometry, Polyline):
            # Paths contain nothing; distance zero counts as a hit for
            # locate-style queries on degenerate paths
            return geometry.distance_to(point=coord) == 0.0

        if isinstance(geometry, Coordinate):
            return geometry == coord

        return False

    def _find_containing(
        self,
        coord: Coordinate,
        local_ids: list[int],
    ) -> Optional[int]:
        """
        Returns the first local id whose geometry contains the
        coordinate.

        Args:
            coord (Coordinate): The query coordinate.
            local_ids (list[int]): Candidate local ids.

        Returns:
            Optional[int]: The containing local id, or None.
        """

        instances: Any = self.table.instances

        for local_id in local_ids:
            entry: Optional[GeometryEntry] = self.entries.get(local_id)

            if entry is None:
                continue

            instance: Optional[Any] = instances.get(local_id) if hasattr(
                instances, "get"
            ) else None

            if instance is None:
                instance = self.table.get(local_id=local_id)

            if instance is None:
                continue

            if self._contains(coord=coord, instance=instance, entry=entry):
                return local_id

        return None

    def locate(self, coord: Coordinate) -> Optional[int]:
        """
        Returns the deepest instance whose geometry contains the
        coordinate (hierarchical descent when a parent attribute is
        declared; flat containment scan otherwise).

        Args:
            coord (Coordinate): The query coordinate.

        Returns:
            Optional[int]: The containing local id, or None.
        """

        if self.parent_attribute is None:
            # Flat: candidates are the entries covering the point's cell
            flat_candidates: list[int] = list(
                self.cells.get(self.cell_of(coord=coord), set())
            )

            flat_found: Optional[int] = self._find_containing(
                coord=coord, local_ids=flat_candidates
            )

            return flat_found

        candidates: list[int] = list(self.roots)
        result: Optional[int] = None

        while candidates:
            found: Optional[int] = self._find_containing(
                coord=coord, local_ids=candidates
            )

            if found is None:
                break

            result = found
            candidates = self.children.get(found, [])

        return result

    def nearest(self, coord: Coordinate, k: int) -> list[int]:
        """
        Returns the local ids of the k indexed instances whose cached
        centers are nearest to the coordinate (chord ranking; ties
        broken arbitrarily).

        Args:
            coord (Coordinate): The query coordinate.
            k (int): Result count.

        Returns:
            list[int]: Up to k local ids, nearest first.
        """

        if k <= 0:
            return []

        radius_bound: float = self._bound_radius(planet=get_current_planet())

        # Cells ordered by their guaranteed minimum distance
        ranked_cells: list[tuple[float, tuple[int, int]]] = []

        for cell in self.cells:
            bound: float = self._cell_lower_bound(
                coord=coord, cell=cell, radius_bound=radius_bound
            )

            ranked_cells.append((bound, cell))

        ranked_cells.sort(key=lambda item: (item[0], item[1]))

        # Top-k by chord to the cached center (deduped: a geometry
        # covering several cells must appear once)
        best: list[tuple[float, int]] = []
        seen: set[int] = set()

        def kth_chord() -> float:
            """
            Returns the current k-th best chord threshold.

            Returns:
                float: The threshold (inf when fewer than k).
            """

            if len(best) < k:
                return math.inf

            return best[k - 1][0]

        for bound, cell in ranked_cells:
            # Safe stop: anything in farther cells is geodesically
            # farther than the current k-th best center distance
            if len(best) >= k:
                kth_best_geodesic: float = max(
                    coord.distance(other=self.entries[local_id].center)
                    for _, local_id in best
                )

                if bound >= kth_best_geodesic:
                    break

            for local_id in self.cells[cell]:

                if local_id in seen:
                    continue

                seen.add(local_id)

                entry: Optional[GeometryEntry] = self.entries.get(local_id)

                if entry is None:
                    continue

                chord: float = coord.chord_distance(other=entry.center)

                if len(best) < k or chord < kth_chord():
                    best.append((chord, local_id))
                    best.sort(key=lambda item: (item[0], item[1]))
                    del best[k:]

        return [local_id for _, local_id in best]

    def within(self, coord: Coordinate, radius_m: float) -> list[int]:
        """
        Returns the local ids whose geometry lies within a radius of
        the coordinate (distance zero for containing geometries).

        Args:
            coord (Coordinate): The query coordinate.
            radius_m (float): Radius in meters.

        Returns:
            list[int]: Matching local ids.
        """

        if radius_m < 0.0:
            raise ValueError(f"Radius must be non-negative, got {radius_m}")

        radius_bound: float = self._bound_radius(planet=get_current_planet())

        result: list[int] = []
        seen: set[int] = set()
        instances: Any = self.table.instances

        for cell, local_ids in self.cells.items():
            bound: float = self._cell_lower_bound(
                coord=coord, cell=cell, radius_bound=radius_bound
            )

            if bound > radius_m:
                continue

            for local_id in local_ids:

                if local_id in seen:
                    continue

                seen.add(local_id)

                entry: Optional[GeometryEntry] = self.entries.get(local_id)

                if entry is None:
                    continue

                # Fast paths on the center distance
                if self._is_definitely_inside(coord=coord, entry=entry):
                    result.append(local_id)

                    continue

                if self._is_definitely_outside(coord=coord, entry=entry):
                    continue

                instance: Optional[Any] = instances.get(local_id) if hasattr(
                    instances, "get"
                ) else None

                if instance is None:
                    instance = self.table.get(local_id=local_id)

                if instance is None:
                    continue

                geometry: Any = self._geometry_of(instance=instance)

                if isinstance(geometry, (Polygon, Polyline)):
                    if geometry.distance_to(point=coord) <= radius_m:
                        result.append(local_id)

                elif isinstance(geometry, Coordinate):
                    if coord.distance(other=geometry) <= radius_m:
                        result.append(local_id)

        return result
