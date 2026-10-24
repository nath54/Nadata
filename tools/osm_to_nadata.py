"""OSM XML to Nadata converter.

Converts an OpenStreetMap XML extract (Overpass API `.osm` or Geofabrik
`.osm` / `.osm.bz2` files) into `.nadata` rows compatible with the
ex13_earth schema:

- every node referenced by a highway way becomes a ``Place`` row
  (junction), named nodes with amenity/shop/tourism/leisure/... tags
  become typed places,
- every highway way becomes a ``Link`` edge: endpoints are its first /
  last node, the geometry is a ``Polyline`` through all of its nodes,
  transport modes derive from the highway class and ``maxspeed`` maps
  to the car speed,
- closed ``boundary=administrative`` ways become ``AdminRegion`` rows,
- building ways become ``Place`` rows positioned at their centroid
  (opt-in via ``--buildings``).

The tool is deliberately stdlib-only and streams with ``iterparse`` so
it handles city-scale extracts on a laptop. Only complete data
converts: ways referencing nodes outside the extract are dropped.

Usage::

    python tools/osm_to_nadata.py map.osm map.nadata --radius 6371008.8
    python tools/osm_to_nadata.py map.osm.bz2 map.nadata --max-ways 1000
"""

# Import Modules
import argparse
import bz2
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional


# Highway class -> transport modes. Roads not listed default to
# car + walk.
_MOTORWAYS = {
    "motorway", "motorway_link", "trunk", "trunk_link",
}

_WALK_ONLY = {
    "footway", "path", "pedestrian", "steps", "cycleway", "track",
    "platform", "elevator", "crossing", "via_ferrata", "bridleway",
}

_WALK_SPEED_KMH = 5.0

# Node tags worth a typed Place row (tag key -> shown kind)
_PLACE_TAG_KEYS = (
    "amenity", "shop", "tourism", "leisure", "healthcare", "craft",
    "office", "emergency", "historic", "man_made",
)


def _open_maybe_bz2(source: Path) -> Any:
    """
    Opens a source file transparently decompressing bzip2.

    Args:
        source (Path): The .osm or .osm.bz2 path.

    Returns:
        Any: A binary file object.
    """

    if source.suffix == ".bz2":
        return bz2.open(source, "rb")

    return open(source, "rb")


def _escape(text: str) -> str:
    """
    Escapes a string for a Nadata Str literal.

    Args:
        text (str): The raw text.

    Returns:
        str: The escaped one-line text.
    """

    return (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", " ")
        .replace("\r", " ")
    )


def _parse_speed(raw: str) -> Optional[float]:
    """
    Parses an OSM maxspeed value into km/h.

    Args:
        raw (str): The tag value ("50", "60 knots", "walk", ...).

    Returns:
        Optional[float]: The km/h value, or None when unparsable.
    """

    digits = ""

    for char in raw.strip():
        if char.isdigit() or char == ".":
            digits += char

        elif digits:
            break

    if not digits:
        return None

    try:
        return float(digits)

    except ValueError:
        return None


def _modes_for(highway: str) -> list[str]:
    """
    Maps a highway class to transport modes.

    Args:
        highway (str): The highway tag value.

    Returns:
        list[str]: The mode list ("car", "walk").
    """

    if highway in _MOTORWAYS:
        return ["car"]

    if highway in _WALK_ONLY:
        return ["walk"]

    return ["car", "walk"]


def _tags_of(element: ET.Element) -> dict[str, str]:
    """
    Collects an element's tags into a dict.

    Args:
        element (ET.Element): The OSM element.

    Returns:
        dict[str, str]: The tag key/value pairs.
    """

    return {
        tag.get("k", ""): tag.get("v", "")
        for tag in element.findall("tag")
    }


class _Collector:
    """
    Two-pass streaming collector: pass one gathers the node ids
    referenced by ways, pass two keeps only their coordinates and the
    tagged place nodes.
    """

    def __init__(self, source: Path) -> None:
        """
        Runs both passes over the source.

        Args:
            source (Path): The .osm / .osm.bz2 path.
        """

        self.node_refs: set[str] = set()
        self.ways: list[tuple[str, dict[str, str], list[str]]] = []
        self.coords: dict[str, tuple[float, float]] = {}
        self.places: list[tuple[str, str, str, tuple[float, float]]] = []
        self.boundaries: list[tuple[str, str, list[tuple[float, float]]]] = []
        self.dropped_ways: int = 0

        self._pass_one(source=source)
        self._pass_two(source=source)

    def _pass_one(self, source: Path) -> None:
        """
        Collects way definitions and the node ids they reference.

        Args:
            source (Path): The source path.
        """

        with _open_maybe_bz2(source=source) as handle:
            for _event, element in ET.iterparse(handle, events=("end",)):
                if element.tag == "way":
                    tags: dict[str, str] = _tags_of(element=element)

                    refs: list[str] = [
                        nd.get("ref", "")
                        for nd in element.findall("nd")
                    ]

                    self.ways.append((element.get("id", ""), tags, refs))
                    self.node_refs.update(refs)

                elif element.tag in ("node", "way", "relation"):
                    # Free the parsed subtree while streaming
                    element.clear()

    def _pass_two(self, source: Path) -> None:
        """
        Keeps coordinates of referenced nodes and extracts places.

        Args:
            source (Path): The source path.
        """

        with _open_maybe_bz2(source=source) as handle:
            for _event, element in ET.iterparse(handle, events=("end",)):
                if element.tag != "node":
                    continue

                node_id: str = element.get("id", "")

                if node_id in self.node_refs:
                    self.coords[node_id] = (
                        float(element.get("lat", "0")),
                        float(element.get("lon", "0")),
                    )

                tags: dict[str, str] = _tags_of(element=element)

                name: str = tags.get("name", "")

                if not name:
                    element.clear()

                    continue

                for key in _PLACE_TAG_KEYS:
                    kind: str = tags.get(key, "")

                    if kind:
                        self.places.append(
                            (
                                node_id,
                                _escape(text=name),
                                _escape(text=kind),
                                (
                                    float(element.get("lat", "0")),
                                    float(element.get("lon", "0")),
                                ),
                            )
                        )

                        break

                element.clear()

        # Building ways become centroid places
        for way_id, tags, refs in self.ways:
            if "building" not in tags or len(refs) < 3:
                continue

            name: str = tags.get("name", "")

            if not name:
                continue

            lats: list[float] = []
            lons: list[float] = []

            for ref in refs:
                coord: Optional[tuple[float, float]] = self.coords.get(ref)

                if coord is not None:
                    lats.append(coord[0])
                    lons.append(coord[1])

            if lats:
                self.places.append(
                    (
                        way_id,
                        _escape(text=name),
                        "building",
                        (sum(lats) / len(lats), sum(lons) / len(lons)),
                    )
                )

    def admin_regions(self) -> list[tuple[str, str, list[tuple[float, float]]]]:
        """
        Extracts closed administrative boundary ways.

        Returns:
            list[tuple[str, str, list[tuple[float, float]]]]: The
            (osm id, name, ring) triples.
        """

        regions: list[tuple[str, str, list[tuple[float, float]]]] = []

        for way_id, tags, refs in self.ways:
            if tags.get("boundary") != "administrative":
                continue

            name: str = tags.get("name", "")

            if not name or len(refs) < 4 or refs[0] != refs[-1]:
                continue

            ring: list[tuple[float, float]] = [
                self.coords[ref]
                for ref in refs
                if ref in self.coords
            ]

            if len(ring) >= 4:
                regions.append((way_id, _escape(text=name), ring))

        return regions


def convert(
    source: Path,
    destination: Path,
    radius: float,
    max_ways: Optional[int] = None,
    skip_roads: bool = False,
    skip_places: bool = False,
    skip_buildings: bool = False,
    skip_boundaries: bool = False,
) -> dict[str, int]:
    """
    Converts an OSM XML extract into a .nadata file.

    Args:
        source (Path): The .osm / .osm.bz2 input path.
        destination (Path): The .nadata output path.
        radius (float): The Coordinate radius in meters.
        max_ways (Optional[int], optional): Way budget (smoke tests).
            Defaults to None.
        skip_roads (bool, optional): Do not emit Link rows. Defaults
            to False.
        skip_places (bool, optional): Do not emit tagged node places.
            Defaults to False.
        skip_buildings (bool, optional): Do not emit building places.
            Defaults to False.
        skip_boundaries (bool, optional): Do not emit AdminRegion
            rows. Defaults to False.

    Returns:
        dict[str, int]: Counters (places, junctions, roads, regions,
        dropped ways).
    """

    collector: _Collector = _Collector(source=source)

    lines: list[str] = [
        "# Generated by tools/osm_to_nadata.py - do not edit by hand.",
        f"# Source: {source.name} (OpenStreetMap, ODbL).",
        "",
    ]

    dropped: int = 0
    roads: list[tuple[str, str, str, list[str], str, str]] = []
    junctions: dict[str, tuple[float, float]] = {}

    if not skip_roads:
        for way_id, tags, refs in collector.ways:
            highway: str = tags.get("highway", "")

            if not highway:
                continue

            if max_ways is not None and len(roads) >= max_ways:
                break

            # Keep only refs present in the extract
            valid_refs: list[str] = [
                ref for ref in refs if ref in collector.coords
            ]

            if len(valid_refs) < 2:
                dropped += 1

                continue

            # Every way node is a shared junction place: ways cross
            # THROUGH intermediate nodes, so the graph needs one
            # segment edge per consecutive node pair to stay
            # connected at real-world junctions
            for ref in valid_refs:
                junctions[ref] = collector.coords[ref]

            name: str = tags.get("name", "") or f"{highway} {way_id}"

            modes: list[str] = _modes_for(highway=highway)

            speed_entries: list[str] = []

            if "maxspeed" in tags and "car" in modes:
                parsed_speed: Optional[float] = _parse_speed(
                    raw=tags["maxspeed"]
                )

                if parsed_speed is not None:
                    if "car" in modes:
                        speed_entries.append(f'"car": {parsed_speed}')

                    if "walk" in modes:
                        speed_entries.append(
                            f'"walk": {_WALK_SPEED_KMH}'
                        )

            # Empty collections are omitted: the schema defaults
            # apply and the loader cannot infer {} / [] types
            extras: list[str] = []

            if modes:
                extras.append(
                    "modes = ["
                    + ", ".join('"' + m + '"' for m in modes)
                    + "]"
                )

            if speed_entries:
                speed_literal: str = "{ " + ", ".join(speed_entries) + " }"

                extras.append(f"speed = {speed_literal}")

            extras_text: str = (", " + ", ".join(extras)) if extras else ""

            name_text: str = _escape(text=name)

            for index in range(len(valid_refs) - 1):
                from_ref: str = valid_refs[index]
                to_ref: str = valid_refs[index + 1]

                lat0, lon0 = collector.coords[from_ref]
                lat1, lon1 = collector.coords[to_ref]

                segment_shape: str = (
                    f"Coordinate({lat0:.7f}, {lon0:.7f}, {radius}), "
                    f"Coordinate({lat1:.7f}, {lon1:.7f}, {radius})"
                )

                roads.append(
                    (
                        f"{way_id}_{index}",
                        name_text,
                        from_ref,
                        to_ref,
                        extras_text,
                        segment_shape,
                    )
                )

    # Tagged places (amenities, shops, ...) keep their nid even when
    # the node is also a road junction: the specific row wins
    place_nids: set[str] = set()
    place_count: int = 0

    if not skip_places:
        for osm_id, name, kind, (lat, lon) in collector.places:
            if skip_buildings and kind == "building":
                continue

            place_nids.add(osm_id)

    junction_count: int = 0

    for ref in sorted(junctions, key=int):
        if ref in place_nids:
            continue

        lat, lon = junctions[ref]

        lines.append(
            f'n{ref} = Place(name = "node_{ref}", nid = "n{ref}", '
            f'kind = "junction", '
            f"position = Coordinate({lat:.7f}, {lon:.7f}, {radius}));"
        )

        junction_count += 1

    if not skip_places:
        for osm_id, name, kind, (lat, lon) in collector.places:
            if skip_buildings and kind == "building":
                continue

            lines.append(
                f'n{osm_id} = Place(name = "{name}", nid = "n{osm_id}", '
                f'kind = "{kind}", '
                f"position = Coordinate({lat:.7f}, {lon:.7f}, {radius}));"
            )

            place_count += 1

    region_count: int = 0

    if not skip_boundaries:
        for way_id, name, ring in collector.admin_regions():
            shape: str = ", ".join(
                f"Coordinate({lat:.7f}, {lon:.7f}, {radius})"
                for lat, lon in ring
            )

            lines.append(
                f'a{way_id} = AdminRegion(name = "{name}", '
                f'nid = "a{way_id}", '
                f"boundary = Polygon([{shape}]));"
            )

            region_count += 1

    # Links last: their endpoint symbols are declared by now
    for way_id, name, first_ref, last_ref, extras, shape in roads:
        extra_text: str = extras

        lines.append(
            f'w{way_id} = Link(name = "{name}", nid = "w{way_id}", '
            f"from_place = n{first_ref}, to_place = n{last_ref}"
            f"{extra_text}, shape = Polyline([{shape}]));"
        )

    destination.write_text(chr(10).join(lines) + chr(10), encoding="utf-8")

    return {
        "roads": len(roads),
        "junctions": junction_count,
        "places": place_count,
        "regions": region_count,
        "dropped_ways": dropped,
    }


def main() -> int:
    """
    CLI entry point.

    Returns:
        int: The process exit code.
    """

    parser = argparse.ArgumentParser(
        description="Convert an OpenStreetMap XML extract to .nadata rows "
        "(ex13_earth schema: Place / Link / AdminRegion).",
    )

    parser.add_argument("source", help="Input .osm or .osm.bz2 file.")
    parser.add_argument("destination", help="Output .nadata file.")
    parser.add_argument(
        "--radius",
        type=float,
        default=6371008.8,
        help="Coordinate radius in meters (default: Earth mean radius).",
    )
    parser.add_argument(
        "--max-ways",
        type=int,
        default=None,
        help="Convert at most this many road ways (smoke tests).",
    )
    parser.add_argument(
        "--skip-roads",
        action="store_true",
        help="Do not emit Link rows.",
    )
    parser.add_argument(
        "--skip-places",
        action="store_true",
        help="Do not emit tagged node places (amenities, shops, ...).",
    )
    parser.add_argument(
        "--skip-buildings",
        action="store_true",
        help="Do not emit building places.",
    )
    parser.add_argument(
        "--skip-boundaries",
        action="store_true",
        help="Do not emit AdminRegion rows.",
    )

    args: Any = parser.parse_args()

    stats: dict[str, int] = convert(
        source=Path(args.source),
        destination=Path(args.destination),
        radius=args.radius,
        max_ways=args.max_ways,
        skip_roads=args.skip_roads,
        skip_places=args.skip_places,
        skip_buildings=args.skip_buildings,
        skip_boundaries=args.skip_boundaries,
    )

    print(
        f"wrote {stats['roads']} road(s), {stats['junctions']} junction(s), "
        f"{stats['places']} place(s), {stats['regions']} region(s) "
        f"({stats['dropped_ways']} dropped incomplete way(s)) to "
        f"{args.destination}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
