#!/usr/bin/env python3
"""
GeoJSON to Nadata AdminRegion converter.

Converts a GeoJSON FeatureCollection of polygons into a .nadata file of
AdminRegion rows (one per feature), for ingestion into an Earth example
style project. Stdlib only.

The generated file declares rows for the requested admin level:

    python tools/geojson_to_nadata.py countries.geojson countries.nadata --level Country

Each feature becomes

    row_0 = Country(name = "<properties.name>", nid = "<properties.iso_a3 or row_0>",
                    boundary = Polygon([Coordinate(lat, lon, 6371008.8), ...]));

Polygons are emitted as their outer ring (holes are ignored; the engine's
Polygon supports antimeridian-crossing rings, so no splitting is needed
for data - converters may split only for rendering). MultiPolygons use
the largest part by vertex count. Child levels reference their parent
row via --parent-nid plus a get_by_nid binding appended to the file:

    python tools/geojson_to_nadata.py regions.geojson regions.nadata --level Region --parent-nid FR

The rows then merge into an existing world:

    python -m src.cli console examples/ex13_earth   # then: import_data();

or simply keep the generated file in the project's data/ folder: its
rows are declared like any other instance declaration.
"""

# Import Modules
import argparse
import json
import math
import sys
from typing import Any

DEFAULT_RADIUS: float = 6371008.8



def outer_ring(geometry: dict[str, Any]) -> list[list[float]]:
    """
    Extracts the outer ring of a GeoJSON geometry.

    Polygons use their first (outer) ring; MultiPolygons use the part
    with the most vertices.

    Args:
        geometry (dict[str, Any]): The GeoJSON geometry object.

    Returns:
        list[list[float]]: Positions as [lon, lat] pairs.

    Raises:
        ValueError: If the geometry type is unsupported.
    """

    kind: str = geometry.get("type", "")

    if kind == "Polygon":
        return list(geometry["coordinates"][0])

    if kind == "MultiPolygon":
        best: list[list[float]] = []

        for part in geometry["coordinates"]:
            ring: list[list[float]] = list(part[0])

            if len(ring) > len(best):
                best = ring

        return best

    raise ValueError(f"Unsupported GeoJSON geometry type '{kind}'")



def feature_name(properties: dict[str, Any]) -> str:
    """
    Picks a display name from a feature's properties.

    Args:
        properties (dict[str, Any]): The feature properties.

    Returns:
        str: A name property or a fallback.
    """

    for key in ("name", "NAME", "name_en", "ADMIN", "admin"):
        value: Any = properties.get(key)

        if isinstance(value, str) and value:
            return value

    return "Unnamed"



def feature_nid(properties: dict[str, Any], index: int) -> str:
    """
    Picks a stable nid from a feature's properties.

    Args:
        properties (dict[str, Any]): The feature properties.
        index (int): Fallback row index.

    Returns:
        str: An ISO-style code or the fallback row_<index>.
    """

    for key in ("nid", "iso_a3", "ISO_A3", "iso_a2", "adm0_a3", "postal"):
        value: Any = properties.get(key)

        if isinstance(value, str) and value and value != "-99":
            return value

    return f"row_{index}"



def escape(text: str) -> str:
    """
    Escapes a string for a Nadata Str literal.

    Args:
        text (str): The raw text.

    Returns:
        str: The escaped text.
    """

    return text.replace("\\", "\\\\").replace('"', '\\"')



def simplify_ring(
    ring: list[list[float]],
    max_points: int,
) -> list[list[float]]:
    """
    Uniformly subsamples a ring to at most max_points vertices.

    Uniform subsampling keeps the shape recognizable without a full
    Douglas-Peucker implementation; ingestion of detailed Natural Earth
    data is a preprocessing step, not an engine concern.

    Args:
        ring (list[list[float]]): The [lon, lat] ring.
        max_points (int): The vertex budget.

    Returns:
        list[list[float]]: The simplified ring.
    """

    if max_points <= 0 or len(ring) <= max_points:
        return ring

    step: float = len(ring) / float(max_points)

    return [ring[int(i * step)] for i in range(max_points)]



def convert(
    source: str,
    destination: str,
    level: str,
    radius: float,
    max_points: int,
    parent_nid: str,
) -> int:
    """
    Converts a GeoJSON file into a .nadata AdminRegion file.

    Args:
        source (str): Input GeoJSON path.
        destination (str): Output .nadata path.
        level (str): The admin level class (Country, Region, ...).
        radius (float): The radius for emitted coordinates.
        max_points (int): Vertex budget per ring.
        parent_nid (str): Optional parent row nid.

    Returns:
        int: The number of rows written.
    """

    with open(source, "r", encoding="utf-8") as handle:
        document: Any = json.load(handle)

    if document.get("type") != "FeatureCollection":
        raise ValueError("Expected a GeoJSON FeatureCollection")

    lines: list[str] = [
        "# Generated by tools/geojson_to_nadata.py - do not edit by hand.",
        f"# Source: {source} -> admin level {level}",
        "",
    ]

    count: int = 0

    for index, feature in enumerate(document.get("features", [])):
        geometry: Any = feature.get("geometry")

        if geometry is None:
            continue

        properties: dict[str, Any] = feature.get("properties") or {}

        try:
            ring: list[list[float]] = simplify_ring(
                ring=outer_ring(geometry=geometry), max_points=max_points
            )
        except ValueError:
            # Skip point / line features silently
            continue

        name: str = feature_name(properties=properties)
        nid: str = feature_nid(properties=properties, index=index)

        vertices: list[str] = [
            f"Coordinate({lat:.6f}, {lon:.6f}, {radius})"
            for lon, lat in ring
        ]

        parent_expr: str = (
            f", parent = get_by_nid(AdminRegion, \"{parent_nid}\")"
            if parent_nid
            else ""
        )

        lines.append(
            f"row_{count} = {level}("
            f"name = \"{escape(name)}\", nid = \"{escape(nid)}\""
            f"{parent_expr}, boundary = Polygon(["
            + ", ".join(vertices)
            + "]))"
        )

        count += 1

    with open(destination, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")

    return count



def main() -> int:
    """
    Runs the converter CLI.

    Returns:
        int: Exit code.
    """

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="geojson_to_nadata",
        description="Convert a GeoJSON FeatureCollection to AdminRegion rows.",
    )

    parser.add_argument("source", help="Input .geojson file.")
    parser.add_argument("destination", help="Output .nadata file.")
    parser.add_argument(
        "--level",
        default="Country",
        help="Admin level class for the rows (default: Country).",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=DEFAULT_RADIUS,
        help="Coordinate radius in meters (default: Earth mean radius).",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=64,
        help="Vertex budget per ring (default: 64).",
    )
    parser.add_argument(
        "--parent-nid",
        default="",
        help="Bind all rows as children of this parent row nid.",
    )

    args: argparse.Namespace = parser.parse_args()

    try:
        written: int = convert(
            source=args.source,
            destination=args.destination,
            level=args.level,
            radius=args.radius,
            max_points=args.max_points,
            parent_nid=args.parent_nid,
        )
    except (OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)

        return 1

    print(f"wrote {written} row(s) to {args.destination}")

    return 0



if __name__ == "__main__":
    raise SystemExit(main())
