"""
Pure geodetic grid math: bilinear sampling over sparse cell
dictionaries. Importable from both the value layer and the sea-level
adapter without circular dependencies.
"""

# Import Modules
import math
from typing import Any, Optional



def bilinear_sample(
    cells: dict[tuple[int, int], Any],
    n_x: int,
    n_y: int,
    cell_deg: float,
    lat_deg: float,
    lon_deg: float,
) -> Optional[Any]:
    """
    Bilinearly samples a sparse cell dictionary at a coordinate.

    The four surrounding cell centers are weighted by their fractional
    distance; cells that are absent contribute nothing and the
    remaining weights are renormalized. Longitude wraps across the
    antimeridian; latitude clamps at the poles (the two edge rows
    collapse into one and the row weights merge).

    Args:
        cells (dict[tuple[int, int], Any]): Occupied cells keyed by
            (column, row).
        n_x (int): Number of longitude columns.
        n_y (int): Number of latitude rows.
        cell_deg (float): Cell edge in degrees.
        lat_deg (float): Sample latitude in degrees.
        lon_deg (float): Sample longitude in degrees.

    Returns:
        Optional[Any]: The interpolated value, or None when no
        surrounding cell has data.

    Raises:
        TypeError: If sampled values are not real numbers.
    """

    fx: float = (lon_deg + 180.0) / cell_deg - 0.5
    fy: float = (lat_deg + 90.0) / cell_deg - 0.5

    # Latitude clamps at the poles; longitude wraps via modulo below
    fy = min(max(fy, 0.0), float(n_y - 1))

    x0: int = int(math.floor(fx))
    y0: int = int(math.floor(fy))

    wx1: float = fx - x0
    wy1: float = fy - y0

    x0 = x0 % n_x
    x1: int = (x0 + 1) % n_x

    y1: int = min(y0 + 1, n_y - 1)
    wy0: float = 1.0 - wy1

    weighted: float = 0.0
    total_weight: float = 0.0

    for x, w_x in ((x0, 1.0 - wx1), (x1, wx1)):
        for y, w_y in ((y0, wy0), (y1, wy1)):
            value: Optional[Any] = cells.get((x, y))

            if value is None:
                continue

            weight: float = w_x * w_y

            if weight <= 0.0:
                continue

            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(
                    f"Bilinear sampling requires numeric cells, got "
                    f"{type(value).__name__}"
                )

            weighted += weight * value
            total_weight += weight

    if total_weight <= 0.0:
        return None

    return weighted / total_weight
