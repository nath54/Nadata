"""
Session sea-level surface registry.

A registered sea-level layer overrides the plain surface radius used
by altitude() (and any consumer of sea level): the layer is a sparse
GeoGrid of sea-surface radii; where it has no data the geometric
surface radius applies.

The registry stores duck-typed adapters only, so it can be imported
from the geodesy core without circular dependencies. Like the session
planet, the registry is process-global state.
"""

# Import Modules
from typing import Any, Optional

from src.core.types.home_types.geogrid import bilinear_sample



class GeoSeaLevelAdapter:
    """
    Adapter exposing a GeoGrid value as a sea-level surface.

    Attributes:
        cells (dict[tuple[int, int], Any]): Occupied cells keyed by
            (column, row).
        n_x (int): Longitude column count.
        n_y (int): Latitude row count.
        cell_deg (float): Cell edge in degrees.
    """

    def __init__(self, grid: Any) -> None:
        """
        Initializes the GeoSeaLevelAdapter instance.

        Args:
            grid (Any): A ValueGeoGrid-like object with cells and a
                TypeGeoGrid-like value_type.
        """

        self.cells: dict[tuple[int, int], Any] = grid.cells
        self.n_x: int = grid.value_type.n_x
        self.n_y: int = grid.value_type.n_y
        self.cell_deg: float = grid.value_type.cell_deg

    def radius_at(self, lat_deg: float, lon_deg: float) -> Optional[float]:
        """
        Samples the sea-surface radius at a coordinate.

        Args:
            lat_deg (float): Latitude in degrees.
            lon_deg (float): Longitude in degrees.

        Returns:
            Optional[float]: The interpolated sea-surface radius in
            meters, or None when no surrounding cell has data.
        """

        return bilinear_sample(
            cells=self.cells,
            n_x=self.n_x,
            n_y=self.n_y,
            cell_deg=self.cell_deg,
            lat_deg=lat_deg,
            lon_deg=lon_deg,
        )



_SEA_LEVEL_SURFACE: Optional[Any] = None



def get_sea_level_surface() -> Optional[Any]:
    """
    Returns the registered sea-level surface, if any.

    Returns:
        Optional[Any]: The surface adapter, or None.
    """

    return _SEA_LEVEL_SURFACE



def set_sea_level_surface(surface: Optional[Any]) -> None:
    """
    Replaces the session sea-level surface.

    Args:
        surface (Optional[Any]): The surface adapter, or None to
            unregister.
    """

    global _SEA_LEVEL_SURFACE

    _SEA_LEVEL_SURFACE = surface
