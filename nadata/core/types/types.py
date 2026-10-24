"""
Core type definitions for Nadata.

Defines the base schema types including sized integers, sized floats,
strings, booleans, arrays, and dictionaries.
"""

# Import Modules
from __future__ import annotations
import math
from collections import deque
from typing import Any, Optional

from nadata.core.types.bits.bits import (
    BitValueTypeInt,
    BitValueTypeFloat,
)
from nadata.core.types.home_types.color import Color
from nadata.core.types.home_types.coordinate import Coordinate
from nadata.core.types.home_types.datetime import DateTime
from nadata.core.types.home_types.geometry import Polygon, Polyline
from nadata.core.types.home_types.planet import DEFAULT_PLANET, Planet



class Type:
    """
    Abstract base type for all Nadata schema types.

    Attributes:
        default_value (Any): The default value associated with this type.
    """

    def __init__(self) -> None:
        """
        Initializes the base Type instance.
        """

        # Default value of the type
        self.default_value: Any = None

    def check_value(self, value: Any) -> None:
        """
        Validates that a value satisfies this type specification.

        Args:
            value (Any): Value to validate.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """

        raise NotImplementedError("check_value must be implemented by subclasses.")

    def __repr__(self) -> str:
        """
        Returns string representation of the Type.

        Returns:
            str: Class name representation.
        """

        return self.__class__.__name__


class TypeStr(Type):
    """
    Schema type representing a Unicode string.

    Attributes:
        default_value (str): Default string value (empty string).
    """

    def __init__(self) -> None:
        """
        Initializes the TypeStr instance.
        """

        # Initialize parent class
        super().__init__()

        # Default value of the string type
        self.default_value: str = ""

    def check_value(self, value: Any) -> None:
        """
        Validates that a value is a string.

        Args:
            value (Any): Value to validate.

        Raises:
            TypeError: If value is not a string.
        """

        # Verify type is strictly a string
        if not isinstance(value, str):
            raise TypeError(f"Expected str, got {type(value).__name__}")

    def __repr__(self) -> str:
        """
        Returns string representation of the TypeStr.

        Returns:
            str: Representation of TypeStr.
        """

        return "Str"


class TypeInt(Type):
    """
    Schema type representing an integer with an exact bit width and signedness.

    Attributes:
        bit_spec (BitValueTypeInt): Underlying bit specification for bounds and validation.
        num_bits (int): Total bit width.
        signed (bool): Whether the integer is signed or unsigned.
        default_value (int): Default integer value (0).
    """

    def __init__(
        self,
        num_bits: int = 32,
        signed: bool = True,
        endian: str = "big"
    ) -> None:
        """
        Initializes the TypeInt instance.

        Args:
            num_bits (int, optional): Total bit width. Defaults to 32.
            signed (bool, optional): Whether integer is signed. Defaults to True.
            endian (str, optional): Byte/bit order ('big' or 'little'). Defaults to 'big'.
        """

        # Initialize parent class
        super().__init__()

        # Construct underlying bit specification
        self.bit_spec: BitValueTypeInt = BitValueTypeInt(
            num_bits=num_bits,
            signed=signed,
            endian=endian
        )

        self.num_bits: int = num_bits
        self.signed: bool = signed

        # Default value of the integer type
        self.default_value: int = 0

    def check_value(self, value: Any) -> None:
        """
        Validates that a value is an integer and within the bit bounds.

        Args:
            value (Any): Value to validate.

        Raises:
            TypeError: If value is not an int or is a bool.
            ValueError: If value is outside the representable bounds.
        """

        # Delegate validation to the bit specification
        self.bit_spec.check_value(value=value)

    def __repr__(self) -> str:
        """
        Returns string representation of the TypeInt.

        Returns:
            str: Representation with bit width and signedness.
        """

        sign_str: str = "Int" if self.signed else "UInt"
        return f"{sign_str}{self.num_bits}"


class TypeFloat(Type):
    """
    Schema type representing a floating-point number with custom bit specifications.

    Attributes:
        bit_spec (BitValueTypeFloat): Underlying bit specification for bounds and validation.
        num_bits (int): Total bit width.
        num_bits_exponent (int): Number of bits reserved for exponent.
        num_bits_mantissa (int): Number of bits reserved for mantissa.
        signed (bool): Whether the float has a sign bit.
        default_value (float): Default float value (0.0).
    """

    def __init__(
        self,
        num_bits: int = 32,
        num_bits_exponent: int = 8,
        signed: bool = True,
        endian: str = "big"
    ) -> None:
        """
        Initializes the TypeFloat instance.

        Args:
            num_bits (int, optional): Total bit width. Defaults to 32.
            num_bits_exponent (int, optional): Number of exponent bits. Defaults to 8.
            signed (bool, optional): Whether float is signed. Defaults to True.
            endian (str, optional): Byte/bit order ('big' or 'little'). Defaults to 'big'.
        """

        # Initialize parent class
        super().__init__()

        # Construct underlying bit specification
        self.bit_spec: BitValueTypeFloat = BitValueTypeFloat(
            num_bits=num_bits,
            num_bits_exponent=num_bits_exponent,
            signed=signed,
            endian=endian
        )

        self.num_bits: int = num_bits
        self.num_bits_exponent: int = num_bits_exponent
        self.num_bits_mantissa: int = self.bit_spec.num_bits_mantissa
        self.signed: bool = signed

        # Default value of the float type
        self.default_value: float = 0.0

    def check_value(self, value: Any) -> None:
        """
        Validates that a value is a float/int within the representable float bounds.

        Args:
            value (Any): Value to validate.

        Raises:
            TypeError: If value is not a float or int, or is a bool.
            ValueError: If unsigned float receives a negative value.
        """

        # Delegate validation to the bit specification
        self.bit_spec.check_value(value=value)

    def __repr__(self) -> str:
        """
        Returns string representation of the TypeFloat.

        Returns:
            str: Representation with bit details.
        """

        prefix: str = "Float" if self.signed else "UFloat"
        exp: int = self.num_bits_exponent
        mant: int = self.num_bits_mantissa
        return f"{prefix}{self.num_bits}(exp={exp}, mant={mant})"




class TypeBool(Type):
    """
    Schema type representing a boolean (True or False).

    Attributes:
        default_value (bool): Default boolean value (False).
    """

    def __init__(self) -> None:
        """
        Initializes the TypeBool instance.
        """

        # Initialize parent class
        super().__init__()

        # Default value of the boolean type
        self.default_value: bool = False

    def check_value(self, value: Any) -> None:
        """
        Validates that a value is a boolean.

        Args:
            value (Any): Value to validate.

        Raises:
            TypeError: If value is not a boolean.
        """

        # Verify type is strictly boolean
        if not isinstance(value, bool):
            raise TypeError(f"Expected bool, got {type(value).__name__}")

    def __repr__(self) -> str:
        """
        Returns string representation of the TypeBool.

        Returns:
            str: Representation of TypeBool.
        """

        return "Bool"


class TypeArray(Type):
    """
    Schema type representing a typed list of elements.

    Attributes:
        value_type (Type): Type of elements stored in the array.
        default_value (list[Any]): Default empty list.
    """

    def __init__(self, value_type: Type) -> None:
        """
        Initializes the TypeArray instance.

        Args:
            value_type (Type): Type of elements allowed in this array.
        """

        # Initialize parent class
        super().__init__()

        self.value_type: Type = value_type

        # Default value of the array type
        self.default_value: list[Any] = []

    def check_value(self, value: Any) -> None:
        """
        Validates that a value is a list and each item matches value_type.

        Args:
            value (Any): List value to validate.

        Raises:
            TypeError: If value is not a list or an item violates value_type.
        """

        # Accept query-like containers duck-typed via to_list (e.g. QuerySet)
        if hasattr(value, "to_list"):
            value = value.to_list()

        # Verify container type
        if not isinstance(value, list):
            raise TypeError(f"Expected list, got {type(value).__name__}")

        # Check each item against element type
        for item in value:
            self.value_type.check_value(value=item)

    def __repr__(self) -> str:
        """
        Returns string representation of the TypeArray.

        Returns:
            str: Representation showing element type.
        """

        return f"Array[{repr(self.value_type)}]"


class TypeDict(Type):
    """
    Schema type representing a typed dictionary of key-value pairs.

    Attributes:
        key_type (Type): Type of keys allowed in the dictionary.
        value_type (Type): Type of values allowed in the dictionary.
        default_value (dict[Any, Any]): Default empty dictionary.
    """

    def __init__(
        self,
        value_type: Type,
        key_type: Optional[Type] = None
    ) -> None:
        """
        Initializes the TypeDict instance.

        Args:
            value_type (Type): Type of values allowed in this dictionary.
            key_type (Optional[Type], optional): Type of keys. Defaults to TypeStr().
        """

        # Initialize parent class
        super().__init__()

        self.value_type: Type = value_type
        self.key_type: Type = key_type if key_type is not None else TypeStr()

        # Default value of the dictionary type
        self.default_value: dict[Any, Any] = {}

    def check_value(self, value: Any) -> None:
        """
        Validates that a value is a dict and all keys/values match specifications.

        Args:
            value (Any): Dictionary value to validate.

        Raises:
            TypeError: If value is not a dict or a key/value violates types.
        """

        # Verify container type
        if not isinstance(value, dict):
            raise TypeError(f"Expected dict, got {type(value).__name__}")

        # Check each key and value
        for key, val in value.items():
            self.key_type.check_value(value=key)
            self.value_type.check_value(value=val)

    def __repr__(self) -> str:
        """
        Returns string representation of the TypeDict.

        Returns:
            str: Representation showing key and value types.
        """

        return f"Dict[{repr(self.key_type)}, {repr(self.value_type)}]"


class TypeSet(Type):
    """
    Schema type representing a typed set of unique elements.

    Attributes:
        value_type (Type): Type of elements stored in the set.
        default_value (set[Any]): Default empty set.
    """

    def __init__(self, value_type: Type) -> None:
        """
        Initializes the TypeSet instance.

        Args:
            value_type (Type): Type of elements allowed in this set.
        """

        # Initialize parent class
        super().__init__()

        self.value_type: Type = value_type

        # Default value of the set type
        self.default_value: set[Any] = set()

    def check_value(self, value: Any) -> None:
        """
        Validates that a value is a set and each item matches value_type.

        Args:
            value (Any): Set value to validate.

        Raises:
            TypeError: If value is not a set or an item violates value_type.
        """

        # Verify container type
        if not isinstance(value, (set, frozenset)):
            raise TypeError(f"Expected set, got {type(value).__name__}")

        # Check each item against element type
        for item in value:
            self.value_type.check_value(value=item)

    def __repr__(self) -> str:
        """
        Returns string representation of the TypeSet.

        Returns:
            str: Representation showing element type.
        """

        return f"Set[{repr(self.value_type)}]"


class TypeQueue(Type):
    """
    Schema type representing a typed FIFO queue of elements.

    Attributes:
        value_type (Type): Type of elements stored in the queue.
        default_value (list[Any]): Default empty queue payload.
    """

    def __init__(self, value_type: Type) -> None:
        """
        Initializes the TypeQueue instance.

        Args:
            value_type (Type): Type of elements allowed in this queue.
        """

        # Initialize parent class
        super().__init__()

        self.value_type: Type = value_type

        # Default value of the queue type (converted to a deque by the
        # runtime wrapper; plain lists survive the JSON struct round trip)
        self.default_value: list[Any] = []

    def check_value(self, value: Any) -> None:
        """
        Validates that a value is a deque (or plain list payload) and each
        item matches value_type.

        Args:
            value (Any): Queue value to validate.

        Raises:
            TypeError: If value is not a deque/list or an item violates
                value_type.
        """

        # Verify container type
        if not isinstance(value, (deque, list)):
            raise TypeError(f"Expected deque, got {type(value).__name__}")

        # Check each item against element type
        for item in value:
            self.value_type.check_value(value=item)

    def __repr__(self) -> str:
        """
        Returns string representation of the TypeQueue.

        Returns:
            str: Representation showing element type.
        """

        return f"Queue[{repr(self.value_type)}]"


class TypePriorityQueue(Type):
    """
    Schema type representing a typed priority queue of elements.

    Elements are ordered by an associated priority of a basic numeric type
    (IntXX / FloatXX only); the lowest priority is served first.

    Attributes:
        value_type (Type): Type of elements stored in the queue.
        priority_type (Type): Numeric type of priorities (IntXX / FloatXX).
        default_value (list[Any]): Default empty queue payload.
    """

    def __init__(self, value_type: Type, priority_type: Type) -> None:
        """
        Initializes the TypePriorityQueue instance.

        Args:
            value_type (Type): Type of elements allowed in this queue.
            priority_type (Type): Numeric priority type.

        Raises:
            TypeError: If priority_type is not a basic IntXX / FloatXX.
        """

        # Initialize parent class
        super().__init__()

        # Restrict priorities to basic numeric types
        if not isinstance(priority_type, (TypeInt, TypeFloat)):
            raise TypeError(
                f"PriorityQueue priority type must be a basic IntXX or "
                f"FloatXX, got '{repr(priority_type)}'"
            )

        self.value_type: Type = value_type
        self.priority_type: Type = priority_type

        # Default value of the queue type (plain list of (priority, element)
        # pairs; survives the JSON struct round trip)
        self.default_value: list[Any] = []

    def check_value(self, value: Any) -> None:
        """
        Validates that a value is a list of (priority, element) pairs with
        numeric priorities and typed elements.

        Args:
            value (Any): Priority queue value to validate.

        Raises:
            TypeError: If value is not a pair list or an entry violates the
                priority or element type.
        """

        # Verify container type
        if not isinstance(value, list):
            raise TypeError(
                f"Expected list of (priority, element) pairs, got "
                f"{type(value).__name__}"
            )

        # Check each priority and element against their types
        for priority, element in value:
            self.priority_type.check_value(value=priority)
            self.value_type.check_value(value=element)

    def __repr__(self) -> str:
        """
        Returns string representation of the TypePriorityQueue.

        Returns:
            str: Representation showing element and priority types.
        """

        return f"PriorityQueue[{repr(self.value_type)}, {repr(self.priority_type)}]"


def flatten_nested_lists(raw: Any) -> list[Any]:
    """
    Flattens nested raw lists into a flat row-major element payload.

    Args:
        raw (Any): Raw payload from a (possibly nested) array literal.

    Returns:
        list[Any]: Flat element list.
    """

    flat: list[Any] = []

    for element in raw:

        if isinstance(element, list):

            flat.extend(flatten_nested_lists(raw=element))



        else:

            flat.append(element)



    return flat



# Supported spatial grid topologies (name -> (dimensions, neighbor offsets))
SPATIAL_TOPOLOGIES: dict[str, tuple[int, list[tuple[int, ...]]]] = {
    "square2d": (2, [(1, 0), (-1, 0), (0, 1), (0, -1)]),
    "hex2d": (2, [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]),
    "cube3d": (3, [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]),
}

# Topology tag per source type name
SPATIAL_GRID_NAMES: dict[str, str] = {
    "GridSquare2D": "square2d",
    "GridHex2D": "hex2d",
    "GridCube3D": "cube3d",
}

# Repr fragment per topology
SPATIAL_GRID_REPRS: dict[str, str] = {
    "square2d": "GridSquare2D",
    "hex2d": "GridHex2D",
    "cube3d": "GridCube3D",
}


class TypeSpatialGrid(Type):
    """
    Schema type representing a sparse spatial grid of typed cells.

    Cells live in a hash map keyed by integer coordinate tuples; absent
    keys are empty cells. The topology fixes the dimensionality and the
    canonical neighbor offsets (square2d: 4-adjacency, hex2d: axial
    6-adjacency, cube3d: 6-adjacency).

    Attributes:
        value_type (Type): Type of cell values.
        topology (str): Topology tag (SPATIAL_TOPOLOGIES key).
        default_value (dict[Any, Any]): Default empty grid.
    """

    def __init__(self, value_type: Type, topology: str) -> None:
        """
        Initializes the TypeSpatialGrid instance.

        Args:
            value_type (Type): Type of cell values.
            topology (str): Topology tag (SPATIAL_TOPOLOGIES key).

        Raises:
            ValueError: If the topology tag is unknown.
        """

        # Initialize parent class
        super().__init__()

        if topology not in SPATIAL_TOPOLOGIES:
            raise ValueError(f"Unknown spatial grid topology '{topology}'")

        self.value_type: Type = value_type
        self.topology: str = topology

        # Default value of the grid type (empty occupancy)
        self.default_value: dict[Any, Any] = {}

    @property
    def dims(self) -> int:
        """
        Returns the coordinate dimensionality of the topology.

        Returns:
            int: Number of coordinate components.
        """

        return SPATIAL_TOPOLOGIES[self.topology][0]

    def neighbor_offsets(self) -> list[tuple[int, ...]]:
        """
        Returns the canonical neighbor coordinate offsets.

        Returns:
            list[tuple[int, ...]]: Fixed-order offset tuples.
        """

        return SPATIAL_TOPOLOGIES[self.topology][1]

    def check_value(self, value: Any) -> None:
        """
        Validates that a value is a coordinate keyed dict of typed cells.

        Args:
            value (Any): Grid value to validate.

        Raises:
            TypeError: If value is not a dict, a coordinate tuple has the
                wrong arity or non-integer components, or a cell violates
                value_type.
        """

        # Verify container type
        if not isinstance(value, dict):
            raise TypeError(f"Expected dict, got {type(value).__name__}")

        # Check each coordinate and cell against the topology / value type
        for coords, cell in value.items():
            if not isinstance(coords, tuple) or len(coords) != self.dims:
                raise TypeError(
                    f"Expected a {self.dims}-component coordinate tuple"
                )

            for component in coords:
                if isinstance(component, bool) or not isinstance(component, int):
                    raise TypeError(
                        f"Expected an integer coordinate component, got "
                        f"{type(component).__name__}"
                    )

            self.value_type.check_value(value=cell)

    def __repr__(self) -> str:
        """
        Returns string representation of the TypeSpatialGrid.

        Returns:
            str: Representation showing the topology and cell type.
        """

        return f"{SPATIAL_GRID_REPRS[self.topology]}[{repr(self.value_type)}]"


class TypeGeoGrid(Type):
    """
    Schema type representing a sparse geodetic field layer.

    Cells live on a quantized lat/lon grid keyed by integer cell
    indices with longitude wrapping at plus/minus 180. The cell size
    is declared in arc-seconds; the effective cell edge is snapped so
    the longitude axis wraps exactly (360 / n_x degrees with n_x
    columns and n_y = n_x / 2 rows).

    Attributes:
        value_type (Type): Type of cell values.
        cell_arcsec (int): Nominal cell size in arc-seconds.
        default_value (dict[Any, Any]): Default empty grid.
    """

    def __init__(self, value_type: Type, cell_arcsec: int = 30) -> None:
        """
        Initializes the TypeGeoGrid instance.

        Args:
            value_type (Type): Type of cell values.
            cell_arcsec (int, optional): Nominal cell size in
                arc-seconds. Defaults to 30 (roughly one kilometer on
                Earth).

        Raises:
            TypeError: If cell_arcsec is not an integer.
            ValueError: If cell_arcsec is not a positive whole number
                of arc-seconds not exceeding a hemisphere.
        """

        # Initialize parent class
        super().__init__()

        if isinstance(cell_arcsec, bool) or not isinstance(cell_arcsec, int):
            raise TypeError(
                f"Expected an integer cell size in arc-seconds, got "
                f"{type(cell_arcsec).__name__}"
            )

        if cell_arcsec <= 0 or cell_arcsec > 648000:
            raise ValueError(
                f"Cell size must be between 1 and 648000 arc-seconds, "
                f"got {cell_arcsec}"
            )

        self.value_type: Type = value_type
        self.cell_arcsec: int = cell_arcsec

        # Snap the column count so longitude wraps exactly; at least
        # two columns (a hemisphere-wide cell) and two rows
        self.n_x: int = max(2, round(360.0 * 3600.0 / cell_arcsec))
        self.n_y: int = max(1, self.n_x // 2)

        # Default value of the grid type (empty occupancy)
        self.default_value: dict[Any, Any] = {}

    @property
    def cell_deg(self) -> float:
        """
        Returns the effective cell edge in degrees.

        Returns:
            float: Cell edge (360 / n_x).
        """

        return 360.0 / self.n_x

    def cell_of(self, lat_deg: float, lon_deg: float) -> tuple[int, int]:
        """
        Returns the cell key containing a coordinate.

        Args:
            lat_deg (float): Latitude in degrees.
            lon_deg (float): Longitude in degrees.

        Returns:
            tuple[int, int]: The (column, row) cell key.
        """

        ix: int = int(math.floor((lon_deg + 180.0) / self.cell_deg)) % self.n_x
        iy: int = int(math.floor((lat_deg + 90.0) / self.cell_deg))

        iy = min(self.n_y - 1, max(0, iy))

        return (ix, iy)

    def cell_center(self, cell: tuple[int, int]) -> tuple[float, float]:
        """
        Returns the center latitude / longitude of a cell.

        Args:
            cell (tuple[int, int]): The (column, row) cell key.

        Returns:
            tuple[float, float]: The (lat, lon) center in degrees.
        """

        lat: float = (cell[1] + 0.5) * self.cell_deg - 90.0
        lon: float = (cell[0] + 0.5) * self.cell_deg - 180.0

        return (lat, lon)

    def check_value(self, value: Any) -> None:
        """
        Validates that a value is a cell keyed dict of typed values.

        Args:
            value (Any): Grid value to validate.

        Raises:
            TypeError: If value is not a dict, a cell key is malformed,
                or a cell violates value_type.
        """

        # Verify container type
        if not isinstance(value, dict):
            raise TypeError(f"Expected dict, got {type(value).__name__}")

        # Check each cell key and value
        for coords, cell in value.items():
            if (
                not isinstance(coords, tuple)
                or len(coords) != 2
                or isinstance(coords[0], bool)
                or isinstance(coords[1], bool)
                or not isinstance(coords[0], int)
                or not isinstance(coords[1], int)
            ):
                raise TypeError(
                    "Expected an integer (column, row) cell key, got "
                    f"{repr(coords)}"
                )

            if not (0 <= coords[0] < self.n_x and 0 <= coords[1] < self.n_y):
                raise TypeError(
                    f"Cell key {coords} outside the "
                    f"{self.n_x} x {self.n_y} grid"
                )

            self.value_type.check_value(value=cell)

    def __repr__(self) -> str:
        """
        Returns string representation of the TypeGeoGrid.

        Returns:
            str: Representation showing the cell type and cell size.
        """

        return f"GeoGrid[{repr(self.value_type)}, {self.cell_arcsec}]"


class TypeTensor(Type):
    """
    Schema type representing a dense tensor with explicit dimensions.

    The shape is part of the type: values are flat row-major lists whose
    length must equal the product of the dimensions. Dimensions are either
    positive integer literals or the names of `const` integer attributes of
    the owning struct (symbolic dims bound per instance).

    Attributes:
        value_type (Type): Type of elements.
        dims (list[int | str]): Dimension extents (ints >= 1) or symbolic
            names resolved against the owning struct's const attributes.
        default_value (list[Any]): Default empty payload.
    """

    def __init__(self, value_type: Type, dims: list[int | str]) -> None:
        """
        Initializes the TypeTensor instance.

        Args:
            value_type (Type): Type of elements.
            dims (list[int | str]): Dimension extents (ints >= 1) or
                symbolic constant names.

        Raises:
            ValueError: If an integer dimension is below 1 or a symbolic
                name is empty.
        """

        # Initialize parent class
        super().__init__()

        for dim in dims:

            if isinstance(dim, str):

                if not dim:
                    raise ValueError("Tensor dimension names must be non-empty")

                continue

            if dim < 1:
                raise ValueError("Tensor dimensions must be >= 1")

        self.value_type: Type = value_type
        self.dims: list[int | str] = list(dims)

        # Default value of the tensor type (never validated; tensors are
        # materialized explicitly)
        self.default_value: list[Any] = []

    @property
    def is_symbolic(self) -> bool:
        """
        Returns whether any dimension is a symbolic constant name.

        Returns:
            bool: True if at least one dimension is symbolic.
        """

        return any(isinstance(dim, str) for dim in self.dims)

    def symbolic_dims(self) -> list[str]:
        """
        Returns the symbolic dimension names in declaration order.

        Returns:
            list[str]: Names of the symbolic dimensions.
        """

        return [dim for dim in self.dims if isinstance(dim, str)]

    def length(self) -> Optional[int]:
        """
        Returns the total element count of the shape when fully literal.

        Returns:
            Optional[int]: Product of the dimensions, or None if any
                dimension is symbolic.
        """

        total: int = 1

        for dim in self.dims:

            if isinstance(dim, str):
                return None

            total *= dim

        return total

    def check_value(self, value: Any) -> None:
        """
        Validates that a value is a flat row-major list matching the shape.

        Symbolic dimensions cannot be checked without an instance (the
        concrete extents live in the const attributes), so only fully
        literal shapes enforce the length here.

        Args:
            value (Any): Tensor value to validate.

        Raises:
            TypeError: If value is not a list, its length does not match a
                literal shape, or an element violates value_type.
        """

        # Verify container type
        if not isinstance(value, list):
            raise TypeError(f"Expected list, got {type(value).__name__}")

        # Verify exact shape (literal shapes only)
        expected: Optional[int] = self.length()

        if expected is not None and len(value) != expected:
            raise TypeError(
                f"Expected {expected} elements for shape "
                f"{self.dims}, got {len(value)}"
            )

        # Check each element against the element type
        for element in value:
            self.value_type.check_value(value=element)

    def __repr__(self) -> str:
        """
        Returns string representation of the TypeTensor.

        Returns:
            str: Representation showing the element type and shape.
        """

        dims_text: str = ", ".join(str(dim) for dim in self.dims)

        return f"Tensor[{repr(self.value_type)}, {dims_text}]"


class TypeDateTime(Type):
    """
    Schema type representing a date and time.
    """

    def __init__(self) -> None:
        """
        Initializes the TypeDateTime instance.
        """

        # Initialize parent class
        super().__init__()

        # Default value is a default DateTime instance
        self.default_value: DateTime = DateTime()

    def check_value(self, value: Any) -> None:
        """
        Validates that a value is a DateTime instance.

        Args:
            value (Any): Value to validate.

        Raises:
            TypeError: If value is not a DateTime.
        """

        if not isinstance(value, DateTime):
            raise TypeError(f"Expected DateTime, got {type(value).__name__}")

    def __repr__(self) -> str:
        """
        Returns string representation of TypeDateTime.

        Returns:
            str: "DateTime".
        """

        return "DateTime"


class TypeCoordinate(Type):
    """
    Schema type representing a geodetic coordinate (latitude,
    longitude, radius from the planet center).
    """

    def __init__(self) -> None:
        """
        Initializes the TypeCoordinate instance.
        """

        # Initialize parent class
        super().__init__()

        # Default value is a sea-level coordinate at the equator
        self.default_value: Coordinate = Coordinate(lat=0.0, lon=0.0, radius=6371008.8)

    def check_value(self, value: Any) -> None:
        """
        Validates that a value is a Coordinate instance.

        Args:
            value (Any): Value to validate.

        Raises:
            TypeError: If value is not a Coordinate.
        """

        if not isinstance(value, Coordinate):
            raise TypeError(f"Expected Coordinate, got {type(value).__name__}")

    def __repr__(self) -> str:
        """
        Returns string representation of TypeCoordinate.

        Returns:
            str: "Coordinate".
        """

        return "Coordinate"


class TypePolyline(Type):
    """
    Schema type representing a geodesic path over Coordinates.
    """

    def __init__(self) -> None:
        """
        Initializes the TypePolyline instance.
        """

        # Initialize parent class
        super().__init__()

        # Default value is the empty path
        self.default_value: Polyline = Polyline(vertices=[])

    def check_value(self, value: Any) -> None:
        """
        Validates that a value is a Polyline instance.

        Args:
            value (Any): Value to validate.

        Raises:
            TypeError: If value is not a Polyline.
        """

        if not isinstance(value, Polyline):
            raise TypeError(f"Expected Polyline, got {type(value).__name__}")

    def __repr__(self) -> str:
        """
        Returns string representation of TypePolyline.

        Returns:
            str: "Polyline".
        """

        return "Polyline"


class TypePolygon(Type):
    """
    Schema type representing a closed ring over Coordinates.
    """

    def __init__(self) -> None:
        """
        Initializes the TypePolygon instance.
        """

        # Initialize parent class
        super().__init__()

        # Default value is the empty polygon
        self.default_value: Polygon = Polygon(vertices=[])

    def check_value(self, value: Any) -> None:
        """
        Validates that a value is a Polygon instance.

        Args:
            value (Any): Value to validate.

        Raises:
            TypeError: If value is not a Polygon.
        """

        if not isinstance(value, Polygon):
            raise TypeError(f"Expected Polygon, got {type(value).__name__}")

    def __repr__(self) -> str:
        """
        Returns string representation of TypePolygon.

        Returns:
            str: "Polygon".
        """

        return "Polygon"


class TypePlanet(Type):
    """
    Schema type representing a planet shape (sphere or spheroid).
    """

    def __init__(self) -> None:
        """
        Initializes the TypePlanet instance.
        """

        # Initialize parent class
        super().__init__()

        # Default value is the default Earth sphere
        self.default_value: Planet = DEFAULT_PLANET

    def check_value(self, value: Any) -> None:
        """
        Validates that a value is a Planet instance.

        Args:
            value (Any): Value to validate.

        Raises:
            TypeError: If value is not a Planet.
        """

        if not isinstance(value, Planet):
            raise TypeError(f"Expected Planet, got {type(value).__name__}")

    def __repr__(self) -> str:
        """
        Returns string representation of TypePlanet.

        Returns:
            str: "Planet".
        """

        return "Planet"


class TypeColor(Type):
    """
    Schema type representing an RGBA color.
    """

    def __init__(self) -> None:
        """
        Initializes the TypeColor instance.
        """

        # Initialize parent class
        super().__init__()

        # Default value is black
        self.default_value: Color = Color(0, 0, 0, 255)

    def check_value(self, value: Any) -> None:
        """
        Validates that a value is a Color instance.

        Args:
            value (Any): Value to validate.

        Raises:
            TypeError: If value is not a Color.
        """

        if not isinstance(value, Color):
            raise TypeError(f"Expected Color, got {type(value).__name__}")

    def __repr__(self) -> str:
        """
        Returns string representation of TypeColor.

        Returns:
            str: "Color".
        """

        return "Color"


# Predefined Sized Integer and Float Aliases
# pylint: disable=invalid-name
Int2: TypeInt = TypeInt(num_bits=2, signed=True)
Int4: TypeInt = TypeInt(num_bits=4, signed=True)
Int8: TypeInt = TypeInt(num_bits=8, signed=True)
Int16: TypeInt = TypeInt(num_bits=16, signed=True)
Int32: TypeInt = TypeInt(num_bits=32, signed=True)
Int64: TypeInt = TypeInt(num_bits=64, signed=True)

UInt2: TypeInt = TypeInt(num_bits=2, signed=False)
UInt4: TypeInt = TypeInt(num_bits=4, signed=False)
UInt8: TypeInt = TypeInt(num_bits=8, signed=False)
UInt16: TypeInt = TypeInt(num_bits=16, signed=False)
UInt32: TypeInt = TypeInt(num_bits=32, signed=False)
UInt64: TypeInt = TypeInt(num_bits=64, signed=False)

# Predefined Sized Float Aliases
Float8_E4M3: TypeFloat = TypeFloat(num_bits=8, num_bits_exponent=4, signed=True)
Float8_E5M2: TypeFloat = TypeFloat(num_bits=8, num_bits_exponent=5, signed=True)
Float16: TypeFloat = TypeFloat(num_bits=16, num_bits_exponent=5, signed=True)
BFloat16: TypeFloat = TypeFloat(num_bits=16, num_bits_exponent=8, signed=True)
Float32: TypeFloat = TypeFloat(num_bits=32, num_bits_exponent=8, signed=True)
Float64: TypeFloat = TypeFloat(num_bits=64, num_bits_exponent=11, signed=True)

# Predefined Unsigned Float Aliases
UFloat8_E4M3: TypeFloat = TypeFloat(num_bits=8, num_bits_exponent=4, signed=False)
UFloat8_E5M2: TypeFloat = TypeFloat(num_bits=8, num_bits_exponent=5, signed=False)
UFloat16: TypeFloat = TypeFloat(num_bits=16, num_bits_exponent=5, signed=False)
UBFloat16: TypeFloat = TypeFloat(num_bits=16, num_bits_exponent=8, signed=False)
UFloat32: TypeFloat = TypeFloat(num_bits=32, num_bits_exponent=8, signed=False)
UFloat64: TypeFloat = TypeFloat(num_bits=64, num_bits_exponent=11, signed=False)
