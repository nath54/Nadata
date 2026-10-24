"""
Binary row and batch encoding for Nadata instance records (format v2).

Encodes instance rows positionally against the struct's declared attribute
types, so type information lives once in the schema documents instead of
being re-encoded per value. The JSON codec remains responsible for meta,
struct, function, and symbol documents; only instance rows switch to the
compact binary form.

Row layout (inside a batch document):

    [presence bitmap: ceil(len(columns) / 8) bytes; bit i set = column i
     carries a value on this row]
    [escape bitmap: same size; bit i set = column i is stored as JSON]
    [typed values for present, non-escaped columns, in column order]
    [varint escaped count]
    [per escaped column: varint column index, varint byte length, JSON
     bytes of the codec-encoded Value]

Value encoding is driven by the declared type:

    - TypeInt: little-endian signed/unsigned of the declared width
    - TypeFloat: '<f' for <=32 bits, '<d' otherwise
    - TypeBool: one byte
    - TypeStr: varint byte length + UTF-8 payload
    - TypeDateTime: 7 x int32 (y, m, d, H, M, S, MS)
    - TypeColor: 4 x uint8 (r, g, b, a)
    - TypeCoordinate: 3 x float64 (lat, lon, radius)
    - TypePolyline / TypePolygon: varint vertex count + per vertex
      3 x float64 (exact round trips; delta compression deferred)
    - TypePlanet: one flag byte (0 sphere, 1 spheroid) + float64
      equatorial radius (+ float64 polar radius when spheroid)
    - Struct type: varint of the referenced entity id (resolved through a
      row resolver on decode, which may decode the referenced row lazily)
    - TypeArray / TypeDict: varint element count + per element value
    - anything else: stored via the escape hatch (JSON codec bytes)
"""

# Import Modules
from __future__ import annotations
from collections import deque
import json
import struct
from typing import Any, Callable, Optional

from nadata.core.database.persistence.serialization import (
    DecodeContext,
    ObjectCodec,
    resolve_instance_refs,
)
from nadata.core.struct.struct import Struct
from nadata.core.types.types import (
    Type,
    TypeArray,
    TypeBool,
    TypeColor,
    TypeCoordinate,
    TypeDateTime,
    TypeDict,
    TypeFloat,
    TypeInt,
    TypePlanet,
    TypePolygon,
    TypePolyline,
    TypePriorityQueue,
    TypeQueue,
    TypeSet,
    TypeSpatialGrid,
    TypeGeoGrid,
    TypeStr,
    TypeTensor,
)
from nadata.core.values.values import Value

# Little-endian struct format characters per (num_bits, signed)
_INT_FORMATS: dict[tuple[int, bool], str] = {
    (8, True): "b",
    (8, False): "B",
    (16, True): "h",
    (16, False): "H",
    (32, True): "i",
    (32, False): "I",
    (64, True): "q",
    (64, False): "Q",
}

# Batch document header fields
_BATCH_KIND: str = "instances"
_BATCH_FORMAT: str = "binary"


def encode_varint(value: int) -> bytes:
    """
    Encodes a non-negative integer as an LEB128 varint.

    Args:
        value (int): Non-negative integer to encode.

    Returns:
        bytes: Encoded varint bytes.

    Raises:
        ValueError: If the value is negative.
    """

    if value < 0:
        raise ValueError(f"Varint values must be non-negative, got {value}")

    out: bytearray = bytearray()

    while True:
        byte: int = value & 0x7F

        value >>= 7

        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)

            return bytes(out)


def decode_varint(payload: bytes, offset: int) -> tuple[int, int]:
    """
    Decodes an LEB128 varint from a payload.

    Args:
        payload (bytes): Payload containing the varint.
        offset (int): Offset of the varint's first byte.

    Returns:
        tuple[int, int]: The decoded value and the offset after it.
    """

    result: int = 0
    shift: int = 0

    while True:
        byte: int = payload[offset]

        offset += 1

        result |= (byte & 0x7F) << shift

        shift += 7

        if not byte & 0x80:
            return result, offset


def encode_zigzag(value: int) -> bytes:
    """
    Encodes a signed integer as a zigzag varint.

    Args:
        value (int): Signed integer to encode.

    Returns:
        bytes: Encoded zigzag varint bytes.
    """

    return encode_varint(value=(value << 1) ^ (value >> 63))


def decode_zigzag(payload: bytes, offset: int) -> tuple[int, int]:
    """
    Decodes a zigzag varint into a signed integer.

    Args:
        payload (bytes): Payload containing the varint.
        offset (int): Offset of the varint's first byte.

    Returns:
        tuple[int, int]: The decoded value and the offset after it.
    """

    decoded: int
    decoded, offset = decode_varint(payload=payload, offset=offset)

    return (decoded >> 1) ^ -(decoded & 1), offset


def _is_struct_type(value_type: Any) -> bool:
    """
    Checks whether a schema type is a struct (duck-typed as a Type).

    Args:
        value_type (Any): Candidate type.

    Returns:
        bool: True if the type is a Struct.
    """

    return isinstance(value_type, Struct)


def _int_format(num_bits: int, signed: bool) -> str:
    """
    Resolves the struct format character for an integer width.

    Args:
        num_bits (int): Declared bit width.
        signed (bool): Signedness.

    Returns:
        str: Struct format character.

    Raises:
        ValueError: On unsupported widths.
    """

    fmt: Optional[str] = _INT_FORMATS.get((num_bits, signed))

    if fmt is None:
        raise ValueError(f"Unsupported integer width {num_bits} (signed={signed})")

    return fmt


def encode_value(raw: Any, value_type: Any) -> bytes:
    """
    Encodes a raw payload against its declared schema type.

    Args:
        raw (Any): Raw payload (from Value.get()).
        value_type (Any): Declared schema type (Type instance or Struct).

    Returns:
        bytes: Encoded value bytes.

    Raises:
        TypeError: If the payload does not fit the declared type; callers
            treat this as a signal to use the escape hatch.
    """

    if _is_struct_type(value_type):
        from nadata.core.database.instance import StructInstance

        if not isinstance(raw, StructInstance):
            raise TypeError(f"Expected a struct instance, got '{type(raw).__name__}'")

        return encode_varint(value=raw.entity_id.value)

    if isinstance(value_type, TypeBool):
        return b"\x01" if raw else b"\x00"

    if isinstance(value_type, TypeInt):
        fmt: str = _int_format(num_bits=value_type.num_bits, signed=value_type.signed)

        return struct.pack("<" + fmt, raw)

    if isinstance(value_type, TypeFloat):
        # Floats always encode as 64-bit to preserve exact round trips
        # (the runtime stores Python floats regardless of declared width)
        return struct.pack("<d", float(raw))

    if isinstance(value_type, TypeStr):
        encoded: bytes = raw.encode("utf-8")

        return encode_varint(value=len(encoded)) + encoded

    if isinstance(value_type, TypeDateTime):
        return struct.pack(
            "<7i",
            raw.year,
            raw.month,
            raw.day,
            raw.hour,
            raw.minute,
            raw.second,
            raw.millisecond,
        )

    if isinstance(value_type, TypeCoordinate):
        return struct.pack("<3d", float(raw.lat), float(raw.lon), float(raw.radius))

    if isinstance(value_type, (TypePolyline, TypePolygon)):
        ring_bytes: bytearray = bytearray(encode_varint(value=len(raw.vertices)))

        for vertex in raw.vertices:
            ring_bytes += struct.pack(
                "<3d", float(vertex.lat), float(vertex.lon), float(vertex.radius)
            )

        return bytes(ring_bytes)

    if isinstance(value_type, TypePlanet):
        if raw.is_sphere:
            return b"\x00" + struct.pack("<d", raw.equatorial_radius)

        return b"\x01" + struct.pack(
            "<2d", raw.equatorial_radius, raw.polar_radius
        )

    if isinstance(value_type, TypeColor):
        return struct.pack("<4B", raw.r, raw.g, raw.b, raw.a)

    if isinstance(value_type, TypeArray):
        out: bytearray = bytearray(encode_varint(value=len(raw)))

        for element in raw:
            out += encode_value(raw=element, value_type=value_type.value_type)

        return bytes(out)

    if isinstance(value_type, TypeDict):
        out = bytearray(encode_varint(value=len(raw)))

        for key, entry_value in raw.items():
            out += encode_value(raw=key, value_type=value_type.key_type)
            out += encode_value(raw=entry_value, value_type=value_type.value_type)

        return bytes(out)

    if isinstance(value_type, TypeSet):
        # Sort for deterministic bytes: sets iterate in hash order, which is
        # unstable across runs for strings
        ordered: list[Any] = _ordered_elements(raw=raw)

        out = bytearray(encode_varint(value=len(ordered)))

        for element in ordered:
            out += encode_value(raw=element, value_type=value_type.value_type)

        return bytes(out)

    if isinstance(value_type, TypeQueue):
        out = bytearray(encode_varint(value=len(raw)))

        for element in raw:
            out += encode_value(raw=element, value_type=value_type.value_type)

        return bytes(out)

    if isinstance(value_type, TypePriorityQueue):
        # Normalize order for deterministic bytes: same content, same bytes
        # regardless of push history
        ordered_entries: list[Any] = sorted(
            raw, key=lambda entry: (entry[0], repr(entry[1]))
        )

        out = bytearray(encode_varint(value=len(ordered_entries)))

        for priority, element in ordered_entries:
            out += encode_value(raw=priority, value_type=value_type.priority_type)
            out += encode_value(raw=element, value_type=value_type.value_type)

        return bytes(out)

    if isinstance(value_type, TypeSpatialGrid):
        # Sort by coordinate tuple for deterministic bytes
        out = bytearray(encode_varint(value=len(raw)))

        for coords in sorted(raw.keys()):
            for component in coords:
                out += encode_zigzag(value=component)

            out += encode_value(raw=raw[coords], value_type=value_type.value_type)

        return bytes(out)

    if isinstance(value_type, TypeGeoGrid):
        # Sorted cells; the per-value sample default is session state
        # and is re-established by scripts after load
        out = bytearray(encode_varint(value=len(raw)))

        for coords in sorted(raw.keys()):
            out += encode_zigzag(value=coords[0])
            out += encode_zigzag(value=coords[1])

            out += encode_value(raw=raw[coords], value_type=value_type.value_type)

        return bytes(out)

    if isinstance(value_type, TypeTensor):
        # Flat row-major payload; the shape lives in the type
        out = bytearray(encode_varint(value=len(raw)))

        for element in raw:
            out += encode_value(raw=element, value_type=value_type.value_type)

        return bytes(out)

    raise TypeError(f"No binary encoding for declared type '{repr(value_type)}'")


def decode_value(
    payload: bytes,
    offset: int,
    value_type: Any,
    resolver: Callable[[int], Any],
) -> tuple[Any, int]:
    """
    Decodes a value encoded by encode_value against its declared type.

    Args:
        payload (bytes): Row payload containing the value.
        offset (int): Offset of the value's first byte.
        value_type (Any): Declared schema type.
        resolver (Callable[[int], Any]): Resolves an entity id to its
            StructInstance (decoding the referenced row if needed).

    Returns:
        tuple[Any, int]: The decoded raw payload and the offset after it.
    """

    if _is_struct_type(value_type):
        entity_id: int
        entity_id, offset = decode_varint(payload=payload, offset=offset)

        return resolver(entity_id), offset

    if isinstance(value_type, TypeBool):
        return bool(payload[offset]), offset + 1

    if isinstance(value_type, TypeInt):
        fmt: str = _int_format(num_bits=value_type.num_bits, signed=value_type.signed)

        return struct.unpack_from("<" + fmt, payload, offset)[0], (
            offset + struct.calcsize("<" + fmt)
        )

    if isinstance(value_type, TypeFloat):
        return struct.unpack_from("<d", payload, offset)[0], offset + 8

    if isinstance(value_type, TypeStr):
        length: int
        length, offset = decode_varint(payload=payload, offset=offset)

        return payload[offset:offset + length].decode("utf-8"), offset + length

    if isinstance(value_type, TypeDateTime):
        fields: tuple[int, ...] = struct.unpack_from("<7i", payload, offset)

        from nadata.core.types.home_types.datetime import DateTime

        return DateTime(
            y=fields[0],
            m=fields[1],
            d=fields[2],
            H=fields[3],
            M=fields[4],
            S=fields[5],
            MS=fields[6],
        ), offset + 28

    if isinstance(value_type, TypeCoordinate):
        lat: float
        lon: float
        radius: float
        lat, lon, radius = struct.unpack_from("<3d", payload, offset)

        from nadata.core.types.home_types.coordinate import Coordinate

        return Coordinate(lat=lat, lon=lon, radius=radius), offset + 24

    if isinstance(value_type, (TypePolyline, TypePolygon)):
        vertex_count: int
        vertex_count, offset = decode_varint(payload=payload, offset=offset)

        from nadata.core.types.home_types.coordinate import Coordinate as _Coordinate
        from nadata.core.types.home_types.geometry import Polygon as _Polygon
        from nadata.core.types.home_types.geometry import Polyline as _Polyline

        vertices: list[Any] = []

        for _ in range(vertex_count):
            v_lat: float
            v_lon: float
            v_radius: float
            v_lat, v_lon, v_radius = struct.unpack_from("<3d", payload, offset)
            offset += 24

            vertices.append(_Coordinate(lat=v_lat, lon=v_lon, radius=v_radius))

        if isinstance(value_type, TypePolyline):
            return _Polyline(vertices=vertices), offset

        return _Polygon(vertices=vertices), offset

    if isinstance(value_type, TypePlanet):
        is_spheroid: bool = payload[offset] == 1
        offset += 1

        from nadata.core.types.home_types.planet import Planet

        if not is_spheroid:
            equatorial: float = struct.unpack_from("<d", payload, offset)[0]

            return Planet(equatorial_radius=equatorial, polar_radius=None), offset + 8

        equatorial_pair: tuple[float, float] = struct.unpack_from(
            "<2d", payload, offset
        )

        return Planet(
            equatorial_radius=equatorial_pair[0],
            polar_radius=equatorial_pair[1],
        ), offset + 16

    if isinstance(value_type, TypeColor):
        channels: tuple[int, ...] = struct.unpack_from("<4B", payload, offset)

        from nadata.core.types.home_types.color import Color

        return Color(r=channels[0], g=channels[1], b=channels[2], a=channels[3]), (
            offset + 4
        )

    if isinstance(value_type, TypeArray):
        count: int
        count, offset = decode_varint(payload=payload, offset=offset)

        elements: list[Any] = []

        for _ in range(count):
            element: Any
            element, offset = decode_value(
                payload=payload,
                offset=offset,
                value_type=value_type.value_type,
                resolver=resolver,
            )

            elements.append(element)

        return elements, offset

    if isinstance(value_type, TypeDict):
        count, offset = decode_varint(payload=payload, offset=offset)

        entries: dict[Any, Any] = {}

        for _ in range(count):
            key: Any
            key, offset = decode_value(
                payload=payload,
                offset=offset,
                value_type=value_type.key_type,
                resolver=resolver,
            )

            entry: Any
            entry, offset = decode_value(
                payload=payload,
                offset=offset,
                value_type=value_type.value_type,
                resolver=resolver,
            )

            entries[key] = entry

        return entries, offset

    if isinstance(value_type, TypeSet):
        set_count: int
        set_count, offset = decode_varint(payload=payload, offset=offset)

        decoded_elements: set[Any] = set()

        for _ in range(set_count):
            decoded_element: Any
            decoded_element, offset = decode_value(
                payload=payload,
                offset=offset,
                value_type=value_type.value_type,
                resolver=resolver,
            )

            decoded_elements.add(decoded_element)

        return decoded_elements, offset

    if isinstance(value_type, TypeQueue):
        queue_count: int
        queue_count, offset = decode_varint(payload=payload, offset=offset)

        queued: deque[Any] = deque()

        for _ in range(queue_count):
            queued_element: Any
            queued_element, offset = decode_value(
                payload=payload,
                offset=offset,
                value_type=value_type.value_type,
                resolver=resolver,
            )

            queued.append(queued_element)

        return queued, offset

    if isinstance(value_type, TypePriorityQueue):
        pq_count: int
        pq_count, offset = decode_varint(payload=payload, offset=offset)

        pq_entries: list[Any] = []

        for _ in range(pq_count):
            priority: Any
            priority, offset = decode_value(
                payload=payload,
                offset=offset,
                value_type=value_type.priority_type,
                resolver=resolver,
            )

            pq_element: Any
            pq_element, offset = decode_value(
                payload=payload,
                offset=offset,
                value_type=value_type.value_type,
                resolver=resolver,
            )

            pq_entries.append((priority, pq_element))

        return pq_entries, offset

    if isinstance(value_type, TypeSpatialGrid):
        grid_count: int
        grid_count, offset = decode_varint(payload=payload, offset=offset)

        grid_cells: dict[tuple[int, ...], Any] = {}

        for _ in range(grid_count):
            coords: list[int] = []

            for _ in range(value_type.dims):
                component: int
                component, offset = decode_zigzag(payload=payload, offset=offset)

                coords.append(component)

            grid_element: Any
            grid_element, offset = decode_value(
                payload=payload,
                offset=offset,
                value_type=value_type.value_type,
                resolver=resolver,
            )

            grid_cells[tuple(coords)] = grid_element

        return grid_cells, offset

    if isinstance(value_type, TypeGeoGrid):
        cell_count: int
        cell_count, offset = decode_varint(payload=payload, offset=offset)

        geo_cells: dict[tuple[int, int], Any] = {}

        for _ in range(cell_count):
            ix: int
            iy: int

            ix, offset = decode_zigzag(payload=payload, offset=offset)
            iy, offset = decode_zigzag(payload=payload, offset=offset)

            cell_element: Any
            cell_element, offset = decode_value(
                payload=payload,
                offset=offset,
                value_type=value_type.value_type,
                resolver=resolver,
            )

            geo_cells[(ix, iy)] = cell_element

        return geo_cells, offset

    if isinstance(value_type, TypeTensor):
        tensor_count: int
        tensor_count, offset = decode_varint(payload=payload, offset=offset)

        tensor_cells: list[Any] = []

        for _ in range(tensor_count):
            tensor_element: Any
            tensor_element, offset = decode_value(
                payload=payload,
                offset=offset,
                value_type=value_type.value_type,
                resolver=resolver,
            )

            tensor_cells.append(tensor_element)

        return tensor_cells, offset

    raise TypeError(f"No binary decoding for declared type '{repr(value_type)}'")


def _ordered_elements(raw: Any) -> list[Any]:
    """
    Returns set elements in a deterministic order for encoding.

    Uses natural ordering when elements are comparable, falling back to
    their repr (e.g. DateTime, struct instances) so the same logical set
    always encodes to the same bytes.

    Args:
        raw (Any): The raw set payload.

    Returns:
        list[Any]: Elements sorted by a stable key.
    """

    try:
        return sorted(raw)

    except TypeError:
        return sorted(raw, key=repr)


def _bitmaps_size(column_count: int) -> int:
    """
    Computes the byte size of one row bitmap.

    Args:
        column_count (int): Number of columns.

    Returns:
        int: Bitmap size in bytes.
    """

    return (column_count + 7) // 8


def _set_bit(bitmap: bytearray, index: int) -> None:
    """
    Sets bit `index` of a bitmap in place.

    Args:
        bitmap (bytearray): Bitmap buffer.
        index (int): Bit index.
    """

    bitmap[index // 8] |= 1 << (index % 8)


def _get_bit(bitmap: bytes, index: int) -> bool:
    """
    Reads bit `index` of a bitmap.

    Args:
        bitmap (bytes): Bitmap buffer.
        index (int): Bit index.

    Returns:
        bool: True if the bit is set.
    """

    return bool(bitmap[index // 8] & (1 << (index % 8)))


def encode_row(
    attributes: dict[str, Value],
    columns: list[str],
    types: dict[str, Type],
) -> bytes:
    """
    Encodes one instance row against the batch's column layout.

    Args:
        attributes (dict[str, Value]): Instance attribute values by name
            (may be sparse: missing names become absent bits).
        columns (list[str]): Batch column names in order.
        types (dict[str, Type]): Declared schema type per column.

    Returns:
        bytes: Encoded row payload.
    """

    bitmap_size: int = _bitmaps_size(column_count=len(columns))
    presence: bytearray = bytearray(bitmap_size)
    escaped: bytearray = bytearray(bitmap_size)
    typed: bytearray = bytearray()
    escaped_payloads: list[tuple[int, bytes]] = []

    for index, column in enumerate(columns):
        if column not in attributes:
            continue

        _set_bit(bitmap=presence, index=index)

        value: Value = attributes[column]

        try:
            typed += encode_value(raw=value.get(), value_type=types[column])

        except (TypeError, ValueError, KeyError, struct.error):
            # Exotic payloads (external objects, mismatched values) fall
            # back to the JSON codec for this column on this row
            _set_bit(bitmap=escaped, index=index)

            escaped_payloads.append((index, _json_value(value=value)))

    out: bytearray = bytearray(presence)
    out += escaped
    out += typed
    out += encode_varint(value=len(escaped_payloads))

    for index, json_bytes in escaped_payloads:
        out += encode_varint(value=index)
        out += encode_varint(value=len(json_bytes))
        out += json_bytes

    return bytes(out)


def decode_row(
    payload: bytes,
    offset: int,
    columns: list[str],
    types: dict[str, Type],
    resolver: Callable[[int], Any],
) -> tuple[dict[str, Value], int]:
    """
    Decodes one row payload into attribute Values.

    Args:
        payload (bytes): Batch payload containing the row.
        offset (int): Offset of the row's first byte.
        columns (list[str]): Batch column names in order.
        types (dict[str, Type]): Declared schema type per column.
        resolver (Callable[[int], Any]): Resolves entity ids to instances.

    Returns:
        tuple[dict[str, Value], int]: The decoded attributes and the offset
            after the row.
    """

    from nadata.core.values.values import make_value

    bitmap_size: int = _bitmaps_size(column_count=len(columns))

    presence: bytes = payload[offset:offset + bitmap_size]
    escaped: bytes = payload[offset + bitmap_size:offset + 2 * bitmap_size]

    cursor: int = offset + 2 * bitmap_size
    attributes: dict[str, Value] = {}

    for index, column in enumerate(columns):
        if not _get_bit(bitmap=presence, index=index):
            continue

        if _get_bit(bitmap=escaped, index=index):
            continue

        raw: Any
        raw, cursor = decode_value(
            payload=payload,
            offset=cursor,
            value_type=types[column],
            resolver=resolver,
        )

        # Struct-typed attributes hold the instance in an opaque wrapper
        if _is_struct_type(value_type=types[column]):
            from nadata.core.values.values import ValueExternal

            attributes[column] = ValueExternal(payload=raw)

        else:
            attributes[column] = make_value(value_type=types[column], raw=raw)

    escaped_count: int
    escaped_count, cursor = decode_varint(payload=payload, offset=cursor)

    codec: ObjectCodec = ObjectCodec()

    for _ in range(escaped_count):
        column_index: int
        column_index, cursor = decode_varint(payload=payload, offset=cursor)

        json_len: int
        json_len, cursor = decode_varint(payload=payload, offset=cursor)

        encoded: Any = json.loads(
            payload[cursor:cursor + json_len].decode("utf-8")
        )

        cursor += json_len

        try:
            value: Value = codec.decode(
                data=encoded, context=_EscapeContext(resolver=resolver)
            )

        except (TypeError, ValueError, KeyError):
            # The payload violated its own type on the way in; keep the
            # raw JSON structure rather than losing the row
            from nadata.core.values.values import ValueExternal

            value = ValueExternal(payload=encoded)

        attributes[columns[column_index]] = _resolve_value_refs(
            value=value, resolver=resolver
        )

    return attributes, cursor


def _json_value(value: Value) -> bytes:
    """
    Encodes a Value through the JSON codec for the escape hatch.

    Args:
        value (Value): Value to encode.

    Returns:
        bytes: Compact UTF-8 JSON bytes.
    """

    codec: ObjectCodec = ObjectCodec()

    return json.dumps(obj=codec.encode(obj=value), separators=(",", ":")).encode("utf-8")


class _EscapeContext(DecodeContext):
    """
    Decode context for escaped values; resolves instance refs through
    the row resolver instead of a materialized registry.

    Note: struct type lookups ('__struct_type__') are not available in
    this context - escaped values referencing struct types decode only
    if they do not need type resolution.

    Attributes:
        resolver (Callable[[int], Any]): Entity id resolver.
    """

    def __init__(self, resolver: Callable[[int], Any]) -> None:
        """
        Initializes the escape context.

        Args:
            resolver (Callable[[int], Any]): Entity id resolver.
        """

        super().__init__()

        self.resolver: Callable[[int], Any] = resolver
        self.instances = _ResolverMap(resolver=resolver)


class _ResolverMap(dict):  # type: ignore[type-arg]
    """
    Dict-like view resolving every key through a resolver callable.

    Used with resolve_instance_refs so escaped JSON values referencing
    pending rows decode on demand.

    Attributes:
        resolver (Callable[[int], Any]): Entity id resolver.
    """

    def __init__(self, resolver: Callable[[int], Any]) -> None:
        """
        Initializes the resolver map.

        Args:
            resolver (Callable[[int], Any]): Entity id resolver.
        """

        super().__init__()

        self.resolver: Callable[[int], Any] = resolver

    def __getitem__(self, key: int) -> Any:
        """
        Resolves an entity id.

        Args:
            key (int): Entity id.

        Returns:
            Any: The resolved StructInstance.
        """

        return self.resolver(key)

    def __contains__(self, key: object) -> bool:
        """
        Reports every key as present.

        Args:
            key (object): Candidate key.

        Returns:
            bool: Always True.
        """

        return True


def _resolve_value_refs(value: Value, resolver: Callable[[int], Any]) -> Value:
    """
    Resolves instance reference placeholders inside a decoded Value.

    Args:
        value (Value): Decoded value.
        resolver (Callable[[int], Any]): Entity id resolver.

    Returns:
        Value: Value with references resolved.
    """

    resolved: Value = resolve_instance_refs(
        value=value, instances=_ResolverMap(resolver=resolver)
    )

    return resolved


def encode_batch_document(
    struct_name: str,
    columns: list[str],
    rows: list[tuple[int, bytes]],
    max_local_id: int,
) -> bytes:
    """
    Builds one binary batch document for a struct table.

    Args:
        struct_name (str): Name of the struct the rows belong to.
        columns (list[str]): Column names in order.
        rows (list[tuple[int, bytes]]): (entity id, encoded row) pairs.
        max_local_id (int): Highest local id in the batch (for lazy
            next_local_id restoration without decoding rows).

    Returns:
        bytes: The complete batch document payload.
    """

    header: dict[str, Any] = {
        "kind": _BATCH_KIND,
        "format": _BATCH_FORMAT,
        "struct_name": struct_name,
        "columns": columns,
        "row_count": len(rows),
        "max_local_id": max_local_id,
    }

    header_bytes: bytes = json.dumps(obj=header, separators=(",", ":")).encode("utf-8")

    out: bytearray = bytearray(encode_varint(value=len(header_bytes)))
    out += header_bytes

    for entity_id, row_payload in rows:
        out += encode_varint(value=entity_id)
        out += encode_varint(value=len(row_payload))
        out += row_payload

    return bytes(out)


def decode_batch_document(
    payload: bytes,
) -> tuple[str, list[str], int, list[tuple[int, bytes]]]:
    """
    Parses a binary batch document into its header and rows.

    Args:
        payload (bytes): The batch document payload.

    Returns:
        tuple[str, list[str], int, list[tuple[int, bytes]]]: Struct name,
            column names, max local id, and (entity id, row payload) pairs.
    """

    header_len: int
    header_len, offset = decode_varint(payload=payload, offset=0)

    header: dict[str, Any] = json.loads(
        payload[offset:offset + header_len].decode("utf-8")
    )

    offset += header_len

    rows: list[tuple[int, bytes]] = []

    for _ in range(header["row_count"]):
        entity_id: int
        entity_id, offset = decode_varint(payload=payload, offset=offset)

        row_len: int
        row_len, offset = decode_varint(payload=payload, offset=offset)

        rows.append((entity_id, payload[offset:offset + row_len]))

        offset += row_len

    return header["struct_name"], header["columns"], header["max_local_id"], rows


def decode_batch_row_spans(
    payload: bytes,
) -> tuple[str, list[str], int, list[tuple[int, int, int]]]:
    """
    Parses a binary batch document into its header and row locations.

    Like decode_batch_document, but rows are returned as (entity id,
    offset, length) spans into the payload instead of copied slices, so
    callers can hold locations rather than bytes.

    Args:
        payload (bytes): The batch document payload.

    Returns:
        tuple[str, list[str], int, list[tuple[int, int, int]]]: Struct
            name, column names, max local id, and (entity id, offset,
            length) spans.
    """

    header_len: int
    header_len, offset = decode_varint(payload=payload, offset=0)

    header: dict[str, Any] = json.loads(
        payload[offset:offset + header_len].decode("utf-8")
    )

    offset += header_len

    spans: list[tuple[int, int, int]] = []

    for _ in range(header["row_count"]):

        entity_id: int
        entity_id, offset = decode_varint(payload=payload, offset=offset)

        row_len: int
        row_len, offset = decode_varint(payload=payload, offset=offset)

        spans.append((entity_id, offset, row_len))

        offset += row_len

    return header["struct_name"], header["columns"], header["max_local_id"], spans
