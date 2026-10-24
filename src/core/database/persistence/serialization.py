"""
Generic JSON based codec for Nadata runtime objects.

Encodes and decodes expression / statement AST nodes, schema types, runtime
values, and home type objects into JSON compatible structures, so they can be
stored as records in slotted pages and reconstructed on load.

Limitations: plain dicts encoded through the generic path must not use the
reserved marker keys ('__class__', '__ref__', '__struct_type__') as keys.
"""

# Import Modules
from __future__ import annotations
import inspect
from collections import deque
from typing import Any, Optional, cast

from src.core.database.entity_id import EntityId
from src.core.database.instance import StructInstance
from src.core.expressions.expression import (
    ArrayLiteralExpr,
    BinaryOpExpr,
    CallExpr,
    DictLiteralExpr,
    Expression,
    IdentifierExpr,
    IndexExpr,
    LambdaExpr,
    LiteralExpr,
    MemberAccessExpr,
    SetLiteralExpr,
    UnaryOpExpr,
)
from src.core.expressions.statement import (
    AssignStmt,
    BlockStmt,
    BreakStmt,
    ContinueStmt,
    ExprStmt,
    ForInStmt,
    ForStmt,
    IfStmt,
    ReturnStmt,
    TargetAssignStmt,
    VarDeclStmt,
    WhileStmt,
)
from src.core.struct.struct import Struct
from src.core.struct.struct_method import StructMethod
from src.core.types.home_types.color import Color
from src.core.types.home_types.coordinate import Coordinate
from src.core.types.home_types.datetime import DateTime
from src.core.types.home_types.geometry import Polygon, Polyline
from src.core.types.home_types.planet import Planet
from src.core.types.types import (
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
    TypeTensor,
    TypeStr,
)
from src.core.values.values import (
    TypeExternal,
    Value,
    ValueArray,
    ValueBool,
    ValueColor,
    ValueCoordinate,
    ValueDateTime,
    ValueDict,
    ValueExternal,
    ValueFloat,
    ValueInt,
    ValuePlanet,
    ValuePolygon,
    ValuePolyline,
    ValuePriorityQueue,
    ValueQueue,
    ValueSet,
    ValueSpatialGrid,
    ValueStr,
    ValueTensor,
)


# Classes reconstructable through the generic __dict__ based codec path
DECODABLE_CLASSES: dict[str, type] = {
    "ArrayLiteralExpr": ArrayLiteralExpr,
    "AssignStmt": AssignStmt,
    "BinaryOpExpr": BinaryOpExpr,
    "BlockStmt": BlockStmt,
    "BreakStmt": BreakStmt,
    "CallExpr": CallExpr,
    "ContinueStmt": ContinueStmt,
    "Color": Color,
    "Coordinate": Coordinate,
    "DateTime": DateTime,
    "Planet": Planet,
    "Polygon": Polygon,
    "Polyline": Polyline,
    "DictLiteralExpr": DictLiteralExpr,
    "EntityId": EntityId,
    "ExprStmt": ExprStmt,
    "ForInStmt": ForInStmt,
    "ForStmt": ForStmt,
    "IdentifierExpr": IdentifierExpr,
    "IfStmt": IfStmt,
    "IndexExpr": IndexExpr,
    "LambdaExpr": LambdaExpr,
    "LiteralExpr": LiteralExpr,
    "MemberAccessExpr": MemberAccessExpr,
    "ReturnStmt": ReturnStmt,
    "SetLiteralExpr": SetLiteralExpr,
    "StructMethod": StructMethod,
    "TypeArray": TypeArray,
    "TypeBool": TypeBool,
    "TypeColor": TypeColor,
    "TypeCoordinate": TypeCoordinate,
    "TypeDateTime": TypeDateTime,
    "TypePlanet": TypePlanet,
    "TypePolygon": TypePolygon,
    "TypePolyline": TypePolyline,
    "TypeDict": TypeDict,
    "TypeExternal": TypeExternal,
    "TypeFloat": TypeFloat,
    "TypeInt": TypeInt,
    "TypePriorityQueue": TypePriorityQueue,
    "TypeQueue": TypeQueue,
    "TypeSet": TypeSet,
    "TypeSpatialGrid": TypeSpatialGrid,
    "TypeTensor": TypeTensor,
    "TypeStr": TypeStr,
    "TargetAssignStmt": TargetAssignStmt,
    "UnaryOpExpr": UnaryOpExpr,
    "ValueArray": ValueArray,
    "ValueBool": ValueBool,
    "ValueColor": ValueColor,
    "ValueCoordinate": ValueCoordinate,
    "ValueDateTime": ValueDateTime,
    "ValueDict": ValueDict,
    "ValueExternal": ValueExternal,
    "ValueFloat": ValueFloat,
    "ValueInt": ValueInt,
    "ValuePlanet": ValuePlanet,
    "ValuePolygon": ValuePolygon,
    "ValuePolyline": ValuePolyline,
    "ValuePriorityQueue": ValuePriorityQueue,
    "ValueQueue": ValueQueue,
    "ValueSet": ValueSet,
    "ValueSpatialGrid": ValueSpatialGrid,
    "ValueTensor": ValueTensor,
    "ValueStr": ValueStr,
    "VarDeclStmt": VarDeclStmt,
    "WhileStmt": WhileStmt,
}

# Constructor parameter cache per class (excludes self and var args)
_CTOR_PARAMS_CACHE: dict[type, list[str]] = {}

# Maps DateTime attributes to their constructor parameter names
_DATETIME_ATTR_TO_PARAM: dict[str, str] = {
    "year": "y",
    "month": "m",
    "day": "d",
    "hour": "H",
    "minute": "M",
    "second": "S",
    "millisecond": "MS",
}


class InstanceRef:
    """
    Placeholder for a struct instance reference encountered during decoding.

    Replaced by the actual StructInstance once all instances are loaded.

    Attributes:
        entity_id (int): Raw 64-bit entity handle of the referenced instance.
    """

    def __init__(self, entity_id: int) -> None:
        """
        Initializes the InstanceRef placeholder.

        Args:
            entity_id (int): Raw 64-bit entity handle of the referenced instance.
        """

        self.entity_id: int = entity_id

    def __repr__(self) -> str:
        """
        Returns string representation of the placeholder.

        Returns:
            str: Placeholder representation.
        """

        return f"InstanceRef({self.entity_id})"


class DecodeContext:
    """
    Registries required to resolve references while decoding.

    Attributes:
        structs (dict[str, Struct]): Loaded structs by name.
        instances (dict[int, StructInstance]): Loaded instances by entity id.
    """

    def __init__(self) -> None:
        """
        Initializes the DecodeContext with empty registries.
        """

        self.structs: dict[str, Struct] = {}
        self.instances: dict[int, StructInstance] = {}


def _ctor_param_names(cls: type) -> list[str]:
    """
    Returns the constructor parameter names of a class (cached).

    Args:
        cls (type): Class to inspect.

    Returns:
        list[str]: Constructor parameter names excluding self and var args.
    """

    if cls in _CTOR_PARAMS_CACHE:
        return _CTOR_PARAMS_CACHE[cls]

    names: list[str] = []

    for name, param in inspect.signature(cls).parameters.items():
        if name == "self":
            continue

        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue

        names.append(name)

    _CTOR_PARAMS_CACHE[cls] = names

    return names


class ObjectCodec:
    """
    Encodes and decodes runtime objects between Python and JSON structures.
    """

    def encode(self, obj: Any) -> Any:
        """
        Encodes a runtime object into a JSON compatible structure.

        Args:
            obj (Any): Object to encode.

        Returns:
            Any: JSON compatible encoded structure.

        Raises:
            TypeError: If the object type is not encodable.
        """

        # JSON scalars pass through (bool before int: bool is an int subclass)
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj

        # Instance payloads encode as entity references
        if isinstance(obj, StructInstance):
            return {"__ref__": obj.entity_id.value}

        # Structs duck type as Type: encode as a named reference
        if isinstance(obj, Struct):
            return {"__struct_type__": obj.name}

        if isinstance(obj, (list, tuple, set, frozenset, deque)):
            return [self.encode(obj=element) for element in obj]

        if isinstance(obj, dict):

            # Tuple keys (spatial grid coordinates) encode as pair lists
            if any(isinstance(key, tuple) for key in obj.keys()):
                return [
                    [list(key), self.encode(obj=value)]
                    for key, value in obj.items()
                ]

            return {key: self.encode(obj=value) for key, value in obj.items()}

        if isinstance(obj, DateTime):
            return self._encode_datetime(datetime_obj=obj)

        if isinstance(obj, ValueDict):
            return self._encode_value_dict(value_dict=obj)

        class_name: str = type(obj).__name__

        if class_name in DECODABLE_CLASSES and DECODABLE_CLASSES[class_name] is not None:
            return {
                "__class__": class_name,
                "fields": self._encode_fields(obj=obj),
            }

        raise TypeError(f"Cannot encode object of type '{class_name}'")

    def decode(self, data: Any, context: DecodeContext) -> Any:
        """
        Decodes a JSON compatible structure back into a runtime object.

        Args:
            data (Any): Encoded structure to decode.
            context (DecodeContext): Registries for reference resolution.

        Returns:
            Any: Decoded runtime object.

        Raises:
            KeyError: If a referenced struct or class is unknown.
        """

        if data is None or isinstance(data, (bool, int, float, str)):
            return data

        if isinstance(data, list):
            return [self.decode(data=element, context=context) for element in data]

        if isinstance(data, dict):

            # Instance reference placeholder
            if "__ref__" in data:
                return InstanceRef(entity_id=data["__ref__"])

            # Struct type reference resolved through the loaded structs
            if "__struct_type__" in data:
                return context.structs[data["__struct_type__"]]

            # Registered class reconstruction
            if "__class__" in data and "fields" in data:
                return self._decode_registered(data=data, context=context)

            # Plain user dict (e.g. call kwargs)
            return {key: self.decode(data=value, context=context) for key, value in data.items()}

        raise TypeError(f"Cannot decode data of type '{type(data).__name__}'")

    # --- Encoding helpers ---

    def _encode_fields(self, obj: Any) -> dict[str, Any]:
        """
        Encodes the constructor parameter fields of a registered object.

        Args:
            obj (Any): Object whose fields are encoded.

        Returns:
            dict[str, Any]: Encoded fields keyed by constructor parameter name.
        """

        fields: dict[str, Any] = {}

        for name in _ctor_param_names(cls=type(obj)):
            if hasattr(obj, name):
                fields[name] = self.encode(obj=getattr(obj, name))

        return fields

    def _encode_datetime(self, datetime_obj: DateTime) -> dict[str, Any]:
        """
        Encodes a DateTime using its constructor parameter names.

        Args:
            datetime_obj (DateTime): DateTime to encode.

        Returns:
            dict[str, Any]: Encoded structure.
        """

        fields: dict[str, Any] = {}

        for attr_name, param_name in _DATETIME_ATTR_TO_PARAM.items():
            fields[param_name] = getattr(datetime_obj, attr_name)

        return {"__class__": "DateTime", "fields": fields}

    def _encode_value_dict(self, value_dict: ValueDict) -> dict[str, Any]:
        """
        Encodes a ValueDict with entries as key / value pairs.

        Entries are encoded as a pair list because JSON dict keys must be
        strings while Nadata dict entries may use any raw key type.

        Args:
            value_dict (ValueDict): ValueDict to encode.

        Returns:
            dict[str, Any]: Encoded structure.
        """

        encoded_entries: list[Any] = [
            [self.encode(obj=key), self.encode(obj=value)]
            for key, value in value_dict.entries.items()
        ]

        return {
            "__class__": "ValueDict",
            "fields": {
                "value_type": self.encode(obj=value_dict.value_type),
                "entries": encoded_entries,
            },
        }

    # --- Decoding helpers ---

    def _decode_registered(self, data: dict[str, Any], context: DecodeContext) -> Any:
        """
        Decodes a registered class instance from its encoded structure.

        Args:
            data (dict[str, Any]): Encoded structure with '__class__' and 'fields'.
            context (DecodeContext): Registries for reference resolution.

        Returns:
            Any: Reconstructed runtime object.

        Raises:
            KeyError: If the class name is unknown.
        """

        class_name: str = data["__class__"]
        fields: dict[str, Any] = data["fields"]

        cls: Optional[type] = DECODABLE_CLASSES.get(class_name)

        if cls is None:
            raise KeyError(f"Unknown encoded class '{class_name}'")

        # ValueDict entries decode from a pair list back into a dict
        if cls is ValueDict:
            return self._decode_value_dict(fields=fields, context=context)

        kwargs: dict[str, Any] = {
            name: self.decode(data=fields[name], context=context)
            for name in _ctor_param_names(cls=cls)
            if name in fields
        }

        return cls(**kwargs)

    def _decode_value_dict(self, fields: dict[str, Any], context: DecodeContext) -> ValueDict:
        """
        Decodes a ValueDict from its pair list entries.

        Args:
            fields (dict[str, Any]): Encoded fields.
            context (DecodeContext): Registries for reference resolution.

        Returns:
            ValueDict: Reconstructed ValueDict.
        """

        entries: dict[Any, Any] = {
            self.decode(data=pair[0], context=context): self.decode(
                data=pair[1], context=context
            )
            for pair in fields["entries"]
        }

        return ValueDict(
            value_type=cast(TypeDict, self.decode(data=fields["value_type"], context=context)),
            entries=entries,
        )


def resolve_instance_refs(
    value: Any,
    instances: dict[int, StructInstance]
) -> Any:
    """
    Replaces instance reference placeholders with the loaded instances.

    Args:
        value (Any): Decoded value potentially containing InstanceRef placeholders.
        instances (dict[int, StructInstance]): Loaded instances by entity id.

    Returns:
        Any: Value with all placeholders resolved.
    """

    if isinstance(value, InstanceRef):
        return instances[value.entity_id]

    if isinstance(value, ValueExternal) and isinstance(value.payload, InstanceRef):
        value.payload = instances[value.payload.entity_id]

        return value

    if isinstance(value, ValueArray):
        value.elements = [_resolve_raw(raw=raw, instances=instances) for raw in value.elements]

        return value

    if isinstance(value, ValueDict):
        value.entries = {
            key: _resolve_raw(raw=raw, instances=instances)
            for key, raw in value.entries.items()
        }

        return value

    if isinstance(value, ValueSet):
        value.elements = {
            _resolve_raw(raw=raw, instances=instances) for raw in value.elements
        }

        return value

    if isinstance(value, ValueQueue):
        value.elements = deque(
            _resolve_raw(raw=raw, instances=instances) for raw in value.elements
        )

        return value

    if isinstance(value, ValuePriorityQueue):
        value.entries = [
            (
                _resolve_raw(raw=priority, instances=instances),
                _resolve_raw(raw=element, instances=instances),
            )
            for priority, element in value.entries
        ]

        return value

    if isinstance(value, ValueSpatialGrid):
        value.cells = {
            coords: _resolve_raw(raw=cell, instances=instances)
            for coords, cell in value.cells.items()
        }

        return value

    if isinstance(value, ValueTensor):
        value.cells = [
            _resolve_raw(raw=cell, instances=instances) for cell in value.cells
        ]

        return value

    return value


def _resolve_raw(raw: Any, instances: dict[int, StructInstance]) -> Any:
    """
    Resolves placeholders inside a raw (non Value) payload.

    Args:
        raw (Any): Raw payload potentially containing InstanceRef placeholders.
        instances (dict[int, StructInstance]): Loaded instances by entity id.

    Returns:
        Any: Payload with placeholders resolved.
    """

    if isinstance(raw, InstanceRef):
        return instances[raw.entity_id]

    if isinstance(raw, list):
        return [_resolve_raw(raw=element, instances=instances) for element in raw]

    if isinstance(raw, dict):
        return {key: _resolve_raw(raw=element, instances=instances) for key, element in raw.items()}

    return raw
