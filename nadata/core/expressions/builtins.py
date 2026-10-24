"""
Built-in runtime members and functions for the Nadata language.

Provides the virtual members usable on instances and struct types
(list_attr, list_methods, getattr, setattr), the type() builtin, and the
database-bound schema_classes() / schema_functions() introspection functions.

Everything here duck-types its targets: struct instances, structs, and
runtime bindings are recognized by their attributes, so this module never
imports the core struct classes (which import expressions).
"""

# Import Modules
from __future__ import annotations
import math
import random
from collections import deque
from typing import Any, Optional, cast

from nadata.core.expressions.autograd import (
    Adam,
    BackwardFunction,
    DetachFunction,
    Optimizer,
    SGD,
    SetLearnableFunction,
    SetTrackableFunction,
    array_to_tensor,
    get_batch_of_partition,
    get_partition_iteration_count,
    tensor_to_array,
)
from nadata.core.expressions.scope import ScopeAwareFunction
from nadata.core.types.home_types.color import Color
from nadata.core.types.home_types.coordinate import (
    Coordinate,
    altitude,
    bearing,
    destination,
    distance,
    geocentric_latitude,
    interpolate,
    surface_radius,
)
from nadata.core.types.home_types.datetime import DateTime
from nadata.core.types.home_types.geometry import (
    Polygon,
    Polyline,
    area,
    circle,
    contains,
    ellipse,
    perimeter,
    regular_polygon,
)
from nadata.core.types.home_types.planet import Planet
from nadata.core.types.types import (
    Float64,
    Int64,
    Type,
    TypeArray,
    TypeBool,
    TypeDict,
    TypeFloat,
    TypeGeoGrid,
    TypeInt,
    TypeStr,
    TypeTensor,
)
from nadata.core.values.values import (
    TYPE_EXTERNAL,
    Value,
    ValueArray,
    ValueFloat,
    ValueBool,
    ValueDict,
    ValueExternal,
    ValueGeoGrid,
    ValueInt,
    ValuePriorityQueue,
    ValueQueue,
    ValueSet,
    ValueSpatialGrid,
    ValueStr,
    ValueTensor,
    make_value,
)

# Members resolved virtually on instances and struct types
VIRTUAL_MEMBERS: set[str] = {"list_attr", "list_methods", "getattr", "setattr"}


def _is_struct_instance(obj: Any) -> bool:
    """
    Checks whether an object is a struct instance.

    Args:
        obj (Any): Object to inspect.

    Returns:
        bool: True if the object carries an entity id and a struct schema.
    """

    return hasattr(obj, "entity_id") and hasattr(obj, "struct")


def _struct_from_object(obj: Any) -> Optional[Any]:
    """
    Extracts a struct schema from a struct, runtime binding, or instance.

    Args:
        obj (Any): Object to inspect.

    Returns:
        Optional[Any]: The struct schema, or None if the object is none.
    """

    if _is_struct_instance(obj=obj):
        return obj.struct

    # Runtime binding: carries both the struct and its owning database
    if hasattr(obj, "struct") and hasattr(obj, "database"):
        return obj.struct

    # A struct schema itself (duck-typed: attribute registry + methods)
    if hasattr(obj, "get_all_attributes") and hasattr(obj, "methods"):
        return obj

    return None


def _string_of(*args: Value) -> ValueStr:
    """
    Native str(x): converts a scalar value to its string form.

    Args:
        *args (Value): Exactly one value.

    Returns:
        ValueStr: The string form of the value.

    Raises:
        TypeError: On wrong arity.
    """

    if len(args) != 1:
        raise TypeError("str() takes exactly one argument")

    raw: Any = args[0].get()

    return ValueStr(value_type=TypeStr(), value=str(raw))


def _int_of(*args: Value) -> ValueInt:
    """
    Native int(s): parses a string into an Int64 (or passes an int
    through). Strict: malformed input raises ValueError.

    Args:
        *args (Value): Exactly one value.

    Returns:
        ValueInt: The parsed integer.

    Raises:
        TypeError: On wrong arity or a non-string/non-int argument.
        ValueError: On a malformed integer string.
    """

    if len(args) != 1:
        raise TypeError("int() takes exactly one argument")

    raw: Any = args[0].get()

    if isinstance(raw, bool):
        raise TypeError("int() does not accept Bool")

    if isinstance(raw, int):
        return ValueInt(value_type=Int64, value=raw)

    if not isinstance(raw, str):
        raise TypeError(
            f"int() expects a Str or Int, got '{type(raw).__name__}'"
        )

    return ValueInt(value_type=Int64, value=int(raw.strip()))


def _float_of(*args: Value) -> ValueFloat:
    """
    Native float(s): parses a string into a Float64 (or passes a
    number through). Strict: malformed input raises ValueError.

    Args:
        *args (Value): Exactly one value.

    Returns:
        ValueFloat: The parsed float.

    Raises:
        TypeError: On wrong arity or a non-string/non-number argument.
        ValueError: On a malformed float string.
    """

    if len(args) != 1:
        raise TypeError("float() takes exactly one argument")

    raw: Any = args[0].get()

    if isinstance(raw, bool):
        raise TypeError("float() does not accept Bool")

    if isinstance(raw, (int, float)):
        return ValueFloat(value_type=Float64, value=float(raw))

    if not isinstance(raw, str):
        raise TypeError(
            f"float() expects a Str or number, got '{type(raw).__name__}'"
        )

    return ValueFloat(value_type=Float64, value=float(raw.strip()))


def _names_array(names: list[str]) -> ValueArray:
    """
    Wraps a list of names into a typed string array Value.

    Args:
        names (list[str]): Names to wrap in declaration order.

    Returns:
        ValueArray: Array of Str values.
    """

    return ValueArray(
        value_type=TypeArray(value_type=TypeStr()),
        elements=list(names),
    )


def _externals_array(payloads: list[Any]) -> ValueArray:
    """
    Wraps a list of opaque payloads into an external-typed array Value.

    Args:
        payloads (list[Any]): Payloads to wrap.

    Returns:
        ValueArray: Array of external values.
    """

    return ValueArray(
        value_type=TypeArray(value_type=TYPE_EXTERNAL),
        elements=[ValueExternal(payload=payload) for payload in payloads],
    )


def _require_arg_count(args: tuple[Value, ...], count: int, member: str) -> None:
    """
    Validates the argument count of a virtual member call.

    Args:
        args (tuple[Value, ...]): Evaluated call arguments.
        count (int): Expected argument count.
        member (str): Virtual member name, for error messages.

    Raises:
        TypeError: If the argument count does not match.
    """

    if len(args) != count:
        raise TypeError(
            f"Virtual member '{member}' takes exactly {count} argument(s), "
            f"got {len(args)}"
        )


class InstanceVirtualCall:
    """
    Callable bound to a virtual member of a struct instance.

    Attributes:
        instance (Any): Receiver instance.
        member (str): Virtual member name.
    """

    def __init__(self, instance: Any, member: str) -> None:
        """
        Initializes the InstanceVirtualCall instance.

        Args:
            instance (Any): Receiver instance.
            member (str): Virtual member name.
        """

        self.instance: Any = instance
        self.member: str = member

    def __call__(self, *args: Value) -> Optional[Value]:
        """
        Invokes the virtual member.

        Args:
            *args (Value): Evaluated call arguments.

        Returns:
            Optional[Value]: Result Value, or None for setattr.

        Raises:
            AttributeError: If the requested attribute is not in the schema.
            TypeError: On invalid argument counts or kinds.
        """

        struct: Any = self.instance.struct

        if self.member == "list_attr":
            return _names_array(names=list(struct.get_all_attributes().keys()))

        if self.member == "list_methods":
            return _names_array(names=list(struct.get_all_methods().keys()))

        if self.member == "getattr":
            _require_arg_count(args=args, count=1, member=self.member)

            name: str = args[0].get()

            if struct.get_attribute(name=name) is None:
                raise AttributeError(
                    f"'{name}' is not declared in the schema of '{struct.name}'"
                )

            return cast(Value, self.instance.get_attribute(name=name))

        if self.member == "setattr":
            _require_arg_count(args=args, count=2, member=self.member)

            # set_attribute validates the schema and the declared type
            self.instance.set_attribute(name=args[0].get(), value=args[1])

            return None

        raise AttributeError(f"Unknown virtual member '{self.member}'")


class StructVirtualCall:
    """
    Callable bound to a virtual member of a struct type.

    Attributes:
        struct (Any): Struct schema the member is invoked on.
        member (str): Virtual member name.
    """

    def __init__(self, struct: Any, member: str) -> None:
        """
        Initializes the StructVirtualCall instance.

        Args:
            struct (Any): Struct schema the member is invoked on.
            member (str): Virtual member name.
        """

        self.struct: Any = struct
        self.member: str = member

    def __call__(self, *args: Value) -> Optional[Value]:
        """
        Invokes the virtual member on the type.

        Args:
            *args (Value): Evaluated call arguments.

        Returns:
            Optional[Value]: Result Value.

        Raises:
            AttributeError: If the member is unsupported on types.
            TypeError: On invalid argument counts or kinds.
        """

        if self.member == "list_attr":
            return _names_array(names=list(self.struct.get_all_attributes().keys()))

        if self.member == "list_methods":
            return _names_array(names=list(self.struct.get_all_methods().keys()))

        if self.member == "getattr":
            _require_arg_count(args=args, count=1, member=self.member)

            name: str = args[0].get()
            attr_def: Any = self.struct.get_attribute(name=name)

            if attr_def is None:
                raise AttributeError(
                    f"'{name}' is not declared in the schema of '{self.struct.name}'"
                )

            return ValueExternal(payload=attr_def.attr_type)

        raise AttributeError(
            f"Virtual member '{self.member}' is not supported on types"
        )


def resolve_virtual_callable(target_obj: Any, member: str) -> Optional[Any]:
    """
    Resolves a virtual member on an instance or struct type into a callable.

    Args:
        target_obj (Any): Unwrapped member access target.
        member (str): Requested member name.

    Returns:
        Optional[Any]: Callable handling the member, or None if the member
        is not virtual for this target.
    """

    if member not in VIRTUAL_MEMBERS:
        return None

    if _is_struct_instance(obj=target_obj):
        return InstanceVirtualCall(instance=target_obj, member=member)

    struct: Optional[Any] = _struct_from_object(obj=target_obj)

    if struct is not None:
        return StructVirtualCall(struct=struct, member=member)

    return None


def builtin_type(*args: Value) -> Value:
    """
    Built-in type(): returns the schema type of a value, instance, or type.

    Args:
        *args (Value): Exactly one argument: an instance (its struct is
            returned), a plain value (its schema type is returned), or an
            already extracted type (returned unchanged).

    Returns:
        Value: External value wrapping the resolved type.

    Raises:
        TypeError: On invalid argument counts or unsupported payloads.
    """

    _require_arg_count(args=args, count=1, member="type")

    arg: Value = args[0]

    if isinstance(arg, ValueExternal):
        payload: Any = arg.payload

        if _is_struct_instance(obj=payload):
            return ValueExternal(payload=payload.struct)

        if hasattr(payload, "struct") and hasattr(payload, "database"):
            return ValueExternal(payload=payload.struct)

        if hasattr(payload, "check_value"):
            return ValueExternal(payload=payload)

        raise TypeError(f"Cannot determine the type of '{repr(payload)}'")

    return ValueExternal(payload=arg.value_type)


class SchemaClassesFunction:
    """
    Built-in schema_classes(): lists every struct registered in the database.

    Attributes:
        database (Any): Database to introspect.
    """

    def __init__(self, database: Any) -> None:
        """
        Initializes the SchemaClassesFunction instance.

        Args:
            database (Any): Database to introspect.
        """

        self.database: Any = database

    def __call__(self, *args: Value) -> Value:
        """
        Returns every registered struct as an array of types.

        Args:
            *args (Value): Ignored.

        Returns:
            Value: Array of external values wrapping each struct.
        """

        structs: list[Any] = [
            table.struct for table in self.database.tables_by_struct_name.values()
        ]

        return _externals_array(payloads=structs)


class SchemaFunctionsFunction:
    """
    Built-in schema_functions(): lists every function registered in the
    database.

    Attributes:
        database (Any): Database to introspect.
    """

    def __init__(self, database: Any) -> None:
        """
        Initializes the SchemaFunctionsFunction instance.

        Args:
            database (Any): Database to introspect.
        """

        self.database: Any = database

    def __call__(self, *args: Value) -> Value:
        """
        Returns every registered function as an array of externals.

        Args:
            *args (Value): Ignored.

        Returns:
            Value: Array of external values wrapping each function.
        """

        return _externals_array(payloads=list(self.database.functions.values()))


class SetLoopLimitFunction(ScopeAwareFunction):
    """
    Native set_loop_limit(n): raises or lowers the loop fuel limit.

    Updates the shared fuel object of the current execution immediately and
    records the new limit in the database settings, so the next dump or
    incremental save persists it.
    """

    def __init__(self, database: Any) -> None:
        """
        Initializes the SetLoopLimitFunction instance.

        Args:
            database (Any): Database whose settings record the limit.
        """

        self.database: Any = database

    def invoke(
        self,
        scope: Any,
        args: list[Any],
        kwargs: dict[str, Any]
    ) -> None:
        """
        Sets the loop fuel limit for the current execution and the database.

        Args:
            scope (Any): Live execution scope of the call site.
            args (list[Any]): Marshaled positional arguments (the new limit).
            kwargs (dict[str, Any]): Marshaled keyword arguments (ignored).

        Raises:
            TypeError: If the limit is not an integer.
            ValueError: If the limit is below 1.
        """

        if len(args) != 1 or kwargs:
            raise TypeError("set_loop_limit takes exactly one argument")

        limit: Any = args[0]

        if isinstance(limit, Value):
            limit = limit.get()

        if not isinstance(limit, int) or isinstance(limit, bool):
            raise TypeError(f"Loop limit must be an integer, got '{type(limit).__name__}'")

        if limit < 1:
            raise ValueError(f"Loop limit must be at least 1, got {limit}")

        # Immediate effect on the running execution
        scope.fuel.limit = limit

        # Persistent effect: recorded in the database settings
        self.database.settings["loop_limit"] = limit


def _resolve_instance(value: Value) -> Any:
    """
    Unwraps a marshaled script value into a StructInstance.

    Args:
        value (Value): The marshaled argument (usually a ValueExternal).

    Returns:
        Any: The StructInstance payload.

    Raises:
        TypeError: If the value does not wrap a struct instance.
    """

    from nadata.core.database.instance import StructInstance

    payload: Any = value.get()

    if isinstance(payload, Value):
        payload = payload.get()

    if not isinstance(payload, StructInstance):
        raise TypeError(
            "Expected a struct instance, got "
            f"'{type(payload).__name__}'"
        )

    return payload


def coerce_nid_value(value: Value) -> str:
    """
    Coerces a marshaled nid argument into its string form.

    Args:
        value (Value): The marshaled nid value.

    Returns:
        str: The nid string.

    Raises:
        TypeError: If the value does not wrap a string.
    """

    raw: Any = value.get()

    if not isinstance(raw, str):
        raise TypeError(
            f"nid must be a string, got '{type(raw).__name__}'"
        )

    return raw


class GetByNidFunction(ScopeAwareFunction):
    """
    Native get_by_nid(Struct, name): O(1) lookup of a named instance.

    Returns None when the nid is unknown (missing is not an error);
    only instances declared with a nid= argument are visible here.
    """

    def invoke(
        self,
        scope: Any,
        args: list[Value],
        kwargs: dict[str, Value],
    ) -> Any:
        """
        Resolves the nid.

        Args:
            scope (Any): Live execution scope (unused).
            args (list[Value]): The struct binding and the nid.
            kwargs (dict[str, Value]): Keyword arguments (rejected).

        Returns:
            Any: External wrapping the instance, or None payload.

        Raises:
            TypeError: On wrong arity or kinds.
        """

        if len(args) != 2 or kwargs:
            raise TypeError("get_by_nid takes exactly two arguments")

        binding: Any = args[0].get()
        nid: str = coerce_nid_value(value=args[1])

        instance: Any = binding.database.get_by_nid(
            struct_name=binding.struct.name,
            nid=nid,
        )

        if instance is None:
            return None

        return ValueExternal(payload=instance)


class FirstFunction(ScopeAwareFunction):
    """
    Native first(Struct): the first instance of a struct, or None.
    """

    def invoke(
        self,
        scope: Any,
        args: list[Value],
        kwargs: dict[str, Value],
    ) -> Any:
        """
        Returns the first instance.

        Args:
            scope (Any): Live execution scope (unused).
            args (list[Value]): The struct binding.
            kwargs (dict[str, Value]): Keyword arguments (rejected).

        Returns:
            Any: External wrapping the instance, or None payload.

        Raises:
            TypeError: On wrong arity or kinds.
        """

        if len(args) != 1 or kwargs:
            raise TypeError("first takes exactly one argument")

        binding: Any = args[0].get()

        found: Any = binding._base_queryset().first()

        return None if found is None else ValueExternal(payload=found)


class LastFunction(ScopeAwareFunction):
    """
    Native last(Struct): the last instance of a struct, or None.
    """

    def invoke(
        self,
        scope: Any,
        args: list[Value],
        kwargs: dict[str, Value],
    ) -> Any:
        """
        Returns the last instance.

        Args:
            scope (Any): Live execution scope (unused).
            args (list[Value]): The struct binding.
            kwargs (dict[str, Value]): Keyword arguments (rejected).

        Returns:
            Any: External wrapping the instance, or None payload.

        Raises:
            TypeError: On wrong arity or kinds.
        """

        if len(args) != 1 or kwargs:
            raise TypeError("last takes exactly one argument")

        binding: Any = args[0].get()

        found: Any = binding._base_queryset().last()

        return None if found is None else ValueExternal(payload=found)


class SpatialIndexFunction(ScopeAwareFunction):
    """
    Native spatial_index(Struct, attribute, lod, parent): builds a
    session-scoped cell index over a geometry attribute for locate /
    nearest / within queries.
    """

    def invoke(
        self,
        scope: Any,
        args: list[Value],
        kwargs: dict[str, Value],
    ) -> Any:
        """
        Builds the index.

        Args:
            scope (Any): Live execution scope (unused).
            args (list[Value]): The struct binding and attribute name.
            kwargs (dict[str, Value]): Optional lod (Int) and parent
                (Str, the self-reference attribute name).

        Returns:
            Any: External wrapping None (statement-style call).

        Raises:
            TypeError: On wrong arity or kinds.
        """

        if len(args) != 2 or ("lod" not in kwargs and "parent" not in kwargs and kwargs):
            raise TypeError(
                "spatial_index takes (Struct, attribute) with optional "
                "lod and parent keyword arguments"
            )

        unknown = set(kwargs) - {"lod", "parent"}

        if unknown:
            raise TypeError(
                f"spatial_index got unexpected keyword arguments: {sorted(unknown)}"
            )

        binding: Any = args[0].get()
        attribute: Any = args[1].get()

        if not isinstance(attribute, str):
            raise TypeError(
                f"spatial_index attribute must be a string, "
                f"got '{type(attribute).__name__}'"
            )

        lod: int = 8

        if "lod" in kwargs:
            lod_value: Any = kwargs["lod"].get()

            if isinstance(lod_value, bool) or not isinstance(lod_value, int):
                raise TypeError("spatial_index lod must be an integer")

            lod = lod_value

        parent_attribute: Optional[str] = None

        if "parent" in kwargs:
            parent_value: Any = kwargs["parent"].get()

            if not isinstance(parent_value, str):
                raise TypeError("spatial_index parent must be a string")

            parent_attribute = parent_value

        table: Any = binding.database.tables_by_struct_name[
            binding.struct.name
        ]

        table.create_spatial_index(
            attribute=attribute,
            lod=lod,
            parent_attribute=parent_attribute,
            database=binding.database,
        )

        return ValueExternal(payload=None)


class NetworkIndexFunction(ScopeAwareFunction):
    """
    Native network_index(NodeStruct, EdgeStruct): builds a
    session-scoped routing network over a node struct (with a
    Coordinate position) and an edge struct (with from / to
    references, optional modes, per-mode speeds, and a Polyline
    shape).
    """

    def invoke(
        self,
        scope: Any,
        args: list[Value],
        kwargs: dict[str, Value],
    ) -> Any:
        """
        Builds the network.

        Args:
            scope (Any): Live execution scope (unused).
            args (list[Value]): The node struct binding and the edge
                struct binding.
            kwargs (dict[str, Value]): Unused.

        Returns:
            Any: External wrapping None (statement-style call).

        Raises:
            TypeError: On wrong arity or non-binding arguments.
        """

        if len(args) != 2 or kwargs:
            raise TypeError(
                "network_index takes exactly (NodeStruct, EdgeStruct)"
            )

        node_binding: Any = args[0].get()
        edge_binding: Any = args[1].get()

        if not hasattr(node_binding, "database") or not hasattr(
            node_binding, "struct"
        ):
            raise TypeError(
                "network_index expects a struct as its first argument"
            )

        if not hasattr(edge_binding, "database") or not hasattr(
            edge_binding, "struct"
        ):
            raise TypeError(
                "network_index expects a struct as its second argument"
            )

        node_table: Any = node_binding.database.tables_by_struct_name[
            node_binding.struct.name
        ]
        edge_table: Any = edge_binding.database.tables_by_struct_name[
            edge_binding.struct.name
        ]

        node_table.create_network(edges_table=edge_table)

        return ValueExternal(payload=None)


class RouteFunction(ScopeAwareFunction):
    """
    Native route(from, to, mode, cost): routes between two coordinates
    over the session's network. Bound to the database at globals
    build time.
    """

    def __init__(self, database: Any) -> None:
        """
        Initializes the RouteFunction instance.

        Args:
            database (Any): The database whose tables host networks.
        """

        self.database: Any = database

    def invoke(
        self,
        scope: Any,
        args: list[Value],
        kwargs: dict[str, Value],
    ) -> Any:
        """
        Routes between two coordinates.

        Args:
            scope (Any): Live execution scope (unused).
            args (list[Value]): Start coordinate, end coordinate,
                transport mode.
            kwargs (dict[str, Value]): Optional cost ("distance" or
                "time").

        Returns:
            Any: External wrapping the Path result.

        Raises:
            TypeError: On wrong arity or kinds.
            ValueError: If no (or several) networks exist, or no path
                is found.
        """

        if len(args) != 3 or set(kwargs) - {"cost"}:
            raise TypeError(
                "route takes (from: Coordinate, to: Coordinate, mode: Str) "
                "with an optional cost keyword"
            )

        from nadata.core.database.network_index import NetworkIndex
        from nadata.core.types.home_types.coordinate import Coordinate

        start: Any = args[0].get()
        end: Any = args[1].get()
        mode: Any = args[2].get()

        if not isinstance(start, Coordinate) or not isinstance(end, Coordinate):
            raise TypeError("route expects Coordinate endpoints")

        if not isinstance(mode, str):
            raise TypeError("route mode must be a string")

        cost: str = "distance"

        if "cost" in kwargs:
            cost_value: Any = kwargs["cost"].get()

            if not isinstance(cost_value, str):
                raise TypeError("route cost must be a string")

            cost = cost_value

        # Gather the session networks across all tables (a network is
        # registered on both its node and edge tables; dedupe by
        # identity)
        networks: list[Any] = []
        seen: list[int] = []

        for table in self.database.tables_by_struct_name.values():
            for candidate in getattr(table, "networks", []):
                if id(candidate) not in seen:
                    seen.append(id(candidate))
                    networks.append(candidate)

        if not networks:
            raise ValueError(
                "route requires a network (build one with network_index)"
            )

        if len(networks) > 1:
            raise ValueError(
                "route is ambiguous: several networks exist in this session"
            )

        network: NetworkIndex = networks[0]

        return ValueExternal(
            payload=network.route(start=start, end=end, mode=mode, cost=cost)
        )


class LocateFunction(ScopeAwareFunction):
    """
    Native locate(Struct, coordinate): the deepest instance whose
    indexed geometry contains the coordinate, via hierarchy descent
    with fast paths. Requires a spatial index on the geometry
    attribute; None when nothing contains the coordinate.
    """

    def invoke(
        self,
        scope: Any,
        args: list[Value],
        kwargs: dict[str, Value],
    ) -> Any:
        """
        Resolves the containing instance.

        Args:
            scope (Any): Live execution scope (unused).
            args (list[Value]): The struct binding and the coordinate.
            kwargs (dict[str, Value]): Rejected.

        Returns:
            Any: External wrapping the instance, or None payload.

        Raises:
            TypeError: On wrong arity, kinds, or a missing index.
        """

        if len(args) != 2 or kwargs:
            raise TypeError("locate takes exactly two arguments")

        binding: Any = args[0].get()
        coord: Any = args[1].get()

        if not isinstance(coord, Coordinate):
            raise TypeError(
                f"locate requires a Coordinate, got '{type(coord).__name__}'"
            )

        table: Any = binding.database.tables_by_struct_name[
            binding.struct.name
        ]

        index: Any = _spatial_index_of(table=table, op_name="locate")

        found: Optional[int] = index.locate(coord=coord)

        if found is None:
            return ValueExternal(payload=None)

        return ValueExternal(payload=index.get_instance(key=found))


class NearestFunction(ScopeAwareFunction):
    """
    Native nearest(Struct, coordinate, k): the k indexed instances
    whose cached geometry centers are nearest to the coordinate,
    ranked by chord distance.
    """

    def invoke(
        self,
        scope: Any,
        args: list[Value],
        kwargs: dict[str, Value],
    ) -> Any:
        """
        Resolves the nearest instances.

        Args:
            scope (Any): Live execution scope (unused).
            args (list[Value]): The struct binding, the coordinate,
                and k.
            kwargs (dict[str, Value]): Rejected.

        Returns:
            Any: External wrapping the instance list (for-in friendly).

        Raises:
            TypeError: On wrong arity, kinds, or a missing index.
        """

        if len(args) != 3 or kwargs:
            raise TypeError("nearest takes exactly three arguments")

        binding: Any = args[0].get()
        coord: Any = args[1].get()
        k: Any = args[2].get()

        if not isinstance(coord, Coordinate):
            raise TypeError(
                f"nearest requires a Coordinate, got '{type(coord).__name__}'"
            )

        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError(f"nearest requires an integer k, got '{type(k).__name__}'")

        table: Any = binding.database.tables_by_struct_name[
            binding.struct.name
        ]

        index: Any = _spatial_index_of(table=table, op_name="nearest")

        keys: list[int] = index.nearest(coord=coord, k=k)

        instances: list[Any] = [
            resolved
            for resolved in (
                index.get_instance(key=key) for key in keys
            )
            if resolved is not None
        ]

        return ValueExternal(payload=instances)


class WithinFunction(ScopeAwareFunction):
    """
    Native within(Struct, coordinate, radius): the indexed instances
    whose geometry lies within a radius of the coordinate.
    """

    def invoke(
        self,
        scope: Any,
        args: list[Value],
        kwargs: dict[str, Value],
    ) -> Any:
        """
        Resolves the instances within the radius.

        Args:
            scope (Any): Live execution scope (unused).
            args (list[Value]): The struct binding, the coordinate,
                and the radius in meters.
            kwargs (dict[str, Value]): Rejected.

        Returns:
            Any: External wrapping the instance list (for-in friendly).

        Raises:
            TypeError: On wrong arity, kinds, or a missing index.
        """

        if len(args) != 3 or kwargs:
            raise TypeError("within takes exactly three arguments")

        binding: Any = args[0].get()
        coord: Any = args[1].get()
        radius: Any = args[2].get()

        if not isinstance(coord, Coordinate):
            raise TypeError(
                f"within requires a Coordinate, got '{type(coord).__name__}'"
            )

        if isinstance(radius, bool) or not isinstance(radius, (int, float)):
            raise TypeError(
                f"within requires a numeric radius, got '{type(radius).__name__}'"
            )

        table: Any = binding.database.tables_by_struct_name[
            binding.struct.name
        ]

        index: Any = _spatial_index_of(table=table, op_name="within")

        keys: list[int] = index.within(coord=coord, radius_m=float(radius))

        instances: list[Any] = [
            resolved
            for resolved in (
                index.get_instance(key=key) for key in keys
            )
            if resolved is not None
        ]

        return ValueExternal(payload=instances)


def _spatial_index_of(table: Any, op_name: str) -> Any:
    """
    Resolves the spatial index of a struct table.

    Args:
        table (Any): The StructTable.
        op_name (str): Operation name for error messages.

    Returns:
        Any: The SpatialIndex.

    Raises:
        TypeError: If the table has no spatial index, or several
            (ambiguous without an attribute argument).
    """

    count: int = len(table.spatial_indexes)

    if count == 1:
        return next(iter(table.spatial_indexes.values()))

    if count == 0:
        raise TypeError(
            f"{op_name} requires a spatial index on Struct "
            f"'{table.struct.name}' (build one with spatial_index)"
        )

    raise TypeError(
        f"{op_name} is ambiguous on Struct '{table.struct.name}': "
        f"multiple spatial indexes exist"
    )


class IndexFunction(ScopeAwareFunction):
    """
    Native index(Struct, attribute): build a session-scoped hash index
    over a scalar attribute for O(1) equality lookups via get_by().
    """

    def invoke(
        self,
        scope: Any,
        args: list[Value],
        kwargs: dict[str, Value],
    ) -> Any:
        """
        Builds the index.

        Args:
            scope (Any): Live execution scope (unused).
            args (list[Value]): The struct binding and attribute name.
            kwargs (dict[str, Value]): Keyword arguments (rejected).

        Returns:
            Any: External wrapping None (statement-style call).

        Raises:
            TypeError: On wrong arity or kinds.
        """

        if len(args) != 2 or kwargs:
            raise TypeError("index takes exactly two arguments")

        binding: Any = args[0].get()
        attribute: Any = args[1].get()

        if not isinstance(attribute, str):
            raise TypeError(
                f"index attribute must be a string, "
                f"got '{type(attribute).__name__}'"
            )

        table: Any = binding.database.tables_by_struct_name[
            binding.struct.name
        ]

        table.create_index(attribute=attribute)

        return ValueExternal(payload=None)


class GetByFunction(ScopeAwareFunction):
    """
    Native get_by(Struct, attribute, value): equality lookup of an
    instance. Uses the hash index built by index() when present;
    otherwise falls back to a scan of the struct's own table.

    Returns the first matching instance, or None when nothing matches.
    """

    def invoke(
        self,
        scope: Any,
        args: list[Value],
        kwargs: dict[str, Value],
    ) -> Any:
        """
        Resolves the match.

        Args:
            scope (Any): Live execution scope (unused).
            args (list[Value]): The struct binding, attribute name, and
                raw value.
            kwargs (dict[str, Value]): Keyword arguments (rejected).

        Returns:
            Any: External wrapping the instance, or None payload.

        Raises:
            TypeError: On wrong arity or kinds.
        """

        if len(args) != 3 or kwargs:
            raise TypeError("get_by takes exactly three arguments")

        binding: Any = args[0].get()
        attribute: Any = args[1].get()

        if not isinstance(attribute, str):
            raise TypeError(
                f"get_by attribute must be a string, "
                f"got '{type(attribute).__name__}'"
            )

        raw: Any = args[2].get()

        table: Any = binding.database.tables_by_struct_name[
            binding.struct.name
        ]

        matches: Any = table.lookup_by_index(attribute=attribute, raw=raw)

        if matches is None:

            # No index: fall back to a scan of this table's rows
            matches = [
                instance
                for instance in dict.values(table.instances)
                if attribute in instance.attributes
                and instance.attributes[attribute].get() == raw
            ]

        if not matches:
            return None

        return ValueExternal(payload=matches[0])


class NidsFunction(ScopeAwareFunction):
    """
    Native nids(Struct): the sorted nid strings of a struct's named
    instances.
    """

    def invoke(
        self,
        scope: Any,
        args: list[Value],
        kwargs: dict[str, Value],
    ) -> Any:
        """
        Returns the nid array.

        Args:
            scope (Any): Live execution scope (unused).
            args (list[Value]): The struct binding.
            kwargs (dict[str, Value]): Keyword arguments (rejected).

        Returns:
            Any: Array of Str nids.

        Raises:
            TypeError: On wrong arity or kinds.
        """

        if len(args) != 1 or kwargs:
            raise TypeError("nids takes exactly one argument")

        binding: Any = args[0].get()

        return _names_array(
            names=binding.database.nid_names(
                struct_name=binding.struct.name
            )
        )


class PinFunction(ScopeAwareFunction):
    """
    Native pin(elt): keeps a struct instance resident in the decode
    cache. Pinned instances are never evicted by clear_cache() or the
    buffer pool bound (the model stays hot for free).
    """

    def invoke(
        self,
        scope: Any,
        args: list[Value],
        kwargs: dict[str, Value],
    ) -> None:
        """
        Pins the instance.

        Args:
            scope (Any): Live execution scope (unused).
            args (list[Value]): The instance to pin.
            kwargs (dict[str, Value]): Keyword arguments (rejected).

        Raises:
            TypeError: On wrong arity or kinds.
        """

        if len(args) != 1 or kwargs:
            raise TypeError("pin takes exactly one argument")

        _resolve_instance(value=args[0]).pinned = True

        return None


class UnpinFunction(ScopeAwareFunction):
    """
    Native unpin(elt): releases a pin, making the instance evictable
    again once it is clean.
    """

    def invoke(
        self,
        scope: Any,
        args: list[Value],
        kwargs: dict[str, Value],
    ) -> None:
        """
        Unpins the instance.

        Args:
            scope (Any): Live execution scope (unused).
            args (list[Value]): The instance to unpin.
            kwargs (dict[str, Value]): Keyword arguments (rejected).

        Raises:
            TypeError: On wrong arity or kinds.
        """

        if len(args) != 1 or kwargs:
            raise TypeError("unpin takes exactly one argument")

        _resolve_instance(value=args[0]).pinned = False

        return None


class FlushFunction(ScopeAwareFunction):
    """
    Native flush(): writes all dirty instances to the database file as
    delta rows. Requires the database to be file-backed; returns the
    number of instances written.
    """

    def __init__(self, database: Any) -> None:
        """
        Initializes the FlushFunction instance.

        Args:
            database (Any): Database to write back to.
        """

        self.database: Any = database

    def invoke(
        self,
        scope: Any,
        args: list[Value],
        kwargs: dict[str, Value],
    ) -> Any:
        """
        Flushes dirty instances.

        Args:
            scope (Any): Live execution scope (unused).
            args (list[Value]): Arguments (rejected; flush takes none).
            kwargs (dict[str, Value]): Keyword arguments (rejected).

        Raises:
            TypeError: On any argument.
        """

        if args or kwargs:
            raise TypeError("flush takes no arguments")

        return int(self.database.flush())


class ClearCacheFunction(ScopeAwareFunction):
    """
    Native clear_cache(): evicts all clean, unpinned instances from the
    decode caches. Returns the number of evicted instances.
    """

    def __init__(self, database: Any) -> None:
        """
        Initializes the ClearCacheFunction instance.

        Args:
            database (Any): Database whose caches to clear.
        """

        self.database: Any = database

    def invoke(
        self,
        scope: Any,
        args: list[Value],
        kwargs: dict[str, Value],
    ) -> Any:
        """
        Clears the decode caches.

        Args:
            scope (Any): Live execution scope (unused).
            args (list[Value]): Arguments (rejected; takes none).
            kwargs (dict[str, Value]): Keyword arguments (rejected).

        Raises:
            TypeError: On any argument.
        """

        if args or kwargs:
            raise TypeError("clear_cache takes no arguments")

        return int(self.database.clear_cache())


class RemoveFromDbFunction(ScopeAwareFunction):
    """
    Native remove_from_db(elt): removes an instance from the database
    and appends a tombstone to the file (when file-backed), so the
    removal survives reloads. The entity id is never reused.
    """

    def __init__(self, database: Any) -> None:
        """
        Initializes the RemoveFromDbFunction instance.

        Args:
            database (Any): Database to remove from.
        """

        self.database: Any = database

    def invoke(
        self,
        scope: Any,
        args: list[Value],
        kwargs: dict[str, Value],
    ) -> None:
        """
        Removes the instance and persists the tombstone.

        Args:
            scope (Any): Live execution scope (unused).
            args (list[Value]): The instance to remove.
            kwargs (dict[str, Value]): Keyword arguments (rejected).

        Raises:
            TypeError: On wrong arity or kinds.
        """

        if len(args) != 1 or kwargs:
            raise TypeError("remove_from_db takes exactly one argument")

        instance: Any = _resolve_instance(value=args[0])

        self.database.remove_instance(instance=instance)

        if self.database.storage_path is not None:

            from nadata.core.database.persistence.dumper import (
                append_instance_removal,
            )

            append_instance_removal(
                database=self.database,
                storage_path=self.database.storage_path,
                entity_id=instance.entity_id.value,
            )

        return None


class DictCollectionCallable:
    """
    Callable bound to the keys() or values() helper of a typed dict value.

    Attributes:
        value_dict (ValueDict): Receiver dict value.
        member (str): Helper name ('keys' or 'values').
    """

    def __init__(self, value_dict: ValueDict, member: str) -> None:
        """
        Initializes the DictCollectionCallable instance.

        Args:
            value_dict (ValueDict): Receiver dict value.
            member (str): Helper name.
        """

        self.value_dict: ValueDict = value_dict
        self.member: str = member

    def __call__(self, *args: Value) -> Value:
        """
        Returns the dict keys or values as a typed array.

        Args:
            *args (Value): Must be empty.

        Returns:
            Value: Array of the declared key or value type.

        Raises:
            TypeError: If any argument is passed.
        """

        _require_arg_count(args=args, count=0, member=self.member)

        if self.member == "keys":
            element_type: Any = self.value_dict.value_type.key_type

        else:
            element_type = self.value_dict.value_type.value_type

        elements: list[Any] = (
            list(self.value_dict.entries.keys())
            if self.member == "keys"
            else list(self.value_dict.entries.values())
        )

        return ValueArray(
            value_type=TypeArray(value_type=element_type),
            elements=elements,
        )


class QueueCallable:
    """
    Callable bound to a typed member of a queue or priority queue value.

    Provides push / pop / peek / size with element and priority validation,
    since the raw Python container fallback bypasses typed checks.

    Attributes:
        target (Value): Receiver ValueQueue or ValuePriorityQueue.
        member (str): Requested member name.
    """

    def __init__(self, target: Value, member: str) -> None:
        """
        Initializes the QueueCallable instance.

        Args:
            target (Value): Receiver queue value.
            member (str): Requested member name.
        """

        self.target: Value = target
        self.member: str = member

    def __call__(self, *args: Value) -> Value:
        """
        Executes the requested queue member.

        Args:
            *args (Value): Evaluated call arguments (unwrapped to raw).

        Returns:
            Value: Typed result (ValueBool for size, ValueExternal for
            mutating members).

        Raises:
            TypeError: On an unknown member or invalid argument count.
            IndexError: On pop / peek against an empty queue.
        """

        raw_args: list[Any] = [arg.get() for arg in args]

        # Priority queue members (entries sorted priority ascending)
        if isinstance(self.target, ValuePriorityQueue):
            return self._call_priority(raw_args=raw_args)

        return self._call_fifo(raw_args=raw_args)

    def _call_fifo(self, raw_args: list[Any]) -> Value:
        """
        Executes a FIFO queue member.

        Args:
            raw_args (list[Any]): Unwrapped call arguments.

        Returns:
            Value: Typed result.

        Raises:
            TypeError: On an unknown member or invalid argument count.
        """

        queue: ValueQueue = cast(ValueQueue, self.target)

        if self.member == "push":
            _require_arg_count(args=tuple(raw_args), count=1, member=self.member)

            queue.value_type.value_type.check_value(value=raw_args[0])

            queue.elements.append(raw_args[0])

            return ValueExternal(payload=None)

        if self.member == "pop":
            _require_arg_count(args=tuple(raw_args), count=0, member=self.member)

            return _wrap_raw(raw=queue.elements.popleft(), value_type=queue.value_type.value_type)

        if self.member == "peek":
            _require_arg_count(args=tuple(raw_args), count=0, member=self.member)

            return _wrap_raw(raw=queue.elements[0], value_type=queue.value_type.value_type)

        if self.member == "size":
            _require_arg_count(args=tuple(raw_args), count=0, member=self.member)

            return ValueInt(
                value_type=TypeInt(num_bits=64, signed=True),
                value=len(queue.elements),
            )

        raise TypeError(f"Unknown queue member '{self.member}'")

    def _call_priority(self, raw_args: list[Any]) -> Value:
        """
        Executes a priority queue member.

        Args:
            raw_args (list[Any]): Unwrapped call arguments.

        Returns:
            Value: Typed result.

        Raises:
            TypeError: On an unknown member or invalid argument count.
        """

        queue: ValuePriorityQueue = cast(ValuePriorityQueue, self.target)

        if self.member == "push":
            _require_arg_count(args=tuple(raw_args), count=2, member=self.member)

            queue.push(element=raw_args[0], priority=raw_args[1])

            return ValueExternal(payload=None)

        if self.member == "pop":
            _require_arg_count(args=tuple(raw_args), count=0, member=self.member)

            return _wrap_raw(
                raw=queue.pop(), value_type=queue.value_type.value_type
            )

        if self.member == "peek":
            _require_arg_count(args=tuple(raw_args), count=0, member=self.member)

            return _wrap_raw(
                raw=queue.peek(), value_type=queue.value_type.value_type
            )

        if self.member == "size":
            _require_arg_count(args=tuple(raw_args), count=0, member=self.member)

            return ValueInt(
                value_type=TypeInt(num_bits=64, signed=True),
                value=len(queue.entries),
            )

        raise TypeError(f"Unknown priority queue member '{self.member}'")


def build_type_globals() -> dict[str, Any]:
    """
    Builds the global symbol table used to evaluate expressions at runtime.

    Contains the home type constructors callable from defaults, scripts, and
    queries (e.g. 'DateTime.now()', 'Color(255, 255, 255)'), and the tensor
    initializer factories (zeros(), fill(v), eye()).

    Returns:
        dict[str, Any]: Global symbols keyed by name.
    """

    return {
        "DateTime": DateTime,
        "Color": Color,
        "Coordinate": Coordinate,
        "Polyline": Polyline,
        "Polygon": Polygon,
        "Planet": Planet,
        "area": area,
        "perimeter": perimeter,
        "contains": contains,
        "circle": circle,
        "ellipse": ellipse,
        "regular_polygon": regular_polygon,
        "distance": distance,
        "bearing": bearing,
        "destination": destination,
        "interpolate": interpolate,
        "surface_radius": surface_radius,
        "altitude": altitude,
        "geocentric_latitude": geocentric_latitude,
        "type": builtin_type,
        "zeros": TensorInitializer(kind="zeros"),
        "fill": TensorInitializer(kind="fill"),
        "eye": TensorInitializer(kind="eye"),
        "randn": TensorInitializer(kind="randn"),
        "arange": ArangeFunction(),
        "backward": BackwardFunction(),
        "set_learnable": SetLearnableFunction(),
        "set_trackable": SetTrackableFunction(),
        "detach": DetachFunction(),
        "pin": PinFunction(),
        "unpin": UnpinFunction(),
        "get_by_nid": GetByNidFunction(),
        "index": IndexFunction(),
        "spatial_index": SpatialIndexFunction(),
        "register_sea_level": RegisterSeaLevelFunction(),
        "clear_sea_level": ClearSeaLevelFunction(),
        "network_index": NetworkIndexFunction(),
        "locate": LocateFunction(),
        "nearest": NearestFunction(),
        "within": WithinFunction(),
        "get_by": GetByFunction(),
        "first": FirstFunction(),
        "last": LastFunction(),
        "nids": NidsFunction(),
        "str": _string_of,
        "int": _int_of,
        "float": _float_of,
        "array_to_tensor": array_to_tensor,
        "tensor_to_array": tensor_to_array,
        "get_batch_of_partition": get_batch_of_partition,
        "get_partition_iteration_count": get_partition_iteration_count,
        "SGD": SGD,
        "Adam": Adam,
        "abs": _math_unary("abs", math.fabs),
        "exp": _math_unary("exp", math.exp),
        "sqrt": _math_unary("sqrt", math.sqrt),
        "log": _math_unary("log", math.log),
        "sin": _math_unary("sin", math.sin),
        "cos": _math_unary("cos", math.cos),
        "pow": _math_binary("pow", lambda a, b: a ** b),
    }


class LengthCallable:
    """
    Callable bound to the `length()` member of arrays and strings.

    Containers with typed members expose size(); arrays and strings have
    no typed fallback, so length() is dispatched here before the raw
    Python attribute lookup (which finds nothing).

    Attributes:
        target (Any): Receiver value (ValueArray or a string wrapper).
    """

    def __init__(self, target: Any) -> None:
        """
        Initializes the LengthCallable instance.

        Args:
            target (Any): Receiver value.
        """

        self.target: Any = target

    def __call__(self) -> ValueInt:
        """
        Returns the receiver's element or character count.

        Returns:
            ValueInt: The Int64 length.
        """

        raw: Any = self.target.get() if hasattr(self.target, "get") else self.target

        return ValueInt(
            value_type=TypeInt(num_bits=64, signed=True),
            value=len(raw),
        )


class StringCallable:
    """
    Callable bound to a typed member of a string value.

    Provides the string manipulation library (upper, lower, strip,
    split, join, replace, contains, starts_with, ends_with) with strict
    argument validation, since the raw Python attribute fallback would
    expose untyped str methods.

    Attributes:
        target (ValueStr): Receiver string value.
        member (str): Requested member name.
    """

    # Members taking zero arguments
    _ZERO_ARG = ("upper", "lower", "strip")

    # Members taking exactly one string argument
    _ONE_STR_ARG = ("contains", "starts_with", "ends_with")

    def __init__(self, target: ValueStr, member: str) -> None:
        """
        Initializes the StringCallable instance.

        Args:
            target (ValueStr): Receiver string value.
            member (str): Requested member name.
        """

        self.target: ValueStr = target
        self.member: str = member

    def __call__(self, *args: Value) -> Value:
        """
        Executes the requested string member.

        Args:
            *args (Value): Evaluated call arguments.

        Returns:
            Value: Typed result (ValueStr, ValueBool, or ValueArray of
            Str for split).

        Raises:
            TypeError: On an unknown member, wrong argument count, or a
                non-Str argument where a Str is required.
        """

        raw: str = self.target.get()
        raw_args: list[Any] = [arg.get() for arg in args]

        if self.member in self._ZERO_ARG:
            _require_arg_count(args=tuple(raw_args), count=0, member=self.member)

            result: str = getattr(raw, self.member)()

            return ValueStr(value_type=TypeStr(), value=result)

        if self.member in self._ONE_STR_ARG:
            _require_arg_count(args=tuple(raw_args), count=1, member=self.member)

            if not isinstance(raw_args[0], str):
                raise TypeError(
                    f"'{self.member}' expects a Str argument, "
                    f"got '{type(raw_args[0]).__name__}'"
                )

            if self.member == "contains":
                return ValueBool(
                    value_type=TypeBool(),
                    value=raw_args[0] in raw,
                )

            if self.member == "starts_with":
                return ValueBool(
                    value_type=TypeBool(),
                    value=raw.startswith(raw_args[0]),
                )

            return ValueBool(
                value_type=TypeBool(),
                value=raw.endswith(raw_args[0]),
            )

        if self.member == "split":
            _require_arg_count(args=tuple(raw_args), count=1, member=self.member)

            if not isinstance(raw_args[0], str):
                raise TypeError(
                    f"'split' expects a Str separator, "
                    f"got '{type(raw_args[0]).__name__}'"
                )

            return ValueArray(
                value_type=TypeArray(value_type=TypeStr()),
                elements=raw.split(raw_args[0]),
            )

        if self.member == "join":
            _require_arg_count(args=tuple(raw_args), count=1, member=self.member)

            parts: Any = raw_args[0]

            if not isinstance(parts, list) or not all(
                isinstance(part, str) for part in parts
            ):
                raise TypeError("'join' expects an Array[Str] argument")

            return ValueStr(value_type=TypeStr(), value=raw.join(parts))

        if self.member == "replace":
            _require_arg_count(args=tuple(raw_args), count=2, member=self.member)

            if not all(isinstance(argument, str) for argument in raw_args):
                raise TypeError(
                    "'replace' expects two Str arguments (old, new)"
                )

            return ValueStr(
                value_type=TypeStr(),
                value=raw.replace(raw_args[0], raw_args[1]),
            )

        raise TypeError(f"Unknown string member '{self.member}'")


class SetCallable:
    """
    Callable bound to a typed member of a set value.

    Provides add / discard / size with element validation, since the raw
    Python container fallback bypasses typed checks.

    Attributes:
        target (ValueSet): Receiver set value.
        member (str): Requested member name.
    """

    def __init__(self, target: ValueSet, member: str) -> None:
        """
        Initializes the SetCallable instance.

        Args:
            target (ValueSet): Receiver set value.
            member (str): Requested member name.
        """

        self.target: ValueSet = target
        self.member: str = member

    def __call__(self, *args: Value) -> Value:
        """
        Executes the requested set member.

        Args:
            *args (Value): Evaluated call arguments (unwrapped to raw).

        Returns:
            Value: Typed result (ValueInt for size, ValueExternal for
            mutating members).

        Raises:
            TypeError: On an unknown member or invalid argument count.
        """

        raw_args: list[Any] = [arg.get() for arg in args]

        # add / discard take the element; size takes nothing
        expected: int = 1 if self.member in ("add", "discard") else 0

        _require_arg_count(args=tuple(raw_args), count=expected, member=self.member)

        if self.member == "add":
            self.target.value_type.value_type.check_value(value=raw_args[0])

            self.target.elements.add(raw_args[0])

            return ValueExternal(payload=None)

        if self.member == "discard":
            self.target.elements.discard(raw_args[0])

            return ValueExternal(payload=None)

        if self.member == "size":
            return ValueInt(
                value_type=TypeInt(num_bits=64, signed=True),
                value=len(self.target.elements),
            )

        raise TypeError(f"Unknown set member '{self.member}'")


def _coords_argument(raw_args: list[Any], dims: int, offset: int, member: str) -> tuple[int, ...]:
    """
    Validates and extracts the leading coordinate components of a call.

    Args:
        raw_args (list[Any]): Unwrapped call arguments.
        dims (int): Expected coordinate arity.
        offset (int): Index of the first coordinate argument.
        member (str): Member name, for error messages.

    Returns:
        tuple[int, ...]: The validated coordinate tuple.

    Raises:
        TypeError: On wrong arity or non-integer components.
    """

    if len(raw_args) < offset + dims:
        raise TypeError(
            f"Member '{member}' expects {dims} coordinate component(s)"
        )

    coords: list[int] = []

    for component in raw_args[offset:offset + dims]:
        if isinstance(component, bool) or not isinstance(component, int):
            raise TypeError(
                f"Member '{member}' expects integer coordinates, got "
                f"{type(component).__name__}"
            )

        coords.append(component)

    return tuple(coords)


def _coords_array(coords_list: list[tuple[int, ...]]) -> ValueArray:
    """
    Wraps coordinate tuples into a typed Array[Array[Int64]] value.

    Args:
        coords_list (list[tuple[int, ...]]): Coordinate tuples.

    Returns:
        ValueArray: The typed coordinate array.
    """

    from nadata.core.types.types import Int64, TypeArray

    return ValueArray(
        value_type=TypeArray(value_type=TypeArray(value_type=Int64)),
        elements=[list(coords) for coords in coords_list],
    )


class GeoGridCallable:
    """
    Callable bound to a typed member of a geodetic field layer value.

    Provides set_at / at / has / remove / size / cell_of / sample /
    set_default with coordinate and element validation.

    Attributes:
        target (ValueGeoGrid): Receiver grid value.
        member (str): Requested member name.
    """

    def __init__(self, target: ValueGeoGrid, member: str) -> None:
        """
        Initializes the GeoGridCallable instance.

        Args:
            target (ValueGeoGrid): Receiver grid value.
            member (str): Requested member name.
        """

        self.target: ValueGeoGrid = target
        self.member: str = member

    def __call__(self, *args: Value) -> Value:
        """
        Executes the requested field layer member.

        Args:
            *args (Value): Evaluated call arguments (unwrapped to raw).

        Returns:
            Value: Typed result.

        Raises:
            TypeError: On an unknown member or invalid arguments.
            KeyError: On 'at' or 'sample' against empty data without a
                default.
        """

        raw_args: list[Any] = [arg.get() for arg in args]
        grid: ValueGeoGrid = self.target
        grid_type: TypeGeoGrid = grid.value_type

        def _coord_arg(index: int, member: str) -> Any:
            """
            Validates and returns one Coordinate argument.

            Args:
                index (int): Argument position.
                member (str): Member name for error messages.

            Returns:
                Any: The raw Coordinate.

            Raises:
                TypeError: If the argument is not a Coordinate.
            """

            from nadata.core.types.home_types.coordinate import Coordinate

            if index >= len(raw_args) or not isinstance(
                raw_args[index], Coordinate
            ):
                raise TypeError(
                    f"Member '{member}' expects a Coordinate argument"
                )

            return raw_args[index]

        if self.member == "set_at":
            coord: Any = _coord_arg(index=0, member=self.member)

            if len(raw_args) != 2:
                raise TypeError(
                    "Member 'set_at' expects a Coordinate and one value"
                )

            grid_type.value_type.check_value(value=raw_args[1])

            grid.cells[grid_type.cell_of(lat_deg=coord.lat, lon_deg=coord.lon)] = (
                raw_args[1]
            )

            return ValueExternal(payload=None)

        if self.member == "at":
            coord = _coord_arg(index=0, member=self.member)

            key: tuple[int, int] = grid_type.cell_of(
                lat_deg=coord.lat, lon_deg=coord.lon
            )

            if key not in grid.cells:
                raise KeyError(f"Empty grid cell at {key}")

            return _wrap_raw(
                raw=grid.cells[key], value_type=grid_type.value_type
            )

        if self.member == "has":
            coord = _coord_arg(index=0, member=self.member)

            key = grid_type.cell_of(lat_deg=coord.lat, lon_deg=coord.lon)

            return ValueBool(value_type=TypeBool(), value=key in grid.cells)

        if self.member == "remove":
            coord = _coord_arg(index=0, member=self.member)

            key = grid_type.cell_of(lat_deg=coord.lat, lon_deg=coord.lon)

            grid.cells.pop(key, None)

            return ValueExternal(payload=None)

        if self.member == "size":
            return ValueInt(
                value_type=TypeInt(num_bits=64, signed=True),
                value=len(grid.cells),
            )

        if self.member == "cell_of":
            coord = _coord_arg(index=0, member=self.member)

            key = grid_type.cell_of(lat_deg=coord.lat, lon_deg=coord.lon)

            return ValueArray(
                value_type=TypeArray(value_type=TypeInt(num_bits=32, signed=True)),
                elements=list(key),
            )

        if self.member == "sample":
            coord = _coord_arg(index=0, member=self.member)

            result: Any = grid.sample(lat_deg=coord.lat, lon_deg=coord.lon)

            if result is None:
                raise KeyError(
                    "No data around the sample point and no default set"
                )

            return _wrap_raw(raw=result, value_type=grid_type.value_type)

        if self.member == "set_default":
            if len(raw_args) != 1:
                raise TypeError("Member 'set_default' expects one value")

            grid_type.value_type.check_value(value=raw_args[0])

            grid.default = raw_args[0]
            grid.has_default = True

            return ValueExternal(payload=None)

        raise TypeError(f"Unknown geodetic grid member '{self.member}'")


class RegisterSeaLevelFunction:
    """
    Registers a geodetic field layer as the session sea-level surface.

    Once registered, altitude() references the interpolated sea-surface
    radius where the layer has data; the geometric surface radius
    applies elsewhere.
    """

    def __call__(self, *args: Value) -> Value:
        """
        Registers the layer.

        Args:
            *args (Value): The GeoGrid value to register.

        Returns:
            Value: An empty opaque Value.

        Raises:
            TypeError: If the argument is not a GeoGrid value.
        """

        if len(args) != 1 or not isinstance(args[0], ValueGeoGrid):
            raise TypeError(
                "register_sea_level expects a GeoGrid value"
            )

        from nadata.core.types.home_types.sea_level import (
            GeoSeaLevelAdapter,
            set_sea_level_surface,
        )

        set_sea_level_surface(surface=GeoSeaLevelAdapter(grid=args[0]))

        return ValueExternal(payload=None)


class ClearSeaLevelFunction:
    """
    Unregisters the session sea-level surface.
    """

    def __call__(self, *args: Value) -> Value:
        """
        Unregisters the layer.

        Args:
            *args (Value): Ignored.

        Returns:
            Value: An empty opaque Value.
        """

        from nadata.core.types.home_types.sea_level import set_sea_level_surface

        set_sea_level_surface(surface=None)

        return ValueExternal(payload=None)


class GridCallable:
    """
    Callable bound to a typed member of a sparse spatial grid value.

    Provides at / set_at / has / remove / size / coords / neighbors /
    occupied_neighbors with coordinate and element validation.

    Attributes:
        target (ValueSpatialGrid): Receiver grid value.
        member (str): Requested member name.
    """

    def __init__(self, target: ValueSpatialGrid, member: str) -> None:
        """
        Initializes the GridCallable instance.

        Args:
            target (ValueSpatialGrid): Receiver grid value.
            member (str): Requested member name.
        """

        self.target: ValueSpatialGrid = target
        self.member: str = member

    def __call__(self, *args: Value) -> Value:
        """
        Executes the requested grid member.

        Args:
            *args (Value): Evaluated call arguments (unwrapped to raw).

        Returns:
            Value: Typed result.

        Raises:
            TypeError: On an unknown member or invalid arguments.
            KeyError: On 'at' against an empty cell.
        """

        raw_args: list[Any] = [arg.get() for arg in args]
        grid: ValueSpatialGrid = self.target
        dims: int = grid.value_type.dims

        if self.member == "at":
            coords: tuple[int, ...] = _coords_argument(
                raw_args=raw_args, dims=dims, offset=0, member=self.member
            )

            if coords not in grid.cells:
                raise KeyError(f"Empty grid cell at {coords}")

            return _wrap_raw(raw=grid.cells[coords], value_type=grid.value_type.value_type)

        if self.member == "set_at":
            coords = _coords_argument(
                raw_args=raw_args, dims=dims, offset=0, member=self.member
            )

            if len(raw_args) != dims + 1:
                raise TypeError(
                    f"Member 'set_at' expects {dims} coordinate(s) and one value"
                )

            grid.value_type.value_type.check_value(value=raw_args[dims])

            grid.cells[coords] = raw_args[dims]

            return ValueExternal(payload=None)

        if self.member == "has":
            coords = _coords_argument(
                raw_args=raw_args, dims=dims, offset=0, member=self.member
            )

            return ValueBool(
                value_type=TypeBool(), value=coords in grid.cells
            )

        if self.member == "remove":
            coords = _coords_argument(
                raw_args=raw_args, dims=dims, offset=0, member=self.member
            )

            grid.cells.pop(coords, None)

            return ValueExternal(payload=None)

        if self.member == "size":
            return ValueInt(
                value_type=TypeInt(num_bits=64, signed=True),
                value=len(grid.cells),
            )

        if self.member == "coords":
            return _coords_array(
                coords_list=sorted(grid.cells.keys())
            )

        if self.member == "neighbors":
            coords = _coords_argument(
                raw_args=raw_args, dims=dims, offset=0, member=self.member
            )

            return _coords_array(coords_list=[
                tuple(base + offset for base, offset in zip(coords, offsets))
                for offsets in grid.value_type.neighbor_offsets()
            ])

        if self.member == "occupied_neighbors":
            coords = _coords_argument(
                raw_args=raw_args, dims=dims, offset=0, member=self.member
            )

            return _coords_array(coords_list=[
                neighbor
                for neighbor in (
                    tuple(base + offset for base, offset in zip(coords, offsets))
                    for offsets in grid.value_type.neighbor_offsets()
                )
                if neighbor in grid.cells
            ])

        raise TypeError(f"Unknown grid member '{self.member}'")


class TensorCallable:
    """
    Callable bound to a typed member of a tensor value.

    Provides at / set_at / size / shape with index bounds and shape
    validation.

    Attributes:
        target (ValueTensor): Receiver tensor value.
        member (str): Requested member name.
    """

    def __init__(self, target: ValueTensor, member: str) -> None:
        """
        Initializes the TensorCallable instance.

        Args:
            target (ValueTensor): Receiver tensor value.
            member (str): Requested member name.
        """

        self.target: ValueTensor = target
        self.member: str = member

    def __call__(self, *args: Value) -> Value:
        """
        Executes the requested tensor member.

        Args:
            *args (Value): Evaluated call arguments (unwrapped to raw).

        Returns:
            Value: Typed result.

        Raises:
            TypeError: On an unknown member or invalid arguments.
            IndexError: On out-of-bounds indices.
        """

        raw_args: list[Any] = [arg.get() for arg in args]
        tensor: ValueTensor = self.target
        dims: int = len(tensor.value_type.dims)

        if self.member == "at":
            indices: tuple[int, ...] = _coords_argument(
                raw_args=raw_args, dims=dims, offset=0, member=self.member
            )

            flat: int = tensor.flat_index(indices=indices)

            return _wrap_raw(
                raw=tensor.cells[flat], value_type=tensor.value_type.value_type
            )

        if self.member == "set_at":
            indices = _coords_argument(
                raw_args=raw_args, dims=dims, offset=0, member=self.member
            )

            if len(raw_args) != dims + 1:
                raise TypeError(
                    f"Member 'set_at' expects {dims} index(es) and one value"
                )

            tensor.value_type.value_type.check_value(value=raw_args[dims])

            tensor.cells[tensor.flat_index(indices=indices)] = raw_args[dims]
            tensor.bump_version()

            return ValueExternal(payload=None)

        if self.member == "size":
            return ValueInt(
                value_type=TypeInt(num_bits=64, signed=True),
                value=len(tensor.cells),
            )

        if self.member == "shape":

            # Symbolic shapes report the resolved extents when bound
            if tensor.concrete_dims is not None:
                resolved_dims: list[int] = tensor.concrete_dims

            elif tensor.value_type.is_symbolic:
                raise TypeError(
                    "Cannot report the shape of a tensor with unbound "
                    "symbolic dimensions"
                )

            else:
                resolved_dims = [
                    dim for dim in tensor.value_type.dims if isinstance(dim, int)
                ]

            return ValueArray(
                value_type=TypeArray(value_type=TypeInt(num_bits=32, signed=True)),
                elements=resolved_dims,
            )

        if self.member == "sum":
            return _wrap_raw(
                raw=sum(tensor.cells),
                value_type=tensor.value_type.value_type,
            )

        if self.member == "min":
            return _wrap_raw(
                raw=min(tensor.cells),
                value_type=tensor.value_type.value_type,
            )

        if self.member == "max":
            return _wrap_raw(
                raw=max(tensor.cells),
                value_type=tensor.value_type.value_type,
            )

        if self.member == "fill":

            if len(raw_args) != 1:
                raise TypeError("Member 'fill' expects exactly one value")

            tensor.value_type.value_type.check_value(value=raw_args[0])

            tensor.cells = [raw_args[0]] * len(tensor.cells)
            tensor.bump_version()

            return ValueExternal(payload=None)

        if self.member == "transpose":

            extents: list[int] = _tensor_extents(tensor=tensor)

            if len(extents) != 2:
                raise TypeError(
                    "Member 'transpose' requires a 2-D tensor, got shape "
                    f"{extents}"
                )

            rows: int = extents[0]
            cols: int = extents[1]

            # Non-in-place: the result is a fresh tensor; the receiver
            # keeps its shape and cells untouched
            return ValueTensor(
                value_type=TypeTensor(
                    value_type=tensor.value_type.value_type,
                    dims=[cols, rows],
                ),
                cells=self._permuted_cells(tensor=tensor, perm=[1, 0]),
            )

        if self.member == "permute":
            return self._permute(args=args)

        if self.member == "map":
            return self._map(args=args)

        if self.member == "matmul":
            return self._matmul(args=args)

        if self.member in ("argmax", "argmin"):
            return self._argreduce(member=self.member)

        if self.member == "mean":
            return self._mean()

        if self.member == "concat":
            return self._concat(args=args)

        if self.member == "reshape":
            return self._reshape(args=args)

        if self.member == "flatten":
            return self._flatten()

        if self.member == "gather":
            return self._gather(args=args)

        if self.member == "topk":
            return self._topk(args=args)

        if self.member == "relu":
            return self._relu()

        if self.member == "sigmoid":
            return self._sigmoid()

        if self.member == "gelu":
            return self._gelu()

        if self.member == "softmax":
            return self._softmax()

        if self.member == "exp":
            return self._exp()

        if self.member == "grad":
            return self._grad()

        if self.member in ("to_float", "to_int"):
            return self._to_numeric(member=self.member)
            return self._to_numeric(member=self.member)

        if self.member in ("conv1d", "conv2d"):
            return self._conv(args=args)

        if self.member in ("max_pool", "min_pool", "avg_pool"):
            return self._pool(args=args)

        raise TypeError(f"Unknown tensor member '{self.member}'")

    def _map(self, args: tuple[Value, ...]) -> ValueTensor:
        """
        Applies a function value-wise over every tensor cell.

        The function may be a lambda ('(v: Float32) -> v * 2.0') or a
        named standalone function passed by identifier; struct methods
        are not supported as values.

        Args:
            args (tuple[Value, ...]): The function Value.

        Returns:
            ValueTensor: New tensor with the mapped cells.

        Raises:
            TypeError: On a missing/invalid function or a result
                violating the element type.
        """

        tensor: ValueTensor = self.target

        if len(args) != 1 or not isinstance(args[0], Value):
            raise TypeError("Member 'map' expects exactly one function")

        fn: Any = args[0].get()

        if not callable(fn) and not hasattr(fn, "execute"):
            raise TypeError(
                f"Member 'map' expects a lambda or function, got "
                f"'{repr(fn)}'"
            )

        element_type: Type = tensor.value_type.value_type
        mapped: list[Any] = []

        for cell in tensor.cells:

            if hasattr(fn, "execute"):

                result: Any = fn.execute(
                    self_instance=None,
                    args=[make_value(value_type=element_type, raw=cell)],
                    kwargs={},
                    globals_dict=build_type_globals(),
                )

                if isinstance(result, Value):
                    result = result.get()

            else:
                result = fn(cell)

                if isinstance(result, Value):
                    result = result.get()

            element_type.check_value(value=result)

            mapped.append(result)

        return ValueTensor(
            value_type=TypeTensor(
                value_type=element_type,
                dims=cast("list[int | str]", _tensor_extents(tensor=tensor)),
            ),
            cells=mapped,
        )

    def _matmul(self, args: tuple[Value, ...]) -> Value:
        """
        Computes a matrix product against another tensor.

        Supports 1-D and 2-D operands: dot product (1-D x 1-D -> scalar),
        matrix-vector (2-D x 1-D), vector-matrix (1-D x 2-D), and full
        matrix-matrix (2-D x 2-D). The result keeps the receiver's
        element type.

        Args:
            args (tuple[Value, ...]): The other tensor Value.

        Returns:
            Value: Scalar element for dot products, otherwise a new
                ValueTensor.

        Raises:
            TypeError: On a non-tensor argument, unsupported ranks,
                inner-dimension mismatch, or integer element types
                receiving float results.
        """

        tensor: ValueTensor = self.target

        if not args or not isinstance(args[0], ValueTensor):
            raise TypeError("Member 'matmul' expects a tensor argument")

        other: ValueTensor = args[0]
        a_extents: list[int] = _tensor_extents(tensor=tensor)
        b_extents: list[int] = _tensor_extents(tensor=other)

        if len(a_extents) not in (1, 2) or len(b_extents) not in (1, 2):
            raise TypeError(
                f"Member 'matmul' supports 1-D and 2-D tensors, got "
                f"shapes {a_extents} and {b_extents}"
            )

        if not isinstance(tensor.value_type.value_type, (TypeInt, TypeFloat)):
            raise TypeError(
                f"Member 'matmul' requires a numeric element type, got "
                f"'{repr(tensor.value_type.value_type)}'"
            )

        if not isinstance(other.value_type.value_type, (TypeInt, TypeFloat)):
            raise TypeError(
                f"Member 'matmul' requires a numeric argument element "
                f"type, got '{repr(other.value_type.value_type)}'"
            )

        result_element: Type = tensor.value_type.value_type

        if isinstance(result_element, TypeInt) and isinstance(
            other.value_type.value_type, TypeFloat
        ):
            raise TypeError(
                "Member 'matmul' with a float argument produces floats; "
                "use a float element type"
            )

        inner: int = a_extents[-1]

        if inner != b_extents[0]:
            raise ValueError(
                f"Member 'matmul' inner dimension mismatch: {a_extents} "
                f"vs {b_extents}"
            )

        dot = lambda a_cell, b_cell: a_cell * b_cell  # noqa: E731

        # 1-D x 1-D: scalar dot product
        if len(a_extents) == 1 and len(b_extents) == 1:

            total: Any = 0

            for index in range(inner):
                total = total + tensor.cells[index] * other.cells[index]

            return _wrap_raw(raw=total, value_type=result_element)

        # 2-D x 2-D: full matrix product
        if len(a_extents) == 2 and len(b_extents) == 2:

            rows: int = a_extents[0]
            cols: int = b_extents[1]
            shared: int = a_extents[1]

            cells: list[Any] = []

            for row in range(rows):

                for col in range(cols):

                    total = 0

                    for index in range(shared):
                        total = total + (
                            tensor.cells[row * shared + index]
                            * other.cells[index * cols + col]
                        )

                    cells.append(total)

            return ValueTensor(
                value_type=TypeTensor(
                    value_type=result_element,
                    dims=cast("list[int | str]", [rows, cols]),
                ),
                cells=cells,
            )

        # 2-D x 1-D: matrix-vector product
        if len(a_extents) == 2:

            rows = a_extents[0]
            cells = []

            for row in range(rows):

                total = 0

                for index in range(inner):
                    total = total + (
                        tensor.cells[row * inner + index] * other.cells[index]
                    )

                cells.append(total)

            return ValueTensor(
                value_type=TypeTensor(
                    value_type=result_element,
                    dims=cast("list[int | str]", [rows]),
                ),
                cells=cells,
            )

        # 1-D x 2-D: vector-matrix product
        cols = b_extents[1]
        cells = []

        for col in range(cols):

            total = 0

            for index in range(inner):
                total = total + (
                    tensor.cells[index] * other.cells[index * cols + col]
                )

            cells.append(total)

        return ValueTensor(
            value_type=TypeTensor(
                value_type=result_element,
                dims=cast("list[int | str]", [cols]),
            ),
            cells=cells,
        )

    def _argreduce(self, member: str) -> Value:
        """
        Computes the index of the extreme cell.

        On a 1-D tensor this is a single Int64 index; on a 2-D tensor it
        produces one index per row (Int64 element type).

        Args:
            member (str): 'argmax' or 'argmin'.

        Returns:
            Value: Int64 index, or a Tensor[Int64, rows] for 2-D input.

        Raises:
            TypeError: On unsupported rank or non-numeric elements.
        """

        tensor: ValueTensor = self.target
        extents: list[int] = _tensor_extents(tensor=tensor)

        if len(extents) not in (1, 2):
            raise TypeError(
                f"Member '{member}' requires a 1-D or 2-D tensor, got "
                f"shape {extents}"
            )

        if not isinstance(tensor.value_type.value_type, (TypeInt, TypeFloat)):
            raise TypeError(
                f"Member '{member}' requires a numeric element type"
            )

        index_type: TypeInt = TypeInt(num_bits=64, signed=True)

        def reduce_row(cells: list[Any]) -> int:
            """Returns the extreme index of one flat row."""

            best: int = 0

            for position in range(1, len(cells)):

                if member == "argmax":
                    if cells[position] > cells[best]:
                        best = position

                elif cells[position] < cells[best]:
                    best = position

            return best

        if len(extents) == 1:
            return ValueInt(
                value_type=index_type,
                value=reduce_row(cells=tensor.cells),
            )

        rows: int = extents[0]
        cols: int = extents[1]

        return ValueTensor(
            value_type=TypeTensor(
                value_type=index_type,
                dims=cast("list[int | str]", [rows]),
            ),
            cells=[
                reduce_row(cells=tensor.cells[row * cols:(row + 1) * cols])
                for row in range(rows)
            ],
        )

    def _mean(self) -> Value:
        """
        Computes the average of every cell.

        Returns:
            Value: The mean wrapped in the element type.

        Raises:
            TypeError: On an integer element type (the average is
                fractional).
        """

        tensor: ValueTensor = self.target

        if not isinstance(tensor.value_type.value_type, (TypeInt, TypeFloat)):
            raise TypeError(
                "Member 'mean' requires a numeric element type"
            )

        if isinstance(tensor.value_type.value_type, TypeInt):
            raise TypeError(
                "Member 'mean' requires a float element type (the mean "
                "is fractional)"
            )

        return _wrap_raw(
            raw=sum(tensor.cells) / len(tensor.cells),
            value_type=tensor.value_type.value_type,
        )

    def _concat(self, args: tuple[Value, ...]) -> ValueTensor:
        """
        Joins another tensor onto this one.

        1-D tensors join end to end; 2-D tensors join along axis 0
        (stacking rows) by default, or along axis 1 with an explicit
        second argument. Element types must be compatible.

        Args:
            args (tuple[Value, ...]): The other tensor, then an optional
                axis (2-D only).

        Returns:
            ValueTensor: The joined tensor.

        Raises:
            TypeError: On a non-tensor argument, rank mismatch, invalid
                axis, or incompatible element types.
            ValueError: If the non-joined axes do not match.
        """

        tensor: ValueTensor = self.target

        if not args or not isinstance(args[0], ValueTensor):
            raise TypeError("Member 'concat' expects a tensor argument")

        other: ValueTensor = args[0]
        a_extents: list[int] = _tensor_extents(tensor=tensor)
        b_extents: list[int] = _tensor_extents(tensor=other)

        if len(a_extents) != len(b_extents) or len(a_extents) not in (1, 2):
            raise TypeError(
                f"Member 'concat' requires matching 1-D or 2-D tensors, "
                f"got shapes {a_extents} and {b_extents}"
            )

        if not isinstance(tensor.value_type.value_type, (TypeInt, TypeFloat)):
            raise TypeError(
                "Member 'concat' requires a numeric element type"
            )

        if isinstance(tensor.value_type.value_type, TypeInt) and isinstance(
            other.value_type.value_type, TypeFloat
        ):
            raise TypeError(
                "Member 'concat' cannot join integer elements with "
                "float elements"
            )

        axis: int = 0

        if len(args) > 1:

            raw_axis: Any = args[1].get()

            if isinstance(raw_axis, bool) or not isinstance(raw_axis, int):
                raise TypeError("Member 'concat' axis must be an integer")

            axis = raw_axis

        if len(a_extents) == 1:

            if axis != 0:
                raise TypeError(
                    "Member 'concat' axis must be 0 for 1-D tensors"
                )

            return ValueTensor(
                value_type=TypeTensor(
                    value_type=tensor.value_type.value_type,
                    dims=cast("list[int | str]", [
                        a_extents[0] + b_extents[0]
                    ]),
                ),
                cells=list(tensor.cells) + list(other.cells),
            )

        if axis not in (0, 1):
            raise TypeError("Member 'concat' axis must be 0 or 1")

        if axis == 0:

            if a_extents[1] != b_extents[1]:
                raise ValueError(
                    f"Member 'concat' column mismatch: {a_extents} vs "
                    f"{b_extents}"
                )

            return ValueTensor(
                value_type=TypeTensor(
                    value_type=tensor.value_type.value_type,
                    dims=cast("list[int | str]", [
                        a_extents[0] + b_extents[0], a_extents[1]
                    ]),
                ),
                cells=list(tensor.cells) + list(other.cells),
            )

        if a_extents[0] != b_extents[0]:
            raise ValueError(
                f"Member 'concat' row mismatch: {a_extents} vs {b_extents}"
            )

        cols: int = a_extents[1]
        other_cols: int = b_extents[1]
        joined: list[Any] = []

        for row in range(a_extents[0]):

            joined.extend(
                tensor.cells[row * cols:(row + 1) * cols]
            )
            joined.extend(
                other.cells[row * other_cols:(row + 1) * other_cols]
            )

        return ValueTensor(
            value_type=TypeTensor(
                value_type=tensor.value_type.value_type,
                dims=cast("list[int | str]", [a_extents[0], cols + other_cols]),
            ),
            cells=joined,
        )

    def _reshape(self, args: tuple[Value, ...]) -> ValueTensor:
        """
        Returns a new tensor with the same cells in a new shape.

        Dimensions come from integer arguments (or a single array of
        integers); at most one dimension may be -1, inferred from the
        total cell count.

        Args:
            args (tuple[Value, ...]): Dimension integers, or one array.

        Returns:
            ValueTensor: The reshaped tensor.

        Raises:
            TypeError: On non-integer dimensions.
            ValueError: If the shape does not match the cell count.
        """

        tensor: ValueTensor = self.target
        total: int = len(tensor.cells)
        dims: list[int] = []

        raw_args: list[Any] = [argument.get() for argument in args]

        if len(raw_args) == 1 and isinstance(raw_args[0], list):
            raw_args = raw_args[0]

        if not raw_args:
            raise TypeError("Member 'reshape' expects at least one dimension")

        for raw_dim in raw_args:

            if isinstance(raw_dim, bool) or not isinstance(raw_dim, int):
                raise TypeError(
                    f"Member 'reshape' dimensions must be integers, got "
                    f"'{repr(raw_dim)}'"
                )

            dims.append(raw_dim)

        if dims.count(-1) > 1:
            raise ValueError(
                "Member 'reshape' allows at most one -1 dimension"
            )

        known: int = 1

        for dim in dims:

            if dim != -1:

                if dim < 1:
                    raise ValueError(
                        f"Member 'reshape' dimensions must be positive or "
                        f"-1, got {dim}"
                    )

                known *= dim

        if -1 in dims:

            if known == 0 or total % known != 0:
                raise ValueError(
                    f"Member 'reshape' cannot infer the -1 dimension for "
                    f"{total} cells with shape {dims}"
                )

            dims = [total // known if dim == -1 else dim for dim in dims]

        product: int = 1

        for dim in dims:
            product *= dim

        if product != total:
            raise ValueError(
                f"Member 'reshape' shape {dims} does not match "
                f"{total} cell(s)"
            )

        return ValueTensor(
            value_type=TypeTensor(
                value_type=tensor.value_type.value_type,
                dims=cast("list[int | str]", dims),
            ),
            cells=list(tensor.cells),
        )

    def _permuted_cells(self, tensor: ValueTensor, perm: list[int]) -> list[Any]:
        """
        Builds the flat payload of a permuted copy.

        Args:
            tensor (ValueTensor): Source tensor.
            perm (list[int]): Output axis d reads source axis perm[d].

        Returns:
            list[Any]: Row-major cells of the permuted tensor.
        """

        extents: list[int] = _tensor_extents(tensor=tensor)
        rank: int = len(extents)

        # Strides of the source axes and of the output axes
        src_strides: list[int] = [1] * rank
        out_strides: list[int] = [1] * rank

        running: int = 1

        for axis in range(rank - 1, -1, -1):
            src_strides[axis] = running
            running *= extents[axis]

        out_extents: list[int] = [extents[axis] for axis in perm]

        running = 1

        for axis in range(rank - 1, -1, -1):
            out_strides[axis] = running
            running *= out_extents[axis]

        cells: list[Any] = []

        for out_flat in range(len(tensor.cells)):

            remaining: int = out_flat
            src_flat: int = 0

            for axis in range(rank):

                coordinate: int = remaining // out_strides[axis]
                remaining %= out_strides[axis]

                # Output axis 'axis' addresses source axis perm[axis]
                src_flat += coordinate * src_strides[perm[axis]]

            cells.append(tensor.cells[src_flat])

        return cells

    def _permute(self, args: tuple[Value, ...]) -> ValueTensor:
        """
        Returns a copy with the axes reordered (non-in-place).

        The result of permute(a0, a1, ...) is a fresh tensor whose axis d
        holds the source's axis a_d; the receiver keeps its shape, cells,
        and identity.

        Args:
            args (tuple[Value, ...]): One axis index per source rank.

        Returns:
            ValueTensor: The permuted copy.

        Raises:
            TypeError: On a wrong axis count or non-integer axes.
            ValueError: If the axes are not a permutation of 0..rank-1.
        """

        tensor: ValueTensor = self.target
        extents: list[int] = _tensor_extents(tensor=tensor)
        rank: int = len(extents)

        if len(args) != rank:
            raise TypeError(
                f"Member 'permute' expects {rank} axis indices for a "
                f"{rank}-D tensor, got {len(args)}"
            )

        perm: list[int] = []

        for argument in args:

            if not isinstance(argument, Value):
                raise TypeError("Member 'permute' expects integer axes")

            raw_axis: Any = argument.get()

            if isinstance(raw_axis, bool) or not isinstance(raw_axis, int):
                raise TypeError(
                    f"Member 'permute' expects integer axes, got "
                    f"'{repr(raw_axis)}'"
                )

            perm.append(raw_axis)

        if sorted(perm) != list(range(rank)):
            raise ValueError(
                f"Member 'permute' axes {perm} are not a permutation of "
                f"0..{rank - 1}"
            )

        return ValueTensor(
            value_type=TypeTensor(
                value_type=tensor.value_type.value_type,
                dims=cast("list[int | str]", [
                    extents[axis] for axis in perm
                ]),
            ),
            cells=self._permuted_cells(tensor=tensor, perm=perm),
        )

    def _flatten(self) -> ValueTensor:
        """
        Returns a 1-D copy of the tensor's cells.

        Returns:
            ValueTensor: 1-D tensor of the same elements.
        """

        tensor: ValueTensor = self.target

        return ValueTensor(
            value_type=TypeTensor(
                value_type=tensor.value_type.value_type,
                dims=cast("list[int | str]", [len(tensor.cells)]),
            ),
            cells=list(tensor.cells),
        )

    def _gather(self, args: tuple[Value, ...]) -> ValueTensor:
        """
        Gathers rows of this tensor by integer indices (embedding lookup).

        The indices tensor may be 1-D or 2-D with an integer element
        type; the result prepends the indices shape to this tensor's
        trailing shape.

        Args:
            args (tuple[Value, ...]): The indices tensor Value.

        Returns:
            ValueTensor: Gathered rows.

        Raises:
            TypeError: On a missing or non-integer indices tensor.
            IndexError: If any index is out of range.
        """

        tensor: ValueTensor = self.target
        extents: list[int] = _tensor_extents(tensor=tensor)

        if not args or not isinstance(args[0], Value):
            raise TypeError(
                "Member 'gather' expects an integer indices tensor or array"
            )

        if isinstance(args[0], ValueArray):

            raw_indices: list[Any] = args[0].get()

            if not raw_indices or any(
                isinstance(cell, bool) or not isinstance(cell, int)
                for cell in raw_indices
            ):
                raise TypeError(
                    "Member 'gather' expects an integer indices tensor "
                    "or array"
                )

            indices = ValueTensor(
                value_type=TypeTensor(
                    value_type=TypeInt(num_bits=64, signed=True),
                    dims=cast("list[int | str]", [len(raw_indices)]),
                ),
                cells=list(raw_indices),
            )

        elif (
            isinstance(args[0], ValueTensor)
            and isinstance(args[0].value_type.value_type, TypeInt)
        ):
            indices = args[0]

        else:
            raise TypeError(
                "Member 'gather' expects an integer indices tensor or array"
            )
        index_extents: list[int] = _tensor_extents(tensor=indices)

        if len(index_extents) not in (1, 2):
            raise TypeError(
                "Member 'gather' expects 1-D or 2-D indices"
            )

        rest: int = 1

        for dim in extents[1:]:
            rest *= dim

        cells: list[Any] = []

        for raw_index in indices.cells:

            if isinstance(raw_index, bool) or not isinstance(raw_index, int):
                raise TypeError(
                    f"Member 'gather' indices must be integers, got "
                    f"'{repr(raw_index)}'"
                )

            if not 0 <= raw_index < extents[0]:
                raise IndexError(
                    f"Member 'gather' index {raw_index} out of range for "
                    f"{extents[0]} row(s)"
                )

            cells.extend(
                tensor.cells[raw_index * rest:(raw_index + 1) * rest]
            )

        return ValueTensor(
            value_type=TypeTensor(
                value_type=tensor.value_type.value_type,
                dims=cast("list[int | str]", index_extents + extents[1:]),
            ),
            cells=cells,
        )

    def _topk(self, args: tuple[Value, ...]) -> ValueTensor:
        """
        Returns the indices of the k largest cells, sorted descending.

        Ties keep source order. The result is an Int64 tensor usable
        directly with gather().

        Args:
            args (tuple[Value, ...]): The count k.

        Returns:
            ValueTensor: Int64 tensor of the k selected indices.

        Raises:
            TypeError: On a missing/invalid k.
            ValueError: If k is not within 1..len(cells).
        """

        tensor: ValueTensor = self.target

        if len(_tensor_extents(tensor=tensor)) != 1:
            raise TypeError("Member 'topk' requires a 1-D tensor")

        if (
            not args
            or not isinstance(args[0], Value)
            or isinstance(args[0].get(), bool)
            or not isinstance(args[0].get(), int)
        ):
            raise TypeError("Member 'topk' expects an integer count")

        count: int = args[0].get()

        if count < 1 or count > len(tensor.cells):
            raise ValueError(
                f"Member 'topk' count must be within 1.."
                f"{len(tensor.cells)}, got {count}"
            )

        order: list[int] = sorted(
            range(len(tensor.cells)),
            key=lambda index: tensor.cells[index],
            reverse=True,
        )

        return ValueTensor(
            value_type=TypeTensor(
                value_type=TypeInt(num_bits=64, signed=True),
                dims=cast("list[int | str]", [count]),
            ),
            cells=order[:count],
        )

    def _relu(self) -> ValueTensor:
        """
        Applies rectified linear unit elementwise.

        Returns:
            ValueTensor: Same-shape tensor of max(cell, 0).

        Raises:
            TypeError: On a non-float element type.
        """

        tensor: ValueTensor = self.target

        if not isinstance(tensor.value_type.value_type, TypeFloat):
            raise TypeError("Member 'relu' requires a float element type")

        return ValueTensor(
            value_type=TypeTensor(
                value_type=tensor.value_type.value_type,
                dims=cast("list[int | str]", _tensor_extents(tensor=tensor)),
            ),
            cells=[
                cell if cell > 0 else 0.0 for cell in tensor.cells
            ],
        )

    def _sigmoid(self) -> ValueTensor:
        """
        Applies the logistic sigmoid elementwise.

        Returns:
            ValueTensor: Same-shape tensor of 1 / (1 + e^-x).

        Raises:
            TypeError: On a non-float element type.
        """

        tensor: ValueTensor = self.target

        if not isinstance(tensor.value_type.value_type, TypeFloat):
            raise TypeError("Member 'sigmoid' requires a float element type")

        return ValueTensor(
            value_type=TypeTensor(
                value_type=tensor.value_type.value_type,
                dims=cast("list[int | str]", _tensor_extents(tensor=tensor)),
            ),
            cells=[_sigmoid_cell(cell) for cell in tensor.cells],
        )

    def _gelu(self) -> ValueTensor:
        """
        Applies the tanh-approximated GELU elementwise.

        Returns:
            ValueTensor: Same-shape tensor of the GELU activation.

        Raises:
            TypeError: On a non-float element type.
        """

        tensor: ValueTensor = self.target

        if not isinstance(tensor.value_type.value_type, TypeFloat):
            raise TypeError("Member 'gelu' requires a float element type")

        return ValueTensor(
            value_type=TypeTensor(
                value_type=tensor.value_type.value_type,
                dims=cast("list[int | str]", _tensor_extents(tensor=tensor)),
            ),
            cells=[_gelu_cell(cell) for cell in tensor.cells],
        )

    def _softmax(self) -> ValueTensor:
        """
        Applies a numerically stable softmax over the last axis.

        Returns:
            ValueTensor: Same-shape tensor of per-row probabilities.

        Raises:
            TypeError: On a non-float element type or rank above 2.
        """

        tensor: ValueTensor = self.target
        extents: list[int] = _tensor_extents(tensor=tensor)

        if not isinstance(tensor.value_type.value_type, TypeFloat):
            raise TypeError("Member 'softmax' requires a float element type")

        if len(extents) not in (1, 2):
            raise TypeError(
                f"Member 'softmax' requires a 1-D or 2-D tensor, got "
                f"shape {extents}"
            )

        width: int = extents[-1]
        cells: list[float] = []

        for row in range(len(tensor.cells) // width):

            window: list[float] = tensor.cells[
                row * width:(row + 1) * width
            ]
            peak: float = max(window)
            exps: list[float] = [math.exp(cell - peak) for cell in window]
            denominator: float = sum(exps)

            cells.extend(value / denominator for value in exps)

        return ValueTensor(
            value_type=TypeTensor(
                value_type=tensor.value_type.value_type,
                dims=cast("list[int | str]", extents),
            ),
            cells=cells,
        )

    def _exp(self) -> ValueTensor:
        """
        Raises e to every cell.

        Returns:
            ValueTensor: Same-shape tensor of exp(cell).

        Raises:
            TypeError: On a non-float element type.
        """

        tensor: ValueTensor = self.target

        if not isinstance(tensor.value_type.value_type, TypeFloat):
            raise TypeError("Member 'exp' requires a float element type")

        return ValueTensor(
            value_type=TypeTensor(
                value_type=tensor.value_type.value_type,
                dims=cast("list[int | str]", _tensor_extents(tensor=tensor)),
            ),
            cells=[math.exp(cell) for cell in tensor.cells],
        )

    def _grad(self) -> ValueTensor:
        """
        Returns the accumulated gradient as a same-shape tensor.

        Returns:
            ValueTensor: The deposited gradient, or zeros when none has
                been computed yet.
        """

        tensor: ValueTensor = self.target
        cells: list[float] = (
            list(tensor.grad) if tensor.grad is not None
            else [0.0] * len(tensor.cells)
        )

        return ValueTensor(
            value_type=TypeTensor(
                value_type=TypeFloat(num_bits=32),
                dims=cast("list[int | str]", _tensor_extents(tensor=tensor)),
            ),
            cells=cells,
        )

    def _to_numeric(self, member: str) -> ValueTensor:
        """
        Converts every cell to a different numeric element type.

        Args:
            member (str): 'to_float' or 'to_int'.

        Returns:
            ValueTensor: Converted copy of the tensor.

        Raises:
            TypeError: On a non-numeric element type.
        """

        tensor: ValueTensor = self.target

        if not isinstance(tensor.value_type.value_type, (TypeInt, TypeFloat)):
            raise TypeError(
                f"Member '{member}' requires a numeric element type"
            )

        if member == "to_float":
            target: Type = TypeFloat(num_bits=32)

            def convert(cell: Any) -> Any:
                return float(cell)

        else:
            target = TypeInt(num_bits=64, signed=True)

            def convert(cell: Any) -> Any:
                return int(cell)

        return ValueTensor(
            value_type=TypeTensor(
                value_type=target,
                dims=cast("list[int | str]", _tensor_extents(tensor=tensor)),
            ),
            cells=[convert(cell=cell) for cell in tensor.cells],
        )

    def _conv(self, args: tuple[Value, ...]) -> ValueTensor:
        """
        Computes a valid cross-correlation of the target tensor.

        Args:
            args (list[Value]): Kernel tensor, then an optional stride.

        Returns:
            ValueTensor: The convolved tensor.

        Raises:
            TypeError: On wrong rank, kernel, stride, or element type.
        """

        tensor: ValueTensor = self.target
        extents: list[int] = _tensor_extents(tensor=tensor)
        expected_rank: int = 1 if self.member == "conv1d" else 2

        if len(extents) != expected_rank:
            raise TypeError(
                f"Member '{self.member}' requires a {expected_rank}-D "
                f"tensor, got shape {extents}"
            )

        if not isinstance(tensor.value_type.value_type, (TypeInt, TypeFloat)):
            raise TypeError(
                f"Member '{self.member}' requires a numeric element type"
            )

        if not args or not isinstance(args[0], ValueTensor):
            raise TypeError(
                f"Member '{self.member}' expects a tensor kernel as its "
                f"first argument"
            )

        kernel: ValueTensor = args[0]
        kernel_extents: list[int] = _tensor_extents(tensor=kernel)

        if len(kernel_extents) != expected_rank:
            raise TypeError(
                f"Member '{self.member}' expects a {expected_rank}-D "
                f"kernel, got shape {kernel_extents}"
            )

        if not isinstance(kernel.value_type.value_type, (TypeInt, TypeFloat)):
            raise TypeError(
                f"Member '{self.member}' requires a numeric kernel"
            )

        stride: int = args[1].get() if len(args) > 1 else 1

        if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
            raise TypeError(
                f"Member '{self.member}' stride must be a positive integer"
            )

        axis: int = 0 if expected_rank == 1 else 1

        for length, kernel_length in zip(extents, kernel_extents):

            if kernel_length > length:
                raise TypeError(
                    f"Member '{self.member}': kernel axis {kernel_length} "
                    f"is larger than input axis {length}"
                )

        if expected_rank == 1:

            return ValueTensor(
                value_type=TypeTensor(
                    value_type=tensor.value_type.value_type,
                    dims=[_conv_valid_length(
                        length=extents[0],
                        kernel=kernel_extents[0],
                        stride=stride,
                    )],
                ),
                cells=_conv1d_cells(
                    signal=tensor.cells,
                    kernel=kernel.cells,
                    stride=stride,
                ),
            )

        return ValueTensor(
            value_type=TypeTensor(
                value_type=tensor.value_type.value_type,
                dims=[
                    _conv_valid_length(
                        length=extents[0], kernel=kernel_extents[0], stride=stride
                    ),
                    _conv_valid_length(
                        length=extents[1], kernel=kernel_extents[1], stride=stride
                    ),
                ],
            ),
            cells=_conv2d_cells(
                cells=tensor.cells,
                rows=extents[0],
                cols=extents[1],
                kernel=kernel.cells,
                krows=kernel_extents[0],
                kcols=kernel_extents[1],
                stride=stride,
            ),
        )

    def _pool(self, args: tuple[Value, ...]) -> ValueTensor:
        """
        Pools the target tensor with max/min/average windows.

        Args:
            args (list[Value]): Window size, then an optional stride
                (defaults to the window size).

        Returns:
            ValueTensor: The pooled tensor.

        Raises:
            TypeError: On wrong rank, window, stride, element type, or
                average pooling on an integer element type.
        """

        tensor: ValueTensor = self.target
        extents: list[int] = _tensor_extents(tensor=tensor)
        reducer: str = self.member.split("_", 1)[0]

        if len(extents) not in (1, 2):
            raise TypeError(
                f"Member '{self.member}' requires a 1-D or 2-D tensor, got "
                f"shape {extents}"
            )

        if not isinstance(tensor.value_type.value_type, (TypeInt, TypeFloat)):
            raise TypeError(
                f"Member '{self.member}' requires a numeric element type"
            )

        if reducer == "avg" and isinstance(tensor.value_type.value_type, TypeInt):
            raise TypeError(
                "Member 'avg_pool' requires a float element type (the "
                "average is fractional)"
            )

        if (
            not args
            or not isinstance(args[0], Value)
            or isinstance(args[0].get(), bool)
            or not isinstance(args[0].get(), int)
        ):
            raise TypeError(
                f"Member '{self.member}' expects an integer window size"
            )

        kernel: int = args[0].get()

        if kernel < 1:
            raise TypeError(
                f"Member '{self.member}' window size must be positive"
            )

        stride: int = args[1].get() if len(args) > 1 else kernel

        if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
            raise TypeError(
                f"Member '{self.member}' stride must be a positive integer"
            )

        for length in extents:

            if kernel > length:
                raise TypeError(
                    f"Member '{self.member}': window {kernel} is larger "
                    f"than input axis {length}"
                )

        if len(extents) == 1:

            return ValueTensor(
                value_type=TypeTensor(
                    value_type=tensor.value_type.value_type,
                    dims=[_conv_valid_length(
                        length=extents[0], kernel=kernel, stride=stride
                    )],
                ),
                cells=_pool_cells(
                    cells=tensor.cells,
                    extents=extents,
                    kernel=kernel,
                    stride=stride,
                    reducer=reducer,
                ),
            )

        return ValueTensor(
            value_type=TypeTensor(
                value_type=tensor.value_type.value_type,
                dims=[
                    _conv_valid_length(length=extents[0], kernel=kernel, stride=stride),
                    _conv_valid_length(length=extents[1], kernel=kernel, stride=stride),
                ],
            ),
            cells=_pool_cells(
                cells=tensor.cells,
                extents=extents,
                kernel=kernel,
                stride=stride,
                reducer=reducer,
            ),
        )


class TensorInit(Value):
    """
    Pending tensor initialization request produced by zeros()/fill()/
    eye()/randn().

    The marker carries the requested kind plus the fill value and seed;
    it is materialized against the target's declared tensor type at the
    assignment site. Using it anywhere else fails loudly.

    Attributes:
        kind (str): Initializer kind ('zeros', 'fill', 'eye', or 'randn').
        fill_value (Optional[Any]): Value for 'fill' initializers.
        seed (Optional[int]): Deterministic seed for 'randn'.
        learnable (bool): Whether backward may deposit a gradient.
        trackable (bool): Whether ops consuming the tensor record.
    """

    def __init__(
        self,
        kind: str,
        fill_value: Optional[Any],
        seed: Optional[int] = None,
        learnable: bool = True,
        trackable: bool = True,
    ) -> None:
        """
        Initializes the TensorInit instance.

        Args:
            kind (str): Initializer kind.
            fill_value (Optional[Any]): Value for 'fill' initializers.
            seed (Optional[int]): Deterministic seed for 'randn'.
            learnable (bool, optional): Gradient deposit flag. Defaults
                to True.
            trackable (bool, optional): Autograd recording flag.
                Defaults to True.
        """

        # Placeholder type; the marker is never stored or checked as-is
        super().__init__(value_type=TypeBool())

        self.kind: str = kind
        self.fill_value: Optional[Any] = fill_value
        self.seed: Optional[int] = seed
        self.learnable: bool = learnable
        self.trackable: bool = trackable

    def get(self) -> Any:
        """
        Raises: pending initializers have no raw value.

        Returns:
            Any: Never.

        Raises:
            TypeError: Always; the initializer must be assigned to a
                declared tensor first.
        """

        raise TypeError(
            f"Tensor initializer '{self.kind}()' must be assigned to a "
            f"declared tensor"
        )

    def __repr__(self) -> str:
        """
        Returns the initializer representation.

        Returns:
            str: Call-style representation.
        """

        if self.kind == "fill":
            return f"fill({repr(self.fill_value)})"

        if self.kind == "randn" and self.seed is not None:
            return f"randn({self.seed})"

        return f"{self.kind}()"


class TensorInitializer:
    """
    Callable bound to a tensor initializer name (zeros, fill, eye).

    Evaluates to a pending TensorInit marker; the assignment site
    materializes it against the target's declared tensor type.

    Attributes:
        kind (str): Initializer kind.
    """

    def __init__(self, kind: str) -> None:
        """
        Initializes the TensorInitializer instance.

        Args:
            kind (str): Initializer kind ('zeros', 'fill', or 'eye').
        """

        self.kind: str = kind

    def __call__(self, *values: Value, **kwargs: Value) -> TensorInit:
        """
        Builds the pending initialization request.

        Args:
            *values (Value): Call arguments (one fill value for 'fill',
                one optional seed for 'randn').
            **kwargs (Value): Optional 'learnable' / 'trackable' flags.

        Returns:
            TensorInit: The pending marker.

        Raises:
            TypeError: On wrong arity, a non-Value fill argument, or an
                invalid flag.
        """

        learnable, trackable = _init_flags(kwargs=kwargs)

        if self.kind == "fill":

            if len(values) != 1 or not isinstance(values[0], Value):
                raise TypeError("fill() expects exactly one value")

            return TensorInit(
                kind=self.kind,
                fill_value=values[0].get(),
                learnable=learnable,
                trackable=trackable,
            )

        if self.kind == "randn":

            if len(values) > 1:
                raise TypeError("randn() expects at most one seed argument")

            if values:

                raw_seed: Any = values[0].get()

                if isinstance(raw_seed, bool) or not isinstance(raw_seed, int):
                    raise TypeError("randn() seed must be an integer")

                return TensorInit(
                    kind=self.kind,
                    fill_value=None,
                    seed=raw_seed,
                    learnable=learnable,
                    trackable=trackable,
                )

            return TensorInit(
                kind=self.kind,
                fill_value=None,
                learnable=learnable,
                trackable=trackable,
            )

        if values:
            raise TypeError(f"{self.kind}() takes no arguments")

        return TensorInit(
            kind=self.kind,
            fill_value=None,
            learnable=learnable,
            trackable=trackable,
        )

    def __repr__(self) -> str:
        """
        Returns the initializer representation.

        Returns:
            str: Call-style representation.
        """

        return f"{self.kind}()"


def materialize_tensor_init(
    init: TensorInit,
    tensor_type: TypeTensor,
    extents: Optional[dict[str, int]] = None,
) -> ValueTensor:
    """
    Materializes a pending initializer against a declared tensor type.

    Args:
        init (TensorInit): Pending initialization request.
        tensor_type (TypeTensor): Declared tensor type providing shape and
            element type.
        extents (Optional[dict[str, int]]): Resolved symbolic extents;
            required when the type has symbolic dimensions.

    Returns:
        ValueTensor: The initialized tensor.

    Raises:
        TypeError: On symbolic dims without extents, non-numeric element
            types for zeros()/eye(), or a fill value violating the
            element type.
    """

    dims: list[int | str] = list(tensor_type.dims)

    for dim in dims:

        if isinstance(dim, str) and (extents is None or dim not in extents):
            raise TypeError(
                f"Cannot initialize symbolic tensor with '{init.kind}()': "
                f"dimension '{dim}' is not bound"
            )

    resolved: list[int] = [
        dim if isinstance(dim, int) else extents[dim] for dim in dims  # type: ignore[index]
    ]

    total: int = 1

    for dim in resolved:
        total *= dim

    element_type: Type = tensor_type.value_type

    if not isinstance(element_type, (TypeInt, TypeFloat)):
        raise TypeError(
            f"'{init.kind}()' requires a numeric element type, got "
            f"'{repr(element_type)}'"
        )

    zero_cell: Any = 0.0 if isinstance(element_type, TypeFloat) else 0
    one_cell: Any = 1.0 if isinstance(element_type, TypeFloat) else 1

    if init.kind == "zeros":

        cells: list[Any] = [zero_cell] * total

    elif init.kind == "eye":

        if len(resolved) != 2 or resolved[0] != resolved[1]:
            raise TypeError(
                f"eye() requires a square 2-D tensor, got shape {resolved}"
            )

        size: int = resolved[0]

        cells = [
            one_cell if index // size == index % size else zero_cell
            for index in range(total)
        ]

    elif init.kind == "randn":

        if not isinstance(element_type, TypeFloat):
            raise TypeError(
                "randn() requires a float element type"
            )

        rng: random.Random = (
            random.Random(init.seed) if init.seed is not None else random.Random()
        )

        cells = []

        for _ in range(total):

            # Box-Muller transform, guarded away from log(0)
            uniform_1: float = rng.random() or 1e-12
            uniform_2: float = rng.random()

            cells.append(
                math.sqrt(-2.0 * math.log(uniform_1))
                * math.cos(2.0 * math.pi * uniform_2)
            )

    else:  # 'fill'
        assert init.fill_value is not None

        element_type.check_value(value=init.fill_value)

        cells = [init.fill_value] * total

    result: ValueTensor = ValueTensor(
        value_type=TypeTensor(
            value_type=element_type,
            dims=cast("list[int | str]", resolved),
        ),
        cells=cells,
        tracked=init.trackable,
        learnable=init.learnable,
    )

    # Symbolic shapes stamp their resolved extents for shape()/slicing
    if tensor_type.is_symbolic:
        result.value_type = tensor_type
        result.concrete_dims = resolved

    return result


# Names that produce pending tensor initializations (materialized
# against the target's declared type)
TENSOR_INIT_NAMES: frozenset[str] = frozenset({"zeros", "fill", "eye", "randn"})


def _init_flags(kwargs: dict[str, Value]) -> tuple[bool, bool]:
    """
    Reads the optional learnable/trackable initializer flags.

    Args:
        kwargs (dict[str, Value]): Keyword arguments of the call.

    Returns:
        tuple[bool, bool]: (learnable, trackable).

    Raises:
        TypeError: On unknown or non-boolean flags.
    """

    learnable: bool = True
    trackable: bool = True

    for key, value in kwargs.items():

        raw: Any = value.get()

        if not isinstance(raw, bool):
            raise TypeError(f"Flag '{key}' must be a boolean")

        if key == "learnable":
            learnable = raw

        elif key == "trackable":
            trackable = raw

        else:
            raise TypeError(f"Unknown initializer flag '{key}'")

    return learnable, trackable


class ArangeFunction:
    """
    Builtin producing a 1-D Int64 tensor over an integer range.

    Called as arange(stop), arange(start, stop), or
    arange(start, stop, step) with a positive step; the result shape
    follows the usual half-open range rules.
    """

    def __call__(self, *args: Value) -> ValueTensor:
        """
        Builds the range tensor.

        Args:
            *args (Value): One to three integer Values.

        Returns:
            ValueTensor: 1-D Int64 tensor of the range.

        Raises:
            TypeError: On wrong arity or non-integer arguments.
            ValueError: On a non-positive step or an empty range.
        """

        raw: list[Any] = []

        if not 1 <= len(args) <= 3:
            raise TypeError("arange() expects one to three arguments")

        for argument in args:

            if not isinstance(argument, Value):
                raise TypeError("arange() expects integer arguments")

            value: Any = argument.get()

            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"arange() expects integer arguments, got '{repr(value)}'"
                )

            raw.append(value)

        if len(raw) == 1:
            start, stop, step = 0, raw[0], 1

        elif len(raw) == 2:
            start, stop, step = raw[0], raw[1], 1

        else:
            start, stop, step = raw

        if step < 1:
            raise ValueError(
                f"arange() step must be positive, got {step}"
            )

        count: int = max(0, (stop - start + step - 1) // step)

        if count < 1:
            raise ValueError(
                f"arange() produces an empty range for "
                f"start={start}, stop={stop}, step={step}"
            )

        return ValueTensor(
            value_type=TypeTensor(
                value_type=TypeInt(num_bits=64, signed=True),
                dims=cast("list[int | str]", [count]),
            ),
            cells=list(range(start, stop, step)),
        )


def _math_unary(name: str, op: Any) -> Any:
    """
    Builds a builtin single-float math function over Value arguments.

    Args:
        name (str): Function name (for error messages).
        op (Any): Underlying float -> float operation.

    Returns:
        Any: A callable taking one Value and returning a float.
    """

    def apply(*args: Value) -> float:
        """Applies the operation to the unwrapped argument."""

        if len(args) != 1:
            raise TypeError(f"{name}() expects exactly one argument")

        raw: Any = args[0].get()

        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise TypeError(
                f"{name}() expects a numeric argument, got '{repr(raw)}'"
            )

        return float(op(raw))

    return apply


def _math_binary(name: str, op: Any) -> Any:
    """
    Builds a builtin two-float math function over Value arguments.

    Args:
        name (str): Function name (for error messages).
        op (Any): Underlying (float, float) -> float operation.

    Returns:
        Any: A callable taking two Values and returning a float.
    """

    def apply(*args: Value) -> float:
        """Applies the operation to the unwrapped arguments."""

        if len(args) != 2:
            raise TypeError(f"{name}() expects exactly two arguments")

        left: Any = args[0].get()
        right: Any = args[1].get()

        for raw in (left, right):

            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise TypeError(
                    f"{name}() expects numeric arguments, got '{repr(raw)}'"
                )

        return float(op(float(left), float(right)))

    return apply


def _conv_valid_length(length: int, kernel: int, stride: int) -> int:
    """
    Computes the valid-mode output length of a conv/pool axis.

    Args:
        length (int): Input axis length.
        kernel (int): Kernel size on that axis.
        stride (int): Stride along that axis.

    Returns:
        int: Output length (>= 1).
    """

    return (length - kernel) // stride + 1


_SQRT_2_OVER_PI: float = math.sqrt(2.0 / math.pi)


def _sigmoid_cell(cell: float) -> float:
    """
    Computes the logistic sigmoid of one cell, overflow-safe.

    Args:
        cell (float): Input value.

    Returns:
        float: 1 / (1 + e^-cell).
    """

    if cell >= 0:
        return 1.0 / (1.0 + math.exp(-cell))

    exponential: float = math.exp(cell)

    return exponential / (1.0 + exponential)


def _gelu_cell(cell: float) -> float:
    """
    Computes the tanh-approximated GELU of one cell.

    Args:
        cell (float): Input value.

    Returns:
        float: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3))).
    """

    inner: float = _SQRT_2_OVER_PI * (cell + 0.044715 * cell ** 3)

    return 0.5 * cell * (1.0 + math.tanh(inner))


def _conv1d_cells(
    signal: list[Any],
    kernel: list[Any],
    stride: int,
) -> list[Any]:
    """
    Computes a valid cross-correlation of a 1-D signal with a kernel.

    Args:
        signal (list[Any]): Flat input payload.
        kernel (list[Any]): Flat kernel payload.
        stride (int): Stride between output positions.

    Returns:
        list[Any]: Flat output payload.
    """

    out_len: int = _conv_valid_length(
        length=len(signal), kernel=len(kernel), stride=stride
    )

    return [
        sum(signal[position * stride + offset] * kernel[offset] for offset in range(len(kernel)))
        for position in range(out_len)
    ]


def _conv2d_cells(
    cells: list[Any],
    rows: int,
    cols: int,
    kernel: list[Any],
    krows: int,
    kcols: int,
    stride: int,
) -> list[Any]:
    """
    Computes a valid cross-correlation of a 2-D image with a kernel.

    Args:
        cells (list[Any]): Flat row-major input payload.
        rows (int): Input row count.
        cols (int): Input column count.
        kernel (list[Any]): Flat row-major kernel payload.
        krows (int): Kernel row count.
        kcols (int): Kernel column count.
        stride (int): Stride between output positions.

    Returns:
        list[Any]: Flat row-major output payload.
    """

    out_rows: int = _conv_valid_length(length=rows, kernel=krows, stride=stride)
    out_cols: int = _conv_valid_length(length=cols, kernel=kcols, stride=stride)

    output: list[Any] = []

    for out_row in range(out_rows):

        for out_col in range(out_cols):

            total: Any = 0

            for k_row in range(krows):

                for k_col in range(kcols):

                    total = total + (
                        cells[(out_row * stride + k_row) * cols + (out_col * stride + k_col)]
                        * kernel[k_row * kcols + k_col]
                    )

            output.append(total)

    return output


def _pool_cells(
    cells: list[Any],
    extents: list[int],
    kernel: int,
    stride: int,
    reducer: str,
) -> list[Any]:
    """
    Pools a 1-D or 2-D tensor with max/min/average windows.

    Args:
        cells (list[Any]): Flat row-major input payload.
        extents (list[int]): Input shape (rank 1 or 2).
        kernel (int): Pooling window size on every axis.
        stride (int): Stride between windows.
        reducer (str): 'max', 'min', or 'avg'.

    Returns:
        list[Any]: Flat row-major output payload.
    """

    if len(extents) == 1:

        length: int = extents[0]
        out_len: int = _conv_valid_length(
            length=length, kernel=kernel, stride=stride
        )

        output: list[Any] = []

        for position in range(out_len):

            window: list[Any] = cells[
                position * stride:position * stride + kernel
            ]

            if reducer == "max":
                output.append(max(window))

            elif reducer == "min":
                output.append(min(window))

            else:
                output.append(sum(window) / kernel)

        return output

    rows: int = extents[0]
    cols: int = extents[1]
    out_rows: int = _conv_valid_length(length=rows, kernel=kernel, stride=stride)
    out_cols: int = _conv_valid_length(length=cols, kernel=kernel, stride=stride)

    pooled: list[Any] = []

    for out_row in range(out_rows):

        for out_col in range(out_cols):

            window = [
                cells[(out_row * stride + k_row) * cols + (out_col * stride + k_col)]
                for k_row in range(kernel)
                for k_col in range(kernel)
            ]

            if reducer == "max":
                pooled.append(max(window))

            elif reducer == "min":
                pooled.append(min(window))

            else:
                pooled.append(sum(window) / (kernel * kernel))

    return pooled


def _tensor_extents(tensor: ValueTensor) -> list[int]:
    """
    Returns the concrete per-axis extents of a tensor.

    Args:
        tensor (ValueTensor): Tensor to inspect.

    Returns:
        list[int]: Extents in axis order.

    Raises:
        TypeError: If the tensor has unbound symbolic dimensions.
    """

    if tensor.concrete_dims is not None:
        return list(tensor.concrete_dims)

    if tensor.value_type.is_symbolic:
        raise TypeError(
            "Cannot operate on a tensor with unbound symbolic dimensions"
        )

    return [dim for dim in tensor.value_type.dims if isinstance(dim, int)]


def _wrap_raw(raw: Any, value_type: Any) -> Value:
    """
    Wraps a raw queue element into the typed Value matching its type.

    Struct instances (opaque at runtime) wrap as externals; everything
    else dispatches through make_value.

    Args:
        raw (Any): Raw element payload.
        value_type (Any): Declared element type.

    Returns:
        Value: The typed wrapper.
    """

    if _is_struct_instance(obj=raw):
        return ValueExternal(payload=raw)

    return make_value(value_type=value_type, raw=raw)
