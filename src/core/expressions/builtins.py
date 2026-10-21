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
from collections import deque
from typing import Any, Optional, cast

from src.core.expressions.scope import ScopeAwareFunction
from src.core.types.types import TypeArray, TypeBool, TypeDict, TypeInt, TypeStr
from src.core.values.values import (
    TYPE_EXTERNAL,
    Value,
    ValueArray,
    ValueBool,
    ValueDict,
    ValueExternal,
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

    from src.core.types.types import Int64, TypeArray

    return ValueArray(
        value_type=TypeArray(value_type=TypeArray(value_type=Int64)),
        elements=[list(coords) for coords in coords_list],
    )


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

            return ValueExternal(payload=None)

        if self.member == "size":
            return ValueInt(
                value_type=TypeInt(num_bits=64, signed=True),
                value=len(tensor.cells),
            )

        if self.member == "shape":
            return ValueArray(
                value_type=TypeArray(value_type=TypeInt(num_bits=32, signed=True)),
                elements=list(tensor.value_type.dims),
            )

        raise TypeError(f"Unknown tensor member '{self.member}'")


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
