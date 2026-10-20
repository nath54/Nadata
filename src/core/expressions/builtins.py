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
from typing import Any, Optional, cast

from src.core.types.types import TypeArray, TypeStr
from src.core.values.values import TYPE_EXTERNAL, Value, ValueArray, ValueExternal, ValueStr

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
