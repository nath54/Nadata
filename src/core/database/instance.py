"""
StructInstance runtime object for Nadata.

Represents an active instance of a Struct schema with typed attributes,
a globally unique EntityId, and bound method execution.
"""

# Import Modules
from __future__ import annotations
from collections import deque
from typing import Any, Optional

from src.core.types.types import (
    TypePriorityQueue,
    TypeQueue,
    TypeTensor,
    flatten_nested_lists,
)
from src.core.values.values import (
    Value,
    ValueArray,
    ValuePriorityQueue,
    ValueQueue,
    ValueTensor,
    make_value,
)

from src.core.values.values import Value, make_value
from src.core.struct.struct import Struct
from src.core.struct.struct_attribute import StructAttribute
from src.core.database.entity_id import EntityId


class StructInstance:
    """
    Runtime instance of a Struct with attributes and bound methods.

    Attributes:
        struct (Struct): Schema struct definition.
        entity_id (EntityId): Globally unique 64-bit entity handle.
        attributes (dict[str, Value]): Stored attribute Values.
    """

    def __init__(
        self,
        struct: Struct,
        entity_id: EntityId,
        attributes: Optional[dict[str, Value]] = None
    ) -> None:
        """
        Initializes the StructInstance.

        Args:
            struct (Struct): Struct schema definition.
            entity_id (EntityId): Globally unique 64-bit entity handle.
            attributes (Optional[dict[str, Value]], optional): Attribute values. Defaults to None.
        """

        self.struct: Struct = struct
        self.entity_id: EntityId = entity_id
        self.attributes: dict[str, Value] = attributes if attributes is not None else {}

    def get_attribute(self, name: str) -> Value:
        """
        Retrieves an attribute Value by name.

        Args:
            name (str): Attribute name.

        Returns:
            Value: Attribute Value.

        Raises:
            AttributeError: If attribute is not found.
        """

        if name in self.attributes:
            return self.attributes[name]

        # Check default value in struct schema definition
        attr_def = self.struct.get_attribute(name=name)

        if attr_def is not None and attr_def.has_default:
            return make_value(
                value_type=attr_def.attr_type,
                raw=attr_def.default_value
            )

        raise AttributeError(f"Instance of '{self.struct.name}' has no attribute '{name}'")

    def ensure_attribute(self, name: str) -> None:
        """
        Materializes a lazily-defaulted attribute onto this instance.

        Copies container defaults (lists, dicts) so mutation through an
        assignment target never aliases the schema's shared default value.

        Args:
            name (str): Attribute name.
        """

        if name in self.attributes:
            return None

        attr_def: Any = self.struct.get_attribute(name=name)

        if attr_def is None or not attr_def.has_default:
            return None

        raw_default: Any = attr_def.default_value

        # Copy container defaults to avoid mutating the schema's shared value
        if isinstance(raw_default, (list, dict, set, deque)):
            raw_default = raw_default.copy()

        self.attributes[name] = make_value(
            value_type=attr_def.attr_type,
            raw=raw_default,
        )

        # Stamp resolved extents onto symbolic tensor defaults when every
        # constant has a schema default (otherwise extents stay unbound)
        stamped: Value = self.attributes[name]

        if (
            isinstance(attr_def.attr_type, TypeTensor)
            and attr_def.attr_type.is_symbolic
            and isinstance(stamped, ValueTensor)
        ):
            self._stamp_default_dims(
                tensor_type=attr_def.attr_type,
                tensor_value=stamped,
            )

    def _stamp_default_dims(self, tensor_type: TypeTensor, tensor_value: ValueTensor) -> None:
        """
        Stamps concrete extents onto a symbolic tensor default value.

        Extents resolve from the const attributes' schema defaults; values
        already bound on the instance take precedence.

        Args:
            tensor_type (TypeTensor): Declared tensor type.
            tensor_value (ValueTensor): The freshly materialized value.
        """

        extents: dict[str, int] = {}

        for dim_name in tensor_type.symbolic_dims():
            dim_attr: Optional[StructAttribute] = self.struct.get_attribute(
                name=dim_name
            )

            if dim_attr is None or not dim_attr.has_default:
                return

            self.ensure_attribute(name=dim_name)

            bound_value: Optional[Value] = self.attributes.get(dim_name)

            if bound_value is None:
                return

            raw_dim: Any = bound_value.get()

            if isinstance(raw_dim, bool) or not isinstance(raw_dim, int):
                return

            extents[dim_name] = raw_dim

        tensor_value.concrete_dims = [
            dim if isinstance(dim, int) else extents[dim]
            for dim in tensor_type.dims
        ]

    def set_attribute(self, name: str, value: Value, allow_const: bool = False) -> None:
        """
        Sets an attribute Value by name after verifying schema constraints.

        Args:
            name (str): Attribute name.
            value (Value): Value to store.
            allow_const (bool): Whether constant attributes may be written
                (creation-time binding only). Defaults to False.

        Raises:
            AttributeError: If attribute is not declared in struct.
            TypeError: If value violates declared attribute type, the
                attribute is immutable, or a symbolic tensor shape does not
                match the instance's bound constants.
        """

        attr_def = self.struct.get_attribute(name=name)

        if attr_def is None:
            raise AttributeError(
                f"Cannot set undeclared attribute '{name}' on Struct '{self.struct.name}'"
            )

        # Immutable after construction: binding happens in the constructor
        if attr_def.is_const and not allow_const:
            raise TypeError(
                f"Cannot assign to immutable attribute '{name}' on Struct "
                f"'{self.struct.name}': const attributes bind at construction"
            )

        # Flatten nested array literals into the flat row-major payload of
        # a tensor declaration (shape checked by the type below)
        if isinstance(attr_def.attr_type, TypeTensor) and isinstance(value, ValueArray):
            value = make_value(
                value_type=attr_def.attr_type,
                raw=flatten_nested_lists(raw=value.get()),
            )

        # Validate type
        attr_def.attr_type.check_value(value=value.get())

        # Symbolic tensor shapes validate against the instance's constants
        if (
            isinstance(attr_def.attr_type, TypeTensor)
            and attr_def.attr_type.is_symbolic
        ):
            self._validate_symbolic_shape(
                tensor_type=attr_def.attr_type,
                value=value,
                attribute_name=name,
            )

        # Re-wrap raw payloads passed under a queue declaration (e.g. a
        # plain array literal in a constructor call)
        if isinstance(attr_def.attr_type, (TypeQueue, TypePriorityQueue)) and not isinstance(
            value, (ValueQueue, ValuePriorityQueue)
        ):
            value = make_value(value_type=attr_def.attr_type, raw=value.get())

        self.attributes[name] = value

    def set_creation_attributes(self, values: dict[str, Value]) -> None:
        """
        Binds construction-time attributes, constants first.

        Constants are written before everything else so symbolic tensor
        shapes validate against already-bound extents regardless of the
        order the arguments appear in the constructor call.

        Args:
            values (dict[str, Value]): Attribute values by name.
        """

        # Constants first (they seed the symbolic tensor shapes)
        for const_name, const_value in values.items():
            attr_def = self.struct.get_attribute(name=const_name)

            if attr_def is not None and attr_def.is_const:
                self.set_attribute(
                    name=const_name, value=const_value, allow_const=True
                )

        # Everything else (unknown names raise in set_attribute)
        for other_name, other_value in values.items():
            attr_def = self.struct.get_attribute(name=other_name)

            if attr_def is None or not attr_def.is_const:
                self.set_attribute(name=other_name, value=other_value)

    def _validate_symbolic_shape(
        self,
        tensor_type: TypeTensor,
        value: Value,
        attribute_name: str,
    ) -> None:
        """
        Validates a symbolic tensor payload against the instance constants.

        Every symbolic dimension must be bound on this instance (explicitly
        or through its schema default); the product of the resolved extents
        must equal the payload length. On success the resolved shape is
        stamped onto the value for shape() introspection.

        Args:
            tensor_type (TypeTensor): Declared tensor type.
            value (Value): The (flat) tensor value being stored.
            attribute_name (str): Attribute being set (for error messages).

        Raises:
            TypeError: If a constant is unbound, non-integer, or the shape
                mismatches the payload length.
        """

        extents: dict[str, int] = {}

        for dim_name in tensor_type.symbolic_dims():

            # Materialize schema defaults of unbound constants
            self.ensure_attribute(name=dim_name)

            dim_value: Optional[Value] = self.attributes.get(dim_name)

            if dim_value is None:
                raise TypeError(
                    f"Cannot set '{attribute_name}': tensor dimension "
                    f"'{dim_name}' is not bound on this instance"
                )

            raw_dim: Any = dim_value.get()

            if isinstance(raw_dim, bool) or not isinstance(raw_dim, int):
                raise TypeError(
                    f"Cannot set '{attribute_name}': tensor dimension "
                    f"'{dim_name}' must be an integer, got "
                    f"{type(raw_dim).__name__}"
                )

            extents[dim_name] = raw_dim

        expected: int = 1

        for dim in tensor_type.dims:

            if isinstance(dim, str):
                expected *= extents[dim]

            else:
                expected *= dim

        payload: list[Any] = value.get()

        if len(payload) != expected:
            raise TypeError(
                f"Cannot set '{attribute_name}': shape "
                f"{tensor_type.dims} with bound dims {extents} requires "
                f"{expected} elements, got {len(payload)}"
            )

        # Stamp the resolved shape onto the value (used by shape())
        if isinstance(value, ValueTensor):
            value.concrete_dims = [
                dim if isinstance(dim, int) else extents[dim]
                for dim in tensor_type.dims
            ]

    def call_method(
        self,
        name: str,
        args: Optional[list[Any]] = None,
        kwargs: Optional[dict[str, Any]] = None,
        globals_dict: Optional[dict[str, Any]] = None
    ) -> Optional[Value]:
        """
        Invokes a struct method bound to this instance.

        Args:
            name (str): Method name.
            args (Optional[list[Any]], optional): Positional arguments. Defaults to None.
            kwargs (Optional[dict[str, Any]], optional): Keyword arguments. Defaults to None.
            globals_dict (Optional[dict[str, Any]], optional): Global scope. Defaults to None.

        Returns:
            Optional[Value]: Method execution return Value.

        Raises:
            AttributeError: If method is not declared on struct.
        """

        method = self.struct.get_method(name=name)

        if method is None:
            raise AttributeError(
                f"Struct '{self.struct.name}' has no method '@{name}'"
            )

        resolved_args: list[Any] = args if args is not None else []
        resolved_kwargs: dict[str, Any] = kwargs if kwargs is not None else {}
        resolved_globals: dict[str, Any] = globals_dict if globals_dict is not None else {}

        return method.execute(
            self_instance=self,
            args=resolved_args,
            kwargs=resolved_kwargs,
            globals_dict=resolved_globals
        )


    def __getattr__(self, name: str) -> Any:
        """
        Dynamic fallback to access attributes or bind methods directly.

        Args:
            name (str): Member name.

        Returns:
            Any: Value or callable bound method.

        Raises:
            AttributeError: If member is not found.
        """

        if name in self.attributes:
            return self.attributes[name]

        # Check struct methods for direct python invocation
        method = self.struct.get_method(name=name)

        if method is not None:
            return lambda *args, **kwargs: self.call_method(
                name=name,
                args=list(args),
                kwargs=kwargs
            )

        # Check default attribute value
        attr_def = self.struct.get_attribute(name=name)

        if attr_def is not None and attr_def.has_default:
            return make_value(
                value_type=attr_def.attr_type,
                raw=attr_def.default_value
            )

        raise AttributeError(f"'{self.struct.name}' instance has no member '{name}'")

    def __repr__(self) -> str:
        """
        Returns string representation of StructInstance.

        Returns:
            str: Struct name with entity_id.
        """

        return f"<{self.struct.name} id={self.entity_id.local_id}>"
