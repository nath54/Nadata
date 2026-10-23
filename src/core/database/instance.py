"""
StructInstance runtime object for Nadata.

Represents an active instance of a Struct schema with typed attributes,
a globally unique EntityId, and bound method execution.
"""

# Import Modules
from __future__ import annotations
from collections import deque
from typing import Any, Optional

from src.core.expressions.builtins import (
    TensorInit,
    build_type_globals,
    materialize_tensor_init,
)
from src.core.expressions.expression import Expression
from src.core.expressions.scope import ExecutionScope
from src.core.types.types import (
    TypeInt,
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
        dirty (bool): True when the instance differs from its stored
            (or not-yet-stored) state; set by mutation, cleared by the
            flush write-back. New instances start dirty.
        pinned (bool): True when explicitly kept resident; pinned
            instances are never evicted from the decode cache.
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
        self.dirty: bool = True
        self.pinned: bool = False
        self.nid: Optional[str] = None

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

            # Expression-backed defaults (tensor initializers, derived
            # consts) materialize through their expression. Tensor results
            # are cached so identity stays stable across reads: autograd
            # flags, gradients, and version counters live on the tensor
            # object itself.
            if attr_def.default_expr is not None:

                value: Value = self._evaluate_derived_default(
                    attr_def=attr_def
                )

                if isinstance(attr_def.attr_type, TypeTensor):
                    self.attributes[name] = value

                return value

            if isinstance(attr_def.attr_type, TypeTensor):

                self.attributes[name] = make_value(
                    value_type=attr_def.attr_type,
                    raw=attr_def.default_value
                )

                return self.attributes[name]

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

        if attr_def is None:
            return None

        # Derived constants evaluate their expression against the bound
        # constants (used when creation paths were bypassed)
        if attr_def.default_expr is not None:
            self.attributes[name] = self._evaluate_derived_default(
                attr_def=attr_def
            )

            return None

        if not attr_def.has_default:
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

        # Pending tensor initializers (zeros()/fill()/eye()) materialize
        # against the declared type, resolving symbolic dims from the
        # instance's bound constants
        if isinstance(attr_def.attr_type, TypeTensor) and isinstance(value, TensorInit):
            value = self._materialize_attribute_init(
                tensor_type=attr_def.attr_type,
                init=value,
                attribute_name=name,
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
        self.dirty = True

    def apply_upsert(self, values: dict[str, Value]) -> None:
        """
        Re-binds creation attributes onto an existing (nid'd) instance.

        Partial update semantics: passed attributes are re-written
        (constants included — upsert is re-construction); attributes
        not passed keep their current value. Derived constants cannot
        be set directly, matching creation semantics.

        Args:
            values (dict[str, Value]): Attribute values by name.

        Raises:
            AttributeError: If a name is not declared in the schema.
            TypeError: If a derived constant is passed directly.
        """

        for name, value in values.items():

            attr_def: Optional[StructAttribute] = self.struct.get_attribute(
                name=name
            )

            if attr_def is None:
                raise AttributeError(
                    f"Cannot set undeclared attribute '{name}' on Struct "
                    f"'{self.struct.name}'"
                )

            if attr_def.is_const and attr_def.default_expr is not None:
                raise TypeError(
                    f"Cannot set '{name}' directly: it is "
                    f"derived from other constants"
                )

            self.set_attribute(name=name, value=value, allow_const=True)

    def set_creation_attributes(self, values: dict[str, Value]) -> None:
        """
        Binds construction-time attributes, constants first.

        Constants are written before everything else so symbolic tensor
        shapes validate against already-bound extents regardless of the
        order the arguments appear in the constructor call. Derived
        constants (expression defaults referencing `self.<const>`) are
        evaluated after the plain constants, in declaration order.

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

        # Derived constants next (in declaration order; each may reference
        # the constants bound above or earlier-derived ones)
        for attr_def in self.struct.get_all_attributes().values():

            if attr_def.is_const and attr_def.default_expr is not None:

                if attr_def.name in values:
                    raise TypeError(
                        f"Cannot set '{attr_def.name}' directly: it is "
                        f"derived from other constants"
                    )

                self.set_attribute(
                    name=attr_def.name,
                    value=self._evaluate_derived_default(attr_def=attr_def),
                    allow_const=True,
                )

        # Everything else (unknown names raise in set_attribute)
        for other_name, other_value in values.items():
            attr_def = self.struct.get_attribute(name=other_name)

            if attr_def is None or not attr_def.is_const:
                self.set_attribute(name=other_name, value=other_value)

    def _evaluate_derived_default(self, attr_def: StructAttribute) -> Value:
        """
        Evaluates a derived const's expression against the bound constants.

        The expression evaluates in a fresh scope exposing `self` and every
        currently bound attribute value by name (so both `self.Dim / 2`
        and bare `Dim / 2` forms work).

        Args:
            attr_def (StructAttribute): The derived const definition.

        Returns:
            Value: The evaluated value.

        Raises:
            TypeError: If a referenced constant is not yet bound.
        """

        parameters: dict[str, Value] = {}

        for name, bound_value in self.attributes.items():
            attr: Optional[StructAttribute] = self.struct.get_attribute(name=name)

            if attr is not None and attr.is_const:
                parameters[name] = bound_value

        scope: ExecutionScope = ExecutionScope(
            self_instance=self,
            parameters=parameters,
            globals_dict=build_type_globals(),
        )

        expression: Optional[Expression] = attr_def.default_expr
        assert expression is not None

        value: Value = expression.evaluate(scope=scope)

        # Pending tensor initializers materialize against the declared
        # type (symbolic dims resolve from the bound constants)
        if isinstance(attr_def.attr_type, TypeTensor) and isinstance(value, TensorInit):
            return self._materialize_attribute_init(
                tensor_type=attr_def.attr_type,
                init=value,
                attribute_name=attr_def.name,
            )

        # Python's '/' yields floats; an integral float binding to a basic
        # integer const coerces (2.0 -> 2). Non-integral values still fail
        # the declared type check in set_attribute.
        raw_value: Any = value.get()

        if (
            isinstance(attr_def.attr_type, TypeInt)
            and isinstance(raw_value, float)
            and raw_value.is_integer()
        ):
            return make_value(value_type=attr_def.attr_type, raw=int(raw_value))

        return value

    def _materialize_attribute_init(
        self,
        tensor_type: TypeTensor,
        init: TensorInit,
        attribute_name: str,
    ) -> ValueTensor:
        """
        Materializes a pending tensor initializer for an attribute.

        Symbolic dimensions resolve from the instance's bound constants
        (schema defaults are materialized first).

        Args:
            tensor_type (TypeTensor): Declared tensor type.
            init (TensorInit): Pending initialization request.
            attribute_name (str): Attribute being set (for errors).

        Returns:
            ValueTensor: The initialized tensor.

        Raises:
            TypeError: If a symbolic dimension is not bound on this
                instance.
        """

        extents: dict[str, int] = {}

        for dim_name in tensor_type.symbolic_dims():

            # Materialize schema defaults of unbound constants
            self.ensure_attribute(name=dim_name)

            dim_value: Optional[Value] = self.attributes.get(dim_name)

            if dim_value is None:
                raise TypeError(
                    f"Cannot initialize '{attribute_name}': tensor dimension "
                    f"'{dim_name}' is not bound on this instance"
                )

            raw_dim: Any = dim_value.get()

            if isinstance(raw_dim, bool) or not isinstance(raw_dim, int):
                raise TypeError(
                    f"Cannot initialize '{attribute_name}': tensor dimension "
                    f"'{dim_name}' must be an integer"
                )

            extents[dim_name] = raw_dim

        return materialize_tensor_init(
            init=init,
            tensor_type=tensor_type,
            extents=extents or None,
        )

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
            return self.get_attribute(name=name)

        raise AttributeError(f"'{self.struct.name}' instance has no member '{name}'")

    def __repr__(self) -> str:
        """
        Returns string representation of StructInstance.

        Returns:
            str: Struct name with entity_id.
        """

        return f"<{self.struct.name} id={self.entity_id.local_id}>"
