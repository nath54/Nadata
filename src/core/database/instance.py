"""
StructInstance runtime object for Nadata.

Represents an active instance of a Struct schema with typed attributes,
a globally unique EntityId, and bound method execution.
"""

# Import Modules
from __future__ import annotations
from typing import Any, Optional

from src.core.values.values import Value, make_value
from src.core.struct.struct import Struct
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
        if isinstance(raw_default, (list, dict, set)):
            raw_default = raw_default.copy()

        self.attributes[name] = make_value(
            value_type=attr_def.attr_type,
            raw=raw_default,
        )

    def set_attribute(self, name: str, value: Value) -> None:
        """
        Sets an attribute Value by name after verifying schema constraints.

        Args:
            name (str): Attribute name.
            value (Value): Value to store.

        Raises:
            AttributeError: If attribute is not declared in struct.
            TypeError: If value violates declared attribute type.
        """

        attr_def = self.struct.get_attribute(name=name)

        if attr_def is None:
            raise AttributeError(
                f"Cannot set undeclared attribute '{name}' on Struct '{self.struct.name}'"
            )

        # Validate type
        attr_def.attr_type.check_value(value=value.get())

        self.attributes[name] = value

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
