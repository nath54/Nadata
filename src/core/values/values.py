"""
Core value definitions for Nadata runtime.

Represents typed runtime values corresponding to schema types,
including integers, floats, strings, booleans, arrays, and dictionaries.
"""

# Import Modules
from __future__ import annotations
from typing import Any, Optional

from src.core.types.bits.bits import (
    Bits,
    BitArray,
)
from src.core.types.types import (
    Type,
    TypeStr,
    TypeInt,
    TypeBool,
    TypeDict,
    TypeColor,
    TypeArray,
    TypeFloat,
    TypeDateTime,
)
from src.core.types.home_types.color import Color
from src.core.types.home_types.datetime import DateTime



class Value:
    """
    Abstract base class for all runtime values in Nadata.

    Attributes:
        value_type (Type): Schema type associated with this value.
    """

    def __init__(self, value_type: Type) -> None:
        """
        Initializes the base Value instance.

        Args:
            value_type (Type): Associated schema type specification.
        """

        # Store schema type
        self.value_type: Type = value_type

    def get(self) -> Any:
        """
        Returns the raw underlying Python value.

        Returns:
            Any: Underlying raw value.
        """

        return getattr(self, "value", None)

    def __repr__(self) -> str:
        """
        Returns string representation of the Value.

        Returns:
            str: Class name representation.
        """

        return f"{self.__class__.__name__}(type={repr(self.value_type)})"


class ValueInt(Value):
    """
    Runtime representation of a sized integer.

    Attributes:
        value_type (TypeInt): Associated integer schema type.
        value (int): Native Python integer stored in memory.
    """

    def __init__(
        self,
        value_type: TypeInt,
        value: int = 0
    ) -> None:
        """
        Initializes the ValueInt instance.

        Args:
            value_type (TypeInt): Integer type specification.
            value (int, optional): Initial integer value. Defaults to 0.

        Raises:
            TypeError: If value is not an int or is a bool.
            ValueError: If value is out of bounds for the bit specification.
        """

        # Initialize parent class
        super().__init__(value_type=value_type)

        # Enforce type and bit bounds invariant
        value_type.check_value(value=value)

        self.value_type: TypeInt = value_type
        self.value: int = value

    def to_bits(self) -> BitArray:
        """
        Serializes this integer to a BitArray using its bit specification.

        Returns:
            BitArray: Encoded bit array.
        """

        # Convert native integer to bit array
        return Bits.int_val_to_bits(
            value=self.value,
            value_type=self.value_type.bit_spec
        )

    def __repr__(self) -> str:
        """
        Returns string representation of the ValueInt.

        Returns:
            str: Representation showing type name and value.
        """

        return f"{repr(self.value_type)}({self.value})"


class ValueFloat(Value):
    """
    Runtime representation of a sized floating-point number.

    Attributes:
        value_type (TypeFloat): Associated float schema type.
        value (float): Native Python float stored in memory.
    """

    def __init__(
        self,
        value_type: TypeFloat,
        value: float = 0.0
    ) -> None:
        """
        Initializes the ValueFloat instance.

        Args:
            value_type (TypeFloat): Float type specification.
            value (float, optional): Initial float value. Defaults to 0.0.

        Raises:
            TypeError: If value is not a float or int, or is a bool.
            ValueError: If an unsigned float receives a negative value.
        """

        # Initialize parent class
        super().__init__(value_type=value_type)

        # Enforce float constraints
        value_type.check_value(value=value)

        self.value_type: TypeFloat = value_type
        self.value: float = float(value)

    def to_bits(self) -> BitArray:
        """
        Serializes this float to a BitArray using its bit specification.

        Returns:
            BitArray: Encoded bit array.
        """

        # Convert native float to bit array
        return Bits.float_val_to_bits(
            value=self.value,
            value_type=self.value_type.bit_spec
        )

    def __repr__(self) -> str:
        """
        Returns string representation of the ValueFloat.

        Returns:
            str: Representation showing type name and value.
        """

        return f"{repr(self.value_type)}({self.value})"


class ValueStr(Value):
    """
    Runtime representation of a string value.

    Attributes:
        value_type (TypeStr): Associated string schema type.
        value (str): Native Python string stored in memory.
    """

    def __init__(
        self,
        value_type: TypeStr,
        value: str = ""
    ) -> None:
        """
        Initializes the ValueStr instance.

        Args:
            value_type (TypeStr): String type specification.
            value (str, optional): Initial string value. Defaults to "".

        Raises:
            TypeError: If value is not a string.
        """

        # Initialize parent class
        super().__init__(value_type=value_type)

        # Enforce string type
        value_type.check_value(value=value)

        self.value_type: TypeStr = value_type
        self.value: str = value

    def get(self) -> str:
        """
        Returns the raw string value.

        Returns:
            str: Underlying string.
        """

        return self.value

    def __repr__(self) -> str:
        """
        Returns string representation of the ValueStr.

        Returns:
            str: Quoted string representation.
        """

        return f'Str("{self.value}")'


class ValueBool(Value):
    """
    Runtime representation of a boolean value.

    Attributes:
        value_type (TypeBool): Associated boolean schema type.
        value (bool): Native Python boolean stored in memory.
    """

    def __init__(
        self,
        value_type: TypeBool,
        value: bool = False
    ) -> None:
        """
        Initializes the ValueBool instance.

        Args:
            value_type (TypeBool): Boolean type specification.
            value (bool, optional): Initial boolean value. Defaults to False.

        Raises:
            TypeError: If value is not a boolean.
        """

        # Initialize parent class
        super().__init__(value_type=value_type)

        # Enforce boolean type
        value_type.check_value(value=value)

        self.value_type: TypeBool = value_type
        self.value: bool = value

    def get(self) -> bool:
        """
        Returns the raw boolean value.

        Returns:
            bool: Underlying boolean.
        """

        return self.value

    def __repr__(self) -> str:
        """
        Returns string representation of the ValueBool.

        Returns:
            str: Boolean representation.
        """

        return f"Bool({self.value})"


class ValueArray(Value):
    """
    Runtime representation of a typed array of values.

    Attributes:
        value_type (TypeArray): Associated array schema type.
        elements (list[Any]): List of contained element values.
    """

    def __init__(
        self,
        value_type: TypeArray,
        elements: Optional[list[Any]] = None
    ) -> None:
        """
        Initializes the ValueArray instance.

        Args:
            value_type (TypeArray): Array type specification.
            elements (Optional[list[Any]], optional): Initial elements. Defaults to None.

        Raises:
            TypeError: If elements violate the array value_type.
        """

        # Initialize parent class
        super().__init__(value_type=value_type)

        resolved_elements: list[Any] = elements if elements is not None else []

        # Validate elements against element type
        value_type.check_value(value=resolved_elements)

        self.value_type: TypeArray = value_type
        self.elements: list[Any] = resolved_elements

    def get(self) -> list[Any]:
        """
        Returns the raw elements list.

        Returns:
            list[Any]: List of contained elements.
        """

        return self.elements

    def __len__(self) -> int:
        """
        Returns count of elements in the array.

        Returns:
            int: Number of elements.
        """

        return len(self.elements)

    def __repr__(self) -> str:
        """
        Returns string representation of the ValueArray.

        Returns:
            str: Array representation.
        """

        return f"{repr(self.value_type)}({self.elements})"


class ValueDict(Value):
    """
    Runtime representation of a typed dictionary of key-value pairs.

    Attributes:
        value_type (TypeDict): Associated dictionary schema type.
        entries (dict[Any, Any]): Dictionary of contained entries.
    """

    def __init__(
        self,
        value_type: TypeDict,
        entries: Optional[dict[Any, Any]] = None
    ) -> None:
        """
        Initializes the ValueDict instance.

        Args:
            value_type (TypeDict): Dictionary type specification.
            entries (Optional[dict[Any, Any]], optional): Initial dictionary. Defaults to None.

        Raises:
            TypeError: If entries violate the dictionary key_type or value_type.
        """

        # Initialize parent class
        super().__init__(value_type=value_type)

        resolved_entries: dict[Any, Any] = entries if entries is not None else {}

        # Validate entries against key and value types
        value_type.check_value(value=resolved_entries)

        self.value_type: TypeDict = value_type
        self.entries: dict[Any, Any] = resolved_entries

    def get(self) -> dict[Any, Any]:
        """
        Returns the raw entries dictionary.

        Returns:
            dict[Any, Any]: Dictionary of contained entries.
        """

        return self.entries

    def __len__(self) -> int:
        """
        Returns count of entries in the dictionary.

        Returns:
            int: Number of entries.
        """

        return len(self.entries)

    def __repr__(self) -> str:
        """
        Returns string representation of the ValueDict.

        Returns:
            str: Dictionary representation.
        """

        return f"{repr(self.value_type)}({self.entries})"


class ValueDateTime(Value):
    """
    Runtime representation of a DateTime value.

    Attributes:
        value_type (TypeDateTime): Associated DateTime schema type.
        value (DateTime): Native DateTime instance stored in memory.
    """

    def __init__(
        self,
        value_type: Optional[TypeDateTime] = None,
        value: Optional[DateTime] = None
    ) -> None:
        """
        Initializes the ValueDateTime instance.

        Args:
            value_type (Optional[TypeDateTime], optional): DateTime type spec. Defaults to None.
            value (Optional[DateTime], optional): Initial DateTime value. Defaults to None.

        Raises:
            TypeError: If value is not a DateTime.
        """

        resolved_type: TypeDateTime = value_type if value_type is not None else TypeDateTime()
        resolved_value: DateTime = value if value is not None else DateTime()

        # Initialize parent class
        super().__init__(value_type=resolved_type)

        resolved_type.check_value(value=resolved_value)

        self.value_type: TypeDateTime = resolved_type
        self.value: DateTime = resolved_value

    def get(self) -> DateTime:
        """
        Returns the raw DateTime instance.

        Returns:
            DateTime: Underlying DateTime.
        """

        return self.value

    def __repr__(self) -> str:
        """
        Returns string representation of the ValueDateTime.

        Returns:
            str: Representation showing the DateTime value.
        """

        return repr(self.value)


class ValueColor(Value):
    """
    Runtime representation of an RGBA Color value.

    Attributes:
        value_type (TypeColor): Associated Color schema type.
        value (Color): Native Color instance stored in memory.
    """

    def __init__(
        self,
        value_type: Optional[TypeColor] = None,
        value: Optional[Color] = None
    ) -> None:
        """
        Initializes the ValueColor instance.

        Args:
            value_type (Optional[TypeColor], optional): Color type spec. Defaults to None.
            value (Optional[Color], optional): Initial Color value. Defaults to None.

        Raises:
            TypeError: If value is not a Color.
        """

        resolved_type: TypeColor = value_type if value_type is not None else TypeColor()
        resolved_value: Color = value if value is not None else Color(0, 0, 0, 255)

        # Initialize parent class
        super().__init__(value_type=resolved_type)

        resolved_type.check_value(value=resolved_value)

        self.value_type: TypeColor = resolved_type
        self.value: Color = resolved_value

    def get(self) -> Color:
        """
        Returns the raw Color instance.

        Returns:
            Color: Underlying Color.
        """

        return self.value

    def __repr__(self) -> str:
        """
        Returns string representation of the ValueColor.

        Returns:
            str: Representation showing the Color value.
        """

        return repr(self.value)


def make_value(value_type: Type, raw: Any) -> Value:
    """
    Wraps a raw value into the typed Value matching its schema type.

    Args:
        value_type (Type): Schema type of the raw value.
        raw (Any): Raw value to wrap.

    Returns:
        Value: The typed Value wrapper.

    Raises:
        TypeError: If the schema type has no matching Value class.
    """

    # Collections and home types have dedicated constructor signatures
    if isinstance(value_type, TypeArray):
        return ValueArray(value_type=value_type, elements=raw)

    if isinstance(value_type, TypeDict):
        return ValueDict(value_type=value_type, entries=raw)

    if isinstance(value_type, TypeDateTime):
        return ValueDateTime(value=raw)

    if isinstance(value_type, TypeColor):
        return ValueColor(value=raw)

    # Primitives share the (value_type, value) signature
    if isinstance(value_type, TypeBool):
        return ValueBool(value_type=value_type, value=raw)

    if isinstance(value_type, TypeInt):
        return ValueInt(value_type=value_type, value=raw)

    if isinstance(value_type, TypeFloat):
        return ValueFloat(value_type=value_type, value=raw)

    if isinstance(value_type, TypeStr):
        return ValueStr(value_type=value_type, value=raw)

    raise TypeError(f"Cannot wrap raw value into '{repr(value_type)}'")


class TypeExternal(Type):
    """
    Schema type accepting any opaque runtime payload.

    Used by ValueExternal wrappers for runtime objects that have no schema
    type of their own, such as query results or lambda closures.
    """

    def check_value(self, value: Any) -> None:
        """
        Accepts any payload value without validation.

        Args:
            value (Any): Payload value to (not) validate.
        """

        # External payloads are intentionally unvalidated
        return

    def __repr__(self) -> str:
        """
        Returns string representation of the type.

        Returns:
            str: Type representation.
        """

        return "External"


# Shared singleton instance used by every ValueExternal wrapper
TYPE_EXTERNAL: TypeExternal = TypeExternal()


class ValueExternal(Value):
    """
    Opaque Value wrapping a runtime object that is not schema-typed.

    Typical payloads are query results (QuerySet), lambda closure callables,
    and struct instance references circulating in loader scopes. Member
    access and callables resolve through get(), so wrapped payloads behave
    transparently in expressions.

    Attributes:
        payload (Any): Wrapped runtime object.
    """

    def __init__(self, payload: Any) -> None:
        """
        Initializes the ValueExternal instance.

        Args:
            payload (Any): Runtime object to wrap.
        """

        # Initialize parent class
        super().__init__(value_type=TYPE_EXTERNAL)

        self.payload: Any = payload

    def get(self) -> Any:
        """
        Returns the wrapped payload object.

        Returns:
            Any: Wrapped runtime object.
        """

        return self.payload

    def __repr__(self) -> str:
        """
        Returns string representation of the wrapper.

        Returns:
            str: Wrapper representation.
        """

        # Struct schemas render as type handles (e.g. from type(x))
        if hasattr(self.payload, "get_all_attributes") and hasattr(self.payload, "name"):
            return f"<Type {self.payload.name}>"

        # Schema types render as type handles (e.g. from type(x.field))
        if hasattr(self.payload, "check_value"):
            return f"<Type {repr(self.payload)}>"

        return f"ValueExternal({repr(self.payload)})"
