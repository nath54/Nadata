"""
Core value definitions for Nadata runtime.

Represents typed runtime values corresponding to schema types,
including integers, floats, strings, booleans, arrays, and dictionaries.
"""

# Import Modules
from __future__ import annotations
from typing import Any

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
    TypeArray,
    TypeFloat,
)


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
        elements: list[Any] | None = None
    ) -> None:
        """
        Initializes the ValueArray instance.

        Args:
            value_type (TypeArray): Array type specification.
            elements (list[Any] | None, optional): Initial elements. Defaults to None.

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
        entries: dict[Any, Any] | None = None
    ) -> None:
        """
        Initializes the ValueDict instance.

        Args:
            value_type (TypeDict): Dictionary type specification.
            entries (dict[Any, Any] | None, optional): Initial dictionary. Defaults to None.

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
