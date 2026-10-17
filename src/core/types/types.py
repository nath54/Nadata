"""
Core type definitions for Nadata.

Defines the base schema types including sized integers, sized floats,
strings, booleans, arrays, and dictionaries.
"""

# Import Modules
from __future__ import annotations
from typing import Any, Optional

from src.core.types.bits.bits import (
    BitValueTypeInt,
    BitValueTypeFloat,
)
from src.core.types.home_types.color import Color
from src.core.types.home_types.datetime import DateTime



class Type:
    """
    Abstract base type for all Nadata schema types.

    Attributes:
        default_value (Any): The default value associated with this type.
    """

    def __init__(self) -> None:
        """
        Initializes the base Type instance.
        """

        # Default value of the type
        self.default_value: Any = None

    def check_value(self, value: Any) -> None:
        """
        Validates that a value satisfies this type specification.

        Args:
            value (Any): Value to validate.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """

        raise NotImplementedError("check_value must be implemented by subclasses.")

    def __repr__(self) -> str:
        """
        Returns string representation of the Type.

        Returns:
            str: Class name representation.
        """

        return self.__class__.__name__


class TypeStr(Type):
    """
    Schema type representing a Unicode string.

    Attributes:
        default_value (str): Default string value (empty string).
    """

    def __init__(self) -> None:
        """
        Initializes the TypeStr instance.
        """

        # Initialize parent class
        super().__init__()

        # Default value of the string type
        self.default_value: str = ""

    def check_value(self, value: Any) -> None:
        """
        Validates that a value is a string.

        Args:
            value (Any): Value to validate.

        Raises:
            TypeError: If value is not a string.
        """

        # Verify type is strictly a string
        if not isinstance(value, str):
            raise TypeError(f"Expected str, got {type(value).__name__}")

    def __repr__(self) -> str:
        """
        Returns string representation of the TypeStr.

        Returns:
            str: Representation of TypeStr.
        """

        return "Str"


class TypeInt(Type):
    """
    Schema type representing an integer with an exact bit width and signedness.

    Attributes:
        bit_spec (BitValueTypeInt): Underlying bit specification for bounds and validation.
        num_bits (int): Total bit width.
        signed (bool): Whether the integer is signed or unsigned.
        default_value (int): Default integer value (0).
    """

    def __init__(
        self,
        num_bits: int = 32,
        signed: bool = True,
        endian: str = "big"
    ) -> None:
        """
        Initializes the TypeInt instance.

        Args:
            num_bits (int, optional): Total bit width. Defaults to 32.
            signed (bool, optional): Whether integer is signed. Defaults to True.
            endian (str, optional): Byte/bit order ('big' or 'little'). Defaults to 'big'.
        """

        # Initialize parent class
        super().__init__()

        # Construct underlying bit specification
        self.bit_spec: BitValueTypeInt = BitValueTypeInt(
            num_bits=num_bits,
            signed=signed,
            endian=endian
        )

        self.num_bits: int = num_bits
        self.signed: bool = signed

        # Default value of the integer type
        self.default_value: int = 0

    def check_value(self, value: Any) -> None:
        """
        Validates that a value is an integer and within the bit bounds.

        Args:
            value (Any): Value to validate.

        Raises:
            TypeError: If value is not an int or is a bool.
            ValueError: If value is outside the representable bounds.
        """

        # Delegate validation to the bit specification
        self.bit_spec.check_value(value=value)

    def __repr__(self) -> str:
        """
        Returns string representation of the TypeInt.

        Returns:
            str: Representation with bit width and signedness.
        """

        sign_str: str = "Int" if self.signed else "UInt"
        return f"{sign_str}{self.num_bits}"


class TypeFloat(Type):
    """
    Schema type representing a floating-point number with custom bit specifications.

    Attributes:
        bit_spec (BitValueTypeFloat): Underlying bit specification for bounds and validation.
        num_bits (int): Total bit width.
        num_bits_exponent (int): Number of bits reserved for exponent.
        num_bits_mantissa (int): Number of bits reserved for mantissa.
        signed (bool): Whether the float has a sign bit.
        default_value (float): Default float value (0.0).
    """

    def __init__(
        self,
        num_bits: int = 32,
        num_bits_exponent: int = 8,
        signed: bool = True,
        endian: str = "big"
    ) -> None:
        """
        Initializes the TypeFloat instance.

        Args:
            num_bits (int, optional): Total bit width. Defaults to 32.
            num_bits_exponent (int, optional): Number of exponent bits. Defaults to 8.
            signed (bool, optional): Whether float is signed. Defaults to True.
            endian (str, optional): Byte/bit order ('big' or 'little'). Defaults to 'big'.
        """

        # Initialize parent class
        super().__init__()

        # Construct underlying bit specification
        self.bit_spec: BitValueTypeFloat = BitValueTypeFloat(
            num_bits=num_bits,
            num_bits_exponent=num_bits_exponent,
            signed=signed,
            endian=endian
        )

        self.num_bits: int = num_bits
        self.num_bits_exponent: int = num_bits_exponent
        self.num_bits_mantissa: int = self.bit_spec.num_bits_mantissa
        self.signed: bool = signed

        # Default value of the float type
        self.default_value: float = 0.0

    def check_value(self, value: Any) -> None:
        """
        Validates that a value is a float/int within the representable float bounds.

        Args:
            value (Any): Value to validate.

        Raises:
            TypeError: If value is not a float or int, or is a bool.
            ValueError: If unsigned float receives a negative value.
        """

        # Delegate validation to the bit specification
        self.bit_spec.check_value(value=value)

    def __repr__(self) -> str:
        """
        Returns string representation of the TypeFloat.

        Returns:
            str: Representation with bit details.
        """

        return f"Float{self.num_bits}(exp={self.num_bits_exponent}, mant={self.num_bits_mantissa})"


class TypeBool(Type):
    """
    Schema type representing a boolean (True or False).

    Attributes:
        default_value (bool): Default boolean value (False).
    """

    def __init__(self) -> None:
        """
        Initializes the TypeBool instance.
        """

        # Initialize parent class
        super().__init__()

        # Default value of the boolean type
        self.default_value: bool = False

    def check_value(self, value: Any) -> None:
        """
        Validates that a value is a boolean.

        Args:
            value (Any): Value to validate.

        Raises:
            TypeError: If value is not a boolean.
        """

        # Verify type is strictly boolean
        if not isinstance(value, bool):
            raise TypeError(f"Expected bool, got {type(value).__name__}")

    def __repr__(self) -> str:
        """
        Returns string representation of the TypeBool.

        Returns:
            str: Representation of TypeBool.
        """

        return "Bool"


class TypeArray(Type):
    """
    Schema type representing a typed list of elements.

    Attributes:
        value_type (Type): Type of elements stored in the array.
        default_value (list[Any]): Default empty list.
    """

    def __init__(self, value_type: Type) -> None:
        """
        Initializes the TypeArray instance.

        Args:
            value_type (Type): Type of elements allowed in this array.
        """

        # Initialize parent class
        super().__init__()

        self.value_type: Type = value_type

        # Default value of the array type
        self.default_value: list[Any] = []

    def check_value(self, value: Any) -> None:
        """
        Validates that a value is a list and each item matches value_type.

        Args:
            value (Any): List value to validate.

        Raises:
            TypeError: If value is not a list or an item violates value_type.
        """

        # Verify container type
        if not isinstance(value, list):
            raise TypeError(f"Expected list, got {type(value).__name__}")

        # Check each item against element type
        for item in value:
            self.value_type.check_value(value=item)

    def __repr__(self) -> str:
        """
        Returns string representation of the TypeArray.

        Returns:
            str: Representation showing element type.
        """

        return f"Array[{repr(self.value_type)}]"


class TypeDict(Type):
    """
    Schema type representing a typed dictionary of key-value pairs.

    Attributes:
        key_type (Type): Type of keys allowed in the dictionary.
        value_type (Type): Type of values allowed in the dictionary.
        default_value (dict[Any, Any]): Default empty dictionary.
    """

    def __init__(
        self,
        value_type: Type,
        key_type: Optional[Type] = None
    ) -> None:
        """
        Initializes the TypeDict instance.

        Args:
            value_type (Type): Type of values allowed in this dictionary.
            key_type (Optional[Type], optional): Type of keys. Defaults to TypeStr().
        """

        # Initialize parent class
        super().__init__()

        self.value_type: Type = value_type
        self.key_type: Type = key_type if key_type is not None else TypeStr()

        # Default value of the dictionary type
        self.default_value: dict[Any, Any] = {}

    def check_value(self, value: Any) -> None:
        """
        Validates that a value is a dict and all keys/values match specifications.

        Args:
            value (Any): Dictionary value to validate.

        Raises:
            TypeError: If value is not a dict or a key/value violates types.
        """

        # Verify container type
        if not isinstance(value, dict):
            raise TypeError(f"Expected dict, got {type(value).__name__}")

        # Check each key and value
        for key, val in value.items():
            self.key_type.check_value(value=key)
            self.value_type.check_value(value=val)

    def __repr__(self) -> str:
        """
        Returns string representation of the TypeDict.

        Returns:
            str: Representation showing key and value types.
        """

        return f"Dict[{repr(self.key_type)}, {repr(self.value_type)}]"


# Predefined Sized Integer and Float Aliases
# pylint: disable=invalid-name
Int2: TypeInt = TypeInt(num_bits=2, signed=True)
Int4: TypeInt = TypeInt(num_bits=4, signed=True)
Int8: TypeInt = TypeInt(num_bits=8, signed=True)
Int16: TypeInt = TypeInt(num_bits=16, signed=True)
Int32: TypeInt = TypeInt(num_bits=32, signed=True)
Int64: TypeInt = TypeInt(num_bits=64, signed=True)

UInt2: TypeInt = TypeInt(num_bits=2, signed=False)
UInt4: TypeInt = TypeInt(num_bits=4, signed=False)
UInt8: TypeInt = TypeInt(num_bits=8, signed=False)
UInt16: TypeInt = TypeInt(num_bits=16, signed=False)
UInt32: TypeInt = TypeInt(num_bits=32, signed=False)
UInt64: TypeInt = TypeInt(num_bits=64, signed=False)

# Predefined Sized Float Aliases
Float8_E4M3: TypeFloat = TypeFloat(num_bits=8, num_bits_exponent=4, signed=True)
Float8_E5M2: TypeFloat = TypeFloat(num_bits=8, num_bits_exponent=5, signed=True)
Float16: TypeFloat = TypeFloat(num_bits=16, num_bits_exponent=5, signed=True)
BFloat16: TypeFloat = TypeFloat(num_bits=16, num_bits_exponent=8, signed=True)
Float32: TypeFloat = TypeFloat(num_bits=32, num_bits_exponent=8, signed=True)
Float64: TypeFloat = TypeFloat(num_bits=64, num_bits_exponent=11, signed=True)


class TypeDateTime(Type):
    """
    Schema type representing a date and time.
    """

    def __init__(self) -> None:
        """
        Initializes the TypeDateTime instance.
        """

        # Initialize parent class
        super().__init__()

        # Default value is a default DateTime instance
        self.default_value: DateTime = DateTime()

    def check_value(self, value: Any) -> None:
        """
        Validates that a value is a DateTime instance.

        Args:
            value (Any): Value to validate.

        Raises:
            TypeError: If value is not a DateTime.
        """

        if not isinstance(value, DateTime):
            raise TypeError(f"Expected DateTime, got {type(value).__name__}")

    def __repr__(self) -> str:
        """
        Returns string representation of TypeDateTime.

        Returns:
            str: "DateTime".
        """

        return "DateTime"


class TypeColor(Type):
    """
    Schema type representing an RGBA color.
    """

    def __init__(self) -> None:
        """
        Initializes the TypeColor instance.
        """

        # Initialize parent class
        super().__init__()

        # Default value is black
        self.default_value: Color = Color(0, 0, 0, 255)

    def check_value(self, value: Any) -> None:
        """
        Validates that a value is a Color instance.

        Args:
            value (Any): Value to validate.

        Raises:
            TypeError: If value is not a Color.
        """

        if not isinstance(value, Color):
            raise TypeError(f"Expected Color, got {type(value).__name__}")

    def __repr__(self) -> str:
        """
        Returns string representation of TypeColor.

        Returns:
            str: "Color".
        """

        return "Color"
