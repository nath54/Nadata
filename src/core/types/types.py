"""
TODO
"""

# Import Modules
from typing import Any


class Type:
    """
    TODO: Abstract Type class
    """

    def __init__(self) -> None:
        """
        Init function
        """

        # Default value of the type
        self.default_value: Any = None


class TypeStr(Type):
    """
    TODO
    """

    def __init__(self) -> None:
        """
        Init function
        """

        # Init parent function
        super().__init__()

        # Default value of the type
        self.default_value: str = ""


class TypeInt(Type):
    """
    TODO
    """

    def __init__(
        self,
        num_bits: int = 32,
        signed: bool = True
    ) -> None:
        """
        Init function
        """

        # Init parent function
        super().__init__()

        # Number of bits
        self.num_bits: int = num_bits

        # Is signed
        self.signed: bool = signed

        # Default value of the type
        self.default_value: int = 0


class TypeFloat(Type):
    """
    TODO
    """

    def __init__(
        self,
        num_bits: int = 32,
        num_bits_exponent: int = 8,
        num_bits_mantissa: int = 23,
        signed: bool = True
    ) -> None:
        """
        Init function
        """

        # Init parent function
        super().__init__()

        # Number of bits
        self.num_bits: int = num_bits
        self.num_bits_exponent: int = num_bits_exponent
        self.num_bits_mantissa: int = num_bits_mantissa

        # Is signed
        self.signed: bool = signed

        # Default value of the type
        self.default_value: float = 0


class TypeBool(Type):
    """
    TODO
    """

    def __init__(self) -> None:
        """
        Init function
        """

        # Init parent function
        super().__init__()

        # Default value of the type
        self.default_value: bool = False


class TypeArray(Type):
    """
    TODO
    """

    def __init__(self, value_type: Type) -> None:
        """
        Init function
        """

        # Init parent function
        super().__init__()

        # Type of the arrays values
        self.value_type: Type = value_type

        # Default value of the type
        self.default_value: list[Any] = []


class TypeDict(Type):
    """
    TODO
    """

    def __init__(self, value_type: Type, key_type: Type = TypeStr()) -> None:
        """
        Init function
        """

        # Init parent function
        super().__init__()

        # Type of the dict keys
        self.key_type: Type = key_type

        # Type of the dict values
        self.value_type: Type = value_type

        # Default value of the type
        self.default_value: dict[Any, Any] = {}


# TODO: implement all the other class types. Maybe put each type in its own python file ?
