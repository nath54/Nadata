"""
TODO
"""

from src.core.types.types import Type

class Value:
    """
    TODO: abstract class
    """

    def __init__(self, value_type: Type) -> None:
        """
        Init function.
        """

        # Store the value type here
        self.value_type: Type = value_type


class ValueInt(Value):
    """
    TODO
    """

    def __init__(self, value_type: Type, value: int = 0) -> None:
        """
        Init function.
        """

        # Init parent
        super().__init__(value_type=value_type)

        # Store value here
        # TODO: is it better to store here in native python int, or in BitArray ?
        self.value: int = value


class ValueFloat(Value):
    """
    TODO
    """

    def __init__(self, value_type: Type, value: float = 0) -> None:
        """
        Init function.
        """

        # Init parent
        super().__init__(value_type=value_type)

        # Store value here
        # TODO: is it better to store here in native python float, or in BitArray ?
        self.value: float = value

# TODO: do all the other types. Maybe implement each type in its own python file.
