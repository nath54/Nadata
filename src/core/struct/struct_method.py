"""
Struct method definition for Nadata schemas.

Represents a computed property or method declared within a struct,
including its name, parameter types, return type, and body expression.
"""

# Import Modules
from __future__ import annotations

from src.core.types.types import Type


class StructMethod:
    """
    Represents a method/computed property within a struct definition.

    Attributes:
        name (str): Method identifier name (e.g. 'age', 'is_broken').
        params (dict[str, Type]): Dictionary mapping parameter names to their Type.
        return_type (Type): Expected return type of the method.
        body_expression (str): Raw expression string executed by the method.
    """

    def __init__(
        self,
        name: str,
        params: dict[str, Type],
        return_type: Type,
        body_expression: str = ""
    ) -> None:
        """
        Initializes the StructMethod instance.

        Args:
            name (str): Identifier name of the method.
            params (dict[str, Type]): Parameters and their types.
            return_type (Type): Return type specification.
            body_expression (str, optional): Body expression. Defaults to "".
        """

        self.name: str = name
        self.params: dict[str, Type] = dict(params)
        self.return_type: Type = return_type
        self.body_expression: str = body_expression.strip()

    def get_signature_str(self) -> str:
        """
        Generates formatted method signature string.

        Returns:
            str: Signature representation (e.g. '@age(now: DateTime) -> Time').
        """

        param_strs: list[str] = [
            f"{param_name}: {repr(param_type)}"
            for param_name, param_type in self.params.items()
        ]
        params_formatted: str = ", ".join(param_strs)

        return f"@{self.name}({params_formatted}) -> {repr(self.return_type)}"

    def __repr__(self) -> str:
        """
        Returns string representation of the StructMethod.

        Returns:
            str: Method signature string.
        """

        return self.get_signature_str()
