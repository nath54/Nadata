"""
EntityId representation and handle encoding for Nadata.

Provides the 64-bit globally unique EntityId encoding the 16-bit TypeId
in the higher bits and the 48-bit instance index in the lower bits.
"""

# Import Modules
from __future__ import annotations
from typing import Any


class EntityId:
    """
    Globally unique 64-bit entity handle.

    Encodes:
        Bits 48-63: type_id (16-bit struct type index, up to 65,536 structs).
        Bits 0-47:  local_id (48-bit instance index, up to 281 trillion instances).

    Attributes:
        value (int): Raw unsigned 64-bit integer.
    """

    # Masks for bit manipulation
    TYPE_ID_SHIFT: int = 48
    LOCAL_ID_MASK: int = 0x0000FFFFFFFFFFFF
    MAX_TYPE_ID: int = 0xFFFF
    MAX_LOCAL_ID: int = 0x0000FFFFFFFFFFFF

    def __init__(self, value: int) -> None:
        """
        Initializes the EntityId instance.

        Args:
            value (int): Unsigned 64-bit integer value.

        Raises:
            ValueError: If value is negative or exceeds 64-bit bounds.
        """

        if not 0 <= value <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError(f"EntityId must be an unsigned 64-bit integer, got {value}")

        self.value: int = value

    @classmethod
    def from_parts(cls, type_id: int, local_id: int) -> EntityId:
        """
        Creates an EntityId from type_id and local_id parts.

        Args:
            type_id (int): 16-bit struct type index.
            local_id (int): 48-bit local instance index.

        Returns:
            EntityId: Encoded EntityId instance.

        Raises:
            ValueError: If type_id or local_id exceed their respective bit widths.
        """

        if not 0 <= type_id <= cls.MAX_TYPE_ID:
            raise ValueError(f"type_id must be in range 0-65535, got {type_id}")

        if not 0 <= local_id <= cls.MAX_LOCAL_ID:
            raise ValueError(f"local_id must be in range 0-281474976710655, got {local_id}")

        combined: int = (type_id << cls.TYPE_ID_SHIFT) | local_id

        return cls(value=combined)

    @property
    def type_id(self) -> int:
        """
        Extracts the 16-bit type index.

        Returns:
            int: Struct type index.
        """

        return (self.value >> self.TYPE_ID_SHIFT) & self.MAX_TYPE_ID

    @property
    def local_id(self) -> int:
        """
        Extracts the 48-bit local instance index.

        Returns:
            int: Local instance index.
        """

        return self.value & self.LOCAL_ID_MASK

    def __int__(self) -> int:
        """
        Returns integer representation of the EntityId.

        Returns:
            int: Raw 64-bit integer.
        """

        return self.value

    def __eq__(self, other: Any) -> bool:
        """
        Checks equality with another EntityId or integer.

        Args:
            other (Any): Object to compare.

        Returns:
            bool: True if equal, False otherwise.
        """

        if isinstance(other, EntityId):
            return self.value == other.value

        if isinstance(other, int):
            return self.value == other

        return False

    def __hash__(self) -> int:
        """
        Returns hash for use in dictionaries and sets.

        Returns:
            int: Integer hash.
        """

        return hash(self.value)

    def __repr__(self) -> str:
        """
        Returns string representation of EntityId.

        Returns:
            str: Hexadecimal representation showing parts.
        """

        return f"EntityId(type={self.type_id}, local={self.local_id}, raw=0x{self.value:016X})"
