"""
Color representations for Nadata.

Provides the Color class representing an RGBA color model with values in 0-255,
and utilities for hex, tuple, and bit conversions.
"""

# Import Modules
from __future__ import annotations
from typing import Any


class Color:
    """
    Represents an RGBA color in Nadata.

    Attributes:
        r (int): Red channel (0-255).
        g (int): Green channel (0-255).
        b (int): Blue channel (0-255).
        a (int): Alpha channel (0-255).
    """

    def __init__(
        self,
        r: int = 0,
        g: int = 0,
        b: int = 0,
        a: int = 255
    ) -> None:
        """
        Initializes the Color instance.

        Args:
            r (int, optional): Red channel (0-255). Defaults to 0.
            g (int, optional): Green channel (0-255). Defaults to 0.
            b (int, optional): Blue channel (0-255). Defaults to 0.
            a (int, optional): Alpha channel (0-255). Defaults to 255.

        Raises:
            ValueError: If any channel is out of the 0-255 range.
        """

        # Check ranges
        for name, val in (("r", r), ("g", g), ("b", b), ("a", a)):
            if not 0 <= val <= 255:
                raise ValueError(f"Color channel '{name}' must be in range 0-255, got {val}")

        self.r: int = r
        self.g: int = g
        self.b: int = b
        self.a: int = a


    def to_rgba(self) -> tuple[int, int, int, int]:
        """
        Returns color channels as an RGBA tuple.

        Returns:
            tuple[int, int, int, int]: (r, g, b, a) tuple.
        """

        return (self.r, self.g, self.b, self.a)

    def to_hex(self) -> str:
        """
        Returns the color formatted as a hex string '#RRGGBBAA'.

        Returns:
            str: Hexadecimal string representation.
        """

        return f"#{self.r:02X}{self.g:02X}{self.b:02X}{self.a:02X}"

    def __eq__(self, other: Any) -> bool:
        """
        Checks equality with another Color.

        Args:
            other (Any): Object to compare with.

        Returns:
            bool: True if all channels match.
        """

        if not isinstance(other, Color):
            return False

        return self.to_rgba() == other.to_rgba()

    def __repr__(self) -> str:
        """
        Returns string representation of the Color.

        Returns:
            str: Color constructor string.
        """

        if self.a == 255:
            return f"Color({self.r}, {self.g}, {self.b})"

        return f"Color({self.r}, {self.g}, {self.b}, {self.a})"
