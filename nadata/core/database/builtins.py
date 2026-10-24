"""
Built-in print function and output formatter for Nadata.

Provides the PrintFunction callable for execution scopes, formatting
primitives, Value wrappers, StructInstances, and QuerySets cleanly.
"""

# Import Modules
from __future__ import annotations
from typing import Any, Optional
import sys


from nadata.core.values.values import Value


class PrintFunction:
    """
    Built-in print implementation for Nadata runtime scopes.

    Attributes:
        output_stream (Any): Output stream (defaults to sys.stdout).
        captured_output (list[str]): Optional list capturing printed lines for testing.
    """

    def __init__(
        self,
        output_stream: Optional[Any] = None,
        capture: bool = False
    ) -> None:
        """
        Initializes the PrintFunction.

        Args:
            output_stream (Optional[Any], optional): Target stream. Defaults to None (sys.stdout).
            capture (bool, optional): Whether to capture lines in memory. Defaults to False.
        """

        self.output_stream: Any = output_stream if output_stream is not None else sys.stdout
        self.capture: bool = capture
        self.captured_output: list[str] = []

    def format_item(self, item: Any) -> str:
        """
        Formats a single item for readable printing.

        Args:
            item (Any): Item to format.

        Returns:
            str: Human-readable string.
        """

        if isinstance(item, Value):
            raw = item.get()

            if isinstance(raw, str):
                return raw

            return str(raw)

        if hasattr(item, "__iter__") and not isinstance(item, (str, bytes)):
            formatted_elements: list[str] = [self.format_item(x) for x in item]
            return f"[{', '.join(formatted_elements)}]"

        return str(item)

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        """
        Executes the print operation.

        Args:
            args (Any): Positional arguments to print.
            kwargs (Any): Keyword arguments (sep, end).
        """

        sep: str = kwargs.get("sep", " ")
        end: str = kwargs.get("end", "\n")

        formatted_args: list[str] = [self.format_item(arg) for arg in args]
        line: str = sep.join(formatted_args)

        if self.capture:
            self.captured_output.append(line)

        self.output_stream.write(line + end)
        self.output_stream.flush()
