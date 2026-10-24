"""
TODO
"""

# Import Modules
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from nadata.parser.utils.brackets import BracketTree


@dataclass
class StructBodyScanState:
    """
    Mutable scanning state while parsing a struct body.

    Holds all the context of the struct being scanned: the case dispatcher
    methods read and update it, while the item-level parsers receive explicit
    per-item arguments extracted from it.

    Attributes:
        struct_name (str): Full header text of the struct being scanned.
        struct_end (int): Index of the closing brace of the struct body.
        content (str): Text of the file being parsed.
        cursor (int): Current character index in the content.
        brackets_inside (list[BracketTree]): Bracket trees of the struct body
            not yet consumed.
        next_bracket (Optional[BracketTree]): Next bracket tree to be consumed.
    """

    struct_name: str
    struct_end: int
    content: str
    cursor: int
    brackets_inside: list[BracketTree] = field(default_factory=list)
    next_bracket: Optional[BracketTree] = None

    def pop_next_bracket(self) -> Optional[BracketTree]:
        """
        Pops the next bracket tree from the waiting list.

        Returns:
            Optional[BracketTree]: Next bracket tree, or None if exhausted.
        """

        if not self.brackets_inside:
            return None

        return self.brackets_inside.pop(0)
