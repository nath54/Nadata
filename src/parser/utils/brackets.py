"""
Bracket tree extraction for Nadata source files.

Provides the BracketTree structure and the BracketsAndStrings scanner which
locate '{' / '}' bracket couples and string literal spans in a single pass.
"""

# Import Modules
from __future__ import annotations
from typing import Optional
import re


class BracketTree:
    """
    Represents a '{' / '}' couple with its nested children bracket trees.

    Attributes:
        open_index (int): Index of the opening bracket '{' in the content.
        close_index (int): Index of the matching closing bracket '}' (-1 if unclosed).
        inside (list[BracketTree]): Nested bracket trees contained in this one.
    """

    def __init__(
        self,
        open_index: int,
        close_index: int = -1,
        inside: Optional[list[BracketTree]] = None
    ) -> None:
        """
        Initializes the BracketTree instance.

        Args:
            open_index (int): Index of the opening bracket.
            close_index (int, optional): Index of the closing bracket. Defaults to -1.
            inside (Optional[list[BracketTree]], optional): Nested trees. Defaults to None.
        """

        self.open_index: int = open_index
        self.close_index: int = close_index
        self.inside: list[BracketTree] = [] if inside is None else inside


class BracketsAndStrings:
    """
    Extracts '{' / '}' bracket trees and string literal spans from source content.

    Brackets appearing inside string literals are ignored. Escaped quotes
    ('\\\"') inside strings are honored. Comments are expected to have been
    removed beforehand (see CommentsRemover).
    """

    # Characters relevant to the scan: brace and quote delimiters
    RELEVANT_CHARS: str = "{}\""

    # Escape character used inside string literals
    ESCAPE_CHAR: str = "\\"

    # Pre-compiled scanner pattern for maximum speed
    _PATTERN: re.Pattern[str] = re.compile("[{}\"]")

    @staticmethod
    def find_unnested(
        content: str,
        target_chars: str,
        start: int,
        end: int
    ) -> int:
        """
        Finds the first occurrence of any target char at bracket/string depth zero.

        Skips string literals (with escapes) and the inside of '(', '[' and '{'
        brackets, so separators nested in calls, arrays, dicts, or strings are
        ignored.

        Args:
            content (str): Comment-free source content to scan.
            target_chars (str): Characters to search for.
            start (int): Index to start searching from (inclusive).
            end (int): Index to stop searching at (exclusive).

        Returns:
            int: Index of the first target char at depth zero, or -1 if none.

        Raises:
            SyntaxError: If an unexpected closing bracket is encountered.
        """

        # Depth stack of currently open brackets
        depth_stack: list[str] = []

        cursor: int = start

        # Single left-to-right scan tracking strings and bracket depth
        while cursor < end:
            char: str = content[cursor]

            # Skip string literals (with escapes)
            if char == "\"":
                cursor += 1

                while cursor < end:
                    if content[cursor] == "\\":
                        cursor += 2
                        continue

                    if content[cursor] == "\"":
                        break

                    cursor += 1

                cursor += 1
                continue

            # Track bracket nesting
            if char in "([{":
                depth_stack.append(char)
            elif char in ")]}" and depth_stack:
                depth_stack.pop()
            elif char in ")]}" and not depth_stack:
                raise SyntaxError(
                    f"Unexpected closing bracket '{char}' at index {cursor}"
                )

            # Target char found at depth zero
            elif char in target_chars and not depth_stack:
                return cursor

            cursor += 1

        return -1

    @staticmethod
    def process(content: str) -> tuple[list[BracketTree], dict[int, int]]:
        """
        Scans the content and extracts top-level bracket trees and string couples.

        Args:
            content (str): Comment-free source content to scan.

        Returns:
            tuple[list[BracketTree], dict[int, int]]:
                - List of top-level (non-nested) BracketTree instances.
                - Dictionary mapping each opening quote index to its closing quote index.

        Raises:
            SyntaxError: If a closing bracket has no opener, a bracket is
                unclosed, or a string literal is unterminated.
        """

        # String couples registry: opening quote index -> closing quote index
        string_couples: dict[int, int] = {}

        # Waiting stack of currently open bracket trees
        waiting_queue: list[BracketTree] = []

        # Completed top-level bracket trees
        larger_trees: list[BracketTree] = []

        # Extract relevant character indices efficiently using finditer
        indices: list[int] = [
            match.start() for match in BracketsAndStrings._PATTERN.finditer(content)
        ]

        # Indicate if we are inside a string or not
        in_string: bool = False
        last_string: int = -1

        # Scan all relevant characters in a single pass
        for i in indices:
            char: str = content[i]

            # String delimiter: toggle the string state unless escaped
            if char == "\"":

                # Skip escaped quotes inside strings (odd backslash count before)
                if in_string and BracketsAndStrings._is_escaped(content=content, index=i):
                    continue

                # Here, we start the string
                if not in_string:
                    in_string = True
                    last_string = i

                # Here, we close the string
                else:
                    in_string = False
                    string_couples[last_string] = i

                continue

            # Brackets inside strings are plain characters, not block delimiters
            if in_string:
                continue

            # Opening bracket: push on the waiting stack
            if char == "{":
                waiting_queue.append(BracketTree(open_index=i))
                continue

            # Closing bracket: pop the matching opener
            if not waiting_queue:
                raise SyntaxError(f"Unexpected closing bracket '}}' at index {i}")

            bracket_tree: BracketTree = waiting_queue.pop()
            bracket_tree.close_index = i

            # This is a top-level bracket, no one is parenting it
            if not waiting_queue:
                larger_trees.append(bracket_tree)

            # Else, adding this tree to its parent tree children
            else:
                waiting_queue[-1].inside.append(bracket_tree)

        # Check for unterminated string literal
        if in_string:
            raise SyntaxError(
                f"Unterminated string literal starting at index {last_string}"
            )

        # Check for unclosed brackets
        if waiting_queue:
            raise SyntaxError(
                f"Missing closing bracket '}}' for bracket at index {waiting_queue[0].open_index}"
            )

        return larger_trees, string_couples

    @staticmethod
    def _is_escaped(content: str, index: int) -> bool:
        """
        Determines if the quote at the given index is escaped by backslashes.

        Args:
            content (str): Full source content.
            index (int): Index of the quote character.

        Returns:
            bool: True if the quote is preceded by an odd number of backslashes.
        """

        # Count consecutive backslashes immediately before the quote
        backslash_count: int = 0
        cursor: int = index - 1

        while cursor >= 0 and content[cursor] == BracketsAndStrings.ESCAPE_CHAR:
            backslash_count += 1
            cursor -= 1

        # Odd count means the quote is escaped
        return backslash_count % 2 == 1
