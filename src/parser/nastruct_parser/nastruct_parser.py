"""
Parser module for .nastruct schema files.
"""

# Import Modules
from __future__ import annotations
from typing import Optional, Pattern
from pathlib import Path

import re

from src.core.struct.struct import Struct
from src.core.database.database import NaDataDatabase
from src.core.struct.struct_method import StructMethod


class NaStructParser:
    """
    Parser for .nastruct schema files.
    """

    # Detailed breakdown of TOP_LEVEL_BLOCK_PATTERN regex:
    # 1. (?:\s+|#[^\n]*\n)* : Non-capturing group matching any leading whitespace
    #                         or comments (starting with '#' until end of line '\n'),
    #                         repeated 0 or more times.
    # 2. (class\s+[^{]+|fn\s+[^{]+) : Capturing Group 1 (SOMETHING1). Matches either:
    #                                 - 'class' followed by whitespace and everything up to '{'
    #                                 - 'fn' followed by whitespace and everything up to '{'
    # 3. \s*\{              : Matches optional whitespace followed by the opening brace '{'.
    TOP_LEVEL_BLOCK_PATTERN: Pattern[str] = re.compile(
        r"(?:\s+|#[^\n]*\n)*(class\s+[^{]+|fn\s+[^{]+)\s*\{"
    )

    def __init__(
        self,
        database: Optional[NaDataDatabase] = None
    ) -> None:
        """
        Init function.
        """

        # Database where we will register the structs and functions
        self.database: NaDataDatabase = (
            database if database is not None else NaDataDatabase()
        )

        # Keys are dependencies we still are waiting for, and values the list of items needing it
        self.unresolved_dependencies: dict[str, list[str]] = {}

        self.waiting_structs: dict[str, Struct] = {}
        self.waiting_structs_deps: dict[str, set[tuple[str, str]]] = {}
        # Key is struct name, value is list of (dep name, dep type (struct or fn))

        self.waiting_fns: dict[str, StructMethod] = {}
        self.waiting_fns_deps: dict[str, set[tuple[str, str]]] = {}

    def parse_folder(self, folder_path: str) -> NaDataDatabase:
        """
        Parses all .nastruct files found in the specified folder into the database.

        Args:
            folder_path (str): Path to directory containing .nastruct files.

        Returns:
            NaDataDatabase: Populated database instance.
        """

        # Locate all .nastruct files in folder
        all_structs_files: list[str] = self.find_all_nastruct_files(
            folder_path=folder_path
        )

        # Parse each file
        for file_path in all_structs_files:
            with open(file_path, "r", encoding="utf-8") as file:
                self.parse_file(content=file.read())

        return self.database

    def find_all_nastruct_files(self, folder_path: str) -> list[str]:
        """
        Finds all .nastruct files recursively in a folder.

        Args:
            folder_path (str): Root folder path to search.

        Returns:
            list[str]: List of absolute file paths.
        """

        return [
            str(file.absolute())
            for file in Path(folder_path).rglob("*.nastruct")
        ]

    def _is_only_whitespace_or_comments(self, text: str) -> bool:
        """
        Checks whether the given text contains only whitespace and comments.

        Args:
            text (str): Suffix text of the file.

        Returns:
            bool: True if there is no remaining code, False otherwise.
        """

        # Check line by line for any non-comment code
        for line in text.splitlines():
            stripped: str = line.strip()

            if stripped and not stripped.startswith("#"):
                return False

        return True

    def _match_next_block(
        self,
        content: str,
        pos: int
    ) -> re.Match[str]:
        """
        Matches the next top-level block header starting at pos.

        Args:
            content (str): Full text content.
            pos (int): Current cursor position.

        Returns:
            re.Match[str]: The regex match for the block header.

        Raises:
            SyntaxError: If no class or fn block header is found at pos.
        """

        # Attempt to match block pattern at current position
        match: Optional[re.Match[str]] = self.TOP_LEVEL_BLOCK_PATTERN.match(
            string=content,
            pos=pos
        )

        # Validate that a block header was matched
        if match is None:
            preview: str = content[pos:pos + 40].replace("\n", " ")
            raise SyntaxError(
                f"Expected 'class' or 'fn' block at index {pos}, found: '{preview}...'"
            )

        return match

    def _find_matching_bracket(
        self,
        content: str,
        open_pos: int
    ) -> int:
        """
        Finds the index of the matching closing curly brace.

        Args:
            content (str): Full text content.
            open_pos (int): Index of the opening curly brace '{'.

        Returns:
            int: Index of the matching closing curly brace '}'.

        Raises:
            SyntaxError: If the matching closing brace is not found before EOF.
        """

        # Depth counter tracking nested curly braces
        bracket_depth: int = 1

        cursor: int = open_pos + 1
        eof: int = len(content)

        # Scan characters until all opened brackets are closed
        while cursor < eof and bracket_depth > 0:
            char: str = content[cursor]

            if char == "{":
                bracket_depth += 1
            elif char == "}":
                bracket_depth -= 1

            cursor += 1

        # Check if closing brace was found
        if bracket_depth != 0:
            raise SyntaxError(
                f"Missing closing bracket '}}' for block starting at index {open_pos}"
            )

        return cursor - 1

    def _dispatch_block(
        self,
        header: str,
        body: str
    ) -> None:
        """
        Dispatches a parsed block to either parse_struct or parse_fn based on its header.

        Args:
            header (str): Header text of the block.
            body (str): Body content within the braces.
        """

        # Dispatch to struct or function parser
        if header.startswith("class"):
            self.parse_struct(name=header, content=body)
        elif header.startswith("fn"):
            self.parse_fn(name=header, content=body)

    def parse_file(self, content: str) -> None:
        """
        Parses source content block by block for classes and standalone functions.

        Args:
            content (str): Full text content of the .nastruct file.

        Raises:
            SyntaxError: If a block header or matching closing brace is not found.
        """

        # Cursor tracking character position in the file content
        i: int = 0
        eof: int = len(content)

        # Iterate over all top-level blocks in the file
        while i < eof:

            # Stop if only whitespace or comments remain
            if self._is_only_whitespace_or_comments(text=content[i:]):
                break

            # Find next block header match
            match: re.Match[str] = self._match_next_block(
                content=content,
                pos=i
            )

            # Extract header and brace positions
            header: str = match.group(1).strip()
            open_pos: int = match.end() - 1
            close_pos: int = self._find_matching_bracket(
                content=content,
                open_pos=open_pos
            )

            # Extract body content and dispatch
            body: str = content[open_pos + 1:close_pos]
            self._dispatch_block(header=header, body=body)

            # Advance cursor past the closing brace
            i = close_pos + 1

    def parse_struct(self, name: str, content: str) -> None:
        """
        Parses a struct definition block.

        Args:
            name (str): Header of the struct.
            content (str): Body content within braces.
        """

        _ = (name, content)

    def parse_fn(self, name: str, content: str) -> None:
        """
        Parses a standalone function definition block.

        Args:
            name (str): Header of the function.
            content (str): Body content within braces.
        """

        _ = (name, content)
