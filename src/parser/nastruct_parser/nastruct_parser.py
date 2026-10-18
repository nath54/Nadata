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
from src.parser.utils.comments_remove import CommentsRemover
from src.parser.utils.brackets import BracketsAndStrings, BracketTree


class NaStructParser:
    """
    Parser for .nastruct schema files.
    """


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


    def parse_file(self, content: str) -> None:
        """
        Parses source content block by block for classes and standalone functions.

        Args:
            content (str): Full text content of the .nastruct file.

        Raises:
            SyntaxError: If a block header or matching closing brace is not found.
        """

        # Remove all the comments present in the file to avoid issues with the rest of the parsing
        content = CommentsRemover.remove_all_comments(content=content)

        # Find all the brackets and string couples
        bracket_trees: list[BracketTree]
        bracket_trees, _strings_couples = BracketsAndStrings.process(content=content)

        # Cursor tracking the end of the last processed block
        cursor: int = 0

        bt: BracketTree

        # For each top-level bracket tree, in file order
        for bt in bracket_trees:

            # Get the text between the previous block and this opening bracket
            header_text: str = content[cursor:bt.start].strip()

            # Extract the block keyword (first token of the header)
            keyword: str = header_text.split(maxsplit=1)[0] if header_text else ""

            # First case: class
            if keyword == "class":
                self.parse_struct(
                    name=header_text,
                    start=bt.start + 1,
                    end=bt.end,
                    brackets_inside=bt.inside,
                    content=content
                )

            # Second case: fn
            elif keyword == "fn":
                self.parse_fn(
                    name=header_text,
                    start=bt.start + 1,
                    end=bt.end,
                    brackets_inside=bt.inside,
                    content=content
                )

            # Syntax error
            else:
                raise SyntaxError(
                    f"Expected 'class' or 'fn' block before index {bt.start}, "
                    f"found: '{header_text[:40]}'"
                )

            # Continue after the closing bracket for the next block
            cursor = bt.end + 1


    def parse_struct(
        self,
        name: str,
        start: int,
        end: int,
        brackets_inside: list[BracketTree],
        content: str
    ) -> None:
        """
        Parses a struct definition block.

        Args:
            name (str): Full header text of the struct (e.g. 'class Human(Entity)').
            start (int): Index of the first content character inside the braces.
            end (int): Index of the closing brace of the block.
            brackets_inside (list[BracketTree]): Nested bracket trees inside the block.
            content (str): text of the file
        """

        # Start of the struct code
        cursor: int = start

        # Prepare for the next bracket
        next_bracket: Optional[BracketTree] = (
            brackets_inside.pop(0) if len(brackets_inside) > 0 else None
        )

        # While we have something in the struct brackets
        while cursor < end:

            # Manage white spaces (spaces, tabs, newlines, and blanked comments)
            if content[cursor].isspace():
                cursor += 1
                continue

            # Class method
            if content[cursor] == "@":

                # Check bracket
                if next_bracket is None:
                    raise SyntaxError(
                        f"Method declaration at index {cursor} has no body '{{...}}'"
                    )

                # Get text before (the full method header: name, params, return type)
                method_header: str = content[cursor + 1:next_bracket.start].strip()

                # Check the text is present
                if not method_header:
                    raise SyntaxError(f"Empty method declaration at index {cursor}")

                # A method declaration requires a parameter list '(...)'
                paren_index: int = method_header.find("(")

                # If there are no parenthesis, it is a syntax error
                if paren_index == -1:
                    raise SyntaxError(
                        f"Method declaration at index {cursor} is missing its "
                        f"parameter list '(...)': '@{method_header[:20]}...'"
                    )

                # The method name is the text before the parameter list
                method_name: str = method_header[:paren_index].strip()

                # Check the method name is present
                if not method_name:
                    raise SyntaxError(f"Missing method name at index {cursor}")

                # Parse the struct method
                self.parse_struct_method(
                    struct_name=name,
                    struct_method_name=method_name,
                    header=method_header,
                    start=next_bracket.start,
                    end=next_bracket.end,
                    content=content,
                    brackets_inside=next_bracket.inside
                )

                # Advance the cursor
                cursor = next_bracket.end + 1

                # Advance the next bracket
                next_bracket = (
                    brackets_inside.pop(0) if len(brackets_inside) > 0 else None
                )

            # Else, it is an attribut definition
            else:

                # Find the closing ; within the struct body only
                i: int = content.find(";", cursor, end)

                # Check closing ;
                if i == -1:
                    raise SyntaxError(
                        f"Missing closing ';' for attribute starting at index {cursor}"
                    )

                bracket_inside: Optional[BracketTree] = None

                # Check next bracket
                if next_bracket is not None and next_bracket.start < i:

                    # Mark the bracket as inside the attribute declaration
                    bracket_inside = next_bracket

                    # Advance the next bracket
                    next_bracket = (
                        brackets_inside.pop(0) if len(brackets_inside) > 0 else None
                    )

                # Process the attribute
                self.parse_struct_attribute(
                    struct_name=name,
                    start=cursor,
                    end=i,
                    content=content,
                    bracket_inside=bracket_inside
                )

                # Advance the cursor
                cursor = i + 1


    def parse_struct_attribute(
        self,
        struct_name: str,
        start: int,
        end: int,
        content: str,
        bracket_inside: Optional[BracketTree]
    ) -> None:
        """
        Parses a single struct attribute declaration.

        Args:
            struct_name (str): Name of the struct owning the attribute.
            start (int): Index of the first character of the declaration.
            end (int): Index of the closing ';' of the declaration.
            content (str): Text of the file.
            bracket_inside (Optional[BracketTree]): Bracket tree contained in the
                declaration (e.g. '{} ' or '[]' default value), None if none.
        """

        pass


    def parse_struct_method(
        self,
        struct_name: str,
        struct_method_name: str,
        header: str,
        start: int,
        end: int,
        content: str,
        brackets_inside: list[BracketTree]
    ) -> None:
        """
        Parses a struct method definition block.

        Args:
            struct_name (str): Name of the struct owning the method.
            struct_method_name (str): Name of the method (without '@').
            header (str): Full header text between '@' and the opening brace
                (e.g. 'age(now: DateTime) -> DateTime').
            start (int): Index of the first content character inside the braces.
            end (int): Index of the closing brace of the method body.
            content (str): Text of the file.
            brackets_inside (list[BracketTree]): Nested bracket trees in the body.
        """

        pass

    def parse_fn(
        self,
        name: str,
        start: int,
        end: int,
        brackets_inside: list[BracketTree],
        content: str
    ) -> None:
        """
        Parses a standalone function definition block.

        Args:
            name (str): Full header text of the function
                (e.g. 'fn is_adult(h: Human) -> Bool').
            start (int): Index of the first content character inside the braces.
            end (int): Index of the closing brace of the block.
            brackets_inside (list[BracketTree]): Nested bracket trees inside the block.
            content (str): text of the file
        """

        pass
