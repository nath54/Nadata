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

        # Find all the brackets and strings couples
        brackets_couples: dict[int, int]
        bracket_trees: list[BracketTree]
        bracket_trees, strings_couples = BracketsAndStrings.process(content=content)

        i: int = 0
        bt: BracketTree

        # For each bracket tree
        for bt in bracket_trees:

            # Get the text before the opening bracket
            txt_before: str = content[bt.start:bt.end]

            # Split the text before the brackets to detect its class
            txtlst: list[str] = txt_before.strip().split(" ")

            # First case: class
            if len(txtlst) == 2:

                # Check if keyword class is present
                if txtlst[0] == "class":
                    self.parse_struct(
                        name=txtlst[1],
                        start=bt.start,
                        end=bt.end,
                        brackets_inside=bt.inside
                    )

                # Check if keyword fn is present
                elif txtlst[0] == "fn":
                    self.parse_fn(
                        name=txtlst[1],
                        start=bt.start,
                        end=bt.end,
                        brackets_inside=bt.inside
                    )

                # Syntax error
                else:
                    return SyntaxError("TODO")

            # Syntax error
            else:
                return SyntaxError("TODO")


            # Continue after for the next block
            i = bt.end + 1

        pass


    def parse_struct(
        self,
        name: str,
        start: int,
        end: int,
        brackets_inside: list[BracketTree]
    ) -> None:
        """
        Parses a struct definition block.

        Args:
            name (str): Name of the struct.
            start (int): index of the content start
            end (int): index of the content end
            brackets_inside (list[BracketTree]): brackets inside the struct
        """

        pass


    def parse_fn(
        self,
        name: str,
        start: int,
        end: int,
        brackets_inside: list[BracketTree]
    ) -> None:
        """
        Parses a standalone function definition block.

        Args:
            name (str): Name of the function.
            start (int): index of the content start
            end (int): index of the content end
            brackets_inside (list[BracketTree]): brackets inside the fn
        """

        pass
