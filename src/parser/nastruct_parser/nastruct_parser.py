"""
Parser module for .nastruct schema files.
"""

# Import Modules
from __future__ import annotations
from typing import Optional
from pathlib import Path

from src.core.database.database import NaDataDatabase
from src.parser.utils.comments_remove import CommentsRemover
from src.parser.utils.brackets import BracketsAndStrings, BracketTree
from src.parser.nastruct_parser.parser_state import NaStructParserState
from src.parser.nastruct_parser.resolver import resolve_parser_state
from src.parser.nastruct_parser.parse_structs import parse_struct, parse_struct_header
from src.parser.nastruct_parser.parse_callables import parse_fn


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
        self.parser_state: NaStructParserState = NaStructParserState(
            database=database
        )


    def resolve(self) -> NaDataDatabase:
        """
        Resolves the intermediate representation into the database.

        Converts every tmp struct and function into core objects: types are
        resolved, defaults evaluated, and everything registered.

        Returns:
            NaDataDatabase: The populated database.
        """

        return resolve_parser_state(parser_state=self.parser_state)

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

        # Resolve the intermediate representation into the database
        self.resolve()

        return self.parser_state.database


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
            header_text: str = content[cursor:bt.open_index].strip()

            # Extract the block keyword (first token of the header)
            keyword: str = header_text.split(maxsplit=1)[0] if header_text else ""

            # First case: class
            if keyword == "class":

                # Extract the struct name and optional parent from the header
                struct_name: str
                struct_parent: Optional[str]
                struct_name, struct_parent = parse_struct_header(
                    header_text=header_text
                )

                parse_struct(
                    parser_state=self.parser_state,
                    struct_name=struct_name,
                    struct_parent=struct_parent,
                    struct_start=bt.open_index + 1,
                    struct_end=bt.close_index,
                    brackets_inside=bt.inside,
                    content=content
                )

            # Second case: fn
            elif keyword == "fn":
                parse_fn(
                    parser_state=self.parser_state,
                    name=header_text,
                    fn_start=bt.open_index + 1,
                    fn_end=bt.close_index,
                    brackets_inside=bt.inside,
                    content=content
                )

            # Syntax error
            else:
                raise SyntaxError(
                    f"Expected 'class' or 'fn' block before index {bt.open_index}, "
                    f"found: '{header_text[:40]}'"
                )

            # Continue after the closing bracket for the next block
            cursor = bt.close_index + 1
