"""
Shared parser state for .nastruct schema parsing.

Provides the NaStructParserState class holding all cross-file parsing
context: the target database, the intermediate representation being built,
and the dependency bookkeeping used by the resolution phase.
"""

# Import Modules
from __future__ import annotations
from typing import Any, Optional

from src.core.database.database import NaDataDatabase
from src.parser.nastruct_parser.tmp_structs import (
    TmpStruct,
    TmpFunction,
)


class NaStructParserState:
    """
    Shared state of a .nastruct parsing session.

    Holds everything the parsing functions need across files and items:
    the database under construction and the intermediate representation
    of the structs and functions discovered so far.

    Attributes:
        database (NaDataDatabase): Database being populated by the parser.
        tmp_structs (dict[str, TmpStruct]): Intermediate struct declarations
            by name, awaiting type resolution.
        tmp_functions (dict[str, TmpFunction]): Intermediate standalone
            function declarations by name, awaiting type resolution.
    """

    def __init__(
        self,
        database: Optional[NaDataDatabase] = None
    ) -> None:
        """
        Initializes the NaStructParserState instance.

        Args:
            database (Optional[NaDataDatabase], optional): Database to populate.
                Defaults to None (a fresh NaDataDatabase is created).
        """

        # Database where we will register the structs and functions
        self.database: NaDataDatabase = (
            database if database is not None else NaDataDatabase()
        )

        # Intermediate representation awaiting resolution
        self.tmp_structs: dict[str, TmpStruct] = {}
        self.tmp_functions: dict[str, TmpFunction] = {}

        # When True (session schema re-merge), re-declared structs /
        # attributes / methods / functions replace instead of raising
        self.merge: bool = False

        # Structs re-declared in the current merge -> member names that
        # the re-declarations explicitly carry (union across parses);
        # members absent from the set were removed from the schema
        self.redeclared_members: dict[str, set[str]] = {}

    def to_ir_dict(self) -> dict[str, Any]:
        """
        Converts the intermediate representation to a plain JSON serializable dict.

        Returns:
            dict[str, Any]: Dict with 'structs' and 'functions' keys mapping
                names to their serialized representations.
        """

        return {
            "structs": {
                name: tmp_struct.to_dict()
                for name, tmp_struct in self.tmp_structs.items()
            },
            "functions": {
                name: tmp_function.to_dict()
                for name, tmp_function in self.tmp_functions.items()
            },
        }
