"""
Debug inspector for .nastruct parsing sessions.

Parses a folder of .nastruct files and pretty-prints the intermediate
representation (TmpStruct / TmpFunction) produced by the scan phase, before
the resolution phase registers anything in the database.

Usage:
    python -m scripts.inspect_parse <folder_path> [--json]
"""

# Import Modules
from __future__ import annotations
import json
import sys

from src.parser.nastruct_parser.nastruct_parser import NaStructParser
from src.parser.nastruct_parser.tmp_structs import TmpAttribute, TmpFunction, TmpStruct


def format_attribute(
    attribute: TmpAttribute,
    include_dependencies: bool = True
) -> str:
    """
    Formats a single attribute / parameter declaration.

    Args:
        attribute (TmpAttribute): Attribute to format.
        include_dependencies (bool, optional): Whether to append the
            dependency comment. Defaults to True.

    Returns:
        str: Formatted declaration line (e.g. 'age: UInt8 = 18').
    """

    line: str = f"{attribute.attribute_name}: {attribute.attribute_type}"

    # Append the raw default value when present
    if attribute.attribute_default_value is not None:
        line += f" = {attribute.attribute_default_value}"

    # Append the unresolved dependency identifiers
    if include_dependencies and attribute.dependencies:
        line += f"    # deps: {', '.join(sorted(attribute.dependencies))}"

    return line


def format_callable(fn: TmpFunction) -> list[str]:
    """
    Formats a method or standalone function declaration.

    Args:
        fn (TmpFunction): Callable to format.

    Returns:
        list[str]: Formatted lines (signature, then indented body).
    """

    # Build the parameter list with types and defaults (no per-param deps)
    params: list[str] = [
        format_attribute(attribute=attribute, include_dependencies=False)
        for attribute in fn.fn_argument.values()
    ]

    signature: str = f"{fn.fn_name}({', '.join(params)}) -> {fn.fn_output_type}"

    if fn.dependencies:
        signature += f"    # deps: {', '.join(sorted(fn.dependencies))}"

    # Indent the statement reprs for readability
    body_lines: list[str] = [
        f"        {repr(statement)}" for statement in fn.fn_body
    ]

    return [signature] + body_lines


def print_struct(tmp_struct: TmpStruct) -> None:
    """
    Prints a single struct intermediate representation.

    Args:
        tmp_struct (TmpStruct): Struct to print.
    """

    parent: str = tmp_struct.struct_parent if tmp_struct.struct_parent else "-"

    print(f"struct {tmp_struct.struct_name} (parent: {parent})")

    # Attributes
    if tmp_struct.attributes:
        print("  attributes:")

        for attribute in tmp_struct.attributes.values():
            print(f"    {format_attribute(attribute=attribute)}")

    # Methods
    if tmp_struct.methods:
        print("  methods:")

        for fn in tmp_struct.methods.values():
            print("    @method")

            for line in format_callable(fn=fn):
                print(f"      {line}")

    print()


def print_function(fn: TmpFunction) -> None:
    """
    Prints a single standalone function intermediate representation.

    Args:
        fn (TmpFunction): Function to print.
    """

    print("fn")
    for line in format_callable(fn=fn):
        print(f"  {line}")

    print()


def main() -> None:
    """
    Parses the folder given as argument and prints the intermediate data.
    """

    # Check the folder argument is present
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.inspect_parse <folder_path> [--json]")
        sys.exit(1)

    folder_path: str = sys.argv[1]
    as_json: bool = "--json" in sys.argv[2:]

    parser: NaStructParser = NaStructParser()
    parser.parse_folder(folder_path=folder_path)

    state = parser.parser_state

    # JSON output mode: dump the whole intermediate representation
    if as_json:
        print(json.dumps(state.to_ir_dict(), indent=4))
        return

    # Summary line
    print(
        f"Parsed {len(state.tmp_structs)} struct(s) and "
        f"{len(state.tmp_functions)} function(s). "
        f"Database tables: {len(state.database.tables_by_struct_name)} "
        f"(empty until the resolution phase is implemented)."
    )
    print()

    # Structs in declaration order
    for tmp_struct in state.tmp_structs.values():
        print_struct(tmp_struct=tmp_struct)

    # Standalone functions in declaration order
    for fn in state.tmp_functions.values():
        print_function(fn=fn)


if __name__ == "__main__":
    main()
