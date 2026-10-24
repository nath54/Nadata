"""
Plain text dumper for NaDataDatabase state.

Provides the NaDataDatabaseDumper class rendering the full database content
(structs, attributes, methods, functions, symbols, and instances) as a
readable text report, highlighting missing or incomplete data for debugging
parser and loader pipelines.
"""

# Import Modules
from __future__ import annotations

from nadata.core.struct.struct import Struct
from nadata.core.struct.struct_attribute import StructAttribute
from nadata.core.database.database import (
    NaDataDatabase,
    StructTable,
)
from nadata.core.database.instance import StructInstance



class NaDataDatabaseDumper:
    """
    Renders a NaDataDatabase as a structured plain text report.

    Attributes:
        database (NaDataDatabase): Database instance to dump.
    """

    # Separator line used between report sections
    SECTION_SEPARATOR: str = "=" * 78

    # Separator line used between items inside a section
    ITEM_SEPARATOR: str = "-" * 78

    def __init__(self, database: NaDataDatabase) -> None:
        """
        Initializes the NaDataDatabaseDumper instance.

        Args:
            database (NaDataDatabase): Database instance to dump.
        """

        self.database: NaDataDatabase = database

    def dump_to_string(self) -> str:
        """
        Renders the full database state as a plain text string.

        Returns:
            str: Multi-line text report of the database content.
        """

        sections: list[str] = [
            self._dump_header(),
            self._dump_structs(),
            self._dump_functions(),
            self._dump_symbols(),
            self._dump_instances(),
        ]

        return "\n\n".join(sections) + "\n"

    def dump_to_file(self, file_path: str) -> None:
        """
        Writes the full database state to a plain text file.

        Args:
            file_path (str): Destination file path.
        """

        with open(file_path, "w", encoding="utf-8") as file:
            file.write(self.dump_to_string())

    def _dump_header(self) -> str:
        """
        Builds the report header with global database statistics.

        Returns:
            str: Header section text.
        """

        instance_count: int = sum(
            len(table.instances)
            for table in self.database.tables_by_type_id.values()
        )

        lines: list[str] = [
            self.SECTION_SEPARATOR,
            "NaDataDatabase Dump",
            f"  structs:   {len(self.database.tables_by_struct_name)}",
            f"  functions: {len(self.database.functions)}",
            f"  symbols:   {len(self.database.symbols)}",
            f"  instances: {instance_count}",
            self.SECTION_SEPARATOR,
        ]

        return "\n".join(lines)

    def _dump_structs(self) -> str:
        """
        Builds the structs section listing inheritance, attributes, and methods.

        Returns:
            str: Structs section text.
        """

        lines: list[str] = ["--- Structs ---"]

        # Handle the empty database case explicitly
        if not self.database.tables_by_struct_name:
            lines.append("  (no struct registered)")
            return "\n".join(lines)

        for table in self.database.tables_by_type_id.values():
            lines.append("")
            lines.append(self._format_struct(table=table))

        return "\n".join(lines)

    def _dump_functions(self) -> str:
        """
        Builds the standalone functions section.

        Returns:
            str: Functions section text.
        """

        lines: list[str] = ["--- Functions ---"]

        # Handle the no function case explicitly
        if not self.database.functions:
            lines.append("  (no function registered)")
            return "\n".join(lines)

        for function in self.database.functions.values():
            lines.append(f"  {repr(function)}")

        return "\n".join(lines)

    def _dump_symbols(self) -> str:
        """
        Builds the symbols section mapping symbol names to entity ids.

        Returns:
            str: Symbols section text.
        """

        lines: list[str] = ["--- Symbols ---"]

        # Handle the no symbol case explicitly
        if not self.database.symbols:
            lines.append("  (no symbol registered)")
            return "\n".join(lines)

        for symbol_name, entity_id in self.database.symbols.items():
            lines.append(f"  {symbol_name} -> {repr(entity_id)}")

        return "\n".join(lines)

    def _dump_instances(self) -> str:
        """
        Builds the instances section listing stored instances per struct table.

        Returns:
            str: Instances section text.
        """

        lines: list[str] = ["--- Instances ---"]

        # Handle the empty database case explicitly
        if not self.database.tables_by_type_id:
            lines.append("  (no instance stored)")
            return "\n".join(lines)

        for table in self.database.tables_by_type_id.values():
            lines.append("")
            lines.append(self._format_table_instances(table=table))

        return "\n".join(lines)

    def _format_struct(self, table: StructTable) -> str:
        """
        Formats a single struct schema with its attributes and methods.

        Args:
            table (StructTable): Table holding the struct definition.

        Returns:
            str: Formatted struct block.
        """

        struct_def: Struct = table.struct

        # Build the inheritance suffix
        parent_suffix: str = (
            f"({struct_def.parent.name})" if struct_def.parent is not None else ""
        )

        lines: list[str] = [
            f"Struct {struct_def.name}{parent_suffix} [type_id={table.type_id}]",
            f"  attributes ({len(struct_def.attributes)}):",
        ]

        # List declared attributes, or a placeholder when none are declared
        if not struct_def.attributes:
            lines.append("    (none declared)")
        else:
            for attribute in struct_def.attributes.values():
                lines.append(f"    {repr(attribute)}")

        lines.append(f"  methods ({len(struct_def.methods)}):")

        # List declared methods, or a placeholder when none are declared
        if not struct_def.methods:
            lines.append("    (none declared)")
        else:
            for method in struct_def.methods.values():
                lines.append(f"    {repr(method)}")

        return "\n".join(lines)

    def _format_table_instances(self, table: StructTable) -> str:
        """
        Formats all instances of a struct table with completeness diagnostics.

        Args:
            table (StructTable): Table whose instances are rendered.

        Returns:
            str: Formatted instances block for the table.
        """

        lines: list[str] = [
            f"[{table.struct.name}] ({len(table.instances)} instances)"
        ]

        # Handle the empty table case explicitly
        if not table.instances:
            lines.append("  (no instance)")
            return "\n".join(lines)

        for instance in table.instances.values():
            lines.append(f"  {self._format_instance(instance=instance)}")

        return "\n".join(lines)

    def _format_instance(self, instance: StructInstance) -> str:
        """
        Formats a single instance with its attribute values and missing fields.

        Args:
            instance (StructInstance): Instance to render.

        Returns:
            str: Formatted instance block.
        """

        lines: list[str] = [f"id={instance.entity_id.local_id}:"]

        all_attributes: dict[str, StructAttribute] = (
            instance.struct.get_all_attributes()
        )

        # Print explicitly set attribute values, flagging undeclared ones
        for attr_name, attr_value in instance.attributes.items():
            if attr_name in all_attributes:
                lines.append(f"    {attr_name} = {repr(attr_value)}")
            else:
                lines.append(f"    {attr_name} = {repr(attr_value)} !! NOT DECLARED IN SCHEMA")

        # Detect schema attributes neither set nor defaulted on the instance
        for attr_name, attr_def in all_attributes.items():
            if attr_name in instance.attributes:
                continue

            if attr_def.has_default:
                lines.append(
                    f"    {attr_name} = <default: {repr(attr_def.default_value)}>"
                )
            else:
                lines.append(f"    {attr_name} = !! MISSING (required, no default)")

        return "\n".join(lines)
