"""
Persistence round trip for Nadata databases.

Provides dump_database and load_database to serialize a fully resolved
database (structs with AST bodies, functions, instances, symbols) into a
slotted page file and reconstruct it, preserving entity ids and cross
instance references.
"""

# Import Modules
from src.core.database.persistence.compaction import compact_database, count_documents
from src.core.database.persistence.dumper import append_database_changes, dump_database
from src.core.database.persistence.loader import load_database

__all__ = [
    "append_database_changes",
    "compact_database",
    "count_documents",
    "dump_database",
    "load_database",
]
