"""
Database package initialization for Nadata.

Exports the core database components including EntityId, SlottedPage,
PageManager, StructTable, QueryOptimizer, QueryExecutor, and NaDataDatabase.
"""

# Import Modules
from src.core.database.instance import StructInstance
from src.core.database.entity_id import EntityId
from src.core.database.slotted_page import SlottedPage
from src.core.database.page_manager import PageManager
from src.core.database.database import (
    StructTable,
    NaDataDatabase,
)
from src.core.database.query_engine import (
    QueryExecutor,
    QueryOptimizer,
)

__all__: list[str] = [
    "EntityId",
    "SlottedPage",
    "PageManager",
    "StructTable",
    "StructInstance",
    "NaDataDatabase",
    "QueryExecutor",
    "QueryOptimizer",
]
