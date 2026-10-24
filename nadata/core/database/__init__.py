"""
Database package initialization for Nadata.

Exports the core database components including EntityId, SlottedPage,
PageManager, StructTable, QueryOptimizer, QueryExecutor, and NaDataDatabase.
"""

# Import Modules
from nadata.core.database.builtins import PrintFunction
from nadata.core.database.dumper import NaDataDatabaseDumper
from nadata.core.database.queryset import QuerySet
from nadata.core.database.instance import StructInstance
from nadata.core.database.entity_id import EntityId
from nadata.core.database.slotted_page import SlottedPage
from nadata.core.database.page_manager import PageManager
from nadata.core.database.database import (
    StructTable,

    NaDataDatabase,
)
from nadata.core.database.query_engine import (
    QueryExecutor,
    QueryOptimizer,
)

__all__: list[str] = [
    "NaDataDatabaseDumper",
    "EntityId",
    "QuerySet",
    "SlottedPage",
    "PageManager",
    "StructTable",
    "PrintFunction",
    "StructInstance",
    "NaDataDatabase",
    "QueryExecutor",
    "QueryOptimizer",
]
