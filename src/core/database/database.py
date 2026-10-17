"""
NaDataDatabase engine for Nadata.

Provides the high-performance database manager supporting typed tables,
globally unique 64-bit EntityIds, lazy paging, symbol registries,
and parallel/optimized queries.
"""

# Import Modules
from __future__ import annotations
from typing import Callable, Optional

from src.core.values.values import Value
from src.core.struct.struct import Struct
from src.core.database.instance import StructInstance
from src.core.database.entity_id import EntityId
from src.core.database.queryset import QuerySet
from src.core.database.page_manager import PageManager
from src.core.database.query_engine import QueryExecutor





class StructTable:
    """
    Storage table for all instances of a specific Struct schema.

    Attributes:
        struct (Struct): Associated struct definition.
        type_id (int): 16-bit type index in the database.
        instances (dict[int, StructInstance]): Local id -> StructInstance mapping.
        next_local_id (int): Next local id index to assign.
    """

    def __init__(self, struct: Struct, type_id: int) -> None:
        """
        Initializes the StructTable instance.

        Args:
            struct (Struct): Struct schema.
            type_id (int): 16-bit type index.
        """

        self.struct: Struct = struct
        self.type_id: int = type_id
        self.instances: dict[int, StructInstance] = {}
        self.next_local_id: int = 0

    def insert(self, attributes: Optional[dict[str, Value]] = None) -> StructInstance:
        """
        Creates and stores a new StructInstance with a globally unique EntityId.

        Args:
            attributes (Optional[dict[str, Value]], optional): Attribute values. Defaults to None.

        Returns:
            StructInstance: Newly created instance.
        """

        local_id: int = self.next_local_id
        self.next_local_id += 1

        entity_id: EntityId = EntityId.from_parts(type_id=self.type_id, local_id=local_id)

        instance: StructInstance = StructInstance(
            struct=self.struct,
            entity_id=entity_id,
            attributes=attributes
        )

        self.instances[local_id] = instance
        return instance

    def get(self, local_id: int) -> Optional[StructInstance]:
        """
        Retrieves an instance by local id.

        Args:
            local_id (int): Local index.

        Returns:
            Optional[StructInstance]: Stored instance or None.
        """

        return self.instances.get(local_id)

    def all(self) -> list[StructInstance]:
        """
        Returns all instances stored in this table.

        Returns:
            list[StructInstance]: List of instances.
        """

        return list(self.instances.values())


class NaDataDatabase:
    """
    Central database engine for Nadata instances.

    Attributes:
        page_manager (PageManager): Slotted page and cache manager.
        tables_by_type_id (dict[int, StructTable]): Type ID -> StructTable registry.
        tables_by_struct_name (dict[str, StructTable]): Struct name -> StructTable.
        symbols (dict[str, EntityId]): Named symbol aliases (e.g. "tom" -> EntityId).
        query_executor (QueryExecutor): Parallel query runner.
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        max_cached_pages: int = 256
    ) -> None:
        """
        Initializes the NaDataDatabase instance.

        Args:
            storage_path (Optional[str], optional): Path to database file. Defaults to None.
            max_cached_pages (int, optional): In-memory page limit. Defaults to 256.
        """

        self.page_manager: PageManager = PageManager(
            storage_path=storage_path,
            max_cached_pages=max_cached_pages
        )

        self.tables_by_type_id: dict[int, StructTable] = {}
        self.tables_by_struct_name: dict[str, StructTable] = {}
        self.symbols: dict[str, EntityId] = {}
        self.query_executor: QueryExecutor = QueryExecutor()
        self.next_type_id: int = 0

    def register_struct(self, struct: Struct) -> StructTable:
        """
        Registers a Struct schema into the database, allocating a 16-bit TypeId.

        Args:
            struct (Struct): Schema to register.

        Returns:
            StructTable: Created or existing storage table.
        """

        if struct.name in self.tables_by_struct_name:
            return self.tables_by_struct_name[struct.name]

        type_id: int = self.next_type_id
        self.next_type_id += 1

        table: StructTable = StructTable(struct=struct, type_id=type_id)
        self.tables_by_type_id[type_id] = table
        self.tables_by_struct_name[struct.name] = table

        return table

    def create_instance(
        self,
        struct: Struct,
        attributes: Optional[dict[str, Value]] = None,
        symbol: Optional[str] = None
    ) -> StructInstance:
        """
        Creates and stores an instance, optionally binding a symbol name.

        Args:
            struct (Struct): Schema of the instance.
            attributes (Optional[dict[str, Value]], optional): Field values. Defaults to None.
            symbol (Optional[str], optional): Symbol alias (e.g. "tom"). Defaults to None.

        Returns:
            StructInstance: Created instance.
        """

        table: StructTable = self.register_struct(struct=struct)
        instance: StructInstance = table.insert(attributes=attributes)

        if symbol is not None:
            self.symbols[symbol] = instance.entity_id

        return instance

    def get_by_id(self, entity_id: EntityId | int) -> Optional[StructInstance]:
        """
        Retrieves an instance by its globally unique EntityId in O(1).

        Args:
            entity_id (EntityId | int): EntityId or raw 64-bit int.

        Returns:
            Optional[StructInstance]: Stored instance or None.
        """

        resolved_id: EntityId = (
            entity_id if isinstance(entity_id, EntityId) else EntityId(entity_id)
        )

        table: Optional[StructTable] = self.tables_by_type_id.get(resolved_id.type_id)

        if table is None:
            return None

        return table.get(local_id=resolved_id.local_id)

    def get_by_symbol(self, name: str) -> Optional[StructInstance]:
        """
        Retrieves an instance by its named symbol alias.

        Args:
            name (str): Symbol alias (e.g. "tom").

        Returns:
            Optional[StructInstance]: Stored instance or None.
        """

        entity_id: Optional[EntityId] = self.symbols.get(name)

        if entity_id is None:
            return None

        return self.get_by_id(entity_id=entity_id)

    def query(
        self,
        struct: Struct,
        predicate: Optional[Callable[[StructInstance], bool]] = None,
        include_subtypes: bool = True
    ) -> QuerySet:
        """
        Queries instances of a Struct with optional parallel predicate filtering.

        Args:
            struct (Struct): Base struct type to query.
            predicate (Optional[Callable[[StructInstance], bool]], optional): Predicate filter.
                Defaults to None.
            include_subtypes (bool, optional): Include derived structs. Defaults to True.

        Returns:
            QuerySet: Matching instances in a chained QuerySet.
        """

        candidates: list[StructInstance] = []

        # Gather tables matching struct or inheriting from it
        for table in self.tables_by_type_id.values():
            if table.struct == struct:
                candidates.extend(table.all())
            elif include_subtypes and table.struct.is_subclass_of(other=struct):
                candidates.extend(table.all())

        queryset: QuerySet = QuerySet(
            struct=struct,
            instances=candidates,
            database=self,
            query_executor=self.query_executor
        )

        if predicate is not None:
            return queryset.where(predicate=predicate)

        return queryset
