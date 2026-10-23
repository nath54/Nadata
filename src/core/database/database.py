"""
NaDataDatabase engine for Nadata.

Provides the high-performance database manager supporting typed tables,
globally unique 64-bit EntityIds, lazy paging, symbol registries,
and parallel/optimized queries.
"""

# Import Modules
from __future__ import annotations
from typing import Any, Callable, Optional

from src.core.values.values import Value
from src.core.struct.struct import Struct
from src.core.database.instance import StructInstance
from src.core.database.entity_id import EntityId
from src.core.database.queryset import QuerySet
from src.core.struct.struct_method import StructMethod
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
        self.storage_path: Optional[str] = storage_path

        # Project root for the I/O builtins (the session folder); None
        # resolves script paths against the current working directory
        self.io_base: Optional[str] = None

        self.tables_by_type_id: dict[int, StructTable] = {}
        self.tables_by_struct_name: dict[str, StructTable] = {}
        self.symbols: dict[str, EntityId] = {}
        self.nids: dict[str, EntityId] = {}
        self._nid_by_entity: dict[int, tuple[str, str]] = {}
        self.functions: dict[str, StructMethod] = {}
        self.query_executor: QueryExecutor = QueryExecutor()
        self.next_type_id: int = 0

        # Database-wide settings (persisted in the meta document of .ndb files)
        self.settings: dict[str, Any] = {}

    def register_function(self, function: StructMethod) -> None:
        """
        Registers a top-level schema function.

        Args:
            function (StructMethod): Function to register.
        """

        self.functions[function.name] = function

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

    def get_by_nid(self, struct_name: str, nid: str) -> Optional[StructInstance]:
        """
        Retrieves an instance by its named id, or None when unknown.

        Args:
            struct_name (str): Struct the nid lives in (per-struct
                namespace).
            nid (str): The named id.

        Returns:
            Optional[StructInstance]: The instance, or None.
        """

        entity_id: Optional[EntityId] = self.nids.get(
            f"{struct_name}/{nid}"
        )

        if entity_id is None:
            return None

        return self.get_by_id(entity_id=entity_id)

    def register_nid(
        self,
        struct_name: str,
        nid: str,
        entity_id: EntityId,
    ) -> None:
        """
        Binds a named id to an entity in the struct's namespace.

        Args:
            struct_name (str): Struct the nid lives in.
            nid (str): The named id.
            entity_id (EntityId): The bound entity.
        """

        key: str = f"{struct_name}/{nid}"

        self.nids[key] = entity_id
        self._nid_by_entity[entity_id.value] = (
            struct_name,
            nid,
        )

    def nid_of_entity(self, struct_name: str, entity_id: int) -> Optional[str]:
        """
        Returns the nid of an entity, or None when anonymous.

        Args:
            struct_name (str): Struct name of the entity.
            entity_id (int): Global 64-bit entity handle.

        Returns:
            Optional[str]: The named id, or None.
        """

        mapped: Optional[tuple[str, str]] = self._nid_by_entity.get(
            entity_id
        )

        if mapped is None or mapped[0] != struct_name:
            return None

        return mapped[1]

    def drop_nids_of_entity(self, entity_id: int) -> None:
        """
        Forgets every nid bound to an entity (removal bookkeeping).

        Args:
            entity_id (int): Global 64-bit entity handle.
        """

        mapped: Optional[tuple[str, str]] = self._nid_by_entity.pop(
            entity_id, None
        )

        if mapped is None:
            return

        self.nids.pop(f"{mapped[0]}/{mapped[1]}", None)

    def nid_names(self, struct_name: str) -> list[str]:
        """
        Returns the sorted nid strings of a struct's named instances.

        Args:
            struct_name (str): Struct name.

        Returns:
            list[str]: Sorted nids.
        """

        prefix: str = f"{struct_name}/"

        return sorted(
            key[len(prefix):]
            for key in self.nids
            if key.startswith(prefix)
        )

    def remove_instance(self, instance: StructInstance) -> None:
        """
        Removes an instance from its table and the decoder caches.

        Purely in-memory: persistence of the removal requires a
        subsequent tombstone append (remove_from_db / flush with
        tombstone support). The entity id is never reused.

        Args:
            instance (StructInstance): The instance to remove.
        """

        table: Optional[StructTable] = self.tables_by_type_id.get(
            instance.entity_id.type_id
        )

        if table is None:
            return

        self.drop_nids_of_entity(entity_id=instance.entity_id.value)

        if (
            table.instances.__contains__(instance.entity_id.local_id)
            and table.instances[instance.entity_id.local_id] is instance
        ):
            del table.instances[instance.entity_id.local_id]

        decoder: Any = getattr(table.instances, "decoder", None)

        if decoder is not None:
            decoder.remove_entity(entity_id=instance.entity_id.value)

    def flush(self) -> int:
        """
        Writes all dirty instances to the database file as delta rows.

        Returns:
            int: Number of instances written (0 if there is no file).
        """

        if self.storage_path is None:
            return 0

        from src.core.database.persistence.dumper import flush_dirty_instances

        return flush_dirty_instances(database=self, storage_path=self.storage_path)

    def clear_cache(self) -> int:
        """
        Evicts all clean, unpinned instances from the decode caches.

        Returns:
            int: Number of instances evicted (0 if there is no file).
        """

        if self.storage_path is None:
            return 0

        evicted: int = 0

        for table in self.tables_by_type_id.values():

            decoder: Any = getattr(table.instances, "decoder", None)

            if decoder is not None:
                evicted += decoder.evict_clean()

        return evicted

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
