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
from src.core.types.types import TypeBool, TypeFloat, TypeInt, TypeStr






def _indexed_raw_of(
    struct: Struct,
    instance: StructInstance,
    attribute: str,
) -> Optional[Any]:
    """
    Returns an instance's raw value for an indexed attribute.

    Absent attributes with a schema default materialize through
    get_attribute; attributes with neither a stored value nor a default
    yield None (the row is skipped by the index — it can never match).

    Args:
        struct (Struct): Table schema.
        instance (StructInstance): Row to read.
        attribute (str): Indexed attribute name.

    Returns:
        Optional[Any]: The raw value, or None when absent.
    """

    if attribute in instance.attributes:
        return instance.attributes[attribute].get()

    attr_def: Any = struct.get_attribute(name=attribute)

    if attr_def is not None and attr_def.has_default:
        return instance.get_attribute(name=attribute).get()

    return None


class StructTable:
    """
    Storage table for all instances of a specific Struct schema.

    Attributes:
        struct (Struct): Associated struct definition.
        type_id (int): 16-bit type index in the database.
        instances (dict[int, StructInstance]): Local id -> StructInstance mapping.
        next_local_id (int): Next local id index to assign.
        indexes (dict[str, dict[Any, set[int]]]): Optional per-attribute
            hash indexes (raw value -> local ids), session scoped.
        spatial_indexes (dict[str, Any]): Optional per-attribute cell
            indexes over geometry attributes (SpatialIndex), session
            scoped.
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
        self.indexes: dict[str, dict[Any, set[int]]] = {}

        # Session-scoped spatial indexes over geometry attributes
        self.spatial_indexes: dict[str, Any] = {}

        # Session-scoped routing networks involving this table (as the
        # node or the edge side)
        self.networks: list[Any] = []

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

        # Write-back hook for hash index maintenance
        instance.table_ref = self

        # New rows join any existing hash indexes (defaults included)
        self.index_instance(instance=instance)

        # New rows join any existing spatial indexes
        self.spatial_index_instance(instance=instance)

        for network in self.networks:
            network.on_insert(instance=instance)

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

    # --- Hash indexes (session scoped) ---

    def create_index(self, attribute: str) -> None:
        """
        Builds a hash index over a scalar attribute of this table.

        Materializes every row once (lazy rows decode), then indexes
        each instance's value for the attribute — schema defaults
        included. Incremental maintenance happens through insert,
        set_attribute, and removal.

        Args:
            attribute (str): Declared scalar attribute name.

        Raises:
            AttributeError: If the attribute is not declared.
            TypeError: If the attribute is not a scalar
                (Int / Float / Str / Bool).
        """

        attr_def: Any = self.struct.get_attribute(name=attribute)

        if attr_def is None:
            raise AttributeError(
                f"Cannot index undeclared attribute '{attribute}' "
                f"on Struct '{self.struct.name}'"
            )

        if not isinstance(
            attr_def.attr_type, (TypeInt, TypeFloat, TypeStr, TypeBool)
        ):
            raise TypeError(
                f"Only scalar attributes (Int / Float / Str / Bool) are "
                f"indexable, but '{self.struct.name}.{attribute}' is "
                f"'{repr(attr_def.attr_type)}'"
            )

        # Build from scratch: materialize lazy rows once
        if hasattr(self.instances, "materialize_all"):
            self.instances.materialize_all()

        index: dict[Any, set[int]] = {}

        for local_id, instance in dict.items(self.instances):

            raw: Optional[Any] = _indexed_raw_of(
                struct=self.struct, instance=instance, attribute=attribute
            )

            if raw is None:
                continue

            index.setdefault(raw, set()).add(local_id)

        self.indexes[attribute] = index

    def _index_add(self, attribute: str, raw: Any, local_id: int) -> None:
        """
        Adds one local id under an indexed attribute value.

        Args:
            attribute (str): Indexed attribute name.
            raw (Any): Raw scalar value.
            local_id (int): Local instance id.
        """

        self.indexes[attribute].setdefault(raw, set()).add(local_id)

    def _index_remove(self, attribute: str, raw: Any, local_id: int) -> None:
        """
        Removes one local id from an indexed attribute value.

        Args:
            attribute (str): Indexed attribute name.
            raw (Any): Previously indexed raw value.
            local_id (int): Local instance id.
        """

        entries: Optional[set[int]] = self.indexes[attribute].get(raw)

        if entries is not None:
            entries.discard(local_id)

            if not entries:
                self.indexes[attribute].pop(raw, None)

    def index_instance(self, instance: StructInstance) -> None:
        """
        Indexes one instance under every existing index of this table.

        Schema defaults materialize through get_attribute, so rows
        created without explicit values are indexed too.

        Args:
            instance (StructInstance): The instance to index.
        """

        if not self.indexes:
            return

        for attribute in self.indexes:

            raw: Optional[Any] = _indexed_raw_of(
                struct=self.struct, instance=instance, attribute=attribute
            )

            if raw is None:
                continue

            self._index_add(
                attribute=attribute,
                raw=raw,
                local_id=instance.entity_id.local_id,
            )

    def index_update(
        self,
        attribute: str,
        old_raw: Optional[Any],
        new_raw: Any,
        local_id: int,
    ) -> None:
        """
        Re-indexes one instance after an attribute write.

        Args:
            attribute (str): The written attribute (must be indexed).
            old_raw (Optional[Any]): Previously indexed raw value, or
                None when the attribute had no indexed value.
            new_raw (Any): The new raw value.
            local_id (int): Local instance id.
        """

        if old_raw is not None:
            self._index_remove(
                attribute=attribute, raw=old_raw, local_id=local_id
            )

        self._index_add(
            attribute=attribute, raw=new_raw, local_id=local_id
        )

    def index_remove_entity(self, local_id: int) -> None:
        """
        Drops one local id from every index of this table.

        Args:
            local_id (int): Local instance id of the removed entity.
        """

        for index in self.indexes.values():

            for raw in list(index.keys()):
                index[raw].discard(local_id)

            empty: list[Any] = [
                raw for raw, entries in index.items() if not entries
            ]

            for raw in empty:
                index.pop(raw, None)

        entity_value: int = EntityId.from_parts(
            type_id=self.type_id, local_id=local_id
        ).value

        for spatial in self.spatial_indexes.values():
            spatial.remove_instance(key=entity_value)

        for network in self.networks:
            network.on_remove(local_id=local_id, table=self)

    # --- Spatial indexes (session scoped) ---

    def create_spatial_index(
        self,
        attribute: str,
        lod: int = 8,
        parent_attribute: Optional[str] = None,
        database: Optional[Any] = None,
    ) -> Any:
        """
        Builds a cell index over a geometry attribute across this
        struct's hierarchy.

        Subclass tables (rows of structs inheriting this struct)
        share the index: one hierarchy, one cell grid, global entity
        id keys. The index registers on every involved table so
        insert / write / removal hooks maintain it.

        Args:
            attribute (str): Declared geometry attribute name.
            lod (int, optional): Level of detail (cell edge =
                360 / 2^lod degrees). Defaults to 8.
            parent_attribute (Optional[str], optional): Self-reference
                attribute enabling hierarchical locate() descent.
                Defaults to None (flat).
            database (Optional[Any], optional): The owning database,
                for subclass table discovery. Defaults to None (the
                base table only).

        Returns:
            Any: The built SpatialIndex.

        Raises:
            AttributeError: If the attribute is not declared.
            TypeError: If the attribute is not a geometry type.
        """

        from src.core.database.spatial_index import SpatialIndex

        # A previous index over the same attribute anywhere in the
        # hierarchy is replaced
        for involved in self._hierarchy_tables(database=database):
            involved.spatial_indexes.pop(attribute, None)

        hierarchy_tables: list[Any] = self._hierarchy_tables(
            database=database
        )

        index: Any = SpatialIndex(
            table=self,
            attribute=attribute,
            lod=lod,
            parent_attribute=parent_attribute,
            tables=hierarchy_tables,
        )

        for involved in hierarchy_tables:
            involved.spatial_indexes[attribute] = index

        return index

    def _hierarchy_tables(
        self,
        database: Optional[Any] = None,
    ) -> list[Any]:
        """
        Returns this table plus every subclass table.

        Args:
            database (Optional[Any], optional): The owning database.
                Defaults to None (base table only).

        Returns:
            list[Any]: The hierarchy's StructTables, base first.
        """

        tables: list[Any] = [self]

        if database is None:
            return tables

        for candidate in database.tables_by_struct_name.values():
            if candidate is self:
                continue

            walker: Any = candidate.struct.parent

            while walker is not None:
                if walker.name == self.struct.name:
                    tables.append(candidate)

                    break

                walker = walker.parent

        return tables

    def spatial_index_instance(self, instance: Any) -> None:
        """
        Indexes one instance under every existing spatial index of
        this table (insert hook).

        Args:
            instance (Any): The instance to index.
        """

        if not self.spatial_indexes:
            return

        for index in self.spatial_indexes.values():
            index.update_instance(instance=instance)

    def create_network(self, edges_table: Any) -> Any:
        """
        Builds a routing network over this table (nodes) and an edge
        table, replacing any previous network for the same pair.

        Args:
            edges_table (Any): The edge StructTable.

        Returns:
            Any: The NetworkIndex.
        """

        from src.core.database.network_index import NetworkIndex

        # Drop a previous network for the same pair from both tables
        previous: list[Any] = [
            existing
            for existing in self.networks
            if existing.edges_table is edges_table
        ]

        for stale in previous:
            self.networks.remove(stale)
            edges_table.networks.remove(stale)

        network: Any = NetworkIndex(nodes_table=self, edges_table=edges_table)

        network.build()

        self.networks.append(network)
        edges_table.networks.append(network)

        return network

    def spatial_index_update(self, attribute: str, local_id: int) -> None:
        """
        Re-indexes one instance after a geometry attribute write.

        Args:
            attribute (str): The written attribute (must be indexed).
            local_id (int): Local instance id.
        """

        index: Optional[Any] = self.spatial_indexes.get(attribute)

        if index is None:
            return

        instance: Optional[Any] = self.instances.get(local_id) if hasattr(
            self.instances, "get"
        ) else None

        if instance is None:
            instance = self.get(local_id=local_id)

        if instance is not None:
            index.update_instance(instance=instance)

    def lookup_by_index(
        self,
        attribute: str,
        raw: Any,
    ) -> Optional[list[StructInstance]]:
        """
        Resolves instances by indexed attribute equality in O(1).

        Args:
            attribute (str): Indexed attribute name.
            raw (Any): Raw scalar value to match.

        Returns:
            Optional[list[StructInstance]]: Matching instances, or None
            when the attribute has no index (caller falls back).
        """

        index: Optional[dict[Any, set[int]]] = self.indexes.get(attribute)

        if index is None:
            return None

        local_ids: Optional[set[int]] = index.get(raw)

        if not local_ids:
            return []

        return [
            self.instances[local_id]
            for local_id in local_ids
            if local_id in self.instances
        ]


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

        if entity_id is not None:
            return self.get_by_id(entity_id=entity_id)

        # Subclass fallback: an instance of Country is also an
        # AdminRegion; walk the registered namespaces by suffix
        suffix: str = f"/{nid}"

        candidates: list[tuple[str, EntityId]] = [
            (key, value)
            for key, value in self.nids.items()
            if key.endswith(suffix)
        ]

        if not candidates:
            return None

        for key, candidate_id in candidates:
            candidate_name: str = key.rsplit("/", 1)[0]

            candidate_table: Optional[Any] = (
                self.tables_by_struct_name.get(candidate_name)
            )

            if candidate_table is None:
                continue

            walker: Any = candidate_table.struct.parent

            while walker is not None:
                if walker.name == struct_name:
                    return self.get_by_id(entity_id=candidate_id)

                walker = walker.parent

        return None

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

        table.index_remove_entity(local_id=instance.entity_id.local_id)

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
