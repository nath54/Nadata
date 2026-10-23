"""
Lazy instance materialization for the binary persistence format (v2).

Instance rows are stored as compact binary payloads inside batch
documents. Loading a database indexes the rows WITHOUT decoding them —
each batch becomes a columnar BatchIndex (parallel arrays of entity ids
and payload spans, ~40 bytes per row). StructInstance objects are
materialized on first access and cached in a bounded LRU; clean,
unpinned instances evict freely and re-decode from disk. Instance
references inside rows resolve on demand through the same decoder, so
cross referenced graphs stream in naturally.
"""

# Import Modules
from __future__ import annotations
from array import array
from bisect import bisect_left
from collections import OrderedDict
from typing import Any, Callable, Iterator, Optional

from src.core.database.entity_id import EntityId
from src.core.database.instance import StructInstance
from src.core.database.persistence.binary_cells import decode_row
from src.core.struct.struct import Struct
from src.core.types.types import Type, TypeFloat, TypeTensor
from src.core.values.values import Value, ValueTensor

# Default bound on simultaneously decoded instances (the working set;
# read-only dataset rows evict freely below this)
DEFAULT_MAX_DECODED: int = 100_000

# Documents whose reassembled payload stays cached (rows of one batch
# decode together; the bound keeps streaming fetches transient)
DEFAULT_DOC_CACHE_DOCS: int = 32

# Oldest entries inspected per eviction attempt (bounded work per decode)
_EVICT_SCAN_WINDOW: int = 32


class RowSource:
    """
    Shared per-batch row metadata (one object per batch document).

    Attributes:
        struct_name (str): Name of the struct the rows belong to.
        columns (list[str]): Column names in order.
        types (dict[str, Type]): Declared schema type per column.
        doc_id (int): Stored document id (-1 for static in-memory rows).
        fetch (Optional[Callable]): Reassembles (doc_id, offset, length)
            spans; unused for static rows.
    """

    __slots__ = ("struct_name", "columns", "types", "doc_id", "fetch")

    def __init__(
        self,
        struct_name: str,
        columns: list[str],
        types: dict[str, Type],
        doc_id: int = -1,
        fetch: Optional[Callable[[tuple[int, int, int]], bytes]] = None,
    ) -> None:
        """
        Initializes the RowSource instance.

        Args:
            struct_name (str): Owning struct name.
            columns (list[str]): Column names in order.
            types (dict[str, Type]): Declared type per column.
            doc_id (int, optional): Document id. Defaults to -1.
            fetch (Optional[Callable], optional): Span reassembler.
                Defaults to None.
        """

        self.struct_name: str = struct_name
        self.columns: list[str] = columns
        self.types: dict[str, Type] = types
        self.doc_id: int = doc_id
        self.fetch: Optional[Callable[[tuple[int, int, int]], bytes]] = fetch


class BatchIndex:
    """
    Columnar pending-row index for one stored batch document.

    Rows live in parallel arrays sorted by entity id, so lookup is a
    bisect and the per-row footprint is four machine words.

    Attributes:
        source (RowSource): Shared batch metadata.
        entity_ids (array): Sorted global entity handles.
        local_ids (array): Sorted local ids (same order).
        offsets (array): Payload start offsets into the document.
        lengths (array): Payload lengths.
    """

    __slots__ = (
        "source",
        "entity_ids",
        "local_ids",
        "offsets",
        "lengths",
        "local_min",
        "local_max",
        "entity_min",
        "entity_max",
    )

    def __init__(
        self,
        source: RowSource,
        rows: list[tuple[int, int, int]],
    ) -> None:
        """
        Initializes the BatchIndex instance.

        Args:
            source (RowSource): Shared batch metadata.
            rows (list[tuple[int, int, int]]): (entity id, offset,
                length) spans in document order.
        """

        ordered: list[tuple[int, int, int]] = sorted(rows)

        self.source: RowSource = source
        self.entity_ids: "array[int]" = array(
            "q", [entity_id for entity_id, _, _ in ordered]
        )
        self.local_ids: "array[int]" = array(
            "q",
            [entity_id & EntityId.LOCAL_ID_MASK for entity_id, _, _ in ordered],
        )
        self.offsets: "array[int]" = array("q", [offset for _, offset, _ in ordered])
        self.lengths: "array[int]" = array("q", [length for _, _, length in ordered])

        # Inclusive local-id range covered by this batch (routing hint:
        # batches are laid out in ascending local ranges, so most
        # lookups touch exactly one batch). Entity ids share the order.
        self.local_min: int = self.local_ids[0]
        self.local_max: int = self.local_ids[-1]
        self.entity_min: int = self.entity_ids[0]
        self.entity_max: int = self.entity_ids[-1]

    def find(self, entity_id: int) -> Optional[int]:
        """
        Locates a row by global entity handle.

        Args:
            entity_id (int): Global entity handle.

        Returns:
            Optional[int]: The row position, or None.
        """

        position: int = bisect_left(self.entity_ids, entity_id)

        if (
            position < len(self.entity_ids)
            and self.entity_ids[position] == entity_id
        ):
            return position

        return None

    def find_local(self, local_id: int) -> Optional[int]:
        """
        Locates a row by local instance id.

        Args:
            local_id (int): Local instance index.

        Returns:
            Optional[int]: The row position, or None.
        """

        position: int = bisect_left(self.local_ids, local_id)

        if (
            position < len(self.local_ids)
            and self.local_ids[position] == local_id
        ):
            return position

        return None

    def span(self, position: int) -> tuple[int, int, int]:
        """
        Returns the document span of a row position.

        Args:
            position (int): Row position.

        Returns:
            tuple[int, int, int]: (doc id, offset, length).
        """

        return (
            self.source.doc_id,
            self.offsets[position],
            self.lengths[position],
        )

    def entity_id_at(self, position: int) -> int:
        """
        Returns the entity id of a row position.

        Args:
            position (int): Row position.

        Returns:
            int: The global entity handle.
        """

        indexed: int = self.entity_ids[position]

        return indexed


class PendingRow:
    """
    One undecoded instance row.

    Rows are either static (an in-memory payload slice) or disk-backed:
    a (doc_id, offset, length) span into a stored batch document,
    fetched through the decoder's cache only when the row is decoded.

    Attributes:
        entity_id (int): Global 64-bit entity handle of the row.
        source (RowSource): Shared batch metadata.
        doc_span (Optional[tuple[int, int, int]]): Disk location, or
            None for static rows.
    """

    __slots__ = ("entity_id", "source", "doc_span", "_static_payload")

    def __init__(
        self,
        entity_id: int,
        struct_name: str,
        columns: list[str],
        types: dict[str, Type],
        payload: Optional[bytes] = None,
        doc_span: Optional[tuple[int, int, int]] = None,
        fetch: Optional[Callable[[tuple[int, int, int]], bytes]] = None,
    ) -> None:
        """
        Initializes the PendingRow instance.

        Args:
            entity_id (int): Global entity handle.
            struct_name (str): Owning struct name.
            columns (list[str]): Batch column names.
            types (dict[str, Type]): Declared type per column.
            payload (Optional[bytes]): Static encoded row payload.
            doc_span (Optional[tuple[int, int, int]]): (doc id, offset,
                length) of the row inside a stored batch document.
            fetch (Optional[Callable]): Reassembles a doc span; required
                when doc_span is given.
        """

        self.entity_id: int = entity_id
        self.source: RowSource = RowSource(
            struct_name=struct_name,
            columns=columns,
            types=types,
            doc_id=doc_span[0] if doc_span is not None else -1,
            fetch=fetch,
        )
        self.doc_span: Optional[tuple[int, int, int]] = doc_span
        self._static_payload: Optional[bytes] = payload

    @classmethod
    def for_span(
        cls,
        source: RowSource,
        entity_id: int,
        doc_span: tuple[int, int, int],
    ) -> "PendingRow":
        """
        Builds a transient row view over a shared batch source.

        Args:
            source (RowSource): Shared batch metadata.
            entity_id (int): Global entity handle.
            doc_span (tuple[int, int, int]): (doc id, offset, length).

        Returns:
            PendingRow: The transient row view.
        """

        row: PendingRow = cls.__new__(cls)

        row.entity_id = entity_id
        row.source = source
        row.doc_span = doc_span
        row._static_payload = None

        return row

    @property
    def struct_name(self) -> str:
        """
        Returns the owning struct name.

        Returns:
            str: The struct name.
        """

        return self.source.struct_name

    @property
    def columns(self) -> list[str]:
        """
        Returns the batch column names.

        Returns:
            list[str]: Column names in order.
        """

        return self.source.columns

    @property
    def types(self) -> dict[str, Type]:
        """
        Returns the declared type per column.

        Returns:
            dict[str, Type]: The types.
        """

        return self.source.types

    @property
    def payload(self) -> bytes:
        """
        Returns the encoded row payload, fetching from disk if needed.

        Returns:
            bytes: The encoded row payload.
        """

        if self.doc_span is None:
            assert self._static_payload is not None

            return self._static_payload

        assert self.source.fetch is not None

        return self.source.fetch(self.doc_span)


class RowDecoder:
    """
    Registry of pending rows with on-demand decoding and ref resolution.

    Acts as the buffer pool for materialized instances: decoded entries
    live in an LRU-ordered dict with a configurable bound. Clean,
    unpinned instances evict freely (they re-decode from disk); dirty
    and pinned instances are never evicted. In-place tensor mutations
    are detected through version-counter snapshots taken at decode time.

    Attributes:
        decoded (OrderedDict[int, StructInstance]): Materialized
            instances by entity id, oldest first (also the resolver
            cache).
        max_decoded (int): Bound on simultaneously decoded instances.
    """

    def __init__(
        self,
        max_decoded: int = DEFAULT_MAX_DECODED,
        doc_cache_docs: int = DEFAULT_DOC_CACHE_DOCS,
    ) -> None:
        """
        Initializes the RowDecoder instance.

        Args:
            max_decoded (int, optional): Decode-cache bound. Defaults to
                DEFAULT_MAX_DECODED.
            doc_cache_docs (int, optional): Reassembled batch documents
                kept cached for row fetches. Defaults to
                DEFAULT_DOC_CACHE_DOCS.
        """

        self.decoded: OrderedDict[int, StructInstance] = OrderedDict()
        self.max_decoded: int = max_decoded
        self._batches: dict[str, list[BatchIndex]] = {}
        self._static_rows: dict[int, PendingRow] = {}
        self._dropped: set[int] = set()
        self._pending_by_local: dict[tuple[str, int], int] = {}
        self._in_progress: set[int] = set()
        self._shells: dict[int, StructInstance] = {}
        self._struct_lookup: Optional[Callable[[str], Struct]] = None
        self._tables: dict[str, Any] = {}
        self._version_snapshots: dict[int, dict[str, int]] = {}
        self._doc_cache: OrderedDict[int, bytes] = OrderedDict()
        self._doc_cache_limit: int = doc_cache_docs
        self._document_reader: Optional[Callable[[int], bytes]] = None

    def set_struct_lookup(self, lookup: Callable[[str], Struct]) -> None:
        """
        Installs the struct lookup used to resolve row schemas.

        Args:
            lookup (Callable[[str], Struct]): Maps a struct name to its
                Struct definition.
        """

        self._struct_lookup = lookup

    def register_table(self, struct_name: str, table: Any) -> None:
        """
        Registers a struct's lazy instance mapping for eviction.

        Args:
            struct_name (str): Owning struct name.
            table (Any): The LazyInstanceDict to drop entries from.
        """

        self._tables[struct_name] = table

    def set_document_reader(
        self, reader: Optional[Callable[[int], bytes]]
    ) -> None:
        """
        Installs the callback reassembling stored documents by id.

        Args:
            reader (Optional[Callable[[int], bytes]]): Document reader,
                or None to detach.
        """

        self._document_reader = reader
        self._doc_cache.clear()

    def fetch_document_span(self, doc_span: tuple[int, int, int]) -> bytes:
        """
        Returns a row payload slice from a cached batch document.

        Args:
            doc_span (tuple[int, int, int]): (doc id, offset, length).

        Returns:
            bytes: The encoded row payload.
        """

        doc_id: int = doc_span[0]

        document: Optional[bytes] = self._doc_cache.get(doc_id)

        if document is None:

            if self._document_reader is None:
                raise RuntimeError(
                    "RowDecoder has no document reader installed"
                )

            document = self._document_reader(doc_id)

            self._doc_cache[doc_id] = document

            while len(self._doc_cache) > self._doc_cache_limit:
                self._doc_cache.popitem(last=False)

        else:
            self._doc_cache.move_to_end(doc_id)

        offset: int = doc_span[1]
        length: int = doc_span[2]

        return document[offset:offset + length]

    def dirty_decoded_entity_ids(self) -> set[int]:
        """
        Returns decoded entity ids whose state changed since decode.

        Covers the explicit dirty flag and in-place tensor mutations
        detected through version-counter snapshots.

        Returns:
            set[int]: Dirty entity ids.
        """

        return {
            entity_id
            for entity_id, instance in self.decoded.items()
            if self.is_dirty(instance=instance)
        }

    def is_dirty(self, instance: StructInstance) -> bool:
        """
        Checks whether an instance changed since it was decoded.

        Args:
            instance (StructInstance): Candidate instance.

        Returns:
            bool: True if the flag is set or a tensor version moved.
        """

        if instance.dirty:
            return True

        snapshot: Optional[dict[str, int]] = (
            self._version_snapshots.get(instance.entity_id.value)
        )

        if snapshot is None:
            return False

        for attr_name, version in snapshot.items():

            value: Any = instance.attributes.get(attr_name)

            if isinstance(value, ValueTensor) and value.version != version:
                return True

        return False

    def mark_clean(self, instance: StructInstance) -> None:
        """
        Clears the dirty state and refreshes version snapshots.

        Args:
            instance (StructInstance): Instance to reset.
        """

        instance.dirty = False
        self._snapshot_versions(instance=instance)

    def evict_clean(self) -> int:
        """
        Drops clean, unpinned decoded instances from the cache.

        Returns:
            int: The number of evicted instances.
        """

        evicted: list[int] = [
            entity_id
            for entity_id, instance in self.decoded.items()
            if not instance.pinned and not self.is_dirty(instance=instance)
        ]

        for entity_id in evicted:
            self._remove_decoded(entity_id=entity_id)

        return len(evicted)

    def _remove_decoded(self, entity_id: int) -> None:
        """
        Removes one decoded instance from the cache and its table.

        Args:
            entity_id (int): The entity id to drop.
        """

        instance: Optional[StructInstance] = self.decoded.pop(
            entity_id, None
        )

        self._version_snapshots.pop(entity_id, None)

        if instance is None:
            return

        table: Any = self._tables.get(instance.struct.name)

        if table is not None:
            table.evict_entity(entity_id=entity_id)

    def _snapshot_versions(self, instance: StructInstance) -> None:
        """
        Records the version counters of the instance's float tensors.

        Args:
            instance (StructInstance): Instance to snapshot.
        """

        snapshot: dict[str, int] = {}

        for attr_name, value in instance.attributes.items():

            if isinstance(value, ValueTensor) and isinstance(
                value.value_type.value_type, TypeFloat
            ):
                snapshot[attr_name] = value.version

        self._version_snapshots[instance.entity_id.value] = snapshot

    def register_batch(
        self,
        struct_name: str,
        columns: list[str],
        types: dict[str, Type],
        doc_id: int,
        rows: list[tuple[int, int, int]],
        fetch: Callable[[tuple[int, int, int]], bytes],
    ) -> None:
        """
        Indexes a whole batch document columnar-wise.

        Args:
            struct_name (str): Owning struct name.
            columns (list[str]): Column names in order.
            types (dict[str, Type]): Declared type per column.
            doc_id (int): The stored document id.
            rows (list[tuple[int, int, int]]): (entity id, offset,
                length) spans in document order.
            fetch (Callable): Reassembles (doc id, offset, length)
                spans.
        """

        batch: BatchIndex = BatchIndex(
            source=RowSource(
                struct_name=struct_name,
                columns=columns,
                types=types,
                doc_id=doc_id,
                fetch=fetch,
            ),
            rows=rows,
        )

        self._batches.setdefault(struct_name, []).append(batch)

    def _locate(self, entity_id: int) -> Optional[PendingRow]:
        """
        Finds the pending row of an entity, static rows first.

        Args:
            entity_id (int): Global entity handle.

        Returns:
            Optional[PendingRow]: The row, or None.
        """

        if entity_id in self._dropped:
            return None

        static: Optional[PendingRow] = self._static_rows.get(entity_id)

        if static is not None:
            return static

        # Later appended batches override earlier rows for the same
        # entity (replay-log semantics), so scan newest first
        for batches in self._batches.values():

            for batch in reversed(batches):

                if not (
                    batch.entity_min <= entity_id <= batch.entity_max
                ):
                    continue

                position: Optional[int] = batch.find(entity_id=entity_id)

                if position is not None:
                    return self._batch_row(
                        batch=batch, position=position
                    )

        return None

    def _batch_row(self, batch: BatchIndex, position: int) -> PendingRow:
        """
        Builds a transient PendingRow view of an indexed row.

        Args:
            batch (BatchIndex): The owning batch index.
            position (int): Row position.

        Returns:
            PendingRow: The transient row view.
        """

        return PendingRow.for_span(
            source=batch.source,
            entity_id=batch.entity_id_at(position=position),
            doc_span=batch.span(position=position),
        )

    def remove_entity(self, entity_id: int) -> None:
        """
        Drops every trace of an entity (pending row, shell, decoded).

        Args:
            entity_id (int): Global 64-bit entity handle to forget.
        """

        self._dropped.add(entity_id)
        self._static_rows.pop(entity_id, None)
        self._shells.pop(entity_id, None)
        self.decoded.pop(entity_id, None)
        self._version_snapshots.pop(entity_id, None)
        self._in_progress.discard(entity_id)

    def register(self, row: PendingRow) -> None:
        """
        Registers a single (usually static) undecoded row.

        Batch documents go through register_batch instead.

        Args:
            row (PendingRow): The row to register.
        """

        self._static_rows[row.entity_id] = row

    def pending_row(self, entity_id: int) -> Optional[PendingRow]:
        """
        Returns the pending row for an entity id.

        Args:
            entity_id (int): Global entity handle.

        Returns:
            Optional[PendingRow]: The pending row, or None.
        """

        return self._locate(entity_id=entity_id)

    def has_pending(self, entity_id: int) -> bool:
        """
        Checks whether an entity id has an undecoded row.

        Args:
            entity_id (int): Global entity handle.

        Returns:
            bool: True if the row is pending.
        """

        return self._locate(entity_id=entity_id) is not None

    def pending_entity_ids(self) -> set[int]:
        """
        Returns every pending entity id.

        Returns:
            set[int]: Pending entity ids.
        """

        entity_ids: set[int] = set(self._static_rows)

        for batches in self._batches.values():

            for batch in batches:
                entity_ids.update(batch.entity_ids)

        return entity_ids - self._dropped

    def pending_entity_id(self, struct_name: str, local_id: int) -> Optional[int]:
        """
        Resolves the entity id of a pending struct/local pair.

        Args:
            struct_name (str): Owning struct name.
            local_id (int): Local instance index.

        Returns:
            Optional[int]: The entity id, or None.
        """

        for batch in reversed(self._batches.get(struct_name, [])):

            if not (batch.local_min <= local_id <= batch.local_max):
                continue

            position: Optional[int] = batch.find_local(local_id=local_id)

            if position is None:
                continue

            entity_id: int = batch.entity_id_at(position=position)

            if entity_id not in self._dropped:
                return entity_id

        for entity_id, row in self._static_rows.items():

            if (
                row.struct_name == struct_name
                and EntityId(value=entity_id).local_id == local_id
            ):
                return entity_id

        return None

    def known_entity_id(self, struct_name: str, local_id: int) -> Optional[int]:
        """
        Resolves the entity id of a struct/local pair, pending or decoded.

        Args:
            struct_name (str): Owning struct name.
            local_id (int): Local instance index.

        Returns:
            Optional[int]: The entity id, or None if unknown.
        """

        pending: Optional[int] = self.pending_entity_id(
            struct_name=struct_name,
            local_id=local_id,
        )

        if pending is not None:
            return pending

        for entity_id, instance in self.decoded.items():
            if (
                instance.struct.name == struct_name
                and instance.entity_id.local_id == local_id
            ):
                return entity_id

        return None

    def pending_local_ids(self, struct_name: str) -> set[int]:
        """
        Returns the local ids of a struct's pending rows.

        Args:
            struct_name (str): Owning struct name.

        Returns:
            set[int]: Pending local ids.
        """

        local_ids: set[int] = {
            EntityId(value=entity_id).local_id
            for entity_id in self._static_rows
            if self._static_rows[entity_id].struct_name == struct_name
        }

        for batch in self._batches.get(struct_name, []):
            local_ids.update(batch.local_ids)

        return {
            local_id
            for local_id in local_ids
            if self.pending_entity_id(
                struct_name=struct_name, local_id=local_id
            )
            is not None
        }

    def pending_payload(self, entity_id: int) -> Optional[bytes]:
        """
        Returns the raw encoded payload of a pending row.

        Args:
            entity_id (int): Global entity handle.

        Returns:
            Optional[bytes]: The payload, or None if decoded or unknown.
        """

        row: Optional[PendingRow] = self._locate(entity_id=entity_id)

        return row.payload if row is not None else None

    def drop_pending(self, entity_id: int) -> None:
        """
        Forgets a pending row (e.g. overwritten by a runtime instance).

        Args:
            entity_id (int): Global entity handle.
        """

        self._dropped.add(entity_id)
        self._static_rows.pop(entity_id, None)
        self._shells.pop(entity_id, None)

    def reregister_static_row(
        self,
        entity_id: int,
        struct_name: str,
        columns: list[str],
        types: dict[str, Type],
        payload: bytes,
    ) -> None:
        """
        Re-points an entity's row at a fresh in-memory payload.

        Used after a flush write-back: the on-disk row the index still
        points at is now superseded by the appended delta, so eviction
        must re-decode from the new payload instead.

        Args:
            entity_id (int): Global entity handle.
            struct_name (str): Owning struct name.
            columns (list[str]): Column names in order.
            types (dict[str, Type]): Declared type per column.
            payload (bytes): The newly written row payload.
        """

        self._dropped.discard(entity_id)

        self._static_rows[entity_id] = PendingRow(
            entity_id=entity_id,
            struct_name=struct_name,
            columns=columns,
            types=types,
            payload=payload,
        )

    def resolve(self, entity_id: int) -> StructInstance:
        """
        Resolves an entity id to its StructInstance, decoding on demand.

        Args:
            entity_id (int): Global entity handle.

        Returns:
            StructInstance: The materialized instance.

        Raises:
            KeyError: If the entity id is unknown.
        """

        if entity_id in self.decoded:
            self.decoded.move_to_end(entity_id)
            return self.decoded[entity_id]

        row: Optional[PendingRow] = self._locate(entity_id=entity_id)

        if row is None:
            raise KeyError(f"Unknown entity id {entity_id} during row resolution")

        # Re-entrant references resolve to a stable shell instance that is
        # filled in once its own row finishes decoding
        if entity_id in self._in_progress:
            return self._shells[entity_id]

        if self._struct_lookup is None:
            raise RuntimeError("RowDecoder has no struct lookup installed")

        struct: Struct = self._struct_lookup(row.struct_name)

        shell: StructInstance = StructInstance(
            struct=struct,
            entity_id=EntityId(value=entity_id),
        )

        self._shells[entity_id] = shell
        self._in_progress.add(entity_id)

        try:
            attributes: dict[str, Any] = self._decode_attributes(row=row)

        finally:
            self._in_progress.discard(entity_id)

        shell.attributes = attributes

        # Stamp resolved extents onto symbolic tensor payloads
        self._stamp_symbolic_dims(struct=struct, attributes=attributes)

        self.decoded[entity_id] = shell
        self._shells.pop(entity_id, None)
        shell.dirty = False
        self._snapshot_versions(instance=shell)
        self._evict_if_over_bound()

        return shell

    def _evict_if_over_bound(self) -> None:
        """
        Evicts oldest clean instances until the cache fits its bound.

        Scans a small window from the LRU front (the oldest entries are
        the likeliest evictable); stops early when the window is all
        dirty or pinned, deferring to the next clear_cache().
        """

        while len(self.decoded) > self.max_decoded:

            evicted: bool = False
            scanned: int = 0

            for entity_id, instance in self.decoded.items():

                scanned += 1

                if scanned > _EVICT_SCAN_WINDOW:
                    break

                if instance.pinned or self.is_dirty(instance=instance):
                    continue

                self._remove_decoded(entity_id=entity_id)
                evicted = True

                break

            if not evicted:
                return

    def _stamp_symbolic_dims(self, struct: Struct, attributes: dict[str, Any]) -> None:
        """
        Fills concrete extents of decoded symbolic tensor payloads.

        The extents live in the row's own const attributes; tensors decode
        count-prefixed so this stamping is optional metadata for shape().

        Args:
            struct (Struct): Schema of the decoded row.
            attributes (dict[str, Any]): Decoded attribute Values.
        """

        for attr_name, attr_def in struct.get_all_attributes().items():

            if not (
                isinstance(attr_def.attr_type, TypeTensor)
                and attr_def.attr_type.is_symbolic
            ):
                continue

            tensor_value: Any = attributes.get(attr_name)

            if not isinstance(tensor_value, ValueTensor):
                continue

            extents: dict[str, int] = {}
            all_bound: bool = True

            for dim_name in attr_def.attr_type.symbolic_dims():
                dim_value: Any = attributes.get(dim_name)

                if dim_value is None:
                    all_bound = False

                    break

                raw_dim: Any = (
                    dim_value.get() if isinstance(dim_value, Value) else dim_value
                )

                if isinstance(raw_dim, bool) or not isinstance(raw_dim, int):
                    all_bound = False

                    break

                extents[dim_name] = raw_dim

            if all_bound:
                tensor_value.concrete_dims = [
                    dim if isinstance(dim, int) else extents[dim]
                    for dim in attr_def.attr_type.dims
                ]

    def _decode_attributes(self, row: PendingRow) -> dict[str, Any]:
        """
        Decodes one pending row into attribute Values.

        Args:
            row (PendingRow): The row to decode.

        Returns:
            dict[str, Any]: Attribute Values by name.
        """

        attributes: dict[str, Any]
        attributes, _ = decode_row(
            payload=row.payload,
            offset=0,
            columns=row.columns,
            types=row.types,
            resolver=self.resolve,
        )

        return attributes


def _struct_of(entity_id: int, batches: dict[str, list["BatchIndex"]]) -> str:
    """
    Placeholder kept for type clarity (unused lookup helper).

    Args:
        entity_id (int): Global entity handle.
        batches (dict[str, list[BatchIndex]]): Batch registry.

    Returns:
        str: Empty string (lookup iterates all batches instead).
    """

    return ""


class LazyInstanceDict(dict):  # type: ignore[type-arg]
    """
    Struct table instance mapping decoding pending rows on first access.

    Satisfies the dict[int, StructInstance] interface used across the
    engine; reads materialize the underlying rows transparently, while
    writes replace any pending payload with a live instance.

    Attributes:
        decoder (RowDecoder): Shared pending row registry and resolver.
        struct_name (str): Name of the struct this mapping belongs to.
    """

    def __init__(self, decoder: RowDecoder, struct_name: str) -> None:
        """
        Initializes the LazyInstanceDict instance.

        Args:
            decoder (RowDecoder): Pending row registry.
            struct_name (str): Owning struct name.
        """

        super().__init__()

        self.decoder: RowDecoder = decoder
        self.struct_name: str = struct_name

    def evict_entity(self, entity_id: int) -> None:
        """
        Drops the live entry for an entity, restoring pending decode.

        Only removes the entry if it still maps to the given entity (a
        newer write under the same key is left untouched). The pending
        row stays registered, so the next read re-decodes from disk.

        Args:
            entity_id (int): Global entity id to evict.
        """

        local_id: int = entity_id & EntityId.LOCAL_ID_MASK

        if not dict.__contains__(self, local_id):
            return

        current: Any = dict.__getitem__(self, local_id)

        if isinstance(current, StructInstance) and (
            current.entity_id.value == entity_id
        ):
            dict.__delitem__(self, local_id)

    def _materialize(self, key: int) -> None:
        """
        Decodes the pending row for a key, if any.

        Args:
            key (int): Local instance id.
        """

        if dict.__contains__(self, key):
            return

        entity_id: Optional[int] = self.decoder.known_entity_id(
            struct_name=self.struct_name,
            local_id=key,
        )

        if entity_id is None:
            return

        instance: StructInstance = self.decoder.resolve(entity_id=entity_id)

        dict.__setitem__(self, key, instance)

    def _local_to_entity(self, local_id: int) -> int:
        """
        Builds the global entity id for a local id using the table's type.

        Args:
            local_id (int): Local instance index.

        Returns:
            int: The encoded global entity id.

        Raises:
            KeyError: If no pending row exists for the local id.
        """

        entity_id: Optional[int] = self.decoder.pending_entity_id(
            struct_name=self.struct_name,
            local_id=local_id,
        )

        if entity_id is None:
            raise KeyError(
                f"No pending row for local id {local_id} of '{self.struct_name}'"
            )

        return entity_id

    def __getitem__(self, key: int) -> StructInstance:
        """
        Returns the instance for a local id, decoding it if pending.

        Args:
            key (int): Local instance id.

        Returns:
            StructInstance: The materialized instance.
        """

        self._materialize(key=key)

        instance: StructInstance = dict.__getitem__(self, key)

        return instance

    def __contains__(self, key: object) -> bool:
        """
        Checks membership without decoding.

        Args:
            key (object): Candidate local id.

        Returns:
            bool: True if the local id is present (pending or decoded).
        """

        if not isinstance(key, int):
            return False

        if dict.__contains__(self, key):
            return True

        return self.decoder.known_entity_id(
            struct_name=self.struct_name,
            local_id=key,
        ) is not None

    def __len__(self) -> int:
        """
        Returns the total instance count (decoded plus pending).

        Returns:
            int: Instance count.
        """

        return dict.__len__(self) + len(self._pending_keys())

    def _pending_keys(self) -> set[int]:
        """
        Returns the local ids of this struct's pending rows.

        Returns:
            set[int]: Pending local ids.
        """

        return self.decoder.pending_local_ids(struct_name=self.struct_name)

    def __iter__(self) -> Iterator[int]:
        """
        Iterates every local id, decoding all pending rows first.

        Returns:
            Iterator[int]: Local id iterator.
        """

        self.materialize_all()

        return dict.__iter__(self)

    def keys(self) -> Any:
        """
        Returns all local ids, decoding pending rows first.

        Returns:
            Any: Decoded local ids.
        """

        self.materialize_all()

        return dict.keys(self)

    def values(self) -> Any:
        """
        Returns all instances, decoding pending rows first.

        Returns:
            Any: Decoded instance values.
        """

        self.materialize_all()

        return dict.values(self)

    def items(self) -> Any:
        """
        Returns all (local id, instance) pairs, decoding pending rows.

        Returns:
            Any: Decoded items.
        """

        self.materialize_all()

        return dict.items(self)

    def get(self, key: int, default: Any = None) -> Any:
        """
        Returns the instance for a local id or a default.

        Args:
            key (int): Local instance id.
            default (Any, optional): Fallback value. Defaults to None.

        Returns:
            Any: The instance or the default.
        """

        if key in self:
            return self[key]

        return default

    def __setitem__(self, key: int, value: StructInstance) -> None:
        """
        Stores an instance, dropping any pending row for the key.

        Args:
            key (int): Local instance id.
            value (StructInstance): The instance to store.
        """

        if dict.__contains__(self, key):
            existing_id: int = dict.__getitem__(self, key).entity_id.value

            self.decoder.drop_pending(entity_id=existing_id)

        else:
            entity_id: Optional[int] = self.decoder.known_entity_id(
                struct_name=self.struct_name,
                local_id=key,
            )

            if entity_id is not None:
                self.decoder.drop_pending(entity_id=entity_id)

        dict.__setitem__(self, key, value)

    def materialize_all(self) -> None:
        """
        Decodes every pending row of this struct.
        """

        for local_id in self._pending_keys():
            self._materialize(key=local_id)

    def all_local_ids(self) -> list[int]:
        """
        Returns every local id (decoded and pending) without decoding.

        Returns:
            list[int]: Sorted local ids.
        """

        return sorted(set(dict.keys(self)) | self._pending_keys())

    def raw_row_if_pending(
        self,
        local_id: int,
        columns: Optional[list[str]] = None,
    ) -> Optional[bytes]:
        """
        Returns the encoded payload of an untouched pending row.

        Used by the dumper to re-emit rows that were never accessed
        without decoding them. When a column layout is given, the payload
        is only returned if the row was encoded with the same layout.

        Args:
            local_id (int): Local instance id.
            columns (Optional[list[str]], optional): Expected column
                layout. Defaults to None (any layout).

        Returns:
            Optional[bytes]: The raw row payload, or None.
        """

        if dict.__contains__(self, local_id):
            return None

        try:
            entity_id: int = self._local_to_entity(local_id=local_id)

        except KeyError:
            return None

        row: Optional[PendingRow] = self.decoder.pending_row(entity_id=entity_id)

        if row is None:
            return None

        if columns is not None and row.columns != columns:
            return None

        return row.payload

    def entity_id_if_pending(self, local_id: int) -> Optional[int]:
        """
        Returns the entity id of an untouched pending row.

        Args:
            local_id (int): Local instance id.

        Returns:
            Optional[int]: The entity id, or None if decoded or unknown.
        """

        if dict.__contains__(self, local_id):
            return None

        try:
            return self._local_to_entity(local_id=local_id)

        except KeyError:
            return None

    def row_attributes(self, local_id: int) -> Optional[dict[str, Any]]:
        """
        Decodes a single pending row and returns its raw attribute payloads.

        Used for change detection without materializing every instance.

        Args:
            local_id (int): Local instance id.

        Returns:
            Optional[dict[str, Any]]: Raw payloads by name, or None when
                the row is not pending.
        """

        try:
            entity_id: int = self._local_to_entity(local_id=local_id)

        except KeyError:
            return None

        if not self.decoder.has_pending(entity_id=entity_id):
            return None

        instance: StructInstance = self.decoder.resolve(entity_id=entity_id)

        dict.__setitem__(self, local_id, instance)

        return {name: value.get() for name, value in instance.attributes.items()}
