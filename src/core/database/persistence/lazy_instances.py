"""
Lazy instance materialization for the binary persistence format (v2).

Instance rows are stored as compact binary payloads inside batch
documents. Loading a database registers the rows without decoding them;
StructInstance objects are materialized on first access and cached.
Instance references inside rows resolve on demand through the same
decoder, so cross referenced graphs stream in naturally.
"""

# Import Modules
from __future__ import annotations
from typing import Any, Callable, Iterator, Optional

from src.core.database.entity_id import EntityId
from src.core.database.instance import StructInstance
from src.core.database.persistence.binary_cells import decode_row
from src.core.struct.struct import Struct
from src.core.types.types import Type, TypeTensor
from src.core.values.values import Value, ValueTensor


class PendingRow:
    """
    One undecoded instance row.

    Attributes:
        entity_id (int): Global 64-bit entity handle of the row.
        struct_name (str): Name of the struct the row belongs to.
        columns (list[str]): Column names of the owning batch, in order.
        types (dict[str, Type]): Declared schema type per column.
        payload (bytes): The encoded row payload.
    """

    def __init__(
        self,
        entity_id: int,
        struct_name: str,
        columns: list[str],
        types: dict[str, Type],
        payload: bytes,
    ) -> None:
        """
        Initializes the PendingRow instance.

        Args:
            entity_id (int): Global entity handle.
            struct_name (str): Owning struct name.
            columns (list[str]): Batch column names.
            types (dict[str, Type]): Declared type per column.
            payload (bytes): Encoded row payload.
        """

        self.entity_id: int = entity_id
        self.struct_name: str = struct_name
        self.columns: list[str] = columns
        self.types: dict[str, Type] = types
        self.payload: bytes = payload


class RowDecoder:
    """
    Registry of pending rows with on-demand decoding and ref resolution.

    Attributes:
        decoded (dict[int, StructInstance]): Materialized instances by
            entity id (also the resolver cache).
    """

    def __init__(self) -> None:
        """
        Initializes the RowDecoder instance.
        """

        self.decoded: dict[int, StructInstance] = {}
        self._pending: dict[int, PendingRow] = {}
        self._pending_by_local: dict[tuple[str, int], int] = {}
        self._in_progress: set[int] = set()
        self._shells: dict[int, StructInstance] = {}
        self._struct_lookup: Optional[Callable[[str], Struct]] = None

    def set_struct_lookup(self, lookup: Callable[[str], Struct]) -> None:
        """
        Installs the struct lookup used to resolve row schemas.

        Args:
            lookup (Callable[[str], Struct]): Maps a struct name to its
                Struct definition.
        """

        self._struct_lookup = lookup

    def register(self, row: PendingRow) -> None:
        """
        Registers an undecoded row (later registrations override earlier).

        Args:
            row (PendingRow): The row to register.
        """

        self._pending[row.entity_id] = row

        local_key: tuple[str, int] = (
            row.struct_name,
            EntityId(value=row.entity_id).local_id,
        )

        self._pending_by_local[local_key] = row.entity_id

    def pending_row(self, entity_id: int) -> Optional[PendingRow]:
        """
        Returns the pending row for an entity id.

        Args:
            entity_id (int): Global entity handle.

        Returns:
            Optional[PendingRow]: The pending row, or None.
        """

        return self._pending.get(entity_id)

    def has_pending(self, entity_id: int) -> bool:
        """
        Checks whether an entity id has an undecoded row.

        Args:
            entity_id (int): Global entity handle.

        Returns:
            bool: True if the row is pending.
        """

        return entity_id in self._pending

    def pending_entity_ids(self) -> set[int]:
        """
        Returns every pending entity id.

        Returns:
            set[int]: Pending entity ids.
        """

        return set(self._pending)

    def pending_entity_id(self, struct_name: str, local_id: int) -> Optional[int]:
        """
        Resolves the entity id of a pending row by struct and local id.

        Args:
            struct_name (str): Owning struct name.
            local_id (int): Local instance index.

        Returns:
            Optional[int]: The entity id, or None if not pending.
        """

        return self._pending_by_local.get((struct_name, local_id))

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

        return {
            local_id
            for (name, local_id) in self._pending_by_local
            if name == struct_name
        }

    def pending_payload(self, entity_id: int) -> Optional[bytes]:
        """
        Returns the raw encoded payload of a pending row.

        Args:
            entity_id (int): Global entity handle.

        Returns:
            Optional[bytes]: The payload, or None if decoded or unknown.
        """

        row: Optional[PendingRow] = self._pending.get(entity_id)

        return row.payload if row is not None else None

    def drop_pending(self, entity_id: int) -> None:
        """
        Forgets a pending row (e.g. overwritten by a runtime instance).

        Args:
            entity_id (int): Global entity handle.
        """

        row: Optional[PendingRow] = self._pending.pop(entity_id, None)

        if row is not None:
            local_key: tuple[str, int] = (
                row.struct_name,
                EntityId(value=entity_id).local_id,
            )

            self._pending_by_local.pop(local_key, None)

        self._shells.pop(entity_id, None)

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
            return self.decoded[entity_id]

        row: Optional[PendingRow] = self._pending.get(entity_id)

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
        self.drop_pending(entity_id=entity_id)

        return shell

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

