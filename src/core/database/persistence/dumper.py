"""
Database dumper serializing a resolved NaDataDatabase into slotted pages.

Writes meta information, struct definitions (with full AST bodies), standalone
functions, instances, and symbol bindings as chunked records through a
RecordStore backed by a file based PageManager.

Struct definitions, functions, and symbols use the JSON codec; instance rows
use the compact binary format (format v2) driven by each struct's declared
attribute types. Lazy pending rows that were never decoded are re-emitted
byte for byte without materialization.
"""

# Import Modules
from __future__ import annotations
import json
import os
from typing import Any, Optional

from src.core.database.database import NaDataDatabase
from src.core.database.entity_id import EntityId
from src.core.database.instance import StructInstance
from src.core.database.page_manager import PageManager
from src.core.database.persistence.binary_cells import encode_batch_document, encode_row
from src.core.database.persistence.loader import load_database
from src.core.database.persistence.record_store import RecordStore
from src.core.database.persistence.serialization import ObjectCodec
from src.core.struct.struct import Struct

# Rows per binary batch document
ROWS_PER_BATCH: int = 512

# Format version written by this dumper and accepted by the loader
FORMAT_VERSION: int = 2


def dump_database(database: NaDataDatabase, storage_path: str) -> None:
    """
    Serializes a fully resolved database into a slotted page file.

    Args:
        database (NaDataDatabase): Database to serialize (structs, functions,
            instances, and symbols).
        storage_path (str): Target file path (must not exist yet).

    Raises:
        ValueError: If the target file already exists.
    """

    if os.path.exists(storage_path):
        raise ValueError(f"Storage path '{storage_path}' already exists")

    page_manager: PageManager = PageManager(storage_path=storage_path)
    record_store: RecordStore = RecordStore(page_manager=page_manager)
    codec: ObjectCodec = ObjectCodec()

    # Meta document: format version, type id counter, and settings
    meta_doc: dict[str, Any] = {
        "kind": "meta",
        "format_version": FORMAT_VERSION,
        "next_type_id": database.next_type_id,
        "settings": dict(database.settings),
    }
    record_store.put_document(payload=_json_bytes(doc=meta_doc))

    # Struct definition documents (insertion order keeps type ids stable)
    for table in database.tables_by_struct_name.values():
        struct_doc: dict[str, Any] = {
            "kind": "struct",
            "struct": _encode_struct(struct=table.struct, codec=codec),
        }
        record_store.put_document(payload=_json_bytes(doc=struct_doc))

    # Standalone function documents
    for function in database.functions.values():
        function_doc: dict[str, Any] = {
            "kind": "function",
            "method": codec.encode(obj=function),
        }
        record_store.put_document(payload=_json_bytes(doc=function_doc))

    # Instance batch documents (per table, entity ids preserved)
    for table in database.tables_by_struct_name.values():
        for batch_payload in _encode_table_batches(table=table):
            record_store.put_document(payload=batch_payload)

    # Symbol binding document (+ nid identity map, merged on load)
    symbols_doc: dict[str, Any] = {
        "kind": "symbols",
        "symbols": {
            name: entity_id.value for name, entity_id in database.symbols.items()
        },
        "nids": {
            key: entity_id.value for key, entity_id in database.nids.items()
        },
    }
    record_store.put_document(payload=_json_bytes(doc=symbols_doc))

    page_manager.flush()


def _all_local_ids(instances: Any) -> list[int]:
    """
    Returns every local id of an instance mapping without decoding.

    Args:
        instances (Any): The instance mapping (possibly lazy).

    Returns:
        list[int]: Sorted local ids.
    """

    if hasattr(instances, "all_local_ids"):
        lazy_ids: list[int] = instances.all_local_ids()

        return lazy_ids

    return sorted(instances.keys())


def _encode_table_batches(table: Any) -> list[bytes]:
    """
    Encodes a struct table's instances into binary batch documents.

    Undecoded lazy rows are copied byte for byte; live instances are
    encoded against the struct's declared attribute types.

    Args:
        table (Any): StructTable whose instances to encode.

    Returns:
        list[bytes]: One binary batch document payload per batch.
    """

    struct: Struct = table.struct
    columns: list[str] = list(struct.get_all_attributes().keys())
    types: dict[str, Any] = {}

    for name in columns:
        attribute: Any = struct.get_attribute(name=name)

        if attribute is None:
            raise KeyError(
                f"Column '{name}' is not declared in struct '{struct.name}'"
            )

        types[name] = attribute.attr_type

    raw_rows: list[tuple[int, bytes]] = []

    # Spatially indexed tables cluster rows by cell key so "load
    # district X" reads contiguous pages
    ordered_ids: list[int] = sorted(_all_local_ids(instances=table.instances))

    if getattr(table, "spatial_indexes", None):
        spatial: Any = next(iter(table.spatial_indexes.values()))

        ordered_ids.sort(
            key=lambda local_id: spatial.sort_key(
                local_id=local_id, table=table
            )
        )

    for local_id in ordered_ids:
        raw_row: Optional[bytes] = None

        if hasattr(table.instances, "raw_row_if_pending"):
            raw_row = table.instances.raw_row_if_pending(
                local_id=local_id, columns=columns
            )

        if raw_row is not None:
            entity_id: Optional[int] = None

            if hasattr(table.instances, "entity_id_if_pending"):
                entity_id = table.instances.entity_id_if_pending(local_id=local_id)

            if entity_id is None:
                # Cannot identify the row without decoding: fall back
                fallback: StructInstance = table.instances[local_id]

                raw_rows.append((
                    fallback.entity_id.value,
                    encode_row(
                        attributes=fallback.attributes,
                        columns=columns,
                        types=types,
                    ),
                ))

                continue

            raw_rows.append((entity_id, raw_row))

            continue

        instance: StructInstance = table.instances[local_id]

        row_payload: bytes = encode_row(
            attributes=instance.attributes,
            columns=columns,
            types=types,
        )

        raw_rows.append((instance.entity_id.value, row_payload))

    batches: list[bytes] = []

    for offset in range(0, len(raw_rows), ROWS_PER_BATCH):
        chunk: list[tuple[int, bytes]] = raw_rows[
            offset:offset + ROWS_PER_BATCH
        ]

        max_local_id: int = max(
            (
                EntityId(value=entity_id).local_id
                for entity_id, _ in chunk
                if entity_id is not None
            ),
            default=0,
        )

        batches.append(
            encode_batch_document(
                struct_name=struct.name,
                columns=columns,
                rows=chunk,
                max_local_id=max_local_id,
            )
        )

    return batches


def _json_bytes(doc: dict[str, Any]) -> bytes:
    """
    Serializes a document dict to compact UTF-8 JSON bytes.

    Args:
        doc (dict[str, Any]): Document to serialize.

    Returns:
        bytes: UTF-8 encoded JSON payload.
    """

    return json.dumps(doc, separators=(",", ":")).encode("utf-8")


def _encode_struct(struct: Struct, codec: ObjectCodec) -> dict[str, Any]:
    """
    Encodes a struct definition into a JSON compatible document.

    The parent is encoded by name to avoid cyclic recursion; attribute types
    referencing structs encode as named references inside the codec.

    Args:
        struct (Struct): Struct to encode.
        codec (ObjectCodec): Codec used for types, defaults, and methods.

    Returns:
        dict[str, Any]: Encoded struct definition.
    """

    return {
        "name": struct.name,
        "parent": struct.parent.name if struct.parent is not None else None,
        "attributes": [
            {
                "name": attribute.name,
                "attr_type": codec.encode(obj=attribute.attr_type),
                "default_value": (
                    codec.encode(obj=attribute.default_value)
                    if attribute.has_default
                    else None
                ),
                "has_default": attribute.has_default,
                "is_const": attribute.is_const,
                "default_expr": (
                    codec.encode(obj=attribute.default_expr)
                    if attribute.default_expr is not None
                    else None
                ),
            }
            for attribute in struct.attributes.values()
        ],
        "methods": [codec.encode(obj=method) for method in struct.methods.values()],
    }


def append_database_changes(database: NaDataDatabase, storage_path: str) -> int:
    """
    Appends only the new or changed data of a database to an existing file.

    The file is a replay log: the loader processes documents in insertion
    order, so appended delta records (new structs, new or changed instances,
    new symbol bindings) override the earlier full snapshot. The current
    state is diffed against the file contents to compute the delta.

    Args:
        database (NaDataDatabase): Database whose changes are appended.
        storage_path (str): Path to the existing .ndb file to extend.

    Returns:
        int: Number of documents appended.

    Raises:
        ValueError: If the file does not exist or is not a Nadata dump.
    """

    if not os.path.exists(storage_path):
        raise ValueError(
            f"Storage path '{storage_path}' does not exist: "
            f"use dump_database for an initial full dump"
        )

    # Reference state: everything currently stored in the file. Lazy rows
    # decode one at a time, so the diff only touches rows the live database
    # claims changed.
    reference: NaDataDatabase = load_database(storage_path=storage_path)

    ref_struct_fields: dict[str, set[str]] = {}
    ref_row_attrs: dict[str, Any] = {}
    ref_symbols: dict[str, int] = {
        name: entity_id.value for name, entity_id in reference.symbols.items()
    }

    for ref_table in reference.tables_by_struct_name.values():
        ref_struct: Struct = ref_table.struct

        ref_struct_fields[ref_struct.name] = (
            set(ref_struct.attributes) | set(ref_struct.methods)
        )

        ref_row_attrs[ref_struct.name] = ref_table.instances

    page_manager: PageManager = PageManager(storage_path=storage_path)
    existing_store: RecordStore = RecordStore(page_manager=page_manager)
    document_count: int = len(existing_store.read_all_documents())

    record_store: RecordStore = RecordStore(
        page_manager=page_manager,
        next_document_id=document_count,
    )
    codec: ObjectCodec = ObjectCodec()
    written: int = 0

    # Meta document: the loader takes the latest one (settings included)
    meta_doc: dict[str, Any] = {
        "kind": "meta",
        "format_version": FORMAT_VERSION,
        "next_type_id": database.next_type_id,
        "settings": dict(database.settings),
    }
    record_store.put_document(payload=_json_bytes(doc=meta_doc))
    written += 1

    # Struct deltas: full definitions for new structs, new fields for existing
    for table in database.tables_by_struct_name.values():
        struct: Struct = table.struct

        if struct.name not in ref_struct_fields:
            struct_doc: dict[str, Any] = {
                "kind": "struct",
                "struct": _encode_struct(struct=struct, codec=codec),
            }
            record_store.put_document(payload=_json_bytes(doc=struct_doc))
            written += 1

            continue

        new_attributes: list[Any] = [
            attribute
            for name, attribute in struct.attributes.items()
            if name not in ref_struct_fields[struct.name]
        ]
        new_methods: list[Any] = [
            method
            for name, method in struct.methods.items()
            if name not in ref_struct_fields[struct.name]
        ]

        if not new_attributes and not new_methods:
            continue

        delta_doc: dict[str, Any] = {
            "kind": "struct",
            "struct": {
                "name": struct.name,
                "parent": None,
                "attributes": [
                    {
                        "name": attribute.name,
                        "attr_type": codec.encode(obj=attribute.attr_type),
                        "default_value": (
                            codec.encode(obj=attribute.default_value)
                            if attribute.has_default
                            else None
                        ),
                        "has_default": attribute.has_default,
                        "is_const": attribute.is_const,
                        "default_expr": (
                            codec.encode(obj=attribute.default_expr)
                            if attribute.default_expr is not None
                            else None
                        ),
                    }
                    for attribute in new_attributes
                ],
                "methods": [codec.encode(obj=method) for method in new_methods],
            },
        }
        record_store.put_document(payload=_json_bytes(doc=delta_doc))
        written += 1

    # Instance deltas: full attribute snapshots for new or changed instances
    for table in database.tables_by_struct_name.values():
        ref_instances: Any = ref_row_attrs.get(table.struct.name)
        changed_rows: list[tuple[int, bytes]] = []

        columns: list[str] = list(table.struct.get_all_attributes().keys())
        types: dict[str, Any] = {
            name: table.struct.get_attribute(name=name).attr_type  # type: ignore[union-attr]
            for name in columns
        }

        for local_id in _all_local_ids(instances=table.instances):
            instance: StructInstance = table.instances[local_id]
            entity_id: int = instance.entity_id.value

            reference_attrs: Optional[dict[str, Any]] = _reference_attributes(
                instances=ref_instances,
                local_id=local_id,
                entity_id=entity_id,
            )

            current_attrs: dict[str, Any] = {
                name: value.get() for name, value in instance.attributes.items()
            }

            if reference_attrs is not None and _attributes_equal(
                current=current_attrs, reference=reference_attrs
            ):
                continue

            changed_rows.append((
                entity_id,
                encode_row(
                    attributes=instance.attributes,
                    columns=columns,
                    types=types,
                ),
            ))

        if changed_rows:
            batch_payload: bytes = encode_batch_document(
                struct_name=table.struct.name,
                columns=columns,
                rows=changed_rows,
                max_local_id=max(
                    (
                        EntityId(value=entity_id).local_id
                        for entity_id, _ in changed_rows
                    ),
                    default=0,
                ),
            )

            record_store.put_document(payload=batch_payload)
            written += 1

    # Symbol deltas: new or re-bound symbols only
    changed_symbols: dict[str, int] = {
        name: entity_id.value
        for name, entity_id in database.symbols.items()
        if ref_symbols.get(name) != entity_id.value
    }

    if changed_symbols:
        symbols_doc: dict[str, Any] = {
            "kind": "symbols",
            "symbols": changed_symbols,
            "nids": {
                key: entity_id.value
                for key, entity_id in database.nids.items()
            },
        }
        record_store.put_document(payload=_json_bytes(doc=symbols_doc))
        written += 1

    page_manager.flush()

    return written


def flush_dirty_instances(database: NaDataDatabase, storage_path: str) -> int:
    """
    Appends only instances flagged dirty since their last persistence.

    The streaming write-back path: no reference file is loaded and no
    diff is computed — the dirty flags (and tensor version snapshots)
    decide what is written. Appended batch rows override earlier rows
    for the same entity id on the next load. After a successful flush,
    written instances are marked clean.

    Args:
        database (NaDataDatabase): Database whose dirty instances to
            write.
        storage_path (str): Path to the existing .ndb file to extend.

    Returns:
        int: Number of instances written.

    Raises:
        ValueError: If the file does not exist.
    """

    if not os.path.exists(storage_path):
        raise ValueError(
            f"Storage path '{storage_path}' does not exist: "
            f"use dump_database for an initial full dump"
        )

    page_manager: PageManager = PageManager(storage_path=storage_path)
    probe: RecordStore = RecordStore(page_manager=page_manager)
    document_count: int = len(probe.document_locations())

    record_store: RecordStore = RecordStore(
        page_manager=page_manager,
        next_document_id=document_count,
    )

    meta_doc: dict[str, Any] = {
        "kind": "meta",
        "format_version": FORMAT_VERSION,
        "next_type_id": database.next_type_id,
        "settings": dict(database.settings),
    }
    record_store.put_document(payload=_json_bytes(doc=meta_doc))

    written: int = 0

    for table in database.tables_by_struct_name.values():

        struct: Struct = table.struct
        columns: list[str] = list(struct.get_all_attributes().keys())
        types: dict[str, Any] = {
            name: struct.get_attribute(name=name).attr_type  # type: ignore[union-attr]
            for name in columns
        }

        dirty_rows: list[tuple[int, bytes]] = []
        dirty_instances: list[StructInstance] = []
        row_payloads: dict[int, bytes] = {}
        seen: set[int] = set()

        decoder: Any = getattr(table.instances, "decoder", None)

        if decoder is not None:
            for entity_id in decoder.dirty_decoded_entity_ids():

                instance: Optional[StructInstance] = (
                    decoder.decoded.get(entity_id)
                )

                if instance is None or entity_id in seen:
                    continue

                seen.add(entity_id)
                dirty_instances.append(instance)

        for instance in dict.values(table.instances):

            if not instance.dirty or instance.entity_id.value in seen:
                continue

            seen.add(instance.entity_id.value)
            dirty_instances.append(instance)

        for instance in dirty_instances:

            row_payload: bytes = encode_row(
                attributes=instance.attributes,
                columns=columns,
                types=types,
            )

            dirty_rows.append((instance.entity_id.value, row_payload))
            row_payloads[instance.entity_id.value] = row_payload

        for offset in range(0, len(dirty_rows), ROWS_PER_BATCH):

            batch: list[tuple[int, bytes]] = (
                dirty_rows[offset:offset + ROWS_PER_BATCH]
            )

            batch_payload: bytes = encode_batch_document(
                struct_name=struct.name,
                columns=columns,
                rows=batch,
                max_local_id=max(
                    (
                        EntityId(value=entity_id).local_id
                        for entity_id, _ in batch
                    ),
                    default=0,
                ),
            )

            record_store.put_document(payload=batch_payload)
            written += len(batch)

        for instance in dirty_instances:

            instance.dirty = False

            if decoder is not None:
                decoder.mark_clean(instance=instance)

        # Re-point row indexes at the freshly written payloads so a later
        # eviction re-decodes the new state, not the superseded row
        if decoder is not None:
            for instance in dirty_instances:

                decoder.reregister_static_row(
                    entity_id=instance.entity_id.value,
                    struct_name=struct.name,
                    columns=columns,
                    types=types,
                    payload=row_payloads[instance.entity_id.value],
                )

    # Nid identity map rides along so upserted instances keep their
    # named identity after the flush (merged on load, full map is safe)
    if written > 0:
        record_store.put_document(payload=_json_bytes(doc={
            "kind": "symbols",
            "symbols": {},
            "nids": {
                key: entity_id.value
                for key, entity_id in database.nids.items()
            },
        }))

    page_manager.flush()

    return written


def append_instance_removal(
    database: NaDataDatabase,
    storage_path: str,
    entity_id: int,
) -> None:
    """
    Appends a tombstone removing one entity from the stored database.

    The tombstone is a replay-log record: the loader drops the entity
    (and any pending row for it) when it reaches this document.

    Args:
        database (NaDataDatabase): Database the entity belongs to (its
            settings ride along in the meta doc).
        storage_path (str): Path to the existing .ndb file to extend.
        entity_id (int): Global 64-bit entity handle to remove.

    Raises:
        ValueError: If the file does not exist.
    """

    if not os.path.exists(storage_path):
        raise ValueError(
            f"Storage path '{storage_path}' does not exist: "
            f"use dump_database for an initial full dump"
        )

    page_manager: PageManager = PageManager(storage_path=storage_path)
    probe: RecordStore = RecordStore(page_manager=page_manager)
    document_count: int = len(probe.document_locations())

    record_store: RecordStore = RecordStore(
        page_manager=page_manager,
        next_document_id=document_count,
    )

    meta_doc: dict[str, Any] = {
        "kind": "meta",
        "format_version": FORMAT_VERSION,
        "next_type_id": database.next_type_id,
        "settings": dict(database.settings),
    }
    record_store.put_document(payload=_json_bytes(doc=meta_doc))

    tombstone_doc: dict[str, Any] = {
        "kind": "instance_removal",
        "entity_id": entity_id,
    }
    record_store.put_document(payload=_json_bytes(doc=tombstone_doc))

    # Drop any pending row index entry so the dead row is never decoded
    type_id: int = EntityId(value=entity_id).type_id

    if type_id in database.tables_by_type_id:

        decoder: Any = getattr(
            database.tables_by_type_id[type_id].instances, "decoder", None
        )

        if decoder is not None:
            decoder.remove_entity(entity_id=entity_id)

    page_manager.flush()


def _reference_attributes(
    instances: Any,
    local_id: int,
    entity_id: int,
) -> Optional[dict[str, Any]]:
    """
    Returns the stored raw attributes of one reference instance.

    Decodes only this single row when the reference mapping is lazy.

    Args:
        instances (Any): Reference instance mapping (possibly lazy).
        local_id (int): Local id of the instance.
        entity_id (int): Global entity handle of the instance.

    Returns:
        Optional[dict[str, Any]]: Raw payloads by name, or None when the
            instance is unknown to the reference file.
    """

    if hasattr(instances, "row_attributes"):
        row_attrs: Optional[dict[str, Any]] = instances.row_attributes(
            local_id=local_id
        )

        if row_attrs is not None:
            return row_attrs

    if local_id not in instances:
        return None

    return {
        name: value.get() for name, value in instances[local_id].attributes.items()
    }


def _attributes_equal(current: dict[str, Any], reference: dict[str, Any]) -> bool:
    """
    Compares two raw instance attribute dictionaries for equality.

    Args:
        current (dict[str, Any]): Current attribute raw payloads by name.
        reference (dict[str, Any]): Reference attribute raw payloads by name.

    Returns:
        bool: True if both dictionaries hold equal raw payloads.
    """

    if set(current) != set(reference):
        return False

    return all(
        _raw_equal(current=name_value, reference=reference[name])
        for name, name_value in current.items()
    )


def _raw_equal(current: Any, reference: Any) -> bool:
    """
    Compares two raw payloads, resolving instance references by entity id.

    Args:
        current (Any): Current raw payload.
        reference (Any): Reference raw payload.

    Returns:
        bool: True if both payloads are equal.
    """

    if isinstance(current, StructInstance) and isinstance(reference, StructInstance):
        return current.entity_id.value == reference.entity_id.value

    if isinstance(current, list) and isinstance(reference, list):
        return len(current) == len(reference) and all(
            _raw_equal(current=current_element, reference=reference_element)
            for current_element, reference_element in zip(current, reference)
        )

    if isinstance(current, dict) and isinstance(reference, dict):
        return set(current) == set(reference) and all(
            _raw_equal(current=current[key], reference=reference[key])
            for key in current
        )

    return bool(current == reference)
