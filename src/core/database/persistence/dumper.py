"""
Database dumper serializing a resolved NaDataDatabase into slotted pages.

Writes meta information, struct definitions (with full AST bodies), standalone
functions, instances, and symbol bindings as chunked records through a
RecordStore backed by a file based PageManager.
"""

# Import Modules
from __future__ import annotations
import json
import os
from typing import Any

from src.core.database.database import NaDataDatabase
from src.core.database.instance import StructInstance
from src.core.database.page_manager import PageManager
from src.core.database.persistence.loader import load_database
from src.core.database.persistence.record_store import RecordStore
from src.core.database.persistence.serialization import ObjectCodec
from src.core.struct.struct import Struct


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
        "format_version": 1,
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

    # Instance documents (per table, entity ids preserved)
    for table in database.tables_by_struct_name.values():
        for instance in table.instances.values():
            instance_doc: dict[str, Any] = {
                "kind": "instance",
                "struct_name": table.struct.name,
                "entity_id": instance.entity_id.value,
                "attributes": {
                    name: codec.encode(obj=value)
                    for name, value in instance.attributes.items()
                },
            }
            record_store.put_document(payload=_json_bytes(doc=instance_doc))

    # Symbol binding document
    symbols_doc: dict[str, Any] = {
        "kind": "symbols",
        "symbols": {
            name: entity_id.value for name, entity_id in database.symbols.items()
        },
    }
    record_store.put_document(payload=_json_bytes(doc=symbols_doc))

    page_manager.flush()


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

    # Reference state: everything currently stored in the file
    reference: NaDataDatabase = load_database(storage_path=storage_path)

    ref_struct_fields: dict[str, set[str]] = {}
    ref_instance_attrs: dict[int, dict[str, Any]] = {}
    ref_symbols: dict[str, int] = {
        name: entity_id.value for name, entity_id in reference.symbols.items()
    }

    for ref_table in reference.tables_by_struct_name.values():
        ref_struct: Struct = ref_table.struct

        ref_struct_fields[ref_struct.name] = (
            set(ref_struct.attributes) | set(ref_struct.methods)
        )

        for ref_instance in ref_table.instances.values():
            ref_instance_attrs[ref_instance.entity_id.value] = {
                name: value.get() for name, value in ref_instance.attributes.items()
            }

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
        "format_version": 1,
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
        for instance in table.instances.values():
            entity_id: int = instance.entity_id.value
            current_attrs: dict[str, Any] = {
                name: value.get() for name, value in instance.attributes.items()
            }

            if entity_id in ref_instance_attrs and _attributes_equal(
                current=current_attrs, reference=ref_instance_attrs[entity_id]
            ):
                continue

            instance_doc: dict[str, Any] = {
                "kind": "instance",
                "struct_name": table.struct.name,
                "entity_id": entity_id,
                "attributes": {
                    name: codec.encode(obj=value)
                    for name, value in instance.attributes.items()
                },
            }
            record_store.put_document(payload=_json_bytes(doc=instance_doc))
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
        }
        record_store.put_document(payload=_json_bytes(doc=symbols_doc))
        written += 1

    page_manager.flush()

    return written


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
