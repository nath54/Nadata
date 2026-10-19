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
from src.core.database.page_manager import PageManager
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

    # Meta document: format version and type id counter
    meta_doc: dict[str, Any] = {
        "kind": "meta",
        "format_version": 1,
        "next_type_id": database.next_type_id,
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
