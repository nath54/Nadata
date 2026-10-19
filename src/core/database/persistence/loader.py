"""
Database loader reconstructing a NaDataDatabase from slotted page records.

Rebuilds structs (shells first, then parents, attributes, and methods so that
forward type references resolve), standalone functions, instances with
preserved entity ids and cross references, and symbol bindings.
"""

# Import Modules
from __future__ import annotations
import json
from typing import Any, cast

from src.core.database.database import NaDataDatabase
from src.core.database.entity_id import EntityId
from src.core.database.instance import StructInstance
from src.core.database.persistence.record_store import RecordStore
from src.core.database.persistence.serialization import (
    DecodeContext,
    ObjectCodec,
    resolve_instance_refs,
)
from src.core.struct.struct import Struct
from src.core.struct.struct_attribute import StructAttribute
from src.core.struct.struct_method import StructMethod
from src.core.types.types import Type


def load_database(storage_path: str) -> NaDataDatabase:
    """
    Loads a database previously written by dump_database.

    Args:
        storage_path (str): Path to the slotted page file.

    Returns:
        NaDataDatabase: Reconstructed database with structs, functions,
            instances, and symbols fully restored.

    Raises:
        ValueError: If the file format is not a Nadata database dump.
    """

    database: NaDataDatabase = NaDataDatabase(storage_path=storage_path)
    record_store: RecordStore = RecordStore(page_manager=database.page_manager)
    codec: ObjectCodec = ObjectCodec()
    context: DecodeContext = DecodeContext()

    documents: list[dict[str, Any]] = [
        json.loads(payload.decode("utf-8"))
        for payload in record_store.read_all_documents().values()
    ]

    meta_docs: list[dict[str, Any]] = [
        doc for doc in documents if doc.get("kind") == "meta"
    ]

    if not meta_docs or meta_docs[0].get("format_version") != 1:
        raise ValueError(f"File '{storage_path}' is not a Nadata database dump")

    struct_docs: list[dict[str, Any]] = [
        doc for doc in documents if doc.get("kind") == "struct"
    ]

    # Phase 1: struct shells registered under stable type ids
    for struct_doc in struct_docs:
        struct: Struct = Struct(name=struct_doc["struct"]["name"])
        database.register_struct(struct=struct)
        context.structs[struct.name] = struct

    # Phase 2: parents, attributes, and methods (struct refs now resolvable)
    for struct_doc in struct_docs:
        _decode_struct(
            struct_doc=struct_doc["struct"],
            struct=context.structs[struct_doc["struct"]["name"]],
            codec=codec,
            context=context,
        )

    # Phase 3: standalone functions
    for doc in documents:
        if doc.get("kind") == "function":
            function: Any = codec.decode(data=doc["method"], context=context)
            database.register_function(function=cast(StructMethod, function))

    # Phase 4: instances with preserved entity ids (refs stay placeholders)
    instance_docs: list[dict[str, Any]] = [
        doc for doc in documents if doc.get("kind") == "instance"
    ]

    for instance_doc in instance_docs:
        table = database.tables_by_struct_name[instance_doc["struct_name"]]

        instance: StructInstance = StructInstance(
            struct=table.struct,
            entity_id=EntityId(value=instance_doc["entity_id"]),
        )

        instance.attributes = {
            name: codec.decode(data=encoded, context=context)
            for name, encoded in instance_doc["attributes"].items()
        }

        context.instances[instance_doc["entity_id"]] = instance

    # Phase 5: resolve cross instance references, then file into tables
    for instance_doc in instance_docs:
        loaded_instance: StructInstance = context.instances[instance_doc["entity_id"]]
        table = database.tables_by_struct_name[instance_doc["struct_name"]]

        loaded_instance.attributes = {
            name: resolve_instance_refs(value=value, instances=context.instances)
            for name, value in loaded_instance.attributes.items()
        }

        local_id: int = loaded_instance.entity_id.local_id
        table.instances[local_id] = loaded_instance
        table.next_local_id = max(table.next_local_id, local_id + 1)

    # Phase 6: symbol bindings
    for doc in documents:
        if doc.get("kind") == "symbols":
            database.symbols = {
                name: EntityId(value=value)
                for name, value in doc["symbols"].items()
            }

    # Restore the type id counter for future registrations
    database.next_type_id = meta_docs[0]["next_type_id"]

    return database


def _decode_struct(
    struct_doc: dict[str, Any],
    struct: Struct,
    codec: ObjectCodec,
    context: DecodeContext
) -> None:
    """
    Populates a struct shell with its parent, attributes, and methods.

    Args:
        struct_doc (dict[str, Any]): Encoded struct definition.
        struct (Struct): The shell struct to populate.
        codec (ObjectCodec): Codec used for types, defaults, and methods.
        context (DecodeContext): Registries for reference resolution.
    """

    parent_name: Any = struct_doc["parent"]

    if parent_name is not None:
        struct.parent = context.structs[parent_name]

    for attribute_doc in struct_doc["attributes"]:
        attribute: StructAttribute = StructAttribute(
            name=attribute_doc["name"],
            attr_type=cast(Type, codec.decode(data=attribute_doc["attr_type"], context=context)),
            default_value=(
                codec.decode(data=attribute_doc["default_value"], context=context)
                if attribute_doc["has_default"]
                else None
            ),
            has_default=attribute_doc["has_default"],
        )

        struct.add_attribute(attribute=attribute)

    for encoded_method in struct_doc["methods"]:
        method: Any = codec.decode(data=encoded_method, context=context)

        struct.add_method(method=cast(StructMethod, method))
