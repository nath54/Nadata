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

    if not meta_docs or meta_docs[-1].get("format_version") != 1:
        raise ValueError(f"File '{storage_path}' is not a Nadata database dump")

    struct_docs: list[dict[str, Any]] = [
        doc for doc in documents if doc.get("kind") == "struct"
    ]

    # Phase 1: struct shells registered under stable type ids
    for struct_doc in struct_docs:
        struct_name: str = struct_doc["struct"]["name"]

        # Re-declared structs (incremental appends) reuse the existing shell
        if struct_name not in context.structs:
            struct: Struct = Struct(name=struct_name)
            database.register_struct(struct=struct)
            context.structs[struct_name] = struct

    # Phase 2: parents, attributes, and methods (struct refs now resolvable);
    # later documents for the same struct merge into the earlier definition
    decoded_structs: set[str] = set()

    for struct_doc in struct_docs:
        struct_name = struct_doc["struct"]["name"]

        _decode_struct(
            struct_doc=struct_doc["struct"],
            struct=context.structs[struct_name],
            codec=codec,
            context=context,
            merge_mode=struct_name in decoded_structs,
        )

        decoded_structs.add(struct_name)

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

        decoded_attributes: dict[str, Any] = {
            name: codec.decode(data=encoded, context=context)
            for name, encoded in instance_doc["attributes"].items()
        }

        # Later documents for the same entity update the existing instance
        if instance_doc["entity_id"] in context.instances:
            context.instances[instance_doc["entity_id"]].attributes = decoded_attributes

            continue

        instance: StructInstance = StructInstance(
            struct=table.struct,
            entity_id=EntityId(value=instance_doc["entity_id"]),
        )

        instance.attributes = decoded_attributes

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

    # Phase 6: symbol bindings (merged, so incremental appends can extend)
    for doc in documents:
        if doc.get("kind") == "symbols":
            database.symbols.update({
                name: EntityId(value=value)
                for name, value in doc["symbols"].items()
            })

    # Restore the type id counter for future registrations
    database.next_type_id = meta_docs[0]["next_type_id"]

    # Settings merge in replay order: later meta documents win per key
    for meta_doc in meta_docs:
        database.settings.update(meta_doc.get("settings", {}))

    return database


def _decode_struct(
    struct_doc: dict[str, Any],
    struct: Struct,
    codec: ObjectCodec,
    context: DecodeContext,
    merge_mode: bool = False
) -> None:
    """
    Populates a struct shell with its parent, attributes, and methods.

    Args:
        struct_doc (dict[str, Any]): Encoded struct definition.
        struct (Struct): The shell struct to populate.
        codec (ObjectCodec): Codec used for types, defaults, and methods.
        context (DecodeContext): Registries for reference resolution.
        merge_mode (bool, optional): When True (incremental re-declaration),
            existing attribute / method names are replaced instead of
            raising. Defaults to False.
    """

    parent_name: Any = struct_doc["parent"]

    if parent_name is not None and struct.parent is None:
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

        if merge_mode and attribute.name in struct.attributes:
            struct.attributes[attribute.name] = attribute

        else:
            struct.add_attribute(attribute=attribute)

    for encoded_method in struct_doc["methods"]:
        method: Any = codec.decode(data=encoded_method, context=context)

        if merge_mode and cast(StructMethod, method).name in struct.methods:
            struct.methods[cast(StructMethod, method).name] = cast(StructMethod, method)

        else:
            struct.add_method(method=cast(StructMethod, method))
