"""
Database loader reconstructing a NaDataDatabase from slotted page records.

Rebuilds structs (shells first, then parents, attributes, and methods so that
forward type references resolve), standalone functions, instances with
preserved entity ids and cross references, and symbol bindings.
"""

# Import Modules
from __future__ import annotations
import json
from typing import Any, Optional, cast

from nadata.core.database.database import NaDataDatabase
from nadata.core.database.entity_id import EntityId
from nadata.core.database.instance import StructInstance
from nadata.core.database.persistence.binary_cells import (
    decode_batch_row_spans,
    decode_row,
    decode_varint,
)
from nadata.core.database.persistence.lazy_instances import (
    LazyInstanceDict,
    RowDecoder,
)
from nadata.core.database.persistence.record_store import RecordStore
from nadata.core.database.persistence.serialization import (
    DecodeContext,
    ObjectCodec,
    resolve_instance_refs,
)
from nadata.core.struct.struct import Struct
from nadata.core.struct.struct_attribute import StructAttribute
from nadata.core.struct.struct_method import StructMethod
from nadata.core.types.types import Type, TypeSet


def _is_json_document(payload: bytes) -> bool:
    """
    Checks whether a stored document payload is a JSON document.

    Args:
        payload (bytes): Document payload.

    Returns:
        bool: True if the payload starts with a JSON object brace.
    """

    return payload[:1] in (b"{", b"[")


def _is_binary_batch(payload: bytes) -> bool:
    """
    Checks whether a stored document payload is a binary instance batch.

    Batch documents start with a varint header length followed by a JSON
    header whose kind is 'instances'; JSON documents start with '{'.

    Args:
        payload (bytes): Document payload.

    Returns:
        bool: True if the payload is a binary instance batch.
    """

    if payload[:1] in (b"{", b"["):
        return False

    try:
        header_len: int
        header_len, offset = decode_varint(payload=payload, offset=0)

        header: dict[str, Any] = json.loads(
            payload[offset:offset + header_len].decode("utf-8")
        )

    except (IndexError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return False

    return header.get("kind") == "instances" and header.get("format") == "binary"


def _column_types(struct: Struct, columns: list[str]) -> dict[str, Type]:
    """
    Resolves the declared type of each batch column from the struct.

    Args:
        struct (Struct): Struct definition to resolve against.
        columns (list[str]): Column names in order.

    Returns:
        dict[str, Type]: Declared type per column.

    Raises:
        KeyError: If a column is not declared in the struct schema.
    """

    types: dict[str, Type] = {}

    for column in columns:
        attribute: Optional[StructAttribute] = struct.get_attribute(name=column)

        if attribute is None:
            raise KeyError(
                f"Batch column '{column}' is not declared in struct '{struct.name}'"
            )

        types[column] = attribute.attr_type

    return types


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

    documents: list[dict[str, Any]] = []

    for doc_id in sorted(record_store.document_locations()):

        payload: bytes = record_store.read_document(doc_id=doc_id)

        if _is_json_document(payload=payload):
            documents.append(json.loads(payload.decode("utf-8")))

    meta_docs: list[dict[str, Any]] = [
        doc for doc in documents if doc.get("kind") == "meta"
    ]

    if not meta_docs or meta_docs[-1].get("format_version") not in (1, 2):
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

    # Phase 4: instances. Binary v2 batches register rows lazily (as doc
    # spans — payloads are reassembled one batch at a time and released);
    # legacy v1 JSON instance documents decode eagerly with placeholder
    # refs.
    row_decoder: RowDecoder = RowDecoder()

    row_decoder.set_struct_lookup(
        lookup=lambda name: context.structs[name],
    )

    row_decoder.set_document_reader(
        reader=lambda doc_id: record_store.read_document(doc_id=doc_id),
    )

    row_decoder.set_nid_provider(
        provider=database.nid_of_entity,
    )

    instance_docs: list[dict[str, Any]] = []

    for doc_id in sorted(record_store.document_locations()):

        payload = record_store.read_document(doc_id=doc_id)

        if not _is_binary_batch(payload=payload):
            continue

        batch_struct_name: str
        columns: list[str]
        max_local_id: int
        rows: list[tuple[int, int, int]]

        (
            batch_struct_name,
            columns,
            max_local_id,
            rows,
        ) = decode_batch_row_spans(payload=payload)

        if batch_struct_name not in context.structs:
            raise KeyError(
                f"Batch references unknown struct '{batch_struct_name}'"
            )

        table = database.tables_by_struct_name[batch_struct_name]
        types: dict[str, Type] = _column_types(
            struct=table.struct, columns=columns
        )

        table.instances = LazyInstanceDict(
            decoder=row_decoder,
            struct_name=batch_struct_name,
        )

        row_decoder.register_table(
            struct_name=batch_struct_name,
            table=table.instances,
        )

        row_decoder.register_batch(
            struct_name=batch_struct_name,
            columns=columns,
            types=types,
            doc_id=doc_id,
            rows=rows,
            fetch=row_decoder.fetch_document_span,
        )

        for entity_id, _row_offset, _row_length in rows:
            table.next_local_id = max(
                table.next_local_id,
                EntityId(value=entity_id).local_id + 1,
            )

        # Header-level max local id also counts (replay may supersede rows)
        table.next_local_id = max(table.next_local_id, max_local_id + 1)

    instance_docs = [
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
    # (legacy v1 rows only; binary rows resolve through the row decoder)
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

        # Persisted state: legacy rows start clean so flush() skips them
        loaded_instance.dirty = False

    # Phase 5.5: removal tombstones (replay order — applied after all
    # instance documents so a removal wins over the row it deletes)
    for doc in documents:
        if doc.get("kind") != "instance_removal":
            continue

        removed_entity_id: int = int(doc["entity_id"])

        row_decoder.remove_entity(entity_id=removed_entity_id)
        database.drop_nids_of_entity(entity_id=removed_entity_id)

        type_id: int = EntityId(value=removed_entity_id).type_id

        if type_id in database.tables_by_type_id:
            removed_table = database.tables_by_type_id[type_id]
            dict.pop(removed_table.instances, EntityId(value=removed_entity_id).local_id, None)  # noqa: E501

    # Phase 6: symbol bindings + nid identity map (merged, so incremental
    # appends can extend)
    for doc in documents:
        if doc.get("kind") == "symbols":
            database.symbols.update({
                name: EntityId(value=value)
                for name, value in doc["symbols"].items()
            })

            for key, value in doc.get("nids", {}).items():

                nid_entity: EntityId = EntityId(value=value)
                database.register_nid(
                    struct_name=key.split("/", 1)[0],
                    nid=key.split("/", 1)[1],
                    entity_id=nid_entity,
                )

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
        attribute_type: Type = cast(
            Type, codec.decode(data=attribute_doc["attr_type"], context=context)
        )

        default_value: Any = (
            codec.decode(data=attribute_doc["default_value"], context=context)
            if attribute_doc["has_default"]
            else None
        )

        # JSON has no set type: set defaults decode as lists
        if (
            isinstance(attribute_type, TypeSet)
            and isinstance(default_value, list)
        ):
            default_value = set(default_value)

        default_expr: Any = (
            codec.decode(data=attribute_doc["default_expr"], context=context)
            if attribute_doc.get("default_expr") is not None
            else None
        )

        attribute: StructAttribute = StructAttribute(
            name=attribute_doc["name"],
            attr_type=attribute_type,
            default_value=default_value,
            has_default=attribute_doc["has_default"],
            is_const=attribute_doc.get("is_const", False),
            default_expr=default_expr,
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
