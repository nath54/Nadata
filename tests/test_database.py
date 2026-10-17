"""
Unit tests for the NaDataDatabase engine, EntityId, SlottedPage,
PageManager, and QueryEngine optimization.
"""

# Import Modules
from __future__ import annotations
import os
import tempfile

from src.core.struct import (
    Struct,
    StructAttribute,
)
from src.core.database import (
    EntityId,
    QuerySet,
    SlottedPage,
    PageManager,
    PrintFunction,
    StructInstance,
    NaDataDatabase,
    QueryOptimizer,
)

from src.core.values.values import (
    ValueStr,
    ValueFloat,
    ValueDateTime,
)
from src.core.types.types import (
    Float32,
    TypeStr,
    TypeDateTime,
)

from src.core.expressions import (
    LiteralExpr,
    BinaryOpExpr,
    ExecutionScope,
    IdentifierExpr,
    MemberAccessExpr,
)
from src.core.types.home_types import DateTime



def test_entity_id_bit_packing() -> None:
    """
    Tests 64-bit EntityId construction, bit shifting, and property decoding.
    """

    type_id: int = 42
    local_id: int = 1234567

    eid: EntityId = EntityId.from_parts(type_id=type_id, local_id=local_id)

    assert eid.type_id == 42
    assert eid.local_id == 1234567

    # Test round trip from raw integer
    eid2: EntityId = EntityId(value=int(eid))
    assert eid2.type_id == 42
    assert eid2.local_id == 1234567
    assert eid == eid2


def test_slotted_page_allocation_and_read() -> None:
    """
    Tests binary slotted page record insertion, slot directory, and offsets.
    """

    page: SlottedPage = SlottedPage(page_id=1, page_size=1024)

    record_a: bytes = b"Hello, Nadata!"
    record_b: bytes = b"Secondary binary payload"

    slot_a: int | None = page.insert_record(record=record_a)
    slot_b: int | None = page.insert_record(record=record_b)

    assert slot_a == 0
    assert slot_b == 1
    assert page.slot_count == 2

    # Verify byte reading
    assert page.read_record(slot_index=0) == record_a
    assert page.read_record(slot_index=1) == record_b


def test_page_manager_lru_and_disk_paging() -> None:
    """
    Tests PageManager lazy loading, LRU eviction, and disk flush.
    """

    with tempfile.TemporaryDirectory() as tmp_dir:
        storage_file: str = os.path.join(tmp_dir, "test_db.nadatab")

        # Create manager with max 2 cached pages
        pm: PageManager = PageManager(storage_path=storage_file, page_size=512, max_cached_pages=2)

        p0: SlottedPage = pm.allocate_page()
        p1: SlottedPage = pm.allocate_page()
        p0.insert_record(b"Page zero data")
        p1.insert_record(b"Page one data")

        # Allocating p2 triggers eviction of p0 to disk
        p2: SlottedPage = pm.allocate_page()
        p2.insert_record(b"Page two data")

        assert 0 not in pm.cache
        assert 1 in pm.cache
        assert 2 in pm.cache

        # Reading p0 reloads lazily from disk
        reloaded_p0: SlottedPage = pm.get_page(page_id=0)
        assert reloaded_p0.read_record(0) == b"Page zero data"


def test_database_table_and_instance_queries() -> None:
    """
    Tests NaDataDatabase registration, O(1) ID lookups, and parallel queries.
    """

    db: NaDataDatabase = NaDataDatabase()

    # Define schemas
    entity_struct: Struct = Struct(name="Entity")
    entity_struct.add_attribute(StructAttribute(name="name", attr_type=TypeStr()))
    entity_struct.add_attribute(
        StructAttribute(name="birthday", attr_type=TypeDateTime())
    )

    human_struct: Struct = Struct(name="Human", parent=entity_struct)
    human_struct.add_attribute(StructAttribute(name="weight", attr_type=Float32))

    # Register in DB
    db.register_struct(struct=entity_struct)
    db.register_struct(struct=human_struct)

    # Insert instances
    tom: StructInstance = db.create_instance(
        struct=human_struct,
        attributes={
            "name": ValueStr(TypeStr(), "Tom"),
            "weight": ValueFloat(Float32, 72.5),
            "birthday": ValueDateTime(value=DateTime(y=2003, m=4, d=12)),
        },
        symbol="tom"
    )

    jean: StructInstance = db.create_instance(
        struct=human_struct,
        attributes={
            "name": ValueStr(TypeStr(), "Jean"),
            "weight": ValueFloat(Float32, 80.0),
            "birthday": ValueDateTime(value=DateTime(y=1972, m=4, d=12)),
        },
        symbol="jean"
    )

    # Test symbol lookup
    lookup_tom = db.get_by_symbol(name="tom")
    assert lookup_tom is not None
    assert lookup_tom.entity_id == tom.entity_id

    # Test O(1) ID lookup
    lookup_jean = db.get_by_id(entity_id=jean.entity_id)
    assert lookup_jean is not None
    assert lookup_jean.attributes["name"].get() == "Jean"

    # Test polymorphic query (querying Entity returns Human instances)
    all_entities = db.query(struct=entity_struct, include_subtypes=True)
    assert len(all_entities) == 2

    # Test query with predicate filter
    heavy_humans = db.query(
        struct=human_struct,
        predicate=lambda h: h.attributes["weight"].get() > 75.0
    )
    assert len(heavy_humans) == 1
    first_heavy = heavy_humans.first()
    assert first_heavy is not None
    assert first_heavy.attributes["name"].get() == "Jean"



def test_query_invariant_hoisting() -> None:
    """
    Tests that QueryOptimizer identifies loop invariants and hoists them.
    """

    tom_instance: StructInstance = StructInstance(
        struct=Struct(name="Human"),
        entity_id=EntityId.from_parts(1, 1),
        attributes={"size": ValueFloat(Float32, 1.80)}
    )

    scope: ExecutionScope = ExecutionScope(globals_dict={"tom": tom_instance})


    # Expression: self.size > tom.size
    # tom.size is invariant with respect to candidate 'self'
    expr: BinaryOpExpr = BinaryOpExpr(
        left=MemberAccessExpr(target=IdentifierExpr("self"), member="size"),
        op=">",
        right=MemberAccessExpr(target=IdentifierExpr("tom"), member="size")
    )

    assert not QueryOptimizer.is_invariant(expr=expr.left, loop_var="self")
    assert QueryOptimizer.is_invariant(expr=expr.right, loop_var="self")

    # Hoist invariants
    optimized: BinaryOpExpr = QueryOptimizer.hoist_invariants(
        expr=expr,
        scope=scope,
        loop_var="self"
    ) # type: ignore

    # The right operand should now be pre-evaluated into a LiteralExpr
    assert isinstance(optimized.right, LiteralExpr)
    assert optimized.right.value.get() == 1.80


def test_queryset_chaining_and_print() -> None:
    """
    Tests fluent QuerySet chaining with .where() and PrintFunction formatting.
    """

    db: NaDataDatabase = NaDataDatabase()

    human_struct: Struct = Struct(name="Human")
    human_struct.add_attribute(StructAttribute(name="name", attr_type=TypeStr()))
    human_struct.add_attribute(StructAttribute(name="weight", attr_type=Float32))
    human_struct.add_attribute(StructAttribute(name="birthplace", attr_type=TypeStr()))

    db.register_struct(struct=human_struct)

    # Insert 3 humans
    db.create_instance(
        struct=human_struct,
        attributes={
            "name": ValueStr(TypeStr(), "Tom"),
            "weight": ValueFloat(Float32, 70.0),
            "birthplace": ValueStr(TypeStr(), "Paris"),
        }
    )
    db.create_instance(
        struct=human_struct,
        attributes={
            "name": ValueStr(TypeStr(), "Jean"),
            "weight": ValueFloat(Float32, 80.0),
            "birthplace": ValueStr(TypeStr(), "Paris"),
        }
    )
    db.create_instance(
        struct=human_struct,
        attributes={
            "name": ValueStr(TypeStr(), "Louise"),
            "weight": ValueFloat(Float32, 65.0),
            "birthplace": ValueStr(TypeStr(), "Lyon"),
        }
    )

    # Initial query
    all_humans: QuerySet = db.query(struct=human_struct)
    assert all_humans.count() == 3

    # First filter: birthplace == "Paris"
    parisians: QuerySet = all_humans.where(
        predicate=lambda h: h.attributes["birthplace"].get() == "Paris"
    )
    assert parisians.count() == 2

    # Chained filter: weight >= 75.0 on the parisians result!
    heavy_parisians: QuerySet = parisians.where(
        predicate=lambda h: h.attributes["weight"].get() >= 75.0
    )
    assert heavy_parisians.count() == 1
    assert heavy_parisians.first() is not None
    assert heavy_parisians.first().attributes["name"].get() == "Jean" # type: ignore

    # Test PrintFunction
    printer: PrintFunction = PrintFunction(capture=True)
    printer("Results:", heavy_parisians)

    assert len(printer.captured_output) == 1
    assert "Results:" in printer.captured_output[0]
    assert "<Human id=1>" in printer.captured_output[0]
