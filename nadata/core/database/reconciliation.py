"""
Schema reconciliation: aligning persisted data with a re-merged schema.

When a session folder re-merges its .nastruct schema files over an
existing world database, stored rows may no longer match the new
schema: attributes may have been removed, or their declared type
changed. Reconciliation runs after every schema merge and enforces two
rules:

1. Removed attributes — values for attributes no longer declared in
   the schema are dropped. Decoded instances are cleaned eagerly (and
   marked dirty, so the next flush / dump persists the removal);
   pending rows drop stale columns at decode time, preserving lazy
   loading.
2. Type-changed attributes — values that no longer satisfy the
   declared type are reported (struct, attribute, entity id, declared
   type) but kept untouched, so the project still opens and a
   migration script can repair the rows. Mismatched values in pending
   rows are detected when the row decodes.

Added attributes need no reconciliation: schema defaults materialize
through StructInstance.get_attribute on first access.
"""

# Import Modules
from __future__ import annotations
from typing import Any

from nadata.core.database.database import NaDataDatabase
from nadata.core.database.instance import StructInstance
from nadata.core.struct.struct import Struct


class ReconciliationReport:
    """
    Outcome of one schema reconciliation pass.

    Eager findings (decoded instances) are stored directly; lazy
    findings accumulate on the armed row decoders and surface through
    this report as rows decode. Totals and mismatches are computed
    dynamically, so a report stays accurate as lazy rows materialize.

    Attributes:
        dropped (dict[str, int]): Stale attribute values dropped from
            eagerly reconciled instances, per struct name.
        eager_mismatches (list[str]): Type mismatches found during the
            eager pass.
        decoders (list[Any]): Armed row decoders contributing
            decode-time findings.
    """

    __slots__ = ("dropped", "eager_mismatches", "decoders")

    def __init__(self) -> None:
        """
        Initializes an empty ReconciliationReport instance.
        """

        self.dropped: dict[str, int] = {}
        self.eager_mismatches: list[str] = []
        self.decoders: list[Any] = []

    @property
    def per_struct_dropped(self) -> dict[str, int]:
        """
        Dropped stale values per struct name (eager plus decode-time).

        Returns:
            dict[str, int]: Counts per struct name.
        """

        combined: dict[str, int] = dict(self.dropped)

        for decoder in self.decoders:

            for struct_name, count in getattr(
                decoder, "reconciliation_dropped", {}
            ).items():
                combined[struct_name] = combined.get(struct_name, 0) + count

        return combined

    @property
    def total_dropped(self) -> int:
        """
        Total number of dropped stale attribute values.

        Returns:
            int: The count across all structs and decoders.
        """

        return sum(self.per_struct_dropped.values())

    @property
    def mismatches(self) -> list[str]:
        """
        All type mismatch reports (eager plus decode-time).

        Returns:
            list[str]: Human readable mismatch descriptions.
        """

        found: list[str] = list(self.eager_mismatches)

        for decoder in self.decoders:
            found.extend(getattr(decoder, "reconciliation_errors", []))

        return found

    @property
    def ok(self) -> bool:
        """
        Whether reconciliation found no type mismatches.

        Returns:
            bool: True when no mismatch was reported.
        """

        return not self.mismatches

    def summary(self) -> str:
        """
        Renders the reconciliation outcome as human readable lines.

        Returns:
            str: Multi line summary ('schema up to date' when clean).
        """

        if self.ok and self.total_dropped == 0:
            return "schema reconciliation: up to date"

        lines: list[str] = ["schema reconciliation:"]

        if self.total_dropped > 0:

            per_struct: dict[str, int] = self.per_struct_dropped

            breakdown: str = ", ".join(
                f"{name}: {count}"
                for name, count in sorted(per_struct.items())
            )

            lines.append(
                f"  dropped {self.total_dropped} stale attribute value(s) "
                f"({breakdown})"
            )

        for mismatch in self.mismatches:
            lines.append(f"  type mismatch: {mismatch}")

        if self.mismatches:
            lines.append(
                "  (values kept as-is; repair them with a migration script)"
            )

        return "\n".join(lines)


def reconcile_instance(
    struct: Struct,
    instance: StructInstance,
    report: ReconciliationReport,
) -> None:
    """
    Aligns one decoded instance with the current struct schema.

    Stale attributes are removed (the instance is marked dirty so the
    removal persists); values violating their declared type are
    reported but kept.

    Args:
        struct (Struct): Current schema definition.
        instance (StructInstance): The instance to reconcile.
        report (ReconciliationReport): Report to record findings in.
    """

    for name in list(instance.attributes.keys()):

        attribute = struct.get_attribute(name=name)

        if attribute is None:
            instance.attributes.pop(name, None)
            instance.dirty = True
            report.dropped[struct.name] = report.dropped.get(struct.name, 0) + 1

            continue

        value: Any = instance.attributes[name]

        if isinstance(value, dict) or not hasattr(value, "get"):
            continue

        try:
            attribute.attr_type.check_value(value=value.get())

        except TypeError as error:
            report.eager_mismatches.append(
                f"{struct.name}.{name} on entity "
                f"{instance.entity_id.value}: {error}"
            )


def prune_undeclared_members(
    database: NaDataDatabase,
    redeclared: dict[str, set[str]],
) -> None:
    """
    Removes struct members that re-declarations no longer carry.

    Merge mode only adds and replaces members, so an attribute removed
    from the .nastruct sources would otherwise linger on the struct
    (and keep surfacing on old rows). This prunes own attributes and
    methods absent from the union of the re-declarations. Inherited
    members are untouched: pruning looks at own declarations only.

    Args:
        database (NaDataDatabase): The world database.
        redeclared (dict[str, set[str]]): Struct name -> member names
            declared by this open's schema files (union across files).
    """

    for struct_name, members in redeclared.items():

        table = database.tables_by_struct_name.get(struct_name)

        if table is None:
            continue

        struct: Struct = table.struct

        for name in list(struct.attributes.keys()):

            if name not in members:
                struct.attributes.pop(name, None)

        for name in list(struct.methods.keys()):

            if name not in members:
                struct.methods.pop(name, None)


def reconcile_schema(database: NaDataDatabase) -> ReconciliationReport:
    """
    Reconciles a world database's data with its (re-merged) schema.

    Decoded instances are validated eagerly; lazy tables get their row
    decoder armed so pending rows reconcile at decode time. The two
    sets of findings are merged into one report.

    Args:
        database (NaDataDatabase): The world database.

    Returns:
        ReconciliationReport: Dropped values and type mismatches found.
    """

    report: ReconciliationReport = ReconciliationReport()

    for table in database.tables_by_type_id.values():

        decoder: Any = getattr(table.instances, "decoder", None)

        if decoder is not None:

            # Lazy table: only decoded entries are checked eagerly —
            # pending rows reconcile through the armed decoder, without
            # defeating lazy loading
            decoder.reconcile_enabled = True
            report.decoders.append(decoder)

            for instance in list(decoder.decoded.values()):
                reconcile_instance(
                    struct=table.struct,
                    instance=instance,
                    report=report,
                )

            continue

        for instance in list(dict.values(table.instances)):
            reconcile_instance(
                struct=table.struct,
                instance=instance,
                report=report,
            )

    return report
