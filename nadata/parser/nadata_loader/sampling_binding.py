"""
Sampling entry points on the language-side struct runtime binding.

Kept apart from the binding itself so the loader module stays focused
on statement passes; the mixin is mixed into StructRuntimeBinding and
uses only its struct/database attributes.
"""

# Import Modules
from __future__ import annotations
import random
from typing import Any, Callable, Optional

from nadata.core.database.instance import StructInstance
from nadata.core.values.values import Value


class StructSamplingMixin:
    """
    Random-sampling entry points mixed into StructRuntimeBinding.

    Provides sample() (unfiltered), sample_by() (indexed equality with
    probe fallback), and sample_where() (predicate probing), all
    polymorphic over subclasses. The sampling algorithms themselves
    live on the tables (nadata.core.database.sampling).
    """

    struct: Any
    database: Any

    def sample(self, count: Value) -> list[StructInstance]:
        """
        Draws random instances of this struct (subclasses included)
        without a full-table decode.

        Args:
            count (Value): Wrapped desired instance count.

        Returns:
            list[StructInstance]: Up to count distinct instances in
            random order.

        Raises:
            TypeError: If the count does not resolve to an integer.
        """

        raw_count: Any = count.get() if isinstance(count, Value) else count

        if not isinstance(raw_count, int) or isinstance(raw_count, bool):
            raise TypeError("sample expects an integer count")

        return self._sample_tables(
            count=raw_count,
            table_operation=lambda table: table.sample(
                count=raw_count
            )
        )

    def sample_by(
        self,
        attribute: Value,
        raw: Value,
        count: Value,
    ) -> list[StructInstance]:
        """
        Draws random instances matching an attribute equality, using the
        hash index when present (O(bucket)) and probing otherwise.

        Args:
            attribute (Value): Wrapped attribute name.
            raw (Value): Wrapped value to match.
            count (Value): Wrapped desired instance count.

        Returns:
            list[StructInstance]: Up to count distinct matching instances
            in random order.

        Raises:
            TypeError: If the attribute is not a string or the count is
                not an integer.
        """

        attribute_name: Any = (
            attribute.get() if isinstance(attribute, Value) else attribute
        )
        raw_value: Any = raw.get() if isinstance(raw, Value) else raw
        raw_count: Any = count.get() if isinstance(count, Value) else count

        if not isinstance(attribute_name, str):
            raise TypeError("sample_by attribute must be a string")

        if not isinstance(raw_count, int) or isinstance(raw_count, bool):
            raise TypeError("sample_by expects an integer count")

        def run(table: Any) -> Optional[list[StructInstance]]:

            indexed: Optional[list[StructInstance]] = table.sample_by(
                attribute=attribute_name,
                raw=raw_value,
                count=raw_count,
            )

            if indexed is not None:
                return indexed

            probed: list[StructInstance] = table.sample(
                count=raw_count,
                predicate=lambda inst: (
                    attribute_name in inst.attributes
                    and inst.attributes[attribute_name].get() == raw_value
                )
            )

            return probed

        return self._sample_tables(count=raw_count, table_operation=run)

    def sample_where(
        self,
        predicate: Value,
        count: Optional[Value] = None,
    ) -> list[StructInstance]:
        """
        Draws random instances passing a lambda predicate, probing rows
        until enough match (then scanning the rest if the filter is too
        sparse). Without a count, every matching row is returned.

        Args:
            predicate (Value): ValueExternal wrapping a callable predicate.
            count (Optional[Value]): Wrapped desired instance count;
                Defaults to None (all matches).

        Returns:
            list[StructInstance]: Up to count distinct matching instances
            in random order (all matches when count is omitted).

        Raises:
            TypeError: If the predicate is not callable or the count is
                not an integer.
        """

        callable_predicate: Any = (
            predicate.get() if isinstance(predicate, Value) else predicate
        )
        raw_count: Any = (
            count.get() if count is not None and isinstance(count, Value) else count
        )

        if not callable(callable_predicate):
            raise TypeError(
                f"'.sample_where' on struct '{self.struct.name}' "
                f"expects a callable predicate"
            )

        if raw_count is not None and (
            not isinstance(raw_count, int) or isinstance(raw_count, bool)
        ):
            raise TypeError("sample_where expects an integer count")

        # An omitted count probes the whole table and keeps every match
        effective_count: int = (
            raw_count if raw_count is not None else 2 ** 62
        )

        return self._sample_tables(
            count=effective_count,
            table_operation=lambda table: table.sample(
                count=effective_count,
                predicate=callable_predicate,
            )
        )

    def _sample_tables(
        self,
        count: int,
        table_operation: Callable[[Any], Optional[list[StructInstance]]],
    ) -> list[StructInstance]:
        """
        Runs a sampling operation over this struct's table and every
        subclass table, then reduces the combined result to count.

        Args:
            count (int): Desired total instance count.
            table_operation (Callable): Per-table sampler returning the
                table's random draw (or None).

        Returns:
            list[StructInstance]: Up to count distinct instances.
        """

        collected: list[StructInstance] = []

        for table in self.database.tables_by_struct_name.values():

            if table.struct is not self.struct and not table.struct.is_subclass_of(
                other=self.struct
            ):
                continue

            drawn: Optional[list[StructInstance]] = table_operation(table)

            if drawn:
                collected.extend(drawn)

        if len(collected) <= count:
            return collected

        return random.sample(population=collected, k=count)
