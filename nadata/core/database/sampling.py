"""
Filtered random sampling over storage tables.

Draws random rows without decoding the whole table: candidate local
ids are probed (decoded + tested) until the quota fills, with a scan
fallback for sparse filters so results stay correct. Indexed equality
sampling reads a hash bucket directly and decodes only the drawn rows.
"""

# Import Modules
from __future__ import annotations
import random
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from nadata.core.database.database import StructTable
    from nadata.core.database.instance import StructInstance


def sample_table_rows(
    table: StructTable,
    count: int,
    predicate: Optional[Callable[[StructInstance], bool]] = None,
    rng: Optional[random.Random] = None,
    probe_factor: int = 8,
) -> list[StructInstance]:
    """
    Draws up to count random instances from a table without a full
    decode.

    Without a predicate, distinct local ids are sampled directly and
    only the chosen rows decode. With a predicate, random rows are
    probed (decode + test) until enough match; if the probe budget is
    exhausted first, the remaining rows are scanned so the result stays
    correct for sparse filters.

    Args:
        table (StructTable): The table to sample from.
        count (int): Desired number of instances.
        predicate (Optional[Callable]): Optional row filter. Defaults to
            None (unfiltered sample).
        rng (Optional[random.Random]): Random source; defaults to the
            module-level generator (seedable via random.seed).
        probe_factor (int): Probe budget = probe_factor * count rows
            before falling back to a scan.

    Returns:
        list[StructInstance]: Up to count distinct matching instances
        in random order.
    """

    source: Any = rng if rng is not None else random

    all_ids: list[int]

    if hasattr(table.instances, "all_local_ids"):
        all_ids = table.instances.all_local_ids()
    else:
        all_ids = list(table.instances.keys())

    if count <= 0 or not all_ids:
        return []

    if predicate is None:
        chosen: list[int] = source.sample(
            population=all_ids, k=min(count, len(all_ids))
        )

        return [table.instances[local_id] for local_id in chosen]

    # Probe phase: random distinct rows, decoded one by one, until
    # enough match or the budget runs out. The first count matches of
    # a random ordering are a uniform random subset.
    probe_budget: int = min(len(all_ids), max(probe_factor * count, count))
    probe_ids: list[int] = (
        source.sample(population=all_ids, k=probe_budget)
        if probe_budget < len(all_ids)
        else source.sample(population=all_ids, k=len(all_ids))
    )

    collected: list[StructInstance] = []

    for local_id in probe_ids:

        if local_id not in table.instances:
            continue

        instance: StructInstance = table.instances[local_id]

        if predicate(instance):
            collected.append(instance)

            if len(collected) >= count:
                return collected

    # Fallback: scan every row not yet probed so sparse filters still
    # fill the quota (correctness over speed).
    probed: set[int] = set(probe_ids)

    for local_id in all_ids:

        if local_id in probed or local_id not in table.instances:
            continue

        instance = table.instances[local_id]

        if predicate(instance):
            collected.append(instance)

    if len(collected) <= count:
        return collected

    reduced: list[StructInstance] = source.sample(
        population=collected, k=count
    )

    return reduced


def sample_indexed_rows(
    table: StructTable,
    attribute: str,
    raw: Any,
    count: int,
    rng: Optional[random.Random] = None,
) -> Optional[list[StructInstance]]:
    """
    Draws up to count random instances matching an indexed equality,
    decoding only the chosen rows.

    Args:
        table (StructTable): The table to sample from.
        attribute (str): Indexed scalar attribute name.
        raw (Any): Raw value to match.
        count (int): Desired number of instances.
        rng (Optional[random.Random]): Random source; defaults to the
            module-level generator.

    Returns:
        Optional[list[StructInstance]]: Up to count matching instances
        in random order, or None when the attribute has no index
        (callers fall back to sample_table_rows with a predicate).
    """

    # None => no index on the attribute; [] => indexed but empty bucket
    matches: Optional[list[StructInstance]] = table.lookup_by_index(
        attribute=attribute, raw=raw
    )

    if matches is None:
        return None

    if len(matches) <= count:
        return matches

    source: Any = rng if rng is not None else random

    chosen: list[StructInstance] = source.sample(
        population=matches, k=count
    )

    return chosen
