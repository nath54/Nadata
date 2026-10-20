"""
Database file compaction: rewriting an append-only .ndb log into a minimal
snapshot of its current state.
"""

# Import Modules
from __future__ import annotations
import os
from typing import Optional

from src.core.database.database import NaDataDatabase
from src.core.database.page_manager import PageManager
from src.core.database.persistence.dumper import dump_database
from src.core.database.persistence.loader import load_database
from src.core.database.persistence.record_store import RecordStore

# Suffix of the temporary file written during compaction
TEMP_SUFFIX: str = ".compact-tmp"


def count_documents(storage_path: str) -> int:
    """
    Counts the documents stored in a database file.

    Args:
        storage_path (str): Path to the .ndb file.

    Returns:
        int: Number of framed documents in the file.

    Raises:
        ValueError: If the file does not exist.
    """

    if not os.path.exists(storage_path):
        raise ValueError(f"Storage path '{storage_path}' does not exist")

    page_manager: PageManager = PageManager(storage_path=storage_path)

    return len(RecordStore(page_manager=page_manager).read_all_documents())


def compact_database(
    storage_path: str,
    output_path: Optional[str] = None
) -> tuple[int, int]:
    """
    Rewrites a database file into a minimal snapshot of its current state.

    The .ndb format is an append-only replay log: incremental saves leave
    superseded documents behind and loads replay the whole history.
    Compaction loads the latest state and rewrites it as a fresh snapshot,
    so the file size and load time are proportional to the current data
    instead of the edit history.

    The output is written to a temporary file next to the target and
    atomically renamed, so a crash mid-compaction never damages the source.

    Args:
        storage_path (str): Path to the existing .ndb file to compact.
        output_path (Optional[str], optional): Where to write the compacted
            file. Defaults to None (compact 'storage_path' in place).

    Returns:
        tuple[int, int]: Document counts (before, after).

    Raises:
        ValueError: If the source does not exist or is not a Nadata dump.
    """

    documents_before: int = count_documents(storage_path=storage_path)

    database: NaDataDatabase = load_database(storage_path=storage_path)

    final_target: str = output_path if output_path is not None else storage_path
    temp_path: str = final_target + TEMP_SUFFIX

    # A stale temporary file from an interrupted compaction is removed
    if os.path.exists(temp_path):
        os.remove(temp_path)

    try:
        dump_database(database=database, storage_path=temp_path)

        documents_after: int = count_documents(storage_path=temp_path)

        os.replace(temp_path, final_target)

    except Exception:

        # Never leave a partial temporary file behind
        if os.path.exists(temp_path):
            os.remove(temp_path)

        raise

    return documents_before, documents_after
