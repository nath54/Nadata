"""
PageManager with LRU caching and lazy disk paging for Nadata.

Manages loading, caching, and evicting SlottedPage instances, ensuring
the database never loads all data into memory simultaneously.
"""

# Import Modules
from __future__ import annotations
from typing import Optional
from collections import OrderedDict
import os

from nadata.core.database.slotted_page import SlottedPage


class PageManager:
    """
    Manages fixed-size SlottedPage allocations, lazy disk reads, and LRU eviction.

    Attributes:
        storage_path (Optional[str]): Optional file path on disk for persistence.
        page_size (int): Size of each page in bytes (default 65536).
        max_cached_pages (int): Max pages retained in memory simultaneously.
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        page_size: int = 65536,
        max_cached_pages: int = 256
    ) -> None:
        """
        Initializes the PageManager instance.

        Args:
            storage_path (Optional[str], optional): File path for page storage. Defaults to None.
            page_size (int, optional): Bytes per page. Defaults to 65536 (64KB).
            max_cached_pages (int, optional): Maximum in-memory cached pages. Defaults to 256.
        """

        self.storage_path: Optional[str] = storage_path
        self.page_size: int = page_size
        self.max_cached_pages: int = max_cached_pages

        # LRU cache: page_id -> SlottedPage (ordered by access)
        self.cache: OrderedDict[int, SlottedPage] = OrderedDict()
        self.next_page_id: int = 0

        # Initialize storage file if specified
        if self.storage_path is not None and os.path.exists(self.storage_path):
            file_size: int = os.path.getsize(self.storage_path)
            self.next_page_id = file_size // self.page_size

    def allocate_page(self) -> SlottedPage:
        """
        Allocates a new SlottedPage and adds it to the cache.

        Returns:
            SlottedPage: Newly created page.
        """

        new_id: int = self.next_page_id
        self.next_page_id += 1

        page: SlottedPage = SlottedPage(page_id=new_id, page_size=self.page_size)
        self._put_in_cache(page=page)

        return page

    def get_page(self, page_id: int) -> SlottedPage:
        """
        Retrieves a page by ID, loading it lazily from disk if not in cache.

        Args:
            page_id (int): Page ID to retrieve.

        Returns:
            SlottedPage: Requested page.

        Raises:
            IndexError: If page_id does not exist.
        """

        # Fast path: check in-memory LRU cache
        if page_id in self.cache:
            page: SlottedPage = self.cache[page_id]
            self.cache.move_to_end(page_id)
            return page

        # Lazy path: load from disk
        page = self._read_from_disk(page_id=page_id)
        self._put_in_cache(page=page)

        return page

    def flush(self) -> None:
        """
        Flushes all dirty cached pages to disk.
        """

        if self.storage_path is None:
            return

        for page in self.cache.values():
            if page.is_dirty:
                self._write_to_disk(page=page)

    def _put_in_cache(self, page: SlottedPage) -> None:
        """
        Inserts a page into the LRU cache, evicting the oldest page if full.

        Args:
            page (SlottedPage): Page to insert.
        """

        # Evict oldest page if capacity reached
        if len(self.cache) >= self.max_cached_pages:
            _, oldest_page = self.cache.popitem(last=False)

            if oldest_page.is_dirty and self.storage_path is not None:
                self._write_to_disk(page=oldest_page)


        self.cache[page.page_id] = page

    def _read_from_disk(self, page_id: int) -> SlottedPage:
        """
        Reads a single page from disk at its calculated offset.

        Args:
            page_id (int): Page ID.

        Returns:
            SlottedPage: Instantiated page.

        Raises:
            IndexError: If page_id is out of bounds.
        """

        if self.storage_path is None or not os.path.exists(self.storage_path):
            raise IndexError(f"Page ID {page_id} does not exist (in-memory mode)")

        offset: int = page_id * self.page_size

        with open(self.storage_path, "rb") as f:
            f.seek(offset)
            raw_bytes: bytes = f.read(self.page_size)

            if len(raw_bytes) != self.page_size:
                raise IndexError(f"Page ID {page_id} out of bounds on disk")

        return SlottedPage(page_id=page_id, page_size=self.page_size, raw_bytes=raw_bytes)

    def _write_to_disk(self, page: SlottedPage) -> None:
        """
        Writes a single page buffer to disk at its calculated offset.

        Args:
            page (SlottedPage): Page to write.
        """

        if self.storage_path is None:
            return

        offset: int = page.page_id * self.page_size

        # Open in r+b or ab
        mode: str = "r+b" if os.path.exists(self.storage_path) else "wb"

        with open(self.storage_path, mode) as f:
            f.seek(offset)
            f.write(page.to_bytes())

        page.is_dirty = False
