"""
Chunked document storage over slotted pages for Nadata.

Stores variable length binary documents (e.g. JSON encoded database records)
across SlottedPage records by splitting them into fixed size chunks with a
framing header, so documents larger than a single page slot are supported.
"""

# Import Modules
from __future__ import annotations
from typing import Optional

from src.core.database.page_manager import PageManager
from src.core.database.slotted_page import SlottedPage


class RecordStore:
    """
    Chunked document store writing binary documents into slotted pages.

    Chunk record layout:
        - doc_id: uint32 (4 bytes)
        - chunk_index: uint16 (2 bytes)
        - chunk_count: uint16 (2 bytes)
        - payload: up to CHUNK_PAYLOAD_SIZE bytes

    Attributes:
        page_manager (PageManager): Underlying page allocator and cache.
        next_document_id (int): Id assigned to the next stored document.
        _current_page (Optional[SlottedPage]): Page currently being filled.
    """

    CHUNK_HEADER_SIZE: int = 8
    CHUNK_PAYLOAD_SIZE: int = 4096

    def __init__(
        self,
        page_manager: PageManager,
        next_document_id: int = 0
    ) -> None:
        """
        Initializes the RecordStore instance.

        Args:
            page_manager (PageManager): Underlying page allocator and cache.
            next_document_id (int, optional): First document id to assign.
                When appending to an existing store, pass the existing
                document count so ids do not collide. Defaults to 0.
        """

        self.page_manager: PageManager = page_manager
        self.next_document_id: int = next_document_id
        self._current_page: Optional[SlottedPage] = None

    def put_document(self, payload: bytes) -> int:
        """
        Stores a binary document, chunked across pages as needed.

        Args:
            payload (bytes): Binary document payload.

        Returns:
            int: The document id assigned to the stored document.
        """

        doc_id: int = self.next_document_id
        self.next_document_id += 1

        chunks: list[bytes] = [
            payload[offset:offset + self.CHUNK_PAYLOAD_SIZE]
            for offset in range(0, len(payload), self.CHUNK_PAYLOAD_SIZE)
        ] or [b""]

        chunk_count: int = len(chunks)

        for chunk_index, chunk in enumerate(chunks):
            record: bytes = (
                doc_id.to_bytes(4, byteorder="little")
                + chunk_index.to_bytes(2, byteorder="little")
                + chunk_count.to_bytes(2, byteorder="little")
                + chunk
            )

            self._insert_record(record=record)

        return doc_id

    def read_all_documents(self) -> dict[int, bytes]:
        """
        Reads every stored document from all pages, reassembling chunks.

        Returns:
            dict[int, bytes]: Document id to reassembled binary payload.
        """

        chunks: dict[int, dict[int, bytes]] = {}
        chunk_counts: dict[int, int] = {}

        for page_id in range(self.page_manager.next_page_id):
            page: SlottedPage = self.page_manager.get_page(page_id=page_id)

            for slot_index in range(page.slot_count):
                record: bytes = page.read_record(slot_index=slot_index)

                doc_id: int = int.from_bytes(record[0:4], byteorder="little")
                chunk_index: int = int.from_bytes(record[4:6], byteorder="little")
                chunk_count: int = int.from_bytes(record[6:8], byteorder="little")

                chunks.setdefault(doc_id, {})[chunk_index] = record[self.CHUNK_HEADER_SIZE:]
                chunk_counts[doc_id] = chunk_count

        return {
            doc_id: b"".join(chunks[doc_id][index] for index in range(chunk_counts[doc_id]))
            for doc_id in sorted(chunks)
        }

    def _insert_record(self, record: bytes) -> None:
        """
        Inserts a single chunk record into the current page, allocating a new
        page when the current one is full.

        Args:
            record (bytes): Framed chunk record to insert.

        Raises:
            ValueError: If the record does not fit into an empty page.
        """

        if self._current_page is None:
            self._current_page = self.page_manager.allocate_page()

        if self._current_page.insert_record(record=record) is not None:
            return

        self._current_page = self.page_manager.allocate_page()

        if self._current_page.insert_record(record=record) is None:
            raise ValueError(
                f"Record of {len(record)} bytes does not fit into an empty page"
            )
