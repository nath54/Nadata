"""
Fixed-size binary slotted page storage for Nadata.

Implements fixed-size memory pages (default 64KB) storing binary records
with a slotted header, slot directory, and free space management.
"""

# Import Modules
from __future__ import annotations
from typing import Optional


class SlottedPage:
    """
    Fixed-size slotted page for binary record storage.

    Layout:
        Header (8 bytes):
            - page_id: uint32 (4 bytes)
            - slot_count: uint16 (2 bytes)
            - free_space_offset: uint16 (2 bytes, points to start of record heap from end)
        Slot Directory:
            - array of (offset: uint16, length: uint16) growing forward from header.
        Record Heap:
            - variable-length binary payloads growing backward from end of page.

    Attributes:
        page_id (int): Identifier of the page.
        page_size (int): Total size in bytes (default 65536 = 64KB).
        data (bytearray): Contiguous byte buffer.
    """

    HEADER_SIZE: int = 8
    SLOT_SIZE: int = 4

    def __init__(
        self,
        page_id: int,
        page_size: int = 65536,
        raw_bytes: Optional[bytes] = None
    ) -> None:
        """
        Initializes the SlottedPage.

        Args:
            page_id (int): Page identifier.
            page_size (int, optional): Page capacity in bytes. Defaults to 65536 (64KB).
            raw_bytes (Optional[bytes], optional): Pre-existing buffer. Defaults to None.
        """

        self.page_id: int = page_id
        self.page_size: int = page_size
        self.is_dirty: bool = False

        if raw_bytes is not None:
            if len(raw_bytes) != page_size:
                raise ValueError(
                    f"raw_bytes length {len(raw_bytes)} does not match page_size {page_size}"
                )

            self.data: bytearray = bytearray(raw_bytes)
        else:
            self.data = bytearray(page_size)
            self._write_header(page_id=page_id, slot_count=0, free_space_offset=page_size)
            self.is_dirty = True

    def _write_header(self, page_id: int, slot_count: int, free_space_offset: int) -> None:
        """
        Writes header fields into page buffer.

        Args:
            page_id (int): Page ID.
            slot_count (int): Current slot count.
            free_space_offset (int): Current end of free space heap.
        """

        # page_id (4 bytes, little endian)
        self.data[0:4] = page_id.to_bytes(4, byteorder="little")

        # slot_count (2 bytes, little endian)
        self.data[4:6] = slot_count.to_bytes(2, byteorder="little")

        # free_space_offset (2 bytes, little endian)
        self.data[6:8] = free_space_offset.to_bytes(2, byteorder="little")

    @property
    def slot_count(self) -> int:
        """
        Returns number of slots currently allocated in the page.

        Returns:
            int: Slot count.
        """

        return int.from_bytes(self.data[4:6], byteorder="little")

    @property
    def free_space_offset(self) -> int:
        """
        Returns the byte offset where the free space heap begins.

        Returns:
            int: Byte offset.
        """

        return int.from_bytes(self.data[6:8], byteorder="little")

    def available_space(self) -> int:
        """
        Calculates remaining free bytes available for new records.

        Returns:
            int: Available byte count.
        """

        directory_end: int = self.HEADER_SIZE + (self.slot_count * self.SLOT_SIZE)
        return self.free_space_offset - directory_end

    def insert_record(self, record: bytes) -> Optional[int]:
        """
        Inserts a byte record into the page, returning its allocated slot index.

        Args:
            record (bytes): Binary payload to store.

        Returns:
            Optional[int]: Slot index, or None if page has insufficient space.
        """

        record_len: int = len(record)
        needed_space: int = record_len + self.SLOT_SIZE

        if self.available_space() < needed_space:
            return None

        current_slots: int = self.slot_count
        new_free_offset: int = self.free_space_offset - record_len

        # Write record bytes into heap
        self.data[new_free_offset:new_free_offset + record_len] = record

        # Write slot entry (offset: 2 bytes, length: 2 bytes)
        slot_pos: int = self.HEADER_SIZE + (current_slots * self.SLOT_SIZE)
        self.data[slot_pos:slot_pos + 2] = new_free_offset.to_bytes(2, byteorder="little")
        self.data[slot_pos + 2:slot_pos + 4] = record_len.to_bytes(2, byteorder="little")

        # Update header
        self._write_header(
            page_id=self.page_id,
            slot_count=current_slots + 1,
            free_space_offset=new_free_offset
        )

        self.is_dirty = True
        return current_slots

    def read_record(self, slot_index: int) -> bytes:
        """
        Reads a record payload from a given slot index.

        Args:
            slot_index (int): Slot index.

        Returns:
            bytes: Stored record bytes.

        Raises:
            IndexError: If slot_index is out of range.
        """

        if slot_index < 0 or slot_index >= self.slot_count:
            raise IndexError(
                f"Slot index {slot_index} out of range (total slots: {self.slot_count})"
            )

        slot_pos: int = self.HEADER_SIZE + (slot_index * self.SLOT_SIZE)
        offset: int = int.from_bytes(self.data[slot_pos:slot_pos + 2], byteorder="little")
        length: int = int.from_bytes(self.data[slot_pos + 2:slot_pos + 4], byteorder="little")

        return bytes(self.data[offset:offset + length])

    def to_bytes(self) -> bytes:
        """
        Returns immutable bytes buffer for disk write.

        Returns:
            bytes: Full page bytes.
        """

        return bytes(self.data)
