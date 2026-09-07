class BitReader:
    """LSB-first bit unpacker."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._byte_pos = 0
        self._bit_buf = 0
        self._bit_count = 0

    def _refill(self) -> None:
        # Load bytes until at least 64 bits are buffered (or the stream ends),
        # so even a 64-bit read can always be satisfied from the buffer. The
        # old cap of 56 bits made read_bits fail when a request arrived with
        # 56 < bit_count < 64 buffered and more data available -- unreachable
        # while every field width was a multiple of 8, but hit by any read
        # pattern that leaves an arbitrary bit remainder (e.g. 10-bit table
        # frequencies at a shrunken RANS_M).
        while self._bit_count < 64 and self._byte_pos < len(self._data):
            self._bit_buf |= self._data[self._byte_pos] << self._bit_count
            self._bit_count += 8
            self._byte_pos += 1

    def read_bits(self, count: int) -> int:
        if self._bit_count < count:
            self._refill()
        if self._bit_count < count and self._byte_pos < len(self._data):
            while self._bit_count < count and self._byte_pos < len(self._data):
                self._bit_buf |= self._data[self._byte_pos] << self._bit_count
                self._bit_count += 8
                self._byte_pos += 1
        if self._bit_count < count:
            raise ValueError(f"truncated bitstream: requested {count} bits but only {self._bit_count} remain")
        mask = (1 << count) - 1
        result = self._bit_buf & mask
        self._bit_buf >>= count
        self._bit_count -= count
        return result

    def read_byte(self) -> int:
        return self.read_bits(8)

    def read_uint16(self) -> int:
        return self.read_bits(16)

    def read_uint32(self) -> int:
        return self.read_bits(32)

    def read_uint64(self) -> int:
        return self.read_bits(64)

    def align_to_byte(self) -> None:
        """Discard any remaining pad bits so the next read starts at a byte
        boundary. Writers pad their final partial byte with zero bits; readers
        that stop mid-byte (e.g. MODE_RAW_TOKENS) must align before reading a
        trailing payload."""
        n = self._bit_count % 8
        if n:
            self._bit_buf >>= n
            self._bit_count -= n

    def has_more(self) -> bool:
        return self._byte_pos < len(self._data) or self._bit_count > 0
