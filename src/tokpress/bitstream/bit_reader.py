class BitReader:
    """LSB-first bit unpacker."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._byte_pos = 0
        self._bit_buf = 0
        self._bit_count = 0

    def _refill(self) -> None:
        while self._bit_count <= 56 and self._byte_pos < len(self._data):
            self._bit_buf |= self._data[self._byte_pos] << self._bit_count
            self._bit_count += 8
            self._byte_pos += 1

    def read_bits(self, count: int) -> int:
        if self._bit_count < count:
            self._refill()
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

    def has_more(self) -> bool:
        return self._byte_pos < len(self._data) or self._bit_count > 0
