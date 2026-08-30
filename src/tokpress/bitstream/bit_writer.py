class BitWriter:
    """LSB-first bit packer."""

    def __init__(self) -> None:
        self._buf = bytearray()
        self._bit_buf = 0
        self._bit_count = 0

    def write_bits(self, value: int, count: int) -> None:
        mask = (1 << count) - 1
        self._bit_buf |= (value & mask) << self._bit_count
        self._bit_count += count
        while self._bit_count >= 8:
            self._buf.append(self._bit_buf & 0xFF)
            self._bit_buf >>= 8
            self._bit_count -= 8

    def write_byte(self, value: int) -> None:
        self.write_bits(value, 8)

    def write_uint16(self, value: int) -> None:
        self.write_bits(value, 16)

    def write_uint32(self, value: int) -> None:
        self.write_bits(value, 32)

    def write_uint64(self, value: int) -> None:
        self.write_bits(value, 64)

    def flush(self) -> None:
        if self._bit_count > 0:
            self._buf.append(self._bit_buf & 0xFF)
            self._bit_buf = 0
            self._bit_count = 0

    def size(self) -> int:
        return len(self._buf)

    def getvalue(self) -> bytes:
        return bytes(self._buf)
