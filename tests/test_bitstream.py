import pytest

from tokpress.bitstream import BitReader, BitWriter
from tokpress.bitstream.varint import read_varint, write_varint


def test_bitstream_mixed_widths_roundtrip():
    values = [(5, 3), (17, 5), (200, 8), (54321, 16), (123456789, 32), (1, 1)]

    w = BitWriter()
    for value, width in values:
        w.write_bits(value, width)
    w.flush()

    r = BitReader(w.getvalue())
    for value, width in values:
        assert r.read_bits(width) == value


def test_bitstream_byte_uint16_uint32_helpers():
    w = BitWriter()
    w.write_byte(0xAB)
    w.write_uint16(0xBEEF)
    w.write_uint32(0xDEADBEEF)
    w.flush()

    r = BitReader(w.getvalue())
    assert r.read_byte() == 0xAB
    assert r.read_uint16() == 0xBEEF
    assert r.read_uint32() == 0xDEADBEEF


def test_bitstream_read_beyond_data_raises():
    """Regression: reading past the end of a (truncated/corrupt) stream used
    to silently return zero-padded garbage instead of raising."""
    w = BitWriter()
    w.write_uint16(0x1234)
    w.flush()
    r = BitReader(w.getvalue())

    assert r.read_uint16() == 0x1234
    with pytest.raises(ValueError):
        r.read_uint32()


def test_varint_roundtrip_and_length_cap():
    w = BitWriter()
    for value in (0, 1, 127, 128, 300, 2**32 - 1, 2**63 - 1):
        write_varint(w, value)
    w.flush()

    r = BitReader(w.getvalue())
    for value in (0, 1, 127, 128, 300, 2**32 - 1, 2**63 - 1):
        assert read_varint(r) == value

    # an all-continuation prefix would shift forever; it must be rejected
    # instead of accumulating an unbounded integer.
    runaway = BitWriter()
    for _ in range(12):
        runaway.write_byte(0x80)
    runaway.flush()
    with pytest.raises(ValueError, match="varint is longer than 64 bits"):
        read_varint(BitReader(runaway.getvalue()))


def test_u64_readable_after_partial_bit_reads():
    """Regression: read_bits could not refill its buffer when 56 < bit_count < 64
    bits were still buffered, so a 64-bit read (e.g. the rANS state word)
    raised "truncated" whenever preceding reads left an arbitrary bit
    remainder (10-bit table frequencies at a shrunken RANS_M expose it;
    16-bit fields at the real RANS_M never did). A 64-bit value must be
    readable after any number of partial-bit reads."""
    from tokpress.bitstream import BitReader, BitWriter

    marker = 0x1122334455667788
    for n10 in range(1, 90):
        w = BitWriter()
        for i in range(n10):
            w.write_bits(i % 1023, 10)
        w.write_uint64(marker)
        w.flush()

        r = BitReader(w.getvalue())
        for i in range(n10):
            assert r.read_bits(10) == i % 1023
        assert r.read_uint64() == marker
