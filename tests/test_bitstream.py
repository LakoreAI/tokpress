import pytest

from tokpress.bitstream import BitReader, BitWriter


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
