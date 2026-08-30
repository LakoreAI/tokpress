"""Byte-oriented LEB128-style varint helpers, plus a sorted-symbol-id-list codec (delta + varint) shared by every wire mode that needs to transmit a sparse set of alphabet indices without spending a fixed 4 bytes/symbol -- o200k_base's ~200k-token alphabet makes that fixed cost dominate a per-record table for any text with a few thousand distinct tokens."""

from .bit_reader import BitReader
from .bit_writer import BitWriter


def write_varint(w: BitWriter, value: int) -> None:
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            w.write_byte(byte | 0x80)
        else:
            w.write_byte(byte)
            return


def read_varint(r: BitReader) -> int:
    result = 0
    shift = 0
    while True:
        byte = r.read_byte()
        result |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return result
        shift += 7


def write_symbol_list(w: BitWriter, sorted_ids: list[int]) -> None:
    w.write_uint32(len(sorted_ids))
    prev = 0
    for sid in sorted_ids:
        write_varint(w, sid - prev)
        prev = sid


def read_symbol_list(r: BitReader) -> list[int]:
    n = r.read_uint32()
    ids = []
    prev = 0
    for _ in range(n):
        prev += read_varint(r)
        ids.append(prev)
    return ids
