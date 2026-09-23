from .core import (
    IndexedBatchWriter,
    batch_record_count,
    benchmark,
    compress,
    compress_each,
    compress_file,
    compress_many,
    decompress,
    decompress_each,
    decompress_file,
    decompress_many,
    indexed_compress,
    indexed_decompress,
    indexed_read,
    iter_decompress_many,
    tokenize_stats,
)
from .dictionary import TokDict

__version__ = "0.2.0"

__all__ = [
    "compress",
    "decompress",
    "compress_file",
    "decompress_file",
    "compress_each",
    "decompress_each",
    "compress_many",
    "decompress_many",
    "iter_decompress_many",
    "batch_record_count",
    "indexed_compress",
    "indexed_decompress",
    "indexed_read",
    "IndexedBatchWriter",
    "benchmark",
    "tokenize_stats",
    "TokDict",
]
