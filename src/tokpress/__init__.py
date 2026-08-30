from .core import (
    IndexedBatchWriter,
    benchmark,
    compress,
    compress_file,
    compress_many,
    decompress,
    decompress_file,
    decompress_many,
    indexed_compress,
    indexed_decompress,
    indexed_read,
    tokenize_stats,
)
from .dictionary import TokDict

__version__ = "0.1.0"

__all__ = [
    "compress",
    "decompress",
    "compress_file",
    "decompress_file",
    "compress_many",
    "decompress_many",
    "indexed_compress",
    "indexed_decompress",
    "indexed_read",
    "IndexedBatchWriter",
    "benchmark",
    "tokenize_stats",
    "TokDict",
]
