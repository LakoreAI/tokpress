from .core import (
    benchmark,
    compress,
    compress_file,
    compress_many,
    decompress,
    decompress_file,
    decompress_many,
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
    "benchmark",
    "tokenize_stats",
    "TokDict",
]
