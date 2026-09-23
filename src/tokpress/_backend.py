"""Optional Rust acceleration. `tokpress._rs` is the compiled extension; when it is missing (pure-Python install) or `TOKPRESS_PURE_PYTHON` is set, every code path falls back to the reference Python implementation, which produces byte-identical streams."""

import os

from .entropy import frequency

_rs = None
if not os.environ.get("TOKPRESS_PURE_PYTHON"):
    try:
        from . import _rs
    except ImportError:  # pragma: no cover - pure-Python install
        _rs = None

_NATIVE_M_BITS = 16


def rust():
    """The compiled extension module, or None when the Python path must be used. The extension bakes in RANS_M=65536, so a test that shrinks RANS_M falls back to Python automatically."""
    if _rs is not None and frequency.RANS_M_BITS == _NATIVE_M_BITS:
        return _rs
    return None
