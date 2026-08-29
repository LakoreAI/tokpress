"""Resource-file access for the pretrained profile data.

The .bin files under <project_root>/data/profileN/ hold each profile's vocab pieces, shared LZ dictionary, and baked order-0/order-1 rANS frequency tables in small length-prefixed / fixed-width binary formats (length-prefixed vocab pieces, u16-pair frequency tables, u16 dictionary tokens), parsed by the loaders in tokenizer/vocab.py, codec/dictionaries.py, and entropy/pretrained_tables.py.

The data directory lives at the project root (../../data relative to this file), NOT under src/tokpress/, so it is not bundled as package data -- this only resolves correctly when running from a repo checkout (including an editable `pip install -e .`), not from a wheel installed elsewhere. Set TOKPRESS_DATA_DIR to override the location explicitly (e.g. if the data/ directory is deployed separately from the source checkout).
"""
import json
import os
from pathlib import Path

NUM_PROFILES = 4
PROFILE_NAMES = ["code", "json", "pkgmeta", "general"]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DATA_ROOT = Path(os.environ["TOKPRESS_DATA_DIR"]) if "TOKPRESS_DATA_DIR" in os.environ else _PROJECT_ROOT / "data"


def _profile_dir(profile_id: int) -> Path:
    d = _DATA_ROOT / f"profile{profile_id}"
    if not d.is_dir():
        raise FileNotFoundError(
            f"tokpress profile data not found at {d}. Expected the data/ "
            f"directory at the project root ({_DATA_ROOT}); set "
            f"TOKPRESS_DATA_DIR to point at it if it's deployed elsewhere."
        )
    return d


def read_binary(profile_id: int, name: str) -> bytes:
    return (_profile_dir(profile_id) / name).read_bytes()


def read_context_ids(profile_id: int) -> list[int]:
    text = (_profile_dir(profile_id) / "context_ids.json").read_text()
    return json.loads(text)


def read_length_prefixed_blobs(profile_id: int, name: str) -> list[bytes]:
    data = read_binary(profile_id, name)
    blobs = []
    offset = 0
    while offset < len(data):
        length = int.from_bytes(data[offset : offset + 4], "little")
        offset += 4
        blobs.append(data[offset : offset + length])
        offset += length
    return blobs
